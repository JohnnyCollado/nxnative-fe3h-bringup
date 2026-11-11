// FSR 2 (dm/third_party/fsr2) on the Android host's Vulkan backend: the temporal upscaler behind vkd_dlss
// (nvn_vk_draw.c). One context serves the whole run; it is recreated when the game's formats, sizes or flags
// change (as nvn_mtl.m's dlss_scaler cache does), the old one destroyed once the GPU can no longer use it
// (vkx_defer). Everything FSR 2 records goes into the submission's own command buffer, between the backend's
// barrier_all calls. The vendored VK backend was patched to keep every image in VK_IMAGE_LAYOUT_GENERAL, the
// layout the rest of the host assumes.
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "fsr2_glue.h"
#include "nvn_vk.h"   // g_vk (the device, vk_driver.c / nvn_vk_mem.c) and vkx_defer
#include "ffx_fsr2.h"
#include "vk/ffx_fsr2_vk.h"

#define FLOG(...) do { flockfile(stderr); fprintf(stderr, "[dm] [fsr2] " __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } while (0)

// -------- the environment switches, read once (as the backend's other DM_* switches are) --------
static int g_sharpen = -1; static float g_sharpness, g_gap_ms;
static void env_once(void) {
    if (g_sharpen >= 0) return;
    const char *e = getenv("DM_FSR2_SHARPEN"); g_sharpen = !e || e[0] != '0';
    e = getenv("DM_FSR2_SHARPNESS"); g_sharpness = e ? (float)atof(e) : 0.3f;
    e = getenv("DM_FSR2_FRAME_DELTA_MAX"); g_gap_ms = e ? (float)atof(e) : 200.0f;
    FLOG("sharpening %s (%g); a frame gap over %g ms resets the history (DM_FSR2_SHARPEN, DM_FSR2_SHARPNESS, DM_FSR2_FRAME_DELTA_MAX)",
         g_sharpen ? "on" : "off", g_sharpness, g_gap_ms);
}

static void fsr2_message(FfxFsr2MsgType type, const wchar_t *msg) {
    char b[1024]; wcstombs(b, msg, sizeof b - 1); b[sizeof b - 1] = 0;
    if (type == FFX_FSR2_MESSAGE_TYPE_ERROR) FLOG("[RED] %s", b); else FLOG("%s", b);
}

static PFN_vkVoidFunction fsr2_gpa(VkDevice d, const char *name) { return vkGetDeviceProcAddr(d, name); }   // volk: the driver's device functions (volkLoadDevice, nvn_vk_mem.c)

// -------- the context and its key --------
static FfxFsr2Context g_ctx; static int g_have;
static void *g_scratch; static size_t g_scratch_n;
static int g_atexit_reg, g_force_reset;
static struct { VkFormat cfmt, dfmt, mfmt, ofmt, efmt; uint32_t iw, ih, ow, oh; uint32_t flags; int disp_mv, have_expo; } g_key; static int g_keyed;

static int disp_mv(const fsr2_job *j) {   // motion vectors at display resolution when their texture is display-sized but not render-sized
    return j->mot_w == j->ow && j->mot_h == j->oh && !(j->mot_w == j->iw && j->mot_h == j->ih);
}

static int key_changed(const fsr2_job *j) {
    if (!g_keyed) return 1;
    return g_key.cfmt != j->color_fmt || g_key.dfmt != j->depth_fmt || g_key.mfmt != j->motion_fmt || g_key.ofmt != j->output_fmt ||
           g_key.efmt != j->exposure_fmt || g_key.flags != j->flags || g_key.disp_mv != disp_mv(j) || g_key.have_expo != (j->exposure != VK_NULL_HANDLE) ||
           j->iw > g_key.iw || j->ih > g_key.ih || j->ow != g_key.ow || j->oh != g_key.oh;
}

static void defer_destroy(void *p) { FfxFsr2Context *c = (FfxFsr2Context *)p; ffxFsr2ContextDestroy(c); free(c); }
static void ctx_discard(void) {   // the game's formats / sizes changed: the old context goes once the GPU is done with it
    FfxFsr2Context *old = (FfxFsr2Context *)malloc(sizeof *old); if (!old) return; *old = g_ctx;
    vkx_defer(defer_destroy, old);
    g_have = 0; g_keyed = 0;
}
static void fs2_exit(void) { if (g_have) ffxFsr2ContextDestroy(&g_ctx); g_have = 0; free(g_scratch); g_scratch = NULL; }

// The camera FSR 2 turns depth into view-space distance with (setupDeviceDepthToViewSpaceDepthParams): its depth clip, which drops the history of pixels that were hidden behind
// something last frame, compares those distances. DLSS takes no camera, so NVN2's DLSS command carries none; left 0 they made 0 / 0 in the reversed-depth transform, every distance NaN,
// and the history was kept where it should have been dropped: ghosting behind moving characters (the character creation screen). As Eden's FSR 3.1 for this game
// (dm/third_party/eden-fefw .../vk_fsr3_upscaler.cpp CAMERA_NEAR, CAMERA_FOV): near 0.1, a vertical field of view of 60 degrees, and with the game's reversed depth no far plane
// (FSR 2 then wants cameraNear FLT_MAX and the near plane in cameraFar, ffx_fsr2.cpp's checks). The game's own camera is not known here: DM_FSR2_NEAR and DM_FSR2_FOV (degrees) tune them
static void camera_of(const fsr2_job *j, float *cn, float *cf, float *fov) {
    static float nearp = -1, fovd;
    if (nearp < 0) { const char *e = getenv("DM_FSR2_NEAR"); nearp = e ? (float)atof(e) : 0.1f; if (!(nearp > 0)) nearp = 0.1f;
                     e = getenv("DM_FSR2_FOV"); fovd = e ? (float)atof(e) : 60.0f; if (!(fovd > 1 && fovd < 179)) fovd = 60.0f;
                     FLOG("camera for the depth clip: near %g, vertical field of view %g degrees (DM_FSR2_NEAR, DM_FSR2_FOV)", nearp, fovd); }
    *fov = fovd * 3.14159265f / 180.0f;
    if (j->flags & DLSS_F_DEPTH_INVERTED) { *cn = FLT_MAX; *cf = nearp; } else { *cn = nearp; *cf = 1000.0f; }
}

static int make_ctx(const fsr2_job *j, FfxFsr2Context *ctx) {
    if (!g_vk.ok || !g_vk.dev) { FLOG("[RED] no Vulkan device: FSR 2 cannot run"); return 0; }
    size_t need = ffxFsr2GetScratchMemorySizeVK(g_vk.pd);
    if (need > g_scratch_n) { free(g_scratch); g_scratch = NULL; }
    if (!g_scratch) { if (posix_memalign(&g_scratch, 64, need)) { FLOG("[RED] FSR 2: %zu bytes of scratch memory cannot be allocated", need); return 0; } g_scratch_n = need; }
    FfxFsr2Interface iface; memset(&iface, 0, sizeof iface);
    FfxErrorCode r = ffxFsr2GetInterfaceVK(&iface, g_scratch, g_scratch_n, g_vk.pd, fsr2_gpa);
    if (r != FFX_OK) { FLOG("[RED] ffxFsr2GetInterfaceVK failed (%d)", r); return 0; }
    FfxFsr2ContextDescription d = {};
    d.flags = FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION;
    if (j->flags & DLSS_F_DEPTH_INVERTED) d.flags |= FFX_FSR2_ENABLE_DEPTH_INVERTED | FFX_FSR2_ENABLE_DEPTH_INFINITE;   // reversed depth with no far plane (camera_of)
    if (j->flags & DLSS_F_AUTO_EXPOSURE) d.flags |= FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR2_ENABLE_AUTO_EXPOSURE;
    if (disp_mv(j)) d.flags |= FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    d.maxRenderSize = {j->iw, j->ih}; d.displaySize = {j->ow, j->oh};
    d.fpMessage = fsr2_message;
    d.device = ffxGetDeviceVK(g_vk.dev);
    d.callbacks = iface;
    r = ffxFsr2ContextCreate(ctx, &d);
    if (r != FFX_OK) { FLOG("[RED] ffxFsr2ContextCreate failed (%d) for colour format %d, depth %d, motion %d, output %d, %ux%u -> %ux%u",
                            r, j->color_fmt, j->depth_fmt, j->motion_fmt, j->output_fmt, j->iw, j->ih, j->ow, j->oh); return 0; }
    FLOG("DLSS: FSR 2 %ux%u -> %ux%u, colour format %d, depth %d, motion %d, output %d%s%s%s", j->iw, j->ih, j->ow, j->oh,
         j->color_fmt, j->depth_fmt, j->motion_fmt, j->output_fmt,
         (d.flags & FFX_FSR2_ENABLE_DEPTH_INVERTED) ? ", depth inverted" : "",
         (d.flags & FFX_FSR2_ENABLE_AUTO_EXPOSURE) ? ", auto exposure (HDR)" : "",
         (d.flags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? ", display-resolution motion vectors" : "");
    if (!g_atexit_reg) { g_atexit_reg = 1; atexit(fs2_exit); }
    return 1;
}

// -------- Context creation off the submit thread: it compiles FSR 2's shaders (~200 ms on this device), which stalled the first upscaled frame --------
// While a context is being made the stand-in draws bilinearly; the frame after it is ready, FSR 2 takes over (with reset, the history starting fresh)
static pthread_mutex_t g_ctx_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_creating, g_fresh; static fsr2_job g_pending;   // fresh: the first dispatch after creation resets (the game's own reset came while the stand-in drew)
static void *ctx_worker(void *unused) {
    (void)unused;
    pthread_mutex_lock(&g_ctx_mu); fsr2_job j = g_pending; pthread_mutex_unlock(&g_ctx_mu);
    FfxFsr2Context ctx; int ok = make_ctx(&j, &ctx);
    pthread_mutex_lock(&g_ctx_mu);
    if (ok) {
        g_ctx = ctx;
        g_key.cfmt = j.color_fmt; g_key.dfmt = j.depth_fmt; g_key.mfmt = j.motion_fmt; g_key.ofmt = j.output_fmt; g_key.efmt = j.exposure_fmt;
        g_key.iw = j.iw; g_key.ih = j.ih; g_key.ow = j.ow; g_key.oh = j.oh; g_key.flags = j.flags;
        g_key.disp_mv = disp_mv(&j); g_key.have_expo = j.exposure != VK_NULL_HANDLE;
        g_keyed = 1; g_have = 1; g_fresh = 1;
    }
    g_creating = 0;
    pthread_mutex_unlock(&g_ctx_mu);
    return NULL;
}

int fsr2_dispatch(const fsr2_job *j) {
    env_once();
    if (!j || !j->color || !j->depth || !j->motion || !j->output || !j->cb) return 0;
    if (!g_vk.ok || !g_vk.dev) return 0;
    pthread_mutex_lock(&g_ctx_mu);
    if (g_creating || !g_have || key_changed(j)) {
        if (!g_creating) {
            if (g_have) ctx_discard();
            g_pending = *j; g_creating = 1;
            pthread_t th;
            if (pthread_create(&th, NULL, ctx_worker, NULL)) g_creating = 0; else pthread_detach(th);
        }
        pthread_mutex_unlock(&g_ctx_mu);
        return 0;   // the stand-in draws this frame; the next one finds the context (or a key change starts another worker)
    }
    int fresh = g_fresh; g_fresh = 0;
    pthread_mutex_unlock(&g_ctx_mu);
    // the frame's delta from the host's own clock: the game stops calling DLSS while paused, so a long gap
    // (a pause, the app in the background) resets the accumulation history, which no longer matches the scene
    static struct timespec last_ts; static int have_ts;
    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    float delta = 16.6f;
    if (have_ts) {
        double ms = (now.tv_sec - last_ts.tv_sec) * 1000.0 + (now.tv_nsec - last_ts.tv_nsec) / 1e6;
        if (ms > g_gap_ms) g_force_reset = 1;
        if (ms < 1.0) ms = 1.0; if (ms > 100.0) ms = 100.0;
        delta = (float)ms;
    }
    last_ts = now; have_ts = 1;
    if (j->pre_exposure <= 0) { static int once; if (!once++) FLOG("[RED] DLSS pre-exposure %g is not positive: 1 used", j->pre_exposure); }
    FfxFsr2DispatchDescription d = {};
    d.commandList = ffxGetCommandListVK(j->cb);
    d.color = ffxGetTextureResourceVK(&g_ctx, j->color, j->color_v, j->iw, j->ih, j->color_fmt, NULL, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.depth = ffxGetTextureResourceVK(&g_ctx, j->depth, j->depth_v, j->iw, j->ih, j->depth_fmt, NULL, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = ffxGetTextureResourceVK(&g_ctx, j->motion, j->motion_v, disp_mv(j) ? j->ow : j->iw, disp_mv(j) ? j->oh : j->ih, j->motion_fmt, NULL, FFX_RESOURCE_STATE_COMPUTE_READ);
    if (j->exposure && !(j->flags & DLSS_F_AUTO_EXPOSURE))
        d.exposure = ffxGetTextureResourceVK(&g_ctx, j->exposure, j->exposure_v, 1, 1, j->exposure_fmt, NULL, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.output = ffxGetTextureResourceVK(&g_ctx, j->output, j->output_v, j->ow, j->oh, j->output_fmt, NULL, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {j->jitter[0], j->jitter[1]};   // the game's jitter as MetalFX takes it (its sign unchecked there too)
    d.motionVectorScale = {j->mv[0], j->mv[1]};
    d.renderSize = {j->iw, j->ih};
    camera_of(j, &d.cameraNear, &d.cameraFar, &d.cameraFovAngleVertical);
    d.enableSharpening = g_sharpen != 0; d.sharpness = g_sharpness;
    d.frameTimeDelta = delta;
    d.preExposure = j->pre_exposure > 0 ? j->pre_exposure : 1.0f;
    d.reset = j->reset != 0 || g_force_reset != 0 || fresh; g_force_reset = 0;
    FfxErrorCode r = ffxFsr2ContextDispatch(&g_ctx, &d);
    if (r != FFX_OK) { static int once; if (!once++) FLOG("[RED] ffxFsr2ContextDispatch failed (%d)", r); return 0; }
    return 1;
}
