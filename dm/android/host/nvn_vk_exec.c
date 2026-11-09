// The Android host's Vulkan backend, step V1b: submissions, fences, textures, render targets, clears, copies and presented frames as PNG files.
// It takes the place of what nvn_mtl.m does on the Mac for these command records (dm/gfx/nvn/nvn_obj.h); draws and dispatches are counted but not executed yet (V1c).
// Without a Vulkan device every entry falls back to the null backend (nvn_null_android.c), which applies the CPU-visible effects at once.
//
// Submissions: each game submission (mtl_sub_begin … mtl_exec … mtl_sub_commit) is one command buffer from a ring, numbered by a timeline semaphore. A completion
// thread applies a submission's CPU-visible effects (event writes, counter reports) once the GPU has finished it, as nvn_mtl.m's post_run does in its completion
// handler, and mtl_wait waits until they are applied. Buffer updates, copies and clears are GPU commands in record order. Buffer clears are written with
// vkCmdUpdateBuffer, not vkCmdFillBuffer, which this driver loses at some offsets (dm/android/tools/vk_alias_test.c).
// Textures: one VkImage per texture serial (hle_nvn.c nvn_new_serial), kept in VK_IMAGE_LAYOUT_GENERAL. A texture is uploaded from its pool memory before its
// first GPU read (block-linear storage deswizzled on the CPU, nvn_deswizzle; linear storage copied straight from the pool's buffer), and again after the CPU
// writes it (mtl_tex_cpu_written), unless the GPU has written it since (render target, clear, copy destination), whose content then lives in the image.
// Render targets: dynamic rendering, begun when a clear (later a draw) needs it, with load and store. Full memory barriers between passes and transfers: correct
// first, fast later.
// DM_VK_SHOT=<frame>[,<frame>…] or DM_VK_SHOT_EVERY=<n>: the presented image of those frames as $DM_ROOT/shots/frame_<n>.png
#include <errno.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "nvn_obj.h"
#include "nvn_timestamp.h"
#include "nvn_deswizzle.h"
#include "nvn_vk.h"
#include "nvn_fmt_vk.h"
#include "nvn_vk_int.h"


void nvn_null_exec(const uint8_t *p);   // nvn_null_android.c: the CPU-only executor
uint64_t nvn_null_commit(void), nvn_null_last_fence(void);
uint64_t guest_ns(void);

// ---------- Submissions and completion ----------
enum { NSUB = 8 };
static sub_t g_sub[NSUB];
sub_t *g_cur;
static VkCommandPool g_cp;
static VkSemaphore g_sem;
uint64_t g_fence;                                          // the last submission's number (submit thread)
static _Atomic uint64_t g_fence_pub; _Atomic uint64_t g_applied;            // published to other threads; the last one whose effects are applied
static pthread_mutex_t g_done_mu = PTHREAD_MUTEX_INITIALIZER, g_q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER, g_work_cv = PTHREAD_COND_INITIALIZER;
static int g_exec_ok = -1;
static volatile uint32_t *g_crumb_map;   // DM_VK_CRUMBS (vkx_crumb)
static VkDeviceMemory g_last_stage_mem;   // the memory of the last untracked staging buffer (s NULL): its caller frees it
static VkBuffer stage_buffer(sub_t *s, uint64_t size, void **map);
// Objects freed while the GPU may still use them (a memory pool's buffer, a released texture's image): destroyed by the completion thread once every submission
// that could reference them has finished. "Could": the one being recorded when it was freed (g_fence + 1), and one more for a lookup racing the removal
typedef struct { uint64_t fence; void (*fn)(void *); void *arg; } grave_t;
static grave_t *g_grave; static int g_ngrave, g_cgrave;
static pthread_mutex_t g_grave_mu = PTHREAD_MUTEX_INITIALIZER;
void vkx_defer(void (*fn)(void *), void *arg) {
    if (g_exec_ok != 1) { fn(arg); return; }
    pthread_mutex_lock(&g_grave_mu);
    GROW(g_grave, g_ngrave, g_cgrave); g_grave[g_ngrave++] = (grave_t){__atomic_load_n(&g_fence, __ATOMIC_ACQUIRE) + 2, fn, arg};
    pthread_mutex_unlock(&g_grave_mu);
}
static void grave_collect(uint64_t applied) {
    pthread_mutex_lock(&g_grave_mu);
    for (int i = 0; i < g_ngrave; ) if (g_grave[i].fence <= applied) { g_grave[i].fn(g_grave[i].arg); g_grave[i] = g_grave[--g_ngrave]; } else i++;
    pthread_mutex_unlock(&g_grave_mu);
}
uint64_t g_stats_draws; static uint64_t g_stats_passes, g_stats_uploads, g_stats_clears, g_stats_copies;


static void apply_posts(sub_t *s) {
    for (int i = 0; i < s->nposts; i++) {
        post_op *o = &s->posts[i];
        if (o->kind == 1) { if (o->val) { if (o->mode == NVN_EVENT_SIGNAL_MODE_ADD) *o->val += o->v; else *o->val = o->v; } }
        else if (o->addr) { uint64_t t = nvn_timestamp_from_ns(guest_ns()), rec[2] = {o->type == 0 || o->type == 0x10 ? t : 0, t}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
    }
    s->nposts = 0;
    for (int i = 0; i < s->nviews; i++) vkDestroyImageView(g_vk.dev, s->views[i], NULL);
    s->nviews = 0;
    for (int i = 0; i < s->nstage; i++) { vkDestroyBuffer(g_vk.dev, s->stage[i].b, NULL); vkFreeMemory(g_vk.dev, s->stage[i].m, NULL); }
    s->nstage = 0;
    vkd_sub_done(s);
}
static void *completion_thread(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_done_mu);
        while (atomic_load(&g_applied) >= atomic_load(&g_fence_pub)) pthread_cond_wait(&g_work_cv, &g_done_mu);
        pthread_mutex_unlock(&g_done_mu);
        uint64_t want = atomic_load(&g_applied) + 1;
        VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_sem, &want};
        VkResult r = vkWaitSemaphores(g_vk.dev, &wi, 2000000000ull);
        if (r == VK_TIMEOUT || r == VK_ERROR_DEVICE_LOST) {
            static int told; if (!told++) { sub_t *f = &g_sub[want % NSUB], *pv = &g_sub[(want + NSUB - 1) % NSUB];
                VLOG("[RED] submission %llu has not completed (%d): the GPU counter reads %u; it held:%s", (unsigned long long)want, r, g_crumb_map ? *g_crumb_map : 0u, f->log); VLOG("the one before it held:%s", pv->log); }
            if (r == VK_TIMEOUT) continue;
            usleep(100000); continue; }
        if (r) { VRED_ONCE("vkWaitSemaphores failed (%d)", r); usleep(1000); continue; }
        apply_posts(&g_sub[want % NSUB]);
        pthread_mutex_lock(&g_done_mu); atomic_store(&g_applied, want); pthread_cond_broadcast(&g_done_cv); pthread_mutex_unlock(&g_done_mu);
        grave_collect(want);
    }
    return NULL;
}
static int exec_init(void) {
    if (g_exec_ok >= 0) return g_exec_ok;
    g_exec_ok = 0;
    if (!vk_ready()) return 0;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, g_vk.qfam};
    if (vkCreateCommandPool(g_vk.dev, &cpi, NULL, &g_cp)) { VLOG("[RED] command pool failed: drawing stays off"); return 0; }
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    for (int i = 0; i < NSUB; i++) if (vkAllocateCommandBuffers(g_vk.dev, &ai, &g_sub[i].cb)) { VLOG("[RED] command buffers failed"); return 0; }
    VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st};
    if (vkCreateSemaphore(g_vk.dev, &si, NULL, &g_sem)) { VLOG("[RED] timeline semaphore failed"); return 0; }
    pthread_t th; if (pthread_create(&th, NULL, completion_thread, NULL)) return 0;
    pthread_detach(th);
    g_exec_ok = 1;
    VLOG("executor on: submissions numbered by a timeline semaphore, %d command buffers", NSUB);
    return 1;
}
void vkx_sublog(sub_t *s, const char *fmt, ...) {
    if (s->nlog >= (int)sizeof s->log - 64) return;
    va_list ap; va_start(ap, fmt); int n = vsnprintf(s->log + s->nlog, sizeof s->log - (size_t)s->nlog, fmt, ap); va_end(ap);
    if (n > 0) s->nlog += n; if (s->nlog > (int)sizeof s->log - 1) s->nlog = (int)sizeof s->log - 1;
}
// DM_VK_CRUMBS: after each draw, dispatch, copy and stretch the GPU writes an increasing number into a host-visible word; when a submission never finishes,
// the word tells the last command that did (the submission's log marks each with #n). Diagnostic only: it ends the render pass after every draw
static int g_crumbs = -1; static VkBuffer g_crumb_buf; static VkDeviceMemory g_crumb_mem;
void vkx_crumb(sub_t *s) {
    if (g_crumbs < 0) {
        const char *e = getenv("DM_VK_CRUMBS"); g_crumbs = e ? (atoi(e) == 2 ? 2 : 1) : 0;   // 2: only at the end of each render pass and after dispatches / copies (passes kept whole)
        if (g_crumbs) { void *m = NULL; g_crumb_buf = stage_buffer(NULL, 256, &m); g_crumb_mem = g_last_stage_mem; g_crumb_map = m; if (!g_crumb_buf) g_crumbs = 0; else VLOG("DM_VK_CRUMBS: a GPU-written counter after every command"); }
    }
    if (!g_crumbs) return;
    if (g_crumbs == 2 && g_rt.active) return;   // a draw inside a pass: its pass's end writes the counter
    end_rendering(); barrier_all(s->cb);
    uint32_t v = (uint32_t)(s->value ? 0 : 0) + ++s->ncrumb;
    vkCmdUpdateBuffer(s->cb, g_crumb_buf, 0, 4, &v);
    barrier_all(s->cb);
    vkx_sublog(s, " #%u", v);
}
void barrier_all(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
void mtl_sub_begin(void) {
    if (!exec_init() || g_cur) return;
    sub_t *s = &g_sub[(g_fence + 1) % NSUB];
    if (s->value) {   // this slot's previous submission must be done (and its effects applied) before its command buffer is reused
        pthread_mutex_lock(&g_done_mu);
        while (atomic_load(&g_applied) < s->value) pthread_cond_wait(&g_done_cv, &g_done_mu);
        pthread_mutex_unlock(&g_done_mu);
    }
    vkResetCommandBuffer(s->cb, 0); s->nlog = 0; s->log[0] = 0; s->ncrumb = 0;
    if (g_crumb_map) *g_crumb_map = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(s->cb, &bi);
    barrier_all(s->cb);   // the CPU's writes to pool memory before this submission (coherent memory: a host-to-device barrier is all it needs)
    s->open = 1; g_cur = s;
}
static int wait_applied(uint64_t fence, uint64_t timeout_ns);
uint64_t mtl_sub_commit(void) {
    if (!exec_init()) return nvn_null_commit();
    if (!g_cur) mtl_sub_begin();
    sub_t *s = g_cur;
    end_rendering();
    vkEndCommandBuffer(s->cb);
    s->value = __atomic_add_fetch(&g_fence, 1, __ATOMIC_RELEASE); s->open = 0; g_cur = NULL; vkd_sub_committed(s->value);
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &s->value};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, 1, &s->cb, 1, &g_sem};
    pthread_mutex_lock(&g_q_mu); VkResult r = vkQueueSubmit(g_vk.q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_q_mu);
    if (r) VRED_ONCE("vkQueueSubmit failed (%d)", r);
    pthread_mutex_lock(&g_done_mu); atomic_store(&g_fence_pub, s->value); pthread_cond_signal(&g_work_cv); pthread_mutex_unlock(&g_done_mu);
    static int sync = -1; if (sync < 0) { sync = getenv("DM_VK_SYNC") != NULL; if (sync) VLOG("DM_VK_SYNC: each submission waits for the GPU before the next is recorded"); }
    struct timespec t0, t1; if (sync) clock_gettime(CLOCK_MONOTONIC, &t0);
    if (sync && wait_applied(s->value, 5000000000ull)) {
        clock_gettime(CLOCK_MONOTONIC, &t1); double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ms > 30) VLOG("DM_VK_SYNC: submission %llu took %.0f ms: %.300s", (unsigned long long)s->value, ms, s->log);
        vkd_after_sync();
    } else if (sync) VRED_ONCE("DM_VK_SYNC: submission %llu did not finish in 5 s; the GPU counter reads %u; it held:%s", (unsigned long long)s->value, g_crumb_map ? *g_crumb_map : 0u, s->log);
    return s->value;
}
uint64_t vkx_last_fence(void) { return atomic_load(&g_fence_pub); }
VkSemaphore vkx_timeline(void) { return g_sem; }
VkResult vkx_submit(const VkSubmitInfo *si, VkFence f) { pthread_mutex_lock(&g_q_mu); VkResult r = vkQueueSubmit(g_vk.q, 1, si, f); pthread_mutex_unlock(&g_q_mu); return r; }
uint64_t mtl_last_fence(void) { return g_exec_ok == 1 ? atomic_load(&g_fence_pub) : nvn_null_last_fence(); }
static int wait_applied(uint64_t fence, uint64_t timeout_ns) {
    if (atomic_load(&g_applied) >= fence) return 1;
    if (!timeout_ns) return 0;
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    uint64_t ns = (uint64_t)t.tv_nsec + (timeout_ns > 600000000000ull ? 600000000000ull : timeout_ns);
    t.tv_sec += (time_t)(ns / 1000000000ull); t.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_done_mu);
    while (atomic_load(&g_applied) < fence) if (pthread_cond_timedwait(&g_done_cv, &g_done_mu, &t) == ETIMEDOUT) break;
    int ok = atomic_load(&g_applied) >= fence;
    pthread_mutex_unlock(&g_done_mu);
    return ok;
}
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { return g_exec_ok == 1 ? wait_applied(fence, timeout_ns) : 1; }
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
void mtl_finish(void) { if (g_exec_ok == 1) wait_applied(atomic_load(&g_fence_pub), ~0ull); }
void mtl_drain(void) { mtl_finish(); }

// ---------- Textures ----------
enum { TBUCKETS = 1 << 14 };
static vk_tex *g_tex[TBUCKETS];
static pthread_mutex_t g_tex_mu = PTHREAD_MUTEX_INITIALIZER;

static uint32_t device_local_type(uint32_t bits) {
    for (uint32_t i = 0; i < g_vk.mp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_vk.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    for (uint32_t i = 0; i < 32; i++) if (bits >> i & 1) return i;
    return 0;
}
static int is_depth(VkFormat f) { return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32; }
static VkImageAspectFlags aspect_of(VkFormat f) {
    if (f == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT) return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    if (is_depth(f)) return VK_IMAGE_ASPECT_DEPTH_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
}
// vkCmdBlitImage offered for this pair: both formats' blit features (and linear filtering when asked); never for depth / stencil (scaled depth blits preceded this driver's hangs)
static int blit_ok(VkFormat src, VkFormat dst, int linear) {
    if (is_depth(src) || is_depth(dst)) return 0;
    VkFormatProperties a, b; vkGetPhysicalDeviceFormatProperties(g_vk.pd, src, &a); vkGetPhysicalDeviceFormatProperties(g_vk.pd, dst, &b);
    if (!(a.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(b.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) return 0;
    return !linear || (a.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
}
static int is_compressed(VkFormat f) { return f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }

static int tex_create(vk_tex *x, VkCommandBuffer cb) {
    uint32_t vf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_vk(x->d.format, &vf, swz) || !vf) { x->failed = 1; VLOG("[RED] texture format %#x has no Vulkan format yet (texture %llu, %ux%u)", x->d.format, (unsigned long long)x->serial, x->d.width, x->d.height); return 0; }
    x->fmt = (VkFormat)vf; x->aspect = aspect_of(x->fmt);
    x->bpb = nvn_fmt_bpb(x->d.format, &x->bw, &x->bh); if (!x->bw) x->bw = 1; if (!x->bh) x->bh = 1;
    x->linear = (x->d.flags & 0x50) != 0;
    int tg = x->d.target;
    x->w = x->d.width ? x->d.width : 1; x->h = x->d.height ? x->d.height : 1; x->depth = 1; x->layers = 1;
    x->type = VK_IMAGE_TYPE_2D; x->vtype = VK_IMAGE_VIEW_TYPE_2D;
    switch (tg) {
        case NVN_TEXTURE_TARGET_1D: x->type = VK_IMAGE_TYPE_1D; x->vtype = VK_IMAGE_VIEW_TYPE_1D; x->h = 1; break;
        case NVN_TEXTURE_TARGET_1D_ARRAY: x->type = VK_IMAGE_TYPE_1D; x->vtype = VK_IMAGE_VIEW_TYPE_1D_ARRAY; x->layers = x->h; x->h = 1; break;
        case NVN_TEXTURE_TARGET_3D: x->type = VK_IMAGE_TYPE_3D; x->vtype = VK_IMAGE_VIEW_TYPE_3D; x->depth = x->d.depth ? x->d.depth : 1; break;
        case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: x->vtype = VK_IMAGE_VIEW_TYPE_2D_ARRAY; x->layers = x->d.depth ? x->d.depth : 1; break;
        case NVN_TEXTURE_TARGET_CUBEMAP: x->vtype = VK_IMAGE_VIEW_TYPE_CUBE; x->layers = 6; break;
        case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: x->vtype = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; x->layers = x->d.depth ? x->d.depth : 6; break;
        case NVN_TEXTURE_TARGET_BUFFER: x->failed = 1; return 0;   // texture buffers are read through the buffer (V1c)
        default: break;
    }
    x->levels = x->d.levels ? x->d.levels : 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vk.pd, x->fmt, &fp);
    VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
    VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageCreateFlags cf = is_compressed(x->fmt) ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    // Storage (a shader writes it as an image): only formats that allow it. An sRGB image is written through its UNORM twin, which the image may carry as an extended usage
    VkFormat twin = x->fmt == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM : x->fmt == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_UNDEFINED;
    x->storage_fmt = VK_FORMAT_UNDEFINED;
    if (x->d.samples <= 1) {
        if (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) { u |= VK_IMAGE_USAGE_STORAGE_BIT; x->storage_fmt = x->fmt; }
        else if (twin) { VkFormatProperties tp; vkGetPhysicalDeviceFormatProperties(g_vk.pd, twin, &tp);
            if (tp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) { u |= VK_IMAGE_USAGE_STORAGE_BIT; cf |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; x->storage_fmt = twin; } }
    }
    x->usage = u;
    if ((tg == NVN_TEXTURE_TARGET_CUBEMAP || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) && x->layers % 6 == 0 && x->w == x->h) cf |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (x->type == VK_IMAGE_TYPE_3D && (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) cf |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkSampleCountFlagBits sc = x->d.samples >= 8 ? VK_SAMPLE_COUNT_8_BIT : x->d.samples >= 4 ? VK_SAMPLE_COUNT_4_BIT : x->d.samples >= 2 ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, cf, x->type, x->fmt, {x->w, x->h, x->depth}, x->levels, x->layers, sc,
                            VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(g_vk.dev, &ii, NULL, &x->img);
    if (r) { x->failed = 1; VLOG("[RED] vkCreateImage %d for texture %llu (format %#x → %d, %ux%ux%u, %u levels, %u layers, target %d)", r, (unsigned long long)x->serial, x->d.format, x->fmt, x->w, x->h, x->depth, x->levels, x->layers, tg); return 0; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, x->img, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
    if ((r = vkAllocateMemory(g_vk.dev, &mai, NULL, &x->dm)) || (r = vkBindImageMemory(g_vk.dev, x->img, x->dm, 0))) {
        x->failed = 1; VLOG("[RED] texture memory %d (%llu bytes)", r, (unsigned long long)mr.size); vkDestroyImage(g_vk.dev, x->img, NULL); x->img = VK_NULL_HANDLE; return 0; }
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                               VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, x->img, {x->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    end_rendering();   // a texture first seen while a draw gathers its resources: barriers are not allowed inside the render pass
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    return 1;
}
// The texture's host image (created on first use when cb is given); NULL when it cannot be made
vk_tex *tex_of(const tex_t *t, VkCommandBuffer cb) {
    if (!t || !t->serial) return NULL;
    pthread_mutex_lock(&g_tex_mu);
    vk_tex **b = &g_tex[t->serial % TBUCKETS], *x = *b;
    while (x && x->serial != t->serial) x = x->next;
    if (!x && cb) {
        x = calloc(1, sizeof *x); x->serial = t->serial; x->d = t->d; x->L = t->L; x->mem = t->mem; x->dirty = 1;
        x->next = *b; *b = x;
        tex_create(x, cb);
    }
    pthread_mutex_unlock(&g_tex_mu);
    return x && !x->failed && x->img ? x : NULL;
}
static void tex_free_gpu(void *arg) {
    vk_tex *x = arg;
    if (x->img) { vkd_tex_release(x); vkDestroyImage(g_vk.dev, x->img, NULL); vkFreeMemory(g_vk.dev, x->dm, NULL); }
    free(x);
}
void mtl_tex_init(const tex_t *t) { (void)t; }   // images are made at first use, when the texture's command buffer is open
// A draw that samples an image its own render pass draws into (a feedback loop) is undefined on a tiled GPU without the feedback-loop layout this driver lacks;
// runs hung the GPU in the game's first 3D frame, and ending the pass after every draw made it go away. Such a draw reads a copy instead (as emulators do)
vk_tex *vkx_clone(sub_t *s, vk_tex *x, int keep) {
    vk_tex *c = calloc(1, sizeof *c);
    c->serial = x->serial; c->d = x->d; c->L = x->L; c->mem = NULL;
    if (!tex_create(c, s->cb)) { free(c); return NULL; }
    c->gpu_owned = 1; c->uploaded = 1;
    end_rendering();
    VkImageCopy rg[16]; uint32_t n = 0;
    for (uint32_t l = 0; l < x->levels && l < c->levels && n < 16; l++)
        rg[n++] = (VkImageCopy){{x->aspect, l, 0, x->layers}, {0, 0, 0}, {c->aspect, l, 0, c->layers}, {0, 0, 0}, {level_dim(x->w, l), level_dim(x->h, l), level_dim(x->depth, l)}};
    barrier_all(s->cb);
    vkCmdCopyImage(s->cb, x->img, VK_IMAGE_LAYOUT_GENERAL, c->img, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier_all(s->cb);
    if (!keep) vkx_defer(tex_free_gpu, c);
    return c;
}
void mtl_tex_release(tex_t *t) {
    if (!t || !t->serial) return;
    pthread_mutex_lock(&g_tex_mu);
    vk_tex **b = &g_tex[t->serial % TBUCKETS];
    while (*b && (*b)->serial != t->serial) b = &(*b)->next;
    vk_tex *x = *b; if (x) *b = x->next;
    pthread_mutex_unlock(&g_tex_mu);
    if (x) vkx_defer(tex_free_gpu, x);   // the submission being recorded may still use it: destroyed once the GPU is done
}
void mtl_tex_cpu_written(const tex_t *t) {
    if (!t || !t->serial) return;
    pthread_mutex_lock(&g_tex_mu);
    vk_tex *x = g_tex[t->serial % TBUCKETS]; while (x && x->serial != t->serial) x = x->next;
    if (x) { x->dirty = 1; x->gpu_owned = 0; }   // an explicit CPU write (WriteTexels: the movie's pictures; the dirty-page path asks mtl_tex_gpu_owned first): the CPU's content is current again
    pthread_mutex_unlock(&g_tex_mu);
}
int mtl_tex_gpu_owned(uint64_t serial) {
    pthread_mutex_lock(&g_tex_mu);
    vk_tex *x = g_tex[serial % TBUCKETS]; while (x && x->serial != serial) x = x->next;
    int r = x && x->gpu_owned;
    pthread_mutex_unlock(&g_tex_mu);
    return r;
}
static VkBuffer stage_buffer(sub_t *s, uint64_t size, void **map) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b)) return VK_NULL_HANDLE;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, g_vk.pool_type};
    VkDeviceMemory m; if (vkAllocateMemory(g_vk.dev, &mai, NULL, &m)) { vkDestroyBuffer(g_vk.dev, b, NULL); return VK_NULL_HANDLE; }
    vkBindBufferMemory(g_vk.dev, b, m, 0); vkMapMemory(g_vk.dev, m, 0, VK_WHOLE_SIZE, 0, map);
    if (s) { GROW(s->stage, s->nstage, s->cstage); s->stage[s->nstage].b = b; s->stage[s->nstage].m = m; s->nstage++; } else g_last_stage_mem = m;
    return b;
}
uint32_t level_dim(uint32_t v, unsigned level) { v >>= level; return v ? v : 1; }
// Upload a texture's storage into its image when the GPU is about to read it and its current content is the CPU's
static void tex_upload(vk_tex *x, sub_t *s) {
    if (x->gpu_owned || !x->dirty || !x->mem) return;
    if (x->d.samples > 1) { x->dirty = 0; return; }
    end_rendering();   // copies are not allowed inside a render pass (a texture a draw samples is uploaded before the draw resumes the pass)
    VkCommandBuffer cb = s->cb;
    if (x->linear) {
        VkBuffer pb; uint64_t off;
        if (!vk_pool_find((uint64_t)(uintptr_t)x->mem, x->L.size ? x->L.size : 1, &pb, &off)) { VRED_ONCE("linear texture %llu is in no pool: not uploaded", (unsigned long long)x->serial); x->dirty = 0; return; }
        uint32_t stride = x->d.stride ? x->d.stride : (uint32_t)(x->L.size / (x->h ? x->h : 1));
        VkBufferImageCopy c = {off, x->bpb ? stride / x->bpb * x->bw : 0, 0, {x->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT ? x->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT : x->aspect, 0, 0, 1}, {0, 0, 0}, {x->w, x->h, 1}};
        barrier_all(cb); vkCmdCopyBufferToImage(cb, pb, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c); barrier_all(cb);
    } else {
        uint64_t total = 0;
        for (unsigned l = 0; l < x->levels; l++) {
            uint32_t bw = (level_dim(x->w, l) + x->bw - 1) / x->bw, bh = (level_dim(x->h, l) + x->bh - 1) / x->bh;
            total += (uint64_t)bw * x->bpb * bh * level_dim(x->depth, l) * x->layers;
        }
        void *map = NULL; VkBuffer sb = stage_buffer(s, total ? total : 1, &map);
        if (!sb) { VRED_ONCE("no staging memory for texture uploads"); return; }
        VkBufferImageCopy *cs = calloc((size_t)x->levels * x->layers, sizeof *cs); int nc = 0; uint64_t o = 0;
        VkImageAspectFlags asp = (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect;
        for (unsigned l = 0; l < x->levels; l++) for (unsigned ly = 0; ly < x->layers; ly++) {
            uint32_t w = level_dim(x->w, l), h = level_dim(x->h, l), dz = level_dim(x->depth, l);
            uint32_t bw = (w + x->bw - 1) / x->bw, bh = (h + x->bh - 1) / x->bh; size_t pitch = (size_t)bw * x->bpb;
            if (nvn_deswizzle(&x->d, l, ly, x->mem, (uint8_t *)map + o, pitch)) { VRED_ONCE("texture %llu (format %#x, target %d): deswizzle refused", (unsigned long long)x->serial, x->d.format, x->d.target); continue; }
            cs[nc++] = (VkBufferImageCopy){o, bw * x->bw, bh * x->bh, {asp, l, ly, 1}, {0, 0, 0}, {w, h, dz}};
            o += pitch * bh * dz;
        }
        if (nc) { barrier_all(cb); vkCmdCopyBufferToImage(cb, sb, x->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nc, cs); barrier_all(cb); }
        free(cs);
    }
    x->dirty = 0; x->uploaded = 1; g_stats_uploads++;
    vkx_sublog(s, " U%llu:%s%d:%ux%ux%u/%ul/%uy", (unsigned long long)x->serial, x->linear ? "lin" : "bl", x->fmt, x->w, x->h, x->depth, x->levels, x->layers);
}
// The texture is about to be written by the GPU: bring in its CPU content first unless it is already the GPU's (a partial write keeps the rest)
vk_tex *tex_for_write(const tex_t *t, sub_t *s) {
    vk_tex *x = tex_of(t, s->cb); if (!x) return NULL;
    if (!x->gpu_owned && !x->uploaded) tex_upload(x, s);
    x->gpu_owned = 1; x->dirty = 0;
    return x;
}
// A linear texture lies over pool memory the CPU may write at any time without telling NVN (the video codec's native images: hle_codec_video.c pool_native_write writes
// each decoded picture there). On the Mac it is made on the pool's buffer and sees every write; here it is an image of its own, so it is copied again from the pool
// the first time each submission reads it (a GPU copy), unless the GPU itself has written it
vk_tex *tex_for_read(const tex_t *t, sub_t *s) {
    vk_tex *x = tex_of(t, s->cb);
    if (x && x->linear && !x->gpu_owned && x->lin_sub != g_fence + 1) { x->dirty = 1; x->lin_sub = g_fence + 1; }
    if (x) tex_upload(x, s);
    return x;
}

// ---------- Render targets ----------
struct vkx_rt g_rt;
static VkImageView make_view(sub_t *s, vk_tex *x, const view_t *v, int has_v, VkImageAspectFlags aspect) {
    uint32_t level = has_v && v->set_levels ? v->base_level : 0, layer = has_v && v->set_layers ? v->min_layer : 0;
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, x->usage & ~VK_IMAGE_USAGE_STORAGE_BIT};   // an attachment view: not storage (its format may not allow it)
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, x->type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, x->fmt,
                                {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY}, {aspect, level, 1, layer, 1}};
    if (x->type == VK_IMAGE_TYPE_3D) vi.viewType = VK_IMAGE_VIEW_TYPE_2D;   // a slice of a 3D target needs the 2D-array-compatible flag set at creation
    VkImageView view; if (vkCreateImageView(g_vk.dev, &vi, NULL, &view)) return VK_NULL_HANDLE;
    GROW(s->views, s->nviews, s->cviews); s->views[s->nviews++] = view;
    return view;
}
void end_rendering(void) {
    if (!g_rt.active || !g_cur) return;
    vkCmdEndRendering(g_cur->cb); g_rt.active = 0;
    barrier_all(g_cur->cb);
    if (g_crumbs == 2) vkx_crumb(g_cur);
}
int begin_rendering(sub_t *s) {
    if (g_rt.active) return 1;
    if (!g_rt.has_rt) return 0;
    VkRenderingAttachmentInfo ca[8]; memset(ca, 0, sizeof ca);
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    uint32_t w = ~0u, h = ~0u; int n = g_rt.rt.n > 8 ? 8 : g_rt.rt.n, any = 0;
    for (int i = 0; i < n; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        vk_tex *x = g_rt.rt.color[i] ? tex_for_write(g_rt.rt.color[i], s) : NULL;
        if (!x || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) continue;
        ca[i].imageView = make_view(s, x, &g_rt.rt.cview[i], g_rt.rt.has_cview[i], VK_IMAGE_ASPECT_COLOR_BIT);
        uint32_t l = g_rt.rt.has_cview[i] && g_rt.rt.cview[i].set_levels ? g_rt.rt.cview[i].base_level : 0;
        if (level_dim(x->w, l) < w) w = level_dim(x->w, l); if (level_dim(x->h, l) < h) h = level_dim(x->h, l); any = 1;
    }
    int has_d = 0, has_s = 0;
    vk_tex *dx = g_rt.rt.depth ? tex_for_write(g_rt.rt.depth, s) : NULL;
    if (dx && (dx->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
        VkImageView dv = make_view(s, dx, &g_rt.rt.dview, g_rt.rt.has_dview, dx->aspect);
        has_d = !!(dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT); has_s = !!(dx->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        da.imageView = sa.imageView = dv; da.imageLayout = sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        uint32_t l = g_rt.rt.has_dview && g_rt.rt.dview.set_levels ? g_rt.rt.dview.base_level : 0;
        if (level_dim(dx->w, l) < w) w = level_dim(dx->w, l); if (level_dim(dx->h, l) < h) h = level_dim(dx->h, l); any = 1;
    }
    if (!any && !g_rt.want.width) return 0;
    if (!any) { w = g_rt.want.width; h = g_rt.want.height; n = 0; }   // a draw that only writes images or storage buffers: a pass with no attachments, the viewport's size
    g_rt.ext = (VkExtent2D){w, h};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_rt.ext}, 1, 0, (uint32_t)n, ca, has_d ? &da : NULL, has_s ? &sa : NULL};
    vkCmdBeginRendering(s->cb, &ri); g_rt.active = 1; g_stats_passes++;
    vkx_sublog(s, " P%ux%u/%d%s", w, h, n, has_d ? "d" : "");
    return 1;
}

// ---------- Execution ----------
static uint64_t g_copy_row_stride, g_copy_image_stride;
static _Atomic uint64_t g_opn[OP_COUNT_ + 1]; static int g_opstat = -1;
static void add_post(sub_t *s, post_op o) { GROW(s->posts, s->nposts, s->cposts); s->posts[s->nposts++] = o; }
static VkClearColorValue clear_value(VkFormat f, const float rgba[4]) {
    VkClearColorValue v;
    int sint = f == VK_FORMAT_R8_SINT || f == VK_FORMAT_R16_SINT || f == VK_FORMAT_R32_SINT || f == VK_FORMAT_R8G8_SINT || f == VK_FORMAT_R16G16_SINT || f == VK_FORMAT_R32G32_SINT ||
               f == VK_FORMAT_R8G8B8A8_SINT || f == VK_FORMAT_R16G16B16A16_SINT || f == VK_FORMAT_R32G32B32A32_SINT;
    int uint = f == VK_FORMAT_R8_UINT || f == VK_FORMAT_R16_UINT || f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R8G8_UINT || f == VK_FORMAT_R16G16_UINT || f == VK_FORMAT_R32G32_UINT ||
               f == VK_FORMAT_R8G8B8A8_UINT || f == VK_FORMAT_R16G16B16A16_UINT || f == VK_FORMAT_R32G32B32A32_UINT || f == VK_FORMAT_A2B10G10R10_UINT_PACK32;
    for (int i = 0; i < 4; i++) { if (sint) v.int32[i] = (int32_t)rgba[i]; else if (uint) v.uint32[i] = (uint32_t)rgba[i]; else v.float32[i] = rgba[i]; }
    return v;
}
static void exec_list(sub_t *s, const uint8_t *p);
void mtl_exec(const uint8_t *p) {
    if (g_opstat < 0) g_opstat = getenv("DM_NVN_OPSTAT") != NULL;
    if (!exec_init()) { nvn_null_exec(p); return; }
    if (!g_cur) mtl_sub_begin();
    exec_list(g_cur, p);
}
static void exec_list(sub_t *s, const uint8_t *p) {
    VkCommandBuffer cb = s->cb;
    for (int guard = 0; p && guard < (1 << 24); guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        if (g_opstat > 0) atomic_fetch_add(&g_opn[h.op <= OP_COUNT_ ? h.op : OP_COUNT_], 1);
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); exec_list(s, (const uint8_t *)(uintptr_t)t); break; }
            // CPU-visible effects: applied when the GPU has finished this submission
            case OP_SIGNAL_EVENT: { post_op o = {1}; memcpy(&o.val, a, 8); memcpy(&o.mode, a + 8, 4); memcpy(&o.v, a + 12, 4); add_post(s, o); break; }
            case OP_REPORT_COUNTER: { post_op o = {2}; memcpy(&o.type, a, 8); memcpy(&o.addr, a + 8, 8); add_post(s, o); break; }
            case OP_FENCE_SYNC: { sync_t *sy; memcpy(&sy, a, 8); if (sy) { sy->fence = g_fence + 1; sy->fenced = 1; } break; }
            // Buffer effects: GPU commands in record order
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); VkBuffer b; uint64_t off;
                if (!n) break;
                end_rendering();
                if ((n & 3) == 0 && (d & 3) == 0 && vk_pool_find(d, n, &b, &off)) { for (uint64_t k = 0; k < n; k += 65536) vkCmdUpdateBuffer(cb, b, off + k, n - k < 65536 ? n - k : 65536, a + 16 + k); barrier_all(cb); vkd_ub_shadow(d, n, a + 16); }
                else memcpy((void *)(uintptr_t)d, a + 16, n);   // not in a pool, or unaligned: written now (the GPU has not read it yet in this submission's order)
                break; }
            case OP_COPY_B2B: { uint64_t sa, d, n; memcpy(&sa, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); VkBuffer sb, db; uint64_t so, dof;
                if (!n) break;
                end_rendering();
                if (vk_pool_find(sa, n, &sb, &so) && vk_pool_find(d, n, &db, &dof)) { VkBufferCopy c = {so, dof, n}; vkCmdCopyBuffer(cb, sb, db, 1, &c); barrier_all(cb); }
                else memmove((void *)(uintptr_t)d, (const void *)(uintptr_t)sa, n);
                break; }
            case OP_CLEAR_BUF: { const p_clear_buf *c = (const p_clear_buf *)a; VkBuffer b; uint64_t off;
                if (!c->size) break;
                end_rendering();
                if (!(c->size & 3) && !(c->addr & 3) && vk_pool_find(c->addr, c->size, &b, &off)) {
                    uint32_t pat[16384]; for (int i = 0; i < 16384; i++) pat[i] = c->value;
                    for (uint64_t k = 0; k < c->size; k += sizeof pat) vkCmdUpdateBuffer(cb, b, off + k, c->size - k < sizeof pat ? c->size - k : sizeof pat, pat);
                    barrier_all(cb);
                } else { uint32_t *d = (uint32_t *)(uintptr_t)c->addr; for (uint64_t i = 0; i < c->size / 4; i++) d[i] = c->value; }
                break; }
            // Render targets and clears
            case OP_SET_RT: end_rendering(); memcpy(&g_rt.rt, a, sizeof g_rt.rt); g_rt.has_rt = 1; break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = (const p_clear_color *)a;
                if (c->index < 0 || c->index >= g_rt.rt.n || !g_rt.rt.color[c->index] || !begin_rendering(s)) break;
                vk_tex *x = tex_of(g_rt.rt.color[c->index], NULL); if (!x) break;
                VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {.color = clear_value(x->fmt, c->rgba)}};
                VkClearRect rect = {{{0, 0}, g_rt.ext}, 0, 1}; vkCmdClearAttachments(cb, 1, &ca, 1, &rect); g_stats_clears++;
                x->nclear++; memcpy(x->last_clear, c->rgba, sizeof x->last_clear);
                break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = (const p_clear_ds *)a;
                if (!g_rt.rt.depth || !begin_rendering(s)) break;
                vk_tex *x = tex_of(g_rt.rt.depth, NULL); if (!x) break;
                VkImageAspectFlags asp = 0;
                if (c->depth_mask && (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (c->stencil_mask && (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (!asp) break;
                VkClearAttachment ca = {asp, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
                VkClearRect rect = {{{0, 0}, g_rt.ext}, 0, 1}; vkCmdClearAttachments(cb, 1, &ca, 1, &rect); g_stats_clears++;
                break; }
            case OP_CLEAR_TEX: { const p_clear_tex *c = (const p_clear_tex *)a;
                end_rendering();
                vk_tex *x = tex_for_write(c->tex, s); if (!x) break;
                uint32_t l = c->has_view && c->view.set_levels ? c->view.base_level : 0, ly = c->has_view && c->view.set_layers ? c->view.min_layer : (uint32_t)c->r.zoffset;
                uint32_t nl = (x->type == VK_IMAGE_TYPE_3D) ? 1 : (uint32_t)(c->r.depth > 0 ? c->r.depth : 1);
                if (ly >= x->layers) ly = 0; if (ly + nl > x->layers) nl = x->layers - ly;
                VkImageSubresourceRange rg = {x->aspect, l, 1, ly, nl};
                if (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
                    VkClearColorValue v; if (c->is_uint) memcpy(v.uint32, c->c.u, 16); else v = clear_value(x->fmt, c->c.f);
                    vkCmdClearColorImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg);
                } else { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; vkCmdClearDepthStencilImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
                barrier_all(cb); g_stats_clears++; x->nclear++; vkx_sublog(s, " CT%llu:%d", (unsigned long long)x->serial, x->fmt);
                break; }
            // Copies
            case OP_COPY_STRIDE: { const p_copy_stride *c = (const p_copy_stride *)a; if (c->image) g_copy_image_stride = (uint64_t)c->stride; else g_copy_row_stride = (uint64_t)c->stride; break; }
            case OP_COPY_T2T: { const p_copy_t2t *c = (const p_copy_t2t *)a;
                end_rendering();
                vk_tex *sx = tex_for_read(c->src, s), *dx = tex_for_write(c->dst, s); if (!sx || !dx) break;
                uint32_t sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
                uint32_t sly = (c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0) + (sx->type == VK_IMAGE_TYPE_3D ? 0 : (uint32_t)c->sr.zoffset);
                uint32_t dly = (c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0) + (dx->type == VK_IMAGE_TYPE_3D ? 0 : (uint32_t)c->dr.zoffset);
                uint32_t nly = (sx->type == VK_IMAGE_TYPE_3D) ? 1 : (uint32_t)(c->sr.depth > 0 ? c->sr.depth : 1);
                if (sly + nly > sx->layers || dly + nly > dx->layers) { VRED_ONCE("texture copy past the layers (src %u+%u of %u, dst %u of %u)", sly, nly, sx->layers, dly, dx->layers); break; }
                if (c->flags & DM_COPY_RESOLVE) {
                    VkImageResolve rs = {{sx->aspect, sl, sly, nly}, {c->sr.xoffset, c->sr.yoffset, 0}, {dx->aspect, dl, dly, nly}, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
                    vkCmdResolveImage(cb, sx->img, VK_IMAGE_LAYOUT_GENERAL, dx->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rs);
                } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && sx->fmt == dx->fmt) {
                    VkImageCopy ic = {{sx->aspect, sl, sly, nly}, {c->sr.xoffset, c->sr.yoffset, sx->type == VK_IMAGE_TYPE_3D ? c->sr.zoffset : 0},
                                      {dx->aspect, dl, dly, nly}, {c->dr.xoffset, c->dr.yoffset, dx->type == VK_IMAGE_TYPE_3D ? c->dr.zoffset : 0},
                                      {(uint32_t)c->sr.width, (uint32_t)c->sr.height, sx->type == VK_IMAGE_TYPE_3D ? (uint32_t)(c->sr.depth > 0 ? c->sr.depth : 1) : 1}};
                    vkCmdCopyImage(cb, sx->img, VK_IMAGE_LAYOUT_GENERAL, dx->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
                } else if (!blit_ok(sx->fmt, dx->fmt, (c->flags & 1) && !is_depth(sx->fmt)) && nly == 1) {   // the driver offers no blit for these formats (depth, here): drawn
                    int32_t ir[4] = {c->sr.xoffset, c->sr.yoffset, c->sr.width, c->sr.height}, orr[4] = {c->dr.xoffset, c->dr.yoffset, c->dr.width, c->dr.height};
                    vkd_stretch(s, sx, sl, sly, ir, dx, dl, dly, orr, (c->flags & 1) != 0);
                } else {
                    VkImageBlit bl = {{sx->aspect, sl, sly, nly}, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + c->sr.height, 1}},
                                      {dx->aspect, dl, dly, nly}, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
                    vkCmdBlitImage(cb, sx->img, VK_IMAGE_LAYOUT_GENERAL, dx->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, (c->flags & 1) && !is_depth(sx->fmt) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                }
                barrier_all(cb); g_stats_copies++; dx->ncopy++;
                vkx_sublog(s, " T2T%llu:%d:%dx%d>%llu:%d:%dx%d%s", (unsigned long long)sx->serial, sx->fmt, c->sr.width, c->sr.height, (unsigned long long)dx->serial, dx->fmt, c->dr.width, c->dr.height,
                           (c->flags & DM_COPY_RESOLVE) ? "R" : (c->sr.width == c->dr.width && c->sr.height == c->dr.height && sx->fmt == dx->fmt) ? "C" : "B");
                vkx_crumb(s);
                break; }
            case OP_COPY_B2T: { const p_copy_b2t *c = (const p_copy_b2t *)a;
                end_rendering();
                vk_tex *x = tex_for_write(c->tex, s); VkBuffer b; uint64_t off; if (!x) break;
                uint64_t need = g_copy_row_stride ? g_copy_row_stride * (uint64_t)(c->r.height > 0 ? c->r.height : 1) : 1;
                if (!vk_pool_find(c->src, need, &b, &off)) { VRED_ONCE("buffer-to-texture copy from outside the pools"); break; }
                uint32_t l = c->has_view && c->view.set_levels ? c->view.base_level : 0;
                uint32_t row = g_copy_row_stride && x->bpb ? (uint32_t)(g_copy_row_stride / (uint64_t)x->bpb) * (uint32_t)x->bw : 0;
                VkBufferImageCopy bc = {off, row, 0, {(x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, l, x->type == VK_IMAGE_TYPE_3D ? 0 : (uint32_t)c->r.zoffset, 1},
                                        {c->r.xoffset, c->r.yoffset, x->type == VK_IMAGE_TYPE_3D ? c->r.zoffset : 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, x->type == VK_IMAGE_TYPE_3D ? (uint32_t)c->r.depth : 1}};
                vkCmdCopyBufferToImage(cb, b, x->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bc); barrier_all(cb); g_stats_copies++; x->ncopy++; vkx_sublog(s, " B2T%llu:%d", (unsigned long long)x->serial, x->fmt);
                break; }
            case OP_COPY_T2B: { const p_copy_b2t *c = (const p_copy_b2t *)a;
                end_rendering();
                vk_tex *x = tex_for_read(c->tex, s); VkBuffer b; uint64_t off; if (!x) break;
                uint64_t need = g_copy_row_stride ? g_copy_row_stride * (uint64_t)(c->r.height > 0 ? c->r.height : 1) : 1;
                if (!vk_pool_find(c->src, need, &b, &off)) { VRED_ONCE("texture-to-buffer copy into outside the pools"); break; }
                uint32_t l = c->has_view && c->view.set_levels ? c->view.base_level : 0;
                uint32_t row = g_copy_row_stride && x->bpb ? (uint32_t)(g_copy_row_stride / (uint64_t)x->bpb) * (uint32_t)x->bw : 0;
                VkBufferImageCopy bc = {off, row, 0, {(x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect, l, x->type == VK_IMAGE_TYPE_3D ? 0 : (uint32_t)c->r.zoffset, 1},
                                        {c->r.xoffset, c->r.yoffset, 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, 1}};
                vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &bc); barrier_all(cb); g_stats_copies++; vkx_sublog(s, " T2B%llu:%d", (unsigned long long)x->serial, x->fmt);
                break; }
            case OP_BARRIER: end_rendering(); barrier_all(cb); break;
            case OP_DRAW: case OP_DRAW_IDX: { g_stats_draws++;
                vk_tex *x = g_rt.has_rt && g_rt.rt.n > 0 && g_rt.rt.color[0] ? tex_of(g_rt.rt.color[0], NULL) : NULL; if (x) x->ndraw++;
                vkd_draw(s, h.op, a, h.size - 8); break; }
            case OP_DRAW_TEX: VRED_ONCE("DrawTexture not drawn yet"); break;
            case OP_DISPATCH: vkd_dispatch(s, a, h.size - 8); break;
            case OP_DLSS: vkd_dlss(s, a); break;
            default: vkd_state_op(h.op, a); break;
        }
        if (h.size < 8) return;
        p += h.size;
    }
}

// ---------- Present: frames as PNG ----------
static uint32_t crc_tab[256];
static uint32_t crc32_of(uint32_t c, const uint8_t *p, size_t n) {
    if (!crc_tab[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t v = i; for (int k = 0; k < 8; k++) v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1; crc_tab[i] = v; }
    c = ~c; for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xff] ^ (c >> 8); return ~c;
}
static void be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static void chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4); fwrite(h, 1, 8, f); if (n) fwrite(data, 1, n, f);
    uint32_t c = crc32_of(0, (const uint8_t *)type, 4); c = crc32_of(c, data, n); uint8_t t[4]; be32(t, c); fwrite(t, 1, 4, f);
}
// An RGBA8 picture as a PNG with stored (uncompressed) deflate blocks: large, but needs no compressor
static int png_write(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10}; fwrite(sig, 1, 8, f);
    uint8_t ih[13]; be32(ih, w); be32(ih + 4, h); ih[8] = 8; ih[9] = 6; ih[10] = ih[11] = ih[12] = 0; chunk(f, "IHDR", ih, 13);
    size_t raw = (size_t)h * (1 + (size_t)w * 4), nblk = (raw + 65534) / 65535, zn = 2 + raw + nblk * 5 + 4;
    uint8_t *z = malloc(zn), *q = z; if (!z) { fclose(f); return 0; }
    *q++ = 0x78; *q++ = 0x01;
    uint32_t a = 1, b = 0; size_t left = raw, y = 0, x = 0;
    uint8_t *line = malloc(1 + (size_t)w * 4);
    while (left) {
        size_t n = left < 65535 ? left : 65535; left -= n;
        *q++ = left ? 0 : 1; *q++ = (uint8_t)n; *q++ = (uint8_t)(n >> 8); *q++ = (uint8_t)~n; *q++ = (uint8_t)(~n >> 8);
        for (size_t k = 0; k < n; k++) {   // the filtered rows (filter 0) streamed through the blocks
            size_t rl = 1 + (size_t)w * 4;
            if (x == 0) { line[0] = 0; memcpy(line + 1, rgba + y * (size_t)w * 4, (size_t)w * 4); }
            uint8_t v = line[x]; *q++ = v; a = (a + v) % 65521; b = (b + a) % 65521;
            if (++x == rl) { x = 0; y++; }
        }
    }
    free(line);
    uint8_t ad[4]; be32(ad, b << 16 | a); memcpy(q, ad, 4); q += 4;
    chunk(f, "IDAT", z, (uint32_t)(q - z)); chunk(f, "IEND", NULL, 0);
    free(z); fclose(f); return 1;
}
uint64_t g_presents;
static int shot_wanted(uint64_t n) {
    static uint64_t list[64]; static int nl = -1, every = 0;
    if (nl < 0) {
        nl = 0; const char *e = getenv("DM_VK_SHOT"); const char *v = getenv("DM_VK_SHOT_EVERY"); every = v ? atoi(v) : 0;
        for (const char *q = e; q && *q && nl < 64; ) { list[nl++] = strtoull(q, (char **)&q, 10); while (*q == ',' || *q == ' ') q++; }
    }
    if (every > 0 && n % (uint64_t)every == 0) return 1;
    for (int i = 0; i < nl; i++) if (list[i] == n) return 1;
    return 0;
}
static float half_to_f(uint16_t hv) { uint32_t s = hv >> 15, e = hv >> 10 & 31, m = hv & 1023; float f = e ? (float)(m + 1024) * (float)(1u << e) / (float)(1u << 25) : (float)m / (float)(1u << 24); return s ? -f : f; }
static void shot(vk_tex *x, uint64_t n, const int crop[4]) {
    if (x->fmt != VK_FORMAT_R8G8B8A8_UNORM && x->fmt != VK_FORMAT_R8G8B8A8_SRGB && x->fmt != VK_FORMAT_B8G8R8A8_UNORM && x->fmt != VK_FORMAT_B8G8R8A8_SRGB &&
        x->fmt != VK_FORMAT_R16G16B16A16_SFLOAT && x->fmt != VK_FORMAT_A2B10G10R10_UNORM_PACK32) { VRED_ONCE("frame shots: presented format %d not converted", x->fmt); return; }
    uint32_t bpp = x->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4; uint64_t size = (uint64_t)x->w * x->h * bpp;
    void *map = NULL; VkBuffer b = stage_buffer(NULL, size, &map); if (!b) return; VkDeviceMemory bm = g_last_stage_mem;
    static VkCommandPool shot_cp;   // its own pool: a command pool belongs to one thread at a time, and the submission thread records into g_cp's buffers
    if (!shot_cp) { VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, g_vk.qfam}; if (vkCreateCommandPool(g_vk.dev, &cpi, NULL, &shot_cp)) { vkDestroyBuffer(g_vk.dev, b, NULL); vkFreeMemory(g_vk.dev, bm, NULL); return; } }
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, shot_cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cb;
    pthread_mutex_lock(&g_q_mu);
    vkAllocateCommandBuffers(g_vk.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
    barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {x->w, x->h, 1}};
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); barrier_all(cb); vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; vkCreateFence(g_vk.dev, &fi, NULL, &fe);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb}; vkQueueSubmit(g_vk.q, 1, &si, fe);
    pthread_mutex_unlock(&g_q_mu);
    vkWaitForFences(g_vk.dev, 1, &fe, VK_TRUE, 5000000000ull); vkDestroyFence(g_vk.dev, fe, NULL);
    uint8_t *rgba = malloc((size_t)x->w * x->h * 4); const uint8_t *src = map;
    for (uint64_t i = 0; i < (uint64_t)x->w * x->h; i++) {
        uint8_t *o = rgba + i * 4;
        if (bpp == 8) { const uint16_t *hp = (const uint16_t *)(src + i * 8); for (int k = 0; k < 4; k++) { float v = half_to_f(hp[k]); o[k] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + 0.5f); } }
        else if (x->fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32) { uint32_t v; memcpy(&v, src + i * 4, 4); o[0] = (uint8_t)((v & 1023) >> 2); o[1] = (uint8_t)((v >> 10 & 1023) >> 2); o[2] = (uint8_t)((v >> 20 & 1023) >> 2); o[3] = 255; }
        else if (x->fmt == VK_FORMAT_B8G8R8A8_UNORM || x->fmt == VK_FORMAT_B8G8R8A8_SRGB) { o[0] = src[i * 4 + 2]; o[1] = src[i * 4 + 1]; o[2] = src[i * 4]; o[3] = src[i * 4 + 3]; }
        else memcpy(o, src + i * 4, 4);
        o[3] = 255;   // the window shows no alpha
    }
    if (vkd_origin_ll()) for (uint32_t y = 0; y < x->h / 2; y++) {   // lower-left origin: the image is stored bottom-up, as the Mac's present flips it (nvn_mtl.m g_origin_ll)
        uint8_t *r0 = rgba + (size_t)y * x->w * 4, *r1 = rgba + (size_t)(x->h - 1 - y) * x->w * 4;
        for (uint32_t k = 0; k < x->w * 4; k++) { uint8_t t = r0[k]; r0[k] = r1[k]; r1[k] = t; }
    }
    char dir[600], path[700]; snprintf(dir, sizeof dir, "%s/shots", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/frame_%06llu.png", dir, (unsigned long long)n);
    uint32_t ow = x->w, oh = x->h;
    if (crop && crop[2] > 0 && crop[3] > 0 && crop[0] >= 0 && crop[1] >= 0 && (uint32_t)(crop[0] + crop[2]) <= x->w && (uint32_t)(crop[1] + crop[3]) <= x->h) {   // the window's crop (what the screen shows)
        ow = (uint32_t)crop[2]; oh = (uint32_t)crop[3];
        for (uint32_t y = 0; y < oh; y++) memmove(rgba + (size_t)y * ow * 4, rgba + ((size_t)(crop[1] + y) * x->w + (size_t)crop[0]) * 4, (size_t)ow * 4);
    }
    if (png_write(path, rgba, ow, oh)) VLOG("frame %llu: %ux%u presented image (texture %llu, crop %ux%u) written to %s; the GPU so far: %llu clears (last %.2f %.2f %.2f %.2f), %llu copies into it, %llu draws to it",
                                                        (unsigned long long)n, x->w, x->h, (unsigned long long)x->serial, ow, oh, path, (unsigned long long)x->nclear, x->last_clear[0], x->last_clear[1], x->last_clear[2], x->last_clear[3], (unsigned long long)x->ncopy, (unsigned long long)x->ndraw);
    free(rgba); vkFreeCommandBuffers(g_vk.dev, shot_cp, 1, &cb); vkDestroyBuffer(g_vk.dev, b, NULL); vkFreeMemory(g_vk.dev, bm, NULL);
}
static float f11(uint32_t v) { uint32_t e = v >> 6 & 31, m = v & 63; return e ? (float)(m + 64) * (float)(1u << e) / (float)(1u << 21) : (float)m / (float)(1u << 20); }
static float f10(uint32_t v) { uint32_t e = v >> 5 & 31, m = v & 31; return e ? (float)(m + 32) * (float)(1u << e) / (float)(1u << 20) : (float)m / (float)(1u << 19); }
static uint8_t u8f(float v) { return (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + 0.5f); }
void vkx_dump(vk_tex *x, const char *tag) {
    VkFormat f = x->fmt; int depth = !!(x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT);
    uint32_t bpp = f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R16G16B16A16_UNORM || f == VK_FORMAT_R32G32_SFLOAT ? 8 : f == VK_FORMAT_R8_UNORM ? 1 :
                   f == VK_FORMAT_R16_SFLOAT || f == VK_FORMAT_R16_UNORM || f == VK_FORMAT_R8G8_UNORM || f == VK_FORMAT_D16_UNORM ? 2 : f == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : 4;
    if (x->type != VK_IMAGE_TYPE_2D || x->d.samples > 1) return;
    uint64_t size = (uint64_t)x->w * x->h * bpp;
    void *map = NULL; VkBuffer b = stage_buffer(NULL, size, &map); if (!b) return; VkDeviceMemory bm = g_last_stage_mem;
    static VkCommandPool cp; if (!cp) { VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, g_vk.qfam}; vkCreateCommandPool(g_vk.dev, &cpi, NULL, &cp); }
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cb;
    pthread_mutex_lock(&g_q_mu);
    vkAllocateCommandBuffers(g_vk.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
    barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {x->w, x->h, 1}};
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); barrier_all(cb); vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; vkCreateFence(g_vk.dev, &fi, NULL, &fe);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb}; vkQueueSubmit(g_vk.q, 1, &si, fe);
    pthread_mutex_unlock(&g_q_mu);
    vkWaitForFences(g_vk.dev, 1, &fe, VK_TRUE, 5000000000ull); vkDestroyFence(g_vk.dev, fe, NULL);
    uint8_t *rgba = malloc((size_t)x->w * x->h * 4); const uint8_t *src = map; uint64_t n = (uint64_t)x->w * x->h;
    float lo = 1e30f, hi = -1e30f;
    if (depth || f == VK_FORMAT_R32_SFLOAT) for (uint64_t i = 0; i < n; i++) { float v; if (f == VK_FORMAT_D24_UNORM_S8_UINT) { uint32_t u; memcpy(&u, src + i * 4, 4); v = (float)(u & 0xffffff) / 16777215.f; } else if (bpp == 2) { uint16_t u; memcpy(&u, src + i * 2, 2); v = u / 65535.f; } else memcpy(&v, src + i * 4, 4); if (v < lo) lo = v; if (v > hi) hi = v; }
    for (uint64_t i = 0; i < n; i++) {
        uint8_t *o = rgba + i * 4; const uint8_t *p = src + i * bpp; o[3] = 255;
        if (depth || f == VK_FORMAT_R32_SFLOAT) { float v; if (f == VK_FORMAT_D24_UNORM_S8_UINT) { uint32_t u; memcpy(&u, p, 4); v = (float)(u & 0xffffff) / 16777215.f; } else if (bpp == 2) { uint16_t u; memcpy(&u, p, 2); v = u / 65535.f; } else memcpy(&v, p, 4);
                                                  o[0] = o[1] = o[2] = u8f(hi > lo ? (v - lo) / (hi - lo) : 0); }
        else if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) { uint32_t v; memcpy(&v, p, 4); o[0] = u8f(f11(v & 2047)); o[1] = u8f(f11(v >> 11 & 2047)); o[2] = u8f(f10(v >> 22)); }
        else if (f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2B10G10R10_UINT_PACK32) { uint32_t v; memcpy(&v, p, 4); o[0] = (uint8_t)((v & 1023) >> 2); o[1] = (uint8_t)((v >> 10 & 1023) >> 2); o[2] = (uint8_t)((v >> 20 & 1023) >> 2); }
        else if (f == VK_FORMAT_R16G16B16A16_SFLOAT) { const uint16_t *h = (const uint16_t *)p; for (int k = 0; k < 3; k++) o[k] = u8f(half_to_f(h[k])); }
        else if (f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB) { o[0] = p[2]; o[1] = p[1]; o[2] = p[0]; }
        else if (bpp >= 4) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; }
        else if (bpp == 2) { o[0] = p[0]; o[1] = p[1]; o[2] = 0; }
        else { o[0] = o[1] = o[2] = p[0]; }
    }
    char dir[600], path[700]; snprintf(dir, sizeof dir, "%s/shots", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/%s.png", dir, tag);
    if (png_write(path, rgba, x->w, x->h)) VLOG("dumped texture %llu (%ux%u format %d%s) to %s", (unsigned long long)x->serial, x->w, x->h, f, depth ? ", depth scaled to its range" : "", path);
    free(rgba); vkFreeCommandBuffers(g_vk.dev, cp, 1, &cb); vkDestroyBuffer(g_vk.dev, b, NULL); vkFreeMemory(g_vk.dev, bm, NULL);
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    (void)period_s; (void)crop;
    uint64_t n = ++g_presents;
    if (exec_init()) { vk_tex *fx = tex_of(t, NULL); if (fx) vkd_front_present(fx, crop); }   // the app's screen (nvn_vk_front.c), when the app started this host
    if (exec_init() && shot_wanted(n)) {
        mtl_finish();   // the submissions before the present have run
        vk_tex *x = tex_of(t, NULL);
        { vk_tex *tg[96]; int nt = vkd_trace_targets(tg, 96); for (int k = 0; k < nt; k++) { char tag[64]; snprintf(tag, sizeof tag, "frame_%06llu_t%02d_s%llu", (unsigned long long)n, k, (unsigned long long)tg[k]->serial); vkx_dump(tg[k], tag); } }
        if (x) shot(x, n, crop); else VLOG("frame %llu: the presented texture has no host image (nothing rendered or copied into it yet)", (unsigned long long)n);
    }
    vkd_pcache_tick(n);
    char dline[256];
    if (n % 600 == 0 && g_exec_ok == 1) { uint64_t sh, dr; vkd_front_stats(&sh, &dr); if (sh || dr) VLOG("frame %llu: %llu frames shown on the app's screen, %llu dropped (the app was behind)", (unsigned long long)n, (unsigned long long)sh, (unsigned long long)dr); }
    if (n % 600 == 0 && g_exec_ok == 1) VLOG("frame %llu: %llu render passes, %llu clears, %llu copies, %llu uploads, %llu draws so far (%s)", (unsigned long long)n,
                                       (unsigned long long)g_stats_passes, (unsigned long long)g_stats_clears, (unsigned long long)g_stats_copies, (unsigned long long)g_stats_uploads, (unsigned long long)g_stats_draws, (vkd_stats_line(dline, sizeof dline), dline));
    if (g_opstat > 0 && n % 300 == 0) {
        char line[2048]; int k = 0;
        for (int i = 0; i <= OP_COUNT_ && k < (int)sizeof line - 32; i++) { uint64_t c = atomic_exchange(&g_opn[i], 0); if (c) k += snprintf(line + k, sizeof line - k, " %d:%.1f", i, c / 300.0); }
        VLOG("commands per frame by type up to present %llu:%s", (unsigned long long)n, line);
    }
    return n;
}
void mtl_present_wait(uint64_t ticket) { (void)ticket; }
