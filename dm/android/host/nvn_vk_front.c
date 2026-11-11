// The Android host's screen: each presented frame, as the window crops it, drawn into an Android hardware buffer the app shows on its SurfaceView (dm_front_proto.h for
// the link). The host stays its own process (the app starts it): the rewritten game code finds its thread slots at fixed offsets from the thread pointer, which a library
// loaded into the app's process cannot promise (its thread-local block is allocated per thread at run time), so the frames cross the process boundary instead.
//   - buffers: DMF_MAX_BUFFERS at the size of the first presented crop, R8G8B8A8, imported into Vulkan (VK_ANDROID_external_memory_android_hardware_buffer) as colour
//     targets; the presented image is stretched into one with the stretch pipeline (nvn_vk_draw.c), after every submission before the present (the timeline semaphore)
//   - a frame goes to the app once the GPU has drawn it (a sender thread waits on its fence); a buffer comes back free when the app has a newer one on screen
//   - no free buffer (the app is behind): the frame is dropped for the screen, the game is never held up
//   - sRGB: the bytes are copied as they are (an sRGB image is read through its UNORM twin): the app shows the bytes on a non-sRGB surface, so they stay encoded
// Controller input from the app goes to player 1 (dm_pad_set, hle_hid.c).
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <android/hardware_buffer.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)
#include "dm_front_proto.h"
#include "nvn_vk_int.h"
#include "lsfg_glue.h"   // Lossless Scaling frame generation (dm/third_party/lsfg-eden, GPL-3.0-or-later): DM_LSFG=1

void dm_pad_set(uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry);   // hle_hid.c
void dm_background(int on); void dm_set_opmode(int m);   // hle_oe.c

enum { FB_FREE, FB_DRAWING, FB_SHOWN };
typedef struct { AHardwareBuffer *ahb; VkImage img; VkDeviceMemory mem; VkImageView view, src; VkFence fence; VkCommandBuffer cb; VkDescriptorSet ds; int state; } fbuf;
static fbuf B[DMF_MAX_BUFFERS]; static int g_nb; static uint32_t g_w, g_h;
static int g_fd = -1, g_dead; static time_t g_try_at;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
// buffers drawn on the GPU, waiting for their fence (the sender thread): which buffer, whose fence says it is drawn (a generated frame shares its real frame's),
// and whether it is a generated frame (sent first; its real frame follows half a frame interval later)
typedef struct { int i, fence_of, gen; } qent;
static qent g_queue[16]; static int g_qn;
static pthread_cond_t g_text_cv = PTHREAD_COND_INITIALIZER;   // the guest thread waiting for the app's text answer (vkd_front_text)
static int g_text_done, g_text_ok; static char g_text_ans[DMF_TEXT_TEXT];
static VkCommandPool g_cp; static VkDescriptorPool g_dp; static VkPipeline g_pipe; static VkPipelineLayout g_pl; static VkDescriptorSetLayout g_dsl; static VkSampler g_samp;
static uint64_t g_shown, g_dropped, g_generated;
static uint64_t g_half_ns = 16666667;   // half the real frames' interval (frame generation's pacing), from their present times
// DM_LSFG=1: the user's Lossless Scaling frame generation ($DM_ROOT/lossless/Lossless.dll, imported in the launcher's settings), set up at the first present
static int lsfg_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_LSFG"); on = e && e[0] == '1' ? lsfg_init(g_vk.dev, g_vk.pd, g_vk.mem_model, g_vk.null_desc, g_vk.fp16) : 0; }
    return on;
}

static int send_msg(const dmf_msg *m) { return send(g_fd, m, sizeof *m, MSG_NOSIGNAL) == (ssize_t)sizeof *m; }
static int send_raw(const void *p, size_t n) { return g_fd >= 0 && send(g_fd, p, n, MSG_NOSIGNAL) == (ssize_t)n; }
static void *reader(void *unused) {
    (void)unused;
    union { dmf_msg m; dmf_text_msg t; } rb;   // the text messages are much bigger than dmf_msg
    for (;;) {
        ssize_t n = recv(g_fd, &rb, sizeof rb, 0);
        if (n <= 0) { VLOG("the app's screen closed the link (%s): frames are no longer shown", n ? strerror(errno) : "closed");
                      pthread_mutex_lock(&g_mu); g_dead = 1; pthread_cond_broadcast(&g_text_cv); pthread_mutex_unlock(&g_mu); return NULL; }
        if (rb.m.type == DMF_TEXT_DONE) {   // the app's user answered the game's keyboard: hand it to the waiting guest thread
            if (n < (ssize_t)sizeof(rb.t)) continue;
            pthread_mutex_lock(&g_mu);
            if (!g_text_done) {   // an answer nobody waits for is dropped
                g_text_ok = rb.t.ok ? 1 : 0;
                if (g_text_ok) { strncpy(g_text_ans, rb.t.text, sizeof g_text_ans - 1); g_text_ans[sizeof g_text_ans - 1] = 0; }
                g_text_done = 1; pthread_cond_broadcast(&g_text_cv);
            }
            pthread_mutex_unlock(&g_mu);
            continue;
        }
        if (n != (ssize_t)sizeof(rb.m)) continue;
        dmf_msg m = rb.m;
        if (m.type == DMF_RELEASE && m.index < (uint32_t)g_nb) { pthread_mutex_lock(&g_mu); if (B[m.index].state == FB_SHOWN) B[m.index].state = FB_FREE; pthread_mutex_unlock(&g_mu); }
        else if (m.type == DMF_INPUT) dm_pad_set(m.buttons, m.lx, m.ly, m.rx, m.ry);
        else if (m.type == DMF_PAUSE) {   // the pause menu: no input while it is open, then the trip to the background (it waits for the GPU work in flight)
            static int paused; int on = m.index != 0;
            if (on) lsfg_reset();   // frame generation starts its history again after the pause
            if (on != paused) { paused = on; if (on) dm_pad_set(0, 0, 0, 0, 0); VLOG("pause menu %s", on ? "opened: the game goes to the background" : "closed: the game resumes"); dm_background(on); }
        } else if (m.type == DMF_OPMODE) dm_set_opmode(m.index ? 1 : 0);
    }
}
static void *sender(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_qn) pthread_cond_wait(&g_cv, &g_mu);
        qent q = g_queue[0]; memmove(g_queue, g_queue + 1, sizeof *g_queue * (size_t)--g_qn);
        pthread_mutex_unlock(&g_mu);
        int i = q.i;
        VkResult r = vkWaitForFences(g_vk.dev, 1, &B[q.fence_of].fence, VK_TRUE, 5000000000ull);
        if (r) { VRED_ONCE("a frame for the screen did not finish drawing (%d)", r); }
        if (!q.gen) vkResetFences(g_vk.dev, 1, &B[q.fence_of].fence);   // a generated frame leaves its real frame's fence to that frame's own turn
        {   // frame generation's pacing: the generated frame goes out at once, its real frame half a frame interval later
            static struct timespec gen_at; static int have_gen;
            if (q.gen) { clock_gettime(CLOCK_MONOTONIC, &gen_at); have_gen = 1; }
            else if (have_gen) {
                have_gen = 0; struct timespec t = gen_at; t.tv_nsec += (long)g_half_ns; while (t.tv_nsec >= 1000000000L) { t.tv_nsec -= 1000000000L; t.tv_sec++; }
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
            }
        }
        dmf_msg m = {DMF_FRAME, (uint32_t)i};
        pthread_mutex_lock(&g_mu);
        int ok = !g_dead && !r && send_msg(&m);
        B[i].state = ok ? FB_SHOWN : FB_FREE; if (ok) g_shown++;
        pthread_mutex_unlock(&g_mu);
    }
}
static int link_up(void) {
    if (g_fd >= 0) return !g_dead;
    const char *name = getenv("DM_FRONT_SOCKET"); if (!name || !*name) return 0;
    time_t now = time(NULL); if (now == g_try_at) return 0; g_try_at = now;   // at most one try a second
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0); if (fd < 0) return 0;
    struct sockaddr_un a; memset(&a, 0, sizeof a); a.sun_family = AF_UNIX; size_t n = strlen(name); if (n > sizeof a.sun_path - 2) n = sizeof a.sun_path - 2;
    memcpy(a.sun_path + 1, name, n);
    if (connect(fd, (struct sockaddr *)&a, (socklen_t)(offsetof(struct sockaddr_un, sun_path) + 1 + n))) { close(fd); VRED_ONCE("the app's screen (socket %s) is not there: %s", name, strerror(errno)); return 0; }
    g_fd = fd;
    pthread_t t; pthread_create(&t, NULL, reader, NULL); pthread_detach(t);
    pthread_create(&t, NULL, sender, NULL); pthread_detach(t);
    VLOG("linked to the app's screen (%s)", name);
    return 1;
}
static int buffers_make(uint32_t w, uint32_t h) {
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID props = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(g_vk.dev, "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (!props) { VLOG("[RED] the driver has no Android hardware buffer import: nothing is shown"); return 0; }
    g_pipe = vkd_stretch_get(VK_FORMAT_R8G8B8A8_UNORM, &g_pl, &g_dsl); g_samp = vkd_linear_sampler();
    if (!g_pipe) { VLOG("[RED] no stretch pipeline for the screen"); return 0; }
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, g_vk.qfam};
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, DMF_MAX_BUFFERS};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, DMF_MAX_BUFFERS, 1, &ps};
    if (vkCreateCommandPool(g_vk.dev, &cpi, NULL, &g_cp) || vkCreateDescriptorPool(g_vk.dev, &dpi, NULL, &g_dp)) return 0;
    for (int i = 0; i < DMF_MAX_BUFFERS; i++) {
        fbuf *b = &B[i];
        AHardwareBuffer_Desc d = {w, h, 1, AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE, 0, 0, 0};
        if (AHardwareBuffer_allocate(&d, &b->ahb)) { VLOG("[RED] screen buffer %ux%u could not be allocated", w, h); return 0; }
        VkAndroidHardwareBufferFormatPropertiesANDROID fp = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
        VkAndroidHardwareBufferPropertiesANDROID pr = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, &fp};
        if (props(g_vk.dev, b->ahb, &pr)) return 0;
        VkExternalMemoryImageCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,   // transfer source: frame generation copies each frame (always allowed on hardware buffers)
                                VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &b->img)) { VLOG("[RED] screen buffer image failed"); return 0; }
        VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, NULL, b->ahb};
        VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, &imp, b->img, VK_NULL_HANDLE};
        uint32_t mt = 0; while (mt < 32 && !(pr.memoryTypeBits >> mt & 1)) mt++;
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ded, pr.allocationSize, mt};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &b->mem) || vkBindImageMemory(g_vk.dev, b->img, b->mem, 0)) { VLOG("[RED] screen buffer memory failed"); return 0; }
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, b->img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dp, 1, &g_dsl};
        if (vkCreateImageView(g_vk.dev, &vi, NULL, &b->view) || vkCreateFence(g_vk.dev, &fi, NULL, &b->fence) || vkAllocateCommandBuffers(g_vk.dev, &cai, &b->cb) ||
            vkAllocateDescriptorSets(g_vk.dev, &dai, &b->ds)) return 0;
    }
    dmf_msg m = {DMF_BUFFERS, DMF_MAX_BUFFERS, w, h};
    if (!send_msg(&m)) return 0;
    for (int i = 0; i < DMF_MAX_BUFFERS; i++) {
        dmf_msg bm = {DMF_BUFFER, (uint32_t)i, w, h};
        if (!send_msg(&bm) || AHardwareBuffer_sendHandleToUnixSocket(B[i].ahb, g_fd)) { VLOG("[RED] screen buffer %d could not be handed to the app", i); return 0; }
    }
    g_nb = DMF_MAX_BUFFERS; g_w = w; g_h = h;
    VLOG("screen: %d buffers of %ux%u handed to the app", g_nb, w, h);
    return 1;
}
// one picture stretched into screen buffer t: src (sampled in layout GENERAL) through rect (u, v, width, height of the source), with the buffer taken from the app
// and, when release is set, handed back to it (with frame generation on, the caller copies the real frame first and hands it back itself)
static void draw_into(VkCommandBuffer cb, fbuf *t, VkImageView src, const float rect[4], int release) {
    VkDescriptorImageInfo di = {g_samp, src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, t->ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &di};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    VkImageMemoryBarrier in = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                               VK_QUEUE_FAMILY_FOREIGN_EXT, g_vk.qfam, t->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &in);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, t->view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {g_w, g_h}}, 1, 0, 1, &ca, NULL, NULL};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl, 0, 1, &t->ds, 0, NULL);
    vkCmdPushConstants(cb, g_pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, rect);
    VkViewport v = {0, 0, (float)g_w, (float)g_h, 0, 1}; VkRect2D sc = {{0, 0}, {g_w, g_h}};
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    if (!release) return;
    VkImageMemoryBarrier out = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                g_vk.qfam, VK_QUEUE_FAMILY_FOREIGN_EXT, t->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &out);
}
static VkFormat unorm_twin(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM : f == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : f; }
void vkd_front_present(vk_tex *x, const int crop[4]) {
    if (!link_up()) return;
    int cx = 0, cy = 0, cw = (int)x->w, ch = (int)x->h;
    if (crop && crop[2] > 0 && crop[3] > 0 && crop[0] >= 0 && crop[1] >= 0 && (uint32_t)(crop[0] + crop[2]) <= x->w && (uint32_t)(crop[1] + crop[3]) <= x->h) { cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3]; }
    if (!g_nb) { static int failed; if (failed || !buffers_make((uint32_t)cw, (uint32_t)ch)) { failed = 1; return; } }
    pthread_mutex_lock(&g_mu);
    int i = -1; for (int k = 0; k < g_nb; k++) if (B[k].state == FB_FREE) { i = k; break; }
    if (i >= 0) B[i].state = FB_DRAWING;
    int dead = g_dead;
    pthread_mutex_unlock(&g_mu);
    if (dead) return;
    if (i < 0) { g_dropped++; return; }
    fbuf *b = &B[i];
    if (b->src) { vkDestroyImageView(g_vk.dev, b->src, NULL); b->src = VK_NULL_HANDLE; }   // its last frame finished (the buffer was free)
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, VK_IMAGE_VIEW_TYPE_2D, unorm_twin(x->fmt), {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(g_vk.dev, &vi, NULL, &b->src)) { vi.format = x->fmt; if (vkCreateImageView(g_vk.dev, &vi, NULL, &b->src)) { b->state = FB_FREE; VRED_ONCE("screen: no view of the presented image"); return; } }
    VkCommandBuffer cb = b->cb; vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
    float rect[4] = {(float)cx / (float)x->w, (float)cy / (float)x->h, (float)cw / (float)x->w, (float)ch / (float)x->h};
    if (vkd_origin_ll()) { rect[1] += rect[3]; rect[3] = -rect[3]; }   // a lower-left origin stores the picture bottom-up (as the Mac's present flips it)
    const int lsfg = lsfg_on();
    draw_into(cb, b, b->src, rect, !lsfg);
    int gi = -1;
    if (lsfg) {   // frame generation: this frame into its history, and the frame halfway from the last one into a second buffer when one is free
        VkImageMemoryBarrier tg = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, b->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tg);
        if (lsfg_frame(cb, b->img, VK_FORMAT_R8G8B8A8_UNORM, g_w, g_h)) {
            pthread_mutex_lock(&g_mu);
            for (int k = 0; k < g_nb; k++) if (k != i && B[k].state == FB_FREE) { gi = k; B[k].state = FB_DRAWING; break; }
            pthread_mutex_unlock(&g_mu);
            if (gi >= 0) { const float full[4] = {0, 0, 1, 1}; draw_into(cb, &B[gi], lsfg_output(), full, 1); g_generated++; }
        }
        VkImageMemoryBarrier rel = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_READ_BIT, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                    g_vk.qfam, VK_QUEUE_FAMILY_FOREIGN_EXT, b->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &rel);
    }
    vkEndCommandBuffer(cb);
    {   // the real frames' interval, for frame generation's pacing (gaps over 100 ms, a load or a pause, are left out)
        static uint64_t last; struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); uint64_t now = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
        if (last && now - last < 100000000ull) g_half_ns = (g_half_ns * 7 + (now - last) / 2) / 8;
        last = now;
    }
    // after every submission so far (the frame's work), as the game presents it
    uint64_t wait = vkx_last_fence(); VkSemaphore sem = vkx_timeline(); VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 1, &wait, 0, NULL};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, wait ? 1u : 0u, &sem, &ws, 1, &cb, 0, NULL};
    if (vkx_submit(&si, b->fence)) { pthread_mutex_lock(&g_mu); b->state = FB_FREE; if (gi >= 0) B[gi].state = FB_FREE; pthread_mutex_unlock(&g_mu); VRED_ONCE("screen: submit failed"); return; }
    pthread_mutex_lock(&g_mu);
    if (gi >= 0) { if (g_qn < 15) g_queue[g_qn++] = (qent){gi, i, 1}; else B[gi].state = FB_FREE; }   // the generated frame first
    if (g_qn < 16) g_queue[g_qn++] = (qent){i, i, 0};
    pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
}
void vkd_front_stats(uint64_t *shown, uint64_t *dropped) { *shown = g_shown; *dropped = g_dropped; }

// The game's keyboard (hle_swkbd.c) on the guest thread: the request crosses to the app, which shows its text box with the system keyboard, and the thread waits here
// until the app's answer arrives through the reader. The game truncates the answer to max_chars (UTF-16 units) on the way back into its string anyway.
int vkd_front_text(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, int min_chars, char *out, int out_cap) {
    if (!link_up()) return -1;
    dmf_text_msg t; memset(&t, 0, sizeof t);
    t.type = DMF_TEXT; t.max_chars = max_chars > 0 ? (uint32_t)max_chars : 0; t.min_chars = min_chars;
    strncpy(t.title, title ? title : "", sizeof t.title - 1);
    strncpy(t.sub, sub ? sub : "", sizeof t.sub - 1);
    strncpy(t.guide, guide ? guide : "", sizeof t.guide - 1);
    strncpy(t.initial, initial ? initial : "", sizeof t.initial - 1);
    pthread_mutex_lock(&g_mu);
    if (g_dead || !send_raw(&t, sizeof t)) { pthread_mutex_unlock(&g_mu); return -1; }
    g_text_done = 0; g_text_ans[0] = 0;
    while (!g_text_done) pthread_cond_wait(&g_text_cv, &g_mu);   // the reader sets g_text_done; it also wakes us when the link dies
    int ok = g_text_ok;
    if (ok && out && out_cap > 0) { size_t len = strlen(g_text_ans); if (len > (size_t)out_cap - 1) len = (size_t)out_cap - 1; memcpy(out, g_text_ans, len); out[len] = 0; }
    pthread_mutex_unlock(&g_mu);
    return ok ? 1 : 0;
}
