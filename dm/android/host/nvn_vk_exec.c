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
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)
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
static int stages_on(void);   // DM_VK_RING_STAGES (the ring capture's section)
static int g_tex_held;
static uint64_t g_n_clear_sc, g_n_clear_mask, g_n_clear_empty;   // clears limited to a scissor, colour clears with a partial channel mask (not honoured), clears with an empty scissor (skipped)        // tex_of holds g_tex_mu while it makes an image: an end_rendering then takes no stage snapshot (its lookup would wait on the lock)
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
static _Atomic uint64_t g_img_bytes, g_img_live;
static uint64_t g_n_evict, g_n_unevict, g_evict_bytes;   // tex_evict: images given up, made again, and the bytes given up   // the host's texture images (tex_create / tex_free_gpu), for the memory line
// The memory line (every 600 frames): what Android's low-memory killer weighs. The process from /proc/self/status, the GPU driver's own count (kgsl), the host's images and kept
// SPIR-V, and what the system has left. Fortune's Weave was killed for low memory after its tutorial (exit reason 3, importance 100) at about 5.3 GB
static long kb_of(const char *path, const char *key) {   // a "Key: N kB" line's N, or a file holding one number (bytes: / 1024); -1 when there is none
    FILE *f = fopen(path, "r"); if (!f) return -1; char l[256]; long v = -1;
    if (!key) { long long b; if (fscanf(f, "%lld", &b) == 1) v = (long)(b / 1024); }
    else while (fgets(l, sizeof l, f)) if (!strncmp(l, key, strlen(key)) && l[strlen(key)] == ':') { v = atol(l + strlen(key) + 1); break; }
    fclose(f); return v;
}
static void img_split(uint64_t *owned, uint64_t *cpu);
static void mem_line(uint64_t n) {
    char p1[96], p2[96]; snprintf(p1, sizeof p1, "/sys/class/kgsl/kgsl/proc/%d/gpumem_mapped", (int)getpid()); snprintf(p2, sizeof p2, "/sys/class/kgsl/kgsl/proc/%d/gpumem_unmapped", (int)getpid());
    long rss = kb_of("/proc/self/status", "VmRSS"), anon = kb_of("/proc/self/status", "RssAnon"), shm = kb_of("/proc/self/status", "RssShmem"), swp = kb_of("/proc/self/status", "VmSwap");
    long gm = kb_of(p1, NULL), gu = kb_of(p2, NULL), avail = kb_of("/proc/meminfo", "MemAvailable");
    uint64_t own, cpu; img_split(&own, &cpu);
    VLOG("[mem] frame %llu: process RSS %ld MB (anon %ld, shared %ld, swapped %ld); GPU driver mapped %ld + unmapped %ld MB; host images %llu MB (%llu live; GPU-written %llu MB, CPU content %llu MB; evicted %llu, %llu MB, made again %llu), arenas %llu MB, kept SPIR-V %llu MB; system available %ld MB",
         (unsigned long long)n, rss / 1024, anon / 1024, shm / 1024, swp / 1024, gm / 1024, gu / 1024, (unsigned long long)(atomic_load(&g_img_bytes) >> 20), (unsigned long long)atomic_load(&g_img_live), (unsigned long long)(own >> 20), (unsigned long long)(cpu >> 20), (unsigned long long)g_n_evict, (unsigned long long)(g_evict_bytes >> 20), (unsigned long long)g_n_unevict,
         (unsigned long long)(vkd_arena_bytes() >> 20),
         (unsigned long long)(vkd_words_bytes() >> 20), avail / 1024);
}


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
// ---------- Frame timing report (every 300 presents): where a frame's time goes ----------
// recording: the CPU turning the game's command lists into Vulkan commands (exec_list); slot wait: waiting for the GPU to finish an older submission before its
// command buffer can be reused (the GPU is behind); screen: handing the frame to the app (vkd_front_present); GPU busy: from each submission's queue time (or the
// previous one's completion) to its completion as the completion thread sees it, so an estimate a little high by that thread's wake-up latency
static _Atomic uint64_t g_t_rec, g_t_slot, g_t_front, g_t_gpu; static uint64_t g_last_done_ns;
static uint64_t now_ns_mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// ---------- Slow-submit breakdown (DM_VK_SLOW_SUB, ms; default 60): a submission that took longer than the threshold reports wall vs this thread's CPU ----------
// wall >> CPU: the submit thread was descheduled (the game's loading threads); wall ~ CPU: the driver calls themselves. Phases: slot wait, record, commit;
// and inside record: texture creates (1), descriptor writes (2), pipeline creates (3), the usual suspects at an area transition
static uint64_t g_sb_t0, g_sb_cpu0, g_sb_slot, g_sb_rec, g_sb_commit, g_sb_ns[5]; static unsigned g_sb_n[5];
static uint64_t cpu_ns(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
uint64_t vkx_now_ns(void) { return now_ns_mono(); }
// what: 1 texture create, 2 descriptor writes, 3 pipeline create, 4 shader module create. Also kept as running totals for hle_nvn.c's frame statistics (vkx_create_stats)
static _Atomic uint64_t g_cr_n[5], g_cr_ns[5];
void vkx_sb_note(int what, uint64_t ns) { if (what > 0 && what < 5) { g_sb_ns[what] += ns; g_sb_n[what]++; g_cr_n[what]++; g_cr_ns[what] += ns; } }
void vkx_create_stats(uint64_t *mods, uint64_t *pipes, uint64_t *mods_ns, uint64_t *pipes_ns) {
    if (mods) *mods = g_cr_n[4]; if (pipes) *pipes = g_cr_n[3]; if (mods_ns) *mods_ns = g_cr_ns[4]; if (pipes_ns) *pipes_ns = g_cr_ns[3];
}
// ---------- GPU time per frame (mtl_gpu_ms: hle_nvn.c's timing line and slow-frame log) ----------
// Each submission writes a timestamp when the GPU starts it and one when it ends; the completion thread adds the difference to the frame it was recorded in. A frame
// counts once all its submissions are done (pending 0), as nvn_mtl.m's gpu_tag_begin / gpu_tag_done do with Metal's command buffer times. Time between submissions
// (the GPU idle, waiting for the next one) is not counted. Off when the queue has no timestamps (timestampValidBits 0)
enum { GRING = 64 };
static struct { _Atomic uint64_t frame, ns; _Atomic int pending; } g_gpuf[GRING];
static VkQueryPool g_ts_pool; static int g_ts_on = -1; static uint64_t g_ts_mask; static _Atomic uint64_t g_vk_frame;
void vkx_set_frame(uint64_t f) { atomic_store(&g_vk_frame, f); }
uint64_t vkx_frame(void) { return atomic_load(&g_vk_frame); }
void vkx_sb_time(int what, uint64_t ns) { if (what > 0 && what < 5) { g_sb_ns[what] += ns; g_cr_ns[what] += ns; } }   // time only, no new item (a wait for a pipeline a worker builds)
static int ts_on(void) {
    if (g_ts_on >= 0) return g_ts_on;
    g_ts_on = 0;
    uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &n, NULL);
    VkQueueFamilyProperties *qf = calloc(n ? n : 1, sizeof *qf); vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &n, qf);
    uint32_t bits = g_vk.qfam < n ? qf[g_vk.qfam].timestampValidBits : 0; free(qf);
    if (!bits || g_vk.lim.timestampPeriod <= 0) { VLOG("GPU time per frame: off (the queue has no timestamps)"); return 0; }
    g_ts_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NSUB, 0};
    if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &g_ts_pool)) { VLOG("[RED] GPU time per frame: no query pool"); return 0; }
    return g_ts_on = 1;
}
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpuf[k].frame) != f) { atomic_store(&g_gpuf[k].frame, f); atomic_store(&g_gpuf[k].ns, 0); atomic_store(&g_gpuf[k].pending, 0); }
    atomic_fetch_add(&g_gpuf[k].pending, 1);
}
static void gpu_tag_done(sub_t *s) {   // completion thread: the submission's two timestamps, read once it has completed
    if (!s->ts) return;
    s->ts = 0; unsigned k = s->frame % GRING; uint32_t q = (uint32_t)(s - g_sub) * 2; uint64_t t[2] = {0, 0};
    VkResult r = vkGetQueryPoolResults(g_vk.dev, g_ts_pool, q, 2, sizeof t, t, sizeof t[0], VK_QUERY_RESULT_64_BIT);
    if (atomic_load(&g_gpuf[k].frame) != s->frame) return;
    if (r == VK_SUCCESS) atomic_fetch_add(&g_gpuf[k].ns, (uint64_t)((double)((t[1] - t[0]) & g_ts_mask) * (double)g_vk.lim.timestampPeriod));
    atomic_fetch_sub(&g_gpuf[k].pending, 1);
}
double vkx_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (g_ts_on != 1 || atomic_load(&g_gpuf[k].frame) != f || atomic_load(&g_gpuf[k].pending)) return -1;
    return atomic_load(&g_gpuf[k].ns) / 1e6;
}
// texture uploads (the slow-frame log): first uploads, re-uploads of a texture the CPU wrote again, bytes, and the CPU time of recording them
static _Atomic uint64_t g_up_first, g_up_again, g_up_bytes, g_up_ns;
void vkx_upload_stats(uint64_t *first, uint64_t *again, uint64_t *bytes) { if (first) *first = g_up_first; if (again) *again = g_up_again; if (bytes) *bytes = g_up_bytes; }
uint64_t vkx_upload_ns_take(void) { return atomic_exchange(&g_up_ns, 0); }
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
        { uint64_t d = now_ns_mono(), st = g_sub[want % NSUB].submit_ns; if (st < g_last_done_ns) st = g_last_done_ns; if (st && d > st) g_t_gpu += d - st; g_last_done_ns = d; }
        gpu_tag_done(&g_sub[want % NSUB]);
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
void barrier_compute(VkCommandBuffer cb) {   // a dispatch's writes visible to what follows: waits on compute only, so a chain of dispatches does not drain the graphics pipeline each time
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
void mtl_sub_begin(void) {
    if (!exec_init() || g_cur) return;
    sub_t *s = &g_sub[(g_fence + 1) % NSUB];
    g_sb_t0 = now_ns_mono(); g_sb_cpu0 = cpu_ns(); g_sb_slot = g_sb_rec = g_sb_commit = 0;
    memset(g_sb_ns, 0, sizeof g_sb_ns); memset(g_sb_n, 0, sizeof g_sb_n);
    if (s->value) {   // this slot's previous submission must be done (and its effects applied) before its command buffer is reused
        uint64_t w0 = atomic_load(&g_applied) < s->value ? now_ns_mono() : 0;
        pthread_mutex_lock(&g_done_mu);
        while (atomic_load(&g_applied) < s->value) pthread_cond_wait(&g_done_cv, &g_done_mu);
        pthread_mutex_unlock(&g_done_mu);
        if (w0) { uint64_t d = now_ns_mono() - w0; g_t_slot += d; g_sb_slot = d; }
    }
    vkResetCommandBuffer(s->cb, 0); s->nlog = 0; s->log[0] = 0; s->ncrumb = 0;
    if (g_crumb_map) *g_crumb_map = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(s->cb, &bi);
    barrier_all(s->cb);   // the CPU's writes to pool memory before this submission (coherent memory: a host-to-device barrier is all it needs)
    s->ts = 0;
    if (ts_on()) {   // GPU time per frame: the start timestamp (the end one is written at commit)
        uint32_t q = (uint32_t)(s - g_sub) * 2; s->frame = atomic_load(&g_vk_frame);
        vkCmdResetQueryPool(s->cb, g_ts_pool, q, 2); vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_ts_pool, q);
        gpu_tag_begin(s->frame); s->ts = 1;
    }
    s->open = 1; g_cur = s;
}
static int wait_applied(uint64_t fence, uint64_t timeout_ns);
uint64_t mtl_sub_commit(void) {
    if (!exec_init()) return nvn_null_commit();
    if (!g_cur) mtl_sub_begin();
    sub_t *s = g_cur;
    uint64_t c0 = now_ns_mono();
    end_rendering();
    if (s->ts) vkCmdWriteTimestamp(s->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_ts_pool, (uint32_t)(s - g_sub) * 2 + 1);
    vkEndCommandBuffer(s->cb);
    s->value = __atomic_add_fetch(&g_fence, 1, __ATOMIC_RELEASE); s->open = 0; g_cur = NULL; vkd_sub_committed(s->value);
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &s->value};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, 1, &s->cb, 1, &g_sem};
    s->submit_ns = now_ns_mono();
    pthread_mutex_lock(&g_q_mu); VkResult r = vkQueueSubmit(g_vk.q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_q_mu);
    if (r) VRED_ONCE("vkQueueSubmit failed (%d)", r);
    pthread_mutex_lock(&g_done_mu); atomic_store(&g_fence_pub, s->value); pthread_cond_signal(&g_work_cv); pthread_mutex_unlock(&g_done_mu);
    g_sb_commit = now_ns_mono() - c0;
    {
        static int slow_ms = -1; if (slow_ms < 0) { const char *e = getenv("DM_VK_SLOW_SUB"); slow_ms = e ? atoi(e) : 60; }
        uint64_t wall = now_ns_mono() - g_sb_t0;
        if (slow_ms && wall > (uint64_t)slow_ms * 1000000ull) {
            uint64_t cpu = cpu_ns() - g_sb_cpu0;
            VLOG("slow submit %llu: %.0f ms wall, %.0f ms CPU (slot %.0f, record %.0f, commit %.0f); tex %u/%.0f ms, desc %u/%.0f ms, pipe %u/%.0f ms:%.300s",
                 (unsigned long long)s->value, wall / 1e6, cpu / 1e6, g_sb_slot / 1e6, g_sb_rec / 1e6, g_sb_commit / 1e6,
                 g_sb_n[1], g_sb_ns[1] / 1e6, g_sb_n[2], g_sb_ns[2] / 1e6, g_sb_n[3], g_sb_ns[3] / 1e6, s->log);
        }
    }
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
    // Cube-compatible: any square, single-sample 2D image of at least six layers, not only the cube targets. The sampling instruction, not the texture's target, picks the
    // dimension: Fire Emblem Fortune's Weave keeps its point-light and light-probe cubes as six layers of a 2D array and samples them as cubes (sample_view makes the cube view).
    // Without the flag that view is invalid Vulkan (VUID-VkImageViewCreateInfo-image-01003); the Adreno accepted it and the start screen's 3D scene came out nearly black.
    // As Eden for this game (Eglantine bc234e54aac5, ported from eden-fefw a1b23ac3a2: "e2DArray views had no cube handle, so those lights sampled the null view")
    if (x->type == VK_IMAGE_TYPE_2D && x->layers >= 6 && x->w == x->h && x->d.samples <= 1) cf |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (x->type == VK_IMAGE_TYPE_3D && (u & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))) cf |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkSampleCountFlagBits sc = x->d.samples >= 8 ? VK_SAMPLE_COUNT_8_BIT : x->d.samples >= 4 ? VK_SAMPLE_COUNT_4_BIT : x->d.samples >= 2 ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, cf, x->type, x->fmt, {x->w, x->h, x->depth}, x->levels, x->layers, sc,
                            VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(g_vk.dev, &ii, NULL, &x->img);
    if (r) { x->failed = 1; VLOG("[RED] vkCreateImage %d for texture %llu (format %#x → %d, %ux%ux%u, %u levels, %u layers, target %d)", r, (unsigned long long)x->serial, x->d.format, x->fmt, x->w, x->h, x->depth, x->levels, x->layers, tg); return 0; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, x->img, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
    if ((r = vkAllocateMemory(g_vk.dev, &mai, NULL, &x->dm)) || (r = vkBindImageMemory(g_vk.dev, x->img, x->dm, 0))) {
        x->failed = 1; VLOG("[RED] texture memory %d (%llu bytes)", r, (unsigned long long)mr.size); vkDestroyImage(g_vk.dev, x->img, NULL); x->img = VK_NULL_HANDLE; return 0; }
    x->dbytes = mr.size; atomic_fetch_add(&g_img_bytes, mr.size); atomic_fetch_add(&g_img_live, 1);
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
        uint64_t tc0 = now_ns_mono(); g_tex_held = 1; tex_create(x, cb); g_tex_held = 0; vkx_sb_note(1, now_ns_mono() - tc0);
    }
    else if (x && cb && !x->img && !x->failed) {   // evicted (tex_evict): made again, and filled from its storage at the next read
        uint64_t tc0 = now_ns_mono(); g_tex_held = 1; tex_create(x, cb); g_tex_held = 0; vkx_sb_note(1, now_ns_mono() - tc0); g_n_unevict++;
    }
    if (x && cb) x->last_ns = now_ns_mono();
    pthread_mutex_unlock(&g_tex_mu);
    return x && !x->failed && x->img ? x : NULL;
}
static void tex_free_gpu(void *arg) {
    vk_tex *x = arg;
    if (x->img) { vkd_tex_release(x); vkDestroyImage(g_vk.dev, x->img, NULL); vkFreeMemory(g_vk.dev, x->dm, NULL); atomic_fetch_sub(&g_img_bytes, x->dbytes); atomic_fetch_sub(&g_img_live, 1); }
    free(x);
}
// Eviction of idle images whose content can be made again. A texture's image holds a copy of what its storage (game memory) holds, unless the GPU has written it since
// (gpu_owned: a render target, a copy or clear destination), and the host made one per texture serial for as long as the serial lived. A game that reuses its texture
// memory without finalizing the old texture objects leaves their serials, and the images, behind: in Fire Emblem Fortune's Weave the host's images grew from 1.9 to
// 4.6 GB in 3 minutes of a battle (7,698 of them, more than the game's 2.7 GB of pools hold), and Android's low-memory killer stopped it after its tutorial.
// An image of CPU content no recording has looked up for DM_VK_TEX_EVICT_S seconds (default 20; 0 keeps every image) is given up; the next lookup makes it again and
// uploads its storage, as at first use. Images the GPU wrote are never given up (their content exists nowhere else)
static void tex_evict(void) {
    static double secs = -1; if (secs < 0) { const char *e = getenv("DM_VK_TEX_EVICT_S"); secs = e ? atof(e) : 20; VLOG("idle texture images of CPU content given up after %g s (DM_VK_TEX_EVICT_S; 0 = never)", secs); }
    if (secs <= 0) return;
    uint64_t now = now_ns_mono(), lim = (uint64_t)(secs * 1e9); int n = 0; uint64_t bytes = 0;
    pthread_mutex_lock(&g_tex_mu);
    for (int i = 0; i < TBUCKETS; i++) for (vk_tex *x = g_tex[i]; x; x = x->next) {
        if (!x->img || x->gpu_owned || x->failed || !x->last_ns || now - x->last_ns < lim || x->wd || x->ws) continue;
        vk_tex *dead = malloc(sizeof *dead); if (!dead) break;
        *dead = *x; dead->next = NULL;   // the image, its memory and its views go once the GPU can no longer use them (tex_free_gpu)
        x->img = VK_NULL_HANDLE; x->dm = VK_NULL_HANDLE; x->sviews = NULL; x->dbytes = 0; x->uploaded = 0; x->dirty = 1; x->lin_sub = 0;
        bytes += dead->dbytes; n++;
        vkx_defer(tex_free_gpu, dead);
    }
    pthread_mutex_unlock(&g_tex_mu);
    g_n_evict += (uint64_t)n; g_evict_bytes += bytes;
}
// Bytes of the live images the GPU wrote (render targets and other GPU-made content) and of those holding CPU content (the memory line)
static void img_split(uint64_t *owned, uint64_t *cpu) {
    *owned = *cpu = 0;
    pthread_mutex_lock(&g_tex_mu);
    for (int i = 0; i < TBUCKETS; i++) for (vk_tex *x = g_tex[i]; x; x = x->next) if (x->img) { if (x->gpu_owned) *owned += x->dbytes; else *cpu += x->dbytes; }
    pthread_mutex_unlock(&g_tex_mu);
}
void mtl_tex_init(const tex_t *t) { (void)t; }   // images are made at first use, when the texture's command buffer is open
// A draw that samples an image its own render pass draws into (a feedback loop) is undefined on a tiled GPU without the feedback-loop layout this driver lacks;
// runs hung the GPU in the game's first 3D frame, and ending the pass after every draw made it go away. Such a draw reads a copy instead (as emulators do)
vk_tex *vkx_clone(sub_t *s, vk_tex *x, int keep) {
    vk_tex *c = calloc(1, sizeof *c);
    c->serial = x->serial; c->d = x->d; c->L = x->L; c->mem = NULL;
    uint64_t tc0 = now_ns_mono(); int tok = tex_create(c, s->cb); vkx_sb_note(1, now_ns_mono() - tc0);
    if (!tok) { free(c); return NULL; }
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
static void tex_upload_do(vk_tex *x, sub_t *s);
static void tex_upload(vk_tex *x, sub_t *s) {   // counted for the slow-frame log (vkx_upload_stats)
    if (x->gpu_owned || !x->dirty || !x->mem) return;
    uint64_t t0 = now_ns_mono(), n0 = x->up_n;
    tex_upload_do(x, s);
    if (x->up_n != n0) { if (n0) g_up_again++; else g_up_first++; g_up_bytes += x->linear ? x->L.size : x->dbytes; g_up_ns += now_ns_mono() - t0; }
}
static void tex_upload_do(vk_tex *x, sub_t *s) {
    if (x->gpu_owned || !x->dirty || !x->mem) return;
    if (x->d.samples > 1) { x->dirty = 0; return; }
    if (stages_on()) {   // DM_VK_RING_STAGES: an upload in the frame, and whether an image the GPU drew holds the same guest memory (its content is not what is uploaded)
        uint64_t over = 0; uintptr_t a0 = (uintptr_t)x->mem;
        pthread_mutex_lock(&g_tex_mu);
        for (int i = 0; i < TBUCKETS && !over; i++) for (vk_tex *y = g_tex[i]; y && !over; y = y->next)
            if (y != x && y->gpu_owned && y->img && y->mem && a0 >= (uintptr_t)y->mem && a0 < (uintptr_t)y->mem + (y->dbytes ? y->dbytes : 1)) over = y->serial;
        pthread_mutex_unlock(&g_tex_mu);
        vkx_ring_note("UPLOAD tex %llu %ux%u fmt %d at %#llx%s%llu", (unsigned long long)x->serial, x->w, x->h, x->fmt, (unsigned long long)a0, over ? " OVER-GPU-IMAGE tex " : "", (unsigned long long)over);
    }
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
        VkBuffer sb; uint64_t soff = 0; uint8_t *map = vkd_stage_alloc(s, total ? total : 1, &sb, &soff);   // suballocated: a scene's first frame uploads thousands of textures, and a vkAllocateMemory each cost a driver round trip
        if (!map) { VRED_ONCE("no staging memory for texture uploads"); return; }
        VkBufferImageCopy *cs = calloc((size_t)x->levels * x->layers, sizeof *cs); int nc = 0; uint64_t o = 0;
        VkImageAspectFlags asp = (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect;
        for (unsigned l = 0; l < x->levels; l++) for (unsigned ly = 0; ly < x->layers; ly++) {
            uint32_t w = level_dim(x->w, l), h = level_dim(x->h, l), dz = level_dim(x->depth, l);
            uint32_t bw = (w + x->bw - 1) / x->bw, bh = (h + x->bh - 1) / x->bh; size_t pitch = (size_t)bw * x->bpb;
            if (nvn_deswizzle(&x->d, l, ly, x->mem, map + o, pitch)) { VRED_ONCE("texture %llu (format %#x, target %d): deswizzle refused", (unsigned long long)x->serial, x->d.format, x->d.target); continue; }
            cs[nc++] = (VkBufferImageCopy){soff + o, bw * x->bw, bh * x->bh, {asp, l, ly, 1}, {0, 0, 0}, {w, h, dz}};
            o += pitch * bh * dz;
        }
        if (nc) { barrier_all(cb); vkCmdCopyBufferToImage(cb, sb, x->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nc, cs); barrier_all(cb); }
        free(cs);
        if (x->bpb == 16 && x->bw > 1) {   // what this upload copied: all-zero blocks decode as ASTC's error colour (magenta)
            uint64_t l0 = (uint64_t)((x->w + x->bw - 1) / x->bw) * ((x->h + x->bh - 1) / x->bh) * x->depth * x->layers * 16;   // level 0, every layer (the copy is level-major)
            uint32_t z = 0, nb = (uint32_t)(o / 16), z0 = 0, n0 = (uint32_t)(l0 < o ? l0 / 16 : o / 16);
            for (uint32_t i = 0; i < nb; i++) { const uint64_t *q = (const uint64_t *)(map + (size_t)i * 16); if (!(q[0] | q[1])) { z++; if (i < n0) z0++; } }
            x->up_zero = z; x->up_blocks = nb; x->up_l0_zero = z0; x->up_l0_blocks = n0;
        }
    }
    x->up_sub = g_fence + 1; x->up_n++;
    x->dirty = 0; x->uploaded = 1; g_stats_uploads++;
    vkx_sublog(s, " U%llu:%s%d:%ux%ux%u/%ul/%uy", (unsigned long long)x->serial, x->linear ? "lin" : "bl", x->fmt, x->w, x->h, x->depth, x->levels, x->layers);
}
// The texture is about to be written by the GPU: bring in its CPU content first unless it is already the GPU's (a partial write keeps the rest)
static void wait_land(vk_tex *x, sub_t *s);
vk_tex *tex_for_write(const tex_t *t, sub_t *s) {
    vk_tex *x = tex_of(t, s->cb); if (!x) return NULL;
    if (x->wd | x->ws) wait_land(x, s);   // a clear still waiting for this image's pass comes first (begin_rendering has taken it already when it becomes the pass's load operation)
    if (!x->gpu_owned && !x->uploaded) tex_upload(x, s);
    x->gpu_owned = 1; x->dirty = 0;
    return x;
}
// A linear texture lies over pool memory the CPU may write at any time without telling NVN (the video codec's native images: hle_codec_video.c pool_native_write writes
// each decoded picture there). On the Mac it is made on the pool's buffer and sees every write; here it is an image of its own, so it is copied again from the pool
// the first time each submission reads it (a GPU copy), unless the GPU itself has written it
vk_tex *tex_for_read(const tex_t *t, sub_t *s) {
    vk_tex *x = tex_of(t, s->cb);
    if (x && (x->wd | x->ws)) wait_land(x, s);
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
// ---------- Clears as load operations (DM_VK_LOAD_CLEAR=0: off, every clear a vkCmdClearAttachments in a pass opened with LOAD, as before) ----------
// A clear of a whole target issued while no pass is open is kept (g_rt.pc / pd / ps) and becomes the load operation of the pass that opens next over these targets:
// loadOp CLEAR, the form a driver handles in full. A clear drawn inside a pass that loaded its attachments (vkCmdClearAttachments) clears the image but, on Qualcomm's
// Adreno driver, not the low-resolution depth it rejects fragments with (LRZ): the next pass tested against the previous frame's depth. Fire Emblem Fortune's Weave's
// depth pre-pass lost its floor wherever the new frame's depth was farther than the frame before's (every one of 1,883 hole pixels, none of 9,600 nearer or equal ones:
// logs_mem3/001546 frames 1405-1407), on camera cuts and fast pans; the G-buffer, drawn with depth EQUAL, then stayed empty there. As BananaNative's vka_rp_clear does.
// Also saves loading attachments that are cleared anyway (a tiler copies them into its tile memory first).
// The kept clears land when the pass opens for a draw, or, with no draw, when the pass would end: a change of targets, a copy, a dispatch, the submission's end.
static int load_clear_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_LOAD_CLEAR"); v = !(e && e[0] == '0'); VLOG("clears of whole targets %s", v ? "become their pass's load operation" : "are drawn inside the pass (DM_VK_LOAD_CLEAR=0)"); } return v; }
static uint64_t g_n_load_clear, g_n_clear_only_pass;
static void pend_reset(void) { g_rt.npend = 0; memset(g_rt.pc_on, 0, sizeof g_rt.pc_on); g_rt.pd_on = g_rt.ps_on = 0; }
// the pass extent these targets give (as begin_rendering): 0 when an attachment has no image yet (the clear then takes the old path, which makes it)
static int pend_extent(VkExtent2D *e) {
    uint32_t w = ~0u, h = ~0u; int any = 0;
    for (int i = 0; i < g_rt.rt.n && i < 8; i++) {
        if (!g_rt.rt.color[i]) continue;
        vk_tex *x = tex_of(g_rt.rt.color[i], NULL); if (!x) return 0;
        uint32_t l = g_rt.rt.has_cview[i] && g_rt.rt.cview[i].set_levels ? g_rt.rt.cview[i].base_level : 0;
        if (level_dim(x->w, l) < w) w = level_dim(x->w, l); if (level_dim(x->h, l) < h) h = level_dim(x->h, l); any = 1;
    }
    if (g_rt.rt.depth) {
        vk_tex *x = tex_of(g_rt.rt.depth, NULL); if (!x) return 0;
        uint32_t l = g_rt.rt.has_dview && g_rt.rt.dview.set_levels ? g_rt.rt.dview.base_level : 0;
        if (level_dim(x->w, l) < w) w = level_dim(x->w, l); if (level_dim(x->h, l) < h) h = level_dim(x->h, l); any = 1;
    }
    if (!any) return 0;
    *e = (VkExtent2D){w, h}; return 1;
}
// whether a clear issued now can wait for the pass: no pass open, the switch on, the has-targets state known, and the scissor covering the whole target
static int pend_can(void) {
    if (g_rt.active || !g_rt.has_rt || g_tex_held || !load_clear_on()) return 0;
    VkExtent2D e; if (!pend_extent(&e)) return 0;
    g_rt.ext = e;   // vkd_clear_rect clamps to it (no pass is open: nothing else reads it)
    VkRect2D r; int part = 0;
    return vkd_clear_rect(&r, &part) && !part;
}
// ---------- A depth clear waits on its image for the pass that draws with it (DM_VK_WAIT_CLEAR=0: off, it lands in a pass that only clears, as the colour clears do) ----------
// Fire Emblem Fortune's Weave clears its scene depth, changes targets (G-buffer clears, shadow maps) and only then draws the depth pre-pass. The kept clear then landed
// at the change of targets as a pass with no draw, and the pre-pass opened later with LOAD. On Qualcomm's Adreno driver that pre-pass lost the near part of the floor in
// single frames (2 of 149 in logs_mem3/001307: frames 1161 and 1182; the depth image itself cleared, the commands the same as the frames around): the fragments were
// rejected by something other than the image's depth, as with the clear drawn inside a pass before (the driver's low-resolution depth; why only in some frames is
// not known). So a depth / stencil clear that would land alone stays on the image (vk_tex wd / ws) and is the load operation of the next pass that has the image as its
// depth attachment, at the same level and layer and covering it whole: clear and draws in one pass, the form every driver sees most. Anything else that uses the image
// first (a shader reading it, a copy, the upscaler: tex_for_read / tex_for_write) lands the clear before it (wait_land: a pass of that image alone).
static int wait_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_WAIT_CLEAR"); v = !(e && e[0] == '0'); VLOG("depth clears with no draw %s", v ? "wait for their image's next pass" : "land in a pass that only clears (DM_VK_WAIT_CLEAR=0)"); } return v; }
static uint64_t g_n_wait, g_n_wait_used, g_n_wait_landed, g_n_ct_depth, g_n_ct_depth_xfer;
static void dview_of(uint32_t *l, uint32_t *y) { *l = g_rt.rt.has_dview && g_rt.rt.dview.set_levels ? g_rt.rt.dview.base_level : 0; *y = g_rt.rt.has_dview && g_rt.rt.dview.set_layers ? g_rt.rt.dview.min_layer : 0; }
// whether a pass of the bound targets covers the whole of its depth image's level
static int wait_covers(vk_tex *x, uint32_t l) { VkExtent2D e; return pend_extent(&e) && e.width == level_dim(x->w, l) && e.height == level_dim(x->h, l); }
// the bound targets' kept depth / stencil clear moves onto the image; 1 when it did
static int wait_keep(void) {
    if (!(g_rt.pd_on || g_rt.ps_on) || !g_rt.rt.depth || !wait_on()) return 0;
    vk_tex *x = tex_of(g_rt.rt.depth, NULL); uint32_t l, y; dview_of(&l, &y);
    if (!x || !x->img || !wait_covers(x, l)) return 0;
    if ((x->wd | x->ws) && (x->wl != l || x->wy != y)) return 0;   // one of another level or layer waits already: this one lands as before (and that one before it, tex_for_write)
    if (g_rt.pd_on) { x->wd = 1; x->wdv = g_rt.pd; }
    if (g_rt.ps_on) { x->ws = 1; x->wsv = g_rt.ps; }
    x->wl = l; x->wy = y; g_rt.pd_on = g_rt.ps_on = 0; g_n_wait++;
    int any = 0; for (int i = 0; i < 8; i++) any |= g_rt.pc_on[i];
    g_rt.npend = any;
    vkx_ring_note("CLEAR depth tex %llu waits for its next pass", (unsigned long long)x->serial);
    return 1;
}
// a waiting clear lands now, in a pass of the image alone (something other than a pass over it is about to use the image)
static void wait_land_kick(VkCommandBuffer cb, vk_tex *x);
static void wait_land(vk_tex *x, sub_t *s) {
    if (!x || !(x->wd | x->ws)) return;
    if (!s || !x->img || g_tex_held) { if (!g_tex_held) x->wd = x->ws = 0; return; }
    if (g_rt.active) end_rendering();
    view_t v; memset(&v, 0, sizeof v); v.set_levels = 1; v.base_level = (uint8_t)x->wl; v.nlevels = 1; v.set_layers = 1; v.min_layer = (uint16_t)x->wy; v.nlayers = 1;
    VkImageView dv = make_view(s, x, &v, 1, x->aspect);
    int wd = x->wd && (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), ws = x->ws && (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    x->wd = x->ws = 0;
    if (!dv || !(wd || ws)) return;
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, dv, VK_IMAGE_LAYOUT_GENERAL}, sa = da;
    da.loadOp = wd ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE; da.clearValue.depthStencil.depth = x->wdv;
    sa.loadOp = ws ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE; sa.clearValue.depthStencil.stencil = x->wsv;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {level_dim(x->w, x->wl), level_dim(x->h, x->wl)}}, 1, 0, 0, NULL,
                          (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? &da : NULL, (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? &sa : NULL};
    vkCmdBeginRendering(s->cb, &ri); wait_land_kick(s->cb, x); vkCmdEndRendering(s->cb); barrier_all(s->cb); g_stats_passes++;
    x->gpu_owned = 1; x->dirty = 0; g_n_wait_landed++; g_n_clear_only_pass++;
    vkx_ring_note("CLEAR depth tex %llu lands alone (its image is used before a pass over it)", (unsigned long long)x->serial);
}
// ---------- DM_VK_GPU_PROF=1: where the GPU's time goes ----------
// A GPU timestamp before and after every render pass and every compute dispatch, read back eight frames later and summed per label (a pass: its first colour target,
// or its depth target, and size; a dispatch: its shader); every 600 frames the log lists the labels by GPU time per frame. A tiled GPU does a pass's work when the pass
// ends, so the times hold per pass (and per dispatch), not per draw. The intervals do not add up to the frame's GPU time: copies, clears outside passes and the time
// the GPU waits are not in them. Needs host query reset and timestamps on the queue; off by default.
enum { PROF_POOLS = 8, PROF_Q = 4096, PROF_LAB = 512 };
typedef struct { VkQueryPool pool; int n; uint16_t lab[PROF_Q / 2]; } prof_pool;
static prof_pool g_prof[PROF_POOLS]; static int g_prof_cur = -1, g_prof_open = -1, g_prof_state = -1;
static struct { char name[48]; double ns; uint64_t n; } g_prof_lab[PROF_LAB]; static int g_prof_nlab; static uint64_t g_prof_frames, g_prof_lost;
static int prof_on(void) {
    if (g_prof_state < 0) {
        const char *e = getenv("DM_VK_GPU_PROF"); g_prof_state = e && e[0] == '1';
        if (g_prof_state && !g_vk.host_qreset) { g_prof_state = 0; VLOG("DM_VK_GPU_PROF: the driver has no host query reset: no GPU profile"); }
        if (g_prof_state) {
            VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, PROF_Q};
            for (int i = 0; i < PROF_POOLS; i++) { if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &g_prof[i].pool)) { g_prof_state = 0; break; } vkResetQueryPool(g_vk.dev, g_prof[i].pool, 0, PROF_Q); }
            if (g_prof_state) { g_prof_cur = 0; VLOG("DM_VK_GPU_PROF: GPU time per render pass and compute dispatch, listed every 600 frames (timestamp period %.1f ns)", g_vk.lim.timestampPeriod); }
        }
    }
    return g_prof_state;
}
static int prof_label(const char *name) {
    for (int i = 0; i < g_prof_nlab; i++) if (!strcmp(g_prof_lab[i].name, name)) return i;
    if (g_prof_nlab >= PROF_LAB) return PROF_LAB - 1;
    snprintf(g_prof_lab[g_prof_nlab].name, sizeof g_prof_lab[0].name, "%s", name); return g_prof_nlab++;
}
void vkx_prof_begin(VkCommandBuffer cb, const char *fmt, ...) {
    if (!prof_on() || g_prof_open >= 0) return;
    prof_pool *p = &g_prof[g_prof_cur]; if (p->n + 2 > PROF_Q) { g_prof_lost++; return; }
    char name[48]; va_list ap; va_start(ap, fmt); vsnprintf(name, sizeof name, fmt, ap); va_end(ap);
    p->lab[p->n / 2] = (uint16_t)prof_label(name); g_prof_open = p->n;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->pool, (uint32_t)p->n);
}
void vkx_prof_end(VkCommandBuffer cb) {
    if (g_prof_state != 1 || g_prof_open < 0) return;
    prof_pool *p = &g_prof[g_prof_cur]; vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p->pool, (uint32_t)g_prof_open + 1); p->n = g_prof_open + 2; g_prof_open = -1;
}
// at a frame's end: the next pool's results (written PROF_POOLS - 1 frames ago) are added up, and the pool starts again
static void prof_frame(uint64_t n) {
    if (!prof_on()) return;
    g_prof_cur = (g_prof_cur + 1) % PROF_POOLS; prof_pool *p = &g_prof[g_prof_cur]; g_prof_open = -1;
    if (p->n) {
        static uint64_t res[PROF_Q * 2];
        VkResult r = vkGetQueryPoolResults(g_vk.dev, p->pool, 0, (uint32_t)p->n, sizeof(uint64_t) * 2 * (size_t)p->n, res, sizeof(uint64_t) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r == VK_SUCCESS || r == VK_NOT_READY) for (int i = 0; i + 1 < p->n; i += 2) {
            if (!res[i * 2 + 1] || !res[i * 2 + 3] || res[i * 2 + 2] < res[i * 2]) { g_prof_lost++; continue; }
            int l = p->lab[i / 2]; g_prof_lab[l].ns += (double)(res[i * 2 + 2] - res[i * 2]) * (double)g_vk.lim.timestampPeriod; g_prof_lab[l].n++;
        }
        g_prof_frames++; vkResetQueryPool(g_vk.dev, p->pool, 0, PROF_Q); p->n = 0;
    }
    if (n % 600 == 0 && g_prof_frames) {
        int order[PROF_LAB]; for (int i = 0; i < g_prof_nlab; i++) order[i] = i;
        for (int i = 0; i < g_prof_nlab; i++) for (int j = i + 1; j < g_prof_nlab; j++) if (g_prof_lab[order[j]].ns > g_prof_lab[order[i]].ns) { int t = order[i]; order[i] = order[j]; order[j] = t; }
        double tot = 0; for (int i = 0; i < g_prof_nlab; i++) tot += g_prof_lab[i].ns;
        VLOG("gpu profile, frame %llu: %.2f ms a frame inside passes and dispatches over %llu frames (%d labels, %llu intervals lost)", (unsigned long long)n, tot / 1e6 / (double)g_prof_frames, (unsigned long long)g_prof_frames, g_prof_nlab, (unsigned long long)g_prof_lost);
        for (int i = 0; i < g_prof_nlab && i < 24; i++) { int l = order[i]; if (g_prof_lab[l].ns <= 0) break;
            VLOG("gpu profile:  %6.2f ms a frame  %5.1f%%  %-40s %6.1f times a frame, %.3f ms each", g_prof_lab[l].ns / 1e6 / (double)g_prof_frames, 100.0 * g_prof_lab[l].ns / tot, g_prof_lab[l].name,
                 (double)g_prof_lab[l].n / (double)g_prof_frames, g_prof_lab[l].ns / 1e6 / (double)(g_prof_lab[l].n ? g_prof_lab[l].n : 1)); }
        for (int i = 0; i < g_prof_nlab; i++) { g_prof_lab[i].ns = 0; g_prof_lab[i].n = 0; } g_prof_frames = 0;
    }
}
// ---------- The early-depth kick (DM_VK_LRZ_KICK=0: off) ----------
// Qualcomm's Adreno 830 driver rejects fragments early against a low-resolution copy of the depth image (LRZ) that it does not always bring up to date with the image:
// Fire Emblem Fortune's Weave's depth pre-pass lost the near floor in single frames, with the image itself cleared and whichever way the clear was given (drawn in the
// pass, a pass of its own, the pre-pass's load operation: logs_mem3/001307 and the play tests after). The work folder's re/lrztest reproduces a stale copy on the Odin 3
// (a transfer clear after a pass has cleared the image: 288 of 300 frames with holes, 514,560 of 518,400 pixels) and shows what makes the driver stop using it: a draw
// with depth writes on whose compare runs the other way from the pass's own draws. The driver keeps the copy for one direction, and drops it for the image when the
// direction turns, until a pass clears the image again. NEVER, or the same draw with depth writes off, does not do it; a fragment shader that discards everything does
// not prevent it. So every pass with a depth attachment begins with two draws of one pixel that discard every fragment, depth writes on: LESS, then GREATER (one of them
// is the other way, whichever way the pass draws; each at the depth that cannot pass its own compare, since the driver wrote the depth of a draw that could, discard or not; lrztest methods f and g: no frame with holes in 300, with shadow passes, copies, samples and four frames in flight
// too). Nothing is written. The cost is the early rejection itself: the driver falls back to the per-pixel depth test.
#include "nvn_vk_lrz_spv.h"
static int lrz_kick_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_LRZ_KICK"); v = !(e && e[0] == '0'); VLOG("depth passes %s", v ? "begin with the early-depth kick (two one-pixel draws that turn the compare direction)" : "do not get the early-depth kick (DM_VK_LRZ_KICK=0)"); } return v; }
typedef struct { uint32_t n; VkFormat cf[8]; VkFormat df; uint32_t asp, samples; VkPipeline p[2]; int failed; } lrz_pipe;
static lrz_pipe g_lrzp[48]; static int g_nlrzp; static VkPipelineLayout g_lrz_pl; static VkShaderModule g_lrz_vs, g_lrz_fs; static uint64_t g_n_lrz_kick, g_n_lrz_skip;
static lrz_pipe *lrz_pipe_of(uint32_t n, const VkFormat *cf, VkFormat df, VkImageAspectFlags asp, uint32_t samples) {
    for (int i = 0; i < g_nlrzp; i++) { lrz_pipe *e = &g_lrzp[i]; if (e->n == n && e->df == df && e->asp == asp && e->samples == samples && !memcmp(e->cf, cf, sizeof e->cf)) return e->failed ? NULL : e; }
    if (g_nlrzp >= 48) return NULL;
    lrz_pipe *e = &g_lrzp[g_nlrzp++]; memset(e, 0, sizeof *e); e->n = n; memcpy(e->cf, cf, sizeof e->cf); e->df = df; e->asp = asp; e->samples = samples;
    if (!g_lrz_pl) {
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; VkShaderModuleCreateInfo mv = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_lrz_vert, k_lrz_vert}, mf = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_lrz_frag, k_lrz_frag};
        if (vkCreatePipelineLayout(g_vk.dev, &li, NULL, &g_lrz_pl) || vkCreateShaderModule(g_vk.dev, &mv, NULL, &g_lrz_vs) || vkCreateShaderModule(g_vk.dev, &mf, NULL, &g_lrz_fs)) { g_lrz_pl = VK_NULL_HANDLE; e->failed = 1; return NULL; }
    }
    for (int k = 0; k < 2; k++) {
        VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_lrz_vs, "main"}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_lrz_fs, "main"}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)samples};
        VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, VK_TRUE, VK_TRUE, k ? VK_COMPARE_OP_GREATER : VK_COMPARE_OP_LESS}; ds.maxDepthBounds = 1;
        VkPipelineColorBlendAttachmentState ba[8]; memset(ba, 0, sizeof ba);   // no colour writes
        VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, n, ba};
        VkDynamicState dy[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dn = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dy};
        VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, n, e->cf, (asp & VK_IMAGE_ASPECT_DEPTH_BIT) ? df : VK_FORMAT_UNDEFINED, (asp & VK_IMAGE_ASPECT_STENCIL_BIT) ? df : VK_FORMAT_UNDEFINED};
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, 2, st, &vi, &ia, NULL, &vp, &rs, &ms, &ds, &cb, &dn, g_lrz_pl};
        VkResult r = vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &e->p[k]);
        if (r) { e->failed = 1; VLOG("[RED] early-depth kick pipeline failed (%d): %u colour targets, depth format %d, %u samples", r, n, df, samples); return NULL; }
    }
    return e;
}
// at the start of a pass (just begun) with a depth attachment: see above
static void lrz_kick(VkCommandBuffer cb, uint32_t n, const VkFormat *cf, VkFormat df, VkImageAspectFlags asp, uint32_t samples, uint32_t w, uint32_t h) {
    if (!(asp & VK_IMAGE_ASPECT_DEPTH_BIT) || !w || !h || !lrz_kick_on()) return;
    lrz_pipe *e = lrz_pipe_of(n, cf, df, asp, samples); if (!e) { g_n_lrz_skip++; return; }
    VkViewport v = {0, 0, (float)w, (float)h, 0, 1}; VkRect2D sc = {{0, 0}, {1, 1}};
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, e->p[0]); vkCmdDraw(cb, 3, 1, 0, 1);   // LESS at depth 1 (the first instance is the depth, shaders/lrz.vert): cannot pass
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, e->p[1]); vkCmdDraw(cb, 3, 1, 0, 0);   // GREATER at depth 0: cannot pass
    g_n_lrz_kick++;
}
static void wait_land_kick(VkCommandBuffer cb, vk_tex *x) { VkFormat none[8]; memset(none, 0, sizeof none); uint32_t sm = x->d.samples >= 8 ? 8 : x->d.samples >= 4 ? 4 : x->d.samples >= 2 ? 2 : 1; lrz_kick(cb, 0, none, x->fmt, x->aspect, sm, level_dim(x->w, x->wl), level_dim(x->h, x->wl)); }
void end_rendering(void) {
    if (!g_rt.active && g_rt.npend && g_cur && !g_rt.in_begin && !g_tex_held) {   // kept clears and no draw
        int kept = wait_keep();                                                    // the depth clear: onto its image, for the pass that draws with it
        if (g_rt.npend) {                                                          // the colour clears: a pass that only clears (without the depth image when its clear waits)
            g_rt.skip_ds = kept; int ok = begin_rendering(g_cur); g_rt.skip_ds = 0;
            if (ok) g_n_clear_only_pass++; else pend_reset();
        }
    }
    if (!g_rt.active || !g_cur) return;
    vkCmdEndRendering(g_cur->cb); g_rt.active = 0; vkx_prof_end(g_cur->cb);
    barrier_all(g_cur->cb);
    if (!g_tex_held && stages_on() && g_rt.rt.depth) vkx_stage_snap_depth(g_cur, tex_of(g_rt.rt.depth, NULL), "pass");
    if (!g_tex_held && stages_on()) for (int a = 0; a < g_rt.rt.n && a < 8; a++) if (g_rt.rt.color[a]) { char w[16]; snprintf(w, sizeof w, "pass c%d", a); vkx_stage_snap(g_cur, tex_of(g_rt.rt.color[a], NULL), w); }
    if (g_crumbs == 2) vkx_crumb(g_cur);
}
int begin_rendering(sub_t *s) {
    if (g_rt.active) return 1;
    if (!g_rt.has_rt) return 0;
    int td = 0, ts = 0; float tdv = 0; uint32_t tsv = 0;   // a clear waiting on the depth image: this pass's load operation when the pass covers the image, landed first otherwise
    if (g_rt.rt.depth && !g_rt.skip_ds) {
        vk_tex *wx = tex_of(g_rt.rt.depth, NULL);
        if (wx && (wx->wd | wx->ws)) {
            uint32_t l, y; dview_of(&l, &y);
            if (wx->wl == l && wx->wy == y && wait_covers(wx, l) && !g_tex_held) { td = wx->wd; tdv = wx->wdv; ts = wx->ws; tsv = wx->wsv; wx->wd = wx->ws = 0; g_n_wait_used++; }
            else wait_land(wx, s);
        }
    }
    g_rt.in_begin = 1;   // an upload of an attachment below ends "the pass": the kept clears are not landed from there
    VkRenderingAttachmentInfo ca[8]; memset(ca, 0, sizeof ca); VkFormat kf[8]; memset(kf, 0, sizeof kf); uint32_t ksamples = 1;   // kf, ksamples: for lrz_kick
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    uint32_t w = ~0u, h = ~0u; int n = g_rt.rt.n > 8 ? 8 : g_rt.rt.n, any = 0;
    for (int i = 0; i < n; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        vk_tex *x = g_rt.rt.color[i] ? tex_for_write(g_rt.rt.color[i], s) : NULL;
        if (!x || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) continue;
        ca[i].imageView = make_view(s, x, &g_rt.rt.cview[i], g_rt.rt.has_cview[i], VK_IMAGE_ASPECT_COLOR_BIT);
        if (g_rt.pc_on[i]) { ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca[i].clearValue.color = g_rt.pc[i]; g_n_load_clear++; }
        if (ca[i].imageView) kf[i] = x->fmt; if (x->d.samples > 1) ksamples = (uint32_t)x->d.samples;
        uint32_t l = g_rt.rt.has_cview[i] && g_rt.rt.cview[i].set_levels ? g_rt.rt.cview[i].base_level : 0;
        if (level_dim(x->w, l) < w) w = level_dim(x->w, l); if (level_dim(x->h, l) < h) h = level_dim(x->h, l); any = 1;
    }
    int has_d = 0, has_s = 0;
    vk_tex *dx = g_rt.rt.depth && !g_rt.skip_ds ? tex_for_write(g_rt.rt.depth, s) : NULL;
    if (dx && (dx->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
        VkImageView dv = make_view(s, dx, &g_rt.rt.dview, g_rt.rt.has_dview, dx->aspect);
        has_d = !!(dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT); has_s = !!(dx->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        da.imageView = sa.imageView = dv; da.imageLayout = sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (g_rt.pd_on && (dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) { da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.clearValue.depthStencil.depth = g_rt.pd; g_n_load_clear++; }
        if (g_rt.ps_on && (dx->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) { sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.clearValue.depthStencil.stencil = g_rt.ps; g_n_load_clear++; }
        if (td && !g_rt.pd_on && (dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) { da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.clearValue.depthStencil.depth = tdv; g_n_load_clear++; }      // the one that waited (a newer kept one wins)
        if (ts && !g_rt.ps_on && (dx->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) { sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.clearValue.depthStencil.stencil = tsv; g_n_load_clear++; }
        uint32_t l = g_rt.rt.has_dview && g_rt.rt.dview.set_levels ? g_rt.rt.dview.base_level : 0;
        if (level_dim(dx->w, l) < w) w = level_dim(dx->w, l); if (level_dim(dx->h, l) < h) h = level_dim(dx->h, l); any = 1;
    }
    g_rt.in_begin = 0;
    if (!any && !g_rt.want.width) return 0;
    if (!any) { w = g_rt.want.width; h = g_rt.want.height; n = 0; }   // a draw that only writes images or storage buffers: a pass with no attachments, the viewport's size
    g_rt.ext = (VkExtent2D){w, h};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_rt.ext}, 1, 0, (uint32_t)n, ca, has_d ? &da : NULL, has_s ? &sa : NULL};
    if (g_prof_state) { vk_tex *p0 = n > 0 && g_rt.rt.color[0] ? tex_of(g_rt.rt.color[0], NULL) : NULL; vkx_prof_begin(s->cb, "pass %s %llu %ux%u", p0 ? "colour" : "depth", (unsigned long long)(p0 ? p0->serial : dx ? dx->serial : 0), w, h); }
    vkCmdBeginRendering(s->cb, &ri); g_rt.active = 1; g_stats_passes++;
    pend_reset();   // the kept clears are this pass's load operations
    if (has_d && dx) { if (dx->d.samples > 1) ksamples = (uint32_t)dx->d.samples; lrz_kick(s->cb, (uint32_t)n, kf, dx->fmt, dx->aspect, ksamples >= 8 ? 8 : ksamples >= 4 ? 4 : ksamples >= 2 ? 2 : 1, w, h); }
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
// DM_VK_RING_FINAL=1 (with DM_VK_RING): the rolling capture keeps the picture the game presents (after upscaling, post-processing and the interface), not the scene
// before upscaling: a fault that is added after the scene (Fire Emblem Fortune's Weave's black tearing on distant buildings: none in 200 scene frames taken while it
// showed, and still there with the upscaler off) is in no scene capture. The presented texture is copied at the start of the next frame's first submission
static tex_t g_final_t; static int g_final_pend;
int vkx_ring_final(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RING_FINAL"); v = e && e[0] == '1'; if (v) VLOG("DM_VK_RING_FINAL: the rolling capture keeps the presented picture"); } return v; }
void mtl_exec(const uint8_t *p) {
    if (g_opstat < 0) g_opstat = getenv("DM_NVN_OPSTAT") != NULL;
    if (!exec_init()) { nvn_null_exec(p); return; }
    if (!g_cur) mtl_sub_begin();
    uint64_t t0 = now_ns_mono();
    if (g_final_pend) { g_final_pend = 0; vk_tex *fx = tex_of(&g_final_t, NULL); if (fx) vkx_ring_record(g_cur, fx, NULL); }   // DM_VK_RING_FINAL: the frame just presented, into the rolling capture
    exec_list(g_cur, p);
    uint64_t d = now_ns_mono() - t0; g_t_rec += d; g_sb_rec += d;
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
            case OP_SET_RT: end_rendering(); pend_reset(); memcpy(&g_rt.rt, a, sizeof g_rt.rt); g_rt.has_rt = 1; break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = (const p_clear_color *)a;
                if (c->index < 0 || c->index >= g_rt.rt.n || c->index >= 8 || !g_rt.rt.color[c->index]) break;
                if ((c->mask & 0xf) == 0xf && pend_can()) {   // the whole target, no pass open: the next pass's load operation (see end_rendering)
                    vk_tex *px = tex_of(g_rt.rt.color[c->index], NULL);
                    if (px && (px->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
                        g_rt.pc[c->index] = clear_value(px->fmt, c->rgba); g_rt.pc_on[c->index] = 1; g_rt.npend++; g_stats_clears++;
                        px->nclear++; memcpy(px->last_clear, c->rgba, sizeof px->last_clear);
                        vkx_ring_note("CLEAR colour tex %llu whole (load op)", (unsigned long long)px->serial);
                        break;
                    }
                }
                if (!begin_rendering(s)) break;
                vk_tex *x = tex_of(g_rt.rt.color[c->index], NULL); if (!x) break;
                VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {.color = clear_value(x->fmt, c->rgba)}};
                // NVN clears inside the scissor (vkd_clear_rect); this cleared the whole target, wiping what the game kept outside the rectangle. A partial channel mask is
                // not honoured yet (the Mac backend draws a triangle for it): logged
                VkClearRect rect = {{{0, 0}, g_rt.ext}, 0, 1}; int part = 0;
                if (!vkd_clear_rect(&rect.rect, &part)) { g_n_clear_empty++; break; }
                if (part && g_n_clear_sc++ < 30) VLOG("scissored colour clear: target %llu (index %d, %ux%u, fmt %d) inside %d,%d %ux%u to %.3f %.3f %.3f %.3f, frame %llu", (unsigned long long)x->serial, c->index,
                                                     g_rt.ext.width, g_rt.ext.height, x->fmt, rect.rect.offset.x, rect.rect.offset.y, rect.rect.extent.width, rect.rect.extent.height, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3], (unsigned long long)g_presents);
                if ((c->mask & 0xf) != 0xf && g_n_clear_mask++ < 30) VLOG("[RED] colour clear with channel mask %#x on target %llu (index %d, fmt %d): every channel is cleared, frame %llu", c->mask, (unsigned long long)x->serial, c->index, x->fmt, (unsigned long long)g_presents);
                vkx_ring_note("CLEAR colour tex %llu %d,%d %ux%u mask %#x", (unsigned long long)x->serial, rect.rect.offset.x, rect.rect.offset.y, rect.rect.extent.width, rect.rect.extent.height, c->mask);
                vkCmdClearAttachments(cb, 1, &ca, 1, &rect); g_stats_clears++;
                x->nclear++; memcpy(x->last_clear, c->rgba, sizeof x->last_clear);
                if (vkd_tracing()) VLOG("trace clear colour: target %llu (index %d, %ux%u) to %.4f %.4f %.4f %.4f", (unsigned long long)x->serial, c->index,
                                        g_rt.ext.width, g_rt.ext.height, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
                break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = (const p_clear_ds *)a;
                if (!g_rt.rt.depth) break;
                {   // the whole target: the load operation of the pass that opens next. A pass open now is ended for it, so a depth clear never is a draw inside a pass
                    // that loaded the depth (the driver's low-resolution depth would keep the old content, see end_rendering); a stencil-only clear inside a pass stays as it was
                    vk_tex *px = tex_of(g_rt.rt.depth, NULL);
                    int want_d = px && c->depth_mask && (px->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), want_s = px && c->stencil_mask && (px->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
                    if (px && (want_d || want_s) && (want_s ? (c->stencil_mask & 0xff) == 0xff : 1) && load_clear_on() && !g_tex_held) {
                        VkRect2D fr; int fpart = 0; int was = g_rt.active;
                        if (was ? (vkd_clear_rect(&fr, &fpart) && !fpart && want_d) : pend_can()) {
                            if (was) end_rendering();
                            if (pend_can()) {
                                if (want_d) { g_rt.pd = c->depth; g_rt.pd_on = 1; }
                                if (want_s) { g_rt.ps = (uint32_t)c->stencil; g_rt.ps_on = 1; }
                                g_rt.npend++; g_stats_clears++;
                                vkx_ring_note("CLEAR depth tex %llu whole to %.4f (load op%s)", (unsigned long long)px->serial, c->depth, was ? ", pass ended for it" : "");
                                break;
                            }
                        }
                    }
                }
                if (!begin_rendering(s)) break;
                vk_tex *x = tex_of(g_rt.rt.depth, NULL); if (!x) break;
                VkImageAspectFlags asp = 0;
                if (c->depth_mask && (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (c->stencil_mask && (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (!asp) break;
                VkClearAttachment ca = {asp, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
                VkClearRect rect = {{{0, 0}, g_rt.ext}, 0, 1}; int part = 0;   // inside the scissor, as the colour clear
                if (!vkd_clear_rect(&rect.rect, &part)) { g_n_clear_empty++; break; }
                if (part && g_n_clear_sc++ < 30) VLOG("scissored depth/stencil clear: target %llu (%ux%u, fmt %d) inside %d,%d %ux%u to depth %.4f stencil %d (masks %d %#x), frame %llu", (unsigned long long)x->serial,
                                                     g_rt.ext.width, g_rt.ext.height, x->fmt, rect.rect.offset.x, rect.rect.offset.y, rect.rect.extent.width, rect.rect.extent.height, c->depth, c->stencil, c->depth_mask, c->stencil_mask, (unsigned long long)g_presents);
                vkx_ring_note("CLEAR depth tex %llu %d,%d %ux%u to %.4f", (unsigned long long)x->serial, rect.rect.offset.x, rect.rect.offset.y, rect.rect.extent.width, rect.rect.extent.height, c->depth);
                vkCmdClearAttachments(cb, 1, &ca, 1, &rect); g_stats_clears++;
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
                } else if (nl == 1 && load_clear_on()) {
                    // nvnCommandBufferClearTexture of a depth image: a pass's load operation, never vkCmdClearDepthStencilImage. On Qualcomm's Adreno 830 driver that transfer
                    // clear leaves the image's low-resolution depth (LRZ) as it was once a render pass has cleared the image before: the next pass that loads it rejects
                    // every fragment the old content would have hidden. Measured with the work folder's re/lrztest on the Odin 3 (method 3 after method 0: 192 of 200
                    // frames with holes, 514,560 of 518,400 pixels at a cut to a farther view; alone, or after only copies into the image, none; a PC GPU: none), and none
                    // with the clear as a load operation (methods 0, 1, 5) or drawn in the pass (2). So the clear waits on the image for its next pass (wait_keep's
                    // mechanism), or lands in a pass of its own before anything else uses the image
                    if (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { x->wd = 1; x->wdv = c->c.f[0]; }
                    if (x->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { x->ws = 1; x->wsv = c->c.u[1]; }
                    x->wl = l; x->wy = ly; g_n_ct_depth++;
                    vkx_ring_note("CLEARTEX depth tex %llu level %u to %.4f (waits for its next pass)", (unsigned long long)x->serial, l, c->c.f[0]);
                    if (!wait_on()) wait_land(x, s);
                } else { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; vkCmdClearDepthStencilImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); g_n_ct_depth_xfer++;
                         VRED_ONCE("a depth texture cleared by a transfer clear (%u layers, or DM_VK_LOAD_CLEAR=0): Qualcomm's driver keeps its low-resolution depth stale", nl); }
                barrier_all(cb); g_stats_clears++; x->nclear++; vkx_sublog(s, " CT%llu:%d", (unsigned long long)x->serial, x->fmt);
                if (vkd_tracing()) VLOG("trace clear texture: %llu level %u to %.4f %.4f %.4f %.4f", (unsigned long long)x->serial, l, c->c.f[0], c->c.f[1], c->c.f[2], c->c.f[3]);
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
                } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && sx->fmt == dx->fmt && !(c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT))) {
                    VkImageCopy ic = {{sx->aspect, sl, sly, nly}, {c->sr.xoffset, c->sr.yoffset, sx->type == VK_IMAGE_TYPE_3D ? c->sr.zoffset : 0},
                                      {dx->aspect, dl, dly, nly}, {c->dr.xoffset, c->dr.yoffset, dx->type == VK_IMAGE_TYPE_3D ? c->dr.zoffset : 0},
                                      {(uint32_t)c->sr.width, (uint32_t)c->sr.height, sx->type == VK_IMAGE_TYPE_3D ? (uint32_t)(c->sr.depth > 0 ? c->sr.depth : 1) : 1}};
                    vkCmdCopyImage(cb, sx->img, VK_IMAGE_LAYOUT_GENERAL, dx->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
                } else if (!blit_ok(sx->fmt, dx->fmt, (c->flags & 1) && !is_depth(sx->fmt)) && nly == 1) {   // the driver offers no blit for these formats (depth, here): drawn
                    if (c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT)) VRED_ONCE("mirrored texture copy of a format the driver cannot blit (%d -> %d): drawn unmirrored", sx->fmt, dx->fmt);
                    int32_t ir[4] = {c->sr.xoffset, c->sr.yoffset, c->sr.width, c->sr.height}, orr[4] = {c->dr.xoffset, c->dr.yoffset, c->dr.width, c->dr.height};
                    vkd_stretch(s, sx, sl, sly, ir, dx, dl, dly, orr, (c->flags & 1) != 0);
                } else {
                    VkImageBlit bl = {{sx->aspect, sl, sly, nly}, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + c->sr.height, 1}},
                                      {dx->aspect, dl, dly, nly}, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
                    // NVN's MIRROR_X / MIRROR_Y copy flags: the blit walks the source from its far edge (a blit with reversed source offsets flips, VkImageBlit). They were ignored, and
                    // Fire Emblem Fortune's Weave's character card in the battle's defeat conditions came out upside down; the Mac's draw does the same (nvn_mtl.m copy_t2t_draw)
                    if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { int32_t t = bl.srcOffsets[0].x; bl.srcOffsets[0].x = bl.srcOffsets[1].x; bl.srcOffsets[1].x = t; }
                    if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { int32_t t = bl.srcOffsets[0].y; bl.srcOffsets[0].y = bl.srcOffsets[1].y; bl.srcOffsets[1].y = t; }
                    vkCmdBlitImage(cb, sx->img, VK_IMAGE_LAYOUT_GENERAL, dx->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, (c->flags & 1) && !is_depth(sx->fmt) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                }
                barrier_all(cb); g_stats_copies++; dx->ncopy++;
                vkx_sublog(s, " T2T%llu:%d:%dx%d>%llu:%d:%dx%d%s", (unsigned long long)sx->serial, sx->fmt, c->sr.width, c->sr.height, (unsigned long long)dx->serial, dx->fmt, c->dr.width, c->dr.height,
                           (c->flags & DM_COPY_RESOLVE) ? "R" : (c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT)) ? "M" : (c->sr.width == c->dr.width && c->sr.height == c->dr.height && sx->fmt == dx->fmt) ? "C" : "B");
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
static uint64_t g_shot_force;   // a present to dump because a trace was armed on demand (the trace_now file)
static int shot_wanted(uint64_t n) {
    static uint64_t list[64]; static int nl = -1, every = 0;
    if (nl < 0) {
        nl = 0; const char *e = getenv("DM_VK_SHOT"); const char *v = getenv("DM_VK_SHOT_EVERY"); every = v ? atoi(v) : 0;
        for (const char *q = e; q && *q && nl < 64; ) { list[nl++] = strtoull(q, (char **)&q, 10); while (*q == ',' || *q == ' ') q++; }
    }
    if (n == g_shot_force) return 1;
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

// ---------- DM_VK_RING=<frames>: a rolling capture, saved after the fact ----------
// Each frame the 3D scene before upscaling (the DLSS command's input colour, blitted to RGBA8) and its depth are copied into the next of <frames> host-visible slots,
// with the frame's draws and dispatches as text (vkx_ring_note). Touching $DM_ROOT/ring_save writes the slots, oldest first, to $DM_ROOT/ring/<frame>/: the moment
// has already happened when the trigger comes (Fire Emblem Fortune's Weave's broken floor lasts 2 frames at a hit, too short to start a capture for). About 5 MB a frame
#define STG_MAX 112
typedef struct { VkBuffer cb, db; VkDeviceMemory cm, dm; uint8_t *cmap, *dmap; uint32_t w, h, dbpp; uint64_t frame; int valid; char *notes; size_t nn;
                 VkBuffer sb; VkDeviceMemory sm; uint8_t *smap; int sn, sfill; char lab[STG_MAX][96]; VkBuffer hb; uint8_t *hmap; uint8_t hset[STG_MAX];                // sb..lab: the stage snapshots (DM_VK_RING_STAGES)
                 VkBuffer xb; VkDeviceMemory xm; uint8_t *xmap; uint64_t xn; int xfill, xd; char *xidx; size_t xin;
                 VkQueryPool op; VkBuffer ob; VkDeviceMemory om; uint8_t *omap; int on, ofill, osaved;
                 VkBuffer lb; VkDeviceMemory lm; uint8_t *lmap; uint64_t lcap; int lfill; char lnote[512]; uint32_t lw, lh; } ring_slot;   // lb..lh: DLSS inputs (DM_VK_RING_DLSS)   // op..osaved: per-draw pixel counts (DM_VK_RING_OCC)
//   // xb..xin: compute buffer dumps (DM_VK_RING_CS)
static ring_slot *g_ring; static int g_ring_n = -1, g_ring_i; static char *g_ring_cur; static size_t g_ring_cn, g_ring_cc;
static VkImage g_ring_img; static VkDeviceMemory g_ring_imem; static uint32_t g_ring_iw, g_ring_ih;
static int g_ring_half;   // DM_VK_RING_HALF=1: the scene at half size and no depth (0.6 MB a frame), for a longer window in the same memory
static int ring_on(void) {
    if (g_ring_n < 0) { const char *e = getenv("DM_VK_RING"); g_ring_n = e ? atoi(e) : 0; if (g_ring_n > 600) g_ring_n = 600; e = getenv("DM_VK_RING_HALF"); g_ring_half = e && e[0] == '1'; if (g_ring_n < 0) g_ring_n = 0;
                        if (g_ring_n) { g_ring = calloc((size_t)g_ring_n, sizeof *g_ring); VLOG("DM_VK_RING: the last %d frames are kept; touch ring_save to write them", g_ring_n); } }
    return g_ring_n > 0;
}
void vkx_ring_note(const char *fmt, ...) {
    if (!ring_on()) return;
    char l[256]; va_list ap; va_start(ap, fmt); int k = vsnprintf(l, sizeof l, fmt, ap); va_end(ap); if (k < 0) return; if (k >= (int)sizeof l) k = sizeof l - 1;
    if (g_ring_cn + (size_t)k + 2 > g_ring_cc) { g_ring_cc = (g_ring_cc ? g_ring_cc * 2 : 65536) + (size_t)k; g_ring_cur = realloc(g_ring_cur, g_ring_cc); }
    memcpy(g_ring_cur + g_ring_cn, l, (size_t)k); g_ring_cn += (size_t)k; g_ring_cur[g_ring_cn++] = '\n'; g_ring_cur[g_ring_cn] = 0;
}
// ---------- DM_VK_RING_OCC=1 (with DM_VK_RING): how many pixels each draw wrote ----------
// An occlusion query around every plain draw (vkd_draw), numbered in the draw's line of the frame's draw list ("occ <n>"); the counts are copied on the GPU into the
// frame's slot at its record and saved as f<frame>_occ.txt ("<n> <samples that passed>"). Fire Emblem Fortune's Weave: a mesh stretches across the screen for a few
// frames in the depth pre-pass only (black ribbons; the capture logs_mem3/001931) - the draw whose count jumps between a clean frame and a streaked one is that mesh
enum { OCC_MAX = 4096 };
static VkQueryPool g_occ_pool;
int vkx_occ_begin(VkCommandBuffer cb) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RING_OCC"); on = ring_on() && e && e[0] == '1'; if (on) VLOG("DM_VK_RING_OCC: pixel counts of up to %d draws a frame kept with the ring", OCC_MAX); }
    if (!on) return -1;
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (!r->op) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, OCC_MAX, 0};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &r->op)) { r->op = VK_NULL_HANDLE; on = 0; return -1; }
        void *m = NULL; r->ob = stage_buffer(NULL, (uint64_t)OCC_MAX * 8, &m); r->om = g_last_stage_mem; r->omap = m; if (!r->ob) { on = 0; return -1; }
    }
    if (r->ofill != g_ring_i + 1) { r->ofill = g_ring_i + 1; r->on = 0; r->osaved = 0; vkResetQueryPool(g_vk.dev, r->op, 0, OCC_MAX); }   // the slot's last use is a ring's length of frames old
    if (r->on >= OCC_MAX) return -1;
    int q = r->on++; g_occ_pool = r->op; vkCmdBeginQuery(cb, r->op, (uint32_t)q, VK_QUERY_CONTROL_PRECISE_BIT); return q;
}
void vkx_occ_end(VkCommandBuffer cb, int q) { if (q >= 0) vkCmdEndQuery(cb, g_occ_pool, (uint32_t)q); }
// ---------- DM_VK_RING_DLSS=1 (with DM_VK_RING): the upscaler's inputs, raw ----------
// At the frame's DLSS call, its colour, motion vectors and depth (4 bytes a texel each: B10G11R11 / R16G16 float / D32 float in Fire Emblem Fortune's Weave) are copied
// as they are into the frame's slot and saved as f<frame>_dlss.bin (colour, then motion, then depth) with f<frame>_dlss.txt (sizes, formats, jitter, motion scale).
// The stage snapshots blit to 8 bits and lose small motion; these keep the values FSR 2 reads, to tell trails on objects with no motion from a wrong sign or scale
void vkx_ring_dlss(sub_t *s, vk_tex *col, vk_tex *mot, vk_tex *dep, const float jit[2], const float mv[2]) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RING_DLSS"); on = ring_on() && e && e[0] == '1'; if (on) VLOG("DM_VK_RING_DLSS: the upscaler's colour, motion vectors and depth kept with the ring"); }
    if (!on || !s || !col || !mot || !dep) return;
    if (mot->w != col->w || mot->h != col->h || dep->w != col->w || dep->h != col->h) { VRED_ONCE("DM_VK_RING_DLSS: the inputs differ in size: not kept"); return; }
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    uint64_t one = (uint64_t)col->w * col->h * 4, need = one * 3 + (uint64_t)col->w * col->h;   // and the reactive mask, a byte a texel (vkx_ring_mask)
    if (r->lb && r->lcap < need) { vkDestroyBuffer(g_vk.dev, r->lb, NULL); vkFreeMemory(g_vk.dev, r->lm, NULL); r->lb = VK_NULL_HANDLE; }
    if (!r->lb) { void *m = NULL; r->lb = stage_buffer(NULL, need, &m); r->lm = g_last_stage_mem; r->lmap = m; r->lcap = need; if (!r->lb) return; }
    end_rendering(); barrier_all(s->cb);
    vk_tex *t[3] = {col, mot, dep};
    for (int k = 0; k < 3; k++) {
        VkImageAspectFlags a = (t[k]->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        VkBufferImageCopy c = {one * (uint64_t)k, 0, 0, {a, 0, 0, 1}, {0, 0, 0}, {col->w, col->h, 1}};
        vkCmdCopyImageToBuffer(s->cb, t[k]->img, VK_IMAGE_LAYOUT_GENERAL, r->lb, 1, &c);
    }
    barrier_all(s->cb);
    r->lfill = g_ring_i + 1; r->lw = col->w; r->lh = col->h;
    snprintf(r->lnote, sizeof r->lnote, "size %ux%u colour fmt %d motion fmt %d depth fmt %d jitter %.4f %.4f motion scale %.3f %.3f\n"
             "textures colour %llu motion %llu at %p (drawn by the GPU %d) depth %llu\n", col->w, col->h, col->fmt, mot->fmt, dep->fmt, jit[0], jit[1], mv[0], mv[1],
             (unsigned long long)col->serial, (unsigned long long)mot->serial, (void *)mot->mem, mot->gpu_owned, (unsigned long long)dep->serial);
}
// the reactive mask made for this frame's DLSS call (nvn_vk_draw.c reactive_run), after the three inputs in f<frame>_dlss.bin
void vkx_ring_mask(sub_t *s, VkImage img, uint32_t w, uint32_t h) {
    if (!ring_on() || !s || !img) return;
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (!r->lb || r->lfill != g_ring_i + 1 || r->lw != w || r->lh != h) return;   // this frame's inputs are kept (DM_VK_RING_DLSS)
    barrier_all(s->cb);
    VkBufferImageCopy c = {(uint64_t)w * h * 12, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(s->cb, img, VK_IMAGE_LAYOUT_GENERAL, r->lb, 1, &c);
    barrier_all(s->cb);
}
static uint32_t depth_bpp(VkFormat f) { return f == VK_FORMAT_D16_UNORM ? 2 : 4; }
// ---------- DM_VK_RING_CS=<key>[,<key>...] (with DM_VK_RING): the buffers of those compute shaders' dispatches ----------
// After each dispatch of a named shader, every uniform and storage buffer bound to the compute stage is copied (its first CSD_PART bytes) into the frame's slot, with an
// index line per buffer; saved as f<frame>_cs.bin and f<frame>_cs.txt. The copies run on the GPU right after the dispatch, so inputs and outputs are what that dispatch
// saw and wrote: the dispatch can then be run again on a PC with the same inputs (the work folder's re/csrun) and its outputs compared. For Fire Emblem Fortune's Weave's
// light culling (35d12d6f98fa05e2), whose shaders give a PC's results on the Adreno with generated inputs while battles tear once it writes its results
enum { CSD_MAX = 40 << 20, CSD_PART = 2 << 20, CSD_IDX = 65536 };
void vkx_ring_csdump(sub_t *s, uint64_t key, int x, int y, int z, const p_bind_buf *ubo, const p_bind_buf *ssbo) {
    static const char *want; static int init; if (!init) { init = 1; want = getenv("DM_VK_RING_CS"); if (want && ring_on()) VLOG("DM_VK_RING_CS: buffers of dispatches of %s kept with the ring, %d MB a frame at most", want, CSD_MAX >> 20); }
    vkx_ring_csdump2(s, want, key, x, y, z, ubo, ssbo);
}
void vkx_ring_csdump2(sub_t *s, const char *want, uint64_t key, int x, int y, int z, const p_bind_buf *ubo, const p_bind_buf *ssbo) {
    if (!want || !ring_on() || !s) return;
    char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)key); if (!strstr(want, k)) return;
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (r->xfill != g_ring_i + 1) { r->xfill = g_ring_i + 1; r->xn = 0; r->xd = 0; r->xin = 0; }
    if (!r->xb) { void *m = NULL; r->xb = stage_buffer(NULL, CSD_MAX, &m); r->xm = g_last_stage_mem; r->xmap = m; r->xidx = malloc(CSD_IDX); if (!r->xb || !r->xidx) { r->xb = VK_NULL_HANDLE; return; } }
    int d = r->xd++;
    barrier_all(s->cb);
    for (int kind = 0; kind < 2; kind++) for (int i = 0; i < 16; i++) {
        const p_bind_buf *b = kind ? &ssbo[i] : &ubo[i]; if (!b->addr || !b->size) continue;
        uint64_t len = b->size < CSD_PART ? b->size : CSD_PART, off = 0, avail = 0; VkBuffer src; int ok = vk_pool_find_part(b->addr, len, &src, &off, &avail);
        if (ok && avail < len) len = avail;
        uint64_t at = r->xn; int fits = ok && len && at + len <= CSD_MAX;
        if (fits) { VkBufferCopy bc = {off, at, len}; vkCmdCopyBuffer(s->cb, src, r->xb, 1, &bc); r->xn += (len + 15) & ~15ull; }
        if (r->xin + 160 < CSD_IDX) r->xin += (size_t)snprintf(r->xidx + r->xin, CSD_IDX - r->xin, "%d %s %dx%dx%d %c %d addr %#llx size %llu kept %llu at %lld\n", d, k, x, y, z, kind ? 'S' : 'U', i,
                                                               (unsigned long long)b->addr, (unsigned long long)b->size, (unsigned long long)(fits ? len : 0), fits ? (long long)at : -1ll);
    }
    barrier_all(s->cb);
}
// ---------- DM_VK_RING_STAGES=1 (with DM_VK_RING): the frame's steps, not only its result ----------
// Fire Emblem Fortune's Weave's battle floor breaks in single frames, with the same draws as the good frames around it; the scene capture shows that a frame is bad,
// not which pass made it so. With this, each frame's slot also keeps a 160x90 snapshot of every 3D-pass colour image (256 to 1919 wide: the game's dynamic resolution goes down to 960x540) at the end of each render
// pass that drew into it and after each compute dispatch that wrote it as a storage image, in order, labelled (what, texture, size, format, guest address) and marked
// in the draw list ("STAGE n"); saved beside the scene as f<frame>_s<nn>.png and f<frame>_stages.txt. Comparing a bad frame with the good one before it, the first
// snapshot that differs is the step that broke. Texture uploads are noted too ("UPLOAD"), with any GPU-written image over the same guest memory
static int g_stg = -1; static const uint32_t g_stg_w = 160, g_stg_h = 90;
static VkImage g_stg_img; static VkDeviceMemory g_stg_imem; static uint64_t g_stg_n, g_stg_skip;
static int g_stg_hdr; static VkImage g_stg_img16; static VkDeviceMemory g_stg_imem16;   // DM_VK_RING_STAGES=2: each colour snapshot also as 16-bit float (f<frame>_s<nn>.f16, 160x90 RGBA half floats), since the 8-bit
                                                                                         // one clips everything above 1 and a lit scene before tone mapping is all above 1 (the world map's dark triangle, logs_mem3/001253)

static int stages_on(void) {
    if (g_stg < 0) { const char *e = getenv("DM_VK_RING_STAGES"); g_stg = ring_on() && e && (e[0] == '1' || e[0] == '2'); g_stg_hdr = g_stg && e[0] == '2'; if (g_stg) VLOG("DM_VK_RING_STAGES: %d snapshots of %ux%u a frame at most, kept with the ring", STG_MAX, g_stg_w, g_stg_h); }
    return g_stg;
}
// one snapshot of x into the slot the current frame fills; outside a render pass (callers: end_rendering, after a dispatch)
void vkx_stage_snap(sub_t *s, vk_tex *x, const char *what) {
    if (!stages_on() || !s || !x || !x->img || x->type != VK_IMAGE_TYPE_2D || x->d.samples > 1 || x->w < 256 || x->w >= 1920 || !(x->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) return;   // the 3D passes (960x540 in battles at low dynamic resolution) and their half / quarter companions; not the UI and upscaled output (1920 and up)
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (r->sfill != g_ring_i + 1) { r->sfill = g_ring_i + 1; r->sn = 0; }
    if (r->sn >= STG_MAX) { g_stg_skip++; return; }
    r->hset[r->sn] = 0;
    if (!blit_ok(x->fmt, VK_FORMAT_R8G8B8A8_UNORM, 1)) { vkx_ring_note("STAGE - %s tex %llu fmt %d: no blit", what, (unsigned long long)x->serial, x->fmt); return; }
    if (!r->sb) { void *m = NULL; r->sb = stage_buffer(NULL, (uint64_t)STG_MAX * g_stg_w * g_stg_h * 4, &m); r->sm = g_last_stage_mem; r->smap = m; if (!r->sb) return; }
    VkCommandBuffer cb = s->cb;
    if (!g_stg_img) {
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {g_stg_w, g_stg_h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &g_stg_img)) { g_stg_img = VK_NULL_HANDLE; return; }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, g_stg_img, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_stg_imem) || vkBindImageMemory(g_vk.dev, g_stg_img, g_stg_imem, 0)) { vkDestroyImage(g_vk.dev, g_stg_img, NULL); g_stg_img = VK_NULL_HANDLE; return; }
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_stg_img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
    barrier_all(cb);
    VkImageBlit bl = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)x->w, (int32_t)x->h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)g_stg_w, (int32_t)g_stg_h, 1}}};
    vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_stg_img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    barrier_all(cb);
    VkBufferImageCopy c = {(VkDeviceSize)r->sn * g_stg_w * g_stg_h * 4, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {g_stg_w, g_stg_h, 1}};
    vkCmdCopyImageToBuffer(cb, g_stg_img, VK_IMAGE_LAYOUT_GENERAL, r->sb, 1, &c);
    barrier_all(cb);
    if (g_stg_hdr && blit_ok(x->fmt, VK_FORMAT_R16G16B16A16_SFLOAT, 1)) {
        if (!r->hb) { void *m = NULL; r->hb = stage_buffer(NULL, (uint64_t)STG_MAX * g_stg_w * g_stg_h * 8, &m); r->hmap = m; }
        if (r->hb && !g_stg_img16) {
            VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {g_stg_w, g_stg_h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
            VkMemoryRequirements mr; VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, 0, 0};
            if (vkCreateImage(g_vk.dev, &ii, NULL, &g_stg_img16)) g_stg_img16 = VK_NULL_HANDLE;
            else { vkGetImageMemoryRequirements(g_vk.dev, g_stg_img16, &mr); mai.allocationSize = mr.size; mai.memoryTypeIndex = device_local_type(mr.memoryTypeBits);
                   if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_stg_imem16) || vkBindImageMemory(g_vk.dev, g_stg_img16, g_stg_imem16, 0)) { vkDestroyImage(g_vk.dev, g_stg_img16, NULL); g_stg_img16 = VK_NULL_HANDLE; } }
            if (g_stg_img16) { VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                                          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_stg_img16, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                               vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); }
        }
        if (r->hb && g_stg_img16) {
            vkCmdBlitImage(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_stg_img16, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
            barrier_all(cb);
            VkBufferImageCopy c16 = {(VkDeviceSize)r->sn * g_stg_w * g_stg_h * 8, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {g_stg_w, g_stg_h, 1}};
            vkCmdCopyImageToBuffer(cb, g_stg_img16, VK_IMAGE_LAYOUT_GENERAL, r->hb, 1, &c16);
            barrier_all(cb); r->hset[r->sn] = 1;
        }
    }
    snprintf(r->lab[r->sn], sizeof r->lab[r->sn], "%s tex %llu %ux%u fmt %d at %#llx", what, (unsigned long long)x->serial, x->w, x->h, x->fmt, (unsigned long long)(uintptr_t)x->mem);
    vkx_ring_note("STAGE %d %s", r->sn, r->lab[r->sn]);
    r->sn++; g_stg_n++;
}
// A depth image's snapshot (32-bit float depth only): its depth copied to a buffer, from there into an R32_SFLOAT image of its size, which is blitted to the 160x90 RGBA8
// like a colour one (red = depth: 1 white, 0 black; reversed depth draws near things bright). Labelled "depth ..."
static VkBuffer g_dsb; static VkDeviceMemory g_dsbm; static uint64_t g_dsb_size;
static struct { VkImage img; VkDeviceMemory mem; uint32_t w, h; } g_dsi[4]; static int g_dsi_n;
void vkx_stage_snap_depth(sub_t *s, vk_tex *x, const char *what) {
    if (!stages_on() || !s || !x || !x->img || x->type != VK_IMAGE_TYPE_2D || x->d.samples > 1 || x->w < 256 || x->w >= 1920) return;
    if (x->fmt != VK_FORMAT_D32_SFLOAT && x->fmt != VK_FORMAT_D32_SFLOAT_S8_UINT) { vkx_ring_note("STAGE - depth %s tex %llu fmt %d: not 32-bit float", what, (unsigned long long)x->serial, x->fmt); return; }
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (r->sfill != g_ring_i + 1) { r->sfill = g_ring_i + 1; r->sn = 0; }
    if (r->sn >= STG_MAX) { g_stg_skip++; return; }
    r->hset[r->sn] = 0;
    if (!blit_ok(VK_FORMAT_R32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, 0)) { vkx_ring_note("STAGE - depth: no R32F blit"); return; }
    if (!r->sb) { void *m = NULL; r->sb = stage_buffer(NULL, (uint64_t)STG_MAX * g_stg_w * g_stg_h * 4, &m); r->sm = g_last_stage_mem; r->smap = m; if (!r->sb) return; }
    VkCommandBuffer cb = s->cb;
    uint64_t need = (uint64_t)x->w * x->h * 4;
    if (need > g_dsb_size && g_dsb) return;   // made once at 16 MB: any depth image under 1920 wide fits (1919 x 1919 x 4 does not, none is that tall)
    if (!g_dsb) {
        need = 16ull << 20;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if (vkCreateBuffer(g_vk.dev, &bi, NULL, &g_dsb)) { g_dsb = VK_NULL_HANDLE; return; }
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, g_dsb, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_dsbm) || vkBindBufferMemory(g_vk.dev, g_dsb, g_dsbm, 0)) { vkDestroyBuffer(g_vk.dev, g_dsb, NULL); g_dsb = VK_NULL_HANDLE; return; }
        g_dsb_size = need;
    }
    int k = -1; for (int i = 0; i < g_dsi_n; i++) if (g_dsi[i].w == x->w && g_dsi[i].h == x->h) k = i;
    if (k < 0) {
        if (g_dsi_n >= 4) return;
        k = g_dsi_n;
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, {x->w, x->h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &g_dsi[k].img)) return;
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, g_dsi[k].img, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_dsi[k].mem) || vkBindImageMemory(g_vk.dev, g_dsi[k].img, g_dsi[k].mem, 0)) { vkDestroyImage(g_vk.dev, g_dsi[k].img, NULL); return; }
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_dsi[k].img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        g_dsi[k].w = x->w; g_dsi[k].h = x->h; g_dsi_n++;
    }
    // the RGBA8 snapshot image is made by vkx_stage_snap; made here too when no colour snapshot came first
    if (!g_stg_img) {
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {g_stg_w, g_stg_h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &g_stg_img)) { g_stg_img = VK_NULL_HANDLE; return; }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, g_stg_img, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_stg_imem) || vkBindImageMemory(g_vk.dev, g_stg_img, g_stg_imem, 0)) { vkDestroyImage(g_vk.dev, g_stg_img, NULL); g_stg_img = VK_NULL_HANDLE; return; }
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_stg_img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    }
    barrier_all(cb);
    VkBufferImageCopy c1 = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {x->w, x->h, 1}};
    vkCmdCopyImageToBuffer(cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_dsb, 1, &c1);
    barrier_all(cb);
    VkBufferImageCopy c2 = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {x->w, x->h, 1}};
    vkCmdCopyBufferToImage(cb, g_dsb, g_dsi[k].img, VK_IMAGE_LAYOUT_GENERAL, 1, &c2);
    barrier_all(cb);
    VkImageBlit bl = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)x->w, (int32_t)x->h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)g_stg_w, (int32_t)g_stg_h, 1}}};
    vkCmdBlitImage(cb, g_dsi[k].img, VK_IMAGE_LAYOUT_GENERAL, g_stg_img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
    barrier_all(cb);
    VkBufferImageCopy c3 = {(VkDeviceSize)r->sn * g_stg_w * g_stg_h * 4, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {g_stg_w, g_stg_h, 1}};
    vkCmdCopyImageToBuffer(cb, g_stg_img, VK_IMAGE_LAYOUT_GENERAL, r->sb, 1, &c3);
    barrier_all(cb);
    snprintf(r->lab[r->sn], sizeof r->lab[r->sn], "depth %s tex %llu %ux%u fmt %d at %#llx", what, (unsigned long long)x->serial, x->w, x->h, x->fmt, (unsigned long long)(uintptr_t)x->mem);
    vkx_ring_note("STAGE %d %s", r->sn, r->lab[r->sn]);
    r->sn++; g_stg_n++;
}
void vkx_ring_record(sub_t *s, vk_tex *col, vk_tex *dep) {
    if (!ring_on() || !col || col->type != VK_IMAGE_TYPE_2D || col->d.samples > 1) return;
    ring_slot *r = &g_ring[g_ring_i % g_ring_n];
    if (r->sfill != g_ring_i + 1) { r->sfill = g_ring_i + 1; r->sn = 0; }   // no snapshots this frame: none from the slot's older round
    if (r->xfill != g_ring_i + 1) r->xin = 0;                               // no dispatch dumped this frame (DM_VK_RING_CS): none saved for it
    g_ring_i++;
    uint32_t w = g_ring_half ? (col->w + 1) / 2 : col->w, h = g_ring_half ? (col->h + 1) / 2 : col->h, db = dep && !g_ring_half ? depth_bpp(dep->fmt) : 0;
    if (r->cb && (r->w != w || r->h != h || r->dbpp != db)) { vkDestroyBuffer(g_vk.dev, r->cb, NULL); vkFreeMemory(g_vk.dev, r->cm, NULL); if (r->db) { vkDestroyBuffer(g_vk.dev, r->db, NULL); vkFreeMemory(g_vk.dev, r->dm, NULL); } r->cb = r->db = VK_NULL_HANDLE; }
    if (!r->cb) { void *m = NULL; r->cb = stage_buffer(NULL, (uint64_t)w * h * 4, &m); r->cm = g_last_stage_mem; r->cmap = m; r->w = w; r->h = h; r->dbpp = db;
                  if (db) { m = NULL; r->db = stage_buffer(NULL, (uint64_t)w * h * db, &m); r->dm = g_last_stage_mem; r->dmap = m; } }
    if (!r->cb) return;
    if (!g_ring_img || g_ring_iw != w || g_ring_ih != h) {   // the RGBA8 image the scene is blitted into (a blit converts the format)
        if (g_ring_img) { vkDestroyImage(g_vk.dev, g_ring_img, NULL); vkFreeMemory(g_vk.dev, g_ring_imem, NULL); g_ring_img = VK_NULL_HANDLE; }
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &g_ring_img)) { g_ring_img = VK_NULL_HANDLE; return; }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, g_ring_img, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_ring_imem) || vkBindImageMemory(g_vk.dev, g_ring_img, g_ring_imem, 0)) { vkDestroyImage(g_vk.dev, g_ring_img, NULL); g_ring_img = VK_NULL_HANDLE; return; }
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_ring_img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
        g_ring_iw = w; g_ring_ih = h;
    }
    end_rendering(); barrier_all(s->cb);
    VkImageBlit bl = {{col->aspect, 0, 0, 1}, {{0, 0, 0}, {(int32_t)col->w, (int32_t)col->h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(s->cb, col->img, VK_IMAGE_LAYOUT_GENERAL, g_ring_img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, g_ring_half ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    barrier_all(s->cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(s->cb, g_ring_img, VK_IMAGE_LAYOUT_GENERAL, r->cb, 1, &c);
    if (dep && r->db && !g_ring_half && dep->w == w && dep->h == h && (dep->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkBufferImageCopy dc = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(s->cb, dep->img, VK_IMAGE_LAYOUT_GENERAL, r->db, 1, &dc);
    }
    if (r->op && r->ob && r->ofill == g_ring_i && r->on > 0) { vkCmdCopyQueryPoolResults(s->cb, r->op, 0, (uint32_t)r->on, r->ob, 0, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT); r->osaved = r->on; } else r->osaved = 0;
    barrier_all(s->cb);
    r->frame = g_presents; r->valid = 1;
    free(r->notes); r->notes = g_ring_cur; r->nn = g_ring_cn; g_ring_cur = NULL; g_ring_cn = g_ring_cc = 0;   // the draws since the last record: this frame's scene
}
static void ring_save_check(uint64_t n) {
    if (!ring_on()) return;
    const char *root = getenv("DM_ROOT") ? getenv("DM_ROOT") : "."; char trig[600]; snprintf(trig, sizeof trig, "%s/ring_save", root);
    if (access(trig, F_OK)) return;
    unlink(trig);
    mtl_finish();   // the slots' copies have run
    char dir[700]; snprintf(dir, sizeof dir, "%s/ring", root); mkdir(dir, 0755); snprintf(dir, sizeof dir, "%s/ring/%06llu", root, (unsigned long long)n); mkdir(dir, 0755);
    int saved = 0;
    for (int k = 0; k < g_ring_n; k++) {
        ring_slot *r = &g_ring[(g_ring_i + k) % g_ring_n]; if (!r->valid || !r->cmap) continue;
        char p[800]; uint32_t np = r->w * r->h;
        uint8_t *rgba = malloc((size_t)np * 4); if (!rgba) break;
        memcpy(rgba, r->cmap, (size_t)np * 4); for (uint32_t i = 0; i < np; i++) rgba[i * 4 + 3] = 255;
        snprintf(p, sizeof p, "%s/f%06llu_scene.png", dir, (unsigned long long)r->frame); png_write(p, rgba, r->w, r->h);
        if (r->dmap && r->dbpp) {   // depth, its range stretched to 0-255
            float lo = 1e30f, hi = -1e30f; float *dv = malloc((size_t)np * sizeof *dv);
            for (uint32_t i = 0; i < np; i++) { float v; if (r->dbpp == 2) { uint16_t u; memcpy(&u, r->dmap + i * 2, 2); v = u / 65535.f; } else memcpy(&v, r->dmap + i * 4, 4); dv[i] = v; if (v < lo) lo = v; if (v > hi) hi = v; }
            for (uint32_t i = 0; i < np; i++) { uint8_t g = hi > lo ? (uint8_t)((dv[i] - lo) / (hi - lo) * 255) : 0; rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = g; }
            free(dv); snprintf(p, sizeof p, "%s/f%06llu_depth.png", dir, (unsigned long long)r->frame); png_write(p, rgba, r->w, r->h);
        }
        free(rgba);
        if (r->notes) { snprintf(p, sizeof p, "%s/f%06llu_draws.txt", dir, (unsigned long long)r->frame); FILE *f = fopen(p, "w"); if (f) { fwrite(r->notes, 1, r->nn, f); fclose(f); } }
        if (r->lmap && r->lfill && r->lfill <= g_ring_i && r->lfill > g_ring_i - g_ring_n) {   // the upscaler's inputs (DM_VK_RING_DLSS)
            snprintf(p, sizeof p, "%s/f%06llu_dlss.txt", dir, (unsigned long long)r->frame); FILE *f = fopen(p, "w"); if (f) { fputs(r->lnote, f); fclose(f); }
            snprintf(p, sizeof p, "%s/f%06llu_dlss.bin", dir, (unsigned long long)r->frame); f = fopen(p, "wb"); if (f) { fwrite(r->lmap, 1, (size_t)r->lw * r->lh * 13, f); fclose(f); }
        }
        if (r->omap && r->osaved > 0) {   // the per-draw pixel counts (DM_VK_RING_OCC)
            snprintf(p, sizeof p, "%s/f%06llu_occ.txt", dir, (unsigned long long)r->frame); FILE *f = fopen(p, "w");
            if (f) { for (int q = 0; q < r->osaved; q++) { uint64_t v; memcpy(&v, r->omap + (size_t)q * 8, 8); fprintf(f, "%d %llu\n", q, (unsigned long long)v); } fclose(f); }
        }
        if (r->xmap && r->xidx && r->xin && r->xfill) {   // the compute buffer dumps (DM_VK_RING_CS)
            snprintf(p, sizeof p, "%s/f%06llu_cs.txt", dir, (unsigned long long)r->frame); FILE *f = fopen(p, "w"); if (f) { fwrite(r->xidx, 1, r->xin, f); fclose(f); }
            snprintf(p, sizeof p, "%s/f%06llu_cs.bin", dir, (unsigned long long)r->frame); f = fopen(p, "wb"); if (f) { fwrite(r->xmap, 1, r->xn, f); fclose(f); }
        }
        if (r->smap && r->sn > 0) {   // the stage snapshots (DM_VK_RING_STAGES)
            snprintf(p, sizeof p, "%s/f%06llu_stages.txt", dir, (unsigned long long)r->frame); FILE *f = fopen(p, "w");
            uint8_t *px = malloc((size_t)g_stg_w * g_stg_h * 4);
            for (int k2 = 0; px && k2 < r->sn; k2++) {
                memcpy(px, r->smap + (size_t)k2 * g_stg_w * g_stg_h * 4, (size_t)g_stg_w * g_stg_h * 4); for (uint32_t i = 0; i < g_stg_w * g_stg_h; i++) px[i * 4 + 3] = 255;
                char q[800]; snprintf(q, sizeof q, "%s/f%06llu_s%02d.png", dir, (unsigned long long)r->frame, k2); png_write(q, px, g_stg_w, g_stg_h);
                if (r->hmap && r->hset[k2]) { snprintf(q, sizeof q, "%s/f%06llu_s%02d.f16", dir, (unsigned long long)r->frame, k2); FILE *hf = fopen(q, "wb");
                                              if (hf) { fwrite(r->hmap + (size_t)k2 * g_stg_w * g_stg_h * 8, 1, (size_t)g_stg_w * g_stg_h * 8, hf); fclose(hf); } }
                if (f) fprintf(f, "%02d %s\n", k2, r->lab[k2]);
            }
            free(px); if (f) fclose(f);
        }
        saved++;
    }
    VLOG("DM_VK_RING: %d frames written to %s; stage snapshots %llu taken, %llu past the %d a frame", saved, dir, (unsigned long long)g_stg_n, (unsigned long long)g_stg_skip, STG_MAX);
}
static uint8_t u8f(float v) { return (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + 0.5f); }
// diagnostic: the texels of a tiny float texture (16 texels at most: the auto exposure's 3x1 R16F image, Fortune's Weave's tonemap reading 2^-(texel 0 + 3), dm/games/04007EF01F8AE000/README.md)
// in the log, read back after the GPU is idle. Run at a traced frame's end for each such texture a draw sampled (vkd_trace_small)
void vkx_log_floats(vk_tex *x, const char *why) {
    VkFormat f = x->fmt; uint32_t ch = f == VK_FORMAT_R16_SFLOAT || f == VK_FORMAT_R32_SFLOAT ? 1 : f == VK_FORMAT_R16G16_SFLOAT || f == VK_FORMAT_R32G32_SFLOAT ? 2 :
                                 f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R32G32B32A32_SFLOAT ? 4 : 0;
    int half = f == VK_FORMAT_R16_SFLOAT || f == VK_FORMAT_R16G16_SFLOAT || f == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!ch || x->type != VK_IMAGE_TYPE_2D || x->d.samples > 1 || (uint64_t)x->w * x->h > 16) return;
    uint64_t n = (uint64_t)x->w * x->h, size = n * ch * (half ? 2 : 4);
    void *map = NULL; VkBuffer b = stage_buffer(NULL, size, &map); if (!b) return; VkDeviceMemory bm = g_last_stage_mem;
    static VkCommandPool cp; if (!cp) { VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, g_vk.qfam}; vkCreateCommandPool(g_vk.dev, &cpi, NULL, &cp); }
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cb;
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
    char line[512]; int o = snprintf(line, sizeof line, "texels of texture %llu (%ux%u format %d, %s):", (unsigned long long)x->serial, x->w, x->h, f, why);
    for (uint64_t i = 0; i < n * ch && o < (int)sizeof line - 24; i++) {
        float v; if (half) v = half_to_f(((const uint16_t *)map)[i]); else memcpy(&v, (const uint8_t *)map + i * 4, 4);
        o += snprintf(line + o, sizeof line - (size_t)o, "%s%.4g", i % ch ? "," : " ", v);
    }
    VLOG("%s", line);
    vkDestroyBuffer(g_vk.dev, b, NULL); vkFreeMemory(g_vk.dev, bm, NULL); vkFreeCommandBuffers(g_vk.dev, cp, 1, &cb);
}
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
// A 1x1 image that reads (0,0,0,0) through its view (every component swizzled to ZERO): what a descriptor slot gets when the game's
// binding holds no texture Vulkan can put there — a 3D texture bound in a 1D array slot, or an array entry the game left empty. Such a
// slot must not be left null: the driver then reads whatever its register holds, and the shader's sample of it comes out as one flat
// colour (the flat, hard-edged block an effect card shows). One tiny image per view kind, made on first use.
static VkImageView g_zero_view[8];
int vkx_zero_blocks_now(const vk_tex *x, uint32_t *zero, uint32_t *blocks, uint32_t *l0_zero, uint32_t *l0_blocks) {
    *zero = *blocks = *l0_zero = *l0_blocks = 0;
    if (!x || !x->mem || x->linear || x->bpb != 16 || x->bw <= 1 || x->d.samples > 1) return 0;
    for (unsigned l = 0; l < x->levels; l++) for (unsigned ly = 0; ly < x->layers; ly++) {
        uint32_t w = level_dim(x->w, l), h = level_dim(x->h, l), dz = level_dim(x->depth, l);
        uint32_t bw = (w + x->bw - 1) / x->bw, bh = (h + x->bh - 1) / x->bh; size_t pitch = (size_t)bw * 16, n = pitch * bh * dz;
        uint8_t *buf = malloc(n ? n : 16); if (!buf) return 0;
        if (!nvn_deswizzle(&x->d, l, ly, x->mem, buf, pitch))
            for (size_t i = 0; i < n / 16; i++) { const uint64_t *q = (const uint64_t *)(buf + i * 16); int z = !(q[0] | q[1]); *zero += z; (*blocks)++; if (!l) { *l0_zero += z; (*l0_blocks)++; } }
        free(buf);
    }
    return 1;
}
VkImageView vkx_zero_view(VkImageViewType vt) {
    int i = (int)vt; if (i < 0 || i > 7) i = 7;
    if (g_zero_view[i]) return g_zero_view[i];
    static int failed[8]; if (failed[i]) return VK_NULL_HANDLE;
    int one_d = vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    uint32_t layers = cube ? 6 : 1;
    static VkImage img[8]; static VkDeviceMemory mem[8];
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                            one_d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {1, 1, 1}, 1, layers, VK_SAMPLE_COUNT_1_BIT,
                            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(g_vk.dev, &ii, NULL, &img[i])) { failed[i] = 1; return VK_NULL_HANDLE; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, img[i], &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, device_local_type(mr.memoryTypeBits)};
    if (vkAllocateMemory(g_vk.dev, &mai, NULL, &mem[i]) || vkBindImageMemory(g_vk.dev, img[i], mem[i], 0)) {
        failed[i] = 1; VLOG("[RED] zero view: image memory"); return VK_NULL_HANDLE;
    }
    {   // UNDEFINED -> GENERAL once: every descriptor that uses it says GENERAL
        VkCommandPool cp; VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, g_vk.qfam};
        if (!vkCreateCommandPool(g_vk.dev, &cpi, NULL, &cp)) {
            VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cb;
            if (!vkAllocateCommandBuffers(g_vk.dev, &ai, &cb)) {
                VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
                VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                           VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
                vkEndCommandBuffer(cb);
                VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; vkCreateFence(g_vk.dev, &fi, NULL, &fe);
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
                pthread_mutex_lock(&g_q_mu); vkQueueSubmit(g_vk.q, 1, &si, fe); pthread_mutex_unlock(&g_q_mu);
                vkWaitForFences(g_vk.dev, 1, &fe, VK_TRUE, 5000000000ull); vkDestroyFence(g_vk.dev, fe, NULL);
                vkFreeCommandBuffers(g_vk.dev, cp, 1, &cb);
            }
            vkDestroyCommandPool(g_vk.dev, cp, NULL);
        }
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img[i], vt, VK_FORMAT_R8G8B8A8_UNORM,
                                {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO},
                                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    if (vkCreateImageView(g_vk.dev, &vi, NULL, &g_zero_view[i])) { failed[i] = 1; return VK_NULL_HANDLE; }
    VLOG("[INFO] zero view made for texture bindings whose slot holds no expressible texture (view type %d)", (int)vt);
    return g_zero_view[i];
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    (void)period_s; (void)crop;
    uint64_t n = ++g_presents;
    {   // DM_ROOT/trace_now: drop this file to trace the next frame and dump its targets, no frame number to guess (vkd_trace_arm)
        static char tpath[600]; static int tinit;
        if (!tinit) { tinit = 1; snprintf(tpath, sizeof tpath, "%s/trace_now", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); }
        if (access(tpath, F_OK) == 0) { remove(tpath); vkd_trace_arm(); g_shot_force = n + 1; }
        static char spath[600]; if (!spath[0]) snprintf(spath, sizeof spath, "%s/shot_now", getenv("DM_ROOT") ? getenv("DM_ROOT") : ".");   // DM_ROOT/shot_now: the next frame as a picture, no trace
        if (access(spath, F_OK) == 0) { remove(spath); g_shot_force = n + 1; }
        vkd_bisect_tick(n);
    }
    if (exec_init()) { uint64_t t0 = now_ns_mono(); vk_tex *fx = tex_of(t, NULL); if (fx) vkd_front_present(fx, crop); g_t_front += now_ns_mono() - t0; }
    if (t && vkx_ring_final()) { g_final_t = *t; g_final_pend = 1; }
    { static uint64_t at; uint64_t t = now_ns_mono();
      if (!at) at = t;
      else if (n % 300 == 0) { double f = 300 * 1e6, wall = (t - at) / f;
          VLOG("frame %llu: per frame over the last 300: %.1f ms apart; CPU recording %.1f ms, waiting for the GPU to free a command buffer %.1f ms, screen hand-off %.1f ms; GPU busy about %.1f ms",
               (unsigned long long)n, wall, g_t_rec / f, g_t_slot / f, g_t_front / f, g_t_gpu / f);
          g_t_rec = g_t_slot = g_t_front = g_t_gpu = 0; at = t; } }   // the app's screen (nvn_vk_front.c), when the app started this host
    if (exec_init() && shot_wanted(n)) {
        mtl_finish();   // the submissions before the present have run
        vk_tex *x = tex_of(t, NULL);
        { vk_tex *sm[32]; int ns = vkd_trace_small(sm, 32); for (int k = 0; k < ns; k++) vkx_log_floats(sm[k], "sampled in the traced frame"); }
        { vk_tex *tg[96]; int nt = vkd_trace_targets(tg, 96); for (int k = 0; k < nt; k++) { char tag[64]; snprintf(tag, sizeof tag, "frame_%06llu_t%02d_s%llu", (unsigned long long)n, k, (unsigned long long)tg[k]->serial); vkx_dump(tg[k], tag); } }
        if (x) shot(x, n, crop); else VLOG("frame %llu: the presented texture has no host image (nothing rendered or copied into it yet)", (unsigned long long)n);
    }
    vkd_pcache_tick(n);
    char dline[256];
    if (n % 150 == 0 && g_exec_ok == 1) tex_evict();
    if (g_exec_ok == 1) ring_save_check(n);
    if (g_exec_ok == 1) prof_frame(n);
    if (n % 600 == 0 && g_exec_ok == 1) mem_line(n);
    if (n % 600 == 0 && g_exec_ok == 1) { uint64_t sh, dr; vkd_front_stats(&sh, &dr); if (sh || dr) VLOG("frame %llu: %llu frames shown on the app's screen, %llu dropped (the app was behind)", (unsigned long long)n, (unsigned long long)sh, (unsigned long long)dr); }
    if (n % 600 == 0 && g_exec_ok == 1) VLOG("frame %llu: clears applied as load operations %llu, passes that only clear %llu (DM_VK_LOAD_CLEAR); depth clears that waited for their image's pass %llu: %llu its load operation, %llu landed alone (DM_VK_WAIT_CLEAR); ClearTexture of depth images %llu as load operations, %llu as transfer clears; early-depth kicks %llu (%llu passes without one)", (unsigned long long)n, (unsigned long long)g_n_load_clear, (unsigned long long)g_n_clear_only_pass, (unsigned long long)g_n_wait, (unsigned long long)g_n_wait_used, (unsigned long long)g_n_wait_landed, (unsigned long long)g_n_ct_depth, (unsigned long long)g_n_ct_depth_xfer, (unsigned long long)g_n_lrz_kick, (unsigned long long)g_n_lrz_skip);
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
