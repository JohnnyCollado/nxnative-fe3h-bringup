// The graphics backend's null form for the Android host, used when no Vulkan device is available (nvn_vk_exec.c falls back to nvn_null_exec), and the rest of what nvn_mtl.m (Metal) provides to hle_nvn.c, the NVN layer.
// hle_nvn.c keeps every NVN object and the game's commands as on the Mac; this backend executes nothing and answers as a GPU that finishes at once:
//   - fences: each commit hands out the next fence number; every wait is already satisfied (mtl_wait returns 1, as nvn_mtl.m does once a fence completed)
//   - present: a ticket that is never waited on; no display link (mtl_link_active 0), so hle_nvn.c paces frames with its own sleep at the game's interval,
//     as the Mac host does without a window; mtl_fps_target answers 60 (callers divide by it)
//   - GPU address lookups find nothing (mtl_addr 0); the statistics and diagnostics are zero
// The boot test needs only this: the game's code runs and its frame loop turns, with nothing drawn. The Vulkan backend replaces this file.
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_obj.h"         // the command records hle_nvn.c writes (op_hdr, OP_*, p_clear_buf, sync_t)
#include "nvn_timestamp.h"   // GPU counter ticks from nanoseconds
#include "nvn_vk.h"

static _Atomic uint64_t g_fence;
uint64_t guest_ns(void);   // hle_os.c: the guest clock, which counter timestamps use (as nvn_mtl.m gpu_ticks)

// ---- submission and fences ----
void mtl_prepare(void) {}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
// A command list's effects the CPU can see, applied at once as a GPU that finishes instantly would leave them; everything that draws or touches textures is skipped.
// These are what nvn_mtl.m mtl_exec does with OP_SIGNAL_EVENT, OP_REPORT_COUNTER, OP_FENCE_SYNC, OP_CLEAR_BUF, OP_COPY_B2B and OP_UPDATE_UB (its post_run applies the
// event and counter writes when the GPU completes). Buffer addresses are host pointers into the memory pools (hle_nvn.c nvnBufferGetAddress). Without them a game that
// waits for a GPU-written value waits forever: Fortune's Weave's movie player never released a picture buffer, so its decoder filled up after four pictures and the
// opening movie stood still
static void apply_report(uint64_t addr, uint64_t type) {
    uint64_t g = guest_ns(), t = nvn_timestamp_from_ns(g), rec[2] = {type == 0 || type == 0x10 ? t : 0, t};   // a timestamp report carries the time in both words (nvn_mtl.m post_run)
    if (addr) memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
}
// DM_NVN_OPSTAT: every 300 frames, the mean count of each command type per frame (what the Vulkan backend must execute first)
static _Atomic uint64_t g_opn[OP_COUNT_ + 1]; static int g_opstat = -1;
void nvn_null_opstat_frame(uint64_t frame) {
    if (g_opstat <= 0 || frame % 300) return;
    char line[2048]; int k = 0;
    for (int i = 0; i <= OP_COUNT_ && k < (int)sizeof line - 32; i++) { uint64_t n = atomic_exchange(&g_opn[i], 0); if (n) k += snprintf(line + k, sizeof line - k, " %d:%.1f", i, n / 300.0); }
    flockfile(stderr); fprintf(stderr, "[dm] [nvn-null] commands per frame by type up to frame %llu:%s\n", (unsigned long long)frame, line); funlockfile(stderr);
}
uint64_t nvn_null_commit(void) { return atomic_fetch_add(&g_fence, 1) + 1; }   // the fence numbers without a Vulkan device (nvn_vk_exec.c falls back here)
uint64_t nvn_null_last_fence(void) { return atomic_load(&g_fence); }
void nvn_null_exec(const uint8_t *p) {
    if (g_opstat < 0) g_opstat = getenv("DM_NVN_OPSTAT") != NULL;
    for (int guard = 0; p && guard < (1 << 24); guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (g_opstat > 0) atomic_fetch_add(&g_opn[h.op <= OP_COUNT_ ? h.op : OP_COUNT_], 1);
        const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); nvn_null_exec((const uint8_t *)(uintptr_t)t); break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4);
                if (val) { if (m == NVN_EVENT_SIGNAL_MODE_ADD) *val += v; else *val = v; } break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); apply_report(ad, t); break; }
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); if (s) { s->fence = atomic_load(&g_fence) + 1; s->fenced = 1; } break; }
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; uint32_t *d = (uint32_t *)(uintptr_t)cb->addr;
                if (d) { for (uint64_t i = 0; i < cb->size / 4; i++) d[i] = cb->value; for (uint64_t i = cb->size & ~3ull; i < cb->size; i++) ((uint8_t *)d)[i] = (uint8_t)(cb->value >> (8 * (i & 3))); } break; }
            case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8);
                if (s && d && n) memmove((void *)(uintptr_t)d, (const void *)(uintptr_t)s, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); if (d && n) memcpy((void *)(uintptr_t)d, a + 16, n); break; }
            default: break;
        }
        if (h.size < 8) return;   // a malformed record: stop rather than loop
        p += h.size;
    }
}
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }   // only a look-ahead hint for nvn_mtl.m (hle_nvn.c runs each list with its own mtl_exec)
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_set_frame(uint64_t f) {
    nvn_null_opstat_frame(f);
    if (f % 600 == 0 && f) { int n, bad; uint64_t b; vk_pool_stats(&n, &b, &bad); flockfile(stderr); fprintf(stderr, "[dm] [vk] frame %llu: %d pools on the GPU, %.1f MB, %d left in ordinary memory\n", (unsigned long long)f, n, b / 1048576.0, bad); funlockfile(stderr); }
}

// ---- presenting and pacing ----
uint32_t mtl_present_qos(void) { return 0; }
double mtl_presented_time(uint64_t n) { (void)n; return 0; }
int mtl_has_window(void) { return 0; }
int mtl_display_hz(void) { return 60; }
int mtl_fps_target(void) { const char *e = getenv("DM_FPS_TARGET"); int v = e ? atoi(e) : 0; return v == 30 || v == 60 ? v : 60; }
int mtl_link_active(void) { return 0; }
int mtl_link_rate(void) { return 60; }
int mtl_link_stalled(void) { return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) {
    (void)n; if (target) *target = 0; if (lead_ms) *lead_ms = 0; if (ticks) *ticks = 0; if (gap_ms) *gap_ms = 0; return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_snap_on(void) { return 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }

// ---- resources ----
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) { (void)addr; (void)len; if (buf) *buf = NULL; if (off) *off = 0; return 0; }
void mtl_pool_add(const void *mem, uint64_t size) { vk_pool_add(mem, size); }   // the pool becomes GPU memory at its own address (nvn_vk_mem.c)
void mtl_pool_del(const void *mem) { vk_pool_del(mem); }
void mtl_alias_drop_pool(const void *pool) { (void)pool; }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    (void)t; (void)level; (void)layer0; (void)r; (void)dst; (void)row; (void)img; }   // nothing was drawn: the destination keeps what it holds
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
void mtl_prog_release(prog_t *p) { (void)p; }
int mtl_pso_prebuild_tool(void) { return 0; }   // DM_PSO_PREBUILD=tool is a Metal pipeline-archive tool: nothing to do here
void mtl_pso_line(char *out, size_t n) { if (n) out[0] = 0; }

// ---- statistics: zero ----
#define Z(p) do { if (p) *(p) = 0; } while (0)
void mtl_stats(uint64_t *passes, uint64_t *draws) { Z(passes); Z(draws); }
void mtl_alloc_stats(uint64_t *a, uint64_t *b, int *c) { Z(a); Z(b); Z(c); }
void mtl_args_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_barrier_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_bw(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_create_ns(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_create_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_cshoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_geohoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_geomesh_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_hybrid_stats(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_lc_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_pkg_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_sm_stats(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_ubd_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_ubp_stats(uint64_t *a, uint64_t *b, uint64_t *c) { Z(a); Z(b); Z(c); }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
void mtl_use_stats(uint64_t *a, uint64_t *b) { Z(a); Z(b); }
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { Z(a); Z(b); Z(c); Z(d); Z(e); Z(f); }
void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long mb) { (void)frame; (void)mb; }
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; Z(t0); Z(t1); return 0; }
unsigned mtl_cpu_tex_count(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_blit_splits(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }

// ---- GPU diagnostics (texture and pass dumps, frame capture): unavailable without a GPU backend ----
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; }
