// The Android Vulkan backend's internals shared by its executor (nvn_vk_exec.c: submissions, textures, render targets, copies) and its draw path (nvn_vk_draw.c:
// shaders, pipelines, bindings, draws). Not an interface for the rest of the host.
#pragma once
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)
#include "nvn_obj.h"
#include "nvn_vk.h"

#define VLOG(...) do { flockfile(stderr); fprintf(stderr, "[dm] [vk] " __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } while (0)
#define VRED_ONCE(...) do { static int once_; if (!once_++) VLOG("[RED] " __VA_ARGS__); } while (0)
#define GROW(arr, n, cap) do { if ((n) == (cap)) { (cap) = (cap) ? (cap) * 2 : 16; (arr) = realloc((arr), (size_t)(cap) * sizeof *(arr)); } } while (0)

typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr, type; } post_op;   // kind 1 event, 2 counter report
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *map; uint64_t size, used; } vkx_arena;   // host-visible memory a submission hands out per draw
typedef struct {
    VkCommandBuffer cb; uint64_t value; int open;
    post_op *posts; int nposts, cposts;
    VkImageView *views; int nviews, cviews;               // freed when the submission completes
    struct { VkBuffer b; VkDeviceMemory m; } *stage; int nstage, cstage;
    VkDescriptorPool *dpools; int ndp, cdp, dp_cur;      // reset when the submission completes (nvn_vk_draw.c)
    vkx_arena *arenas; int nar, car, ar_cur;             // likewise
    uint64_t submit_ns;                                   // when it went to the queue (the frame timing report: GPU busy estimate)
    char log[8192]; int nlog; uint32_t ncrumb;                            // what it holds (draw / dispatch / DLSS), printed if the GPU never finishes it
} sub_t;

typedef struct vk_tex {
    struct vk_tex *next; uint64_t serial; nvn_texdesc d; nvn_texlayout L; uint8_t *mem;
    VkImage img; VkDeviceMemory dm; VkFormat fmt; VkImageAspectFlags aspect; VkImageType type; VkImageViewType vtype;
    uint32_t layers, levels, w, h, depth; int bw, bh, bpb, linear, failed;
    int gpu_owned, uploaded, dirty;
    VkImageUsageFlags usage; VkFormat storage_fmt;       // its usages; the format a storage view takes (an sRGB image's UNORM twin), 0 when it cannot be one
    uint64_t lin_sub;                                     // a linear texture: the submission that last copied it from its pool (re-copied once per submission)
    uint64_t nclear, ncopy, ndraw; float last_clear[4];   // what the GPU did to it (frame-shot diagnostics)
    void *sviews;                                         // the draw path's cached sampling views (nvn_vk_draw.c), destroyed with the texture
    uint64_t dbytes;                                      // its image's device memory (the memory line's host-image total)
    uint64_t last_ns;                                     // when a recording last looked it up (tex_of): idle images of CPU content are evicted (tex_evict)
    uint8_t wd, ws; float wdv; uint32_t wsv, wl, wy;      // a depth / stencil clear of the whole image (level wl, layer wy) waiting for the image's next pass (nvn_vk_exec.c wait_*)
    uint32_t up_zero, up_blocks, up_l0_zero, up_l0_blocks; uint64_t up_sub, up_n;   // 16-byte compressed formats (ASTC, BC6/7): all-zero blocks in what the last upload
                                                          // copied (all levels, and level 0), the submission it ran in, uploads so far (trace diagnostics)
} vk_tex;
// all-zero 16-byte blocks the texture's guest storage holds now, deswizzled (all levels; level 0 apart); 0 when its format has no 16-byte blocks or it has no storage
int vkx_zero_blocks_now(const vk_tex *x, uint32_t *zero, uint32_t *blocks, uint32_t *l0_zero, uint32_t *l0_blocks);

struct vkx_rt { int has_rt, active; p_set_rt rt; VkExtent2D ext, want;     // want: the render area of a pass with no attachments (the draw's viewport)
                // clears waiting for the pass of these targets to open (nvn_vk_exec.c pend_*): applied as its load operations
                int npend, in_begin, skip_ds; uint8_t pc_on[8], pd_on, ps_on; VkClearColorValue pc[8]; float pd; uint32_t ps; };
extern struct vkx_rt g_rt;            // the bound render targets and the open render pass
extern sub_t *g_cur;                  // the submission being recorded
extern uint64_t g_fence;              // the last submitted submission's number
extern _Atomic uint64_t g_applied;    // the last one the GPU finished (and whose effects are applied)
extern uint64_t g_stats_draws;
extern uint64_t g_presents;           // frames presented so far

uint64_t vkx_now_ns(void);                              // monotonic nanoseconds (the slow-submit breakdown's clock)
void vkx_sb_note(int what, uint64_t ns);                // per-submission driver-call time: 1 texture create, 2 descriptor writes, 3 pipeline create
uint8_t *vkd_stage_alloc(sub_t *s, uint64_t n, VkBuffer *buf, uint64_t *off);   // host-visible staging from the submission's arenas (freed when it completes)
void barrier_all(VkCommandBuffer cb);
void vkx_ring_record(sub_t *s, vk_tex *col, vk_tex *dep);   // DM_VK_RING: this frame's scene and depth into the rolling capture
void vkx_ring_note(const char *fmt, ...);                     // DM_VK_RING: a line of this frame's draw list
void barrier_compute(VkCommandBuffer cb);             // after a dispatch: compute writes fenced, graphics pipeline left running
void end_rendering(void);
int begin_rendering(sub_t *s);
vk_tex *tex_of(const tex_t *t, VkCommandBuffer cb);
vk_tex *tex_for_read(const tex_t *t, sub_t *s);
vk_tex *tex_for_write(const tex_t *t, sub_t *s);       // the GPU is about to write it: its content lives in the image from now on       // uploads the CPU's content first when needed (outside a render pass)
uint32_t level_dim(uint32_t v, unsigned level);

// nvn_vk_draw.c
int vkd_state_op(uint32_t op, const void *a);          // a state record (binds, viewports, …): 1 when it is one
void vkd_draw(sub_t *s, uint32_t op, const void *a, uint32_t size);   // OP_DRAW / OP_DRAW_IDX; size: the payload bytes
void vkd_dispatch(sub_t *s, const void *a, uint32_t size);   // OP_DISPATCH
void vkd_dlss(sub_t *s, const void *a);
int vkd_stretch(sub_t *s, vk_tex *in, uint32_t il, uint32_t ilayer, const int32_t ir[4], vk_tex *out, uint32_t ol, uint32_t olayer, const int32_t orr[4], int linear);   // by drawing; 1 when done                // OP_DLSS: a bilinear stand-in
void vkd_sub_done(sub_t *s);                           // the completion thread: s's descriptor pools and arenas are free again
void vkd_ub_shadow(uint64_t addr, uint64_t n, const void *data);   // an UPDATE_UB recorded as a GPU command: its bytes, for handle reads before the GPU runs it
void vkd_sub_committed(uint64_t value);                // the submission just recorded is number value
void vkd_tex_release(vk_tex *x);                       // destroys x's cached views (the GPU is idle)
void vkd_stats_line(char *out, size_t n);
void vkx_sublog(sub_t *s, const char *fmt, ...);
void vkx_crumb(sub_t *s);
void vkx_dump(vk_tex *x, const char *tag);              // diagnostic: level 0 / layer 0 of x as $DM_ROOT/shots/<tag>.png (after the GPU is idle)
VkImageView vkx_zero_view(VkImageViewType vt);         // a 1x1 all-zero view of that kind: what a texture binding with no expressible texture samples
uint64_t vkd_arena_bytes(void);                       // per-submission arenas alive (nvn_vk_draw.c), for the memory line
uint64_t vkd_words_bytes(void);                       // SPIR-V kept for variants (nvn_vk_draw.c), for the memory line
void vkx_ring_csdump(sub_t *s, uint64_t key, int x, int y, int z, const p_bind_buf *ubo, const p_bind_buf *ssbo);   // nvn_vk_exec.c: DM_VK_RING_CS, a dispatch's buffers into the rolling capture
void vkx_prof_begin(VkCommandBuffer cb, const char *fmt, ...);   // nvn_vk_exec.c: DM_VK_GPU_PROF, GPU time of what is recorded up to the matching vkx_prof_end
void vkx_prof_end(VkCommandBuffer cb);
int vkx_occ_begin(VkCommandBuffer cb); void vkx_occ_end(VkCommandBuffer cb, int q);   // nvn_vk_exec.c: DM_VK_RING_OCC, a draw's pixel count into the rolling capture (-1: not counted)
int vkx_ring_final(void);   // nvn_vk_exec.c: DM_VK_RING_FINAL, the rolling capture keeps the presented picture
void vkd_bisect_tick(uint64_t n);   // nvn_vk_draw.c: DM_VK_BISECT, the skip range read again
void vkx_ring_csdump2(sub_t *s, const char *want, uint64_t key, int x, int y, int z, const p_bind_buf *ubo, const p_bind_buf *ssbo);   // nvn_vk_exec.c: as vkx_ring_csdump for a list given by the caller
void vkx_sg_probe(void);   // nvn_vk_sgprobe.c: logs how the driver groups compute threads into subgroups
int vkd_clear_rect(VkRect2D *r, int *partial);   // nvn_vk_draw.c: the scissor a clear is limited to, clamped to the target; 0 = empty
void vkx_stage_snap_depth(sub_t *s, vk_tex *x, const char *what);   // nvn_vk_exec.c: DM_VK_RING_STAGES snapshot of a 32-bit float depth image
void vkx_stage_snap(sub_t *s, vk_tex *x, const char *what);   // nvn_vk_exec.c: DM_VK_RING_STAGES snapshot of a scene-sized colour image
int vk_pool_find_part(uint64_t addr, uint64_t len, VkBuffer *buf, uint64_t *off, uint64_t *avail);   // nvn_vk_mem.c: a range past its pool's end resolves to what the pool has
int vkd_trace_small(vk_tex **out, int max);            // tiny float textures the traced frame's draws sampled (their texels logged at the traced present)
void vkx_log_floats(vk_tex *x, const char *why);       // diagnostic: a tiny float texture's texels in the log (after the GPU is idle)
int vkd_trace_targets(vk_tex **out, int max);          // the G-buffer pass's targets seen while tracing (dumped at the traced present)
int vkd_tracing(void);                                 // the frame is being traced: the executor logs its clears too
void vkd_trace_arm(void);                              // trace the next frame's draws on demand (a DM_ROOT/trace_now file); pair with a forced shot at the next present
vk_tex *vkx_clone(sub_t *s, vk_tex *x, int keep);      // a copy of x's image as it is now (outside a render pass), freed once this submission is done (keep: never freed, for diagnostics)                              // DM_VK_CRUMBS: the GPU writes a counter after this command (ends the render pass)       // appends to s->log
int vkd_origin_ll(void);
void vkd_pcache_tick(uint64_t frame);
VkPipeline vkd_stretch_get(VkFormat f, VkPipelineLayout *pl, VkDescriptorSetLayout *dsl);   // the colour stretch pipeline (one combined sampler, a 16-byte rectangle push)
VkSampler vkd_linear_sampler(void);                    // linear filtering, clamp to edge
uint64_t vkx_last_fence(void);                         // the last submission's number on the timeline semaphore
VkSemaphore vkx_timeline(void);
VkResult vkx_submit(const VkSubmitInfo *si, VkFence f);   // a submission of the caller's own, with the queue's lock
void vkd_front_present(vk_tex *x, const int crop[4]); // nvn_vk_front.c: the presented image to the app's screen
void vkd_front_stats(uint64_t *shown, uint64_t *dropped);                  // saves the pipeline cache to disk now and then
int vkd_front_text(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, int min_chars, char *out, int out_cap);   // the app's user types the text (1 OK / 0 cancel / -1 no link)
void vkd_after_sync(void);                             // DM_VK_SYNC: a submission just finished (diagnostic dumps)                               // the window origin is lower left (GL): images are stored bottom-up
