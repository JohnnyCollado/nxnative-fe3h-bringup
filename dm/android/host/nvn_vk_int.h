// The Android Vulkan backend's internals shared by its executor (nvn_vk_exec.c: submissions, textures, render targets, copies) and its draw path (nvn_vk_draw.c:
// shaders, pipelines, bindings, draws). Not an interface for the rest of the host.
#pragma once
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
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
} vk_tex;

struct vkx_rt { int has_rt, active; p_set_rt rt; VkExtent2D ext, want; };   // want: the render area of a pass with no attachments (the draw's viewport)
extern struct vkx_rt g_rt;            // the bound render targets and the open render pass
extern sub_t *g_cur;                  // the submission being recorded
extern uint64_t g_fence;              // the last submitted submission's number
extern _Atomic uint64_t g_applied;    // the last one the GPU finished (and whose effects are applied)
extern uint64_t g_stats_draws;
extern uint64_t g_presents;           // frames presented so far

void barrier_all(VkCommandBuffer cb);
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
int vkd_trace_targets(vk_tex **out, int max);          // the G-buffer pass's targets seen while tracing (dumped at the traced present)
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
void vkd_after_sync(void);                             // DM_VK_SYNC: a submission just finished (diagnostic dumps)                               // the window origin is lower left (GL): images are stored bottom-up
