// The Android host's Vulkan backend, step V1c: draws. What nvn_mtl.m does on the Mac for a draw, in Vulkan form: the bound program's shaders (SPIR-V from
// dm/gfx/tools/vk_spirv.py, described by dm/android/tools/vk_index.py's index.txt), a pipeline for the bound state, the shaders' resources resolved from the
// NVN bindings, and the draw itself. Compute dispatches, DrawTexture, transform feedback and the texture-pool shaders (95 of 16,888) are not done yet.
//
// Shaders: the translator numbers each shader's resources from 0 in descriptor set 0 (support buffer, then textures, uniform buffers, storage buffers), independently per
// stage, so the vertex and pixel shaders of one program collide. Each module is patched as it is loaded: its set becomes its stage's (vertex 0, pixel 1, geometry 2,
// compute 0) and its bindings are renumbered densely. The descriptor types, array sizes, image dimensions and vertex inputs are read from the SPIR-V itself; what each
// binding means (which uniform slot, which texture handle) from the index (Ryujinx's ShaderProgramInfo: dm/third_party/ryujinx …/TextureHandle.cs for the handles).
// Constant buffer slots as dmsh_build.py reflect reads them: 0 and 2 the driver's segments (c0: storage buffer table, base instance; c2: the bound handles,
// dmsh_abi.h), 1 the shader's own c1, 3-16 uniform slot n - 3, 24 the push constants.
// State: everything this driver allows as dynamic state is set per draw (viewport, scissor, depth / stencil, cull, front face, vertex input, primitive restart,
// depth bias); blend, channel masks, target formats, topology, polygon mode and sample count are the pipeline's key (no extended dynamic state 3 here).
// Orientation as the Mac: with the lower-left (GL) origin the image is stored bottom-up; Vulkan's clip y points down where Metal's points up, so the viewport is
// flipped exactly when the Mac's is not (nvn_mtl.m g_origin_ll). Depth in the GL convention (near = -w) uses depth clip control's -1..1 range.
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <vulkan/vulkan.h>
#include "dmsh_abi.h"
#include "nvn_fmt_vk.h"
#include "nvn_vk_int.h"

int nvn_dpool_base(const void *pool, int tex);   // hle_nvn.c: the numbering base of a texture / sampler pool

// ---------- Orientation and depth convention (hle_nvn.c sets them) ----------
static _Atomic int g_origin_ll = 1, g_queue_origin_ll, g_depth_zero;
void mtl_set_origin_lower_left(int on) { if (atomic_exchange(&g_origin_ll, on) != on) VLOG("window origin: %s", on ? "lower left (GL: images stored bottom-up)" : "upper left"); }
void mtl_set_queue_origin_lower_left(int on) { g_queue_origin_ll = on != 0; }
void mtl_set_depth_zero(int on) { if (atomic_exchange(&g_depth_zero, on) != on) VLOG("depth convention: %s", on ? "near = 0 (Vulkan's)" : "near = -w (GL: depth clip control -1..1)"); }
int vkd_origin_ll(void) { return atomic_load(&g_origin_ll); }

// ---------- Registered textures and samplers (texture / sampler pool IDs) ----------
typedef struct { int ok, has_v; tex_t t; view_t v; } treg_t;
static treg_t *g_treg; static int g_ntreg;
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static int g_nsreg;
static pthread_mutex_t g_reg_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_set_tex_pool_n(int n) { (void)n; }
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (id < 0 || id >= (1 << 20)) return;
    pthread_mutex_lock(&g_reg_mu);
    if (id >= g_ntreg) { int n = id + 1024; g_treg = realloc(g_treg, (size_t)n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (size_t)(n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    treg_t *r = &g_treg[id]; r->ok = t != NULL; if (t) r->t = *t; r->has_v = v != NULL; if (v) r->v = *v;
    pthread_mutex_unlock(&g_reg_mu);
}
void mtl_reg_image(int id, const tex_t *t, const view_t *v) { mtl_reg_tex(id, t, v); }
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0 || id >= (1 << 12) * 16) return;
    pthread_mutex_lock(&g_reg_mu);
    if (id >= g_nsreg) { int n = id + 256; g_sreg = realloc(g_sreg, (size_t)n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, (size_t)n); memset(g_sreg_ok + g_nsreg, 0, (size_t)(n - g_nsreg)); g_nsreg = n; }
    g_sreg[id] = *p; g_sreg_ok[id] = 1;
    pthread_mutex_unlock(&g_reg_mu);
}
static int treg_get(int id, treg_t *out) {
    pthread_mutex_lock(&g_reg_mu); int ok = id >= 0 && id < g_ntreg && g_treg[id].ok; if (ok) *out = g_treg[id]; pthread_mutex_unlock(&g_reg_mu); return ok;
}
static int sreg_get(int id, samp_par *out) {
    pthread_mutex_lock(&g_reg_mu); int ok = id >= 0 && id < g_nsreg && g_sreg_ok[id]; if (ok) *out = g_sreg[id]; pthread_mutex_unlock(&g_reg_mu); return ok;
}

// ---------- The shader index and the shaders ----------
enum { BK_ZERO, BK_SUPPORT, BK_CBUF, BK_SBUF, BK_TEX, BK_IMG, BK_POOL_TEX, BK_POOL_SAMP };
// The texture-pool shaders (dmsh_abi.h DMSH_KIND_TEX_POOL / SAMP_POOL): arrays the shader indexes by the texture / sampler ID it reads from the game's data (handle type
// Direct). Declared with a million elements (every possible ID); the array is shrunk to pool_n() elements, which covers the IDs the game registers, and its unused
// elements stay empty (a partially bound binding). Element i is the bound pool's ID i (registry ID tex_base + i, as nvn_mtl.m pool_gpu adds the pool's base)
static uint32_t pool_n(void) {
    static uint32_t n; if (n) return n;
    uint32_t lim = g_vk.lim.maxPerStageDescriptorSampledImages, want = 8192;
    n = lim > 2 * want + 256 ? want : lim > 512 ? (lim - 256) / 2 : 128;
    VLOG("texture-pool arrays: %u elements (the driver allows %u sampled images per stage)", n, lim);
    return n;
}
typedef struct {
    uint32_t vb; VkDescriptorType dt; uint32_t count;   // the dense binding, its descriptor type and array size (from the SPIR-V)
    uint8_t kind; int slot;                             // BK_CBUF: constant buffer slot; BK_SBUF: NVN storage buffer slot
    int cbs, handle, flags;                             // BK_TEX / BK_IMG: Ryujinx's cbuf slots and packed handle offsets
    uint8_t dim, arrayed, ms;                           // SPIR-V image dimension (0 1D, 1 2D, 2 3D, 3 cube, 5 buffer)
} sbind;
typedef struct shd {
    uint64_t key; int stage;                            // NVN stage: 0 vertex, 1 pixel, 2 geometry, 5 compute
    int loaded, bad; VkShaderModule mod; VkDescriptorSetLayout dsl;
    sbind *b; int nb;
    uint32_t in_mask; uint8_t in_kind[32];              // vertex inputs by location: 0 float, 1 int, 2 uint (a pixel shader: its input locations)
    uint32_t out_mask;                                  // a vertex shader's output locations
    uint32_t out_frag;                                  // a pixel shader's colour outputs (location bits)
    uint32_t *words; size_t nwords;                     // a vertex shader's patched SPIR-V, for variants with extra outputs (vs_variant)
    struct { uint32_t add; VkShaderModule m; } var[4]; int nvar;
    uint8_t *c1; uint32_t c1n; VkBuffer c1buf; VkDeviceMemory c1mem;
    const char *idx;                                    // its block in the index
} shd;
enum { NSHD = 1 << 15 };
static shd *g_shd[NSHD];
static char *g_index; static int g_index_tried;
static char g_spirv_dir[512];
static pthread_mutex_t g_shd_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_n_shaders, g_n_pipelines, g_n_pipe_fail, g_n_draws_done, g_n_skip_noprog, g_n_skip_bad, g_n_skip_rt, g_n_skip_other;

static uint64_t hex64(const char *s) { uint64_t v = 0; for (int i = 0; i < 16 && s[i]; i++) { char c = s[i]; v = v << 4 | (uint64_t)(c <= '9' ? c - '0' : (c | 32) - 'a' + 10); } return v; }
static uint64_t key_u64(const uint8_t k[8]) { uint64_t v = 0; for (int i = 0; i < 8; i++) v = v << 8 | k[i]; return v; }   // the key's hex digits as one number
static void index_load(void) {
    if (g_index_tried) return;
    g_index_tried = 1;
    const char *e = getenv("DM_VK_SPIRV");
    if (e) snprintf(g_spirv_dir, sizeof g_spirv_dir, "%s", e);
    else { const char *r = getenv("DM_ROOT"); snprintf(g_spirv_dir, sizeof g_spirv_dir, "%s/../spirv", r ? r : "."); }
    char p[600]; snprintf(p, sizeof p, "%s/index.txt", g_spirv_dir);
    FILE *f = fopen(p, "rb"); if (!f) { VLOG("[RED] shader index %s missing: nothing is drawn (dm/android/tools/vk_index.py writes it)", p); return; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    g_index = malloc((size_t)n + 1); size_t got = fread(g_index, 1, (size_t)n, f); fclose(f); g_index[got] = 0;
    int cnt = 0;
    for (char *q = g_index; q && *q; ) {
        if (q[0] == 'S' && q[1] == ' ') {
            shd *s = calloc(1, sizeof *s); s->key = hex64(q + 2); s->idx = q;
            uint32_t h = (uint32_t)(s->key ^ s->key >> 29) & (NSHD - 1); while (g_shd[h]) h = (h + 1) & (NSHD - 1);
            g_shd[h] = s; cnt++;
            const char *st = q + 19; s->stage = !strncmp(st, "vs", 2) ? 0 : !strncmp(st, "ps", 2) ? 1 : !strncmp(st, "gs", 2) ? 2 : !strncmp(st, "cs", 2) ? 5 : -1;
        }
        q = strchr(q, '\n'); if (q) q++;
    }
    VLOG("shader index: %d shaders from %s", cnt, p);
}
static shd *shd_find(uint64_t key) {
    if (!g_index) return NULL;
    uint32_t h = (uint32_t)(key ^ key >> 29) & (NSHD - 1);
    for (; g_shd[h]; h = (h + 1) & (NSHD - 1)) if (g_shd[h]->key == key) return g_shd[h];
    return NULL;
}
static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_vk.mp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_vk.mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}
// a host-visible buffer of n bytes with the given contents (NULL: zeros)
static VkBuffer host_buffer(uint64_t n, const void *data, VkDeviceMemory *mem, uint8_t **map_out) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b)) return VK_NULL_HANDLE;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    if (mai.memoryTypeIndex == ~0u || vkAllocateMemory(g_vk.dev, &mai, NULL, mem)) { vkDestroyBuffer(g_vk.dev, b, NULL); return VK_NULL_HANDLE; }
    vkBindBufferMemory(g_vk.dev, b, *mem, 0);
    uint8_t *m = NULL; vkMapMemory(g_vk.dev, *mem, 0, VK_WHOLE_SIZE, 0, (void **)&m);
    if (m) { if (data) memcpy(m, data, n); else memset(m, 0, n); }
    if (map_out) *map_out = m;
    return b;
}

// The SPIR-V: resources and vertex inputs read, sets and bindings patched (see the file header)
enum { SPV_DECORATE = 71, SPV_VARIABLE = 59, SPV_TYPE_POINTER = 32, SPV_TYPE_ARRAY = 28, SPV_TYPE_RTARRAY = 29, SPV_TYPE_IMAGE = 25, SPV_TYPE_SAMPLER = 26,
       SPV_TYPE_SAMPLED_IMAGE = 27, SPV_TYPE_STRUCT = 30, SPV_TYPE_VECTOR = 23, SPV_TYPE_INT = 21, SPV_TYPE_FLOAT = 22, SPV_CONSTANT = 43,
       DEC_BLOCK = 2, DEC_BUFFER_BLOCK = 3, DEC_BUILTIN = 11, DEC_LOCATION = 30, DEC_BINDING = 33, DEC_SET = 34,
       SC_UNIFORM_CONSTANT = 0, SC_INPUT = 1, SC_UNIFORM = 2, SC_STORAGE_BUFFER = 12 };
static int idx_line(const char *blk, char tag, int binding, int *f) {   // the index line of this tag and original binding: its fields after the binding into f
    for (const char *q = strchr(blk, '\n'); q && q[1] && q[1] != 'S'; q = strchr(q + 1, '\n')) {
        if (q[1] != tag || q[2] != ' ') continue;
        int b = atoi(q + 3); if (b != binding) continue;
        const char *r = q + 3; while (*r && *r != ' ' && *r != '\n') r++;
        for (int i = 0; i < 6; i++) { while (*r == ' ') r++; if (*r == '\n' || !*r) { f[i] = 0; continue; } f[i] = atoi(r); while (*r && *r != ' ' && *r != '\n') r++; }
        return 1;
    }
    return 0;
}
static int c0_ssbo_base(int stage) { return stage == 0 ? DMSH_C0_SSBO_VS : stage == 1 ? DMSH_C0_SSBO_PS : stage == 2 ? DMSH_C0_SSBO_GS : stage == 5 ? DMSH_C0_SSBO_CS : -1; }
static int shd_load(shd *s) {
    if (s->loaded) return !s->bad;
    s->loaded = 1; s->bad = 1;
    char p[600]; snprintf(p, sizeof p, "%s/%016llx.spv", g_spirv_dir, (unsigned long long)s->key);
    FILE *f = fopen(p, "rb"); if (!f) { VLOG("[RED] shader %016llx: %s missing", (unsigned long long)s->key, p); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)n); size_t nw = fread(w, 1, (size_t)n, f) / 4; fclose(f);
    if (nw < 5 || w[0] != 0x07230203) { VLOG("[RED] shader %016llx: not SPIR-V", (unsigned long long)s->key); free(w); return 0; }
    uint32_t bound = w[3];
    uint32_t *def = calloc(bound, 4), *bind_w = calloc(bound, 4), *set_w = calloc(bound, 4), *loc = calloc(bound, 4); uint8_t *blk = calloc(bound, 1), *bi = calloc(bound, 1);
    uint32_t *vars = malloc(nw * 4); int nv = 0; uint32_t *name_at = calloc(bound, 4);
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == SPV_DECORATE && len >= 3 && w[i + 1] < bound) {
            uint32_t t = w[i + 1], d = w[i + 2];
            if (d == DEC_BINDING) bind_w[t] = (uint32_t)i + 3; else if (d == DEC_SET) set_w[t] = (uint32_t)i + 3; else if (d == DEC_LOCATION) loc[t] = w[i + 3] + 1;
            else if (d == DEC_BUILTIN) bi[t] = 1; else if (d == DEC_BLOCK || d == DEC_BUFFER_BLOCK) blk[t] = (uint8_t)d;
        }
        else if (op == SPV_VARIABLE && len >= 4) vars[nv++] = (uint32_t)i;
        else if (op == 5 /* OpName */ && len >= 3 && w[i + 1] < bound) name_at[w[i + 1]] = (uint32_t)i + 2;
        else if ((op >= SPV_TYPE_INT && op <= SPV_TYPE_POINTER) || op == SPV_CONSTANT) { uint32_t id = op == SPV_CONSTANT ? w[i + 2] : w[i + 1]; if (id < bound) def[id] = (uint32_t)i; }
        i += len;
    }
    const char *bl = s->idx; int stage = s->stage, set = stage == 1 ? 1 : stage == 2 ? 2 : 0; uint64_t pool_flags = 0;
    s->b = calloc((size_t)nv + 1, sizeof *s->b);
    VkDescriptorSetLayoutBinding *lb = calloc((size_t)nv + 1, sizeof *lb);
    int ok = 1;
    for (int k = 0; k < nv; k++) {
        size_t i = vars[k]; uint32_t ptr = w[i + 1], id = w[i + 2], sc = w[i + 3];
        if (sc == 3 /* Output */ && stage == 0 && loc[id] && !bi[id] && loc[id] - 1 < 32) { s->out_mask |= 1u << (loc[id] - 1); continue; }
        if (sc == SC_INPUT && stage == 1 && loc[id] && !bi[id] && loc[id] - 1 < 32) { s->in_mask |= 1u << (loc[id] - 1); continue; }
        if (sc == 3 && stage == 1 && loc[id] && !bi[id] && loc[id] - 1 < 32) { s->out_frag |= 1u << (loc[id] - 1); continue; }
        if (sc == SC_INPUT && stage == 0 && loc[id] && !bi[id] && ptr < bound && def[ptr]) {   // a vertex input
            uint32_t t = w[def[ptr] + 3]; if (t < bound && def[t] && (w[def[t]] & 0xffff) == SPV_TYPE_VECTOR) t = w[def[t] + 2];
            uint32_t l = loc[id] - 1; if (l < 32 && t < bound && def[t]) { s->in_mask |= 1u << l; s->in_kind[l] = (w[def[t]] & 0xffff) == SPV_TYPE_INT ? (w[def[t] + 3] ? 1 : 2) : 0; }
            continue;
        }
        if (!bind_w[id] || (sc != SC_UNIFORM_CONSTANT && sc != SC_UNIFORM && sc != SC_STORAGE_BUFFER)) continue;
        uint32_t t = ptr < bound && def[ptr] ? w[def[ptr] + 3] : 0, count = 1;
        uint32_t len_id = 0;
        if (t < bound && def[t] && (w[def[t]] & 0xffff) == SPV_TYPE_ARRAY) { uint32_t c = w[def[t] + 3]; len_id = c; count = c < bound && def[c] ? w[def[c] + 3] : 1; t = w[def[t] + 2]; }
        else if (t < bound && def[t] && (w[def[t]] & 0xffff) == SPV_TYPE_RTARRAY) { count = 4096; t = w[def[t] + 2]; }
        sbind *b = &s->b[s->nb]; uint32_t orig = w[bind_w[id]];
        b->vb = (uint32_t)s->nb; b->count = count;
        uint32_t top = t < bound && def[t] ? w[def[t]] & 0xffff : 0, img = top == SPV_TYPE_SAMPLED_IMAGE ? w[def[t] + 2] : top == SPV_TYPE_IMAGE ? t : 0;
        if (img && img < bound && def[img]) { b->dim = (uint8_t)w[def[img] + 3]; b->arrayed = (uint8_t)w[def[img] + 5]; b->ms = (uint8_t)w[def[img] + 6]; }
        int fl[6] = {0};
        if (sc == SC_STORAGE_BUFFER || (sc == SC_UNIFORM && blk[t] == DEC_BUFFER_BLOCK)) {
            b->dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b->kind = BK_ZERO;
            if (idx_line(bl, 'B', (int)orig, fl) && fl[0] == 0 && c0_ssbo_base(stage) >= 0) {
                int byte = fl[1] * 4; b->kind = BK_SBUF;
                if (stage == 5 && byte >= DMSH_C0_CS_UBO && byte < DMSH_C0_CS_UBO + DMSH_C0_CS_UBO_SLOTS * DMSH_C0_ROW) b->slot = 100 + (byte - DMSH_C0_CS_UBO) / DMSH_C0_ROW;   // compute: a uniform slot read through its c0 address
                else b->slot = (byte - c0_ssbo_base(stage)) / DMSH_C0_ROW;
            }
        } else if (sc == SC_UNIFORM) {
            b->dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            if (idx_line(bl, 'C', (int)orig, fl)) { b->kind = BK_CBUF; b->slot = fl[0]; }
            else {
                // not in the description: the translator leaves driver segment c0 out of it when the shader reads it only for a storage buffer's base (vp_c0 in
                // 14c868df02293941); its variable name says the bank, as dmsh_build.py reflect reads it (vp_c<n>, fp_c<n>, cp_c<n>, gp_c<n>)
                const char *nm = name_at[id] ? (const char *)&w[name_at[id]] : ""; const char *c = strstr(nm, "_c"); int bank = -1;
                if (c && c[2] >= '0' && c[2] <= '9') { char *e; long v = strtol(c + 2, &e, 10); if (!*e) bank = (int)v; }
                if (bank >= 0 && strcmp(nm, "support_buffer")) { b->kind = BK_CBUF; b->slot = bank; }
                else b->kind = orig == 0 || !strcmp(nm, "support_buffer") ? BK_SUPPORT : BK_ZERO;
            }
        } else if (top == SPV_TYPE_SAMPLED_IMAGE) {
            b->dt = b->dim == 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (idx_line(bl, 'T', (int)orig, fl)) { b->kind = BK_TEX; b->cbs = fl[1]; b->handle = fl[2]; b->flags = fl[4]; } else b->kind = BK_ZERO;
        } else if (top == SPV_TYPE_IMAGE) {
            int storage = w[def[img] + 7] == 2;
            b->dt = b->dim == 5 ? (storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) : storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (idx_line(bl, storage ? 'I' : 'T', (int)orig, fl)) { b->kind = storage ? BK_IMG : BK_TEX; b->cbs = fl[1]; b->handle = fl[2]; b->flags = fl[4]; } else b->kind = BK_ZERO;
        } else if (top == SPV_TYPE_SAMPLER) { b->dt = VK_DESCRIPTOR_TYPE_SAMPLER; b->kind = BK_ZERO; if (idx_line(bl, 'T', (int)orig, fl)) { b->cbs = fl[1]; b->handle = fl[2]; b->flags = fl[4]; } }
        else { VLOG("[RED] shader %016llx: resource binding %u of an unknown type (op %u)", (unsigned long long)s->key, orig, top); ok = 0; continue; }
        if (count > 1024) {   // a pool array: indexed by the IDs themselves (handle type 4, Direct)
            int direct = (uint32_t)b->handle >> 28 == 4;
            if (direct && b->dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE && b->dim != 5 && count == pool_n()) b->kind = BK_POOL_TEX;   // its length constant, shared with another pool array, was shrunk already
            else if (direct && b->dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE && b->dim != 5 && count > pool_n()) {
                // shrink the declared length: its constant must be used by array types alone (else another use would change)
                int other = 0;
                for (size_t i = 5; i < nw && !other; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) break;
                    if (!(op == SPV_CONSTANT && w[i + 2] == len_id) && !(op == SPV_TYPE_ARRAY && w[i + 3] == len_id) && op != SPV_DECORATE && op != 5 /* OpName */)
                        for (uint32_t k = 1; k < len; k++) if (w[i + k] == len_id) { other = 1; break; }
                    i += len; }
                if (!other && len_id < bound && def[len_id]) { w[def[len_id] + 3] = pool_n(); count = pool_n(); b->kind = BK_POOL_TEX; }
            } else if (direct && b->dt == VK_DESCRIPTOR_TYPE_SAMPLER && count <= 4096) b->kind = BK_POOL_SAMP;
            b->count = count;
            if (b->kind != BK_POOL_TEX && b->kind != BK_POOL_SAMP) { ok = 0; VRED_ONCE("a shader array of %u elements the host cannot bind (first: %016llx)", count, (unsigned long long)s->key); }
        }
        w[bind_w[id]] = b->vb; if (set_w[id]) w[set_w[id]] = (uint32_t)set;
        if (b->kind == BK_POOL_TEX || b->kind == BK_POOL_SAMP) pool_flags |= 1ull << (s->nb < 63 ? s->nb : 63);
        lb[s->nb] = (VkDescriptorSetLayoutBinding){b->vb, b->dt, count > 4096 && b->kind != BK_POOL_TEX ? 1 : count, stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : stage == 1 ? VK_SHADER_STAGE_FRAGMENT_BIT :
                                                   stage == 2 ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_COMPUTE_BIT, NULL};
        s->nb++;
    }
    // c1: the shader's own constant segment, from the index's S line (hex)
    const char *c = bl; for (int sp = 0; sp < 4 && c; sp++) { c = strchr(c, ' '); if (c) c++; }
    if (c && *c != '-') {
        const char *e = c; while (*e && *e != '\n' && *e != ' ') e++;
        s->c1n = (uint32_t)(e - c) / 2; s->c1 = malloc(s->c1n + 16);
        for (uint32_t i = 0; i < s->c1n; i++) { unsigned v; sscanf(c + 2 * i, "%2x", &v); s->c1[i] = (uint8_t)v; }
    }
    {   // DM_VK_DUMP_SPV: the patched modules as $DM_ROOT/spv_patched/<key>.spv (checked on the PC with spirv-val)
        static int dump = -1; if (dump < 0) dump = getenv("DM_VK_DUMP_SPV") != NULL;
        if (dump) { char d[600], q[700]; snprintf(d, sizeof d, "%s/spv_patched", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(d, 0755);
            snprintf(q, sizeof q, "%s/%016llx.spv", d, (unsigned long long)s->key); FILE *o = fopen(q, "wb"); if (o) { fwrite(w, 4, nw, o); fclose(o); } }
    }
    if (ok) {
        VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, nw * 4, w};
        VkDescriptorBindingFlags *bf = calloc((size_t)s->nb + 1, sizeof *bf);
        for (int i = 0; i < s->nb && i < 64; i++) if (pool_flags >> i & 1) bf[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, NULL, (uint32_t)s->nb, bf};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pool_flags ? &bfi : NULL, 0, (uint32_t)s->nb, lb};
        VkResult r1 = vkCreateShaderModule(g_vk.dev, &mi, NULL, &s->mod), r2 = r1 ? r1 : vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &s->dsl);
        free(bf);
        if (r1 || r2) VLOG("[RED] shader %016llx: module %d / set layout %d", (unsigned long long)s->key, r1, r2);
        else {
            s->bad = 0; g_n_shaders++;
            if (s->c1n) { uint64_t sz = (s->c1n + 255) & ~255ull; uint8_t *tmp = calloc(1, sz); memcpy(tmp, s->c1, s->c1n); s->c1buf = host_buffer(sz, tmp, &s->c1mem, NULL); free(tmp); }
        }
    }
    if (stage == 0 && !s->bad) { s->words = w; s->nwords = nw; w = NULL; }
    free(name_at); free(w); free(def); free(bind_w); free(set_w); free(loc); free(blk); free(bi); free(vars); free(lb);
    return !s->bad;
}
// A program's shaders by NVN stage, cached by the program's serial (a new SetShaders gives a new serial)
typedef struct { uint64_t serial; shd *st[6]; } prog_ent;
enum { NPROG = 4096 };
static prog_ent g_prog[NPROG];
static void prog_shaders(const prog_t *p, shd *out[6]) {
    memset(out, 0, 6 * sizeof *out);
    if (!p || !p->serial) return;
    prog_ent *e = &g_prog[p->serial % NPROG];
    if (e->serial == p->serial) { memcpy(out, e->st, sizeof e->st); return; }
    pthread_mutex_lock(&g_shd_mu); index_load(); pthread_mutex_unlock(&g_shd_mu);
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {
        uint8_t k[8]; if (!prog_key(p, i, k)) continue;
        shd *s = shd_find(key_u64(k));
        if (!s) { VLOG("[RED] program %llu stage %u: shader %016llx is not in the index", (unsigned long long)p->serial, i, (unsigned long long)key_u64(k)); continue; }
        if (s->stage >= 0 && s->stage < 6) out[s->stage] = s;
    }
    e->serial = p->serial; memcpy(e->st, out, sizeof e->st);
}

// ---------- Bound state ----------
static struct {
    shd *sh[6]; int has_prog;
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t attr[16]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][256], samp[6][128], img[6][32];
    float vp[4]; int has_vp; float dr[2]; p_scissor sc; int has_sc;
    uint32_t st_vmask[2], st_wmask[2], st_ref[2]; p_poly_offset po; float blend_color[4];
    int depth_clamp, rast_discard; p_prim_restart restart;
    uint8_t push[2][DMSH_PUSH_SIZE];
    int tex_base, samp_base, init;
} S;
static void state_init(void) {
    if (S.init) return;
    S.init = 1;
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){(uint64_t)i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu; S.ms.enable = 1;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds.df = NVN_DEPTH_FUNC_LESS; S.ds.sf_f = S.ds.sf_b = NVN_STENCIL_FUNC_ALWAYS;
    S.ds.fail_f = S.ds.zfail_f = S.ds.zpass_f = S.ds.fail_b = S.ds.zfail_b = S.ds.zpass_b = NVN_STENCIL_OP_KEEP;
    S.dr[0] = 0; S.dr[1] = 1; S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff;
}
static inline void bh_fresh(const p_bind_handles *b) { if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); }
int vkd_state_op(uint32_t op, const void *a) {
    state_init();
    switch (op) {
        case OP_BIND_PROGRAM: { const p_bind_program *b = a; shd *st[6]; prog_shaders(b->prog, st);
            for (int i = 0; i < 6; i++) if (b->stages >> i & 1) S.sh[i] = st[i];   // stage bits 0 vertex 1 pixel 2 geometry 3 / 4 tessellation 5 compute (NVN_SHADER_STAGE_*_BIT)
            S.has_prog = 1; return 1; }
        case OP_BIND_STATE: { const p_bind_state *b = a; blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); if (bs.target < 8) S.blend[bs.target] = bs; break;
                case 1: memcpy(&S.color, b->bytes, sizeof S.color); break;
                case 2: memcpy(&S.chmask, b->bytes, sizeof S.chmask); break;
                case 3: memcpy(&S.ms, b->bytes, sizeof S.ms); break;
                case 4: memcpy(&S.poly, b->bytes, sizeof S.poly); break;
                case 5: memcpy(&S.ds, b->bytes, sizeof S.ds); break;
            }
            return 1; }
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(S.attr + b->first, b->st, 4 * (size_t)b->n); if (b->first + b->n > S.nattr) S.nattr = b->first + b->n; } return 1; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(S.stream + b->first, b->st, 8 * (size_t)b->n); if (b->first + b->n > S.nstream) S.nstream = b->first + b->n; } return 1; }
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf t; const p_bind_vbuf *b = vbuf_rec(op, a, &t); if (b->first >= 0 && b->first + b->n <= 16) memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * (size_t)b->n); return 1; }
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf t; const p_bind_buf *b = bind_buf_rec(op, a, &t); if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ubo[b->stage][b->index] = *b; return 1; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ssbo[b->stage][b->index] = *b; return 1; }
        case OP_BIND_TEX: case OP_BIND_TEX1: { p_bind_handles t; const p_bind_handles *b = bind_rec(op, a, &t); if (b->n < 0 || b->n > 32) return 1;
            bh_fresh(b); if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= 256) memcpy(&S.tex[b->stage][b->first], b->h, 8 * (size_t)b->n); return 1; }
        case OP_BIND_SAMP: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= 128) memcpy(&S.samp[b->stage][b->first], b->h, 8 * (size_t)b->n); return 1; }
        case OP_BIND_IMG: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) memcpy(&S.img[b->stage][b->first], b->h, 8 * (size_t)b->n); return 1; }
        case OP_STENCIL: { const p_stencil *b = a; for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = (uint32_t)b->value_mask; else if (b->which == 1) S.st_wmask[f] = (uint32_t)b->mask; else S.st_ref[f] = (uint32_t)b->ref; } return 1; }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; return 1;
        case OP_VIEWPORTS: { const p_viewports *b = a; if (b->first == 0 && b->n > 0) { memcpy(S.vp, b->v[0], 16); S.has_vp = 1; } return 1; }
        case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.has_sc = 1; return 1;
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; return 1; }
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)a)->on; return 1;
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); return 1;
        case OP_PUSH_CONST: { const p_push_const *b = a; if ((uint32_t)b->offset + b->size <= DMSH_PUSH_SIZE) memcpy(S.push[b->compute ? 1 : 0] + b->offset, b->data, b->size); return 1; }
        case OP_PRIM_RESTART: S.restart = *(const p_prim_restart *)a; return 1;
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)a)->on; return 1;
        case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: { int tex = op == OP_SET_TEX_POOL, b = nvn_dpool_base(((const p_set_pool *)a)->pool, tex); if (tex) S.tex_base = b; else S.samp_base = b; return 1; }
        case OP_ALPHA_REF: case OP_VP_SWIZZLE: case OP_TF_BIND: case OP_TF_STATE: case OP_BEGIN: return 1;
        case OP_RENDER_COND: VRED_ONCE("conditional rendering is ignored (everything draws)"); return 1;
        default: return 0;
    }
}

// ---------- UPDATE_UB shadow: uniform bytes the GPU has not written yet, for handle reads at record time ----------
enum { SH_ENT = 1 << 15, SH_BUCKETS = 1 << 16, SH_BYTES = 16 << 20 };
typedef struct { uint64_t addr, n, seq, fence, prev[1]; uint64_t pos; } sh_ent;   // prev: the bucket chain (seq + 1, 0 = end), for the bucket of addr >> 8
static sh_ent g_she[SH_ENT]; static uint64_t g_sh_bucket[SH_BUCKETS]; static uint8_t *g_sh_bytes; static uint64_t g_sh_seq, g_sh_pos;
void vkd_ub_shadow(uint64_t addr, uint64_t n, const void *data) {
    if (!n || n > 65536) return;
    if (!g_sh_bytes) g_sh_bytes = malloc(SH_BYTES);
    uint64_t pos = g_sh_pos; if (pos % SH_BYTES + n > SH_BYTES) pos += SH_BYTES - pos % SH_BYTES;
    memcpy(g_sh_bytes + pos % SH_BYTES, data, n); g_sh_pos = pos + n;
    for (uint64_t blk = addr >> 8; blk <= (addr + n - 1) >> 8; blk++) {
        uint64_t seq = ++g_sh_seq; sh_ent *e = &g_she[seq % SH_ENT];
        uint32_t bk = (uint32_t)(blk * 0x9E3779B1u) & (SH_BUCKETS - 1);
        *e = (sh_ent){addr, n, seq, g_fence + 1, {g_sh_bucket[bk]}, pos};
        g_sh_bucket[bk] = seq + 1;
    }
}
void vkd_sub_committed(uint64_t value) { (void)value; }
// the 32-bit word at addr as the GPU will see it once the submissions recorded so far have run
static uint32_t ub_word(uint64_t addr) {
    uint64_t applied = atomic_load(&g_applied);
    uint32_t bk = (uint32_t)((addr >> 8) * 0x9E3779B1u) & (SH_BUCKETS - 1);
    for (uint64_t link = g_sh_bucket[bk], steps = 0; link && steps < 4096; steps++) {
        uint64_t seq = link - 1; sh_ent *e = &g_she[seq % SH_ENT];
        if (e->seq != seq || g_sh_seq - seq >= SH_ENT || g_sh_pos - e->pos > SH_BYTES - 65536) break;   // overwritten: older entries are gone too
        if (e->fence <= applied) break;                                                                  // the GPU has written it (and everything older): memory holds it
        if (addr >= e->addr && addr + 4 <= e->addr + e->n) { uint32_t v; memcpy(&v, g_sh_bytes + e->pos % SH_BYTES + (addr - e->addr), 4); return v; }
        link = e->prev[0];
    }
    uint32_t v; memcpy(&v, (const void *)(uintptr_t)addr, 4); return v;
}

// ---------- Per-submission memory: constant blocks and descriptor sets ----------
static uint8_t *arena_alloc(sub_t *s, uint64_t n, VkBuffer *buf, uint64_t *off) {
    const uint64_t al = 256; n = (n + al - 1) & ~(al - 1);
    for (;;) {
        if (s->ar_cur < s->nar) {
            vkx_arena *a = &s->arenas[s->ar_cur];
            if (a->used + n <= a->size) { *buf = a->b; *off = a->used; a->used += n; return a->map + *off; }
            s->ar_cur++; continue;
        }
        GROW(s->arenas, s->nar, s->car); vkx_arena *a = &s->arenas[s->nar];
        a->size = n > (4u << 20) ? n : 4u << 20; a->used = 0;
        a->b = host_buffer(a->size, NULL, &a->m, &a->map);
        if (!a->b) { VRED_ONCE("per-draw constant memory could not be allocated"); return NULL; }
        s->nar++;
    }
}
static VkDescriptorSet set_alloc(sub_t *s, VkDescriptorSetLayout l) {
    for (;;) {
        if (s->dp_cur < s->ndp) {
            VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, s->dpools[s->dp_cur], 1, &l}; VkDescriptorSet ds;
            VkResult r = vkAllocateDescriptorSets(g_vk.dev, &ai, &ds);
            if (!r) return ds;
            if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) { VRED_ONCE("vkAllocateDescriptorSets %d", r); return VK_NULL_HANDLE; }
            s->dp_cur++; continue;
        }
        VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192},
                                     {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 65536}, {VK_DESCRIPTOR_TYPE_SAMPLER, 16384}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
                                     {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2048, sizeof ps / sizeof *ps, ps};
        VkDescriptorPool p; if (vkCreateDescriptorPool(g_vk.dev, &pi, NULL, &p)) { VRED_ONCE("descriptor pool creation failed"); return VK_NULL_HANDLE; }
        GROW(s->dpools, s->ndp, s->cdp); s->dpools[s->ndp++] = p;
    }
}
void vkd_sub_done(sub_t *s) {
    for (int i = 0; i < s->ndp && i <= s->dp_cur; i++) vkResetDescriptorPool(g_vk.dev, s->dpools[i], 0);
    s->dp_cur = 0;
    for (int i = 0; i < s->nar; i++) s->arenas[i].used = 0;
    s->ar_cur = 0;
}

// ---------- Shared buffers: the support buffer and a zero buffer ----------
static VkBuffer g_support, g_zero; static VkDeviceMemory g_support_m, g_zero_m; static uint8_t *g_support_map;
static void shared_buffers(void) {
    if (g_support) return;
    g_support = host_buffer(65536, NULL, &g_support_m, &g_support_map);
    // Ryujinx SupportBuffer (SupportBuffer.cs; nvn_mtl.m support_buf): alpha_test 0, is_bgra[8] 16, viewport_inverse 144, viewport_size 160, frag_scale_count 176,
    // render_scale float4[73] 192: scale 1 everywhere
    if (g_support_map) for (int i = 0; i < 73; i++) { float one = 1.0f; memcpy(g_support_map + 192 + 16 * i, &one, 4); }
    g_zero = host_buffer(65536, NULL, &g_zero_m, NULL);
}

// ---------- Samplers and sampling views ----------
typedef struct { samp_par p; VkSampler s; } samp_ent;
enum { NSAMP = 4096 };
static samp_ent g_samp[NSAMP]; static int g_nsamp_used;
static VkSamplerAddressMode wrap_vk(int w) {
    switch (w) { case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case NVN_WRAP_MODE_MIRRORED_REPEAT: case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
        case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case NVN_WRAP_MODE_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static pthread_mutex_t g_samp_mu = PTHREAD_MUTEX_INITIALIZER;
static VkSampler sampler_of_(const samp_par *sp);
static VkSampler sampler_of(const samp_par *sp) { pthread_mutex_lock(&g_samp_mu); VkSampler s = sampler_of_(sp); pthread_mutex_unlock(&g_samp_mu); return s; }
VkSampler vkd_linear_sampler(void) { return sampler_of(NULL); }
static VkSampler sampler_of_(const samp_par *sp) {
    static samp_par def; static int def_init;
    if (!sp) { if (!def_init) { def_init = 1; def.minf = NVN_MIN_FILTER_LINEAR; def.magf = NVN_MAG_FILTER_LINEAR; def.ws = def.wt = def.wr = NVN_WRAP_MODE_CLAMP_TO_EDGE; def.lod_max = 1000; } sp = &def; }
    uint32_t h = 2166136261u; const uint8_t *q = (const uint8_t *)sp; for (size_t i = 0; i < sizeof *sp; i++) h = (h ^ q[i]) * 16777619u;
    for (uint32_t i = h & (NSAMP - 1), k = 0; k < NSAMP; i = (i + 1) & (NSAMP - 1), k++) {
        samp_ent *e = &g_samp[i];
        if (e->s && !memcmp(&e->p, sp, sizeof *sp)) return e->s;
        if (e->s) continue;
        int mip = sp->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
        VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, NULL, {.float32 = {sp->border[0], sp->border[1], sp->border[2], sp->border[3]}}, VK_FORMAT_UNDEFINED};
        int border = sp->ws == NVN_WRAP_MODE_CLAMP_TO_BORDER || sp->wt == NVN_WRAP_MODE_CLAMP_TO_BORDER || sp->wr == NVN_WRAP_MODE_CLAMP_TO_BORDER;
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, border && g_vk.border ? &cb : NULL, 0,
            sp->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, (sp->minf & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
            sp->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
            wrap_vk(sp->ws), wrap_vk(sp->wt), wrap_vk(sp->wr), sp->lod_bias, sp->aniso > 1, sp->aniso > 1 ? (sp->aniso < g_vk.lim.maxSamplerAnisotropy ? sp->aniso : g_vk.lim.maxSamplerAnisotropy) : 1,
            sp->cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, sp->cmp_func >= 1 && sp->cmp_func <= 8 ? (VkCompareOp)(sp->cmp_func - 1) : VK_COMPARE_OP_ALWAYS,
            sp->lod_min, mip ? sp->lod_max : 0.25f,
            border && g_vk.border ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT : (sp->border[3] > 0.5f ? (sp->border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK) : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK), VK_FALSE};
        if (si.maxLod < si.minLod) si.maxLod = si.minLod;
        if (g_nsamp_used >= NSAMP * 3 / 4) { VRED_ONCE("more than %d distinct samplers", NSAMP * 3 / 4); return g_samp[0].s; }
        if (vkCreateSampler(g_vk.dev, &si, NULL, &e->s)) { e->s = VK_NULL_HANDLE; VRED_ONCE("vkCreateSampler failed"); return VK_NULL_HANDLE; }
        e->p = *sp; g_nsamp_used++;
        return e->s;
    }
    return VK_NULL_HANDLE;
}
typedef struct sview { struct sview *next; uint64_t key; VkImageView v; } sview;
static VkComponentSwizzle swz_vk(int s) { return s == NVN_SWZ_ZERO ? VK_COMPONENT_SWIZZLE_ZERO : s == NVN_SWZ_ONE ? VK_COMPONENT_SWIZZLE_ONE : (VkComponentSwizzle)(VK_COMPONENT_SWIZZLE_R + s - NVN_SWZ_R); }
// The view a shader binding samples (or writes, storage) of a registered texture: its dimension from the shader, format and range from the NVN view, the channel
// remapping composed of the view's, the texture's (TextureBuilderSetSwizzle) and the format's. VK_NULL_HANDLE when the texture cannot be seen that way
static VkImageView sample_view(vk_tex *x, const treg_t *r, const sbind *b, int storage) {
    const view_t *v = r->has_v ? &r->v : NULL;
    VkFormat fmt = x->fmt; uint8_t fswz[4];
    int vf = v && v->set_format ? v->format : r->t.d.format; uint32_t vk = 0;
    if (nvn_fmt_to_vk(vf, &vk, fswz) && vk) { if (vk != (uint32_t)x->fmt && (x->bpb && !(x->bw > 1))) fmt = (VkFormat)vk; } else nvn_fmt_to_vk(r->t.d.format, NULL, fswz);
    VkImageViewType vt; uint32_t layers_need = 1;
    switch (b->dim) {
        case 0: vt = b->arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D; if (x->type != VK_IMAGE_TYPE_1D) return VK_NULL_HANDLE; break;
        case 1: vt = b->arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; if (x->type != VK_IMAGE_TYPE_2D) return VK_NULL_HANDLE; if ((x->d.samples > 1) != (b->ms != 0)) return VK_NULL_HANDLE; break;
        case 2: vt = VK_IMAGE_VIEW_TYPE_3D; if (x->type != VK_IMAGE_TYPE_3D) return VK_NULL_HANDLE; break;
        case 3: vt = b->arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE; layers_need = 6; if (x->type != VK_IMAGE_TYPE_2D || x->layers < 6 || x->w != x->h) return VK_NULL_HANDLE; break;
        default: return VK_NULL_HANDLE;
    }
    uint32_t base_l = v && v->set_levels ? v->base_level : 0, nl = v && v->set_levels && v->nlevels ? v->nlevels : x->levels - base_l;
    uint32_t base_y = v && v->set_layers ? v->min_layer : 0, ny = v && v->set_layers && v->nlayers ? v->nlayers : x->layers - base_y;
    if (base_l >= x->levels) base_l = 0; if (base_l + nl > x->levels) nl = x->levels - base_l;
    if (base_y >= x->layers) base_y = 0; if (base_y + ny > x->layers) ny = x->layers - base_y;
    if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_3D) ny = 1;
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6; if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny -= ny % 6; if (ny < layers_need) return VK_NULL_HANDLE;
    if (storage) { nl = 1; if (!x->storage_fmt) { VRED_ONCE("texture %llu (format %d) bound as a storage image, which its format cannot be: left empty", (unsigned long long)x->serial, x->fmt); return VK_NULL_HANDLE; } fmt = x->storage_fmt; }
    VkImageAspectFlags asp = x->aspect;
    if (asp & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) asp = ((r->t.stencil || (v && v->set_dsmode && v->dsmode == 1)) && (asp & VK_IMAGE_ASPECT_STENCIL_BIT)) ? VK_IMAGE_ASPECT_STENCIL_BIT : (asp & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : asp;
    uint8_t sw[4];
    for (int i = 0; i < 4; i++) {
        int s0 = v && v->set_swizzle ? v->swz[i] : NVN_SWZ_R + i;
        if (s0 >= NVN_SWZ_R && s0 <= NVN_SWZ_A && r->t.set_swz) s0 = r->t.swz >> (3 * (s0 - NVN_SWZ_R)) & 7;
        if (s0 >= NVN_SWZ_R && s0 <= NVN_SWZ_A) s0 = fswz[s0 - NVN_SWZ_R];
        sw[i] = (uint8_t)s0;
    }
    if (storage) memcpy(sw, (uint8_t[]){NVN_SWZ_R, NVN_SWZ_G, NVN_SWZ_B, NVN_SWZ_A}, 4);   // storage views take no remapping
    uint64_t key = (uint64_t)vt | (uint64_t)fmt << 4 | (uint64_t)base_l << 20 | (uint64_t)nl << 25 | (uint64_t)(base_y & 0x7ff) << 30 | (uint64_t)(ny & 0x7ff) << 41 |
                   ((uint64_t)(sw[0] | sw[1] << 3 | sw[2] << 6 | sw[3] << 9) << 52 ^ (uint64_t)asp << 62 ^ (uint64_t)storage << 63);
    for (sview *c = x->sviews; c; c = c->next) if (c->key == key) return c->v;
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, vt, fmt, {swz_vk(sw[0]), swz_vk(sw[1]), swz_vk(sw[2]), swz_vk(sw[3])}, {asp, base_l, nl, base_y, ny}};
    VkImageView view;
    if (vkCreateImageView(g_vk.dev, &vi, NULL, &view)) { if (storage) { VRED_ONCE("storage view of texture %llu (format %d) failed", (unsigned long long)x->serial, fmt); return VK_NULL_HANDLE; } vi.format = x->fmt; if (vkCreateImageView(g_vk.dev, &vi, NULL, &view)) { VRED_ONCE("sampling view of texture %llu (format %d, view type %d) failed", (unsigned long long)x->serial, fmt, vt); return VK_NULL_HANDLE; } }
    sview *c = malloc(sizeof *c); c->key = key; c->v = view; c->next = x->sviews; x->sviews = c;
    return view;
}
void vkd_tex_release(vk_tex *x) {
    for (sview *c = x->sviews, *n; c; c = n) { n = c->next; vkDestroyImageView(g_vk.dev, c->v, NULL); free(c); }
    x->sviews = NULL;
}

// ---------- Handles: a texture binding's packed texture / sampler ID ----------
static uint32_t c2_word(int stage, uint32_t byte) {   // the driver's handle tables, as the bound handles fill them (dmsh_abi.h DMSH_C2_*)
    if (byte & 4) return 0;
    if (byte >= DMSH_C2_TEX && byte < DMSH_C2_IMG) return (uint32_t)S.tex[stage][(byte - DMSH_C2_TEX) / DMSH_C2_ROW];
    if (byte >= DMSH_C2_IMG && byte < DMSH_C2_SEPTEX) { uint32_t r = (byte - DMSH_C2_IMG) / DMSH_C2_ROW; return r < 32 ? (uint32_t)S.img[stage][r] : 0; }
    if (byte >= DMSH_C2_SEPTEX && byte < DMSH_C2_SAMP) return (uint32_t)S.tex[stage][DMSH_SEPTEX_SLOT0 + (byte - DMSH_C2_SEPTEX) / DMSH_C2_ROW];
    if (byte >= DMSH_C2_SAMP && byte < DMSH_C2_SAMP + DMSH_C2_SEP_SLOTS * DMSH_C2_ROW) return (uint32_t)S.samp[stage][(byte - DMSH_C2_SAMP) / DMSH_C2_ROW];
    return 0;
}
static uint32_t cb_word(int stage, int cb, uint32_t word, int *from_ubo) {
    *from_ubo = 0;
    if (cb == 2) return c2_word(stage, word * 4);
    if (cb >= 3 && cb <= 16) {
        const p_bind_buf *u = &S.ubo[stage][cb - 3];
        if (!u->addr || (uint64_t)word * 4 + 4 > u->size) return 0;
        *from_ubo = 1; return ub_word(u->addr + (uint64_t)word * 4);
    }
    if (cb == DMSH_PUSH_BANK) { uint32_t v = 0; if (word * 4 + 4 <= DMSH_PUSH_SIZE) memcpy(&v, S.push[stage == 5] + word * 4, 4); return v; }
    return 0;
}
// element k's texture and sampler IDs (Ryujinx TextureHandle.ReadPackedId); 0 when the binding has no handle source
static int binding_ids(int stage, const sbind *b, uint32_t k, int *tid, int *sid) {
    int tcb, scb; if (b->cbs < 0) tcb = scb = 2; else { uint32_t hi = (uint32_t)b->cbs >> 16; tcb = (uint16_t)b->cbs; scb = hi ? (int)hi - 1 : tcb; }
    uint32_t to = (uint32_t)b->handle & 0x3fff, so = (uint32_t)b->handle >> 14 & 0x3fff, type = (uint32_t)b->handle >> 28;
    if (type == 4) return 0;   // a direct pool index (the pool shaders)
    uint32_t step = 2;   // array element k is 8 bytes on, in c2 (its rows) as in a uniform buffer: nvn_mtl.m fills it from the array's offset + 8k (NVN_RB_TEXU; docs/reference/shaders.md
                         // "A table of handles past the array's length"). A 4-byte step left every odd element empty and the characters' textures mismatched (see-through characters)
    int fu, fs; uint32_t h = cb_word(stage, tcb, to + k * step, &fu);
    if (type != 0) {
        uint32_t sh = type == 3 ? so : cb_word(stage, scb, so, &fs);
        if (type == 2 || type == 3) sh <<= 20;
        h = (h & 0xfffff) | (sh & 0xfff00000u);
    }
    *tid = (int)(h & 0xfffff); *sid = (int)(h >> 20 & 0xfff);
    if (fu) { if (*tid) *tid += S.tex_base; if (*sid) *sid += S.samp_base; }   // read from game memory: the IDs are the bound pool's (nvn_mtl.m OP_SET_TEX_POOL)
    return 1;
}

// ---------- Pipelines ----------
typedef struct {
    shd *vs, *ps, *gs; uint32_t topo, poly_mode, samples, a2c, depth_clamp, clip_m1, logic_op, ncol;
    uint32_t col_fmt[8], ds_fmt, d_on, s_on, chmask; uint32_t blend_en; blend_st blend[8];
} pkey;
typedef struct pent { struct pent *next; pkey k; VkPipeline p; VkPipelineLayout pl; } pent;
enum { NPIPE = 4096 };
static pent *g_pipe[NPIPE];
static VkPipelineCache g_pcache;
// The driver's pipeline cache, kept on disk ($DM_ROOT/vk_pipelines.bin) so the compiled pipelines carry over between runs: without it the first 3D scene froze
// for 11-14 s on every start while some 140 pipelines compiled. The driver checks the data's header (its own vendor, device and cache UUID) and ignores data that
// is not its own, so a driver update or a different driver just starts empty. Saved every 300 presents while new pipelines appear (the host has no clean exit)
static uint64_t g_pcache_saved_n;
static uint64_t g_n_cpipes_all(void);
static void pcache_path(char *p, size_t n) { snprintf(p, n, "%s/vk_pipelines.bin", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); }
static void pcache_open(void) {
    if (g_pcache) return;
    char p[600]; pcache_path(p, sizeof p);
    void *data = NULL; size_t n = 0; FILE *f = fopen(p, "rb");
    if (f) { fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET); if (l > 0 && (data = malloc((size_t)l))) n = fread(data, 1, (size_t)l, f); fclose(f); }
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, n, data};
    if (vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_pcache)) { ci.initialDataSize = 0; ci.pInitialData = NULL; vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_pcache); n = 0; }
    VLOG("pipeline cache: %zu bytes loaded from %s", n, p);
    free(data);
}
void vkd_pcache_tick(uint64_t frame) {
    uint64_t made = g_n_pipelines + g_n_cpipes_all();
    if (!g_pcache || frame % 300 || made == g_pcache_saved_n) return;
    size_t n = 0; if (vkGetPipelineCacheData(g_vk.dev, g_pcache, &n, NULL) || !n) return;
    void *d = malloc(n); if (!d) return;
    if (!vkGetPipelineCacheData(g_vk.dev, g_pcache, &n, d)) {
        char p[600], t[620]; pcache_path(p, sizeof p); snprintf(t, sizeof t, "%s.tmp", p);
        FILE *f = fopen(t, "wb"); if (f) { int ok = fwrite(d, 1, n, f) == n; ok &= fclose(f) == 0; if (ok && !rename(t, p)) { g_pcache_saved_n = made; VLOG("pipeline cache: %zu bytes saved (%llu pipelines made this run)", n, (unsigned long long)made); } }
    }
    free(d);
}
static VkDescriptorSetLayout g_empty_dsl;
typedef struct { VkDescriptorSetLayout l[3]; VkPipelineLayout pl; } plent;
static plent *g_pl; static int g_npl, g_cpl;
static VkPipelineLayout layout_of(shd *vs, shd *ps, shd *gs) {
    if (!g_empty_dsl) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &g_empty_dsl); }
    VkDescriptorSetLayout l[3] = {vs ? vs->dsl : g_empty_dsl, ps ? ps->dsl : g_empty_dsl, gs ? gs->dsl : g_empty_dsl};
    for (int i = 0; i < g_npl; i++) if (!memcmp(g_pl[i].l, l, sizeof l)) return g_pl[i].pl;
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 3, l};
    VkPipelineLayout pl; if (vkCreatePipelineLayout(g_vk.dev, &pi, NULL, &pl)) return VK_NULL_HANDLE;
    GROW(g_pl, g_npl, g_cpl); memcpy(g_pl[g_npl].l, l, sizeof l); g_pl[g_npl].pl = pl; g_npl++;
    return pl;
}
static VkBlendFactor bf_vk(uint32_t f) {
    switch (f) {
        case NVN_BLEND_FUNC_ZERO: return VK_BLEND_FACTOR_ZERO; case NVN_BLEND_FUNC_ONE: return VK_BLEND_FACTOR_ONE;
        case NVN_BLEND_FUNC_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case NVN_BLEND_FUNC_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case NVN_BLEND_FUNC_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case NVN_BLEND_FUNC_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case NVN_BLEND_FUNC_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case NVN_BLEND_FUNC_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case NVN_BLEND_FUNC_CONSTANT_COLOR: return VK_BLEND_FACTOR_CONSTANT_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case NVN_BLEND_FUNC_CONSTANT_ALPHA: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp beq_vk(uint32_t e) { return e == NVN_BLEND_EQUATION_SUB ? VK_BLEND_OP_SUBTRACT : e == NVN_BLEND_EQUATION_REVERSE_SUB ? VK_BLEND_OP_REVERSE_SUBTRACT : e == NVN_BLEND_EQUATION_MIN ? VK_BLEND_OP_MIN : e == NVN_BLEND_EQUATION_MAX ? VK_BLEND_OP_MAX : VK_BLEND_OP_ADD; }
static const VkDynamicState k_dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
    VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_VERTEX_INPUT_EXT};
// A pixel shader may read a location its vertex shader never writes (NVIDIA's hardware then gives it leftovers; the game's 4c8a48999b6c0ddf reads location 11 after
// 4d0323e7d9fa80fe, which writes 0, 1, 4-7 and 10). Vulkan wants every input matched, and this driver refused such pipelines (-13). The vertex shader gets the missing
// locations as unwritten vec4 outputs: the pixel shader reads an undefined value there, as on the Switch
static VkShaderModule vs_variant(shd *vs, uint32_t add) {
    for (int i = 0; i < vs->nvar; i++) if (vs->var[i].add == add) return vs->var[i].m;
    if (!vs->words || vs->nvar == 4) return vs->mod;
    const uint32_t *w = vs->words; size_t nw = vs->nwords;
    // the Output vec4 pointer type and where to put things: after the first OpDecorate, after the first Output variable, the entry point's interface list
    size_t dec_at = 0, var_at = 0, ep_at = 0; uint32_t ptr = 0;
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) break;
        if (op == 15 && !ep_at) ep_at = i;
        if (op == SPV_DECORATE && !dec_at) dec_at = i + len;
        if (op == SPV_VARIABLE && w[i + 3] == 3 && !var_at) {   // an Output variable: its pointer type must be to a vec4 of 32-bit floats
            for (size_t j = 5; j < nw; ) { uint32_t o = w[j] & 0xffff, l = w[j] >> 16; if (!l) break;
                if (o == SPV_TYPE_POINTER && w[j + 1] == w[i + 1]) {
                    uint32_t t = w[j + 3];
                    for (size_t q = 5; q < nw; ) { uint32_t oo = w[q] & 0xffff, ll = w[q] >> 16; if (!ll) break; if (oo == SPV_TYPE_VECTOR && w[q + 1] == t && w[q + 3] == 4) { ptr = w[i + 1]; break; } q += ll; }
                    break; }
                j += l; }
            if (ptr) var_at = i + len;
        }
        i += len;
    }
    if (!ep_at || !dec_at || !var_at || !ptr) { VRED_ONCE("vertex shader %016llx: no place to add the outputs a pixel shader reads", (unsigned long long)vs->key); return vs->mod; }
    int n = __builtin_popcount(add);
    uint32_t *o = malloc((nw + (size_t)n * 9) * 4); size_t no = 0; uint32_t bound = w[3], ids[32]; int k = 0;
    for (int l = 0; l < 32; l++) if (add >> l & 1) ids[k++] = bound++;
    for (size_t i = 0; i < nw; ) {
        uint32_t len = i < 5 ? 1 : w[i] >> 16;
        if (i == ep_at) { memcpy(o + no, w + i, len * 4); o[no] = (len + (uint32_t)n) << 16 | 15; no += len; for (int j = 0; j < n; j++) o[no++] = ids[j]; i += len; continue; }
        if (i == dec_at) { k = 0; for (int l = 0; l < 32; l++) if (add >> l & 1) { o[no++] = 4u << 16 | SPV_DECORATE; o[no++] = ids[k++]; o[no++] = DEC_LOCATION; o[no++] = (uint32_t)l; } }
        if (i == var_at) { for (int j = 0; j < n; j++) { o[no++] = 4u << 16 | SPV_VARIABLE; o[no++] = ptr; o[no++] = ids[j]; o[no++] = 3; } }
        memcpy(o + no, w + i, len * 4); no += len; i += len;
    }
    o[3] = bound;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, no * 4, o};
    VkShaderModule m; VkResult r = vkCreateShaderModule(g_vk.dev, &mi, NULL, &m); free(o);
    if (r) return vs->mod;
    vs->var[vs->nvar].add = add; vs->var[vs->nvar].m = m; vs->nvar++;
    VLOG("vertex shader %016llx: unwritten outputs added at locations %#x for its pixel shader", (unsigned long long)vs->key, add);
    return m;
}
static pent *pipeline_of(const pkey *k) {
    uint32_t h = 2166136261u; const uint8_t *q = (const uint8_t *)k; for (size_t i = 0; i < sizeof *k; i++) h = (h ^ q[i]) * 16777619u;
    pent **b = &g_pipe[h & (NPIPE - 1)];
    for (pent *e = *b; e; e = e->next) if (!memcmp(&e->k, k, sizeof *k)) return e->p ? e : NULL;
    pent *e = calloc(1, sizeof *e); e->k = *k; e->next = *b; *b = e;
    pcache_open();
    e->pl = layout_of(k->vs, k->ps, k->gs); if (!e->pl) { g_n_pipe_fail++; return NULL; }
    VkPipelineShaderStageCreateInfo st[3]; uint32_t ns = 0;
    VkShaderModule vmod = k->vs->mod;
    if (k->ps && !k->gs && (k->ps->in_mask & ~k->vs->out_mask)) vmod = vs_variant(k->vs, k->ps->in_mask & ~k->vs->out_mask);
    st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vmod, "main", NULL};
    if (k->gs) st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_GEOMETRY_BIT, k->gs->mod, "main", NULL};
    if (k->ps) st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, k->ps->mod, "main", NULL};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, VK_FALSE};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, VK_TRUE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip_m1 ? &dcc : NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, VK_FALSE,
        k->poly_mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : k->poly_mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, VK_FALSE, 0, NULL, k->a2c, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState ba[8];
    for (uint32_t i = 0; i < k->ncol; i++) {
        const blend_st *bs = &k->blend[i];
        ba[i] = (VkPipelineColorBlendAttachmentState){(k->blend_en >> i & 1) ? VK_TRUE : VK_FALSE, bf_vk(bs->src_rgb), bf_vk(bs->dst_rgb), beq_vk(bs->eq_rgb), bf_vk(bs->src_a), bf_vk(bs->dst_a), beq_vk(bs->eq_a),
                                                      (VkColorComponentFlags)(k->chmask >> (4 * i) & 15)};
        if (bs->eq_rgb == NVN_BLEND_EQUATION_MIN || bs->eq_rgb == NVN_BLEND_EQUATION_MAX) { ba[i].srcColorBlendFactor = ba[i].dstColorBlendFactor = VK_BLEND_FACTOR_ONE; }
        if (bs->eq_a == NVN_BLEND_EQUATION_MIN || bs->eq_a == NVN_BLEND_EQUATION_MAX) { ba[i].srcAlphaBlendFactor = ba[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; }
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, k->logic_op != NVN_LOGIC_OP_COPY, (VkLogicOp)k->logic_op, k->ncol, ba, {0, 0, 0, 0}};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof k_dyn / sizeof *k_dyn - (g_vk.vi_dyn ? 0 : 1), k_dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->ncol; i++) cf[i] = (VkFormat)k->col_fmt[i];
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, cf, k->d_on ? (VkFormat)k->ds_fmt : VK_FORMAT_UNDEFINED, k->s_on ? (VkFormat)k->ds_fmt : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, ns, st, &vi, &ia, NULL, &vp, &rs, &ms, &ds, &cb, &dy, e->pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_pcache, 1, &gi, NULL, &e->p);
    if (r) { e->p = VK_NULL_HANDLE; g_n_pipe_fail++; VLOG("[RED] pipeline for shaders %016llx / %016llx failed (%d)", (unsigned long long)k->vs->key, k->ps ? (unsigned long long)k->ps->key : 0ull, r); return NULL; }
    g_n_pipelines++;
    return e;
}

// ---------- Vertex formats ----------
static VkFormat vfmt_vk(uint32_t f) {
    static const uint16_t rgb[12] = {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM,
                                     VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    static const uint16_t i2f[24] = {VK_FORMAT_R8_USCALED, VK_FORMAT_R8_SSCALED, VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SSCALED, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT,
                                     VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
                                     VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
                                     VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R16G16B16A16_USCALED, VK_FORMAT_R16G16B16A16_SSCALED, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT};
    if (f >= NVN_FORMAT_RGB8 && f <= NVN_FORMAT_RGB32I) return (VkFormat)rgb[f - NVN_FORMAT_RGB8];
    if (f >= NVN_FORMAT_R8_UI2F && f <= NVN_FORMAT_RGBA32_I2F) return (VkFormat)i2f[f - NVN_FORMAT_R8_UI2F];
    uint32_t v = 0; nvn_fmt_to_vk((int)f, &v, NULL); return (VkFormat)v;
}
static VkPrimitiveTopology topo_vk(int prim, int *quads) {
    *quads = 0;
    switch (prim) {
        case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case NVN_DRAW_PRIMITIVE_LINE_LOOP: VRED_ONCE("line loops drawn as line strips"); return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; case NVN_DRAW_PRIMITIVE_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        case NVN_DRAW_PRIMITIVE_QUADS: *quads = 1; return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        default: return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}

// ---------- DM_VK_TRACE=<frame>: every draw of the frame after that many presents, with its shaders, target and textures ----------
static int g_trace_lines;
static vk_tex *g_trace_tg[64]; static int g_trace_ntg; static uint64_t g_trace_pass_first[40]; static int g_trace_npass;   // the multi-target passes' targets in the traced frame
static int tracing(void);
// DM_VK_SNAP_PS=<ps key>[,…]: while tracing, a copy of target 0 right after the last draw with that pixel shader (vkx_clone), dumped with the targets
static vk_tex *g_snap[32]; static uint64_t g_snap_ser[32]; static int g_nsnap;   // serial 9KKKKT: K the key's low 4 digits, T the target index
static void snap_after(sub_t *s, shd *ps) {
    static const char *want; static int init; if (!init) { init = 1; want = getenv("DM_VK_SNAP_PS"); if (want) VLOG("snapshots after pixel shaders %s while tracing", want); }
    if (!want || !ps || !tracing()) return;
    char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)ps->key); if (!strstr(want, k)) return;
    for (int t = 0; t < g_rt.rt.n && t < 8; t++) {
        vk_tex *x = g_rt.rt.color[t] ? tex_of(g_rt.rt.color[t], NULL) : NULL; if (!x) continue;
        uint64_t ser = 900000 + ps->key % 10000 * 10 + (uint64_t)t;
        int i = 0; while (i < g_nsnap && g_snap_ser[i] != ser) i++;
        if (i == g_nsnap) { if (g_nsnap == 32) return; g_nsnap++; }
        g_snap_ser[i] = ser; g_snap[i] = vkx_clone(s, x, 1);   // the clone ends the render pass; the next draw opens it again. Kept (a diagnostic leak, one traced frame): the dump runs after the submission's fence
        VLOG("snapshot of target %d (texture %llu) after ps %016llx: %s", t, (unsigned long long)x->serial, (unsigned long long)ps->key, g_snap[i] ? "taken" : "the copy could not be made");
    }
}
int vkd_trace_targets(vk_tex **out, int max) { int n = g_trace_ntg < max ? g_trace_ntg : max; memcpy(out, g_trace_tg, sizeof *out * (size_t)n);
    for (int i = 0; i < g_nsnap && n < max; i++) if (g_snap[i]) { g_snap[i]->serial = g_snap_ser[i]; out[n++] = g_snap[i]; }
    g_trace_ntg = 0; g_trace_npass = 0; g_nsnap = 0; return n; }
static int g_cs_trace;   // set while a dispatch DM_VK_CS_TRACE names gathers its bindings
// DM_VK_SYNC with DM_VK_CS_TRACE: the storage buffers that dispatch was given, summarized once the GPU has run it (they are pool memory the CPU sees)
static struct { int on; uint64_t addr[16], size[16]; } g_cs_dump;
void vkd_after_sync(void) {
    if (!g_cs_dump.on) return;
    g_cs_dump.on = 0;
    for (int r = 0; r < 16; r++) {
        if (!g_cs_dump.addr[r] || !g_cs_dump.size[r]) continue;
        const uint32_t *w = (const uint32_t *)(uintptr_t)g_cs_dump.addr[r]; uint64_t n = g_cs_dump.size[r] / 4, nz = 0, big = 0; uint32_t mx = 0;
        for (uint64_t i = 0; i < n; i++) { uint32_t v = w[i]; if (v) nz++; if (v > mx) mx = v; if (v > 100000) big++; }
        VLOG("trace after the GPU ran it: storage slot %d at %#llx, %llu words: %llu non-zero, %llu above 100000, largest %u; first words %u %u %u %u %u %u %u %u", r,
             (unsigned long long)g_cs_dump.addr[r], (unsigned long long)n, (unsigned long long)nz, (unsigned long long)big, mx, n > 0 ? w[0] : 0, n > 1 ? w[1] : 0, n > 2 ? w[2] : 0, n > 3 ? w[3] : 0,
             n > 4 ? w[4] : 0, n > 5 ? w[5] : 0, n > 6 ? w[6] : 0, n > 7 ? w[7] : 0);
    }
}
static int tracing(void) {
    if (g_cs_trace) return 1;
    static long long at = -2; if (at == -2) { const char *e = getenv("DM_VK_TRACE"); at = e ? atoll(e) : -1; }
    return at >= 0 && g_presents == (uint64_t)at && g_trace_lines < 60000;
}
#define TRACE(...) do { if (tracing()) { g_trace_lines++; VLOG("trace " __VA_ARGS__); } } while (0)

// ---------- Descriptors for one stage ----------
static uint32_t g_draw_binst, g_dispatch_dim[3];
static uint64_t g_n_fbcopy, g_n_pool_desc;
static int fb_copy_on(void) { static int v = -1; if (v < 0) v = getenv("DM_VK_NO_FBCOPY") == NULL; return v; }
static int stage_set(sub_t *s, shd *sh, VkDescriptorSet *out, int *missing_tex) {
    if (!sh->nb) { *out = VK_NULL_HANDLE; return 1; }
    VkDescriptorSet ds = set_alloc(s, sh->dsl); if (!ds) return 0;
    int stage = sh->stage;
    enum { NW = 64, NI = 256 };
    VkWriteDescriptorSet wr[NW]; VkDescriptorBufferInfo bi[NW]; VkDescriptorImageInfo ii[NI]; int nw = 0, nii = 0; int pools = 0;
    for (int i = 0; i < sh->nb && nw < NW; i++) {
        const sbind *b = &sh->b[i];
        VkWriteDescriptorSet *w = &wr[nw]; *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, b->vb, 0, b->count > 1024 ? 1 : b->count, b->dt};
        if (b->dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b->dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            VkDescriptorBufferInfo *d = &bi[nw]; *d = (VkDescriptorBufferInfo){g_zero, 0, 65536}; w->descriptorCount = 1;
            uint64_t range_max = b->dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? g_vk.lim.maxUniformBufferRange : 0x7fffffff;
            if (b->kind == BK_SUPPORT) *d = (VkDescriptorBufferInfo){g_support, 0, 65536};
            else if (b->kind == BK_CBUF && b->slot == 1) { if (sh->c1buf) *d = (VkDescriptorBufferInfo){sh->c1buf, 0, VK_WHOLE_SIZE}; }
            else if (b->kind == BK_CBUF && (b->slot == 0 || b->slot == 2)) {
                VkBuffer ab; uint64_t ao; uint8_t *m = arena_alloc(s, DMSH_DRV_SIZE, &ab, &ao);
                if (m) {
                    memset(m, 0, DMSH_DRV_SIZE);
                    if (b->slot == 0) {
                        int base = c0_ssbo_base(stage);
                        if (base >= 0) for (int r = 0; r < DMSH_C0_SSBO_SLOTS; r++) { uint64_t a = S.ssbo[stage][r].addr; uint32_t n = (uint32_t)S.ssbo[stage][r].size; memcpy(m + base + DMSH_C0_ROW * r, &a, 8); memcpy(m + base + DMSH_C0_ROW * r + 8, &n, 4); }
                        if (stage != 5) { const float max_point = 2047.0f; memcpy(m + DMSH_C0_POINT_SIZE, &max_point, 4); memcpy(m + DMSH_C0_BASE_INSTANCE, &g_draw_binst, 4); }
                        else {   // compute: the dispatch's group counts (and NVN2's copy of them below its own layout), the uniform slots' addresses (nvn_mtl.m encode_args)
                            memcpy(m + DMSH_C0_DISPATCH, g_dispatch_dim, 12); memcpy(m + DMSH_C0_DISPATCH_NVN2, g_dispatch_dim, 12);
                            for (int r = 0; r < DMSH_C0_CS_UBO_SLOTS; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(m + DMSH_C0_CS_UBO + DMSH_C0_ROW * r, &a, 8); memcpy(m + DMSH_C0_CS_UBO + DMSH_C0_ROW * r + 8, &n, 4); }
                        }
                    } else for (uint32_t off = 0; off + 4 <= DMSH_DRV_SIZE; off += 4) { uint32_t v = c2_word(stage, off); if (v) memcpy(m + off, &v, 4); }
                    *d = (VkDescriptorBufferInfo){ab, ao, DMSH_DRV_SIZE};
                }
            } else if (b->kind == BK_CBUF && b->slot == DMSH_PUSH_BANK) {
                VkBuffer ab; uint64_t ao; uint8_t *m = arena_alloc(s, DMSH_PUSH_SIZE, &ab, &ao);
                if (m) { memcpy(m, S.push[stage == 5], DMSH_PUSH_SIZE); *d = (VkDescriptorBufferInfo){ab, ao, DMSH_PUSH_SIZE}; }
            } else if ((b->kind == BK_CBUF && b->slot >= 3 && b->slot <= 16) || b->kind == BK_SBUF) {
                const p_bind_buf *u = b->kind == BK_CBUF ? &S.ubo[stage][b->slot - 3] : b->slot >= 100 && b->slot < 116 ? &S.ubo[stage][b->slot - 100] : b->slot >= 0 && b->slot < 16 ? &S.ssbo[stage][b->slot] : NULL;
                VkBuffer pb; uint64_t po;
                if (u && u->addr && u->size) {
                    uint64_t addr = u->addr, size = u->size;
                    if (b->kind == BK_SBUF) { uint64_t al = addr & ~15ull; size += addr - al; addr = al; }   // the shader indexes from the 16-byte aligned base (Ryujinx GlobalToStorage, host alignment 16)
                    uint64_t al_req = b->kind == BK_SBUF ? g_vk.lim.minStorageBufferOffsetAlignment : g_vk.lim.minUniformBufferOffsetAlignment;
                    if (vk_pool_find(addr, size, &pb, &po)) {
                        if (po % al_req) VRED_ONCE("%s buffer at %#llx is not %llu-byte aligned in its pool: bound at the lower boundary (reads shifted)", b->kind == BK_SBUF ? "a storage" : "a uniform", (unsigned long long)addr, (unsigned long long)al_req);
                        uint64_t aligned = po - po % al_req; size += po - aligned;
                        *d = (VkDescriptorBufferInfo){pb, aligned, size < range_max ? size : range_max};
                    }
                }
            }
            w->pBufferInfo = d;
            TRACE("  st%d b%u %s kind %d slot %d: buffer %p offset %llu range %llu%s", stage, b->vb, b->dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? "storage" : "uniform", b->kind, b->slot,
                  (void *)d->buffer, (unsigned long long)d->offset, (unsigned long long)d->range, d->buffer == g_zero ? " (zero buffer)" : d->buffer == g_support ? " (support)" : "");
        } else if (b->dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b->dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || b->dt == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            uint32_t n = w->descriptorCount; if (nii + (int)n > NI) { VRED_ONCE("more than %d texture descriptors in one stage", NI); continue; }
            w->pImageInfo = &ii[nii];
            int storage = b->dt == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            for (uint32_t k = 0; k < n; k++) {
                VkDescriptorImageInfo *d = &ii[nii++]; *d = (VkDescriptorImageInfo){VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
                int tid = 0, sid = 0; treg_t r; samp_par sp;
                if (b->kind == BK_TEX || b->kind == BK_IMG) binding_ids(stage, b, k, &tid, &sid);
                if (b->dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) d->sampler = sampler_of(sreg_get(sid, &sp) ? &sp : NULL);
                if (tid && treg_get(tid, &r)) {
                    vk_tex *x = tex_for_read(&r.t, s);
                    if (x && !storage && g_rt.has_rt && fb_copy_on()) {   // one of the pass's own images: read a copy (vkx_clone)
                        int att = g_rt.rt.depth && tex_of(g_rt.rt.depth, NULL) == x;
                        for (int a = 0; a < g_rt.rt.n && a < 8 && !att; a++) att = g_rt.rt.color[a] && tex_of(g_rt.rt.color[a], NULL) == x;
                        if (att) { vk_tex *c = vkx_clone(s, x, 0); if (c) { x = c; g_n_fbcopy++; vkx_sublog(s, " F%llu", (unsigned long long)c->serial); } }
                    }
                    if (x && (!storage || x->d.samples <= 1)) d->imageView = sample_view(x, &r, b, storage);
                }
                if (!d->imageView) (*missing_tex)++;
                TRACE("  st%d b%u %s dim %d arr %d kind %d cbs %d handle %#x [%u]: tid %d sid %d reg %d ser %llu %ux%u fmt %#x tgt %d mem %p -> view %s", stage, b->vb, storage ? "storage image" : "texture", b->dim, b->arrayed, b->kind, b->cbs, (unsigned)b->handle, k, tid, sid,
                      tid ? treg_get(tid, &r) : 0, tid && treg_get(tid, &r) ? (unsigned long long)r.t.serial : 0ull, tid && treg_get(tid, &r) ? r.t.d.width : 0, tid && treg_get(tid, &r) ? r.t.d.height : 0,
                      tid && treg_get(tid, &r) ? r.t.d.format : 0, tid && treg_get(tid, &r) ? r.t.d.target : -1, tid && treg_get(tid, &r) ? (void *)r.t.mem : NULL, d->imageView ? "ok" : "none");
            }
            if (!g_vk.null_desc) for (uint32_t k = 0; k < n; k++) if (!ii[nii - n + k].imageView) { VRED_ONCE("a texture binding has no texture and the driver has no null descriptors: draw skipped"); return 0; }
        } else if (b->kind == BK_POOL_TEX || b->kind == BK_POOL_SAMP) { pools = 1; continue; }   // written below, element by element
        else continue;   // other separate samplers and texel buffers: left unwritten
        nw++;
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)nw, wr, 0, NULL);
    if (pools) {   // the pool arrays: one descriptor per registered ID of the bound pool (the rest stay empty)
        uint32_t cap = 0, np = 0; VkWriteDescriptorSet *pw = NULL; VkDescriptorImageInfo *pi = NULL;
        for (int i = 0; i < sh->nb; i++) {
            const sbind *b = &sh->b[i]; if (b->kind != BK_POOL_TEX && b->kind != BK_POOL_SAMP) continue;
            for (uint32_t e = 0; e < b->count; e++) {
                VkDescriptorImageInfo d = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
                if (b->kind == BK_POOL_SAMP) { samp_par sp; if (!sreg_get(S.samp_base + (int)e, &sp)) continue; d.sampler = sampler_of(&sp); }
                else { treg_t r; if (!treg_get(S.tex_base + (int)e, &r)) continue; vk_tex *x = tex_for_read(&r.t, s); if (!x) continue; d.imageView = sample_view(x, &r, b, 0); }
                if (!d.sampler && !d.imageView) continue;
                if (np == cap) { cap = cap ? cap * 2 : 1024; pw = realloc(pw, cap * sizeof *pw); pi = realloc(pi, cap * sizeof *pi); }
                pi[np] = d; pw[np] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, b->vb, e, 1, b->dt}; np++;
            }
        }
        if (tracing()) for (int i = 0; i < sh->nb; i++) { const sbind *b = &sh->b[i]; if (b->kind != BK_POOL_TEX && b->kind != BK_POOL_SAMP) continue;
            uint32_t c = 0, lo = ~0u, hi = 0; for (uint32_t k = 0; k < np; k++) if (pw[k].dstBinding == b->vb) { c++; if (pw[k].dstArrayElement < lo) lo = pw[k].dstArrayElement; if (pw[k].dstArrayElement > hi) hi = pw[k].dstArrayElement; }
            TRACE("  st%d b%u %s pool (dim %d arr %d): %u elements written, IDs %u-%u, bases tex %d samp %d", stage, b->vb, b->kind == BK_POOL_TEX ? "texture" : "sampler", b->dim, b->arrayed, c, c ? lo : 0, hi, S.tex_base, S.samp_base); }
        for (uint32_t k = 0; k < np; k++) pw[k].pImageInfo = &pi[k];
        if (np) vkUpdateDescriptorSets(g_vk.dev, np, pw, 0, NULL);
        g_n_pool_desc += np;
        free(pw); free(pi);
    }
    *out = ds;
    return 1;
}

// ---------- The draw ----------
static PFN_vkCmdSetVertexInputEXT p_set_vi;
static uint64_t g_n_missing_tex;
static VkFormat dummy_attr_fmt(uint8_t kind) { return kind == 1 ? VK_FORMAT_R32G32B32A32_SINT : kind == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT; }
void vkd_draw(sub_t *s, uint32_t op, const void *rec, uint32_t size) {
    // a direct draw's record stops before ind (hle_nvn.c rec_draw): the rest reads as zero
    union { p_draw d; p_draw_idx i; } full; memset(&full, 0, sizeof full); memcpy(&full, rec, size < sizeof full ? size : sizeof full); const void *a = &full;
    state_init(); shared_buffers();
    if (!p_set_vi && g_vk.vi_dyn) p_set_vi = (PFN_vkCmdSetVertexInputEXT)vkGetDeviceProcAddr(g_vk.dev, "vkCmdSetVertexInputEXT");
    if (!p_set_vi) { VRED_ONCE("no vertex input dynamic state: nothing is drawn"); return; }
    shd *vs = S.sh[0], *ps = S.sh[1], *gs = S.sh[2];
    if (tracing()) { vk_tex *t0 = g_rt.has_rt && g_rt.rt.n > 0 && g_rt.rt.color[0] ? tex_of(g_rt.rt.color[0], NULL) : NULL;
        TRACE("draw op %u: vs %016llx ps %016llx; target 0 ser %llu %ux%u fmt %d mem %p (%d targets, depth %d); viewport %.0f %.0f %.0f %.0f scissor %d %d %d %d", op, vs ? (unsigned long long)vs->key : 0ull,
              ps ? (unsigned long long)ps->key : 0ull, t0 ? (unsigned long long)t0->serial : 0ull, t0 ? t0->w : 0, t0 ? t0->h : 0, t0 ? t0->fmt : 0, g_rt.has_rt && g_rt.rt.n > 0 && g_rt.rt.color[0] ? (void *)g_rt.rt.color[0]->mem : NULL, g_rt.rt.n, g_rt.rt.depth != NULL,
              S.vp[0], S.vp[1], S.vp[2], S.vp[3], S.sc.x, S.sc.y, S.sc.w, S.sc.h); }
    if (!vs) { g_n_skip_noprog++; return; }
    if (!shd_load(vs) || (ps && !shd_load(ps)) || (gs && !shd_load(gs))) { g_n_skip_bad++; TRACE("  skipped: shader unusable"); return; }
    const p_draw *dr = op == OP_DRAW ? &full.d : NULL; const p_draw_idx *di = op == OP_DRAW_IDX ? &full.i : NULL; (void)a;
    int quads; VkPrimitiveTopology topo = topo_vk(dr ? dr->prim : di->prim, &quads);
    if (topo == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) { VRED_ONCE("primitive %d not drawn", dr ? dr->prim : di->prim); g_n_skip_other++; return; }
    if ((dr ? dr->mdi_cnt : di->mdi_cnt) || (quads && (dr ? dr->ind : di->ind))) { VRED_ONCE("indirect-count draws and indirect quads not drawn yet"); g_n_skip_other++; return; }
    if (di && di->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE && !g_vk.idx_u8 && !quads) { VRED_ONCE("8-bit indices without the driver's support: not drawn"); g_n_skip_other++; return; }
    g_draw_binst = (uint32_t)(dr ? dr->base_instance : di->base_instance);
    // resources first: uploads end the render pass, which must be open for the draw
    VkDescriptorSet sets[3] = {0}; int missing = 0;
    if (!stage_set(s, vs, &sets[0], &missing) || (ps && !stage_set(s, ps, &sets[1], &missing)) || (gs && !stage_set(s, gs, &sets[2], &missing))) { g_n_skip_other++; return; }
    if (missing) g_n_missing_tex += (uint64_t)missing;
    g_rt.want = (VkExtent2D){S.has_vp && S.vp[2] >= 1 ? (uint32_t)(S.vp[0] + S.vp[2]) : 0, S.has_vp && S.vp[3] >= 1 ? (uint32_t)(S.vp[1] + S.vp[3]) : 0};
    if (!g_rt.want.height) g_rt.want.width = 0;
    if (g_rt.active && g_rt.ext.width == 0) end_rendering();
    if (!begin_rendering(s)) { g_n_skip_rt++; TRACE("  skipped: no render target"); return; }
    if (tracing() && g_rt.rt.n >= 1) {   // each distinct target set (G-buffer and the like): its targets, dumped at the traced present, and named in the trace
        char names[200]; int k = 0; names[0] = 0;
        for (int i = 0; i < g_rt.rt.n && i < 8; i++) { vk_tex *x = g_rt.rt.color[i] ? tex_of(g_rt.rt.color[i], NULL) : NULL; k += snprintf(names + k, sizeof names - (size_t)k, " %llu:%d", x ? (unsigned long long)x->serial : 0ull, x ? x->fmt : 0); }
        vk_tex *x0 = g_rt.rt.color[0] ? tex_of(g_rt.rt.color[0], NULL) : NULL; uint64_t first = x0 ? x0->serial : 0; int seen = 0;
        for (int i = 0; i < g_trace_npass; i++) if (g_trace_pass_first[i] == first) seen = 1;
        if (!seen && g_trace_npass < 40) {
            g_trace_pass_first[g_trace_npass++] = first;
            TRACE("  pass %d targets:%s", g_trace_npass, names);
            for (int i = 0; i < g_rt.rt.n && i < 8 && g_trace_ntg < 63; i++) { vk_tex *x = g_rt.rt.color[i] ? tex_of(g_rt.rt.color[i], NULL) : NULL; if (x) g_trace_tg[g_trace_ntg++] = x; }
            vk_tex *dx0 = g_rt.rt.depth ? tex_of(g_rt.rt.depth, NULL) : NULL; if (dx0 && g_trace_ntg < 64) g_trace_tg[g_trace_ntg++] = dx0;
        } }
    // the pipeline for this state and these targets
    pkey k; memset(&k, 0, sizeof k);
    k.vs = vs; k.ps = ps; k.gs = gs; k.topo = (uint32_t)topo; k.poly_mode = S.poly.mode; k.depth_clamp = S.depth_clamp != 0; k.clip_m1 = g_vk.clip_ctl && !atomic_load(&g_depth_zero);
    k.logic_op = S.color.logic_op; k.a2c = S.ms.a2c; k.chmask = S.chmask.mask; k.blend_en = S.color.blend_en;
    int n = g_rt.rt.n > 8 ? 8 : g_rt.rt.n; k.ncol = (uint32_t)(n < 0 ? 0 : n);
    { int any = 0; for (int i = 0; i < n; i++) { vk_tex *x = g_rt.rt.color[i] ? tex_of(g_rt.rt.color[i], NULL) : NULL; if (x && (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) any = 1; }
      vk_tex *d0 = g_rt.rt.depth ? tex_of(g_rt.rt.depth, NULL) : NULL; if (!any && !d0) n = 0, k.ncol = 0; }   // the pass began with no attachments (begin_rendering) k.samples = VK_SAMPLE_COUNT_1_BIT;
    for (int i = 0; i < n; i++) {
        vk_tex *x = g_rt.rt.color[i] ? tex_of(g_rt.rt.color[i], NULL) : NULL;
        k.col_fmt[i] = x && (x->aspect & VK_IMAGE_ASPECT_COLOR_BIT) ? (uint32_t)x->fmt : VK_FORMAT_UNDEFINED;
        if (x && x->d.samples > 1) k.samples = x->d.samples;
        k.blend[i] = S.blend[i]; k.blend[i].target = 0;
    }
    vk_tex *dx = g_rt.rt.depth ? tex_of(g_rt.rt.depth, NULL) : NULL;
    if (dx) { k.ds_fmt = (uint32_t)dx->fmt; k.d_on = !!(dx->aspect & VK_IMAGE_ASPECT_DEPTH_BIT); k.s_on = !!(dx->aspect & VK_IMAGE_ASPECT_STENCIL_BIT); if (dx->d.samples > 1) k.samples = dx->d.samples; }
    pent *pe = pipeline_of(&k); if (!pe) { g_n_skip_other++; return; }
    VkCommandBuffer cb = s->cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pe->p);
    for (uint32_t i = 0; i < 3; i++) if (sets[i]) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pe->pl, i, 1, &sets[i], 0, NULL);
    // dynamic state
    float vx = S.has_vp ? S.vp[0] : 0, vy = S.has_vp ? S.vp[1] : 0, vw = S.has_vp ? S.vp[2] : (float)g_rt.ext.width, vh = S.has_vp ? S.vp[3] : (float)g_rt.ext.height;
    VkViewport v = {vx, vy, vw, vh, S.dr[0], S.dr[1]};
    if (!atomic_load(&g_origin_ll)) { v.y = vy + vh; v.height = -vh; }   // see the file header
    if (v.minDepth < 0) v.minDepth = 0; if (v.minDepth > 1) v.minDepth = 1; if (v.maxDepth < 0) v.maxDepth = 0; if (v.maxDepth > 1) v.maxDepth = 1;
    vkCmdSetViewport(cb, 0, 1, &v);
    int32_t sx = S.has_sc ? S.sc.x : 0, sy = S.has_sc ? S.sc.y : 0; int64_t sw = S.has_sc ? S.sc.w : (int64_t)g_rt.ext.width, sh_ = S.has_sc ? S.sc.h : (int64_t)g_rt.ext.height;
    if (sx < 0) { sw += sx; sx = 0; } if (sy < 0) { sh_ += sy; sy = 0; } if (sw < 0) sw = 0; if (sh_ < 0) sh_ = 0;
    if (sx + sw > (int64_t)g_rt.ext.width) sw = (int64_t)g_rt.ext.width - sx; if (sy + sh_ > (int64_t)g_rt.ext.height) sh_ = (int64_t)g_rt.ext.height - sy; if (sw < 0) sw = 0; if (sh_ < 0) sh_ = 0;
    VkRect2D sc = {{sx, sy}, {(uint32_t)sw, (uint32_t)sh_}}; vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdSetLineWidth(cb, 1.0f);
    vkCmdSetDepthBiasEnable(cb, S.poly.offset_en != 0); vkCmdSetDepthBias(cb, S.po.units, S.po.clamp, S.po.factor);
    vkCmdSetBlendConstants(cb, S.blend_color);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_vmask[0]); vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_vmask[1]);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_wmask[0]); vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, S.st_wmask[1]);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, S.st_ref[0]); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, S.st_ref[1]);
    vkCmdSetCullMode(cb, (VkCullModeFlags)S.poly.cull);
    int front_ccw = (S.poly.front == NVN_FRONT_FACE_CCW) != g_queue_origin_ll;   // as nvn_mtl.m's winding
    vkCmdSetFrontFace(cb, front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    vkCmdSetDepthTestEnable(cb, S.ds.dt && k.d_on); vkCmdSetDepthWriteEnable(cb, S.ds.dw && k.d_on);
    VkCompareOp dop = S.ds.df >= 1 && S.ds.df <= 8 ? (VkCompareOp)(S.ds.df - 1) : VK_COMPARE_OP_ALWAYS;
    { static int dbg = -1; if (dbg < 0) dbg = getenv("DM_VK_EQ_ALWAYS") != NULL;   // diagnostic: the equal tests always pass (does the pass rasterize at all?)
      if (dbg && dop == VK_COMPARE_OP_EQUAL) dop = VK_COMPARE_OP_ALWAYS; }
    vkCmdSetDepthCompareOp(cb, dop); vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
    vkCmdSetStencilTestEnable(cb, S.ds.st && k.s_on);
#define SOP(o) ((o) >= 1 && (o) <= 8 ? (VkStencilOp)((o) - 1) : VK_STENCIL_OP_KEEP)
#define SFN(f) ((f) >= 1 && (f) <= 8 ? (VkCompareOp)((f) - 1) : VK_COMPARE_OP_ALWAYS)
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, SOP(S.ds.fail_f), SOP(S.ds.zpass_f), SOP(S.ds.zfail_f), SFN(S.ds.sf_f));
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, SOP(S.ds.fail_b), SOP(S.ds.zpass_b), SOP(S.ds.zfail_b), SFN(S.ds.sf_b));
#undef SOP
#undef SFN
    vkCmdSetPrimitiveRestartEnable(cb, di && S.restart.on && topo != VK_PRIMITIVE_TOPOLOGY_POINT_LIST && topo != VK_PRIMITIVE_TOPOLOGY_LINE_LIST && topo != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetRasterizerDiscardEnable(cb, S.rast_discard != 0);
    // vertex input: every location the vertex shader reads, from its NVN attribute, or zeros
    VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32]; uint32_t nvb = 0, nva = 0, used = 0;
    VkBuffer bufs[17]; VkDeviceSize offs[17]; int dummy = -1;
    for (uint32_t l = 0; l < 32; l++) {
        if (!(vs->in_mask >> l & 1)) continue;
        va_st at; memset(&at, 0, sizeof at); if ((int)l < S.nattr) memcpy(&at, &S.attr[l], 4);
        VkFormat f = at.fmt ? vfmt_vk(at.fmt) : VK_FORMAT_UNDEFINED; uint32_t st = at.stream;
        VkBuffer pb; uint64_t po;
        if (f && st < 16 && S.vbuf[st].address && vk_pool_find(S.vbuf[st].address, S.vbuf[st].size ? S.vbuf[st].size : 1, &pb, &po)) {
            if (!(used >> st & 1)) {
                used |= 1u << st; const vs_st *ss = &S.stream[st];
                vb[nvb] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, st, ss->stride, ss->divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, 1};
                if (ss->divisor > 1) VRED_ONCE("vertex divisor %u drawn as 1", ss->divisor);
                bufs[nvb] = pb; offs[nvb] = po; nvb++;
            }
            va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, l, st, f, at.offset};
        } else {
            if (dummy < 0) { dummy = (int)nvb; vb[nvb] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, 16, 0, VK_VERTEX_INPUT_RATE_VERTEX, 1}; bufs[nvb] = g_zero; offs[nvb] = 0; nvb++; }
            va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, l, 16, dummy_attr_fmt(vs->in_kind[l]), 0};
        }
    }
    p_set_vi(cb, nvb, vb, nva, va);
    for (uint32_t i = 0; i < nvb; i++) vkCmdBindVertexBuffers(cb, vb[i].binding, 1, &bufs[i], &offs[i]);
    uint32_t inst = (uint32_t)((dr ? dr->instances : di->instances) ? (dr ? dr->instances : di->instances) : 1);
    if (dr && !quads) {
        if (dr->ind) { VkBuffer ib; uint64_t io; if (!vk_pool_find(dr->ind, 16, &ib, &io)) { g_n_skip_other++; return; } vkCmdDrawIndirect(cb, ib, io, 1, 16); }
        else vkCmdDraw(cb, (uint32_t)dr->count, inst, (uint32_t)dr->first, (uint32_t)dr->base_instance);
    } else if (dr) {   // quads: as two triangles each, 0 1 2 / 0 2 3
        uint32_t nq = (uint32_t)dr->count / 4; if (!nq) return;
        VkBuffer ab; uint64_t ao; uint32_t *ix = (uint32_t *)arena_alloc(s, (uint64_t)nq * 24, &ab, &ao); if (!ix) return;
        for (uint32_t q = 0; q < nq; q++) { uint32_t b0 = (uint32_t)dr->first + 4 * q; uint32_t t[6] = {b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3}; memcpy(ix + 6 * q, t, 24); }
        vkCmdBindIndexBuffer(cb, ab, ao, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, nq * 6, inst, 0, 0, (uint32_t)dr->base_instance);
    } else {
        int isz = di->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 : di->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4;
        if (quads) {
            uint32_t nq = (uint32_t)di->count / 4; if (!nq) return;
            VkBuffer ab; uint64_t ao; uint32_t *ix = (uint32_t *)arena_alloc(s, (uint64_t)nq * 24, &ab, &ao); if (!ix) return;
            const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf;
            for (uint32_t q = 0; q < nq; q++) { uint32_t c[4]; for (int j = 0; j < 4; j++) { uint32_t i = 4 * q + (uint32_t)j; c[j] = isz == 1 ? src[i] : isz == 2 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i]; }
                uint32_t t[6] = {c[0], c[1], c[2], c[0], c[2], c[3]}; memcpy(ix + 6 * q, t, 24); }
            vkCmdBindIndexBuffer(cb, ab, ao, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, nq * 6, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        } else {
            VkBuffer ib; uint64_t io;
            if (!vk_pool_find(di->ibuf, (uint64_t)(di->count ? di->count : 1) * (uint64_t)isz, &ib, &io)) { VRED_ONCE("an index buffer outside every pool: not drawn"); g_n_skip_other++; return; }
            vkCmdBindIndexBuffer(cb, ib, io, isz == 1 ? VK_INDEX_TYPE_UINT8_EXT : isz == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
            if (di->ind) { VkBuffer xb; uint64_t xo; if (!vk_pool_find(di->ind, 20, &xb, &xo)) { g_n_skip_other++; return; } vkCmdDrawIndexedIndirect(cb, xb, xo, 1, 20); }
            else vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        }
    }
    g_n_draws_done++;
    snap_after(s, ps);
    vkx_sublog(s, " D%llx/%llx", (unsigned long long)(vs->key >> 32), ps ? (unsigned long long)(ps->key >> 32) : 0ull);
    vkx_crumb(s);
    TRACE("  ps outputs %#x, target formats %u %u %u %u %u", ps ? ps->out_frag : 0, k.col_fmt[0], k.col_fmt[1], k.col_fmt[2], k.col_fmt[3], k.col_fmt[4]);
    TRACE("  drawn: %s count %d, inputs %#x, blend %#x mask %#x, cull %d front %d; depth test %d write %d func %d (attached %d fmt %d), range %.3f-%.3f; stencil %d; ncol %u; feedback copies so far %llu",
          dr ? "arrays" : "elements", dr ? dr->count : di->count, vs->in_mask, S.color.blend_en, S.chmask.mask, S.poly.cull, S.poly.front, S.ds.dt, S.ds.dw, S.ds.df, k.d_on, k.ds_fmt, S.dr[0], S.dr[1], S.ds.st, k.ncol,
          (unsigned long long)g_n_fbcopy);
}
// ---------- Compute ----------
typedef struct cpent { struct cpent *next; shd *cs; VkPipeline p; VkPipelineLayout pl; int failed; } cpent;
static cpent *g_cpipe[256];
static uint64_t g_n_dispatch, g_n_dispatch_skip, g_n_cpipes;
static uint64_t g_n_cpipes_all(void) { return g_n_cpipes; }
static cpent *cpipe_of(shd *cs) {
    cpent **b = &g_cpipe[(cs->key ^ cs->key >> 17) & 255];
    for (cpent *e = *b; e; e = e->next) if (e->cs == cs) return e->failed ? NULL : e;
    cpent *e = calloc(1, sizeof *e); e->cs = cs; e->next = *b; *b = e;
    VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &cs->dsl};
    if (vkCreatePipelineLayout(g_vk.dev, &li, NULL, &e->pl)) { e->failed = 1; return NULL; }
    pcache_open();
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs->mod, "main", NULL}, e->pl, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_pcache, 1, &ci, NULL, &e->p);
    if (r) { e->failed = 1; VLOG("[RED] compute pipeline for shader %016llx failed (%d)", (unsigned long long)cs->key, r); return NULL; }
    g_n_cpipes++;
    VLOG("compute shader %016llx first dispatched (%d bindings)", (unsigned long long)cs->key, cs->nb);
    return e;
}
void vkd_dispatch(sub_t *s, const void *rec, uint32_t size) {
    state_init(); shared_buffers();
    p_dispatch d; memset(&d, 0, sizeof d); memcpy(&d, rec, size < sizeof d ? size : sizeof d);
    shd *cs = S.sh[5];
    if (!cs || !shd_load(cs)) { g_n_dispatch_skip++; return; }
    {   // DM_VK_CS_SKIP=<key>[,<key>…]: those compute shaders are not run (to find one that faults the GPU)
        static const char *skip; static int init; if (!init) { init = 1; skip = getenv("DM_VK_CS_SKIP"); }
        if (skip) { char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)cs->key); if (!strcmp(skip, "all") || strstr(skip, k)) { g_n_dispatch_skip++; return; } }
    }
    if (d.ind) { VRED_ONCE("indirect dispatch: shaders reading the group counts from c0 see 0"); memset(g_dispatch_dim, 0, sizeof g_dispatch_dim); }
    else { g_dispatch_dim[0] = (uint32_t)d.x; g_dispatch_dim[1] = (uint32_t)d.y; g_dispatch_dim[2] = (uint32_t)d.z; }
    int missing = 0; VkDescriptorSet set = VK_NULL_HANDLE;
    {   static const char *ct; static int init, n; if (!init) { init = 1; ct = getenv("DM_VK_CS_TRACE"); }
        char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)cs->key);
        g_cs_trace = ct && strstr(ct, k) && n++ < 3;
        if (g_cs_trace) { VLOG("trace dispatch %s %dx%dx%d, %d bindings", k, d.x, d.y, d.z, cs->nb);
            g_cs_dump.on = 1; for (int r = 0; r < 16; r++) { g_cs_dump.addr[r] = S.ssbo[5][r].addr; g_cs_dump.size[r] = S.ssbo[5][r].size; } } }
    if (!stage_set(s, cs, &set, &missing)) { g_cs_trace = 0; g_n_dispatch_skip++; return; }
    g_cs_trace = 0;
    if (missing) g_n_missing_tex += (uint64_t)missing;
    cpent *pe = cpipe_of(cs); if (!pe) { g_n_dispatch_skip++; return; }
    end_rendering();   // no compute inside a render pass
    VkCommandBuffer cb = s->cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pe->p);
    if (set) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pe->pl, 0, 1, &set, 0, NULL);
    if (d.ind) { VkBuffer ib; uint64_t io; if (!vk_pool_find(d.ind, 12, &ib, &io)) { g_n_dispatch_skip++; return; } vkCmdDispatchIndirect(cb, ib, io); }
    else { if (!d.x || !d.y || !d.z) return; vkCmdDispatch(cb, (uint32_t)d.x, (uint32_t)d.y, (uint32_t)d.z); }
    barrier_all(cb);
    g_n_dispatch++;
    vkx_sublog(s, " C%016llx(%dx%dx%d)", (unsigned long long)cs->key, d.x, d.y, d.z);
    vkx_crumb(s);
    TRACE("dispatch cs %016llx %dx%dx%d%s", (unsigned long long)cs->key, d.x, d.y, d.z, d.ind ? " (indirect)" : "");
}

// ---------- Stretching one image over another by drawing ----------
// A full-screen triangle sampling the source (shaders/stretch.vert, stretch.frag, stretch_depth.frag; nvn_vk_blit_spv.h). Used instead of vkCmdBlitImage wherever the
// driver does not offer the blit for the formats (depth formats here) and for the DLSS stand-in: runs with blits on this driver lost the device (a scaled D32 blit
// preceded every hang), so a draw, the path the game's own frames take, stands in
#include "nvn_vk_blit_spv.h"
static VkShaderModule g_st_vs, g_st_fs, g_st_fsd; static VkDescriptorSetLayout g_st_dsl; static VkPipelineLayout g_st_pl;
typedef struct { VkFormat f; int depth; VkPipeline p; } st_pipe;
static st_pipe g_st_pipes[16]; static int g_nst;
static pthread_mutex_t g_st_mu = PTHREAD_MUTEX_INITIALIZER;   // the screen's present thread takes a pipeline here too
static VkPipeline stretch_pipeline_(VkFormat f, int depth);
static VkPipeline stretch_pipeline(VkFormat f, int depth) { pthread_mutex_lock(&g_st_mu); VkPipeline p = stretch_pipeline_(f, depth); pthread_mutex_unlock(&g_st_mu); return p; }
VkPipeline vkd_stretch_get(VkFormat f, VkPipelineLayout *pl, VkDescriptorSetLayout *dsl) { VkPipeline p = stretch_pipeline(f, 0); *pl = g_st_pl; *dsl = g_st_dsl; return p; }
static VkPipeline stretch_pipeline_(VkFormat f, int depth) {
    for (int i = 0; i < g_nst; i++) if (g_st_pipes[i].f == f && g_st_pipes[i].depth == depth) return g_st_pipes[i].p;
    if (!g_st_vs) {
        VkShaderModuleCreateInfo mv = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_stretch_vert, k_stretch_vert}, mf = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_stretch_frag, k_stretch_frag},
                                 md = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_stretch_depth_frag, k_stretch_depth_frag};
        VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &lb};
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        if (vkCreateShaderModule(g_vk.dev, &mv, NULL, &g_st_vs) || vkCreateShaderModule(g_vk.dev, &mf, NULL, &g_st_fs) || vkCreateShaderModule(g_vk.dev, &md, NULL, &g_st_fsd) ||
            vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &g_st_dsl)) return VK_NULL_HANDLE;
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_st_dsl, 1, &pr};
        if (vkCreatePipelineLayout(g_vk.dev, &pli, NULL, &g_st_pl)) return VK_NULL_HANDLE;
    }
    if (g_nst == 16) return VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_st_vs, "main", NULL},
                                            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, depth ? g_st_fsd : g_st_fs, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, depth, depth, VK_COMPARE_OP_ALWAYS};
    VkPipelineColorBlendAttachmentState ba = {VK_FALSE, 0, 0, 0, 0, 0, 0, 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, depth ? 0 : 1, &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    int has_s = depth && (f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT);
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, depth ? 0 : 1, depth ? NULL : &f, depth ? f : VK_FORMAT_UNDEFINED, has_s ? f : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, 2, st, &vi, &ia, NULL, &vp, &rs, &ms, &dss, &cb, &dy, g_st_pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    pcache_open();
    VkPipeline pp; if (vkCreateGraphicsPipelines(g_vk.dev, g_pcache, 1, &gi, NULL, &pp)) return VK_NULL_HANDLE;
    g_st_pipes[g_nst++] = (st_pipe){f, depth, pp};
    return pp;
}
static VkImageView plain_view(sub_t *s, vk_tex *x, VkImageUsageFlags usage, VkImageAspectFlags asp, uint32_t level, uint32_t layer) {   // freed with the submission
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, usage};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, VK_IMAGE_VIEW_TYPE_2D, x->fmt, {0, 0, 0, 0}, {asp, level, 1, layer, 1}};
    VkImageView v; if (vkCreateImageView(g_vk.dev, &vi, NULL, &v)) return VK_NULL_HANDLE;
    GROW(s->views, s->nviews, s->cviews); s->views[s->nviews++] = v;
    return v;
}
// Stretch rectangle ir (x, y, w, h) of in's level / layer over rectangle orr of out's: 1 when drawn. Outside a render pass; leaves none open
int vkd_stretch(sub_t *s, vk_tex *in, uint32_t il, uint32_t iy_layer, const int32_t ir[4], vk_tex *out, uint32_t ol, uint32_t oy_layer, const int32_t orr[4], int linear) {
    state_init(); shared_buffers();
    int depth = (out->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    if (depth != ((in->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0)) { VRED_ONCE("stretch between a depth and a colour image (formats %d / %d)", in->fmt, out->fmt); return 0; }
    if (in->type == VK_IMAGE_TYPE_3D || out->type == VK_IMAGE_TYPE_3D || in->d.samples > 1 || out->d.samples > 1) { VRED_ONCE("stretch of a 3D or multisampled image"); return 0; }
    if (!(in->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !(out->usage & (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))) {
        VRED_ONCE("stretch: format %d cannot be sampled or %d drawn into", in->fmt, out->fmt); return 0; }
    if (depth && (out->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) VRED_ONCE("stretch of a depth-stencil image: depth only (stencil left as it was)");
    VkPipeline pp = stretch_pipeline(out->fmt, depth); if (!pp) { VRED_ONCE("stretch: no pipeline for format %d", out->fmt); return 0; }
    end_rendering();
    uint32_t iwl = level_dim(in->w, il), ihl = level_dim(in->h, il), owl = level_dim(out->w, ol), ohl = level_dim(out->h, ol);
    int32_t ox = orr[0] < 0 ? 0 : orr[0], oy = orr[1] < 0 ? 0 : orr[1], ow = orr[2], oh = orr[3];
    if (ox + ow > (int32_t)owl) ow = (int32_t)owl - ox; if (oy + oh > (int32_t)ohl) oh = (int32_t)ohl - oy;
    if (ow <= 0 || oh <= 0 || ir[2] <= 0 || ir[3] <= 0) return 1;
    VkImageView iv = plain_view(s, in, VK_IMAGE_USAGE_SAMPLED_BIT, depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, il, iy_layer);
    VkImageView ov = plain_view(s, out, depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, depth ? out->aspect : VK_IMAGE_ASPECT_COLOR_BIT, ol, oy_layer);
    VkDescriptorSet ds = set_alloc(s, g_st_dsl); if (!iv || !ov || !ds) { VRED_ONCE("stretch: views or descriptor set failed"); return 0; }
    samp_par sp; memset(&sp, 0, sizeof sp); sp.minf = linear && !depth ? NVN_MIN_FILTER_LINEAR : NVN_MIN_FILTER_NEAREST; sp.magf = linear && !depth ? NVN_MAG_FILTER_LINEAR : NVN_MAG_FILTER_NEAREST;
    sp.ws = sp.wt = sp.wr = NVN_WRAP_MODE_CLAMP_TO_EDGE;
    VkDescriptorImageInfo ii = {sampler_of(&sp), iv, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    VkCommandBuffer cb = s->cb;
    VkRenderingAttachmentInfo at = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, ov, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
    int has_s = depth && (out->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {owl, ohl}}, 1, 0, depth ? 0 : 1, depth ? NULL : &at, depth ? &at : NULL, has_s ? &at : NULL};
    barrier_all(cb);
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_st_pl, 0, 1, &ds, 0, NULL);
    float rect[4] = {(float)ir[0] / (float)iwl, (float)ir[1] / (float)ihl, (float)ir[2] / (float)iwl, (float)ir[3] / (float)ihl};
    vkCmdPushConstants(cb, g_st_pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, rect);
    VkViewport v = {(float)ox, (float)oy, (float)ow, (float)oh, 0, 1}; VkRect2D sc = {{ox, oy}, {(uint32_t)ow, (uint32_t)oh}};
    vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    barrier_all(cb);
    vkx_sublog(s, " S%llu:%d>%llu:%d", (unsigned long long)in->serial, in->fmt, (unsigned long long)out->serial, out->fmt);
    vkx_crumb(s);
    return 1;
}

// ---------- DLSS: the colour input stretched bilinearly over the output (as nvn_mtl.m does where MetalFX has no scaler) ----------
static uint64_t g_n_dlss;
void vkd_dlss(sub_t *s, const void *a) {
    const p_dlss *c = a;
    { static int off = -1; if (off < 0) off = getenv("DM_VK_NO_DLSS") != NULL; if (off) return; }
    if (!c->tex[DLSS_COLOR] || !c->tex[DLSS_OUTPUT]) { VRED_ONCE("DLSS evaluated without %s", c->tex[DLSS_COLOR] ? "an output" : "an input colour"); return; }
    end_rendering();
    vk_tex *in = tex_for_read(c->tex[DLSS_COLOR], s), *out = tex_for_write(c->tex[DLSS_OUTPUT], s);
    if (!in || !out) { VRED_ONCE("DLSS: input or output texture has no image"); return; }
    int32_t ir[4] = {c->base[DLSS_COLOR][0], c->base[DLSS_COLOR][1], c->iw > 0 ? c->iw : (int32_t)in->w, c->ih > 0 ? c->ih : (int32_t)in->h};
    int32_t orr[4] = {c->base[DLSS_OUTPUT][0], c->base[DLSS_OUTPUT][1], c->ow > 0 ? c->ow : (int32_t)out->w, c->oh > 0 ? c->oh : (int32_t)out->h};
    if (!vkd_stretch(s, in, 0, 0, ir, out, 0, 0, orr, 1)) return;
    vkx_sublog(s, " DLSS");
    if (!g_n_dlss++) VLOG("DLSS stand-in: %dx%d of a %ux%u input (format %d) drawn bilinearly over %dx%d of the output (format %d)", ir[2], ir[3], in->w, in->h, in->fmt, orr[2], orr[3], out->fmt);
}

void vkd_stats_line(char *out, size_t n) {
    snprintf(out, n, "%llu drawn; skipped: %llu no program, %llu shader unusable, %llu no target, %llu other; %llu dispatches (%llu skipped, %llu compute pipelines), %llu DLSS, %llu feedback copies, %llu pool descriptors; %llu shaders, %llu pipelines (%llu failed), %llu empty texture slots",
             (unsigned long long)g_n_draws_done, (unsigned long long)g_n_skip_noprog, (unsigned long long)g_n_skip_bad, (unsigned long long)g_n_skip_rt, (unsigned long long)g_n_skip_other,
             (unsigned long long)g_n_dispatch, (unsigned long long)g_n_dispatch_skip, (unsigned long long)g_n_cpipes, (unsigned long long)g_n_dlss, (unsigned long long)g_n_fbcopy, (unsigned long long)g_n_pool_desc, (unsigned long long)g_n_shaders, (unsigned long long)g_n_pipelines, (unsigned long long)g_n_pipe_fail, (unsigned long long)g_n_missing_tex);
}
