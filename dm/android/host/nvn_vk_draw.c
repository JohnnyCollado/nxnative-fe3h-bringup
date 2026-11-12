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
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)
#include "dmsh_abi.h"
#include "nvn_fmt_vk.h"
#include "nvn_vk_int.h"
#include "fsr2_glue.h"   // FSR 2 behind a C interface (fsr2_glue.cpp): vkd_dlss's temporal upscaler

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
enum { BK_ZERO, BK_SUPPORT, BK_CBUF, BK_SBUF, BK_TEX, BK_IMG, BK_POOL_TEX, BK_POOL_SAMP, BK_ALPHA };
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
    uint8_t sfloat;                                     // the image's sampled type is float (the translator's only type for sampled textures)
} sbind;
typedef struct shd {
    uint64_t key; int stage;                            // NVN stage: 0 vertex, 1 pixel, 2 geometry, 5 compute
    int loaded, bad; VkShaderModule mod; VkDescriptorSetLayout dsl;
    sbind *b; int nb;
    uint32_t in_mask; uint8_t in_kind[32];              // vertex inputs by location: 0 float, 1 int, 2 uint (a pixel shader: its input locations)
    uint32_t out_mask;                                  // a vertex shader's output locations
    uint32_t out_frag;                                  // a pixel shader's colour outputs (location bits)
    uint32_t *words; size_t nwords;                     // the patched SPIR-V, for variants: a vertex shader's extra outputs (vs_variant), integer slots (int_mod)
    uint64_t imask_u, imask_s;                          // set by stage_set for the draw: bindings (b[] index) holding unsigned / signed integer textures in float slots
    struct { uint64_t u, s; VkShaderModule m; } ivar[8]; int nivar, int_fail;   // its integer-slot variants (int_patch); int_fail: one could not be made
    struct { uint32_t add; VkShaderModule m; } var[4]; int nvar;
    VkShaderModule alpha_mod; int has_alpha;            // a pixel shader's alpha-test variant (alpha_inject) and its extra BK_ALPHA binding
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
// NVN's fixed-function alpha test (ColorStateSetAlphaTest + SetAlphaRef), which the translator bakes out (Program.cs passes AlphaTestOp.Always, so the SPIR-V has no
// discard). The Mac route rebuilds a <key>_ba library offline (dmsh_alpha.py); here the same test is injected into the fragment SPIR-V at load, as a variant module
// used only while alpha_func != ALWAYS (the base module keeps its early-Z). The function and reference are data, read from a dm_alpha_state {uint func; float ref;}
// uniform buffer the host fills per draw (stage_set BK_ALPHA), so one variant serves every function and a reference that changes every draw. The test reads colour 0's
// alpha (.w) before each return of the entry point and OpKill on failure, exactly dmsh_alpha.py's dm_alpha_pass (func 1 never, 2 <, 3 ==, 4 <=, 5 >, 6 !=, 7 >=, else pass).
// Refuses (returns 0) when colour 0 is not a vec4 of 32-bit float: an integer or absent colour 0 has no alpha to test, as on the Mac. Validated with spirv-val.
enum { AI_ENTRYPOINT=15, AI_TYPEBOOL=20, AI_TYPEINT=21, AI_TYPEFLOAT=22, AI_TYPEVECTOR=23, AI_TYPESTRUCT=30, AI_TYPEPOINTER=32,
       AI_FUNCTION=54, AI_FUNCTIONEND=56, AI_VARIABLE=59, AI_LOAD=61, AI_ACCESSCHAIN=65, AI_DECORATE=71, AI_MEMBERDECORATE=72,
       AI_CONSTTRUE=41, AI_CONSTFALSE=42, AI_CONSTANT=43, AI_SELECT=169, AI_IEQUAL=170, AI_FORDEQUAL=180, AI_FUNORDNOTEQUAL=183,
       AI_FORDLESSTHAN=184, AI_FORDGREATERTHAN=186, AI_FORDLESSEQUAL=188, AI_FORDGREATEREQUAL=190, AI_LABEL=248, AI_BRANCHCOND=250,
       AI_KILL=252, AI_RETURN=253, AI_SELECTIONMERGE=247, AI_DEC_BLOCK=2, AI_DEC_OFFSET=35, AI_DEC_BINDING=33, AI_DEC_SET=34, AI_DEC_LOCATION=30,
       AI_SC_UNIFORM=2, AI_SC_OUTPUT=3 };
typedef struct { uint32_t *w; size_t n, cap; } ai_buf;
static void ai_put(ai_buf *b, uint32_t v) { if (b->n == b->cap) { b->cap = b->cap ? b->cap*2 : 4096; b->w = realloc(b->w, b->cap*4); } b->w[b->n++] = v; }
static void ai_op(ai_buf *b, uint32_t opc, int n, const uint32_t *a) { ai_put(b, ((uint32_t)(n+1)<<16)|opc); for (int i=0;i<n;i++) ai_put(b, a[i]); }
static uint32_t ai_find_type(const uint32_t *w, size_t nw, uint32_t opc, uint32_t a1, uint32_t a2, int na) {
    for (size_t i=5;i<nw;) { uint32_t o=w[i]&0xffff, l=w[i]>>16; if(!l) break;
        if (o==opc) { int ok=1; if (na>=1 && w[i+2]!=a1) ok=0; if (na>=2 && w[i+3]!=a2) ok=0; if (ok) return w[i+1]; }
        i+=l; }
    return 0;
}
static int alpha_inject(const uint32_t *w, size_t nw, uint32_t alpha_bind, uint32_t set, uint32_t **out, size_t *out_nw) {
    if (nw<5 || w[0]!=0x07230203) return 0;
    uint32_t bound=w[3], entry=0, col0=0; size_t ann_end=0, func_start=0, entry_start=0;
    uint32_t *loc=calloc(bound,4); uint8_t *bi=calloc(bound,1);
    for (size_t i=5;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l||i+l>nw) break;
        if (o==AI_ENTRYPOINT && w[i+1]==4) entry=w[i+2];
        if (o==AI_DECORATE && l>=3 && w[i+1]<bound) { if (w[i+2]==AI_DEC_LOCATION) loc[w[i+1]]=w[i+3]+1; if (w[i+2]==11) bi[w[i+1]]=1; }
        if (o==AI_DECORATE||o==AI_MEMBERDECORATE||o==73||o==74||o==75||o==332||o==5632) ann_end=i+l;
        if (o==AI_FUNCTION && !func_start) func_start=i;
        i+=l; }
    if (!entry||!func_start) { free(loc); free(bi); return 0; }
    for (size_t i=func_start;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l) break; if (o==AI_FUNCTION && w[i+2]==entry) { entry_start=i; break; } i+=l; }
    if (!entry_start) { free(loc); free(bi); return 0; }
    uint32_t uint_t=ai_find_type(w,nw,AI_TYPEINT,32,0,2), float_t=ai_find_type(w,nw,AI_TYPEFLOAT,32,0,1), v4float_t=ai_find_type(w,nw,AI_TYPEVECTOR,float_t,4,2);
    for (size_t i=5;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l) break;
        if (o==AI_VARIABLE && l>=4 && w[i+3]==AI_SC_OUTPUT && w[i+2]<bound && loc[w[i+2]]==1 && !bi[w[i+2]]) {
            uint32_t ptr=w[i+1];
            for (size_t j=5;j<nw;) { uint32_t oo=w[j]&0xffff,ll=w[j]>>16; if(!ll) break;
                if (oo==AI_TYPEPOINTER && w[j+1]==ptr && w[j+3]==v4float_t) col0=w[i+2];
                j+=ll; }
        }
        i+=l; }
    if (!col0||!uint_t||!float_t||!v4float_t) { free(loc); free(bi); return 0; }
    size_t f_end=0; int depth=0;
    for (size_t i=entry_start;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l) break; if(o==AI_FUNCTION) depth++; if(o==AI_FUNCTIONEND){depth--; if(!depth){f_end=i+l;break;}} i+=l; }
    if (!f_end) { free(loc); free(bi); return 0; }
    uint32_t next=bound;
    #define AINEW() (next++)
    uint32_t bool_t=ai_find_type(w,nw,AI_TYPEBOOL,0,0,0), ptr_u_uint=ai_find_type(w,nw,AI_TYPEPOINTER,AI_SC_UNIFORM,uint_t,2),
             ptr_u_float=ai_find_type(w,nw,AI_TYPEPOINTER,AI_SC_UNIFORM,float_t,2), ptr_o_float=ai_find_type(w,nw,AI_TYPEPOINTER,AI_SC_OUTPUT,float_t,2);
    uint32_t uc[8]={0}, t_id=0, f_id=0;
    for (size_t i=5;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l) break;
        if (o==AI_CONSTANT && l>=4 && w[i+1]==uint_t && w[i+3]<8) uc[w[i+3]]=w[i+2];
        if (o==AI_CONSTTRUE) t_id=w[i+2]; if (o==AI_CONSTFALSE) f_id=w[i+2];
        i+=l; }
    uint32_t alpha_struct=AINEW(), ptr_u_alpha=AINEW(), alpha_var=AINEW();
    uint32_t need_bool=bool_t?0:AINEW(), need_puuint=ptr_u_uint?0:AINEW(), need_pufloat=ptr_u_float?0:AINEW(), need_pofloat=ptr_o_float?0:AINEW();
    uint32_t need_uc[8]; for (int v=0;v<8;v++) need_uc[v]=uc[v]?0:AINEW();
    uint32_t need_true=t_id?0:AINEW(), need_false=f_id?0:AINEW();
    if (!bool_t) bool_t=need_bool; if (!ptr_u_uint) ptr_u_uint=need_puuint; if (!ptr_u_float) ptr_u_float=need_pufloat; if (!ptr_o_float) ptr_o_float=need_pofloat;
    for (int v=0;v<8;v++) if (!uc[v]) uc[v]=need_uc[v];
    if (!t_id) t_id=need_true; if (!f_id) f_id=need_false;
    ai_buf b={0}; b.cap=nw+8192; b.w=malloc(b.cap*4);
    size_t i=0;
    for (;i<ann_end;) { uint32_t l=(i<5)?1:w[i]>>16; for(size_t k=0;k<l;k++) ai_put(&b,w[i+k]); i+=l; }
    { uint32_t a[2]={alpha_struct,AI_DEC_BLOCK}; ai_op(&b,AI_DECORATE,2,a); }
    { uint32_t a[4]={alpha_struct,0,AI_DEC_OFFSET,0}; ai_op(&b,AI_MEMBERDECORATE,4,a); }
    { uint32_t a[4]={alpha_struct,1,AI_DEC_OFFSET,4}; ai_op(&b,AI_MEMBERDECORATE,4,a); }
    { uint32_t a[3]={alpha_var,AI_DEC_SET,set}; ai_op(&b,AI_DECORATE,3,a); }
    { uint32_t a[3]={alpha_var,AI_DEC_BINDING,alpha_bind}; ai_op(&b,AI_DECORATE,3,a); }
    for (;i<func_start;) { uint32_t l=w[i]>>16; for(size_t k=0;k<l;k++) ai_put(&b,w[i+k]); i+=l; }
    if (need_bool) { uint32_t a[1]={bool_t}; ai_op(&b,AI_TYPEBOOL,1,a); }
    { uint32_t a[3]={alpha_struct,uint_t,float_t}; ai_op(&b,AI_TYPESTRUCT,3,a); }
    { uint32_t a[3]={ptr_u_alpha,AI_SC_UNIFORM,alpha_struct}; ai_op(&b,AI_TYPEPOINTER,3,a); }
    if (need_puuint) { uint32_t a[3]={ptr_u_uint,AI_SC_UNIFORM,uint_t}; ai_op(&b,AI_TYPEPOINTER,3,a); }
    if (need_pufloat) { uint32_t a[3]={ptr_u_float,AI_SC_UNIFORM,float_t}; ai_op(&b,AI_TYPEPOINTER,3,a); }
    if (need_pofloat) { uint32_t a[3]={ptr_o_float,AI_SC_OUTPUT,float_t}; ai_op(&b,AI_TYPEPOINTER,3,a); }
    for (int v=0;v<8;v++) if (need_uc[v]) { uint32_t a[3]={uint_t,uc[v],(uint32_t)v}; ai_op(&b,AI_CONSTANT,3,a); }
    if (need_true) { uint32_t a[2]={bool_t,t_id}; ai_op(&b,AI_CONSTTRUE,2,a); }
    if (need_false) { uint32_t a[2]={bool_t,f_id}; ai_op(&b,AI_CONSTFALSE,2,a); }
    { uint32_t a[3]={ptr_u_alpha,alpha_var,AI_SC_UNIFORM}; ai_op(&b,AI_VARIABLE,3,a); }
    for (;i<nw;) { uint32_t o=w[i]&0xffff,l=w[i]>>16; if(!l) break;
        if (i>=entry_start && i<f_end && o==AI_RETURN) {
            uint32_t fp=AINEW(),fv=AINEW(),rp=AINEW(),rv=AINEW(),cp=AINEW(),av=AINEW();
            uint32_t lt=AINEW(),eq=AINEW(),le=AINEW(),gt=AINEW(),ge=AINEW(),ne=AINEW(), is[8];
            for(int v=1;v<8;v++) is[v]=AINEW();
            uint32_t kill=AINEW(),merge=AINEW(),a[5];
            a[0]=ptr_u_uint;a[1]=fp;a[2]=alpha_var;a[3]=uc[0]; ai_op(&b,AI_ACCESSCHAIN,4,a);
            a[0]=uint_t;a[1]=fv;a[2]=fp; ai_op(&b,AI_LOAD,3,a);
            a[0]=ptr_u_float;a[1]=rp;a[2]=alpha_var;a[3]=uc[1]; ai_op(&b,AI_ACCESSCHAIN,4,a);
            a[0]=float_t;a[1]=rv;a[2]=rp; ai_op(&b,AI_LOAD,3,a);
            a[0]=ptr_o_float;a[1]=cp;a[2]=col0;a[3]=uc[3]; ai_op(&b,AI_ACCESSCHAIN,4,a);
            a[0]=float_t;a[1]=av;a[2]=cp; ai_op(&b,AI_LOAD,3,a);
            a[0]=bool_t;a[1]=lt;a[2]=av;a[3]=rv; ai_op(&b,AI_FORDLESSTHAN,4,a);
            a[0]=bool_t;a[1]=eq;a[2]=av;a[3]=rv; ai_op(&b,AI_FORDEQUAL,4,a);
            a[0]=bool_t;a[1]=le;a[2]=av;a[3]=rv; ai_op(&b,AI_FORDLESSEQUAL,4,a);
            a[0]=bool_t;a[1]=gt;a[2]=av;a[3]=rv; ai_op(&b,AI_FORDGREATERTHAN,4,a);
            a[0]=bool_t;a[1]=ge;a[2]=av;a[3]=rv; ai_op(&b,AI_FORDGREATEREQUAL,4,a);
            a[0]=bool_t;a[1]=ne;a[2]=av;a[3]=rv; ai_op(&b,AI_FUNORDNOTEQUAL,4,a);
            for (int v=1;v<8;v++) { a[0]=bool_t;a[1]=is[v];a[2]=fv;a[3]=uc[v]; ai_op(&b,AI_IEQUAL,4,a); }
            uint32_t pass=t_id, sel[8]={0,f_id,lt,eq,le,gt,ne,ge};
            for (int v=1;v<8;v++) { uint32_t p=AINEW(); a[0]=bool_t;a[1]=p;a[2]=is[v];a[3]=sel[v];a[4]=pass; ai_op(&b,AI_SELECT,5,a); pass=p; }
            { uint32_t m[2]={merge,0}; ai_op(&b,AI_SELECTIONMERGE,2,m); }
            { uint32_t bc[3]={pass,merge,kill}; ai_op(&b,AI_BRANCHCOND,3,bc); }
            { uint32_t lb[1]={kill}; ai_op(&b,AI_LABEL,1,lb); }
            ai_op(&b,AI_KILL,0,0);
            { uint32_t lb[1]={merge}; ai_op(&b,AI_LABEL,1,lb); }
            ai_op(&b,AI_RETURN,0,0);
            i+=l; continue;
        }
        for(size_t k=0;k<l;k++) ai_put(&b,w[i+k]); i+=l;
    }
    b.w[3]=next;
    #undef AINEW
    free(loc); free(bi);
    *out=b.w; *out_nw=b.n; return 1;
}
static int alpha_enabled(void) { static int v=-1; if (v<0) v=getenv("DM_VK_NO_ALPHA")==NULL; return v; }
// Registers read before any write read zero. The translator keeps each Maxwell register in a Function-storage variable declared without an initialiser, and a path
// that reads one before writing it (a predicated or branch-only write) reads an undefined value: NVIDIA's drivers give zero, the Adreno whatever the register held.
// In Fire Emblem: Fortune's Weave the lit-material pixel shader 5743b0078253f3ed has 32 such variables and painted flat magenta rectangles into the scene (about 1 in 10
// of the game's shaders have some). Every such variable gets an OpConstantNull initialiser of its type, as Eden does for its OpUndef
// (dm/third_party/eden-fefw: emit_spirv_undefined.cpp, "stray coloured patches on AMD"); a driver drops the store where a write always comes first.
// DM_VK_UNDEF_ZERO=0 keeps the module as translated. Returns the number of variables given an initialiser (the words are replaced when > 0)
static uint32_t zero_locals(uint32_t **pw, size_t *pnw) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_UNDEF_ZERO"); on = !(e && e[0] == '0'); }
    if (!on) return 0;
    uint32_t *w = *pw; size_t nw = *pnw; const uint32_t bound0 = w[3]; uint32_t bound = bound0;   // bound0 sizes the tables; bound grows as IDs are handed out
    uint32_t *pointee = calloc(bound, 4), *null_of = calloc(bound, 4); uint8_t *need = calloc(bound, 1);
    size_t first_fn = 0; uint32_t nvar = 0, nnew = 0;
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == SPV_TYPE_POINTER && len == 4 && w[i + 1] < bound && w[i + 2] == 7 /* Function */) pointee[w[i + 1]] = w[i + 3];
        else if (op == 46 /* OpConstantNull */ && len == 3 && w[i + 1] < bound) null_of[w[i + 1]] = w[i + 2];
        else if (op == 54 /* OpFunction */ && !first_fn) first_fn = i;
        else if (op == SPV_VARIABLE && len == 4 && w[i + 3] == 7 && w[i + 1] < bound && pointee[w[i + 1]]) { uint32_t t = pointee[w[i + 1]]; if (t < bound && !null_of[t] && !need[t]) { need[t] = 1; nnew++; } nvar++; }
        i += len;
    }
    if (!nvar || !first_fn) { free(pointee); free(null_of); free(need); return 0; }
    uint32_t *o = malloc((nw + (size_t)nvar + 3 * (size_t)nnew) * 4), no = 5;
    memcpy(o, w, 5 * 4);   // the header (its bound, word 3, is set below)
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) { memcpy(o + no, w + i, (nw - i) * 4); no += (uint32_t)(nw - i); break; }
        if (i == first_fn) for (uint32_t t = 0; t < bound0; t++) if (need[t]) { o[no++] = 3u << 16 | 46; o[no++] = t; o[no++] = null_of[t] = bound++; }
        if (op == SPV_VARIABLE && len == 4 && w[i + 3] == 7 && w[i + 1] < bound0 && pointee[w[i + 1]]) {
            o[no++] = 5u << 16 | SPV_VARIABLE; o[no++] = w[i + 1]; o[no++] = w[i + 2]; o[no++] = 7; o[no++] = null_of[pointee[w[i + 1]]];
        } else { memcpy(o + no, w + i, len * 4); no += len; }
        i += len;
    }
    o[3] = bound;
    free(pointee); free(null_of); free(need); free(w);
    *pw = o; *pnw = no;
    return nvar;
}
static _Atomic uint64_t g_words_bytes;   // SPIR-V kept for variants (the memory line)
uint64_t vkd_words_bytes(void) { return atomic_load(&g_words_bytes); }
static int shd_load_do(shd *s);
// Vertex outputs start at zero. A translated vertex shader writes only what its Maxwell / Ampere code wrote: some components of a location never (Fire Emblem
// Fortune's Weave's dust particles: 3d43d86f8e7702f5 writes location 6's x, y and z, and 300c5dd0fd84cf79 reads its w too), and little or nothing on a path that
// returns early. What such an output holds is the driver's. Qualcomm's gives 0 (the work folder's re/texprobe/vsrun.c: its capture of that shader on generated
// inputs is all zero); Turnip gives leftovers (a third of the bytes never written, the rest other values' remains), and the dust came out as solid flat quads.
// Every Output variable with a location is stored zero (OpConstantNull) at the start of the entry function, and so is the position, through an access chain into
// its block, so both drivers see the same (vsrun on both: identical captures). Stores, not OpVariable initializers; and not a store of the whole block, which
// crashed both drivers' compilers. DM_VK_ZERO_OUT=0: as translated. Returns the new words (w is freed) or w
static uint32_t *spv_zero_outputs(uint32_t *w, size_t *pnw, int *count) {
    size_t nw = *pnw; uint32_t bound = w[3], entry = 0, iconst[8] = {0}; *count = 0;
    uint32_t *pointee = calloc(bound, 4), *optr = calloc(bound, 4), *nullc = calloc(bound, 4), *struct_at = calloc(bound, 4), *posm = calloc(bound, 4);
    uint8_t *want = calloc(bound, 1), *is_int = calloc(bound, 1); size_t nvar = 0;
    for (size_t i = 5; i < nw; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) break;
        if (op == 15 && !entry) entry = w[i + 2];
        if (op == SPV_TYPE_POINTER && len == 4 && w[i + 1] < bound && w[i + 3] < bound && w[i + 2] == 3) { pointee[w[i + 1]] = w[i + 3]; optr[w[i + 3]] = w[i + 1]; }
        if (op == SPV_TYPE_STRUCT && w[i + 1] < bound) struct_at[w[i + 1]] = (uint32_t)i;
        if (op == SPV_TYPE_INT && w[i + 1] < bound && w[i + 2] == 32) is_int[w[i + 1]] = 1;
        if (op == SPV_CONSTANT && len == 4 && w[i + 1] < bound && is_int[w[i + 1]] && w[i + 3] < 8 && !iconst[w[i + 3]]) iconst[w[i + 3]] = w[i + 2];
        if (op == SPV_DECORATE && len >= 4 && w[i + 2] == DEC_LOCATION && w[i + 1] < bound) want[w[i + 1]] = 1;
        if (op == 72 && len == 5 && w[i + 3] == DEC_BUILTIN && w[i + 4] == 0 && w[i + 1] < bound) posm[w[i + 1]] = w[i + 2] + 1;   // OpMemberDecorate ... BuiltIn Position: the member, plus 1
        if (op == 46 && len == 3 && w[i + 1] < bound) nullc[w[i + 1]] = w[i + 2];   // an OpConstantNull the module already has, by type
        if (op == SPV_VARIABLE && w[i + 3] == 3) nvar++;
        i += len; }
    uint32_t *o = malloc((nw + nvar * 12) * 4), *todo = malloc((nvar + 1) * 16); size_t no = 0, nt = 0; int state = 0;   // todo: variable, null constant, pointer type and index constant of a member (0: the variable itself)
    for (size_t i = 0; i < nw; ) { uint32_t op = i < 5 ? 0 : w[i] & 0xffff, len = i < 5 ? 1 : w[i] >> 16; if (!len) { memcpy(o + no, w + i, (nw - i) * 4); no += nw - i; break; }
        if (op == SPV_VARIABLE && w[i + 3] == 3 && w[i + 2] < bound && w[i + 1] < bound && pointee[w[i + 1]]) {
            uint32_t t = pointee[w[i + 1]], mt = 0, m = 0;
            if (!want[w[i + 2]]) {   // not a located output: the block with the position, if its member's pointer type and index constant are in the module
                t = 0;
                if (struct_at[pointee[w[i + 1]]] && posm[pointee[w[i + 1]]]) { m = posm[pointee[w[i + 1]]] - 1; size_t sa = struct_at[pointee[w[i + 1]]];
                    if (m + 2 < (w[sa] >> 16) && m < 8 && iconst[m]) { mt = w[sa + 2 + m]; if (mt < bound && optr[mt]) t = mt; } }
            }
            if (t) {
                if (!nullc[t]) { nullc[t] = bound++; o[no++] = 3u << 16 | 46; o[no++] = t; o[no++] = nullc[t]; }   // before the variable: its type is declared by now
                todo[nt * 4] = w[i + 2]; todo[nt * 4 + 1] = nullc[t]; todo[nt * 4 + 2] = mt ? optr[mt] : 0; todo[nt * 4 + 3] = mt ? iconst[m] : 0; nt++; }
        }
        if (state == 2 && op != SPV_VARIABLE) {   // after the entry function's own variables
            for (size_t k = 0; k < nt; k++) { uint32_t dst = todo[k * 4];
                if (todo[k * 4 + 2]) { dst = bound++; o[no++] = 5u << 16 | 65; o[no++] = todo[k * 4 + 2]; o[no++] = dst; o[no++] = todo[k * 4]; o[no++] = todo[k * 4 + 3]; }   // OpAccessChain to the position
                o[no++] = 3u << 16 | 62; o[no++] = dst; o[no++] = todo[k * 4 + 1]; }
            state = 3; *count = (int)nt; }
        memcpy(o + no, w + i, len * 4); no += len;
        if (op == 54 && w[i + 2] == entry) state = 1; else if (state == 1 && op == 248) state = 2;
        i += len; }
    free(pointee); free(optr); free(nullc); free(struct_at); free(posm); free(want); free(is_int); free(todo);
    if (!*count) { free(o); return w; }
    o[3] = bound; free(w); *pnw = no; return o;
}
static int shd_load(shd *s) {   // a shader's first use reads its SPIR-V from storage and makes its module: timed for the slow-frame log (vkx_sb_note 4)
    if (s->loaded) return !s->bad;
    uint64_t t0 = vkx_now_ns(); int ok = shd_load_do(s); vkx_sb_note(4, vkx_now_ns() - t0); return ok;
}
// A per-pixel index into an array of textures or samplers is marked NonUniform. A translated shader picks a texture by an index it computes for each pixel (Fire
// Emblem Fortune's Weave's lighting draw 88d7f48d93663021: the reflection probe a pixel belongs to, out of an array of 32 cube maps); Vulkan leaves such an index
// undefined unless the access is decorated NonUniform, and the translator does not decorate. Qualcomm's driver takes each pixel's own index anyway; Turnip took one
// index for several pixels, so pixels beside an edge got their neighbour's probe: light or dark squares beside stone edges and stepped shadow edges, with Turnip only
// (the work folder's re/csrun/psrun_cap drew the captured draw on both drivers: 2,620 pixels differed, 140 by rounding once the accesses were marked; nonuniform.py
// there is this pass). Marked: each access chain into such an array with an index that is not a constant, what is loaded through it, and a sampled image made from
// that; with the ShaderNonUniform and SampledImageArrayNonUniformIndexing capabilities and SPV_EXT_descriptor_indexing. Only where the device has the feature
// (g_vk.nonuniform); DM_VK_NONUNIFORM=0: as translated. Returns the ids marked
static uint32_t spv_nonuniform(uint32_t **pw, size_t *pnw) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_NONUNIFORM"); on = g_vk.nonuniform && !(e && e[0] == '0');
        VLOG("texture arrays indexed per pixel %s", on ? "are marked NonUniform (each pixel takes its own index on any driver)" : g_vk.nonuniform ? "as translated (DM_VK_NONUNIFORM=0)" : "as translated (the device lacks shaderSampledImageArrayNonUniformIndexing)"); }
    if (!on) return 0;
    uint32_t *w = *pw; size_t nw = *pnw; uint32_t bound = w[3], n = 0;
    uint8_t *opaque = calloc(bound, 1), *arrv = calloc(bound, 1), *cst = calloc(bound, 1), *mark = calloc(bound, 1); uint32_t *arr = calloc(bound, 4), *ptr = calloc(bound, 4);
    for (size_t i = 5; i < nw; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == SPV_TYPE_IMAGE && len >= 8 && w[i + 1] < bound) opaque[w[i + 1]] = w[i + 7] != 2;              // not a storage image (another capability)
        else if ((op == SPV_TYPE_SAMPLER || op == 27) && w[i + 1] < bound) opaque[w[i + 1]] = 1;                   // 27: OpTypeSampledImage
        else if ((op == SPV_TYPE_ARRAY || op == SPV_TYPE_RTARRAY) && w[i + 1] < bound) arr[w[i + 1]] = w[i + 2];
        else if (op == SPV_TYPE_POINTER && len == 4 && w[i + 1] < bound) ptr[w[i + 1]] = w[i + 3];
        else if ((op == 41 || op == 42 || op == 43 || op == 44 || op == 46) && len >= 3 && w[i + 2] < bound) cst[w[i + 2]] = 1;   // constants
        else if (op == SPV_VARIABLE && len >= 4 && w[i + 2] < bound && w[i + 1] < bound) { uint32_t t = ptr[w[i + 1]]; arrv[w[i + 2]] = t < bound && arr[t] && arr[t] < bound && opaque[arr[t]]; }
        else if ((op == 65 || op == 66) && len >= 5 && w[i + 2] < bound && w[i + 3] < bound && arrv[w[i + 3]] && w[i + 4] < bound && !cst[w[i + 4]]) { mark[w[i + 2]] = 1; n++; }   // OpAccessChain
        else if (op == 61 && len >= 4 && w[i + 2] < bound && w[i + 3] < bound && mark[w[i + 3]]) { mark[w[i + 2]] = 1; n++; }                                                     // OpLoad
        else if (op == 86 && len >= 5 && w[i + 2] < bound && ((w[i + 3] < bound && mark[w[i + 3]]) || (w[i + 4] < bound && mark[w[i + 4]]))) { mark[w[i + 2]] = 1; n++; }        // OpSampledImage
        else if (op == 100 && len >= 4 && w[i + 2] < bound && w[i + 3] < bound && mark[w[i + 3]]) { mark[w[i + 2]] = 1; n++; }                                                    // OpImage
        i += len; }
    if (n) {
        static const char ext[] = "SPV_EXT_descriptor_indexing"; uint32_t extw[8] = {0}; memcpy(extw, ext, sizeof ext); uint32_t nextw = (uint32_t)(sizeof ext + 3) / 4;
        uint32_t *o = malloc((nw + 4 + 1 + nextw + 3 * (size_t)n) * 4); size_t no = 0; int done_ext = w[1] >= 0x10500, done_dec = 0;
        memcpy(o, w, 20); no = 5; o[no++] = 2u << 16 | 17; o[no++] = 5301; o[no++] = 2u << 16 | 17; o[no++] = 5307;   // OpCapability ShaderNonUniform, SampledImageArrayNonUniformIndexing
        for (size_t i = 5; i < nw; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
            if (!done_ext && op != 17 && op != 10) { o[no++] = (nextw + 1) << 16 | 10; memcpy(o + no, extw, nextw * 4); no += nextw; done_ext = 1; }   // OpExtension, after the capabilities and extensions
            if (!done_dec && op != 17 && op != 10 && op != 11 && op != 14 && op != 15 && op != 16 && !(op >= 3 && op <= 8) && op != 330 && op != 331) {   // the annotations start here at the latest
                for (uint32_t id = 0; id < bound; id++) if (mark[id]) { o[no++] = 3u << 16 | SPV_DECORATE; o[no++] = id; o[no++] = 5300; }
                done_dec = 1; }
            memcpy(o + no, w + i, len * 4); no += len; i += len; }
        free(w); *pw = o; *pnw = no;
    }
    free(opaque); free(arrv); free(cst); free(mark); free(arr); free(ptr);
    return n;
}
static int shd_load_do(shd *s) {
    s->loaded = 1; s->bad = 1;
    char p[600]; snprintf(p, sizeof p, "%s/%016llx.spv", g_spirv_dir, (unsigned long long)s->key);
    FILE *f = fopen(p, "rb"); if (!f) { VLOG("[RED] shader %016llx: %s missing", (unsigned long long)s->key, p); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)n); size_t nw = fread(w, 1, (size_t)n, f) / 4; fclose(f);
    if (nw < 5 || w[0] != 0x07230203) { VLOG("[RED] shader %016llx: not SPIR-V", (unsigned long long)s->key); free(w); return 0; }
    { static uint64_t nshd, nvars; uint32_t z = zero_locals(&w, &nw); if (z) { nshd++; nvars += z; if (nshd == 1 || nshd % 500 == 0) VLOG("registers read before a write read zero: %llu shaders, %llu variables given a zero initialiser so far", (unsigned long long)nshd, (unsigned long long)nvars); } }
    { static uint64_t nshd, nids; uint32_t m = spv_nonuniform(&w, &nw); if (m) { nshd++; nids += m; if (nshd == 1 || nshd % 200 == 0) VLOG("texture array accesses marked NonUniform: %llu shaders, %llu ids so far", (unsigned long long)nshd, (unsigned long long)nids); } }
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
        if (img && img < bound && def[img]) { b->dim = (uint8_t)w[def[img] + 3]; b->arrayed = (uint8_t)w[def[img] + 5]; b->ms = (uint8_t)w[def[img] + 6];
                                             uint32_t st = w[def[img] + 2]; b->sfloat = st < bound && def[st] && (w[def[st]] & 0xffff) == SPV_TYPE_FLOAT; }
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
    if (ok && stage == 1 && alpha_enabled()) {   // the alpha-test variant: a BK_ALPHA uniform binding the base module ignores, read by alpha_mod while alpha_func != ALWAYS
        uint32_t *aw; size_t anw;
        if (alpha_inject(w, nw, (uint32_t)s->nb, 1, &aw, &anw)) {
            VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, anw * 4, aw};
            if (!vkCreateShaderModule(g_vk.dev, &mi, NULL, &s->alpha_mod)) {
                s->has_alpha = 1;
                s->b[s->nb] = (sbind){.vb = (uint32_t)s->nb, .dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .kind = BK_ALPHA};
                lb[s->nb] = (VkDescriptorSetLayoutBinding){(uint32_t)s->nb, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
                s->nb++;
                if (getenv("DM_VK_DUMP_SPV")) { char d[600], q[700]; snprintf(d, sizeof d, "%s/spv_patched", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(d, 0755);
                    snprintf(q, sizeof q, "%s/%016llx_a.spv", d, (unsigned long long)s->key); FILE *o = fopen(q, "wb"); if (o) { fwrite(aw, 4, anw, o); fclose(o); } }
            } else VLOG("[RED] shader %016llx: alpha variant module failed", (unsigned long long)s->key);
            free(aw);
        }
    }
    {   // DM_VK_DUMP_SPV: the patched modules as $DM_ROOT/spv_patched/<key>.spv (checked on the PC with spirv-val)
        static int dump = -1; if (dump < 0) dump = getenv("DM_VK_DUMP_SPV") != NULL;
        if (dump) { char d[600], q[700]; snprintf(d, sizeof d, "%s/spv_patched", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(d, 0755);
            snprintf(q, sizeof q, "%s/%016llx.spv", d, (unsigned long long)s->key); FILE *o = fopen(q, "wb"); if (o) { fwrite(w, 4, nw, o); fclose(o); } }
    }
    if (ok && stage == 0) {
        static int zo = -1; if (zo < 0) { const char *e = getenv("DM_VK_ZERO_OUT"); zo = !(e && e[0] == '0'); VLOG("vertex outputs %s", zo ? "are stored zero first (what a shader does not write is 0 on any driver)" : "as translated (DM_VK_ZERO_OUT=0)"); }
        if (zo) { int nz = 0; w = spv_zero_outputs(w, &nw, &nz); }
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
    {   // kept for variants: a vertex shader's extra outputs (vs_variant), and any shader with a sampled float slot an integer texture may be bound to (int_mod)
        int sampled = 0; for (int i = 0; i < s->nb; i++) sampled |= s->b[i].sfloat && (s->b[i].dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || s->b[i].dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        if (!s->bad && (stage == 0 || sampled)) { s->words = w; s->nwords = nw; w = NULL; atomic_fetch_add(&g_words_bytes, (uint64_t)nw * 4); }
    }
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
    float alpha_ref;                                    // NVN SetAlphaRef (OP_ALPHA_REF): the alpha test's reference, data for the BK_ALPHA buffer
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
        // A bind replaces the whole table (nvnCommandBufferBindVertexAttribState / StreamState record first 0, n): attributes and streams past n are off, as in nvn_mtl.m.
        // The counts only grew here, so a mesh bound with fewer attributes kept the last mesh's entries above its own: Fire Emblem Fortune's Weave's wind shader then read
        // its sway weight (location 4) for 24-byte-vertex props from a stale RG16F entry at offset 24, the next corner's position bytes: weights up to 20,000, the props
        // stretched across the screen in the depth pre-pass and differently in the G-buffer pass (black ribbons; the captures logs_mem3/003910 and 001566, re/tools/windrun.py)
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(S.attr + b->first, b->st, 4 * (size_t)b->n); S.nattr = b->first + b->n; } return 1; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(S.stream + b->first, b->st, 8 * (size_t)b->n); S.nstream = b->first + b->n; } return 1; }
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
        case OP_ALPHA_REF: S.alpha_ref = ((const p_alpha_ref *)a)->ref; return 1;
        case OP_VP_SWIZZLE: case OP_TF_BIND: case OP_TF_STATE: case OP_BEGIN: return 1;
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
// Diagnostic (always on, logged only on a hit): storage buffers the compute dispatches recorded in the current submission write. The GPU has not run them while the
// rest of the submission is recorded, so a texture handle the host reads from game memory inside one of these ranges (binding_ids -> cb_word) is the value from before
// the dispatch: Eden's notes for Fire Emblem Fortune's Weave say it builds texture-handle tables with compute shaders and draws with them a few draws later
// (dm/third_party/eden-fefw CHANGELOG, "Smoother 60 FPS in busy scenes"). Each hit is logged (the first 50) and noted in the rolling capture's draw list (DM_VK_RING)
typedef struct { uint64_t lo, hi, cs; } cs_range;
static cs_range g_csw[256]; static int g_ncsw; static uint64_t g_cs_cur;
static uint64_t g_hstale_n; static int g_hstale_hit; static uint64_t g_hstale_addr;
void vkd_sub_committed(uint64_t value) { (void)value; g_ncsw = 0; }
static void csw_add(uint64_t addr, uint64_t size) { if (g_ncsw < 256 && size) { g_csw[g_ncsw].lo = addr; g_csw[g_ncsw].hi = addr + size; g_csw[g_ncsw].cs = g_cs_cur; g_ncsw++; } }
static int csw_hit(uint64_t addr, uint64_t *cs) { for (int i = 0; i < g_ncsw; i++) if (addr >= g_csw[i].lo && addr < g_csw[i].hi) { *cs = g_csw[i].cs; return 1; } return 0; }
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
// n bytes at addr as the GPU reads them at this point of the recording: game memory, with the UPDATE_UB writes recorded but not yet run laid over it, oldest first (ub_word
// for a block). For a uniform buffer bound through a copy (stage_set: one not aligned to the device's minUniformBufferOffsetAlignment)
static void ub_copy(uint8_t *dst, uint64_t addr, uint64_t n) {
    memcpy(dst, (const void *)(uintptr_t)addr, n);
    if (!g_sh_seq) return;
    uint64_t applied = atomic_load(&g_applied), seqs[256]; int ns = 0;
    for (uint64_t blk = addr >> 8; blk <= (addr + n - 1) >> 8 && ns < 256; blk++) {
        uint32_t bk = (uint32_t)(blk * 0x9E3779B1u) & (SH_BUCKETS - 1);
        for (uint64_t link = g_sh_bucket[bk], steps = 0; link && steps < 4096 && ns < 256; steps++) {
            uint64_t seq = link - 1; sh_ent *e = &g_she[seq % SH_ENT];
            if (e->seq != seq || g_sh_seq - seq >= SH_ENT || g_sh_pos - e->pos > SH_BYTES - 65536) break;
            if (e->fence <= applied) break;
            if (e->addr < addr + n && e->addr + e->n > addr) { int dup = 0; for (int i = 0; i < ns; i++) if (g_she[seqs[i] % SH_ENT].pos == e->pos) dup = 1; if (!dup) seqs[ns++] = seq; }
            link = e->prev[0];
        }
    }
    for (int i = 1; i < ns; i++) for (int j = i; j > 0 && seqs[j - 1] > seqs[j]; j--) { uint64_t t = seqs[j]; seqs[j] = seqs[j - 1]; seqs[j - 1] = t; }   // oldest first: the newest wins
    for (int i = 0; i < ns; i++) {
        sh_ent *e = &g_she[seqs[i] % SH_ENT];
        uint64_t lo = e->addr > addr ? e->addr : addr, hi = e->addr + e->n < addr + n ? e->addr + e->n : addr + n;
        memcpy(dst + (lo - addr), g_sh_bytes + e->pos % SH_BYTES + (lo - e->addr), hi - lo);
    }
}
static uint64_t g_n_ubo_copy;   // uniform buffers bound through an aligned copy (not aligned to minUniformBufferOffsetAlignment in their pool)

// ---------- Per-submission memory: constant blocks and descriptor sets ----------
static _Atomic uint64_t g_arena_bytes;   // host-visible arena memory alive (the memory line)
uint64_t vkd_arena_bytes(void) { return atomic_load(&g_arena_bytes); }
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
        atomic_fetch_add(&g_arena_bytes, a->size); s->nar++;
    }
}
uint8_t *vkd_stage_alloc(sub_t *s, uint64_t n, VkBuffer *buf, uint64_t *off) { return arena_alloc(s, n, buf, off); }   // texture-upload staging (nvn_vk_exec.c): the arenas live until the submission completes, exactly a staging buffer's lifetime
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
    // The arenas: a few standard ones are kept for the next submission in this slot; the rest go. Kept all, each slot held the most it had ever needed: a battle's
    // texture uploads (staging, vkd_stage_alloc) left about 2.2 GB of host-visible memory in Fire Emblem Fortune's Weave, which Android's low-memory killer then
    // stopped after its tutorial. The GPU is done with this submission, so they can be freed here
    int keep = 0;
    for (int i = 0; i < s->nar; i++) {
        vkx_arena *a = &s->arenas[i];
        if (keep < 8 && a->size == (4u << 20)) { a->used = 0; s->arenas[keep++] = *a; }
        else { vkDestroyBuffer(g_vk.dev, a->b, NULL); vkFreeMemory(g_vk.dev, a->m, NULL); atomic_fetch_sub(&g_arena_bytes, a->size); }
    }
    s->nar = keep; s->ar_cur = 0;
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
typedef struct { samp_par p; VkSampler s; VkSamplerCreateInfo si; int custom; } samp_ent;
static const samp_ent *g_samp_last;   // the sampler sampler_of gave last (DM_VK_PS_DUMP)
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
        if (e->s && !memcmp(&e->p, sp, sizeof *sp)) { g_samp_last = e; return e->s; }
        if (e->s) continue;
        int mip = sp->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
        VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, NULL, {.float32 = {sp->border[0], sp->border[1], sp->border[2], sp->border[3]}}, VK_FORMAT_UNDEFINED};
        int border = sp->ws == NVN_WRAP_MODE_CLAMP_TO_BORDER || sp->wt == NVN_WRAP_MODE_CLAMP_TO_BORDER || sp->wr == NVN_WRAP_MODE_CLAMP_TO_BORDER;
        // The three border colours Vulkan names are given by name, a custom colour only for any other: a custom colour without a format
        // (customBorderColorWithoutFormat) is the driver's to pack for whatever texture is sampled, and the named ones do not depend on that
        const float *bc = sp->border; int std_bc = (bc[0] == 0 && bc[1] == 0 && bc[2] == 0 && (bc[3] == 0 || bc[3] == 1)) || (bc[0] == 1 && bc[1] == 1 && bc[2] == 1 && bc[3] == 1);
        { static int once; if (border && !once) { once = 1; VLOG("samplers: border colours %s", std_bc ? "by name where they are transparent black, opaque black or opaque white" : "custom (first seen is not a named one)"); } }
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, border && g_vk.border && !std_bc ? &cb : NULL, 0,
            sp->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, (sp->minf & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
            sp->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
            wrap_vk(sp->ws), wrap_vk(sp->wt), wrap_vk(sp->wr), sp->lod_bias, sp->aniso > 1, sp->aniso > 1 ? (sp->aniso < g_vk.lim.maxSamplerAnisotropy ? sp->aniso : g_vk.lim.maxSamplerAnisotropy) : 1,
            sp->cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, sp->cmp_func >= 1 && sp->cmp_func <= 8 ? (VkCompareOp)(sp->cmp_func - 1) : VK_COMPARE_OP_ALWAYS,
            sp->lod_min, mip ? sp->lod_max : 0.25f,
            border && g_vk.border && !std_bc ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT : (sp->border[3] > 0.5f ? (sp->border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK) : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK), VK_FALSE};
        if (si.maxLod < si.minLod) si.maxLod = si.minLod;
        if (g_nsamp_used >= NSAMP * 3 / 4) { VRED_ONCE("more than %d distinct samplers", NSAMP * 3 / 4); return g_samp[0].s; }
        if (vkCreateSampler(g_vk.dev, &si, NULL, &e->s)) { e->s = VK_NULL_HANDLE; VRED_ONCE("vkCreateSampler failed"); return VK_NULL_HANDLE; }
        e->p = *sp; g_nsamp_used++; e->custom = si.pNext != NULL; e->si = si; e->si.pNext = NULL; g_samp_last = e;
        return e->s;
    }
    return VK_NULL_HANDLE;
}
static int g_sv_alias_ok = 1;   // stage_set: 0 while the shader being bound reads integer textures through an integer variant (int_mod), so they keep integer views
// 1 unsigned, 2 signed: an integer format, whose texels a float-declared slot cannot read in Vulkan. A stencil aspect reads as unsigned
static int fmt_int_kind(VkFormat f, VkImageAspectFlags asp) {
    if (asp == VK_IMAGE_ASPECT_STENCIL_BIT) return 1;
    switch (f) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_A8B8G8R8_UINT_PACK32:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_S8_UINT: return 1;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_SINT: case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT: return 2;
        default: return 0;
    }
}
static int int_variants_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_INT_VARIANTS"); on = !(e && e[0] == '0'); } return on; }   // 0: off (A/B)
typedef struct sview { struct sview *next; uint64_t key; VkImageView v; VkFormat fmt; VkImageAspectFlags asp; VkImageViewCreateInfo ci; } sview;
static const sview *g_sv_last; static vk_tex *g_sv_x;   // the view sample_view gave last, and its texture (DM_VK_PS_DUMP)
static VkFormat g_sv_fmt; static VkImageAspectFlags g_sv_asp;   // the format and aspect of the view sample_view last returned (stage_set's integer-slot test)
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
        case 1: vt = b->arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; if (x->type != VK_IMAGE_TYPE_2D && !(b->arrayed && x->type == VK_IMAGE_TYPE_3D)) return VK_NULL_HANDLE; if ((x->d.samples > 1) != (b->ms != 0)) return VK_NULL_HANDLE; break;
        case 2: vt = VK_IMAGE_VIEW_TYPE_3D; if (x->type != VK_IMAGE_TYPE_3D) return VK_NULL_HANDLE; break;
        case 3: vt = b->arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE; layers_need = 6; if (x->type != VK_IMAGE_TYPE_2D || x->layers < 6 || x->w != x->h) return VK_NULL_HANDLE; break;
        default: return VK_NULL_HANDLE;
    }
    uint32_t base_l = v && v->set_levels ? v->base_level : 0, nl = v && v->set_levels && v->nlevels ? v->nlevels : x->levels - base_l;
    uint32_t layers_total = x->layers;
    if (b->arrayed && x->type == VK_IMAGE_TYPE_3D) layers_total = x->depth;
    uint32_t base_y = v && v->set_layers ? v->min_layer : 0, ny = v && v->set_layers && v->nlayers ? v->nlayers : layers_total - base_y;
    if (base_l >= x->levels) base_l = 0; if (base_l + nl > x->levels) nl = x->levels - base_l;
    if (base_y >= layers_total) base_y = 0; if (base_y + ny > layers_total) ny = layers_total - base_y;
    if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_3D) ny = 1;
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6; if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny -= ny % 6; if (ny < layers_need) return VK_NULL_HANDLE;
    // An integer texture in a slot the shader declares float: Maxwell's fetch returns the texel's bits and the shader takes them back with a bit cast (floatBitsToInt).
    // Vulkan leaves an integer image read through a float type undefined (the Apple GPU keeps the bits, docs/reference/shaders.md "Integer textures and images through
    // float declarations keep their bits"; Adreno is unchecked). A 32-bit integer format is viewed as the float format of the same size, which reads the stored bits, as the
    // Metal side does for buffer textures (nvn_mtl.m tbuf_view_format). The image has MUTABLE_FORMAT (vkx tex_create). 8- and 16-bit integers have no such float twin
    static int alias_on = -1; if (alias_on < 0) { const char *e = getenv("DM_VK_INT_FLOAT_VIEW"); alias_on = !(e && e[0] == '0'); }   // 0: the integer view as before (A/B)
    if (!storage && b->sfloat && alias_on && g_sv_alias_ok) {   // the fallback where the shader has no integer variant (int_mod)
        VkFormat ff = fmt == VK_FORMAT_R32_UINT || fmt == VK_FORMAT_R32_SINT ? VK_FORMAT_R32_SFLOAT : fmt == VK_FORMAT_R32G32_UINT || fmt == VK_FORMAT_R32G32_SINT ? VK_FORMAT_R32G32_SFLOAT :
                      fmt == VK_FORMAT_R32G32B32A32_UINT || fmt == VK_FORMAT_R32G32B32A32_SINT ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_UNDEFINED;
        static const VkFormat small_int[] = {VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
            VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT,
            VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT};
        int small = 0; for (size_t i = 0; i < sizeof small_int / sizeof *small_int; i++) small |= fmt == small_int[i];
        if (ff) { static int said; if (said < 4) { said++; VLOG("texture %llu (format %d) in a float-declared slot: viewed as format %d (same bits)", (unsigned long long)x->serial, fmt, ff); } fmt = ff; }
        else if (small)
            VRED_ONCE("texture %llu (format %d): an 8/16-bit integer texture read through a float declaration, which has no same-bits float view (reads undefined)", (unsigned long long)x->serial, fmt);
    }
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
    for (sview *c = x->sviews; c; c = c->next) if (c->key == key) { g_sv_fmt = c->fmt; g_sv_asp = c->asp; g_sv_last = c; g_sv_x = x; return c->v; }
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, vt, fmt, {swz_vk(sw[0]), swz_vk(sw[1]), swz_vk(sw[2]), swz_vk(sw[3])}, {asp, base_l, nl, base_y, ny}};
    VkImageView view;
    if (vkCreateImageView(g_vk.dev, &vi, NULL, &view)) { if (storage) { VRED_ONCE("storage view of texture %llu (format %d) failed", (unsigned long long)x->serial, fmt); return VK_NULL_HANDLE; } vi.format = x->fmt; if (vkCreateImageView(g_vk.dev, &vi, NULL, &view)) { VRED_ONCE("sampling view of texture %llu (format %d, view type %d) failed", (unsigned long long)x->serial, fmt, vt); return VK_NULL_HANDLE; } }
    sview *c = malloc(sizeof *c); c->key = key; c->v = view; c->fmt = vi.format; c->asp = asp; c->ci = vi; c->ci.pNext = NULL; g_sv_fmt = vi.format; g_sv_asp = asp; c->next = x->sviews; x->sviews = c; g_sv_last = c; g_sv_x = x;
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
static uint64_t g_cs_cur_hit;
static uint64_t g_n_part_buf, g_n_nopool_buf, g_n_part_vtx, g_n_nopool_vtx;   // bindings past a pool's end / in no pool (logged; see stage_set and the vertex input)
static uint32_t cb_word(int stage, int cb, uint32_t word, int *from_ubo) {
    *from_ubo = 0;
    if (cb == 2) return c2_word(stage, word * 4);
    if (cb >= 3 && cb <= 16) {
        const p_bind_buf *u = &S.ubo[stage][cb - 3];
        if (!u->addr || (uint64_t)word * 4 + 4 > u->size) return 0;
        *from_ubo = 1;
        { uint64_t a = u->addr + (uint64_t)word * 4, cs; if (g_ncsw && csw_hit(a, &cs)) { g_hstale_hit = 1; g_hstale_addr = a; g_cs_cur_hit = cs; } }
        return ub_word(u->addr + (uint64_t)word * 4);
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
    uint32_t alpha;                                     // the pixel shader's alpha-test variant is used (alpha_func != ALWAYS and the shader has a float colour 0)
    uint64_t imu[3], ims[3];                            // vertex, pixel, geometry: bindings retyped for integer textures (int_mod; stage_set's masks)
} pkey;
typedef struct pent { struct pent *next; pkey k; VkPipeline p; VkPipelineLayout pl; _Atomic int building; } pent;   // building: a worker is still making p (pipe_wait)
enum { NPIPE = 4096 };
static pent *g_pipe[NPIPE];
static VkPipelineCache g_pcache;
// The driver's pipeline cache, kept on disk so the compiled pipelines carry over between runs: without it the first 3D scene froze for 11-14 s on every start while
// some 140 pipelines compiled. The driver checks the data's header (its own vendor, device and cache UUID) and ignores data that is not its own, so each driver keeps
// a file of its own, named by its cache UUID ($DM_ROOT/vk_pipelines-<uuid>.bin): switching between the system driver and a custom one (vk_driver.c) no longer
// throws the other's pipelines away. A first run on a driver without its file takes the old shared vk_pipelines.bin, which only the driver that wrote it accepts. Saved every 300 presents while new pipelines appear (the host has no clean exit)
static uint64_t g_pcache_saved_n; static size_t g_pcache_saved_size;
static uint64_t g_n_cpipes_all(void);
static void pcache_path(char *p, size_t n) {
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(g_vk.pd, &pr);
    char u[33]; for (int i = 0; i < 16; i++) snprintf(u + 2 * i, 3, "%02x", pr.pipelineCacheUUID[i]);
    snprintf(p, n, "%s/vk_pipelines-%s.bin", getenv("DM_ROOT") ? getenv("DM_ROOT") : ".", u);
}
static void pcache_open(void) {
    if (g_pcache) return;
    char p[600]; pcache_path(p, sizeof p);
    void *data = NULL; size_t n = 0; FILE *f = fopen(p, "rb");
    if (!f) { char old[600]; snprintf(old, sizeof old, "%s/vk_pipelines.bin", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); f = fopen(old, "rb"); }   // the file before per-driver names
    if (f) { fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET); if (l > 0 && (data = malloc((size_t)l))) n = fread(data, 1, (size_t)l, f); fclose(f); }
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, n, data};
    if (vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_pcache)) { ci.initialDataSize = 0; ci.pInitialData = NULL; vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_pcache); n = 0; }
    VLOG("pipeline cache: %zu bytes loaded from %s", n, p);
    free(data);
    size_t now = 0; if (!vkGetPipelineCacheData(g_vk.dev, g_pcache, &now, NULL)) g_pcache_saved_size = now;   // what the driver holds once loaded: unchanged means nothing to save
}
// The save costs a frame when done in full: the cache's whole data (47 MB in Fire Emblem Fortune's Weave) copied out of the driver and written to storage, on the
// thread that presents. "made" counts pipelines requested, cache hits included, so with a warm cache every 300th frame rewrote the same file (a 52 ms frame every
// 10 s; logs_mem3/log_perf2.txt: 21 saves of 46,858,457 bytes). Now the driver is asked for the size alone first (no copy) and nothing is saved while it has not
// changed; when it has, the data is copied here and written by a thread of its own
static _Atomic int g_pcache_writing;
typedef struct { void *d; size_t n; uint64_t made; } pcache_job;
static void *pcache_write(void *arg) {
    pcache_job *j = arg; char p[600], t[620]; pcache_path(p, sizeof p); snprintf(t, sizeof t, "%s.tmp", p);
    FILE *f = fopen(t, "wb");
    if (f) { int ok = fwrite(j->d, 1, j->n, f) == j->n; ok &= fclose(f) == 0; if (ok && !rename(t, p)) VLOG("pipeline cache: %zu bytes saved (%llu pipelines made this run)", j->n, (unsigned long long)j->made); else VLOG("[RED] pipeline cache: the save to %s failed", p); }
    free(j->d); free(j); atomic_store(&g_pcache_writing, 0); return NULL;
}
static _Atomic uint64_t g_bg_n, g_bg_ns, g_n_pipe_wait_skip, g_n_pipe_waited, g_n_pipe_must;   // the pipeline workers' counts (below)
void vkd_pcache_tick(uint64_t frame) {
    if (frame % 300 == 0 && g_bg_n) {   // what the pipeline workers did in the last 300 frames
        uint64_t n = atomic_exchange(&g_bg_n, 0), ns = atomic_exchange(&g_bg_ns, 0), w = atomic_exchange(&g_n_pipe_waited, 0), sk = atomic_exchange(&g_n_pipe_wait_skip, 0), mu = atomic_exchange(&g_n_pipe_must, 0);
        VLOG("pipelines: %llu built over the last 300 frames, %.1f ms each; draws that waited for one %llu (%llu of them opaque, whole-screen or in passes with no depth image, which are never skipped), skipped until it was ready %llu", (unsigned long long)n, n ? ns / 1e6 / (double)n : 0.0, (unsigned long long)w, (unsigned long long)mu, (unsigned long long)sk);
    }
    uint64_t made = g_n_pipelines + g_n_cpipes_all();
    if (!g_pcache || frame % 300 || made == g_pcache_saved_n || atomic_load(&g_pcache_writing)) return;
    size_t n = 0; if (vkGetPipelineCacheData(g_vk.dev, g_pcache, &n, NULL) || !n) return;
    if (n == g_pcache_saved_size) { g_pcache_saved_n = made; return; }   // only cache hits since the last save: nothing new to write
    pcache_job *j = malloc(sizeof *j); void *d = malloc(n); if (!j || !d) { free(j); free(d); return; }
    if (vkGetPipelineCacheData(g_vk.dev, g_pcache, &n, d)) { free(j); free(d); return; }
    *j = (pcache_job){d, n, made}; atomic_store(&g_pcache_writing, 1);
    pthread_t th; if (pthread_create(&th, NULL, pcache_write, j)) { pcache_write(j); } else pthread_detach(th);
    g_pcache_saved_n = made; g_pcache_saved_size = n;
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
    uint32_t *o = malloc((nw + (size_t)n * 10) * 4); size_t no = 0; uint32_t bound = w[3], ids[32]; int k = 0;
    for (int l = 0; l < 32; l++) if (add >> l & 1) ids[k++] = bound++;
    for (size_t i = 0; i < nw; ) {
        uint32_t len = i < 5 ? 1 : w[i] >> 16;
        if (i == ep_at) { memcpy(o + no, w + i, len * 4); o[no] = (len + (uint32_t)n) << 16 | 15; no += len; for (int j = 0; j < n; j++) o[no++] = ids[j]; i += len; continue; }
        if (i == dec_at) { k = 0; for (int l = 0; l < 32; l++) if (add >> l & 1) { o[no++] = 4u << 16 | SPV_DECORATE; o[no++] = ids[k++]; o[no++] = DEC_LOCATION; o[no++] = (uint32_t)l; } }
        if (i == var_at) { for (int j = 0; j < n; j++) { o[no++] = 4u << 16 | SPV_VARIABLE; o[no++] = ptr; o[no++] = ids[j]; o[no++] = 3; } }
        memcpy(o + no, w + i, len * 4); no += len; i += len;
    }
    o[3] = bound;
    { const char *e = getenv("DM_VK_ZERO_OUT"); if (!(e && e[0] == '0')) { int nz = 0; o = spv_zero_outputs(o, &no, &nz); } }   // the added outputs are stored zero too
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, no * 4, o};
    VkShaderModule m; VkResult r = vkCreateShaderModule(g_vk.dev, &mi, NULL, &m); free(o);
    if (r) return vs->mod;
    vs->var[vs->nvar].add = add; vs->var[vs->nvar].m = m; vs->nvar++;
    VLOG("vertex shader %016llx: unwritten outputs added at locations %#x for its pixel shader", (unsigned long long)vs->key, add);
    return m;
}
// ---------- Integer textures in float-declared slots: shader variants ----------
// Maxwell's texture instructions return a texel's raw bits whatever the texture's format; the translator declares every sampled texture float and the shader takes the bits
// back with a bit cast (floatBitsToInt). Vulkan leaves an integer image read through a float type undefined, and the Adreno converts (Fire Emblem Fortune's Weave's light-probe
// relight, ee2444c53652b5a0, reads an RGBA8_UINT material texture through a float samplerCube and joins its four bytes into an ID). int_patch retypes the slots a draw binds
// integer textures to: their image type gets an integer sampled type (signed for SINT formats), the loads, access chains and image extracts that carry them follow, and every
// sample / fetch / gather from them returns an integer vector that an OpBitcast turns back into the float vector the rest of the shader reads (the same bits, as on Maxwell).
// Anything else that reaches such a value (a depth compare, a sparse read, a phi) makes the patch give up: the shader keeps its original module and a red line says so.
// slots: the binding numbers (vb) to retype in descriptor set `set`; sgn: which of them are signed. Returns the new words (malloc) or NULL
enum { SPV_OP_FUNCTION = 54, SPV_OP_LOAD = 61, SPV_OP_ACCESS_CHAIN = 65, SPV_OP_IB_ACCESS_CHAIN = 66, SPV_OP_COPY_OBJECT = 83, SPV_OP_SAMPLED_IMAGE = 86, SPV_OP_IMAGE = 100, SPV_OP_BITCAST = 124 };
static uint32_t *int_patch(const uint32_t *w, size_t nw, uint32_t set, const uint32_t *slots, const uint8_t *sgn, int nslot, size_t *out_n, const char **why) {
    uint32_t bound = w[3];
    uint32_t *def = calloc(bound, 4), *bind = calloc(bound, 4), *dset = calloc(bound, 4); uint8_t *hasb = calloc(bound, 1);
    uint32_t *o = NULL; size_t no = 0; *why = NULL;
    uint32_t tgt_var[64]; uint8_t tgt_sgn[64]; int ntgt = 0; size_t first_fn = 0, first_tgt = 0;
    uint32_t int_t[2] = {0, 0}, ivec4[2] = {0, 0};   // existing OpTypeInt 32 0/1 and their 4-vectors
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) { *why = "malformed"; goto out; }
        if (op == SPV_DECORATE && len >= 4 && w[i + 1] < bound) { if (w[i + 2] == DEC_BINDING) { bind[w[i + 1]] = w[i + 3]; hasb[w[i + 1]] = 1; } else if (w[i + 2] == DEC_SET) dset[w[i + 1]] = w[i + 3]; }
        else if (op >= 19 && op <= 39 && len >= 2 && w[i + 1] < bound) def[w[i + 1]] = (uint32_t)i;
        else if (op == SPV_CONSTANT && len >= 3 && w[i + 2] < bound) def[w[i + 2]] = (uint32_t)i;
        if (op == SPV_TYPE_INT && len == 4 && w[i + 2] == 32) int_t[w[i + 3] ? 1 : 0] = w[i + 1];
        if (op == SPV_OP_FUNCTION && !first_fn) first_fn = i;
        if (op == SPV_VARIABLE && len >= 4 && !first_fn && w[i + 2] < bound && hasb[w[i + 2]] && dset[w[i + 2]] == set)
            for (int k = 0; k < nslot; k++) if (bind[w[i + 2]] == slots[k] && ntgt < 64) { if (!first_tgt) first_tgt = i; tgt_sgn[ntgt] = sgn[k]; tgt_var[ntgt++] = w[i + 2]; }
        i += len;
    }
    for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == SPV_TYPE_VECTOR && len == 4 && w[i + 3] == 4) for (int s = 0; s < 2; s++) if (int_t[s] && w[i + 2] == int_t[s]) ivec4[s] = w[i + 1];
        i += len; }
    if (!ntgt) { *why = "no variable with those bindings"; goto out; }
    for (int s = 0; s < 2; s++) if (int_t[s] && def[int_t[s]] > first_tgt) { *why = "its integer type is declared after the variables"; goto out; }
    {
    // the new types, per target: I' (image, integer sampled type), SI' (sampled image of I'), the variable's pointer (to SI' / I' or an array of it), the element pointer
    struct { uint32_t var, ptr_old, pointee, elem, img, arr_len, sampled, img2, si2, arr2, ptr2, eptr2; uint8_t s; } t[64];
    const uint32_t bound0 = bound;   // ids of the input are below it (the kind table's size); bound grows as ids are handed out
    size_t cap = nw + 64 * 40 + 64;
    memset(t, 0, sizeof t);
    for (int k = 0; k < ntgt; k++) {
        uint32_t v = tgt_var[k]; t[k].var = v; t[k].s = tgt_sgn[k];
        size_t vi = 0; for (size_t i = 5; i < first_fn; ) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (op == SPV_VARIABLE && w[i + 2] == v) { vi = i; break; } i += len; }
        uint32_t p = w[vi + 1]; if (p >= bound || !def[p] || (w[def[p]] & 0xffff) != SPV_TYPE_POINTER) { *why = "a variable's type is not a pointer"; goto out; }
        uint32_t e = w[def[p] + 3]; t[k].ptr_old = p; t[k].pointee = e; t[k].arr_len = 0;
        if (e < bound && def[e] && (w[def[e]] & 0xffff) == SPV_TYPE_ARRAY) { t[k].arr_len = w[def[e] + 3]; e = w[def[e] + 2]; }
        t[k].elem = e; if (e >= bound || !def[e]) { *why = "unknown element type"; goto out; }
        uint32_t eop = w[def[e]] & 0xffff, img = eop == SPV_TYPE_SAMPLED_IMAGE ? w[def[e] + 2] : eop == SPV_TYPE_IMAGE ? e : 0;
        if (!img || img >= bound || !def[img]) { *why = "not an image slot"; goto out; }
        uint32_t st = w[def[img] + 2]; if (st >= bound || !def[st] || (w[def[st]] & 0xffff) != SPV_TYPE_FLOAT) { *why = "its sampled type is not float"; goto out; }
        t[k].img = img; t[k].sampled = eop == SPV_TYPE_SAMPLED_IMAGE;
    }
    o = malloc(cap * 4); memcpy(o, w, 20); no = 5;
    uint32_t *kind = calloc(bound0, 4);   // per id: 0 not retyped; 1 + target<<4 | sub: 1 variable pointer, 2 element pointer, 3 sampled image, 4 image
    #define KIND(id, tk, sub) (kind[id] = (uint32_t)(1 + ((tk) << 4) + (sub)))
    for (int k = 0; k < ntgt; k++) KIND(t[k].var, k, 1);
    for (size_t i = 5; i < nw; ) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (no + len + 64 * 12 + 8 > cap) { cap = cap * 2 + len + 1024; o = realloc(o, cap * 4); }
        if (i == first_tgt) {   // the integer scalar and 4-vector types, before the first retyped variable
            for (int s = 0; s < 2; s++) {
                int need = 0; for (int k = 0; k < ntgt; k++) need |= t[k].s == s;
                if (!need) continue;
                if (!int_t[s]) { int_t[s] = bound++; o[no++] = 4u << 16 | SPV_TYPE_INT; o[no++] = int_t[s]; o[no++] = 32; o[no++] = (uint32_t)s; }
                if (!ivec4[s]) { ivec4[s] = bound++; o[no++] = 4u << 16 | SPV_TYPE_VECTOR; o[no++] = ivec4[s]; o[no++] = int_t[s]; o[no++] = 4; }
            }
        }
        if (op == SPV_VARIABLE && len >= 4 && i < first_fn && w[i + 2] < bound0 && kind[w[i + 2]]) {   // a retyped variable: its new types just before it (an array's length
            int k = (int)((kind[w[i + 2]] - 1) >> 4);                                                   // constant may come after another target's variable)
            // types are declared once each (SPIR-V forbids duplicate image, sampled-image and pointer types): the image pair per (image, signedness), the rest per whole shape
            int dupi = -1, dup = -1;
            for (int q = 0; q < ntgt; q++) if (q != k && t[q].img2 && t[q].img == t[k].img && t[q].s == t[k].s) { dupi = q; if (t[q].ptr2 && t[q].arr_len == t[k].arr_len && t[q].sampled == t[k].sampled) { dup = q; break; } }
            if (dup >= 0) { t[k].img2 = t[dup].img2; t[k].si2 = t[dup].si2; t[k].arr2 = t[dup].arr2; t[k].ptr2 = t[dup].ptr2; t[k].eptr2 = t[dup].eptr2; }
            else {
                if (dupi >= 0) { t[k].img2 = t[dupi].img2; t[k].si2 = t[dupi].si2; }
                else {
                    const uint32_t *iw = w + def[t[k].img]; uint32_t il = iw[0] >> 16;
                    t[k].img2 = bound++; o[no] = iw[0]; o[no + 1] = t[k].img2; o[no + 2] = int_t[t[k].s]; memcpy(o + no + 3, iw + 3, (il - 3) * 4); no += il;
                    t[k].si2 = bound++; o[no++] = 3u << 16 | SPV_TYPE_SAMPLED_IMAGE; o[no++] = t[k].si2; o[no++] = t[k].img2;
                }
                uint32_t el = t[k].sampled ? t[k].si2 : t[k].img2;
                if (t[k].arr_len) { t[k].arr2 = bound++; o[no++] = 4u << 16 | SPV_TYPE_ARRAY; o[no++] = t[k].arr2; o[no++] = el; o[no++] = t[k].arr_len; }
                t[k].ptr2 = bound++; o[no++] = 4u << 16 | SPV_TYPE_POINTER; o[no++] = t[k].ptr2; o[no++] = SC_UNIFORM_CONSTANT; o[no++] = t[k].arr_len ? t[k].arr2 : el;
                if (t[k].arr_len) { t[k].eptr2 = bound++; o[no++] = 4u << 16 | SPV_TYPE_POINTER; o[no++] = t[k].eptr2; o[no++] = SC_UNIFORM_CONSTANT; o[no++] = el; }
            }
        }
        size_t at = no; memcpy(o + no, w + i, len * 4); no += len;
        if (op == SPV_VARIABLE && len >= 4 && w[i + 2] < bound0 && kind[w[i + 2]]) o[at + 1] = t[(kind[w[i + 2]] - 1) >> 4].ptr2;
        else if (i > first_fn) {
            uint32_t r = len >= 3 ? w[i + 2] : 0;
            #define TK(id) ((id) < bound0 && kind[id] ? (int)((kind[id] - 1) >> 4) : -1)
            #define SUB(id) ((kind[id] - 1) & 15)
            if ((op == SPV_OP_ACCESS_CHAIN || op == SPV_OP_IB_ACCESS_CHAIN) && len >= 4 && TK(w[i + 3]) >= 0) {
                int k = TK(w[i + 3]); if (SUB(w[i + 3]) != 1 || !t[k].arr_len || len != 5) { *why = "an access chain into it the patch does not know"; goto fail; }
                o[at + 1] = t[k].eptr2; KIND(r, k, 2);
            } else if (op == SPV_OP_LOAD && len >= 4 && TK(w[i + 3]) >= 0) {
                int k = TK(w[i + 3]); o[at + 1] = t[k].sampled ? t[k].si2 : t[k].img2; KIND(r, k, t[k].sampled ? 3 : 4);
            } else if (op == SPV_OP_COPY_OBJECT && len == 4 && TK(w[i + 3]) >= 0) {
                int k = TK(w[i + 3]), sub = SUB(w[i + 3]); o[at + 1] = sub == 3 ? t[k].si2 : sub == 4 ? t[k].img2 : sub == 2 ? t[k].eptr2 : t[k].ptr2; KIND(r, k, sub);
            } else if (op == SPV_OP_IMAGE && len == 4 && TK(w[i + 3]) >= 0) {
                int k = TK(w[i + 3]); o[at + 1] = t[k].img2; KIND(r, k, 4);
            } else if (op == SPV_OP_SAMPLED_IMAGE && len == 5 && TK(w[i + 3]) >= 0) {
                int k = TK(w[i + 3]); o[at + 1] = t[k].si2; KIND(r, k, 3);
            } else if ((op == 87 || op == 88 || op == 91 || op == 92 || op == 95 || op == 96) && len >= 5 && TK(w[i + 3]) >= 0) {
                // sample (implicit / explicit / projective), fetch, gather: an integer vector, then its bits as the float vector the shader reads
                int k = TK(w[i + 3]); uint32_t rt = w[i + 1];
                if (rt >= bound || !def[rt] || (w[def[rt]] & 0xffff) != SPV_TYPE_VECTOR || w[def[rt] + 3] != 4) { *why = "a read whose result is not a 4-vector"; goto fail; }
                uint32_t r2 = bound++; o[at + 1] = ivec4[t[k].s]; o[at + 2] = r2;
                o[no++] = 4u << 16 | SPV_OP_BITCAST; o[no++] = rt; o[no++] = r; o[no++] = r2;
            } else if (op >= 103 && op <= 107) {   // size, levels, samples, LOD queries: their results do not depend on the sampled type
            } else if (op == 5 || op == 6 || op == SPV_DECORATE || op == 72) {   // names, decorations
            } else {
                for (uint32_t q = 1; q < len; q++) if (w[i + q] < bound0 && kind[w[i + q]] && TK(w[i + q]) >= 0) { static char b[96]; snprintf(b, sizeof b, "an instruction it does not know uses it (op %u)", op); *why = b; goto fail; }
            }
        }
        i += len;
    }
    free(kind); o[3] = bound; *out_n = no; goto out;
    fail: free(kind); free(o); o = NULL;
    }
out:
    free(def); free(bind); free(dset); free(hasb);
    return o;
}

// The module of shader sh with the bindings in u (unsigned) and s (signed) retyped as integer slots (int_patch), cached per mask; sh->mod when there is none to make
static VkShaderModule int_mod(shd *sh, uint64_t u, uint64_t sg) {
    if (!(u | sg)) return sh->mod;
    for (int i = 0; i < sh->nivar; i++) if (sh->ivar[i].u == u && sh->ivar[i].s == sg) return sh->ivar[i].m;
    if (!sh->words || sh->nivar == 8 || sh->int_fail) return sh->mod;
    uint32_t slots[64]; uint8_t sgn[64]; int n = 0;
    for (int i = 0; i < sh->nb && i < 64; i++) if ((u | sg) >> i & 1) { slots[n] = sh->b[i].vb; sgn[n] = (uint8_t)(sg >> i & 1); n++; }
    uint32_t set = sh->stage == 1 ? 1 : sh->stage == 2 ? 2 : 0; size_t on = 0; const char *why = NULL;
    uint32_t *o = int_patch(sh->words, sh->nwords, set, slots, sgn, n, &on, &why);
    VkShaderModule m = VK_NULL_HANDLE;
    if (o) { VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, on * 4, o};
             if (vkCreateShaderModule(g_vk.dev, &mi, NULL, &m)) { m = VK_NULL_HANDLE; why = "vkCreateShaderModule refused it"; }
             if (getenv("DM_VK_DUMP_SPV")) { char d[600], q[700]; snprintf(d, sizeof d, "%s/spv_patched", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); mkdir(d, 0755);
                 snprintf(q, sizeof q, "%s/%016llx_i%llx_%llx.spv", d, (unsigned long long)sh->key, (unsigned long long)u, (unsigned long long)sg); FILE *f = fopen(q, "wb"); if (f) { fwrite(o, 4, on, f); fclose(f); } }
             free(o); }
    if (!m) { sh->int_fail = 1; VLOG("[RED] shader %016llx: no integer-slot variant (bindings %#llx unsigned, %#llx signed): %s; its integer textures read through float views", (unsigned long long)sh->key, (unsigned long long)u, (unsigned long long)sg, why ? why : "?"); return sh->mod; }
    sh->ivar[sh->nivar].u = u; sh->ivar[sh->nivar].s = sg; sh->ivar[sh->nivar].m = m; sh->nivar++;
    VLOG("shader %016llx: integer-slot variant for bindings %#llx unsigned, %#llx signed", (unsigned long long)sh->key, (unsigned long long)u, (unsigned long long)sg);
    return m;
}

// ---------- The pipeline list ($DM_ROOT/vk_pipeline_list.bin; DM_VK_PIPE_LIST=0 turns it off) ----------
// Every graphics and compute pipeline the game asks for, recorded once across runs: the recipe to build it again before play (a prebuild at start, and a list shipped
// with the game profile, the Vulkan counterpart of the Mac's pso_manifest.tsv). The Switch 2 never compiles at run time (its shaders ship as machine code and NVN binds
// state separately); here every new shader + state combination is a pipeline the driver compiles (about 28 ms on the Adreno 830), so first visits stuttered.
// File: header {"DMPL", version, sizeof(pkey)}, then records {kind, vs, ps, gs, body}: kind 1 graphics (body: the pkey with its shader pointers cleared), kind 2
// compute (vs = the compute shader, body: its two integer-slot masks). A header of another version or key size is set aside as .old and a new list started
enum { PLIST_VERSION = 1 };
typedef struct { uint32_t kind, pad; uint64_t vs, ps, gs; } plist_rec;
static FILE *g_pl_f; static int g_pl_on = -1; static uint64_t *g_pl_set; static size_t g_pl_n, g_pl_cap; static uint64_t g_pl_new;
static uint64_t plist_hash(const plist_rec *r, const void *body, size_t n) {
    uint64_t h = 1469598103934665603ull; const uint8_t *p = (const uint8_t *)r;
    for (size_t i = 0; i < sizeof *r; i++) h = (h ^ p[i]) * 1099511628211ull;
    p = body; for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h ? h : 1;
}
static int plist_insert(uint64_t h) {   // 1 when new
    if ((g_pl_n + 1) * 2 > g_pl_cap) {
        size_t nc = g_pl_cap ? g_pl_cap * 2 : 4096; uint64_t *ns = calloc(nc, sizeof *ns); if (!ns) return 0;
        for (size_t i = 0; i < g_pl_cap; i++) if (g_pl_set[i]) { size_t j = g_pl_set[i] & (nc - 1); while (ns[j]) j = (j + 1) & (nc - 1); ns[j] = g_pl_set[i]; }
        free(g_pl_set); g_pl_set = ns; g_pl_cap = nc;
    }
    size_t j = h & (g_pl_cap - 1); while (g_pl_set[j]) { if (g_pl_set[j] == h) return 0; j = (j + 1) & (g_pl_cap - 1); }
    g_pl_set[j] = h; g_pl_n++; return 1;
}
static size_t plist_body(uint32_t kind) { return kind == 1 ? sizeof(pkey) : kind == 2 ? 2 * sizeof(uint64_t) : 0; }
static int plist_open(void) {
    if (g_pl_on >= 0) return g_pl_on;
    const char *e = getenv("DM_VK_PIPE_LIST"); g_pl_on = 0; if (e && e[0] == '0') return 0;
    char p[600]; snprintf(p, sizeof p, "%s/vk_pipeline_list.bin", getenv("DM_ROOT") ? getenv("DM_ROOT") : ".");
    uint32_t want[3] = {0x4c504d44u /* "DMPL" */, PLIST_VERSION, (uint32_t)sizeof(pkey)}, have[3] = {0, 0, 0};
    FILE *f = fopen(p, "rb"); size_t recs = 0;
    if (f) {
        if (fread(have, sizeof have, 1, f) == 1 && !memcmp(have, want, sizeof want)) {
            plist_rec r; uint8_t body[sizeof(pkey) + 16];
            while (fread(&r, sizeof r, 1, f) == 1) { size_t n = plist_body(r.kind); if (!n || fread(body, n, 1, f) != 1) break; plist_insert(plist_hash(&r, body, n)); recs++; }
            fclose(f);
        } else { fclose(f); char o[620]; snprintf(o, sizeof o, "%s.old", p); rename(p, o); VLOG("pipeline list: %s is of another version or key size: kept as %s, a new list started", p, o); }
    }
    g_pl_f = fopen(p, "ab");
    if (!g_pl_f) { VLOG("[RED] pipeline list: cannot write %s", p); return 0; }
    if (!recs && ftell(g_pl_f) == 0) fwrite(want, sizeof want, 1, g_pl_f);
    VLOG("pipeline list: %zu pipelines known from earlier runs (%s); new ones are added", recs, p);
    return g_pl_on = 1;
}
static void plist_add(uint32_t kind, uint64_t a, uint64_t b, uint64_t c, const void *body) {
    if (!plist_open()) return;
    plist_rec r = {kind, 0, a, b, c}; size_t n = plist_body(kind);
    if (!plist_insert(plist_hash(&r, body, n))) return;
    if (fwrite(&r, sizeof r, 1, g_pl_f) == 1 && fwrite(body, n, 1, g_pl_f) == 1) { fflush(g_pl_f); g_pl_new++; }
}
static void plist_add_graphics(const pkey *k) {
    pkey c = *k; c.vs = c.ps = c.gs = NULL;
    plist_add(1, k->vs ? k->vs->key : 0, k->ps ? k->ps->key : 0, k->gs ? k->gs->key : 0, &c);
}
static void plist_add_compute(const shd *cs, uint64_t imu, uint64_t ims) { uint64_t m[2] = {imu, ims}; plist_add(2, cs->key, 0, 0, m); }

// ---------- Graphics pipelines built by worker threads ----------
// A pipeline the driver has not compiled before takes about 28 ms on the Adreno 830, and a scene's first visit asks for dozens in one frame: Fire Emblem Fortune's
// Weave froze for up to 1.9 s after its tutorial (logs_mem3/log_perf1.txt: 343 pipelines, 10.2 s over the frame budget in 3.5 minutes; with the driver's cache warm,
// 0.2 s). The recording thread still prepares what is cheap (layout, shader variants) and hands the driver call to one of PIPE_WORKERS threads (Vulkan lets several
// threads create pipelines into one cache: it is not made externally synchronized). A draw waits for its pipeline while the frame's waiting budget lasts
// (DM_VK_PIPE_WAIT_MS, default 4), else it is skipped this frame and drawn once the pipeline is ready: on a first visit an object may appear a frame or two late, or
// leave a short-lived hole where the depth pre-pass and the G-buffer pass disagree. A draw in a pass with no depth image always waits (pipe_wait). DM_VK_ASYNC_PIPE=0 builds on the recording thread, as before. Compute pipelines
// stay synchronous: a skipped dispatch can leave lasting wrong results (exposure, histograms)
enum { PIPE_WORKERS = 3 };
typedef struct pjob {
    struct pjob *next; pent *e;
    VkPipelineShaderStageCreateInfo st[3]; VkPipelineVertexInputStateCreateInfo vi; VkPipelineInputAssemblyStateCreateInfo ia; VkPipelineViewportDepthClipControlCreateInfoEXT dcc;
    VkPipelineViewportStateCreateInfo vp; VkPipelineRasterizationStateCreateInfo rs; VkPipelineMultisampleStateCreateInfo ms; VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState ba[8]; VkPipelineColorBlendStateCreateInfo cb; VkPipelineDynamicStateCreateInfo dy; VkFormat cf[8]; VkPipelineRenderingCreateInfo ri;
    VkGraphicsPipelineCreateInfo gi;
} pjob;
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER, g_pdone_cv = PTHREAD_COND_INITIALIZER;
static pjob *g_pq_head, *g_pq_tail;
static int async_pipe_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_ASYNC_PIPE"); v = !(e && e[0] == '0'); } return v; }
static uint64_t pipe_wait_budget_ns(void) { static int64_t v = -1; if (v < 0) { const char *e = getenv("DM_VK_PIPE_WAIT_MS"); v = (int64_t)((e ? atof(e) : 4.0) * 1e6); if (v < 0) v = 0; } return (uint64_t)v; }
static void pjob_run(pjob *j) {
    uint64_t t0 = vkx_now_ns(); VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_pcache, 1, &j->gi, NULL, &p);
    g_bg_ns += vkx_now_ns() - t0; g_bg_n++;
    pthread_mutex_lock(&g_pq_mu);
    if (r) { p = VK_NULL_HANDLE; g_n_pipe_fail++; VLOG("[RED] pipeline for shaders %016llx / %016llx failed (%d)", (unsigned long long)j->e->k.vs->key, j->e->k.ps ? (unsigned long long)j->e->k.ps->key : 0ull, r); }
    j->e->p = p; atomic_store(&j->e->building, 0); pthread_cond_broadcast(&g_pdone_cv);
    pthread_mutex_unlock(&g_pq_mu);
    free(j);
}
static void *pipe_worker(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (!g_pq_head) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        pjob *j = g_pq_head; g_pq_head = j->next; if (!g_pq_head) g_pq_tail = NULL;
        pthread_mutex_unlock(&g_pq_mu);
        pjob_run(j);
    }
    return NULL;
}
static int pipe_workers_start(void) {
    static int ok = -1; if (ok >= 0) return ok;
    ok = 0; for (int i = 0; i < PIPE_WORKERS; i++) { pthread_t t; if (!pthread_create(&t, NULL, pipe_worker, NULL)) { pthread_detach(t); ok = 1; } }
    VLOG("pipelines: %s", ok ? "built by worker threads; a draw waits up to the frame's budget, then is skipped until its pipeline is ready (DM_VK_ASYNC_PIPE=0: on the recording thread)" : "[RED] no worker thread: built on the recording thread");
    return ok;
}
// a pipeline still being built: wait while this frame's budget lasts. Returns the entry once ready (NULL if its build failed), or NULL to skip the draw this frame
static int g_pipe_must;   // the draw being looked up is a whole-screen one (three or four vertices, no index buffer): set by the draw, read by pipe_wait
static pent *pipe_wait(pent *e) {
    static uint64_t frame = ~0ull, used;
    uint64_t f = vkx_frame(); if (f != frame) { frame = f; used = 0; }
    if (atomic_load(&e->building)) {
        uint64_t b = pipe_wait_budget_ns(), left = used < b ? b - used : 0;
        if ((!e->k.d_on && !e->k.s_on) || g_pipe_must) {   // a pass with no depth image: a full-screen step (lighting, combining, tone mapping) or the interface. Skipped, the whole picture is
                                           // black or stale for the frame (log_blackzoom.txt: up to 954 draws skipped in 300 frames with a cold cache, a black flash on first zoom), so it waits
            uint64_t t0 = vkx_now_ns();
            pthread_mutex_lock(&g_pq_mu); while (atomic_load(&e->building)) pthread_cond_wait(&g_pdone_cv, &g_pq_mu); pthread_mutex_unlock(&g_pq_mu);
            uint64_t d = vkx_now_ns() - t0; used += d; vkx_sb_time(3, d); g_n_pipe_waited++; g_n_pipe_must++;
        } else if (left) {
            uint64_t t0 = vkx_now_ns(); struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            uint64_t dl = (uint64_t)ts.tv_nsec + left; ts.tv_sec += (time_t)(dl / 1000000000ull); ts.tv_nsec = (long)(dl % 1000000000ull);
            pthread_mutex_lock(&g_pq_mu);
            while (atomic_load(&e->building)) if (pthread_cond_timedwait(&g_pdone_cv, &g_pq_mu, &ts)) break;
            pthread_mutex_unlock(&g_pq_mu);
            uint64_t d = vkx_now_ns() - t0; used += d; vkx_sb_time(3, d); g_n_pipe_waited++;
        }
        if (atomic_load(&e->building)) { g_n_pipe_wait_skip++; return NULL; }
    }
    return e->p ? e : NULL;
}
static pent *pipeline_of(const pkey *k) {
    uint32_t h = 2166136261u; const uint8_t *q = (const uint8_t *)k; for (size_t i = 0; i < sizeof *k; i++) h = (h ^ q[i]) * 16777619u;
    pent **b = &g_pipe[h & (NPIPE - 1)];
    for (pent *e = *b; e; e = e->next) if (!memcmp(&e->k, k, sizeof *k)) return atomic_load(&e->building) ? pipe_wait(e) : e->p ? e : NULL;
    pent *e = calloc(1, sizeof *e); e->k = *k; e->next = *b; *b = e;
    pcache_open();
    uint64_t pc0 = vkx_now_ns();
    e->pl = layout_of(k->vs, k->ps, k->gs); if (!e->pl) { g_n_pipe_fail++; return NULL; }
    plist_add_graphics(k);
    pjob *j = calloc(1, sizeof *j); if (!j) { g_n_pipe_fail++; return NULL; }
    j->e = e;
    VkPipelineShaderStageCreateInfo *st = j->st; uint32_t ns = 0;
    VkShaderModule vmod = k->vs->mod;
    if (k->ps && !k->gs && (k->ps->in_mask & ~k->vs->out_mask)) vmod = vs_variant(k->vs, k->ps->in_mask & ~k->vs->out_mask);
    if (k->imu[0] | k->ims[0]) {   // integer slots (int_mod) on top of the base module; a vertex shader needing extra outputs as well keeps those (rare: logged)
        if (vmod != k->vs->mod) VRED_ONCE("vertex shader %016llx needs both extra outputs and integer slots: integer slots left float", (unsigned long long)k->vs->key);
        else vmod = int_mod(k->vs, k->imu[0], k->ims[0]);
    }
    VkShaderModule pmod = k->ps ? (k->alpha && k->ps->alpha_mod ? k->ps->alpha_mod : k->ps->mod) : VK_NULL_HANDLE;
    if (k->ps && (k->imu[1] | k->ims[1])) {
        if (k->alpha && k->ps->alpha_mod) VRED_ONCE("pixel shader %016llx needs both the alpha test and integer slots: integer slots left float", (unsigned long long)k->ps->key);
        else pmod = int_mod(k->ps, k->imu[1], k->ims[1]);
    }
    st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vmod, "main", NULL};
    if (k->gs) st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_GEOMETRY_BIT, int_mod(k->gs, k->imu[2], k->ims[2]), "main", NULL};
    static const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo k_rsg64 = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 64};   // nvn_vk_mem.c "Subgroup size": pixel shaders too
    if (k->ps) st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, g_vk.sg_pin_fs ? &k_rsg64 : NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, pmod, "main", NULL};
    j->vi = (VkPipelineVertexInputStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    j->ia = (VkPipelineInputAssemblyStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, VK_FALSE};
    j->dcc = (VkPipelineViewportDepthClipControlCreateInfoEXT){VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, VK_TRUE};
    j->vp = (VkPipelineViewportStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip_m1 ? &j->dcc : NULL, 0, 1, NULL, 1, NULL};
    j->rs = (VkPipelineRasterizationStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, VK_FALSE,
        k->poly_mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : k->poly_mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1};
    j->ms = (VkPipelineMultisampleStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, VK_FALSE, 0, NULL, k->a2c, VK_FALSE};
    j->ds = (VkPipelineDepthStencilStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState *ba = j->ba;
    for (uint32_t i = 0; i < k->ncol; i++) {
        const blend_st *bs = &k->blend[i];
        ba[i] = (VkPipelineColorBlendAttachmentState){(k->blend_en >> i & 1) ? VK_TRUE : VK_FALSE, bf_vk(bs->src_rgb), bf_vk(bs->dst_rgb), beq_vk(bs->eq_rgb), bf_vk(bs->src_a), bf_vk(bs->dst_a), beq_vk(bs->eq_a),
                                                      (VkColorComponentFlags)(k->chmask >> (4 * i) & 15)};
        if (bs->eq_rgb == NVN_BLEND_EQUATION_MIN || bs->eq_rgb == NVN_BLEND_EQUATION_MAX) { ba[i].srcColorBlendFactor = ba[i].dstColorBlendFactor = VK_BLEND_FACTOR_ONE; }
        if (bs->eq_a == NVN_BLEND_EQUATION_MIN || bs->eq_a == NVN_BLEND_EQUATION_MAX) { ba[i].srcAlphaBlendFactor = ba[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; }
    }
    j->cb = (VkPipelineColorBlendStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, k->logic_op != NVN_LOGIC_OP_COPY, (VkLogicOp)k->logic_op, k->ncol, ba, {0, 0, 0, 0}};
    j->dy = (VkPipelineDynamicStateCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof k_dyn / sizeof *k_dyn - (g_vk.vi_dyn ? 0 : 1), k_dyn};
    for (uint32_t i = 0; i < k->ncol; i++) j->cf[i] = (VkFormat)k->col_fmt[i];
    j->ri = (VkPipelineRenderingCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, j->cf, k->d_on ? (VkFormat)k->ds_fmt : VK_FORMAT_UNDEFINED, k->s_on ? (VkFormat)k->ds_fmt : VK_FORMAT_UNDEFINED};
    j->gi = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &j->ri, 0, ns, j->st, &j->vi, &j->ia, NULL, &j->vp, &j->rs, &j->ms, &j->ds, &j->cb, &j->dy, e->pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    g_n_pipelines++;
    if (async_pipe_on() && pipe_workers_start()) {   // to a worker: wait for it within the frame's budget (pipe_wait)
        atomic_store(&e->building, 1);
        pthread_mutex_lock(&g_pq_mu); if (g_pq_tail) g_pq_tail->next = j; else g_pq_head = j; g_pq_tail = j; pthread_cond_signal(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
        vkx_sb_note(3, vkx_now_ns() - pc0);
        return pipe_wait(e);
    }
    pjob_run(j);   // on this thread (DM_VK_ASYNC_PIPE=0)
    vkx_sb_note(3, vkx_now_ns() - pc0);
    return e->p ? e : NULL;
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
static int g_snap_seq;   // DM_VK_SNAP_ALL: the matching draws so far (the flip-book's frame number)
static char g_snap_all_file[256];   // DM_ROOT/snap_all.txt, read when a trace is armed: the serials to flip-book, changeable between traces
static int in_serial_list(const char *list, uint64_t ser) {   // "5677,5735": is ser one of the comma-separated serials?
    while (list && *list) { while (*list == ',' || *list == ' ') list++; char *e; unsigned long long v = strtoull(list, &e, 10); if (e == list) break;
        if (v == ser) return 1; list = e; while (*list && *list != ',') list++; }
    return 0;
}
static void snap_after(sub_t *s, shd *ps) {
    static const char *want, *all_env, *said; static int init;
    if (!init) { init = 1; want = getenv("DM_VK_SNAP_PS"); all_env = getenv("DM_VK_SNAP_ALL"); if (want) VLOG("snapshots after pixel shaders %s while tracing", want); }
    const char *all = g_snap_all_file[0] ? g_snap_all_file : all_env;
    if (all && all != said) { said = all; VLOG("a snapshot of target 0 after every draw into %s while tracing: a flip-book of that pass (32 draws at most)", all); }
    if (!tracing() || (!want && !all)) return;
    uint64_t base = 0; int nt = 8;
    if (want && ps) { char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)ps->key); if (strstr(want, k)) base = 900000 + ps->key % 10000 * 10; }
    if (all) {   // DM_VK_SNAP_ALL=<serial>[,…]: a copy of target 0 after each draw into that target, named 950001, 950011, … in draw order
        vk_tex *z = g_rt.rt.n > 0 && g_rt.rt.color[0] ? tex_of(g_rt.rt.color[0], NULL) : NULL;
        if (z && in_serial_list(all, z->serial)) { base = 950000 + (uint64_t)(g_snap_seq % 30) * 10 + 1; nt = 1; g_snap_seq++; } }
    if (!base) return;
    for (int t = 0; t < g_rt.rt.n && t < nt; t++) {
        vk_tex *x = g_rt.rt.color[t] ? tex_of(g_rt.rt.color[t], NULL) : NULL; if (!x) continue;
        uint64_t ser = base + (uint64_t)t;
        int i = 0; while (i < g_nsnap && g_snap_ser[i] != ser) i++;
        if (i == g_nsnap) { if (g_nsnap == 32) return; g_nsnap++; }
        g_snap_ser[i] = ser; g_snap[i] = vkx_clone(s, x, 1);   // the clone ends the render pass; the next draw opens it again. Kept (a diagnostic leak, one traced frame): the dump runs after the submission's fence
        VLOG("snapshot of target %d (texture %llu) after ps %016llx: %s", t, (unsigned long long)x->serial, (unsigned long long)ps->key, g_snap[i] ? "taken" : "the copy could not be made");
    }
}
static vk_tex *g_trace_small[32]; static int g_trace_nsmall;   // tiny float textures the traced frame's draws sampled (the auto exposure's image): their texels are logged at its end
int vkd_trace_small(vk_tex **out, int max) { int n = g_trace_nsmall < max ? g_trace_nsmall : max; memcpy(out, g_trace_small, sizeof *out * (size_t)n); g_trace_nsmall = 0; return n; }
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
static long long g_trace_at = -2;   // DM_VK_TRACE=<frame>, or armed on demand by vkd_trace_arm (a DM_ROOT/trace_now file)
void vkd_trace_arm(void) { g_trace_at = (long long)g_presents; g_trace_lines = 0; g_trace_ntg = 0; g_trace_npass = 0;
    g_snap_all_file[0] = 0;   // DM_ROOT/snap_all.txt: the targets to flip-book, re-read per trace so a pass can be chosen at the scene, without a restart
    { const char *r = getenv("DM_ROOT") ? getenv("DM_ROOT") : "."; char p[600]; snprintf(p, sizeof p, "%s/snap_all.txt", r); FILE *f = fopen(p, "r");
      if (f) { size_t n = fread(g_snap_all_file, 1, sizeof g_snap_all_file - 1, f); g_snap_all_file[n] = 0; fclose(f);
          for (char *q = g_snap_all_file; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';
          VLOG("flip-book list from snap_all.txt: %s", g_snap_all_file); } }
    VLOG("trace armed on demand: tracing the next frame's draws (at present %llu)", (unsigned long long)g_presents); }
static int tracing(void) {
    if (g_cs_trace) return 1;
    if (g_trace_at == -2) { const char *e = getenv("DM_VK_TRACE"); g_trace_at = e ? atoll(e) : -1; }
    return g_trace_at >= 0 && g_presents == (uint64_t)g_trace_at && g_trace_lines < 60000;
}
int vkd_tracing(void) { return tracing(); }   // for the clears and copies (nvn_vk_exec.c), which the trace logs too
#define TRACE(...) do { if (tracing()) { g_trace_lines++; VLOG("trace " __VA_ARGS__); } } while (0)

// ---------- Descriptors for one stage ----------
static uint32_t g_draw_binst, g_dispatch_dim[3];
static uint64_t g_n_fbcopy, g_n_pool_desc;
static int fb_copy_on(void) { static int v = -1; if (v < 0) v = getenv("DM_VK_NO_FBCOPY") == NULL; return v; }
// The rectangle a clear covers: NVN clears inside the scissor, as Maxwell's clear with its scissor flag does and as the Mac backend does (nvn_mtl.m clr_scissor):
// the whole target when no scissor is set, else the scissor clamped to the target; 0 when that is empty (nothing is cleared). Set *partial when it is smaller than the target
int vkd_clear_rect(VkRect2D *r, int *partial) {
    uint32_t tw = g_rt.ext.width, th = g_rt.ext.height; *r = (VkRect2D){{0, 0}, {tw, th}}; *partial = 0;
    if (!S.has_sc) return tw && th;
    int64_t x0 = S.sc.x < 0 ? 0 : S.sc.x, y0 = S.sc.y < 0 ? 0 : S.sc.y, x1 = (int64_t)S.sc.x + S.sc.w, y1 = (int64_t)S.sc.y + S.sc.h;
    if (x1 > tw) x1 = tw; if (y1 > th) y1 = th;
    if (x1 <= x0 || y1 <= y0) return 0;
    *r = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    *partial = x0 > 0 || y0 > 0 || x1 < tw || y1 < th;
    return 1;
}
static vk_tex *g_simg[8]; static int g_nsimg;   // a dispatch's storage images (stage_set), for its DM_VK_RING_STAGES snapshots
static uint64_t g_n_sbuf_open;
// ---------- DM_VK_PS_DUMP=<pixel shader key>: everything one draw reads, for running it again outside the game ----------
// In a traced frame (the trace_now file), the first draw with that pixel shader has both stages' descriptor sets written out as Vulkan sees them: each buffer's bound
// range, each sampled texture (every level and layer, with the image's, the view's and the sampler's create parameters), as $DM_ROOT/psdump.txt (one line per item)
// and psdump.bin (the bytes). The copies are recorded before the draw, so they hold what the draw reads. The work folder's re/csrun/psrun_cap draws it again from these
// through a chosen driver. For Fire Emblem Fortune's Weave's lighting draw (88d7f48d93663021), whose shadow edges are stepped with Turnip and smooth with Qualcomm's
// driver while every compute shader of the frame gives both the same results
enum { PSD_MAX = 160 << 20, PSD_TXT = 1 << 20 };
typedef struct { uint32_t vb, elem; int storage, zero, zvt; vk_tex *x; VkImageViewCreateInfo vci; int has_s, custom; VkSamplerCreateInfo si; float bc[4]; } psd_img;
static uint64_t g_psd_key; static int g_psd_on; static long long g_psd_done_at = -3;
static psd_img g_psd_img[512]; static int g_psd_nimg;
static VkBuffer g_psd_buf; static VkDeviceMemory g_psd_mem; static uint8_t *g_psd_map; static uint64_t g_psd_n; static char *g_psd_txt; static size_t g_psd_tn;
static vk_tex *g_psd_tex[128]; static int g_psd_ntex;
#define PSD(...) do { if (g_psd_tn + 600 < PSD_TXT) g_psd_tn += (size_t)snprintf(g_psd_txt + g_psd_tn, PSD_TXT - g_psd_tn, __VA_ARGS__); } while (0)
static int psd_want(const shd *ps) {
    static int init; if (!init) { init = 1; const char *e = getenv("DM_VK_PS_DUMP"); g_psd_key = e ? strtoull(e, NULL, 16) : 0; if (g_psd_key) VLOG("DM_VK_PS_DUMP: the inputs of the first draw of pixel shader %016llx in a traced frame go to psdump.txt / psdump.bin", (unsigned long long)g_psd_key); }
    return g_psd_key && ps && ps->key == g_psd_key && tracing() && g_psd_done_at != g_trace_at;
}
static int psd_begin(void) {
    if (!g_psd_buf) {
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, PSD_MAX, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(g_vk.dev, &bci, NULL, &g_psd_buf)) { g_psd_buf = VK_NULL_HANDLE; return 0; }
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, g_psd_buf, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, g_vk.pool_type}; void *m = NULL;
        if (vkAllocateMemory(g_vk.dev, &mai, NULL, &g_psd_mem) || vkBindBufferMemory(g_vk.dev, g_psd_buf, g_psd_mem, 0) || vkMapMemory(g_vk.dev, g_psd_mem, 0, VK_WHOLE_SIZE, 0, &m)) { VLOG("[RED] DM_VK_PS_DUMP: no buffer"); g_psd_buf = VK_NULL_HANDLE; return 0; }
        g_psd_map = m; g_psd_txt = malloc(PSD_TXT);
    }
    g_psd_n = 0; g_psd_tn = 0; g_psd_ntex = 0; g_psd_nimg = 0; return g_psd_txt != NULL;
}
static void psd_note(const sbind *b, uint32_t k, int storage, int zeroed, const VkDescriptorImageInfo *d) {
    if (g_psd_nimg >= 512) return;
    psd_img *q = &g_psd_img[g_psd_nimg++]; memset(q, 0, sizeof *q); q->vb = b->vb; q->elem = k; q->storage = storage;
    q->zvt = b->dim == 0 ? (b->arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : b->dim == 2 ? VK_IMAGE_VIEW_TYPE_3D : b->dim == 3 ? (b->arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : (b->arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
    if (!d->imageView || zeroed || !g_sv_last || !g_sv_x) q->zero = d->imageView ? 1 : 2;   // 1: the host's zero view; 2: nothing bound
    else { q->x = g_sv_x; q->vci = g_sv_last->ci; }
    if (d->sampler && g_samp_last) { q->has_s = 1; q->si = g_samp_last->si; q->custom = g_samp_last->custom; memcpy(q->bc, g_samp_last->p.border, sizeof q->bc); }
}
static void psd_emit(sub_t *s, const shd *sh, const VkWriteDescriptorSet *wr, int nw) {
    end_rendering(); barrier_all(s->cb);
    PSD("SH %d %016llx imask %llx %llx\n", sh->stage, (unsigned long long)sh->key, (unsigned long long)sh->imask_u, (unsigned long long)sh->imask_s);
    for (int i = 0; i < nw; i++) {
        const VkWriteDescriptorSet *w = &wr[i];
        if (w->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || w->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            const VkDescriptorBufferInfo *d = w->pBufferInfo; uint64_t n = d->range == VK_WHOLE_SIZE ? ((uint64_t)sh->c1n + 255) & ~255ull : d->range; long long at = -1;
            if (d->buffer != g_zero && n && g_psd_n + n <= PSD_MAX) { VkBufferCopy bc = {d->offset, g_psd_n, n}; vkCmdCopyBuffer(s->cb, d->buffer, g_psd_buf, 1, &bc); at = (long long)g_psd_n; g_psd_n += (n + 15) & ~15ull; }
            PSD("B %d %u %c size %llu at %lld%s\n", sh->stage, w->dstBinding, w->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? 'U' : 'S', (unsigned long long)n, at, d->buffer == g_zero ? " zero" : "");
        } else for (uint32_t k = 0; k < w->descriptorCount; k++) {
            const psd_img *q = NULL; for (int j = 0; j < g_psd_nimg && !q; j++) if (g_psd_img[j].vb == w->dstBinding && g_psd_img[j].elem == k) q = &g_psd_img[j];
            if (!q) { PSD("I %d %u %u unknown\n", sh->stage, w->dstBinding, k); continue; }
            char sm[400] = "nosamp";
            if (q->has_s) snprintf(sm, sizeof sm, "samp %d %d %d %d %d %d %g %d %g %d %d %g %g %d custom %d %g %g %g %g", q->si.magFilter, q->si.minFilter, q->si.mipmapMode, q->si.addressModeU, q->si.addressModeV, q->si.addressModeW,
                                  q->si.mipLodBias, q->si.anisotropyEnable, q->si.maxAnisotropy, q->si.compareEnable, q->si.compareOp, q->si.minLod, q->si.maxLod, q->si.borderColor, q->custom, q->bc[0], q->bc[1], q->bc[2], q->bc[3]);
            char kind = q->storage ? 'X' : w->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 'C' : 'T';
            if (q->zero) { PSD("I %d %u %u %c zero %d vt %d %s\n", sh->stage, w->dstBinding, k, kind, q->zero, q->zvt, sm); continue; }
            vk_tex *x = q->x; int known = 0; for (int j = 0; j < g_psd_ntex; j++) known |= g_psd_tex[j] == x;
            if (!known && g_psd_ntex < 128) {
                g_psd_tex[g_psd_ntex++] = x;
                int dep = (x->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0; VkImageAspectFlags ca = dep ? VK_IMAGE_ASPECT_DEPTH_BIT : x->aspect;
                PSD("X %llu type %d fmt %d w %u h %u d %u layers %u levels %u samples %d aspect %u copied %u bpb %d bw %d bh %d :", (unsigned long long)x->serial, x->type, x->fmt, x->w, x->h, x->depth, x->layers, x->levels, x->d.samples, x->aspect, ca,
                    dep ? (x->fmt == VK_FORMAT_D16_UNORM ? 2 : 4) : x->bpb, dep ? 1 : x->bw, dep ? 1 : x->bh);
                for (uint32_t l = 0; l < x->levels && x->d.samples <= 1 && !(ca & VK_IMAGE_ASPECT_STENCIL_BIT); l++) {
                    uint32_t lw = level_dim(x->w, l), lh = level_dim(x->h, l), ld = level_dim(x->depth, l), bw = dep ? 1 : (uint32_t)x->bw, bh = dep ? 1 : (uint32_t)x->bh;
                    uint64_t n = (uint64_t)((lw + bw - 1) / bw) * ((lh + bh - 1) / bh) * (dep ? (x->fmt == VK_FORMAT_D16_UNORM ? 2u : 4u) : (uint32_t)x->bpb) * ld * x->layers;
                    if (!n || g_psd_n + n > PSD_MAX) { PSD(" L%u at -1 size %llu", l, (unsigned long long)n); continue; }
                    VkBufferImageCopy c = {g_psd_n, 0, 0, {ca, l, 0, x->layers}, {0, 0, 0}, {lw, lh, ld}};
                    vkCmdCopyImageToBuffer(s->cb, x->img, VK_IMAGE_LAYOUT_GENERAL, g_psd_buf, 1, &c);
                    PSD(" L%u at %llu size %llu", l, (unsigned long long)g_psd_n, (unsigned long long)n); g_psd_n += (n + 15) & ~15ull;
                }
                PSD("\n");
            }
            PSD("I %d %u %u %c tex %llu view %d %d swz %d %d %d %d asp %u lv %u %u ly %u %u %s\n", sh->stage, w->dstBinding, k, kind, (unsigned long long)x->serial, q->vci.viewType, q->vci.format,
                q->vci.components.r, q->vci.components.g, q->vci.components.b, q->vci.components.a, q->vci.subresourceRange.aspectMask, q->vci.subresourceRange.baseMipLevel, q->vci.subresourceRange.levelCount,
                q->vci.subresourceRange.baseArrayLayer, q->vci.subresourceRange.layerCount, sm);
        }
    }
    for (int i = 0; i < sh->nb; i++) if (sh->b[i].kind == BK_POOL_TEX || sh->b[i].kind == BK_POOL_SAMP) PSD("P %d %u pool array (not dumped)\n", sh->stage, sh->b[i].vb);
    barrier_all(s->cb); g_psd_nimg = 0;
}
static void psd_write(void *arg) {   // once the GPU has made the copies (vkx_defer)
    (void)arg; char q[700]; const char *root = getenv("DM_ROOT") ? getenv("DM_ROOT") : ".";
    snprintf(q, sizeof q, "%s/psdump.txt", root); FILE *f = fopen(q, "w"); if (f) { fwrite(g_psd_txt, 1, g_psd_tn, f); fclose(f); }
    snprintf(q, sizeof q, "%s/psdump.bin", root); f = fopen(q, "wb"); if (f) { fwrite(g_psd_map, 1, g_psd_n, f); fclose(f); }
    VLOG("DM_VK_PS_DUMP: %llu bytes and %llu bytes of index written to %s/psdump.bin / .txt", (unsigned long long)g_psd_n, (unsigned long long)g_psd_tn, root);
}
static int sbuf_open_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SBUF_EXACT"); v = !(e && e[0] == '1'); VLOG("storage buffers are bound %s", v ? "up to 1 MB past their size (their pool's end at most): a read past the size returns the memory there, as on the console" : "at their size only (DM_VK_SBUF_EXACT=1)"); } return v; }
static int stage_set(sub_t *s, shd *sh, VkDescriptorSet *out, int *missing_tex) {
    if (!sh->nb) { *out = VK_NULL_HANDLE; return 1; }
    VkDescriptorSet ds = set_alloc(s, sh->dsl); if (!ds) return 0;
    int stage = sh->stage;
    // integer textures in float slots (int_mod): which bindings hold unsigned / signed ones, and which hold float ones (an array of both is left as it is)
    uint64_t imu = 0, ims = 0, imf = 0; int ivar = int_variants_on() && sh->words && !sh->int_fail;
    uint64_t csw_new[2][32]; int ncsw_new = 0; if (stage == 5) g_cs_cur = sh->key;   // this dispatch's storage ranges (csw_add once its own handles are read)
    g_sv_alias_ok = !ivar;
    enum { NW = 64, NI = 256 };
    VkWriteDescriptorSet wr[NW]; VkDescriptorBufferInfo bi[NW]; VkDescriptorImageInfo ii[NI]; int nw = 0, nii = 0; int pools = 0;
    for (int i = 0; i < sh->nb && nw < NW; i++) {
        const sbind *b = &sh->b[i];
        VkWriteDescriptorSet *w = &wr[nw]; *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, b->vb, 0, b->count > 1024 ? 1 : b->count, b->dt};
        if (b->dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b->dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            VkDescriptorBufferInfo *d = &bi[nw]; *d = (VkDescriptorBufferInfo){g_zero, 0, 65536}; w->descriptorCount = 1;
            uint64_t range_max = b->dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? g_vk.lim.maxUniformBufferRange : 0x7fffffff;
            if (b->kind == BK_SUPPORT) *d = (VkDescriptorBufferInfo){g_support, 0, 65536};
            else if (b->kind == BK_ALPHA) {   // dm_alpha_state {uint func; float ref;}: the alpha test's function and reference, per draw (alpha_inject)
                VkBuffer ab; uint64_t ao; uint8_t *m = arena_alloc(s, 16, &ab, &ao);
                if (m) { uint32_t fn = S.color.alpha_func; memcpy(m, &fn, 4); memcpy(m + 4, &S.alpha_ref, 4); *d = (VkDescriptorBufferInfo){ab, ao, 16}; }
            }
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
                    if (stage == 5 && b->kind == BK_SBUF && ncsw_new < 32) { csw_new[0][ncsw_new] = u->addr; csw_new[1][ncsw_new] = u->size; ncsw_new++; }
                    uint64_t addr = u->addr, size = u->size;
                    if (b->kind == BK_SBUF) { uint64_t al = addr & ~15ull; size += addr - al; addr = al; }   // the shader indexes from the 16-byte aligned base (Ryujinx GlobalToStorage, host alignment 16)
                    // A storage buffer is bound from the boundary its shader was translated for, 64 bytes (vk_spirv.py SB_ALIGN, Program.cs sb_align: the shader reads at
                    // address - (address & -64)), not from the device's own minimum. Qualcomm's driver asks for 64, so the two agreed; Turnip asks for 4, and every
                    // storage buffer off a 64-byte boundary was bound 16, 32 or 48 bytes too far in and read shifted (its log: "alignment uniform 64 storage 4").
                    // A device asking for more than 64 keeps its own value (and the line below reports the shift)
                    uint64_t sb_al = g_vk.lim.minStorageBufferOffsetAlignment > 64 ? g_vk.lim.minStorageBufferOffsetAlignment : 64;
                    uint64_t al_req = b->kind == BK_SBUF ? sb_al : g_vk.lim.minUniformBufferOffsetAlignment;
                    uint64_t avail = 0; int part = 0;
                    if (!vk_pool_find(addr, size, &pb, &po) && vk_pool_find_part(addr, size, &pb, &po, &avail) && avail >= 16) {
                        // the range runs past its pool's end: bound to the end instead of the zero buffer (the shader reads the part it uses)
                        part = 1; size = avail; g_n_part_buf++;
                        if (g_n_part_buf <= 20) VLOG("[RED] %s buffer at %#llx+%#llx runs past its pool's end (%#llx bytes there): bound to the end, was the zero buffer; shader %016llx st%d slot %d, frame %llu",
                                                    b->kind == BK_SBUF ? "a storage" : "a uniform", (unsigned long long)addr, (unsigned long long)u->size, (unsigned long long)avail, (unsigned long long)sh->key, stage, b->slot, (unsigned long long)g_presents);
                        vkx_ring_note("BUF-PAST-POOL shader %016llx st%d slot %d addr %#llx size %#llx avail %#llx", (unsigned long long)sh->key, stage, b->slot, (unsigned long long)addr, (unsigned long long)u->size, (unsigned long long)avail);
                    } else if (!part && !vk_pool_find(addr, size, &pb, &po)) {
                        g_n_nopool_buf++;
                        if (g_n_nopool_buf <= 20) VLOG("[RED] %s buffer at %#llx+%#llx is in no GPU pool: the zero buffer is bound; shader %016llx st%d slot %d, frame %llu",
                                                       b->kind == BK_SBUF ? "a storage" : "a uniform", (unsigned long long)addr, (unsigned long long)u->size, (unsigned long long)sh->key, stage, b->slot, (unsigned long long)g_presents);
                        vkx_ring_note("BUF-NO-POOL shader %016llx st%d slot %d addr %#llx size %#llx", (unsigned long long)sh->key, stage, b->slot, (unsigned long long)addr, (unsigned long long)u->size);
                    }
                    if ((part || vk_pool_find(addr, size, &pb, &po)) && b->kind != BK_SBUF && (po % al_req)) {
                        // A uniform buffer the device cannot bind where it lies (minUniformBufferOffsetAlignment, 64 on the Adreno; NVN allows less): bound through an aligned
                        // copy of its bytes as the GPU reads them here (ub_copy). It was bound at the lower boundary, so the shader read its constants shifted by up to 63
                        // bytes. Storage buffers are offset in the shader instead (Program.cs sb_align). Fire Emblem Fortune's Weave: the constant ring's position moves
                        // with each frame's draws, so a frame with more of them (a shadow cascade refresh) put the floor's G-buffer constants off the boundary, and the floor
                        // was drawn with wrong constants: missing from the G-buffer but for stripes, in single frames (the stage capture, logs_mem3/001546)
                        uint64_t n = size < range_max ? size : range_max; VkBuffer ab; uint64_t ao; uint8_t *m = arena_alloc(s, n, &ab, &ao);
                        if (m) { ub_copy(m, addr, n); *d = (VkDescriptorBufferInfo){ab, ao, n}; }
                        if (g_n_ubo_copy++ < 20) VLOG("uniform buffer at %#llx+%#llx is %llu bytes past a %llu-byte boundary in its pool: bound through an aligned copy; shader %016llx st%d slot %d, frame %llu",
                                                     (unsigned long long)addr, (unsigned long long)u->size, (unsigned long long)(po % al_req), (unsigned long long)al_req, (unsigned long long)sh->key, stage, b->slot, (unsigned long long)g_presents);
                        vkx_ring_note("UBO-COPY shader %016llx st%d slot %d addr %#llx +%llu size %#llx", (unsigned long long)sh->key, stage, b->slot, (unsigned long long)addr, (unsigned long long)(po % al_req), (unsigned long long)u->size);
                    } else if (part || vk_pool_find(addr, size, &pb, &po)) {
                        if (po % al_req && !(b->kind == BK_SBUF && al_req == 64)) VRED_ONCE("%s buffer at %#llx is not %llu-byte aligned in its pool: bound at the lower boundary (reads shifted)", b->kind == BK_SBUF ? "a storage" : "a uniform", (unsigned long long)addr, (unsigned long long)al_req);
                        uint64_t aligned = po - po % al_req; size += po - aligned;
                        if (b->kind == BK_SBUF && sbuf_open_on()) {
                            // A storage buffer is bound past its size, up to 1 MB or its pool's end (DM_VK_SBUF_EXACT=1: its size only, as before). On the console a shader
                            // reaches a storage buffer through its address and nothing checks the size given at the bind: a read past it returns the memory that lies
                            // there. Here the size became the descriptor's range, and with robust buffer access a read past it gives zeros or any value of the range, which
                            // may differ from one shader to the next. Fire Emblem Fortune's Weave's skinned meshes read their bone matrices this way in three vertex
                            // shaders (depth pre-pass 14c868df02293941 and its like, the lit pass's 89490faf74ddfb4f / 388febb65aa9504f); the bone tables they are given
                            // are byte for byte the same within a frame (logs_mem3/001824), yet the pre-pass threw single corners far off, to one point of the scene: black
                            // wedges where the G-buffer then had nothing (logs_mem3/001476). Open: that a read past the table is what differs has not been measured
                            VkBuffer b2; uint64_t o2, av = 0;
                            if (vk_pool_find_part(addr, 1ull << 40, &b2, &o2, &av) && b2 == pb) { uint64_t open = av + (po - aligned); if (open > size + (1u << 20)) open = size + (1u << 20); if (open > size) { size = open; g_n_sbuf_open++; } }
                        }
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
                if (g_psd_on) { g_sv_last = NULL; g_sv_x = NULL; g_samp_last = NULL; }
                int tid = 0, sid = 0, zeroed = 0; treg_t r; samp_par sp;
                if (b->kind == BK_TEX || b->kind == BK_IMG) binding_ids(stage, b, k, &tid, &sid);
                if (g_hstale_hit) {   // the handle came from memory a dispatch earlier in this submission writes, before the GPU ran it
                    g_hstale_hit = 0; g_hstale_n++;
                    if (g_hstale_n <= 50) VLOG("[RED] stale texture handle: shader %016llx st%d b%u [%u] read its handle at %#llx, which compute shader %016llx (recorded earlier in this submission, not run yet) writes; frame %llu, tid %d",
                                               (unsigned long long)sh->key, stage, b->vb, k, (unsigned long long)g_hstale_addr, (unsigned long long)g_cs_cur_hit, (unsigned long long)g_presents, tid);
                    vkx_ring_note("HANDLE-STALE shader %016llx st%d b%u [%u] from cs %016llx tid %d", (unsigned long long)sh->key, stage, b->vb, k, (unsigned long long)g_cs_cur_hit, tid);
                }
                int have_sp = b->dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && sreg_get(sid, &sp);
                if (b->dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) d->sampler = sampler_of(have_sp ? &sp : NULL);
                if (tid && treg_get(tid, &r)) {
                    vk_tex *x = tex_for_read(&r.t, s);
                    if (x && !storage && g_rt.has_rt && fb_copy_on()) {   // one of the pass's own images: read a copy (vkx_clone)
                        int att = g_rt.rt.depth && tex_of(g_rt.rt.depth, NULL) == x;
                        for (int a = 0; a < g_rt.rt.n && a < 8 && !att; a++) att = g_rt.rt.color[a] && tex_of(g_rt.rt.color[a], NULL) == x;
                        if (att) { vk_tex *c = vkx_clone(s, x, 0); if (c) { x = c; g_n_fbcopy++; vkx_sublog(s, " F%llu", (unsigned long long)c->serial); } }
                    }
                    if (x && (!storage || x->d.samples <= 1)) d->imageView = sample_view(x, &r, b, storage);
                    if (storage && stage == 5 && x && g_nsimg < 8) g_simg[g_nsimg++] = x;   // DM_VK_RING_STAGES: the images this dispatch writes (snapshot after it)
                    if (ivar && d->imageView && !storage && b->sfloat && b->dim != 5 && i < 64) {
                        int kd = fmt_int_kind(g_sv_fmt, g_sv_asp);
                        if (!kd) imf |= 1ull << i;
                        else {
                            if (kd == 2) ims |= 1ull << i; else imu |= 1ull << i;
                            if (b->dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {   // integer formats are not filtered (Vulkan requires a nearest sampler; Maxwell does not filter them)
                                samp_par np; if (have_sp) np = sp; else { memset(&np, 0, sizeof np); np.ws = np.wt = np.wr = NVN_WRAP_MODE_CLAMP_TO_EDGE; np.lod_max = 1000; }
                                np.magf = NVN_MAG_FILTER_NEAREST; np.minf = np.minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : NVN_MIN_FILTER_NEAREST;
                                np.aniso = 1; np.cmp_mode = NVN_COMPARE_MODE_NONE; d->sampler = sampler_of(&np);
                            }
                        }
                    }
                    if (x && !storage && tracing() && (uint64_t)x->w * x->h <= 16 && x->type == VK_IMAGE_TYPE_2D &&
                        (x->fmt == VK_FORMAT_R16_SFLOAT || x->fmt == VK_FORMAT_R32_SFLOAT || x->fmt == VK_FORMAT_R16G16_SFLOAT || x->fmt == VK_FORMAT_R16G16B16A16_SFLOAT)) {
                        int dup = 0; for (int q = 0; q < g_trace_nsmall && !dup; q++) dup = g_trace_small[q] == x;
                        if (!dup && g_trace_nsmall < 32) g_trace_small[g_trace_nsmall++] = x;
                    }
                    if (x && x->bpb == 16 && x->bw > 1 && tracing()) {   // all-zero blocks (ASTC's magenta) in what the image holds against the guest storage now; once per texture per trace
                        static uint64_t seen[1024]; static long long seen_at = -3; static int nseen;
                        if (seen_at != g_trace_at) { seen_at = g_trace_at; nseen = 0; }
                        int dup = 0; for (int q = 0; q < nseen && !dup; q++) dup = seen[q] == x->serial;
                        if (!dup && nseen < 1024) { seen[nseen++] = x->serial; uint32_t z, nb, z0, n0; vkx_zero_blocks_now(x, &z, &nb, &z0, &n0);
                            TRACE("  zero blocks ser %llu fmt %#x %ux%u %ul/%uy: at upload %u/%u (level 0 %u/%u), now %u/%u (level 0 %u/%u); uploads %llu, last in submission %llu (now %llu)%s",
                                  (unsigned long long)x->serial, x->d.format, x->w, x->h, x->levels, x->layers, x->up_zero, x->up_blocks, x->up_l0_zero, x->up_l0_blocks, z, nb, z0, n0,
                                  (unsigned long long)x->up_n, (unsigned long long)x->up_sub, (unsigned long long)g_fence + 1, x->gpu_owned ? ", GPU-owned" : ""); } }
                }
                if (!d->imageView && !storage) {   // nothing Vulkan can put in this slot (a 3D texture in a 1D array slot, an entry the game left empty):
                    VkImageViewType zvt = b->dim == 0 ? (b->arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : b->dim == 2 ? VK_IMAGE_VIEW_TYPE_3D :
                                          b->dim == 3 ? (b->arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) :
                                          (b->arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
                    d->imageView = vkx_zero_view(zvt); if (d->imageView) zeroed = 1;   // a null descriptor here reads as one flat colour
                }
                if (!d->imageView) (*missing_tex)++;
                if (g_psd_on) psd_note(b, k, storage, zeroed, d);
                TRACE("  st%d b%u %s dim %d arr %d kind %d cbs %d handle %#x [%u]: tid %d sid %d reg %d ser %llu %ux%ux%u fmt %#x tgt %d mem %p -> view %s", stage, b->vb, storage ? "storage image" : "texture", b->dim, b->arrayed, b->kind, b->cbs, (unsigned)b->handle, k, tid, sid,
                      tid ? treg_get(tid, &r) : 0, tid && treg_get(tid, &r) ? (unsigned long long)r.t.serial : 0ull, tid && treg_get(tid, &r) ? r.t.d.width : 0, tid && treg_get(tid, &r) ? r.t.d.height : 0, tid && treg_get(tid, &r) ? r.t.d.depth : 0,
                      tid && treg_get(tid, &r) ? r.t.d.format : 0, tid && treg_get(tid, &r) ? r.t.d.target : -1, tid && treg_get(tid, &r) ? (void *)r.t.mem : NULL, d->imageView ? (zeroed ? "zero" : "ok") : "none");
                if (have_sp) TRACE("    sampler %d: filter min %d mag %d, wrap %d %d %d, compare %d/%d, lod %.2f to %.2f bias %.2f, aniso %.1f, border %.3f %.3f %.3f %.3f", sid, sp.minf, sp.magf, sp.ws, sp.wt, sp.wr,
                                   sp.cmp_mode, sp.cmp_func, (double)sp.lod_min, (double)sp.lod_max, (double)sp.lod_bias, (double)sp.aniso, (double)sp.border[0], (double)sp.border[1], (double)sp.border[2], (double)sp.border[3]);
            }
            if (!g_vk.null_desc) for (uint32_t k = 0; k < n; k++) if (!ii[nii - n + k].imageView) { VRED_ONCE("a texture binding has no texture and the driver has no null descriptors: draw skipped"); return 0; }
        } else if (b->kind == BK_POOL_TEX || b->kind == BK_POOL_SAMP) { pools = 1; continue; }   // written below, element by element
        else continue;   // other separate samplers and texel buffers: left unwritten
        nw++;
    }
    if (ivar) {   // the integer-slot masks the pipeline takes (pkey / cpipe_of), and the variant made now, so that a shader it cannot be made for falls back from the next draw
        uint64_t mixed = ((imu | ims) & imf) | (imu & ims);
        if (mixed) VRED_ONCE("shader %016llx: a texture array holds integer and float textures (bindings %#llx): left float", (unsigned long long)sh->key, (unsigned long long)mixed);
        sh->imask_u = imu & ~mixed; sh->imask_s = ims & ~mixed;
        if ((sh->imask_u | sh->imask_s) && int_mod(sh, sh->imask_u, sh->imask_s) == sh->mod) sh->imask_u = sh->imask_s = 0;
    } else sh->imask_u = sh->imask_s = 0;
    uint64_t dw0 = vkx_now_ns(); vkUpdateDescriptorSets(g_vk.dev, (uint32_t)nw, wr, 0, NULL); vkx_sb_note(2, vkx_now_ns() - dw0);
    if (pools) {   // the pool arrays: one descriptor per registered ID of the bound pool (the rest stay empty)
        g_sv_alias_ok = 1;   // pool arrays are never retyped (int_mod): their 32-bit integer textures keep the same-bits float view
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
        if (np) { uint64_t dw0 = vkx_now_ns(); vkUpdateDescriptorSets(g_vk.dev, np, pw, 0, NULL); vkx_sb_note(2, vkx_now_ns() - dw0); }
        g_n_pool_desc += np;
        free(pw); free(pi);
    }
    for (int i = 0; i < ncsw_new; i++) csw_add(csw_new[0][i], csw_new[1][i]);
    if (g_psd_on) psd_emit(s, sh, wr, nw);
    *out = ds;
    return 1;
}

// ---------- The draw ----------
static PFN_vkCmdSetVertexInputEXT p_set_vi;
static uint64_t g_n_missing_tex;
static VkFormat dummy_attr_fmt(uint8_t kind) { return kind == 1 ? VK_FORMAT_R32G32B32A32_SINT : kind == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT; }
// ---------- DM_VK_BISECT=1: which draw makes a fault ----------
// $DM_ROOT/draw_skip holds two hexadecimal numbers, lo and hi: draws whose signature (a hash of the vertex shader, the pixel shader and the element count) lies in
// [lo, hi) are not drawn. The file is read again every 8 frames, so a script on the PC halves the range while the game holds a view that shows the fault, taking a frame
// shot at each step ($DM_ROOT/shot_now, nvn_vk_exec.c): Fire Emblem Fortune's Weave's black slivers across the sky at some camera angles, with the same draw list as a
// clean view (logs_mem3/005398). The signature is the same from frame to frame, which a draw's index is not (the shadow cascades take turns).
static uint32_t g_bis_lo, g_bis_hi; static uint64_t g_n_bis_skip;
uint32_t vkd_draw_sig(uint64_t vs, uint64_t ps, uint32_t count) {
    uint64_t h = vs * 0x9E3779B97F4A7C15ull ^ ps * 0xC2B2AE3D27D4EB4Full ^ (uint64_t)count * 0x165667B19E3779F9ull; h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32; return (uint32_t)h;
}
void vkd_bisect_tick(uint64_t n) {
    static int on = -1; static char path[600];
    if (on < 0) { const char *e = getenv("DM_VK_BISECT"); on = e && e[0] == '1'; snprintf(path, sizeof path, "%s/draw_skip", getenv("DM_ROOT") ? getenv("DM_ROOT") : "."); if (on) VLOG("DM_VK_BISECT: draws named by %s are skipped", path); }
    if (!on || n % 8) return;
    uint32_t lo = 0, hi = 0; FILE *f = fopen(path, "r"); if (f) { if (fscanf(f, "%x %x", &lo, &hi) != 2) lo = hi = 0; fclose(f); }
    if (lo != g_bis_lo || hi != g_bis_hi) { VLOG("DM_VK_BISECT: frame %llu, skipping signatures %08x to %08x (%llu draws skipped so far)", (unsigned long long)n, lo, hi, (unsigned long long)g_n_bis_skip); g_bis_lo = lo; g_bis_hi = hi; }
}
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
    if (g_bis_hi) { uint32_t sg = vkd_draw_sig(vs->key, ps ? ps->key : 0, (uint32_t)(dr ? dr->count : di->count)); if (sg >= g_bis_lo && sg < g_bis_hi) { g_n_bis_skip++; return; } }   // DM_VK_BISECT
    g_draw_binst = (uint32_t)(dr ? dr->base_instance : di->base_instance);
    {   // DM_VK_RING_VS=<key>[,<key>...] (with DM_VK_RING): the uniform and storage buffers bound to the vertex stage at each draw of those vertex shaders, into the rolling
        // capture as the compute dumps are (vkx_ring_csdump; x y z: the draw's element count, 0, 0). Fire Emblem Fortune's Weave's skinned meshes: their depth pre-pass
        // throws single corners far off in some frames (black wedges where the G-buffer then has nothing), and the bone matrices they read are the first thing to look at
        static const char *want; static int init; if (!init) { init = 1; want = getenv("DM_VK_RING_VS"); }
        if (want) { char k[17]; snprintf(k, sizeof k, "%016llx", (unsigned long long)vs->key); if (strstr(want, k)) {
            end_rendering(); vkx_ring_csdump2(s, want, vs->key, (int)(dr ? dr->count : di->count), 0, 0, S.ubo[0], S.ssbo[0]);
            // DM_VK_RING_VTX=1: the draw's vertex streams (as the 'U' entries of a second dump: U i = stream i) and its index buffer (S 0), with the attribute layout in
            // the draw list ("VTX ..."), so the vertex shader can be run again by hand on the very vertices
            static int vtx = -1; if (vtx < 0) { const char *e = getenv("DM_VK_RING_VTX"); vtx = e && e[0] == '1'; }
            if (vtx) {
                p_bind_buf vx[16], ix[16]; memset(vx, 0, sizeof vx); memset(ix, 0, sizeof ix);
                for (int i = 0; i < 16; i++) if (S.vbuf[i].address) { vx[i].addr = S.vbuf[i].address; vx[i].size = S.vbuf[i].size ? S.vbuf[i].size : 1; }
                int isz = di ? (di->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 : di->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4) : 0;
                if (di && di->ibuf) { ix[0].addr = di->ibuf; ix[0].size = (uint64_t)di->count * (uint64_t)isz; }
                vkx_ring_csdump2(s, want, vs->key, (int)(dr ? dr->count : di->count), di ? di->base_vertex : dr->first, isz, vx, ix);
                char line[900]; int n = snprintf(line, sizeof line, "VTX vs %016llx count %d base %d index bytes %d:", (unsigned long long)vs->key, (int)(dr ? dr->count : di->count), di ? di->base_vertex : dr->first, isz);
                for (uint32_t l = 0; l < 32 && n < 820; l++) if (vs->in_mask >> l & 1) { va_st at; memset(&at, 0, sizeof at); if ((int)l < S.nattr) memcpy(&at, &S.attr[l], 4);
                    n += snprintf(line + n, sizeof line - (size_t)n, " L%u fmt %u vk %d stream %u offset %u stride %u div %u;", l, (unsigned)at.fmt, at.fmt ? (int)vfmt_vk(at.fmt) : 0, (unsigned)at.stream, (unsigned)at.offset, at.stream < 16 ? S.stream[at.stream].stride : 0, at.stream < 16 ? S.stream[at.stream].divisor : 0); }
                vkx_ring_note("%s", line);
            }
        } }
    }
    // resources first: uploads end the render pass, which must be open for the draw
    VkDescriptorSet sets[3] = {0}; int missing = 0;
    g_psd_on = psd_want(ps) && psd_begin();
    { int ok_sets = stage_set(s, vs, &sets[0], &missing) && (!ps || stage_set(s, ps, &sets[1], &missing)) && (!gs || stage_set(s, gs, &sets[2], &missing));
      if (g_psd_on) { g_psd_on = 0; g_psd_done_at = g_trace_at; vkx_defer(psd_write, NULL); }
      if (!ok_sets) { g_n_skip_other++; return; } }
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
    k.alpha = ps && ps->has_alpha && S.color.alpha_func && S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS;
    k.imu[0] = vs->imask_u; k.ims[0] = vs->imask_s;
    if (ps) { k.imu[1] = ps->imask_u; k.ims[1] = ps->imask_s; }
    if (gs) { k.imu[2] = gs->imask_u; k.ims[2] = gs->imask_s; }
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
    {   // diagnostics: DM_VK_SKIP_PS=<ps key list> draws nothing for those pixel shaders; DM_VK_FORCE_BLEND / _ADD give them
        // SRC_ALPHA/1-SRC_ALPHA or ONE/ONE — is a flat, hard-edged quad a blend state the game set and we dropped?
        static int inited; static const char *skip, *fb, *fa;
        if (!inited) { inited = 1; skip = getenv("DM_VK_SKIP_PS"); fb = getenv("DM_VK_FORCE_BLEND"); fa = getenv("DM_VK_FORCE_ADD"); }
        char key[17]; key[0] = 0;
        if (ps) snprintf(key, sizeof key, "%016llx", (unsigned long long)ps->key);
        if (key[0] && skip && strstr(skip, key)) { static int said; if (!said) { said = 1; VLOG("DM_VK_SKIP_PS: ps %s is not drawn", key); } g_n_skip_other++; return; }
        int add = key[0] && fa && strstr(fa, key);
        if (key[0] && (add || (fb && strstr(fb, key)))) {
            static int said; if (!said) { said = 1; VLOG("DM_VK_FORCE_%s: ps %s", add ? "ADD" : "BLEND", key); }
            k.blend_en = 0xffffffffu;
            for (int i = 0; i < 8; i++) {
                k.blend[i].src_rgb = add ? NVN_BLEND_FUNC_ONE : NVN_BLEND_FUNC_SRC_ALPHA;
                k.blend[i].dst_rgb = add ? NVN_BLEND_FUNC_ONE : NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA;
                k.blend[i].src_a = NVN_BLEND_FUNC_ONE;
                k.blend[i].dst_a = add ? NVN_BLEND_FUNC_ONE : NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA;
                k.blend[i].eq_rgb = k.blend[i].eq_a = NVN_BLEND_EQUATION_ADD; k.blend[i].target = 0; k.blend[i].adv = 0;
            }
        }
    }
    // A whole-screen draw is never skipped for a pipeline still being built, in a pass with a depth image either: Fire Emblem Fortune's Weave draws some of its
    // full-screen steps with the depth image attached (fd88ba60fe74bd15 into the scene with depth test on), and a battle zoom on a cold cache was one black
    // frame (Turnip's own, smaller pipeline cache: up to 542 draws skipped in 300 frames, logs_mem3/log_black2_prev.txt)
    // Nor is an opaque draw (one that writes depth, or has blending off): the game draws its scenery in a depth pass and then a colour pass that takes only pixels of
    // equal depth, and a mesh skipped in one of the two came out black (the arena's whole floor for a moment, on a cold cache: logs_mem3/log_cold1.txt, 830 draws
    // skipped in 900 frames). What may still be skipped is blended and writes no depth: particles and other effects, which only appear a frame late.
    // DM_VK_PIPE_SKIP_OPAQUE=1: opaque draws skipped too, as before
    { static int so = -1; if (so < 0) { const char *e = getenv("DM_VK_PIPE_SKIP_OPAQUE"); so = e && e[0] == '1'; }
      g_pipe_must = (dr && dr->count <= 4) || (!so && (S.ds.dw || !S.color.blend_en)); }
    pent *pe = pipeline_of(&k); g_pipe_must = 0; if (!pe) { g_n_skip_other++; return; }
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
    {   // DM_VK_EQ_RELAX=ge|le (workaround, off by default): a depth EQUAL test without depth writes (a G-buffer drawn over a depth pre-pass) tests GREATER_OR_EQUAL (reversed
        // depth, as Fire Emblem Fortune's Weave's pre-pass) or LESS_OR_EQUAL instead, the fragment moved toward the viewer by DM_VK_EQ_BIAS depth units (default 4). For a GPU
        // compiler that computes two vertex shaders' identical position math to different depths despite Invariant / NoContraction: the floor dropped out of the G-buffer
        // in single frames on Adreno (docs/reference/shaders.md "Position invariance across shaders"); as BananaNative's mesh_cut relaxes EQUAL (vka_draw.c)
        static int rel = -1; static float bias = 4;
        if (rel < 0) { const char *e = getenv("DM_VK_EQ_RELAX"); rel = !e ? 0 : e[0] == 'g' ? 1 : e[0] == 'l' ? 2 : 0; e = getenv("DM_VK_EQ_BIAS"); if (e) bias = (float)atof(e);
                       if (rel) VLOG("DM_VK_EQ_RELAX: depth EQUAL without depth writes tests %s, biased %g units toward the viewer", rel == 1 ? "GREATER_OR_EQUAL" : "LESS_OR_EQUAL", bias); }
        if (rel && dop == VK_COMPARE_OP_EQUAL && !(S.ds.dw && k.d_on)) {
            dop = rel == 1 ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_LESS_OR_EQUAL;
            vkCmdSetDepthBiasEnable(cb, VK_TRUE); vkCmdSetDepthBias(cb, rel == 1 ? bias : -bias, 0, 0);
        }
    }
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
        int vfound = 0;
        if (f && st < 16 && S.vbuf[st].address) {
            uint64_t vsz = S.vbuf[st].size ? S.vbuf[st].size : 1, avail = 0;
            if (vk_pool_find(S.vbuf[st].address, vsz, &pb, &po)) vfound = 1;
            else if (vk_pool_find_part(S.vbuf[st].address, vsz, &pb, &po, &avail)) {   // past its pool's end: bound anyway (the draw reads the vertices it has), was zeros
                vfound = 1; g_n_part_vtx++;
                if (g_n_part_vtx <= 20) VLOG("[RED] vertex stream %u at %#llx+%#llx runs past its pool's end (%#llx bytes there): bound, was zeros; vs %016llx, frame %llu", st,
                                             (unsigned long long)S.vbuf[st].address, (unsigned long long)vsz, (unsigned long long)avail, (unsigned long long)vs->key, (unsigned long long)g_presents);
                vkx_ring_note("VTX-PAST-POOL vs %016llx stream %u addr %#llx size %#llx avail %#llx", (unsigned long long)vs->key, st, (unsigned long long)S.vbuf[st].address, (unsigned long long)vsz, (unsigned long long)avail);
            } else {
                g_n_nopool_vtx++;
                if (g_n_nopool_vtx <= 20) VLOG("[RED] vertex stream %u at %#llx+%#llx is in no GPU pool: zeros; vs %016llx, frame %llu", st, (unsigned long long)S.vbuf[st].address, (unsigned long long)vsz, (unsigned long long)vs->key, (unsigned long long)g_presents);
                vkx_ring_note("VTX-NO-POOL vs %016llx stream %u addr %#llx size %#llx", (unsigned long long)vs->key, st, (unsigned long long)S.vbuf[st].address, (unsigned long long)vsz);
            }
        }
        if (vfound) {
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
    int oq = -1;   // DM_VK_RING_OCC: this draw's query (the plain draws only)
    if (dr && !quads) {
        if (dr->ind) { VkBuffer ib; uint64_t io; if (!vk_pool_find(dr->ind, 16, &ib, &io)) { g_n_skip_other++; return; } vkCmdDrawIndirect(cb, ib, io, 1, 16); }
        else { oq = vkx_occ_begin(cb); vkCmdDraw(cb, (uint32_t)dr->count, inst, (uint32_t)dr->first, (uint32_t)dr->base_instance); vkx_occ_end(cb, oq); }
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
            else { oq = vkx_occ_begin(cb); vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance); vkx_occ_end(cb, oq); }
        }
    }
    g_n_draws_done++;
    snap_after(s, ps);
    vkx_sublog(s, " D%llx/%llx", (unsigned long long)(vs->key >> 32), ps ? (unsigned long long)(ps->key >> 32) : 0ull);
    vkx_ring_note("draw vs %016llx ps %016llx target %llu %ux%u %s %d depth test %d write %d func %d blend %#x ds %llu occ %d", (unsigned long long)vs->key, ps ? (unsigned long long)ps->key : 0ull,
                  g_rt.rt.n && g_rt.rt.color[0] ? (unsigned long long)g_rt.rt.color[0]->serial : 0ull, g_rt.ext.width, g_rt.ext.height, dr ? "arrays" : "elements", dr ? dr->count : di->count,
                  S.ds.dt, S.ds.dw, S.ds.df, S.color.blend_en, (unsigned long long)(g_rt.rt.depth ? g_rt.rt.depth->serial : 0), oq);
    vkx_crumb(s);
    TRACE("  ps outputs %#x, target formats %u %u %u %u %u", ps ? ps->out_frag : 0, k.col_fmt[0], k.col_fmt[1], k.col_fmt[2], k.col_fmt[3], k.col_fmt[4]);
    TRACE("  drawn: %s count %d, inputs %#x, blend %#x mask %#x, cull %d front %d; depth test %d write %d func %d (attached %d fmt %d), range %.3f-%.3f; stencil %d; ncol %u; feedback copies so far %llu",
          dr ? "arrays" : "elements", dr ? dr->count : di->count, vs->in_mask, S.color.blend_en, S.chmask.mask, S.poly.cull, S.poly.front, S.ds.dt, S.ds.dw, S.ds.df, k.d_on, k.ds_fmt, S.dr[0], S.dr[1], S.ds.st, k.ncol,
          (unsigned long long)g_n_fbcopy);
}
// ---------- Compute ----------
typedef struct cpent { struct cpent *next; shd *cs; uint64_t imu, ims; VkPipeline p; VkPipelineLayout pl; int failed; } cpent;   // imu / ims: integer slots (int_mod)
static cpent *g_cpipe[256];
static uint64_t g_n_dispatch, g_n_dispatch_skip, g_n_cpipes;
static uint64_t g_n_cpipes_all(void) { return g_n_cpipes; }
static cpent *cpipe_of(shd *cs) {
    cpent **b = &g_cpipe[(cs->key ^ cs->key >> 17) & 255];
    for (cpent *e = *b; e; e = e->next) if (e->cs == cs && e->imu == cs->imask_u && e->ims == cs->imask_s) return e->failed ? NULL : e;
    cpent *e = calloc(1, sizeof *e); e->cs = cs; e->imu = cs->imask_u; e->ims = cs->imask_s; e->next = *b; *b = e;
    VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &cs->dsl};
    if (vkCreatePipelineLayout(g_vk.dev, &li, NULL, &e->pl)) { e->failed = 1; return NULL; }
    pcache_open(); plist_add_compute(cs, e->imu, e->ims);
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rsg = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 64};   // see nvn_vk_mem.c "Subgroup size"
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, g_vk.sg_pin_cs ? &rsg : NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, int_mod(cs, e->imu, e->ims), "main", NULL}, e->pl, VK_NULL_HANDLE, -1};
    uint64_t pc0 = vkx_now_ns(); VkResult r = vkCreateComputePipelines(g_vk.dev, g_pcache, 1, &ci, NULL, &e->p); vkx_sb_note(3, vkx_now_ns() - pc0);
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
    g_nsimg = 0;
    if (!stage_set(s, cs, &set, &missing)) { g_cs_trace = 0; g_n_dispatch_skip++; return; }
    g_cs_trace = 0;
    if (missing) g_n_missing_tex += (uint64_t)missing;
    cpent *pe = cpipe_of(cs); if (!pe) { g_n_dispatch_skip++; return; }
    end_rendering();   // no compute inside a render pass
    VkCommandBuffer cb = s->cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pe->p);
    if (set) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pe->pl, 0, 1, &set, 0, NULL);
    vkx_prof_begin(cb, "compute %016llx", (unsigned long long)cs->key);
    if (d.ind) { VkBuffer ib; uint64_t io; if (!vk_pool_find(d.ind, 12, &ib, &io)) { g_n_dispatch_skip++; vkx_prof_end(cb); return; } vkCmdDispatchIndirect(cb, ib, io); }
    else { if (!d.x || !d.y || !d.z) { vkx_prof_end(cb); return; } vkCmdDispatch(cb, (uint32_t)d.x, (uint32_t)d.y, (uint32_t)d.z); }
    vkx_prof_end(cb);
    barrier_compute(cb);
    g_n_dispatch++;
    vkx_ring_csdump(s, cs->key, d.x, d.y, d.z, S.ubo[5], S.ssbo[5]);
    vkx_sublog(s, " C%016llx(%dx%dx%d)", (unsigned long long)cs->key, d.x, d.y, d.z);
    vkx_crumb(s);
    TRACE("dispatch cs %016llx %dx%dx%d%s", (unsigned long long)cs->key, d.x, d.y, d.z, d.ind ? " (indirect)" : "");
    vkx_ring_note("dispatch cs %016llx %dx%dx%d%s", (unsigned long long)cs->key, d.x, d.y, d.z, d.ind ? " (indirect)" : "");
    for (int i = 0; i < g_nsimg; i++) { char w[40]; snprintf(w, sizeof w, "cs %016llx", (unsigned long long)cs->key); vkx_stage_snap(s, g_simg[i], w); }
}

// ---------- Stretching one image over another by drawing ----------
// A full-screen triangle sampling the source (shaders/stretch.vert, stretch.frag, stretch_depth.frag; nvn_vk_blit_spv.h). Used instead of vkCmdBlitImage wherever the
// driver does not offer the blit for the formats (depth formats here) and for the DLSS stand-in: runs with blits on this driver lost the device (a scaled D32 blit
// preceded every hang), so a draw, the path the game's own frames take, stands in
#include "nvn_vk_blit_spv.h"
#include "nvn_vk_reactive_spv.h"   // shaders/reactive.comp: FSR 2's reactive mask (reactive_run)
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
    VkPipeline pp; uint64_t pc0 = vkx_now_ns(); VkResult pr = vkCreateGraphicsPipelines(g_vk.dev, g_pcache, 1, &gi, NULL, &pp); vkx_sb_note(3, vkx_now_ns() - pc0); if (pr) return VK_NULL_HANDLE;
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
static VkImageView storage_view(sub_t *s, vk_tex *x) {   // what a storage image sees of x: its UNORM twin for an sRGB image (tex_create, nvn_vk_exec.c)
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, VK_IMAGE_USAGE_STORAGE_BIT};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vu, 0, x->img, VK_IMAGE_VIEW_TYPE_2D, x->storage_fmt, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
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

// ---------- DLSS: the game's temporal upscale (the Mac's MetalFX scaler): FSR 2 (dm/third_party/fsr2 via fsr2_glue.cpp); the stretch stand-in draws what FSR 2 cannot take ----------
static uint64_t g_n_dlss, g_n_dlss_st, g_n_dlss_fsr;
// ---------- FSR 2's reactive mask (shaders/reactive.comp; DM_FSR2_REACTIVE=0 turns it off) ----------
// Fire Emblem Fortune's Weave gives moving objects a previous transform equal to the current one, so their motion vectors are zero, and FSR 2 kept their old
// positions: ghosting behind turning gears and walking characters (the captures logs_mem3/002156 and 000778; the Eden fork notes the same ghosting with the
// game's own DLSS network). This pass marks where the picture changed since the last frame at the place the motion vector points to, and FSR 2 trusts its
// history less there. DM_FSR2_REACTIVE_T0 / _T1: the relative brightness change where the mask starts and where it is full (default 0.06, 0.2);
// DM_FSR2_REACTIVE_MAX: its strongest value (default 0.9; 1 would drop the history entirely)
static struct { VkPipeline p; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkImage img[3]; VkDeviceMemory mem[3]; VkImageView view[3]; uint32_t w, h; int cur, have_prev, state; } g_ra;
static float g_ra_t0, g_ra_t1, g_ra_max, g_ra_decay;
static int ra_mem_type(uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_vk.pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return (int)i;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if (bits >> i & 1) return (int)i;
    return -1;
}
static int ra_setup(void) {   // once: the switches, the format check and the pipeline. 1 when the pass can run
    if (g_ra.state) return g_ra.state > 0;
    g_ra.state = -1;
    const char *e = getenv("DM_FSR2_REACTIVE"); if (e && e[0] == '0') { VLOG("DLSS: no reactive mask (DM_FSR2_REACTIVE=0)"); return 0; }
    e = getenv("DM_FSR2_REACTIVE_T0"); g_ra_t0 = e ? (float)atof(e) : 0.06f;
    e = getenv("DM_FSR2_REACTIVE_T1"); g_ra_t1 = e ? (float)atof(e) : 0.2f;
    e = getenv("DM_FSR2_REACTIVE_MAX"); g_ra_max = e ? (float)atof(e) : 0.9f;
    e = getenv("DM_FSR2_REACTIVE_DECAY"); g_ra_decay = e ? (float)atof(e) : 0.85f;   // the share of last frame's mask kept: about ten frames to fade
    VkFormat need[2] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R16_SFLOAT};
    for (int i = 0; i < 2; i++) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vk.pd, need[i], &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) || !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) { VLOG("[RED] DLSS: no reactive mask: format %d cannot be a storage image here", need[i]); return 0; } }
    if (!vkCmdPushDescriptorSetKHR) { VLOG("[RED] DLSS: no reactive mask: push descriptors are not available"); return 0; }
    VkDescriptorSetLayoutBinding b[5];
    for (uint32_t i = 0; i < 5; i++) b[i] = (VkDescriptorSetLayoutBinding){i, i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 5, b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 28};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_reactive_comp, k_reactive_comp};
    VkShaderModule m;
    if (vkCreateDescriptorSetLayout(g_vk.dev, &dli, NULL, &g_ra.dsl) || vkCreateShaderModule(g_vk.dev, &mi, NULL, &m)) { VLOG("[RED] DLSS: no reactive mask: its layout or shader failed"); return 0; }
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_ra.dsl, 1, &pr};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL},
                                      VK_NULL_HANDLE, VK_NULL_HANDLE, -1};
    int ok = !vkCreatePipelineLayout(g_vk.dev, &pli, NULL, &g_ra.pl) && (ci.layout = g_ra.pl, !vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_ra.p));
    vkDestroyShaderModule(g_vk.dev, m, NULL);
    if (!ok) { VLOG("[RED] DLSS: no reactive mask: its pipeline failed"); return 0; }
    VLOG("DLSS: reactive mask on: brightness change from %.2f to %.2f of the pixel's level, up to %.2f, %.2f of it kept each next frame; also as the transparency and composition mask (DM_FSR2_REACTIVE, _T0, _T1, _MAX, _DECAY)",
         g_ra_t0, g_ra_t1, g_ra_max, g_ra_decay);
    g_ra.state = 1; return 1;
}
static int ra_images(VkCommandBuffer cb, uint32_t w, uint32_t h) {   // the mask and the two brightness histories at the render size
    if (g_ra.img[0] && g_ra.w == w && g_ra.h == h) return 1;
    if (g_ra.img[0]) { vkDeviceWaitIdle(g_vk.dev); for (int i = 0; i < 3; i++) { vkDestroyImageView(g_vk.dev, g_ra.view[i], NULL); vkDestroyImage(g_vk.dev, g_ra.img[i], NULL); vkFreeMemory(g_vk.dev, g_ra.mem[i], NULL); g_ra.img[i] = VK_NULL_HANDLE; } }
    VkImageMemoryBarrier ib[3];
    for (int i = 0; i < 3; i++) {
        VkFormat f = i ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R8_UNORM;
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_vk.dev, &ii, NULL, &g_ra.img[i])) { g_ra.img[i] = VK_NULL_HANDLE; return 0; }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, g_ra.img[i], &mr); int t = ra_mem_type(mr.memoryTypeBits);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)(t < 0 ? 0 : t)};
        if (t < 0 || vkAllocateMemory(g_vk.dev, &mai, NULL, &g_ra.mem[i]) || vkBindImageMemory(g_vk.dev, g_ra.img[i], g_ra.mem[i], 0)) return 0;
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_ra.img[i], VK_IMAGE_VIEW_TYPE_2D, f, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(g_vk.dev, &vi, NULL, &g_ra.view[i])) return 0;
        ib[i] = (VkImageMemoryBarrier){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                       VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_ra.img[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 3, ib);
    g_ra.w = w; g_ra.h = h; g_ra.have_prev = 0; g_ra.cur = 0;
    return 1;
}
// records the mask for this DLSS call into the submission (after the game's draws, before FSR 2) and gives it to the job; nothing when it cannot run
static void reactive_run(sub_t *s, fsr2_job *j) {
    if (!ra_setup() || !ra_images(s->cb, j->iw, j->ih)) return;
    int prev = 1 + (g_ra.cur ^ 1), cur = 1 + g_ra.cur;
    VkSampler smp = vkd_linear_sampler();
    VkDescriptorImageInfo ii[5] = {{smp, j->color_v, VK_IMAGE_LAYOUT_GENERAL}, {smp, j->motion_v, VK_IMAGE_LAYOUT_GENERAL}, {smp, g_ra.view[prev], VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, g_ra.view[0], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, g_ra.view[cur], VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[5];
    for (uint32_t i = 0; i < 5; i++) w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, i, 0, 1,
                                                                 i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[i], NULL, NULL};
    struct { int32_t w, h; float t0, t1, rmax; int32_t first; float decay; } pc = {(int32_t)j->iw, (int32_t)j->ih, g_ra_t0, g_ra_t1, g_ra_max, (j->reset || !g_ra.have_prev) ? 1 : 0, g_ra_decay};
    barrier_all(s->cb);   // last frame's FSR 2 read of the mask and this frame's draws of colour and motion
    vkCmdBindPipeline(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ra.p);
    vkCmdPushDescriptorSetKHR(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ra.pl, 0, 5, w);
    vkCmdPushConstants(s->cb, g_ra.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(s->cb, (j->iw + 7) / 8, (j->ih + 7) / 8, 1);
    barrier_all(s->cb);
    g_ra.cur ^= 1; g_ra.have_prev = 1;
    vkx_ring_mask(s, g_ra.img[0], j->iw, j->ih);   // DM_VK_RING_DLSS: the mask as FSR 2 gets it
    j->reactive = g_ra.img[0]; j->reactive_v = g_ra.view[0];
}

void vkd_dlss(sub_t *s, const void *a) {
    const p_dlss *c = a;
    { static int off = -1; if (off < 0) off = getenv("DM_VK_NO_DLSS") != NULL; if (off) return; }
    if (!c->tex[DLSS_COLOR] || !c->tex[DLSS_OUTPUT]) { VRED_ONCE("DLSS evaluated without %s", c->tex[DLSS_COLOR] ? "an output" : "an input colour"); return; }
    end_rendering();
    vk_tex *in = tex_for_read(c->tex[DLSS_COLOR], s), *out = tex_for_write(c->tex[DLSS_OUTPUT], s);
    if (!in || !out) { VRED_ONCE("DLSS: input or output texture has no image"); return; }
    vk_tex *dep = c->tex[DLSS_DEPTH] ? tex_for_read(c->tex[DLSS_DEPTH], s) : NULL;
    if (!vkx_ring_final()) vkx_ring_record(s, in, dep);   // DM_VK_RING: the scene before upscaling, for the rolling capture
    vk_tex *mot = c->tex[DLSS_MOTION] ? tex_for_read(c->tex[DLSS_MOTION], s) : NULL;
    vk_tex *expo = c->tex[DLSS_EXPOSURE] ? tex_for_read(c->tex[DLSS_EXPOSURE], s) : NULL;
    vkx_ring_dlss(s, in, mot, dep, c->jitter, c->mv);   // DM_VK_RING_DLSS: these inputs as FSR 2 reads them
    if (c->tex[DLSS_BIAS]) VRED_ONCE("DLSS: the game's bias mask is not used by FSR 2 (as the Mac's MetalFX path does not use it)");
    int32_t iw = c->iw > 0 ? c->iw : (int32_t)in->w, ih = c->ih > 0 ? c->ih : (int32_t)in->h;
    int32_t ow = c->ow > 0 ? c->ow : (int32_t)out->w, oh = c->oh > 0 ? c->oh : (int32_t)out->h;
    int32_t ir[4] = {c->base[DLSS_COLOR][0], c->base[DLSS_COLOR][1], iw, ih};
    int32_t orr[4] = {c->base[DLSS_OUTPUT][0], c->base[DLSS_OUTPUT][1], ow, oh};
    fsr2_job j = {0};
    int can = 0;   // whether the images are what FSR 2 takes: full 2D single-sampled textures, samplable input / depth / motion, a storage-capable output
    static int fsr_off = -1; if (fsr_off < 0) { const char *e = getenv("DM_FSR2_OFF"); fsr_off = e && e[0] == '1'; if (fsr_off) VLOG("DLSS: FSR 2 off (DM_FSR2_OFF=1): the stretch stand-in draws"); }   // A/B: the temporal upscaler off
    if (fsr_off) {
    } else if (!dep || !mot) {   // a temporal upscaler cannot run without them (nor does the Mac's MetalFX scaler)
        if (!g_n_dlss) VLOG("DLSS: no %s: FSR 2 cannot run, the stretch stand-in draws", dep ? "motion vectors" : "depth");
    } else if (in->type != VK_IMAGE_TYPE_2D || out->type != VK_IMAGE_TYPE_2D || dep->type != VK_IMAGE_TYPE_2D || mot->type != VK_IMAGE_TYPE_2D ||
               in->d.samples > 1 || out->d.samples > 1 || dep->d.samples > 1 || mot->d.samples > 1) {
        VRED_ONCE("DLSS: a 3D or multisampled image cannot feed FSR 2: the stretch stand-in draws");
    } else if (!(in->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !(dep->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !(mot->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        VRED_ONCE("DLSS: the input, depth or motion vectors cannot be sampled (formats %d / %d / %d): the stretch stand-in draws", in->fmt, dep->fmt, mot->fmt);
    } else if (!(dep->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VRED_ONCE("DLSS: the depth texture (format %d) has no depth aspect: the stretch stand-in draws", dep->fmt);
    } else if (!out->storage_fmt) {
        VRED_ONCE("DLSS: the output (format %d) cannot be a storage image: the stretch stand-in draws", out->fmt);
    } else if (ir[0] || ir[1] || orr[0] || orr[1] || iw != (int32_t)in->w || ih != (int32_t)in->h || ow != (int32_t)out->w || oh != (int32_t)out->h) {
        VRED_ONCE("DLSS: a sub-rectangle (%d,%d %dx%d of a %ux%u input; %d,%d %dx%d of a %ux%u output) cannot feed FSR 2: the stretch stand-in draws",
                  ir[0], ir[1], ir[2], ir[3], in->w, in->h, orr[0], orr[1], orr[2], orr[3], out->w, out->h);
    } else can = 1;
    if (can) {
        j.color = in->img; j.depth = dep->img; j.motion = mot->img; j.output = out->img;
        j.color_fmt = in->fmt; j.depth_fmt = dep->fmt; j.motion_fmt = mot->fmt; j.output_fmt = out->storage_fmt;   // what the storage view sees
        j.color_v = plain_view(s, in, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
        j.depth_v = plain_view(s, dep, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0);
        j.motion_v = plain_view(s, mot, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0);
        j.output_v = storage_view(s, out);
        if (expo) { j.exposure = expo->img; j.exposure_fmt = expo->fmt; j.exposure_v = plain_view(s, expo, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0); }
        j.mot_w = mot->w; j.mot_h = mot->h;
        j.iw = (uint32_t)iw; j.ih = (uint32_t)ih; j.ow = (uint32_t)ow; j.oh = (uint32_t)oh;
        j.jitter[0] = c->jitter[0]; j.jitter[1] = c->jitter[1]; j.mv[0] = c->mv[0]; j.mv[1] = c->mv[1];
        j.pre_exposure = c->pre_exposure; j.flags = c->flags; j.reset = c->reset; j.cb = s->cb;
        if (!j.color_v || !j.depth_v || !j.motion_v || !j.output_v || (expo && !j.exposure_v)) {
            VRED_ONCE("DLSS: a view for FSR 2 could not be made: the stretch stand-in draws"); can = 0;
        }
    }
    // what the temporal upscaler takes: the game's sub-pixel camera jitter, motion vector scale and history reset, for the first evaluations and then every 300th
    if (g_n_dlss < 8 || g_n_dlss % 300 == 0)
        VLOG("DLSS %llu: jitter %.4f %.4f, motion vector scale %.4f %.4f, reset %u, pre-exposure %.3f, depth %s, motion vectors %s, exposure %s, flags %#x", (unsigned long long)g_n_dlss,
             c->jitter[0], c->jitter[1], c->mv[0], c->mv[1], c->reset, c->pre_exposure, dep ? "given" : "none", mot ? "given" : "none", expo ? "given" : "none", c->flags);
    if (!can) {
        if (!vkd_stretch(s, in, 0, 0, ir, out, 0, 0, orr, 1)) return;
        vkx_sublog(s, " DLSS");
        if (!g_n_dlss_st++) VLOG("DLSS stand-in: %dx%d of a %ux%u input (format %d) drawn bilinearly over %dx%d of the output (format %d)", ir[2], ir[3], in->w, in->h, in->fmt, orr[2], orr[3], out->fmt);
        g_n_dlss++;
        return;
    }
    barrier_all(s->cb);   // the game's last draws on these images, then FSR 2's dispatches into the submission's buffer (fsr2_glue.cpp)
    reactive_run(s, &j);   // where the picture changed, FSR 2 trusts its history less (the game's moving objects have no motion vectors)
    if (!fsr2_dispatch(&j)) {   // an FFX error or a context that cannot be made: the stand-in still draws the frame
        if (!vkd_stretch(s, in, 0, 0, ir, out, 0, 0, orr, 1)) return;
        vkx_sublog(s, " DLSS");
        g_n_dlss++;
        return;
    }
    barrier_all(s->cb);
    vkx_sublog(s, " DLSS(FSR2)");
    if (!g_n_dlss_fsr++) VLOG("DLSS: FSR 2 upscales %dx%d to %dx%d (colour %d, depth %d, motion %d, output %d)", iw, ih, ow, oh, in->fmt, dep->fmt, mot->fmt, out->storage_fmt);
    vkx_crumb(s);
    g_n_dlss++;
}

void vkd_stats_line(char *out, size_t n) {
    snprintf(out, n, "%llu drawn; skipped: %llu no program, %llu shader unusable, %llu no target, %llu other; %llu dispatches (%llu skipped, %llu compute pipelines), %llu DLSS, %llu feedback copies, %llu pool descriptors; %llu shaders, %llu pipelines (%llu failed), %llu empty texture slots",
             (unsigned long long)g_n_draws_done, (unsigned long long)g_n_skip_noprog, (unsigned long long)g_n_skip_bad, (unsigned long long)g_n_skip_rt, (unsigned long long)g_n_skip_other,
             (unsigned long long)g_n_dispatch, (unsigned long long)g_n_dispatch_skip, (unsigned long long)g_n_cpipes, (unsigned long long)g_n_dlss, (unsigned long long)g_n_fbcopy, (unsigned long long)g_n_pool_desc, (unsigned long long)g_n_shaders, (unsigned long long)g_n_pipelines, (unsigned long long)g_n_pipe_fail, (unsigned long long)g_n_missing_tex);
}
