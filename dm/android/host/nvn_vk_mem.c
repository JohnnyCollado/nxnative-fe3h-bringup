// The Android host's Vulkan device and the NVN memory pools as GPU memory (step V1a of the Vulkan backend).
// The game's NVN memory pools live at addresses it chose in its own heap, and both the CPU (the game) and the GPU use them: buffers, vertex and index data, uniform
// blocks, textures and command memory all sit in pool memory. The Mac backend makes Metal buffers directly over that memory; the Adreno driver has no
// VK_EXT_external_memory_host (dm/android/tools/vk_probe.c), so here each pool is backed by driver memory that is moved onto the pool's address instead:
//   1. allocate host-visible, coherent, cached device memory of the pool's size and map it (vkMapMemory)
//   2. copy the pool's current bytes into it
//   3. move the driver's mapping onto the pool's address (mremap MREMAP_MAYMOVE | MREMAP_FIXED), which replaces the game's pages there; the CPU and the GPU now share the
//      pool's memory at the address the game knows (dm/android/tools/vk_alias_test.c: a moved mapping behaves as the driver's own)
//   4. reserve the range the driver's mapping left (PROT_NONE): the driver unmaps that range when the memory is freed, and must not hit another mapping placed there meanwhile
// When the pool is finalized the bytes are copied back into ordinary anonymous pages at the same address before the device memory is freed, so the game's heap keeps its
// memory and its contents. Every pool also gets a VkBuffer over its whole memory, which the backend's buffer address lookups (vk_pool_find) resolve into.
// DM_VK_POOLS=0 leaves pools in ordinary memory (the null backend's behaviour).
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)
#include "nvn_vk.h"

#define VLOG(...) do { flockfile(stderr); fprintf(stderr, "[dm] [vk] " __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } while (0)

vk_ctx g_vk;
static pthread_once_t g_vk_once = PTHREAD_ONCE_INIT;

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }

int vk_driver_load(void); const char *vk_driver_desc(void);   // vk_driver.c
static void vk_init_once(void) {
    if (!vk_driver_load()) { VLOG("[RED] no Vulkan driver: pools stay in ordinary memory"); return; }
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkResult r = vkCreateInstance(&ici, NULL, &g_vk.inst);
    if (r) { VLOG("[RED] vkCreateInstance failed (%d): no Vulkan device, pools stay in ordinary memory", r); return; }
    volkLoadInstance(g_vk.inst);
    uint32_t n = 1; if (vkEnumeratePhysicalDevices(g_vk.inst, &n, &g_vk.pd) < 0 || !n) { VLOG("[RED] no Vulkan physical device"); return; }
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(g_vk.pd, &p);
    vkGetPhysicalDeviceMemoryProperties(g_vk.pd, &g_vk.mp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &nq, NULL);
    VkQueueFamilyProperties qf[8]; nq = nq > 8 ? 8 : nq; vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &nq, qf);
    g_vk.qfam = 0; for (uint32_t i = 0; i < nq; i++) if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { g_vk.qfam = i; break; }
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(g_vk.pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne, sizeof *ext); vkEnumerateDeviceExtensionProperties(g_vk.pd, NULL, &ne, ext);
    const char *want[] = {"VK_EXT_depth_clip_control", "VK_KHR_push_descriptor", "VK_EXT_custom_border_color", "VK_EXT_robustness2", "VK_EXT_extended_dynamic_state",
                          "VK_EXT_extended_dynamic_state2", "VK_EXT_vertex_input_dynamic_state", "VK_EXT_index_type_uint8", "VK_EXT_provoking_vertex", "VK_EXT_depth_clip_enable",
                          "VK_EXT_vertex_attribute_divisor", "VK_ANDROID_external_memory_android_hardware_buffer", "VK_EXT_queue_family_foreign"};   // the last two: the app's screen (nvn_vk_front.c)
    const char *on[16]; uint32_t non = 0;
    for (size_t i = 0; i < sizeof want / sizeof *want; i++) if (has_ext(ext, ne, want[i])) on[non++] = want[i]; else VLOG("device extension %s not offered", want[i]);
    free(ext);
    // Features the translated shaders and NVN state use (dm/android/tools/vk_probe.c lists what this driver offers)
    // The extensions' own features: an extension turned on without its feature bit offers nothing (vertex input dynamic state, nullDescriptor and depth clip control are
    // needed by the draw path, nvn_vk_draw.c). Chained only for the extensions the driver offers
    int x_vi = 0, x_rb = 0, x_dc = 0, x_u8 = 0, x_cb = 0, x_dv = 0;
    for (uint32_t i = 0; i < non; i++) { x_vi |= !strcmp(on[i], "VK_EXT_vertex_input_dynamic_state"); x_rb |= !strcmp(on[i], "VK_EXT_robustness2"); x_dc |= !strcmp(on[i], "VK_EXT_depth_clip_control");
                                         x_u8 |= !strcmp(on[i], "VK_EXT_index_type_uint8"); x_cb |= !strcmp(on[i], "VK_EXT_custom_border_color"); x_dv |= !strcmp(on[i], "VK_EXT_vertex_attribute_divisor"); }
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT hvi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT hrb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT hdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT hu8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT hcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT hdv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    void *xchain = NULL;
#define LINK(cond, st) if (cond) { (st).pNext = xchain; xchain = &(st); }
    LINK(x_vi, hvi) LINK(x_rb, hrb) LINK(x_dc, hdc) LINK(x_u8, hu8) LINK(x_cb, hcb) LINK(x_dv, hdv)
#undef LINK
    VkPhysicalDeviceVulkan13Features h13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, xchain};
    VkPhysicalDeviceVulkan12Features h12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &h13};
    VkPhysicalDeviceVulkan11Features h11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &h12};
    VkPhysicalDeviceFeatures2 have = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &h11}; vkGetPhysicalDeviceFeatures2(g_vk.pd, &have);
    VkPhysicalDeviceFeatures f = {0}; const VkPhysicalDeviceFeatures *h = &have.features;
#define ON(x) f.x = h->x
    ON(robustBufferAccess);   // out-of-range buffer accesses read 0 / are dropped instead of faulting the GPU (the translated shaders index storage buffers the game sized)
    ON(shaderStorageImageWriteWithoutFormat); ON(shaderStorageImageReadWithoutFormat); ON(geometryShader); ON(depthClamp); ON(depthBiasClamp); ON(depthBounds);
    ON(independentBlend); ON(dualSrcBlend); ON(logicOp); ON(multiDrawIndirect); ON(drawIndirectFirstInstance); ON(fillModeNonSolid); ON(wideLines); ON(samplerAnisotropy);
    ON(textureCompressionBC); ON(textureCompressionASTC_LDR); ON(textureCompressionETC2); ON(shaderInt64); ON(shaderInt16); ON(sampleRateShading); ON(imageCubeArray);
    ON(fragmentStoresAndAtomics); ON(vertexPipelineStoresAndAtomics); ON(shaderClipDistance); ON(shaderCullDistance); ON(multiViewport); ON(occlusionQueryPrecise);
#undef ON
    // Each is asked for only when the driver offers it (a feature it lacks fails vkCreateDevice with VK_ERROR_FEATURE_NOT_PRESENT)
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, NULL};
    f13.dynamicRendering = h13.dynamicRendering; f13.synchronization2 = h13.synchronization2; f13.maintenance4 = h13.maintenance4; f13.shaderDemoteToHelperInvocation = h13.shaderDemoteToHelperInvocation;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = h12.timelineSemaphore; f12.shaderFloat16 = h12.shaderFloat16; f12.shaderInt8 = h12.shaderInt8; f12.scalarBlockLayout = h12.scalarBlockLayout;
    f12.descriptorIndexing = h12.descriptorIndexing; f12.runtimeDescriptorArray = h12.runtimeDescriptorArray; f12.descriptorBindingPartiallyBound = h12.descriptorBindingPartiallyBound;
    {   // the Vulkan memory model, only when frame generation is on (DM_LSFG=1: the Lossless shaders need it, lsfg_glue.cpp). Turned on for every run it was
        // the one device change before Fire Emblem Fortune's Weave's light culling went wrong in a triangle of tiles on the world map (logs_mem3/001253): kept off otherwise
        const char *lf = getenv("DM_LSFG"); int want = lf && lf[0] == '1';
        if (want) { f12.vulkanMemoryModel = h12.vulkanMemoryModel; f12.vulkanMemoryModelDeviceScope = h12.vulkanMemoryModelDeviceScope; }
        g_vk.mem_model = want && h12.vulkanMemoryModel; g_vk.fp16 = h12.shaderFloat16;
    }
    f12.drawIndirectCount = h12.drawIndirectCount; f12.hostQueryReset = h12.hostQueryReset; g_vk.host_qreset = h12.hostQueryReset; f12.shaderSubgroupExtendedTypes = h12.shaderSubgroupExtendedTypes;
    f12.uniformAndStorageBuffer8BitAccess = h12.uniformAndStorageBuffer8BitAccess; f12.storageBuffer8BitAccess = h12.storageBuffer8BitAccess;
    // the same extension structs, now asking for what the query found (vkGetPhysicalDeviceFeatures2 filled them in place)
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT fvi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT, NULL, hvi.vertexInputDynamicState};
    VkPhysicalDeviceRobustness2FeaturesEXT frb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, NULL, h->robustBufferAccess ? hrb.robustBufferAccess2 : 0, hrb.robustImageAccess2, hrb.nullDescriptor};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, NULL, hdc.depthClipControl};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT fu8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, NULL, hu8.indexTypeUint8};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, NULL, hcb.customBorderColors, hcb.customBorderColorWithoutFormat};
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT fdv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, NULL, hdv.vertexAttributeInstanceRateDivisor, hdv.vertexAttributeInstanceRateZeroDivisor};
    void *fchain = NULL;
#define LINK(cond, st) if (cond) { (st).pNext = fchain; fchain = &(st); }
    LINK(x_vi, fvi) LINK(x_rb, frb) LINK(x_dc, fdc) LINK(x_u8, fu8) LINK(x_cb, fcb) LINK(x_dv, fdv)
#undef LINK
    f13.pNext = fchain;
    {   // Subgroup size: the compute shaders that pass values between threads (Maxwell / Ampere warps of 32 lanes) are translated for 64-lane subgroups. Qualcomm's driver
        // runs 64; a driver whose size varies (Turnip: 64-128) is told 64 per compute pipeline (nvn_vk_draw.c cpipe_of), when it lets the compute stage be pinned
        VkPhysicalDeviceVulkan13Properties q13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES}; VkPhysicalDeviceVulkan11Properties q11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &q13};
        VkPhysicalDeviceProperties2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &q11}; vkGetPhysicalDeviceProperties2(g_vk.pd, &q2);
        g_vk.sg_size = q11.subgroupSize; g_vk.sg_min = q13.minSubgroupSize ? q13.minSubgroupSize : q11.subgroupSize; g_vk.sg_max = q13.maxSubgroupSize ? q13.maxSubgroupSize : q11.subgroupSize;
        f13.subgroupSizeControl = h13.subgroupSizeControl; f13.computeFullSubgroups = h13.computeFullSubgroups; g_vk.sg_full = h13.computeFullSubgroups;   // sg_full: nvn_vk_sgprobe.c
        g_vk.sg_pin_cs = h13.subgroupSizeControl && (q13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) && g_vk.sg_min <= 64 && g_vk.sg_max >= 64 && (g_vk.sg_min != g_vk.sg_max || g_vk.sg_size != 64);
        VLOG("subgroups: %u lanes (%u-%u), operations %#x in stages %#x; compute pipelines %s", q11.subgroupSize, g_vk.sg_min, g_vk.sg_max, q11.subgroupSupportedOperations, q11.subgroupSupportedStages,
             g_vk.sg_pin_cs ? "pinned to 64" : g_vk.sg_size == 64 || g_vk.sg_size == 32 ? "as the device runs them (right for shaders translated for 64)" : "[RED] not 64 and not pinnable: warp shaders read the wrong lanes");
    }
    g_vk.vi_dyn = x_vi && hvi.vertexInputDynamicState; g_vk.null_desc = x_rb && hrb.nullDescriptor; g_vk.clip_ctl = x_dc && hdc.depthClipControl;
    g_vk.idx_u8 = x_u8 && hu8.indexTypeUint8; g_vk.border = x_cb && hcb.customBorderColors && hcb.customBorderColorWithoutFormat; g_vk.divisor = x_dv && hdv.vertexAttributeInstanceRateDivisor;
    g_vk.lim = p.limits;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = h11.storageBuffer16BitAccess; f11.uniformAndStorageBuffer16BitAccess = h11.uniformAndStorageBuffer16BitAccess; f11.shaderDrawParameters = h11.shaderDrawParameters;
    float prio = 1; VkDeviceQueueCreateInfo dqi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, g_vk.qfam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11, 0, 1, &dqi, 0, NULL, non, on, &f};
    r = vkCreateDevice(g_vk.pd, &dci, NULL, &g_vk.dev);
    if (r) { VLOG("[RED] vkCreateDevice failed (%d): pools stay in ordinary memory", r); g_vk.dev = VK_NULL_HANDLE; return; }
    volkLoadDevice(g_vk.dev);   // device functions straight from the driver, without the loader's dispatch
    vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.q);
    // The pool memory type: host visible and coherent (the CPU and the GPU see each other's writes without flushes), cached when there is one (the game reads pools on the CPU)
    g_vk.pool_type = ~0u;
    for (int pass = 0; pass < 2 && g_vk.pool_type == ~0u; pass++) {
        VkMemoryPropertyFlags want_f = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | (pass ? 0 : VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        for (uint32_t i = 0; i < g_vk.mp.memoryTypeCount; i++) if ((g_vk.mp.memoryTypes[i].propertyFlags & want_f) == want_f) { g_vk.pool_type = i; break; }
    }
    g_vk.ok = 1;
    VLOG("robust access: buffers %d, buffers2 %d, images2 %d", h->robustBufferAccess, h->robustBufferAccess ? hrb.robustBufferAccess2 : 0, hrb.robustImageAccess2);
    VLOG("features: vertex input dynamic state %d, null descriptors %d, depth clip control %d, 8-bit indices %d, border colours %d, divisors %d; alignment uniform %llu storage %llu, uniform range %u",
         g_vk.vi_dyn, g_vk.null_desc, g_vk.clip_ctl, g_vk.idx_u8, g_vk.border, g_vk.divisor, (unsigned long long)g_vk.lim.minUniformBufferOffsetAlignment,
         (unsigned long long)g_vk.lim.minStorageBufferOffsetAlignment, g_vk.lim.maxUniformBufferRange);
    VkPhysicalDeviceDriverProperties dp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dp};
    vkGetPhysicalDeviceProperties2(g_vk.pd, &p2);
    VLOG("driver: %s %s (driver id %d, conformance %u.%u.%u), loaded as %s", dp.driverName, dp.driverInfo, dp.driverID, dp.conformanceVersion.major, dp.conformanceVersion.minor,
         dp.conformanceVersion.subminor, vk_driver_desc());
    VLOG("device: %s, Vulkan %u.%u.%u, %u device extensions on; pool memory type %u (flags %#x)", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
         VK_API_VERSION_PATCH(p.apiVersion), non, g_vk.pool_type, g_vk.pool_type == ~0u ? 0 : g_vk.mp.memoryTypes[g_vk.pool_type].propertyFlags);
    void vkx_sg_probe(void); vkx_sg_probe();   // how the driver groups compute threads into subgroups, against what the warp shaders assume (nvn_vk_sgprobe.c)
}
int vk_ready(void) { pthread_once(&g_vk_once, vk_init_once); return g_vk.ok; }

// ---------- Pools ----------
typedef struct { uint8_t *mem; uint64_t size; VkDeviceMemory dm; VkBuffer buf; } vk_pool;
static vk_pool *g_pools; static int g_npools, g_cpools;
static pthread_rwlock_t g_pools_mu = PTHREAD_RWLOCK_INITIALIZER;
static uint64_t g_pool_bytes; static int g_pool_fail;

static int pools_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOLS"); v = !(e && !strcmp(e, "0")); } return v; }

// The pool's memory as device memory at its own address: 0 on success
static int pool_back(vk_pool *o) {
    uint64_t pg = (uint64_t)getpagesize();
    if ((uintptr_t)o->mem % pg || o->size % pg) { VLOG("[RED] pool %p+%#llx is not page aligned: left in ordinary memory", (void *)o->mem, (unsigned long long)o->size); return -1; }
    VkExternalMemoryBufferCreateInfo *none = NULL; (void)none;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, o->size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkResult r = vkCreateBuffer(g_vk.dev, &bci, NULL, &o->buf);
    if (r) { VLOG("[RED] pool %p+%#llx: vkCreateBuffer %d", (void *)o->mem, (unsigned long long)o->size, r); return -1; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, o->buf, &mr);
    if (!(mr.memoryTypeBits >> g_vk.pool_type & 1)) { VLOG("[RED] pool buffers cannot use memory type %u (allowed %#x)", g_vk.pool_type, mr.memoryTypeBits); vkDestroyBuffer(g_vk.dev, o->buf, NULL); return -1; }
    uint64_t asz = (mr.size + pg - 1) & ~(pg - 1);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, asz, g_vk.pool_type};
    r = vkAllocateMemory(g_vk.dev, &mai, NULL, &o->dm);
    if (r) { VLOG("[RED] pool %p+%#llx: vkAllocateMemory %d", (void *)o->mem, (unsigned long long)o->size, r); vkDestroyBuffer(g_vk.dev, o->buf, NULL); return -1; }
    uint8_t *drv = NULL;
    if ((r = vkMapMemory(g_vk.dev, o->dm, 0, VK_WHOLE_SIZE, 0, (void **)&drv)) || ((uintptr_t)drv % pg)) {
        VLOG("[RED] pool %p+%#llx: vkMapMemory %d", (void *)o->mem, (unsigned long long)o->size, r); vkFreeMemory(g_vk.dev, o->dm, NULL); vkDestroyBuffer(g_vk.dev, o->buf, NULL); return -1; }
    memcpy(drv, o->mem, o->size);
    void *at = mremap(drv, o->size, o->size, MREMAP_MAYMOVE | MREMAP_FIXED, o->mem);
    if (at == MAP_FAILED) {
        VLOG("[RED] pool %p+%#llx: moving the driver's mapping failed (errno %d): left in ordinary memory", (void *)o->mem, (unsigned long long)o->size, errno);
        vkUnmapMemory(g_vk.dev, o->dm); vkFreeMemory(g_vk.dev, o->dm, NULL); vkDestroyBuffer(g_vk.dev, o->buf, NULL); return -1;
    }
    // the vacated range: kept reserved until the driver unmaps it at vkFreeMemory (asz may exceed size: the tail stays the driver's own mapping)
    if (mmap(drv, o->size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0) == MAP_FAILED) { extern uint64_t g_fence; VLOG("pool %p: placeholder over the driver's old range %p failed (errno %d) at submission %llu", (void *)o->mem, (void *)drv, errno, (unsigned long long)__atomic_load_n(&g_fence, __ATOMIC_ACQUIRE)); }
    if ((r = vkBindBufferMemory(g_vk.dev, o->buf, o->dm, 0))) VLOG("[RED] pool %p: vkBindBufferMemory %d", (void *)o->mem, r);
    return 0;
}
static void pool_free_gpu(void *arg) {
    vk_pool *o = arg;
    vkDestroyBuffer(g_vk.dev, o->buf, NULL);
    vkUnmapMemory(g_vk.dev, o->dm); vkFreeMemory(g_vk.dev, o->dm, NULL);
    free(o);
}
// Back to ordinary pages with the same bytes, then the device memory goes
static void pool_unback(vk_pool *o) {
    uint8_t *copy = malloc(o->size);
    if (copy) memcpy(copy, o->mem, o->size);
    if (mmap(o->mem, o->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) VLOG("[RED] pool %p: ordinary pages could not replace it (errno %d)", (void *)o->mem, errno);
    else if (copy) memcpy(o->mem, copy, o->size);
    free(copy);
    // The buffer and its memory go once the GPU is done: a submission still being recorded (or in flight) may read them. The CPU side moved
    // back above at once, because the game reuses the memory as soon as the pool is finalized (and the GPU sees the same bytes until then)
    extern uint64_t g_fence;
    static int leak = -1; if (leak < 0) leak = getenv("DM_VK_POOL_LEAK") != NULL;   // diagnostic: the GPU side is never freed
    VLOG("pool %p+%#llx released at submission %llu%s", (void *)o->mem, (unsigned long long)o->size, (unsigned long long)__atomic_load_n(&g_fence, __ATOMIC_ACQUIRE), leak ? " (GPU side kept: DM_VK_POOL_LEAK)" : "");
    if (leak) return;
    vk_pool *dead = malloc(sizeof *dead); *dead = *o;
    vkx_defer(pool_free_gpu, dead);
}

void vk_pool_add(const void *mem, uint64_t size) {
    if (!pools_on() || !mem || !size || !vk_ready() || g_vk.pool_type == ~0u) return;
    vk_pool o = {(uint8_t *)mem, size, VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (pool_back(&o)) { g_pool_fail++; return; }
    pthread_rwlock_wrlock(&g_pools_mu);
    if (g_npools == g_cpools) { g_cpools = g_cpools ? g_cpools * 2 : 64; g_pools = realloc(g_pools, (size_t)g_cpools * sizeof *g_pools); }
    int i = g_npools; while (i > 0 && g_pools[i - 1].mem > o.mem) { g_pools[i] = g_pools[i - 1]; i--; }   // sorted by address
    g_pools[i] = o; g_npools++; g_pool_bytes += size;
    int n = g_npools; uint64_t total = g_pool_bytes;
    pthread_rwlock_unlock(&g_pools_mu);
    if (n <= 8 || !(n & (n - 1))) VLOG("pool %p+%#llx on the GPU (%d pools, %.1f MB)", mem, (unsigned long long)size, n, total / 1048576.0);
}
void vk_pool_del(const void *mem) {
    vk_pool o = {0}; int found = 0;
    pthread_rwlock_wrlock(&g_pools_mu);
    for (int i = 0; i < g_npools; i++) if (g_pools[i].mem == mem) { o = g_pools[i]; memmove(&g_pools[i], &g_pools[i + 1], (size_t)(g_npools - i - 1) * sizeof *g_pools); g_npools--; g_pool_bytes -= o.size; found = 1; break; }
    pthread_rwlock_unlock(&g_pools_mu);
    if (found) pool_unback(&o);   // no vkQueueWaitIdle here: the queue belongs to the submitting thread (unsynchronized use lost the device)
}
// The pool holding [addr, addr + len): its buffer and the offset in it
int vk_pool_find(uint64_t addr, uint64_t len, VkBuffer *buf, uint64_t *off) {
    int ok = 0;
    pthread_rwlock_rdlock(&g_pools_mu);
    int lo = 0, hi = g_npools - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2; const vk_pool *p = &g_pools[m];
        if (addr < (uint64_t)(uintptr_t)p->mem) hi = m - 1;
        else if (addr >= (uint64_t)(uintptr_t)p->mem + p->size) lo = m + 1;
        else { if (addr + len <= (uint64_t)(uintptr_t)p->mem + p->size) { *buf = p->buf; *off = addr - (uint64_t)(uintptr_t)p->mem; ok = 1; } break; }
    }
    pthread_rwlock_unlock(&g_pools_mu);
    return ok;
}
// As vk_pool_find, but a range that starts inside a pool and runs past its end resolves too, *avail being the bytes from addr to the pool's end (the whole len when it fits).
// 0 when addr is in no pool. NVN binds uniform buffers and vertex streams with a size the shader need not read to its end; a binding near a pool's end (a ring buffer's
// position moves each frame) must not lose its data
int vk_pool_find_part(uint64_t addr, uint64_t len, VkBuffer *buf, uint64_t *off, uint64_t *avail) {
    int ok = 0;
    pthread_rwlock_rdlock(&g_pools_mu);
    int lo = 0, hi = g_npools - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2; const vk_pool *p = &g_pools[m];
        uint64_t base = (uint64_t)(uintptr_t)p->mem, end = base + p->size;
        if (addr < base) hi = m - 1;
        else if (addr >= end) lo = m + 1;
        else { *buf = p->buf; *off = addr - base; *avail = addr + len <= end ? len : end - addr; ok = 1; break; }
    }
    pthread_rwlock_unlock(&g_pools_mu);
    return ok;
}
void vk_pool_stats(int *n, uint64_t *bytes, int *failed) { pthread_rwlock_rdlock(&g_pools_mu); *n = g_npools; *bytes = g_pool_bytes; *failed = g_pool_fail; pthread_rwlock_unlock(&g_pools_mu); }
