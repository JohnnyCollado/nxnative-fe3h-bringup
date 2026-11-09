// Vulkan feature probe for the Android host's planned Vulkan backend (V0): what the device's driver offers that the NVN → Vulkan translation needs.
// Prints the GPU, driver and API version; the extensions and features the backend relies on, each marked yes / NO; texture compression support (Switch games ship BC
// and ASTC textures); and two functional tests:
//   - host memory import (VK_EXT_external_memory_host): a GPU buffer made directly over anonymous memory the CPU writes, the way the Mac backend makes Metal buffers
//     over the game's memory pools; a compute-free check: the buffer is filled by vkCmdFillBuffer and the CPU reads the result through its own pointer
//   - the translated shaders: every .spv in a folder (argument 1, optional) is handed to vkCreateShaderModule, and each of the first compute shaders to a compute
//     pipeline built from its own .res.json bindings, so the driver's acceptance of the translator's SPIR-V is measured on the device
// Build: NDK clang --target=aarch64-linux-android35 vk_probe.c -lvulkan -o vk_probe. Usage on the device: ./vk_probe [spirv folder]
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <vulkan/vulkan.h>

static int g_bad;
static void mark(const char *what, int ok) { printf("  %-52s %s\n", what, ok ? "yes" : "NO"); if (!ok) g_bad++; }
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)z + 1); if (p && fread(p, 1, (size_t)z, f) != (size_t)z) { free(p); p = NULL; }
    fclose(f); if (p) { p[z] = 0; *n = (size_t)z; } return p;
}

// The bindings a .res.json declares, as descriptor set 0 (binding 0 is the translator's support buffer, a uniform buffer)
static int res_bindings(const char *json, VkDescriptorSetLayoutBinding *b, int cap, VkShaderStageFlags stage) {
    int n = 0;
    b[n++] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, NULL};
    static const struct { const char *key; VkDescriptorType type; } kinds[] = {
        {"\"cbufs\"", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {"\"sbufs\"", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {"\"textures\"", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, {"\"images\"", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}};
    for (int k = 0; k < 4; k++) {
        const char *s = strstr(json, kinds[k].key); if (!s) continue;
        const char *end = strchr(s, ']'); if (!end) continue;
        for (const char *q = s; (q = strstr(q, "\"binding\":")) && q < end; q++) {
            int bd = atoi(q + 10); if (n < cap && bd > 0) b[n++] = (VkDescriptorSetLayoutBinding){(uint32_t)bd, kinds[k].type, 1, stage, NULL};
        }
    }
    return n;
}

int main(int argc, char **argv) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative vk_probe", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VkInstance inst; VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if (r) { printf("vkCreateInstance failed: %d\n", r); return 1; }
    uint32_t npd = 0; vkEnumeratePhysicalDevices(inst, &npd, NULL);
    if (!npd) { printf("no Vulkan device\n"); return 1; }
    VkPhysicalDevice pds[4]; npd = npd > 4 ? 4 : npd; vkEnumeratePhysicalDevices(inst, &npd, pds);
    VkPhysicalDevice pd = pds[0];

    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, &drv};
    VkPhysicalDevicePushDescriptorPropertiesKHR pushp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR, &hostp};
    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pushp};
    vkGetPhysicalDeviceProperties2(pd, &props);
    VkPhysicalDeviceProperties *p = &props.properties;
    printf("GPU: %s (vendor %#x device %#x)\n", p->deviceName, p->vendorID, p->deviceID);
    printf("API %u.%u.%u, driver %s %s (version %#x)\n", VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion),
           drv.driverName, drv.driverInfo, p->driverVersion);
    printf("limits: max image 2D %u, max descriptor set samplers %u / sampled images %u / storage buffers %u, uniform buffer range %u, push constants %u B\n",
           p->limits.maxImageDimension2D, p->limits.maxDescriptorSetSamplers, p->limits.maxDescriptorSetSampledImages, p->limits.maxDescriptorSetStorageBuffers,
           p->limits.maxUniformBufferRange, p->limits.maxPushConstantsSize);
    printf("  min uniform / storage buffer offset alignment %llu / %llu, timestamp period %.3f ns\n",
           (unsigned long long)p->limits.minUniformBufferOffsetAlignment, (unsigned long long)p->limits.minStorageBufferOffsetAlignment, p->limits.timestampPeriod);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("memory heaps:"); for (uint32_t i = 0; i < mp.memoryHeapCount; i++) printf(" %.1f GB%s", mp.memoryHeaps[i].size / 1073741824.0, mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " (device local)" : "");
    printf("\nmemory types:"); for (uint32_t i = 0; i < mp.memoryTypeCount; i++) printf(" %u:%#x", i, mp.memoryTypes[i].propertyFlags); printf("\n");

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne, sizeof *ext); vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
    printf("\nextensions the backend needs or wants (%u offered in all):\n", ne);
    static const char *const want[] = {
        "VK_EXT_external_memory_host", "VK_EXT_depth_clip_control", "VK_KHR_push_descriptor", "VK_KHR_dynamic_rendering", "VK_KHR_synchronization2",
        "VK_EXT_extended_dynamic_state", "VK_EXT_extended_dynamic_state2", "VK_EXT_extended_dynamic_state3", "VK_EXT_vertex_input_dynamic_state",
        "VK_EXT_custom_border_color", "VK_EXT_provoking_vertex", "VK_EXT_transform_feedback", "VK_EXT_robustness2", "VK_EXT_depth_clip_enable",
        "VK_EXT_index_type_uint8", "VK_KHR_draw_indirect_count", "VK_EXT_primitive_topology_list_restart", "VK_EXT_conditional_rendering",
        "VK_KHR_shader_float_controls", "VK_EXT_shader_demote_to_helper_invocation", "VK_EXT_descriptor_indexing", "VK_KHR_buffer_device_address",
        "VK_EXT_line_rasterization", "VK_EXT_color_write_enable", "VK_KHR_maintenance5", "VK_EXT_attachment_feedback_loop_layout"};
    for (size_t i = 0; i < sizeof want / sizeof *want; i++) mark(want[i], has_ext(ext, ne, want[i]));

    // Extension feature structures go into the chain only for extensions the device offers
    VkPhysicalDeviceDepthClipControlFeaturesEXT dcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT rb2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    void *chain = NULL;
    if (has_ext(ext, ne, "VK_EXT_depth_clip_control")) { dcc.pNext = chain; chain = &dcc; }
    if (has_ext(ext, ne, "VK_EXT_custom_border_color")) { cbc.pNext = chain; chain = &cbc; }
    if (has_ext(ext, ne, "VK_EXT_robustness2")) { rb2.pNext = chain; chain = &rb2; }
    if (has_ext(ext, ne, "VK_EXT_extended_dynamic_state3")) { eds3.pNext = chain; chain = &eds3; }
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, chain};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f);
    VkPhysicalDeviceFeatures *f0 = &f.features;
    printf("\nfeatures:\n");
    mark("shaderStorageImageWriteWithoutFormat (38 shaders)", f0->shaderStorageImageWriteWithoutFormat);
    mark("shaderStorageImageReadWithoutFormat", f0->shaderStorageImageReadWithoutFormat);
    mark("geometryShader (10 shaders)", f0->geometryShader);
    mark("depthClamp", f0->depthClamp); mark("depthBiasClamp", f0->depthBiasClamp); mark("depthBounds", f0->depthBounds);
    mark("independentBlend", f0->independentBlend); mark("dualSrcBlend", f0->dualSrcBlend); mark("logicOp", f0->logicOp);
    mark("multiDrawIndirect", f0->multiDrawIndirect); mark("drawIndirectFirstInstance", f0->drawIndirectFirstInstance);
    mark("fillModeNonSolid", f0->fillModeNonSolid); mark("wideLines", f0->wideLines); mark("samplerAnisotropy", f0->samplerAnisotropy);
    mark("textureCompressionBC", f0->textureCompressionBC); mark("textureCompressionASTC_LDR", f0->textureCompressionASTC_LDR); mark("textureCompressionETC2", f0->textureCompressionETC2);
    mark("shaderInt64", f0->shaderInt64); mark("shaderInt16", f0->shaderInt16); mark("shaderFloat64", f0->shaderFloat64);
    mark("sampleRateShading", f0->sampleRateShading); mark("imageCubeArray", f0->imageCubeArray); mark("fragmentStoresAndAtomics", f0->fragmentStoresAndAtomics);
    mark("vertexPipelineStoresAndAtomics", f0->vertexPipelineStoresAndAtomics); mark("shaderClipDistance", f0->shaderClipDistance); mark("shaderCullDistance", f0->shaderCullDistance);
    mark("multiViewport", f0->multiViewport); mark("occlusionQueryPrecise", f0->occlusionQueryPrecise);
    mark("storageBuffer16BitAccess (1.1)", f11.storageBuffer16BitAccess); mark("shaderDrawParameters (1.1)", f11.shaderDrawParameters);
    mark("shaderFloat16 (1.2)", f12.shaderFloat16); mark("shaderInt8 (1.2)", f12.shaderInt8); mark("drawIndirectCount (1.2)", f12.drawIndirectCount);
    mark("descriptorIndexing (1.2)", f12.descriptorIndexing); mark("runtimeDescriptorArray (1.2)", f12.runtimeDescriptorArray);
    mark("descriptorBindingPartiallyBound (1.2)", f12.descriptorBindingPartiallyBound); mark("bufferDeviceAddress (1.2)", f12.bufferDeviceAddress);
    mark("timelineSemaphore (1.2)", f12.timelineSemaphore); mark("scalarBlockLayout (1.2)", f12.scalarBlockLayout); mark("shaderSubgroupExtendedTypes (1.2)", f12.shaderSubgroupExtendedTypes);
    mark("dynamicRendering (1.3)", f13.dynamicRendering); mark("synchronization2 (1.3)", f13.synchronization2); mark("maintenance4 (1.3)", f13.maintenance4);
    mark("shaderDemoteToHelperInvocation (1.3)", f13.shaderDemoteToHelperInvocation);
    if (has_ext(ext, ne, "VK_EXT_depth_clip_control")) mark("depthClipControl (NVN's -1..1 depth range)", dcc.depthClipControl);
    if (has_ext(ext, ne, "VK_EXT_custom_border_color")) mark("customBorderColors", cbc.customBorderColors);
    if (has_ext(ext, ne, "VK_EXT_robustness2")) mark("nullDescriptor (robustness2)", rb2.nullDescriptor);
    if (has_ext(ext, ne, "VK_EXT_extended_dynamic_state3")) { mark("extendedDynamicState3ColorBlendEnable", eds3.extendedDynamicState3ColorBlendEnable); mark("extendedDynamicState3ColorBlendEquation", eds3.extendedDynamicState3ColorBlendEquation); }
    printf("  push descriptors: up to %u per set; host pointer import alignment %llu B\n", pushp.maxPushDescriptors, (unsigned long long)hostp.minImportedHostPointerAlignment);

    // Texture formats Switch games use: BC1-7 and ASTC 4x4 sampled; the render targets
    printf("\nformats (sampled / color attachment / depth-stencil attachment / storage):\n");
    static const struct { VkFormat f; const char *n; } fm[] = {
        {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "BC1"}, {VK_FORMAT_BC3_UNORM_BLOCK, "BC3"}, {VK_FORMAT_BC4_UNORM_BLOCK, "BC4"}, {VK_FORMAT_BC5_UNORM_BLOCK, "BC5"},
        {VK_FORMAT_BC6H_UFLOAT_BLOCK, "BC6H"}, {VK_FORMAT_BC7_UNORM_BLOCK, "BC7"}, {VK_FORMAT_ASTC_4x4_UNORM_BLOCK, "ASTC 4x4"}, {VK_FORMAT_ASTC_8x8_SRGB_BLOCK, "ASTC 8x8 sRGB"},
        {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8"}, {VK_FORMAT_R8G8B8A8_SRGB, "RGBA8 sRGB"}, {VK_FORMAT_A2B10G10R10_UNORM_PACK32, "RGB10A2"}, {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "R11G11B10F"},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "RGBA16F"}, {VK_FORMAT_R32G32B32A32_SFLOAT, "RGBA32F"}, {VK_FORMAT_R16_SFLOAT, "R16F"}, {VK_FORMAT_R32_UINT, "R32UI"},
        {VK_FORMAT_D24_UNORM_S8_UINT, "D24S8"}, {VK_FORMAT_D32_SFLOAT, "D32F"}, {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32FS8"}, {VK_FORMAT_D16_UNORM, "D16"}};
    for (size_t i = 0; i < sizeof fm / sizeof *fm; i++) {
        VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(pd, fm[i].f, &fp); VkFormatFeatureFlags t = fp.optimalTilingFeatures;
        printf("  %-14s %-4s %-4s %-4s %s\n", fm[i].n, t & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ? "yes" : "-", t & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT ? "yes" : "-",
               t & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ? "yes" : "-", t & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? "yes" : "-");
    }

    // A device with what the tests need
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qf[8]; nq = nq > 8 ? 8 : nq; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo dqi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qi, 1, &prio};
    const char *dext[2]; uint32_t ndext = 0; int host_ok = has_ext(ext, ne, "VK_EXT_external_memory_host");
    if (host_ok) dext[ndext++] = "VK_EXT_external_memory_host";
    VkPhysicalDeviceFeatures en = {0}; en.shaderStorageImageWriteWithoutFormat = f0->shaderStorageImageWriteWithoutFormat; en.shaderStorageImageReadWithoutFormat = f0->shaderStorageImageReadWithoutFormat;
    en.shaderInt64 = f0->shaderInt64; en.shaderInt16 = f0->shaderInt16; en.fragmentStoresAndAtomics = f0->fragmentStoresAndAtomics; en.shaderFloat64 = f0->shaderFloat64;
    en.vertexPipelineStoresAndAtomics = f0->vertexPipelineStoresAndAtomics; en.imageCubeArray = f0->imageCubeArray; en.shaderClipDistance = f0->shaderClipDistance; en.geometryShader = f0->geometryShader;
    VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; e12.shaderFloat16 = f12.shaderFloat16; e12.shaderInt8 = f12.shaderInt8; e12.scalarBlockLayout = f12.scalarBlockLayout;
    VkPhysicalDeviceVulkan11Features e11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &e12}; e11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess; e11.shaderDrawParameters = f11.shaderDrawParameters;
    VkPhysicalDeviceVulkan13Features e13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &e11}; e13.shaderDemoteToHelperInvocation = f13.shaderDemoteToHelperInvocation;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &e13, 0, 1, &dqi, 0, NULL, ndext, dext, &en};
    VkDevice dev; r = vkCreateDevice(pd, &dci, NULL, &dev);
    if (r) { printf("\nvkCreateDevice failed: %d\n", r); return 1; }
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qi};
    VkCommandPool cp; vkCreateCommandPool(dev, &cpi, NULL, &cp);

    // Test 1: host memory import
    printf("\ntest: GPU buffer over the CPU's own memory (VK_EXT_external_memory_host)\n");
    if (!host_ok) printf("  skipped: extension missing\n");
    else {
        PFN_vkGetMemoryHostPointerPropertiesEXT gp = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
        size_t sz = 16 << 20; uint8_t *mem = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memset(mem, 0x11, sz);
        VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        r = gp ? gp(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem, &hp) : VK_ERROR_EXTENSION_NOT_PRESENT;
        printf("  host pointer properties: result %d, usable memory types %#x\n", r, hp.memoryTypeBits);
        uint32_t mt = 0; while (mt < 32 && !(hp.memoryTypeBits >> mt & 1)) mt++;
        VkExternalMemoryBufferCreateInfo emb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &emb, 0, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer buf; VkDeviceMemory dm; r = r ? r : vkCreateBuffer(dev, &bci, NULL, &buf);
        VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem};
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, sz, mt};
        if (!r) r = vkAllocateMemory(dev, &mai, NULL, &dm);
        if (!r) r = vkBindBufferMemory(dev, buf, dm, 0);
        if (!r) {
            VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &cai, &cb);
            VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(cb, &bi); vkCmdFillBuffer(cb, buf, 4096, 4096, 0xCAFEF00Du);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL); vkEndCommandBuffer(cb);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
            vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(q);
            uint32_t got = *(volatile uint32_t *)(mem + 4096), before = *(volatile uint32_t *)(mem + 4092), after = *(volatile uint32_t *)(mem + 8192);
            mark("the CPU reads what the GPU wrote, through its own pointer", got == 0xCAFEF00Du && before == 0x11111111u && after == 0x11111111u);
        } else { printf("  import failed: %d\n", r); g_bad++; }
    }

    // Test 2: the translator's SPIR-V on this driver
    if (argc > 1) {
        printf("\ntest: the translated shaders in %s\n", argv[1]);
        DIR *d = opendir(argv[1]); if (!d) { printf("  cannot open the folder\n"); return 1; }
        struct dirent *e; int nmod = 0, badmod = 0, ncs = 0, badcs = 0; double t0 = now_ms(), tcs = 0; char path[1024];
        while ((e = readdir(d))) {
            size_t l = strlen(e->d_name); if (l < 5 || strcmp(e->d_name + l - 4, ".spv")) continue;
            snprintf(path, sizeof path, "%s/%s", argv[1], e->d_name);
            size_t n; uint8_t *code = slurp(path, &n); if (!code) continue;
            VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n, (const uint32_t *)code};
            VkShaderModule sm; r = vkCreateShaderModule(dev, &smi, NULL, &sm); nmod++;
            if (r) { if (badmod++ < 5) printf("  module %s refused: %d\n", e->d_name, r); free(code); continue; }
            // compute shaders: a whole pipeline (the driver's compiler runs), with the bindings their .res.json declares
            snprintf(path, sizeof path, "%s/%.*s.res.json", argv[1], (int)(l - 4), e->d_name);
            size_t jn; char *json = (char *)slurp(path, &jn);
            if (json && strstr(json, "\"stage\": \"cs\"") && ncs < 200) {
                VkDescriptorSetLayoutBinding b[64]; int nb = res_bindings(json, b, 64, VK_SHADER_STAGE_COMPUTE_BIT);
                VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nb, b};
                VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkPipeline pipe; double c0 = now_ms();
                r = vkCreateDescriptorSetLayout(dev, &li, NULL, &dsl);
                VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 0, NULL};
                if (!r) r = vkCreatePipelineLayout(dev, &pli, NULL, &pl);
                VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl, VK_NULL_HANDLE, 0};
                if (!r) r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
                tcs += now_ms() - c0; ncs++;
                if (r) { if (badcs++ < 5) printf("  compute pipeline %s refused: %d\n", e->d_name, r); }
                else vkDestroyPipeline(dev, pipe, NULL);
                if (pl) vkDestroyPipelineLayout(dev, pl, NULL);
                if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, NULL);
            }
            free(json); vkDestroyShaderModule(dev, sm, NULL); free(code);
        }
        closedir(d);
        printf("  shader modules: %d, refused %d (%.0f ms in all)\n", nmod, badmod, now_ms() - t0);
        printf("  compute pipelines compiled: %d, refused %d (%.1f ms each on average)\n", ncs, badcs, ncs ? tcs / ncs : 0);
        if (badmod || badcs) g_bad++;
    }
    printf("\n%d check(s) not met\n", g_bad);
    return 0;
}
