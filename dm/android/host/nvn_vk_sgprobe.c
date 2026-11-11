// Subgroup layout probe: at start-up, one small compute job per work group shape the warp shaders use, and a log of how the driver groups its threads.
//
// The guest's compute shaders pass values between the 32 threads of a warp: threads 32 w .. 32 w + 31 of the work group, in order (SHFL, REDUX, MATCH). The translation
// runs them on host subgroups and assumes lane L of a 64-lane subgroup is thread 64 k + L, so that the subgroup's two halves are two warps (dm/third_party/ryujinx:
// EmitterContextInsts.WarpGather, InstEmitAmpereTensor.ARedux). Vulkan promises neither that order nor, without VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
// full subgroups, and a PC GPU with 32-lane subgroups never runs the second half. This runs the translation's own forms (shaders/sgprobe.comp) and compares them with
// what a guest warp would give. Lines start "sgprobe"; a wrong layout is a [RED] line. DM_VK_SG_PROBE=0 skips it.
// Shapes: 256 x 1 (Fire Emblem Fortune's Weave's light culling and tile depth bounds) and 16 x 16 (its exposure histogram), each as the host creates compute
// pipelines today (no flags) and, where Vulkan allows it (x a multiple of the subgroup size), with full subgroups required.
// Built alone for a PC with -DSGPROBE_PC (the work folder's re/sgprobe).
#ifdef SGPROBE_PC
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define VLOG(...) do { printf(__VA_ARGS__); putchar('\n'); } while (0)
static struct { VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam; VkPhysicalDeviceMemoryProperties mp; uint32_t sg_size, sg_min, sg_max; int sg_pin_cs, sg_full; } g_vk;
#else
#include "nvn_vk_int.h"
#endif
#include "nvn_vk_sgprobe_spv.h"

enum { SGP_THREADS = 256, SGP_GROUPS = 2, SGP_WORDS = 12 };

static void sgp_run(VkShaderModule mod, VkDescriptorSetLayout dsl, VkPipelineLayout pl, VkDescriptorSet ds, VkCommandBuffer cb, uint32_t *out, uint32_t lx, uint32_t ly, int full) {
    char name[64]; snprintf(name, sizeof name, "%u x %u%s", lx, ly, full ? ", full subgroups required" : "");
    (void)dsl;
    uint32_t sp[2] = {lx, ly}; VkSpecializationMapEntry se[2] = {{0, 0, 4}, {1, 4, 4}}; VkSpecializationInfo si = {2, se, sizeof sp, sp};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rsg = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 64};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, g_vk.sg_pin_cs ? &rsg : NULL, full ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
         VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", &si}, pl, VK_NULL_HANDLE, -1};
    VkPipeline pipe; VkResult r = vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe);
    if (r) { VLOG("sgprobe %s: pipeline failed (%d)", name, r); return; }
    memset(out, 0xee, SGP_GROUPS * SGP_THREADS * SGP_WORDS * 4);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkResetCommandBuffer(cb, 0); vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, SGP_GROUPS, 1, 1); vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; vkCreateFence(g_vk.dev, &fi, NULL, &f);
    VkSubmitInfo sub = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    r = vkQueueSubmit(g_vk.q, 1, &sub, f); if (!r) r = vkWaitForFences(g_vk.dev, 1, &f, VK_TRUE, 2000000000ull);
    vkDestroyFence(g_vk.dev, f, NULL);
    if (r) { VLOG("sgprobe %s: run failed (%d)", name, r); vkDestroyPipeline(g_vk.dev, pipe, NULL); return; }

    // what a guest warp gives thread i: the sum and the minimum of its warp's thread numbers, thread 5 of its warp, and the sum with every fourth thread inactive
    int n = SGP_GROUPS * SGP_THREADS, ran = 0, linear = 0, sum_ok = 0, min_ok = 0, gather_ok = 0, part_ok = 0, part_n = 0, block_ok = 0;
    uint32_t size = 0, nsub = 0, bal_min = ~0u, bal_max = 0, pbal_min = ~0u, pbal_max = 0;
    for (int t = 0; t < n; t++) {
        const uint32_t *o = out + (size_t)t * SGP_WORDS; uint32_t i = (uint32_t)t % SGP_THREADS, w = i >> 5;
        if (o[11] != 0x600d) continue;
        ran++; size = o[1]; nsub = o[6];
        if (o[3] < bal_min) bal_min = o[3]; if (o[3] > bal_max) bal_max = o[3];
        linear += size && o[0] == i % size;
        block_ok += size && o[2] / size == i / size;   // its subgroup's first lane is a thread of the same block of `size` consecutive threads
        sum_ok += o[4] == 32 * 32 * w + 496; min_ok += o[8] == 32 * w; gather_ok += o[5] == 32 * w + 5;
        if (i & 3) { part_n++; part_ok += o[9] == 24 * 32 * w + 384; if (o[10] < pbal_min) pbal_min = o[10]; if (o[10] > pbal_max) pbal_max = o[10]; }
    }
    int good = ran == n && sum_ok == n && min_ok == n && gather_ok == n && part_ok == part_n;
    VLOG("%ssgprobe %s: %d of %d threads ran; subgroup size %u, %u subgroups a work group, active lanes %u-%u (with a quarter off: %u-%u); lane = thread %% size for %d, "
         "subgroup within its block of consecutive threads for %d; as a guest warp: sum %d, min %d, gather %d of %d, sum with a quarter off %d of %d",
         good ? "" : "[RED] ", name, ran, n, size, nsub, bal_min, bal_max, pbal_min, pbal_max, linear, block_ok, sum_ok, min_ok, gather_ok, n, part_ok, part_n);
    if (!good || linear != n) {   // the layout itself: each thread's lane, then the first thread of its subgroup, 32 threads a row (the first work group)
        for (int row = 0; row < SGP_THREADS / 32; row++) {
            char a[160], b[200]; int na = 0, nb = 0;
            for (int c = 0; c < 32; c++) {
                const uint32_t *o = out + (size_t)(row * 32 + c) * SGP_WORDS;
                na += snprintf(a + na, sizeof a - na, "%02x ", o[0] & 0xff); nb += snprintf(b + nb, sizeof b - nb, "%3u ", o[2] & 0x3ff);
            }
            VLOG("sgprobe %s: threads %3d-%3d lanes %s| subgroup's first thread %s", name, row * 32, row * 32 + 31, a, b);
        }
        for (int t = 0, shown = 0; t < n && shown < 6; t++) {
            const uint32_t *o = out + (size_t)t * SGP_WORDS; uint32_t i = (uint32_t)t % SGP_THREADS, w = i >> 5;
            if (o[4] == 32 * 32 * w + 496 && o[5] == 32 * w + 5 && (!(i & 3) || o[9] == 24 * 32 * w + 384)) continue;
            VLOG("sgprobe %s: group %d thread %u lane %u: sum %u (warp: %u), gather %u (%u), sum with a quarter off %u (%u), active %u / %u", name, t / SGP_THREADS, i, o[0], o[4], 32 * 32 * w + 496,
                 o[5], 32 * w + 5, o[9], 24 * 32 * w + 384, o[3], o[10]); shown++;
        }
    }
    vkDestroyPipeline(g_vk.dev, pipe, NULL);
}

void vkx_sg_probe(void) {
    const char *e = getenv("DM_VK_SG_PROBE"); if (e && e[0] == '0') return;
    VkDevice d = g_vk.dev; VkResult r;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_sgprobe_comp, k_sgprobe_comp}; VkShaderModule mod;
    if ((r = vkCreateShaderModule(d, &mi, NULL, &mod))) { VLOG("sgprobe: shader module failed (%d)", r); return; }
    VkDeviceSize bytes = SGP_GROUPS * SGP_THREADS * SGP_WORDS * 4;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}; VkBuffer buf; vkCreateBuffer(d, &bci, NULL, &buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(d, buf, &mr); uint32_t mt = ~0u;
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < g_vk.mp.memoryTypeCount; i++) if ((mr.memoryTypeBits >> i & 1) && (g_vk.mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt}; VkDeviceMemory mem = VK_NULL_HANDLE; void *map = NULL;
    if (mt == ~0u || vkAllocateMemory(d, &ai, NULL, &mem) || vkBindBufferMemory(d, buf, mem, 0) || vkMapMemory(d, mem, 0, bytes, 0, &map)) {
        VLOG("sgprobe: no buffer"); vkDestroyBuffer(d, buf, NULL); if (mem) vkFreeMemory(d, mem, NULL); vkDestroyShaderModule(d, mod, NULL); return;
    }
    VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &lb}; VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(d, &li, NULL, &dsl);
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl}; VkPipelineLayout pl; vkCreatePipelineLayout(d, &pli, NULL, &pl);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VkDescriptorPool dp; vkCreateDescriptorPool(d, &dpi, NULL, &dp);
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl}; VkDescriptorSet ds; vkAllocateDescriptorSets(d, &dai, &ds);
    VkDescriptorBufferInfo dbi = {buf, 0, bytes}; VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi, NULL};
    vkUpdateDescriptorSets(d, 1, &wr, 0, NULL);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, g_vk.qfam}; VkCommandPool cp; vkCreateCommandPool(d, &cpi, NULL, &cp);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cb; vkAllocateCommandBuffers(d, &cai, &cb);

    uint32_t size = g_vk.sg_pin_cs ? 64 : g_vk.sg_size;
    VLOG("sgprobe: device subgroup size %u (%u-%u)%s, full subgroups %s", g_vk.sg_size, g_vk.sg_min, g_vk.sg_max, g_vk.sg_pin_cs ? ", pinned to 64" : "", g_vk.sg_full ? "can be required" : "cannot be required");
    sgp_run(mod, dsl, pl, ds, cb, map, 256, 1, 0);
    sgp_run(mod, dsl, pl, ds, cb, map, 16, 16, 0);
    if (g_vk.sg_full && size && 256 % size == 0) sgp_run(mod, dsl, pl, ds, cb, map, 256, 1, 1);
    if (g_vk.sg_full && size && 16 % size == 0) sgp_run(mod, dsl, pl, ds, cb, map, 16, 16, 1);

    vkDestroyCommandPool(d, cp, NULL); vkDestroyDescriptorPool(d, dp, NULL); vkDestroyPipelineLayout(d, pl, NULL); vkDestroyDescriptorSetLayout(d, dsl, NULL);
    vkDestroyBuffer(d, buf, NULL); vkFreeMemory(d, mem, NULL); vkDestroyShaderModule(d, mod, NULL);
}

#ifdef SGPROBE_PC
int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "sgprobe", 0, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app}; VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst)) return 2;
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &g_vk.pd); if (!n) return 2;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(g_vk.pd, &p); printf("device: %s\n", p.deviceName);
    vkGetPhysicalDeviceMemoryProperties(g_vk.pd, &g_vk.mp);
    uint32_t nq = 16; VkQueueFamilyProperties qp[16]; vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_vk.qfam = i; break; }
    VkPhysicalDeviceVulkan13Properties q13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES}; VkPhysicalDeviceVulkan11Properties q11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &q13};
    VkPhysicalDeviceProperties2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &q11}; vkGetPhysicalDeviceProperties2(g_vk.pd, &q2);
    g_vk.sg_size = q11.subgroupSize; g_vk.sg_min = q13.minSubgroupSize; g_vk.sg_max = q13.maxSubgroupSize;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.subgroupSizeControl = 1; f13.computeFullSubgroups = 1; g_vk.sg_full = 1;
    float prio = 1; VkDeviceQueueCreateInfo dqi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, g_vk.qfam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &dqi}; if (vkCreateDevice(g_vk.pd, &dci, NULL, &g_vk.dev)) return 2;
    vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.q);
    vkx_sg_probe(); return 0;
}
#endif
