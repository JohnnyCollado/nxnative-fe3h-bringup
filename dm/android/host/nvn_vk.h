// The Android host's Vulkan backend (shared state between its files): the device and the NVN memory pools as GPU memory (nvn_vk_mem.c).
#pragma once
#include <stdint.h>
#include "volk.h"   // Vulkan through function pointers from the chosen driver (vk_driver.c)

#ifdef __cplusplus
extern "C" {   // C++ users (fsr2_glue.cpp) see the backend's symbols with their C names
#endif

typedef struct {
    int ok;                              // the device exists
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t pool_type;                  // the memory type pools are backed by (host visible + coherent, cached when possible), ~0 if none
    VkPhysicalDeviceLimits lim;
    uint32_t sg_size, sg_min, sg_max; int sg_pin_cs, sg_pin_fs, sg_full, host_qreset;   // host_qreset: query pools can be reset from the CPU (nvn_vk_exec.c prof_*)
    uint32_t ts_bits;   // the queue family's timestamp bits (0: no timestamps)
    int _prof_pad;   // the device's subgroup size and its range; compute pipelines are pinned to 64 lanes (what the shaders are translated for: vk_spirv.py SUBGROUP)
    int mem_model, fp16;   // the Vulkan memory model and 16-bit float shaders turned on (what the Lossless frame generation shaders need)
    int nonuniform;                      // shaderSampledImageArrayNonUniformIndexing is on (the draw path marks translated shaders' texture array accesses)
    int vi_dyn, null_desc, clip_ctl, idx_u8, border, divisor;   // extension features turned on: vertex input dynamic state, robustness2 nullDescriptor, depth clip control,
                                                                // 8-bit indices, custom border colours, vertex attribute divisors
} vk_ctx;
extern vk_ctx g_vk;

int vk_ready(void);                                       // creates the device at the first call; 0 when there is no usable Vulkan device
void vk_pool_add(const void *mem, uint64_t size);         // an NVN memory pool becomes device memory at its own address
void vk_pool_del(const void *mem);                        // back to ordinary pages, same bytes
int vk_pool_find(uint64_t addr, uint64_t len, VkBuffer *buf, uint64_t *off);   // the pool buffer holding [addr, addr + len)
void vk_pool_stats(int *n, uint64_t *bytes, int *failed);
void vk_pool_churn(uint64_t *adds, uint64_t *add_b, uint64_t *dels, uint64_t *del_b);
void vkx_defer_bytes(int64_t d);   // bytes of GPU memory whose free waits for the GPU (vkx_defer)
// nvn_vk_exec.c: fn(arg) runs once the GPU has finished every submission that may still use what it frees (at once when the executor is off)
void vkx_defer(void (*fn)(void *), void *arg);

#ifdef __cplusplus
}
#endif
