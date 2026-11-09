// The Android host's Vulkan backend (shared state between its files): the device and the NVN memory pools as GPU memory (nvn_vk_mem.c).
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef struct {
    int ok;                              // the device exists
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t pool_type;                  // the memory type pools are backed by (host visible + coherent, cached when possible), ~0 if none
    VkPhysicalDeviceLimits lim;
    int vi_dyn, null_desc, clip_ctl, idx_u8, border, divisor;   // extension features turned on: vertex input dynamic state, robustness2 nullDescriptor, depth clip control,
                                                                // 8-bit indices, custom border colours, vertex attribute divisors
} vk_ctx;
extern vk_ctx g_vk;

int vk_ready(void);                                       // creates the device at the first call; 0 when there is no usable Vulkan device
void vk_pool_add(const void *mem, uint64_t size);         // an NVN memory pool becomes device memory at its own address
void vk_pool_del(const void *mem);                        // back to ordinary pages, same bytes
int vk_pool_find(uint64_t addr, uint64_t len, VkBuffer *buf, uint64_t *off);   // the pool buffer holding [addr, addr + len)
void vk_pool_stats(int *n, uint64_t *bytes, int *failed);
// nvn_vk_exec.c: fn(arg) runs once the GPU has finished every submission that may still use what it frees (at once when the executor is off)
void vkx_defer(void (*fn)(void *), void *arg);
