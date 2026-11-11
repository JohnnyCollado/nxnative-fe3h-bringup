// Compatibility layer (dm/third_party/lsfg-eden/README.md): Eden's MemoryAllocator as one allocation per object on the host's device (frame generation makes a
// few dozen images and small buffers when it starts or the size changes, never per frame).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "video_core/vulkan_common/vulkan_wrapper.h"
VkPhysicalDevice lsfg_vk_physical();   // lsfg_glue.cpp
namespace Vulkan {
enum class MemoryUsage { DeviceLocal, Upload, Download };
class MemoryAllocator {
public:
    vk::Image CreateImage(const VkImageCreateInfo& ci) {
        VkDevice d = lsfg_vk_device(); VkImage img = VK_NULL_HANDLE;
        if (vkCreateImage(d, &ci, nullptr, &img)) return {};
        VkMemoryRequirements r; vkGetImageMemoryRequirements(d, img, &r);
        VkDeviceMemory m = Alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (!m || vkBindImageMemory(d, img, m, 0)) { vkDestroyImage(d, img, nullptr); if (m) vkFreeMemory(d, m, nullptr); return {}; }
        return vk::Image{img, m};
    }
    vk::Buffer CreateBuffer(const VkBufferCreateInfo& ci, MemoryUsage usage) {
        VkDevice d = lsfg_vk_device(); VkBuffer b = VK_NULL_HANDLE;
        if (vkCreateBuffer(d, &ci, nullptr, &b)) return {};
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(d, b, &r);
        const bool host = usage != MemoryUsage::DeviceLocal; u32 flags = 0;
        VkDeviceMemory m = Alloc(r, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, host ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0, &flags);
        void* map = nullptr;
        if (!m || vkBindBufferMemory(d, b, m, 0) || (host && vkMapMemory(d, m, 0, VK_WHOLE_SIZE, 0, &map))) { vkDestroyBuffer(d, b, nullptr); if (m) vkFreeMemory(d, m, nullptr); return {}; }
        return vk::Buffer{b, m, map, ci.size, (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0};
    }
private:
    // a memory type with all of need (and want if one has it)
    static VkDeviceMemory Alloc(const VkMemoryRequirements& r, u32 need, u32 want, u32* got = nullptr) {
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(lsfg_vk_physical(), &mp);
        int pick = -1;
        for (u32 pass = 0; pass < 2 && pick < 0; pass++)
            for (u32 i = 0; i < mp.memoryTypeCount; i++) {
                const u32 f = mp.memoryTypes[i].propertyFlags, w = pass ? need : need | want;
                if ((r.memoryTypeBits >> i & 1) && (f & w) == w) { pick = (int)i; break; }
            }
        if (pick < 0) return VK_NULL_HANDLE;
        if (got) *got = mp.memoryTypes[pick].propertyFlags;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, (u32)pick};
        VkDeviceMemory m = VK_NULL_HANDLE; if (vkAllocateMemory(lsfg_vk_device(), &ai, nullptr, &m)) return VK_NULL_HANDLE;
        return m;
    }
};
} // namespace Vulkan
