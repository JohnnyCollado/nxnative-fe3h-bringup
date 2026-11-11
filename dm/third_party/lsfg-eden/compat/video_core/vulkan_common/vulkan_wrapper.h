// Compatibility layer (dm/third_party/lsfg-eden/README.md): the part of Eden's vk:: wrapper the frame generation sources use. Owning handles destroy their
// object on the host's device (lsfg_vk_device(), set by lsfg_glue.cpp); vk::CommandBuffer is a plain handle with the few commands the passes record.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>
#include "volk.h"
#include "common/common_types.h"

VkDevice lsfg_vk_device();   // lsfg_glue.cpp

namespace Vulkan::vk {

template <typename H, void (*Destroy)(H)>
class Handle {
public:
    Handle() = default;
    explicit Handle(H h) : h{h} {}
    ~Handle() { Reset(); }
    Handle(Handle&& o) noexcept : h{std::exchange(o.h, (H)VK_NULL_HANDLE)} {}
    Handle& operator=(Handle&& o) noexcept { if (this != &o) { Reset(); h = std::exchange(o.h, (H)VK_NULL_HANDLE); } return *this; }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    H operator*() const { return h; }
    explicit operator bool() const { return h != VK_NULL_HANDLE; }
    void Reset() { if (h != VK_NULL_HANDLE) { Destroy(h); h = VK_NULL_HANDLE; } }
private:
    H h = VK_NULL_HANDLE;
};
inline void DestroyImageView(VkImageView h) { vkDestroyImageView(lsfg_vk_device(), h, nullptr); }
inline void DestroySampler(VkSampler h) { vkDestroySampler(lsfg_vk_device(), h, nullptr); }
inline void DestroyDsl(VkDescriptorSetLayout h) { vkDestroyDescriptorSetLayout(lsfg_vk_device(), h, nullptr); }
inline void DestroyPipelineLayout(VkPipelineLayout h) { vkDestroyPipelineLayout(lsfg_vk_device(), h, nullptr); }
inline void DestroyPipeline(VkPipeline h) { vkDestroyPipeline(lsfg_vk_device(), h, nullptr); }
inline void DestroyDescriptorPool(VkDescriptorPool h) { vkDestroyDescriptorPool(lsfg_vk_device(), h, nullptr); }
inline void DestroyShaderModule(VkShaderModule h) { vkDestroyShaderModule(lsfg_vk_device(), h, nullptr); }
using ImageView = Handle<VkImageView, DestroyImageView>;
using Sampler = Handle<VkSampler, DestroySampler>;
using DescriptorSetLayout = Handle<VkDescriptorSetLayout, DestroyDsl>;
using PipelineLayout = Handle<VkPipelineLayout, DestroyPipelineLayout>;
using Pipeline = Handle<VkPipeline, DestroyPipeline>;
using DescriptorPool = Handle<VkDescriptorPool, DestroyDescriptorPool>;
using ShaderModule = Handle<VkShaderModule, DestroyShaderModule>;

// an image with its own memory
class Image {
public:
    Image() = default;
    Image(VkImage i, VkDeviceMemory m) : img{i}, mem{m} {}
    ~Image() { Reset(); }
    Image(Image&& o) noexcept : img{std::exchange(o.img, (VkImage)VK_NULL_HANDLE)}, mem{std::exchange(o.mem, (VkDeviceMemory)VK_NULL_HANDLE)} {}
    Image& operator=(Image&& o) noexcept { if (this != &o) { Reset(); img = std::exchange(o.img, (VkImage)VK_NULL_HANDLE); mem = std::exchange(o.mem, (VkDeviceMemory)VK_NULL_HANDLE); } return *this; }
    VkImage operator*() const { return img; }
    explicit operator bool() const { return img != VK_NULL_HANDLE; }
    void Reset() { if (img) vkDestroyImage(lsfg_vk_device(), img, nullptr); if (mem) vkFreeMemory(lsfg_vk_device(), mem, nullptr); img = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; }
private:
    VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
};

// a buffer with its own memory, mapped when host-visible
class Buffer {
public:
    Buffer() = default;
    Buffer(VkBuffer b, VkDeviceMemory m, void* map, VkDeviceSize n, bool coherent) : buf{b}, mem{m}, ptr{static_cast<u8*>(map)}, size{n}, coh{coherent} {}
    ~Buffer() { Reset(); }
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) { Reset(); buf = std::exchange(o.buf, (VkBuffer)VK_NULL_HANDLE); mem = std::exchange(o.mem, (VkDeviceMemory)VK_NULL_HANDLE); ptr = std::exchange(o.ptr, nullptr); size = o.size; coh = o.coh; }
        return *this;
    }
    VkBuffer operator*() const { return buf; }
    explicit operator bool() const { return buf != VK_NULL_HANDLE; }
    std::span<u8> Mapped() const { return {ptr, ptr ? static_cast<std::size_t>(size) : 0}; }
    void Flush() const { if (!coh && mem) { VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, mem, 0, VK_WHOLE_SIZE}; vkFlushMappedMemoryRanges(lsfg_vk_device(), 1, &r); } }
    void Invalidate() const { if (!coh && mem) { VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, mem, 0, VK_WHOLE_SIZE}; vkInvalidateMappedMemoryRanges(lsfg_vk_device(), 1, &r); } }
    void Reset() { if (buf) vkDestroyBuffer(lsfg_vk_device(), buf, nullptr); if (mem) vkFreeMemory(lsfg_vk_device(), mem, nullptr); buf = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; ptr = nullptr; }
private:
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; u8* ptr = nullptr; VkDeviceSize size = 0; bool coh = true;
};

// descriptor sets from a pool (freed with the pool)
class DescriptorSets {
public:
    DescriptorSets() = default;
    explicit DescriptorSets(std::vector<VkDescriptorSet> s) : sets{std::move(s)} {}
    VkDescriptorSet operator[](std::size_t i) const { return sets[i]; }
    std::size_t size() const { return sets.size(); }
private:
    std::vector<VkDescriptorSet> sets;
};

class CommandBuffer {
public:
    CommandBuffer() = default;
    CommandBuffer(VkCommandBuffer c) : cb{c} {}
    VkCommandBuffer operator*() const { return cb; }
    void PipelineBarrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, std::span<const VkMemoryBarrier> mem,
                         std::span<const VkBufferMemoryBarrier> buf, std::span<const VkImageMemoryBarrier> img) const {
        vkCmdPipelineBarrier(cb, src, dst, dep, (u32)mem.size(), mem.data(), (u32)buf.size(), buf.data(), (u32)img.size(), img.data());
    }
    void PipelineBarrier(VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, std::span<const VkMemoryBarrier> mem,
                         std::span<const VkBufferMemoryBarrier> buf, const VkImageMemoryBarrier& img) const {
        PipelineBarrier(src, dst, dep, mem, buf, std::span<const VkImageMemoryBarrier>(&img, 1));
    }
    void BindPipeline(VkPipelineBindPoint bp, VkPipeline p) const { vkCmdBindPipeline(cb, bp, p); }
    void BindDescriptorSets(VkPipelineBindPoint bp, VkPipelineLayout l, u32 first, VkDescriptorSet s, std::span<const u32> dyn) const {
        vkCmdBindDescriptorSets(cb, bp, l, first, 1, &s, (u32)dyn.size(), dyn.data());
    }
    void Dispatch(u32 x, u32 y, u32 z) const { vkCmdDispatch(cb, x, y, z); }
    void CopyImage(VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, const VkImageCopy& r) const { vkCmdCopyImage(cb, src, sl, dst, dl, 1, &r); }
private:
    VkCommandBuffer cb = VK_NULL_HANDLE;
};

// the device's calls the sources make through Device::GetLogical()
class Device {
public:
    void UpdateDescriptorSets(std::span<const VkWriteDescriptorSet> w, std::span<const VkCopyDescriptorSet> c) const {
        vkUpdateDescriptorSets(lsfg_vk_device(), (u32)w.size(), w.data(), (u32)c.size(), c.data());
    }
    Sampler CreateSampler(const VkSamplerCreateInfo& ci) const { VkSampler s = VK_NULL_HANDLE; vkCreateSampler(lsfg_vk_device(), &ci, nullptr, &s); return Sampler{s}; }
};

} // namespace Vulkan::vk
