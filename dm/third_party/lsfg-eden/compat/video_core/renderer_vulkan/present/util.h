// Compatibility layer (dm/third_party/lsfg-eden/README.md): the CreateWrapped* helpers of Eden's present/util.h that the frame generation passes call,
// on the host's device. Each binding of a layout holds one descriptor, numbered from 0 in the order given (as the Lossless shaders are renumbered, lsfg_translate.cpp).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <initializer_list>
#include <span>
#include <vector>
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
namespace Vulkan {
inline vk::ImageView CreateWrappedImageView(const Device&, const vk::Image& image, VkFormat format) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, *image, VK_IMAGE_VIEW_TYPE_2D, format, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView v = VK_NULL_HANDLE; vkCreateImageView(lsfg_vk_device(), &ci, nullptr, &v); return vk::ImageView{v};
}
inline vk::DescriptorSetLayout CreateWrappedDescriptorSetLayout(const Device&, std::span<const VkDescriptorType> types, VkShaderStageFlags stages) {
    std::vector<VkDescriptorSetLayoutBinding> b(types.size());
    for (size_t i = 0; i < types.size(); i++) b[i] = VkDescriptorSetLayoutBinding{(u32)i, types[i], 1, stages, nullptr};
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, (u32)b.size(), b.data()};
    VkDescriptorSetLayout l = VK_NULL_HANDLE; vkCreateDescriptorSetLayout(lsfg_vk_device(), &ci, nullptr, &l); return vk::DescriptorSetLayout{l};
}
inline vk::PipelineLayout CreateWrappedPipelineLayout(const Device&, const vk::DescriptorSetLayout& dsl) {
    VkDescriptorSetLayout l = *dsl;
    VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &l, 0, nullptr};
    VkPipelineLayout p = VK_NULL_HANDLE; vkCreatePipelineLayout(lsfg_vk_device(), &ci, nullptr, &p); return vk::PipelineLayout{p};
}
inline vk::Pipeline CreateWrappedComputePipeline(const Device&, const vk::PipelineLayout& layout, VkShaderModule module) {
    if (module == VK_NULL_HANDLE) return {};
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                                   {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr}, *layout, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE; vkCreateComputePipelines(lsfg_vk_device(), VK_NULL_HANDLE, 1, &ci, nullptr, &p); return vk::Pipeline{p};
}
inline vk::DescriptorPool CreateWrappedDescriptorPool(const Device&, u32 per_type, u32 max_sets, std::initializer_list<VkDescriptorType> types) {
    std::vector<VkDescriptorPoolSize> s; for (VkDescriptorType t : types) s.push_back({t, per_type});
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, max_sets, (u32)s.size(), s.data()};
    VkDescriptorPool p = VK_NULL_HANDLE; vkCreateDescriptorPool(lsfg_vk_device(), &ci, nullptr, &p); return vk::DescriptorPool{p};
}
inline vk::DescriptorSets CreateWrappedDescriptorSets(const vk::DescriptorPool& pool, const std::vector<VkDescriptorSetLayout>& layouts) {
    std::vector<VkDescriptorSet> sets(layouts.size());
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, *pool, (u32)layouts.size(), layouts.data()};
    if (!layouts.empty() && vkAllocateDescriptorSets(lsfg_vk_device(), &ai, sets.data())) sets.assign(layouts.size(), VK_NULL_HANDLE);
    return vk::DescriptorSets{std::move(sets)};
}
inline vk::ShaderModule CreateWrappedShaderModule(const Device&, std::span<const u32> words) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, words.size() * 4, words.data()};
    VkShaderModule m = VK_NULL_HANDLE; vkCreateShaderModule(lsfg_vk_device(), &ci, nullptr, &m); return vk::ShaderModule{m};
}
} // namespace Vulkan
