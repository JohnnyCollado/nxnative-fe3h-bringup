// Compatibility layer (dm/third_party/lsfg-eden/README.md): the host's Vulkan device as Eden's Device, with the three features the Lossless shaders need
// (filled by lsfg_glue.cpp from what nvn_vk_mem.c turned on).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "video_core/vulkan_common/vulkan_wrapper.h"
namespace Vulkan {
class Device {
public:
    bool memory_model{}, null_descriptor{}, float16{};
    [[nodiscard]] const vk::Device& GetLogical() const { return logical; }
    [[nodiscard]] bool IsVulkanMemoryModelSupported() const { return memory_model; }
    [[nodiscard]] bool HasNullDescriptor() const { return null_descriptor; }
    [[nodiscard]] bool IsFloat16Supported() const { return float16; }
private:
    vk::Device logical;
};
} // namespace Vulkan
