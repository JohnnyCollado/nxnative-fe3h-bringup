// Lossless Scaling frame generation for the Android host (lsfg_glue.h): Eden's LSFG pass chain (video_core/, copied unchanged from the Eden fork) run on the
// host's device. In Eden the present manager drives it (renderer_vulkan/present/frame_gen.cpp, frame_gen_pacer.cpp); here nvn_vk_front.c does, one generated frame
// between each pair of real ones (2x), written into an image of this module that the screen code stretches into an app buffer of its own, sent half a frame
// interval before the real one.
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project (the frame generation sources this drives)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdio>
#include <cstdlib>
#include <optional>
#include "lsfg_glue.h"
#include "video_core/frame_gen/lossless_dll.h"
#include "video_core/renderer_vulkan/present/lsfg_chain.h"
#include "video_core/renderer_vulkan/present/lsfg_shaders.h"
#include "video_core/renderer_vulkan/present/util.h"

static VkDevice g_dev; static VkPhysicalDevice g_pd;
VkDevice lsfg_vk_device() { return g_dev; }
VkPhysicalDevice lsfg_vk_physical() { return g_pd; }

#define LLOG(...) do { std::fprintf(stderr, "[dm] [lsfg] " __VA_ARGS__); std::fputc('\n', stderr); } while (0)

namespace {
using namespace Vulkan;
Device g_device; MemoryAllocator g_alloc;
std::optional<LsfgShaders> g_shaders;
std::optional<LsfgChain> g_chain;
LsfgImage g_out;                 // the generated frame
uint32_t g_w, g_h; VkFormat g_fmt; float g_flow;
uint64_t g_count;                // real frames copied into the history since the chain was built or reset
int g_ok = -1;

float flow_scale() {   // the optical flow's resolution against the output: Eden's automatic choice gives 0.5 for a 960-wide picture shown at 1920 (DM_LSFG_FLOW_SCALE)
    const char* e = std::getenv("DM_LSFG_FLOW_SCALE"); float v = e ? (float)std::atof(e) : 0.5f;
    return v < 0.25f ? 0.25f : v > 1.0f ? 1.0f : v;
}
const char* status_text(VideoCore::FrameGen::LosslessStatus s) {
    using S = VideoCore::FrameGen::LosslessStatus;
    switch (s) {
    case S::Ok: return "ready"; case S::NotInstalled: return "no Lossless.dll (import it in the launcher's settings)"; case S::UnreadableFile: return "Lossless.dll cannot be read";
    case S::NotPortableExecutable: return "Lossless.dll is not a Windows DLL"; case S::MissingShaders: return "Lossless.dll has none of the expected shaders (another version?)";
    case S::TranslationFailed: return "its shaders could not be prepared"; case S::CacheUnusable: return "the shader cache could not be written";
    }
    return "?";
}
} // namespace

extern "C" int lsfg_init(VkDevice dev, VkPhysicalDevice pd, int memory_model, int null_descriptor, int float16) {
    if (g_ok >= 0) return g_ok;
    g_dev = dev; g_pd = pd; g_ok = 0;
    g_device.memory_model = memory_model != 0; g_device.null_descriptor = null_descriptor != 0; g_device.float16 = float16 != 0;
    if (!memory_model || !null_descriptor || !float16) {
        LLOG("[RED] frame generation off: the device lacks %s%s%s", memory_model ? "" : "the Vulkan memory model ", null_descriptor ? "" : "null descriptors ", float16 ? "" : "16-bit float shaders");
        return 0;
    }
    const auto st = VideoCore::FrameGen::BuildShaderCache();
    if (st != VideoCore::FrameGen::LosslessStatus::Ok) { LLOG("[RED] frame generation off: %s (%s)", status_text(st), VideoCore::FrameGen::GetLosslessDllPath().string().c_str()); return 0; }
    g_shaders.emplace(g_device);
    if (!g_shaders->IsValid()) { LLOG("[RED] frame generation off: the shaders from Lossless.dll did not load"); g_shaders.reset(); return 0; }
    LLOG("frame generation ready: Lossless Scaling's shaders from %s, one generated frame between each two (2x)", VideoCore::FrameGen::GetLosslessDllPath().string().c_str());
    return g_ok = 1;
}

extern "C" void lsfg_reset(void) { g_count = 0; }

extern "C" VkImageView lsfg_output(void) { return g_out.View(); }

extern "C" int lsfg_frame(VkCommandBuffer cb, VkImage src, VkFormat fmt, uint32_t w, uint32_t h) {
    if (g_ok != 1) return 0;
    const float flow = flow_scale();
    if (!g_chain || w != g_w || h != g_h || fmt != g_fmt || flow != g_flow) {   // (re)built at the first frame and when the size changes; the caller's earlier frames are done
        vkDeviceWaitIdle(g_dev);
        g_chain.reset();
        g_chain.emplace(g_device, g_alloc, *g_shaders, VkExtent2D{w, h}, fmt, flow);
        g_out = LsfgImage(g_device, g_alloc, VkExtent2D{w, h}, fmt);
        g_w = w; g_h = h; g_fmt = fmt; g_flow = flow; g_count = 0;
        LLOG("frame generation chain for %ux%u (format %d), optical flow at %.2f", w, h, (int)fmt, flow);
    }
    const uint64_t n = g_count++;
    vk::CommandBuffer c{cb};
    // this frame into the history (as Eden's CopyPresentedFrame: the source stays GENERAL)
    LsfgImage& in = g_chain->Input(n);
    VkImageMemoryBarrier pre[2] = {
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, src, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, in.Layout(), VK_IMAGE_LAYOUT_GENERAL,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, in.Handle(), {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, pre);
    VkImageCopy r{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {w, h, 1}};
    vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, in.Handle(), VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    VkImageMemoryBarrier post = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, in.Handle(), {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &post);
    in.SetLayout(VK_IMAGE_LAYOUT_GENERAL);
    if (n < 1) return 0;                         // the passes compare two frames
    g_chain->DispatchShared(c, n);
    if (n < 2) return 0;                         // as Eden: generate once the shared passes have run on two frames in a row
    g_chain->SetTarget(g_device, 1, 0, 0, g_out.View());
    g_chain->DispatchGeneration(c, n, 1, 0, 0, g_out.Handle(), VkExtent2D{w, h});
    VkImageMemoryBarrier done = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_out.Handle(), {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &done);
    g_out.SetLayout(VK_IMAGE_LAYOUT_GENERAL);
    return 1;
}
