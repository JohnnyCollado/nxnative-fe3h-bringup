// FSR 2 (dm/third_party/fsr2) behind a C interface for the Vulkan backend: what nvn_vk_draw.c's vkd_dlss hands
// the C++ glue (fsr2_glue.cpp), which owns the context, the scratch memory and the frame clock. The job carries
// the game's DLSS command (p_dlss, nvn_obj.h) resolved to images and views, so the C side stays free of the
// FidelityFX headers.
#pragma once
#include <volk.h>

#ifdef __cplusplus
extern "C" {   // defined in fsr2_glue.cpp, called from C (nvn_vk_draw.c)
#endif

// The game's DLSS flags (the NVN2 DLSS command's flags word), as nvn_mtl.m reads them
enum { DLSS_F_DEPTH_INVERTED = 1 << 3, DLSS_F_AUTO_EXPOSURE = 1 << 6 };

typedef struct {
    VkImage color, depth, motion, output, exposure;         // exposure may be VK_NULL_HANDLE
    VkImageView color_v, depth_v, motion_v, output_v, exposure_v;
    VkFormat color_fmt, depth_fmt, motion_fmt, output_fmt, exposure_fmt;   // output_fmt: what the storage view sees (an sRGB image's UNORM twin)
    uint32_t mot_w, mot_h;                                   // the motion texture's size: display-resolution vectors are detected from it
    uint32_t iw, ih, ow, oh;                                 // the render and display sizes of this call
    float jitter[2], mv[2], pre_exposure;
    uint32_t flags, reset;                                   // the game's DLSS flags and its history reset
    VkCommandBuffer cb;                                      // the submission's buffer: FSR 2 records into it
    VkImage reactive; VkImageView reactive_v;                // FSR 2's reactive mask (R8, render size; nvn_vk_draw.c reactive_run), or VK_NULL_HANDLE
} fsr2_job;

int fsr2_dispatch(const fsr2_job *j);   // 1 when FSR 2 ran (recorded into j->cb); 0 when it cannot (the caller stretches bilinearly)

#ifdef __cplusplus
}
#endif
