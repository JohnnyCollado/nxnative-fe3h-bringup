// Lossless Scaling frame generation for the Android host, behind a C interface (lsfg_glue.cpp; this folder's README.md). GPL-3.0-or-later, as the Eden
// sources it drives: the app that ships it is distributed under the GPL.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdint.h>
#include "volk.h"
#ifdef __cplusplus
extern "C" {
#endif
// Once, with the host's device and the features nvn_vk_mem.c turned on. 1 when the user's Lossless.dll is there and its shaders are ready
// ($DM_ROOT/lossless/Lossless.dll; the SPIR-V taken from it is cached beside it as lsfg_spirv.cache)
int lsfg_init(VkDevice dev, VkPhysicalDevice pd, int memory_model, int null_descriptor, int float16);
// Recorded into cb after a real frame was drawn into src (w x h, format fmt, layout GENERAL, its colour writes done): copies it into the history and, from the
// second frame on, writes the frame halfway between the previous one and this into the frame generator's own image. 1 when that image now holds a new frame
// (lsfg_output; layout GENERAL, ready for a fragment shader to sample)
int lsfg_frame(VkCommandBuffer cb, VkImage src, VkFormat fmt, uint32_t w, uint32_t h);
VkImageView lsfg_output(void);
// The history no longer follows the picture (a pause, a scene cut): the next frames are not interpolated until two new ones are in
void lsfg_reset(void);
#ifdef __cplusplus
}
#endif
