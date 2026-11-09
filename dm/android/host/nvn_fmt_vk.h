// NVNformat → VkFormat (nvn_fmt_vk.c). Returns 1 exact, 2 substitute, 0 none; swz: the component remapping (NVN_SWZ_*, as dm/gfx/tex/nvn_fmt_mtl.h)
#pragma once
#include <stdint.h>
enum { NVN_SWZ_ZERO = 0, NVN_SWZ_ONE = 1, NVN_SWZ_R = 2, NVN_SWZ_G = 3, NVN_SWZ_B = 4, NVN_SWZ_A = 5 };
int nvn_fmt_to_vk(int nvn_format, uint32_t *vk_format, uint8_t swz[4]);
