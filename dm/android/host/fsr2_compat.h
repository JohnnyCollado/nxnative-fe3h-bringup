// FSR 2 (dm/third_party/fsr2, written for Windows) on Android: the Microsoft C runtime's checked string functions it calls, on the standard ones. Force-included into its
// sources by build.py, with FFX_GCC (no __declspec export); its Vulkan calls go through volk, as the host's do.
#pragma once
#ifdef __cplusplus
#define _LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT 1   // its Vulkan backend turns shader resource names into wide strings with std::wstring_convert
#define _LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS 1
#include "volk.h"   // the host's Vulkan function pointers (vk_driver.c): the backend calls a few instance functions by name
#include <codecvt>
#include <locale>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cwchar>

#define _countof(a) (sizeof(a) / sizeof((a)[0]))

// wcscpy_s(dst array, src): copies at most the array, always terminated
template <size_t N> inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t *src) { wcsncpy(dst, src, N - 1); dst[N - 1] = 0; return 0; }
inline int wcscpy_s(wchar_t *dst, size_t n, const wchar_t *src) { if (!n) return 1; wcsncpy(dst, src, n - 1); dst[n - 1] = 0; return 0; }
// wcstombs_s(&converted, dst, dst size, src, max count): converted counts the terminator, as Microsoft's does
inline int wcstombs_s(size_t *ret, char *dst, size_t dst_size, const wchar_t *src, size_t count) {
    if (!dst || !dst_size) { if (ret) *ret = 0; return 1; }
    size_t lim = count < dst_size - 1 ? count : dst_size - 1;
    size_t n = wcstombs(dst, src, lim);
    if (n == (size_t)-1) { dst[0] = 0; if (ret) *ret = 0; return 1; }
    dst[n < lim ? n : lim] = 0; if (ret) *ret = (n < lim ? n : lim) + 1; return 0;
}
inline int sprintf_s(char *buf, size_t n, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vsnprintf(buf, n, fmt, ap); va_end(ap); return r; }
#endif
