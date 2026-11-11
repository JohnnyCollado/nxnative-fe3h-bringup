// Compatibility layer (dm/third_party/lsfg-eden/README.md): Eden hashes Lossless.dll with CityHash only to notice a changed DLL beside its shader cache.
// Any 64-bit hash does that job, so this is FNV-1a: a cache written by Eden itself is not read here (the host keeps its own cache file).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include "common/common_types.h"
namespace Common {
inline u64 CityHash64(const char* data, std::size_t size) {
    u64 h = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        h = (h ^ static_cast<u8>(data[i])) * 1099511628211ull;
    }
    return h;
}
} // namespace Common
