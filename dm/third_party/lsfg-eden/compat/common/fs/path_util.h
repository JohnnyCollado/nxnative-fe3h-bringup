// Compatibility layer (dm/third_party/lsfg-eden/README.md): Eden's per-user folders. Only the Lossless folder is used: $DM_ROOT/lossless on the device, where the
// launcher places the user's Lossless.dll and the host keeps the shaders taken from it (the run folder the app gives the host as DM_ROOT).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdlib>
#include <filesystem>
namespace Common::FS {
enum class EdenPath { LosslessDir };
inline std::filesystem::path GetEdenPath(EdenPath) {
    const char* root = std::getenv("DM_ROOT");
    return std::filesystem::path(root && *root ? root : ".") / "lossless";
}
} // namespace Common::FS
