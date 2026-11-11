# lsfg-eden: Lossless Scaling frame generation for the Android host

**Licence: GPL-3.0-or-later** (`LICENSE.txt`). This folder is not MIT like the rest of the repository: an app built with it is distributed under the GPL as a whole.

## What it is

Lossless Scaling (a paid Windows app, on Steam) generates frames between two real ones (LSFG). Its shaders live inside its `Lossless.dll`. The user imports their own copy of that DLL in the launcher (Settings, "Import Lossless.dll…"); it is never shipped with this app. With "Frame generation: 2x" on, the host (`DM_LSFG=1`) inserts one generated frame between each two real ones, so the game's 30 fps is shown as 60, with about half a frame interval more delay.

## Where it comes from

- `video_core/frame_gen/` and `video_core/renderer_vulkan/present/lsfg_*`: copied unchanged from the Eden fork in `dm/third_party/eden-fefw/source/src/video_core/` (Copyright 2026 Eden Emulator Project, GPL-3.0-or-later). Eden's files credit lsfg-vk (Copyright 2025 lsfg-vk, GPL-3.0-or-later at the time), which first ran these shaders under Vulkan. lsfg-vk itself has since moved to CC BY-NC-ND 4.0 (August 2026); nothing here is taken from lsfg-vk's later releases.
  - `lossless_dll.cpp` reads the DLL's shader resources (IDs 255, 256, 280-302, and the 16-bit float variants 49 above) and caches them as `lsfg_spirv.cache` beside the DLL; `lsfg_translate.cpp` renumbers their bindings.
  - `lsfg_*.cpp`: the pass chain (mipmaps, alpha, beta, gamma, delta, generate) on the host's Vulkan device.
- `compat/`: written here. It stands in for the parts of Eden the copied files include: its integer names, `vk::` wrapper, `Device`, `MemoryAllocator`, the `CreateWrapped*` helpers of `present/util.h`, and `Common::FS`. The DLL hash is FNV-1a instead of CityHash (it only notices a changed DLL), and Eden's Lossless folder is `$DM_ROOT/lossless`.
- `lsfg_glue.cpp` / `lsfg_glue.h`: written here, after Eden's `present/frame_gen.cpp`. A C interface for `dm/android/host/nvn_vk_front.c`: each real frame shown is copied into the history, and from the third frame on the frame halfway from the last one is generated into an image of this module. The screen code stretches that image into a spare app buffer and sends it half a real-frame interval before the real one. Eden's pacer, multi-frame generation (3x, 4x) and debug dumps are not ported.

## Settings

- `DM_LSFG=1`: on (the launcher's Frame generation setting).
- `DM_LSFG_FLOW_SCALE` (0.25 to 1, default 0.5): the optical flow's resolution against the output, as Eden's automatic choice gives for a 960-wide picture shown at 1920.

## Requirements

The device's Vulkan memory model, null descriptors (robustness2) and 16-bit float shaders (`nvn_vk_mem.c` turns them on when present). Without them, or without a valid DLL, the log says why and frames are shown as before.
