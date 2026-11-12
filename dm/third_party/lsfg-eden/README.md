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
- `DM_LSFG_SKIP=<n>` (0 or unset: measured and logged only): no generated frame when the mean change of a 32x18 brightness grid between two real frames is n of 255 or more.
- `DM_LSFG_FLOW_SCALE` (0.25 to 1, default 0.5): the optical flow's resolution against the output, as Eden's automatic choice gives for a 960-wide picture shown at 1920.

## What it costs and what it cannot do (Odin 3, Adreno 830, Fire Emblem Fortune's Weave)

GPU busy on the title screen, ten one-second readings of `/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage` each (2026-11-12, Turnip):

| Setup | GPU busy | Frames shown per 600 real |
|---|---|---|
| Docked, off | 44 % | 600 |
| Docked, flow 0.25 | 79 % | 1,192 |
| Docked, flow 0.5 | 87 % | 1,187 |
| Docked, flow 1.0 | 92 % | 1,190 |
| Handheld, off | 66 % (GPU clock 1.05 GHz, the others 1.10) | 599 |
| Handheld, flow 0.5 | 72 % | 1,190 |

In play (looping a staircase with the camera turning, the same day): docked at flow 0.5 87 % and handheld at flow 0.5 72 % (GPU clock 1.05 GHz), both at 33.3 ms a real frame with every generated frame shown; the quality model at flow 0.75 99 %, 39.9 ms a frame and a third of the generated frames.

- Docked mode has the game upscale to 2048x1152 and the chain work at that size; handheld mode gives 1920x1080 for both, which is where the room comes from.
- **The quality model is not here.** Lossless.dll also holds LSFG 3.1's quality shaders (the performance ones' resource IDs minus 23; lsfg-vk's `v3.1_src` beside `v3.1p_src`, the same six stages with about twice the feature images). Ported for a test: 90 % at flow 0.25, 99 % at 0.5 with a third of the generated frames made and the game at 25 fps; no difference to the eye on a fast camera turn. Taken out again.
- **Warped frames on a fast camera turn** (a ghost outline where a character was, shredded edges, worst over repeating steps) are the generator's limit at 30 real frames a second: it sees two pictures and nothing of depth or motion. Flow 1.0 and the quality model did not change it. The host can leave the generated frame out across a large change (`DM_LSFG_SKIP`, the app's Fast motion setting; `dm/android/host/nvn_vk_front.c`, after LSFG-Android's anti-artifacts option): its measure reads about 0.4 on the calm title screen and up to 101 on cuts and fast action. The thresholds behind the setting (26, 16) are LSFG-Android's and a guess, not yet set from play.
- The game has a motion blur task of its own (`gfxa::TaskMotionBlur` in main). A game's blur sized for 30 frames a second is twice too long at 60 shown; whether it runs outside cutscenes has not been checked.

## Requirements

The device's Vulkan memory model, null descriptors (robustness2) and 16-bit float shaders (`nvn_vk_mem.c` turns them on when present). Without them, or without a valid DLL, the log says why and frames are shown as before.
