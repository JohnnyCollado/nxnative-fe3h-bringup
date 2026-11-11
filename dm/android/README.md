# Android launcher

The Android app (`dev.nxnative.fortunesweave`): it installs the game data, starts the game (`GameActivity`, which runs the host `libdmhost.so` as a process of its own and shows its frames), and holds the settings.

## Game data

The RomFS (the game's read-only data) goes to the device as parts and is joined into one image on the device:

1. On the PC, pack the dump into one image (`scripts/pack_dump.py`, or `dm/tools/romfs_image.py`), then split it:
   `python scripts/split_romfs.py <RomFS.img> --title <title ID> --version <version>`.
   This writes `<RomFS.img>.parts/`: parts of 4 GiB less 1 MiB, and `parts.sha256`, which lists each part's SHA-256 and the image's identity line (`dm/host/romfs_identity.h`).
2. Push `parts.sha256` and the parts into the app's import folder: `/sdcard/Android/data/dev.nxnative.fortunesweave/files/import/`.
3. Open the app. It starts the install by itself once `parts.sha256` is there, and it can run while parts are still being pushed. In a foreground service (`InstallService.kt`), it does this for each part in order:
   - appends the part to `files/romfs.img.joining`, checking its SHA-256 in the same pass;
   - deletes the part.

   So the device needs about one part of free space beyond the image. When a part fails its check, the installer cuts it back off the image, deletes it and waits for it to be pushed again. A paused or killed install resumes from the last verified part.
4. The finished image is checked against the manifest's size and identity (its index, not its data, which the parts' checks covered), then renamed to `files/romfs.img`, and its identity line is written to `files/romfs.id`.

The install is one-time. A start reads `romfs.id` and compares the image's size from the file system (`Installer.installed()`), without reading the image's data. A new install happens only when a different `parts.sha256` is pushed.

The data is in the app's own storage folder, so uninstalling removes it; the manifest sets `hasFragileUserData`, so Android offers to keep it. Update the app with `adb install -r`, which keeps it.

`scripts/join_romfs.sh` does the same join from an adb shell, without the app.

## Playing

- **Pause menu:** Back, or − and + held together for a second, opens it over the game; leaving the app opens it too. While it is open the game is in the background (the host's `dm_background`: focus out, sound off, the game's clock and graphics stopped; `DMF_PAUSE` in `host/dm_front_proto.h`). The d-pad or stick moves, A (the right button) picks, B (the bottom one) or Back closes it. It holds Resume, the settings, Restart game and Quit to launcher.
- **Settings** (`Settings.kt`, shown by `SettingsPanel.kt` in the pause menu and on the launcher's Settings button). Each becomes an environment setting of the host:

  | Setting | Variable | Applies |
  |---|---|---|
  | Console mode (docked / handheld) | `DM_OPMODE` | at once (`DMF_OPMODE`, the host's `dm_set_opmode`) |
  | Game language | `DM_LANG` | at the next start |
  | Sound | `DM_AUDIO` | at the next start |
  | GPU driver | `DM_GPU_DRIVER`, `DM_HOOK_DIR` | at the next start |
  | Frame-rate counter | (the app's) | at once: the host's finished frames a second, top left |

  The frame-rate cap (`DM_FPS_TARGET`) is not offered: this game has no known frame limit for the host to set (`dm/host/hle_fpslock.c`).

## GPU drivers

The host calls Vulkan through [volk](../third_party/volk), so the driver is chosen when it starts (`host/vk_driver.c`): the system's, or a custom one loaded through [libadrenotools](../third_party/libadrenotools), whose hook libraries `build.py` builds into the app's native library folder (`DM_HOOK_DIR`). A custom driver that does not load is logged with `[RED]` and the system driver is used instead. The log names the driver in use (`driver: ...`).

Driver packages are zips in the format other Android emulators use: the driver library and a `meta.json` whose `libraryName` names it. Install one with the launcher's Import GPU driver button, or push the zip into `/sdcard/Android/data/dev.nxnative.fortunesweave/files/import/drivers/`; the launcher installs it the next time it opens. Drivers are kept in the app's internal storage (`files/drivers/`), because libadrenotools refuses drivers on shared storage. Pick one under Settings › GPU driver.

## DLSS

The game's DLSS (`nvnCommandBufferEvaluateDlss`, NVIDIA's neural upscaler, a part of NVN2) runs as [FSR 2](../third_party/fsr2), AMD's temporal upscaler ([NVN2's DLSS on FSR 2](../../docs/reference/rendering.md#nvn2s-dlss-on-fsr-2-the-android-host)), where the Mac host runs it on MetalFX. Before FSR 2 there was a bilinear stretch. The game renders at 1024x576 and the upscaler takes it to 2048x1152. What FSR 2 cannot take (no depth or motion vectors, a sub-rectangle, an output whose format cannot be a storage image) still draws with the bilinear stretch, and the host logs which path each call took.

| Variable | Meaning |
|---|---|
| `DM_VK_NO_DLSS` | skip the DLSS command entirely (the output is left as it is) |
| `DM_FSR2_SHARPEN` | `0` turns FSR 2's sharpening off (on by default) |
| `DM_FSR2_SHARPNESS` | the sharpness, 0 to 1 (default 0.3) |
| `DM_FSR2_FRAME_DELTA_MAX` | the longest gap between DLSS calls in ms before the accumulation history resets (default 200, a pause) |

## Building

From this folder: `./gradlew --offline :app:assembleDebug` (output `app/build/outputs/apk/debug/app-debug.apk`) and `./gradlew --offline :app:testDebugUnitTest`.
The plugin versions (AGP 8.9.1, Kotlin 2.0.21, Gradle 8.13) are the ones the local Gradle cache already holds, so the build needs no downloads.
`InstallerTest` runs the installer on the build machine: a full install, a bad part, a missing part, an interrupted copy, a wrong identity, and the identity of an image that romfs_image.py computed.
