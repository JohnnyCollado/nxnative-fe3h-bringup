# Android launcher

The Android app (`dev.nxnative.fortunesweave`). So far it installs the game data. Starting the game waits for the host to run on Android.

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

## Building

From this folder: `./gradlew --offline :app:assembleDebug` (output `app/build/outputs/apk/debug/app-debug.apk`) and `./gradlew --offline :app:testDebugUnitTest`.
The plugin versions (AGP 8.9.1, Kotlin 2.0.21, Gradle 8.13) are the ones the local Gradle cache already holds, so the build needs no downloads.
`InstallerTest` runs the installer on the build machine: a full install, a bad part, a missing part, an interrupted copy, a wrong identity, and the identity of an image that romfs_image.py computed.
