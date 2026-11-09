#!/system/bin/sh
# Runs the Android host (dmhost, dm/android/host/build.py) on the device from an adb shell, the way the Mac's run command does (docs/development.md, "Build by hand").
# The folder (default /data/local/tmp/nx) holds: dmhost, patched/ (the modules rewritten with DM_TARGET=android and their tables), font/ (nintendo_ext.ttf), spirv/, and
# RomFS.img (or, without it, the launcher app's installed romfs.img is used).
# Saves and temporary files go to run/ beside them. The log is written to run/log.txt and to the terminal; extra KEY=VALUE arguments are passed to the host's environment.
# Usage: sh run_device.sh [folder] [KEY=VALUE ...]
D=${1:-/data/local/tmp/nx}
[ $# -gt 0 ] && shift
mkdir -p "$D/run/save" "$D/run/temp"
export DM_ROOT="$D/run" DM_SAVE="$D/run/save" DM_TEMP="$D/run/temp" DM_FONT_DIR="$D/font"
# the game image: the folder's own RomFS.img, else the launcher app's installed copy (the shell can read the app's external folder)
APP_IMG=/sdcard/Android/data/dev.nxnative.fortunesweave/files/romfs.img
[ -z "$DM_ROMFS" ] && { [ -f "$D/RomFS.img" ] && DM_ROMFS="$D/RomFS.img" || DM_ROMFS="$APP_IMG"; }
export DM_ROMFS
# Mods: each folder mods/<name>/romfs mirrors the game's file paths, and a file there is read instead of the image's file of the same path (DM_ROMFS_OVERLAY,
# dm/host/hle_fs.c rom_overlay_path; the image itself is never changed). Several mods stack in folder-name order, the first one holding a file wins. Delete or
# rename a mod's folder to turn it off. An explicit DM_ROMFS_OVERLAY (an argument) replaces this list
mkdir -p "$D/mods"
OV=""
for m in "$D"/mods/*/romfs; do [ -d "$m" ] && OV="${OV:+$OV:}$m"; done
[ -n "$OV" ] && [ -z "$DM_ROMFS_OVERLAY" ] && export DM_ROMFS_OVERLAY="$OV" && echo "[dm] mods over the game files, first match wins: $OV"
[ -z "$DM_SHARED_FONT" ] && [ -f /system/fonts/Roboto-Regular.ttf ] && export DM_SHARED_FONT=/system/fonts/Roboto-Regular.ttf   # no standard Switch font here: the device's own
for kv in "$@"; do export "$kv"; done
cd "$D" || exit 1
chmod 755 ./dmhost   # adb push of a new build drops the execute bit
exec ./dmhost "$D/patched" 2>&1 | tee "$D/run/log.txt"
