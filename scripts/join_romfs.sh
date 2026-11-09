#!/system/bin/sh
# Joins the parts written by split_romfs.py back into one RomFS image, on an Android device (toybox sh) or any POSIX shell.
# Each part's SHA-256 is checked before it is appended, and the part is deleted once appended, so the device needs about one part of space beyond the image.
# A join that stops partway (a part missing, a bad copy, a full disk) can be run again: it truncates <image>.joining to the parts already appended and goes on.
# The finished image is checked against the whole image's size and SHA-256, then renamed from <image>.joining to <image>.
# Usage: sh join_romfs.sh <parts folder> <image path>   (non-zero exit on error)
set -u
dir=${1:?usage: join_romfs.sh <parts folder> <image path>}
img=${2:?usage: join_romfs.sh <parts folder> <image path>}
list="$dir/parts.sha256"
[ -f "$list" ] || { echo "[RED] no $list"; exit 1; }
work="$img.joining"
[ -f "$work" ] || : > "$work" || { echo "[RED] cannot write $work"; exit 1; }
off=0
while read -r hash name size; do
    case $hash in image|identity) continue;; esac
    end=$((off + size))
    have=$(stat -c %s "$work")
    if [ "$have" -ge "$end" ]; then
        off=$end; continue   # appended by an earlier run
    fi
    part="$dir/$name"
    [ -f "$part" ] || { echo "[RED] $part is missing: push it, then run this again"; exit 1; }
    [ "$(stat -c %s "$part")" = "$size" ] || { echo "[RED] $part is $(stat -c %s "$part") bytes, expected $size: push it again"; exit 1; }
    got=$(sha256sum "$part" | cut -d' ' -f1)
    [ "$got" = "$hash" ] || { echo "[RED] $part failed its SHA-256 check: push it again"; exit 1; }
    [ "$have" = "$off" ] || truncate -s "$off" "$work" || { echo "[RED] cannot truncate $work"; exit 1; }
    cat "$part" >> "$work" || { echo "[RED] appending $part failed (disk full?); free space and run this again"; truncate -s "$off" "$work"; exit 1; }
    rm -f "$part"
    echo "appended $name"
    off=$end
done < "$list"
set -- $(grep '^image ' "$list")
want_hash=$2; want_size=$3
[ "$(stat -c %s "$work")" = "$want_size" ] || { echo "[RED] $work is $(stat -c %s "$work") bytes, expected $want_size"; exit 1; }
echo "checking the whole image (a few minutes for a large one)"
[ "$(sha256sum "$work" | cut -d' ' -f1)" = "$want_hash" ] || { echo "[RED] the joined image failed its SHA-256 check: delete $work and push the parts again"; exit 1; }
mv -f "$work" "$img" || exit 1
echo "done: $img ($want_size bytes)"
