#!/usr/bin/env python3
# Split a RomFS image (dm/tools/romfs_image.py) into parts for transfer to a device, which joins them back into one image with join_romfs.sh.
# Copying one 27 GB file over adb is all-or-nothing: a dropped connection starts it again. Parts of 4 GiB less 1 MiB (small enough for a FAT32 staging card too)
# are pushed one at a time, and only a failed part is sent again. The device appends each checked part to the image and deletes it, so it needs about one part of
# space beyond the image itself.
# Writes into <out folder> (default: <image>.parts beside the image):
#   <image name>.000, .001, ...   the parts, in order
#   parts.sha256                  one line per part: its SHA-256, its name, its size (read by join_romfs.sh and the Android launcher's installer), then
#                                 "image <SHA-256 of the whole image> <size>" and "identity <the image's identity line>" (romfs_image.py --identity; dm/host/romfs_identity.h),
#                                 which names the game and lets an installer check the joined image without reading its 27 GB again
#   join_romfs.sh                 the device-side join script (a copy of scripts/join_romfs.sh)
# The image's own index check (romfs_image.h) also refuses an image cut off partway, so a join that stopped early cannot be mounted.
# Usage: split_romfs.py <image> [--out DIR] [--part-size BYTES] [--title ID] [--version V]   (exit 1 on error)
#        split_romfs.py --identity-only <image> [--out DIR] [--title ID] [--version V]   adds the identity line to an existing split's parts.sha256
import argparse, hashlib, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dm/tools'))
import romfs_image

PART = (4 << 30) - (1 << 20)
CHUNK = 8 << 20


def split(img, out, part_size=PART, say=print):
    size = img.stat().st_size
    if size == 0: raise ValueError(f'{img} is empty')
    with open(img, 'rb') as f:
        if f.read(8) != b'DMROMFS1': raise ValueError(f'{img}: not a RomFS image (no DMROMFS1 header)')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(img.name + '.[0-9][0-9][0-9]'): old.unlink()   # parts of an earlier split with another part size
    whole, lines, k = hashlib.sha256(), [], 0
    with open(img, 'rb') as f:
        while True:
            name = f'{img.name}.{k:03d}'
            h, n = hashlib.sha256(), 0
            with open(out / (name + '.tmp'), 'wb') as p:
                while n < part_size:
                    b = f.read(min(CHUNK, part_size - n))
                    if not b: break
                    p.write(b); h.update(b); whole.update(b); n += len(b)
            if n == 0: (out / (name + '.tmp')).unlink(); break
            os.replace(out / (name + '.tmp'), out / name)
            lines.append(f'{h.hexdigest()}  {name} {n}')
            say(f'  {name}: {n:,} bytes')
            k += 1
            if n < part_size: break
    if sum(int(l.rsplit(' ', 1)[1]) for l in lines) != size: raise RuntimeError(f'{img} changed size while splitting; split it again')
    lines.append(f'image {whole.hexdigest()} {size}')
    (out / 'parts.sha256').write_text('\n'.join(lines) + '\n', newline='\n')
    shutil.copyfile(Path(__file__).with_name('join_romfs.sh'), out / 'join_romfs.sh')
    return k


def add_identity(img, out, title='', version=''):   # (re)writes the identity line of parts.sha256 from the image (its header and index only: a moment)
    lst = out / 'parts.sha256'
    lines = [l for l in lst.read_text().splitlines() if l and not l.startswith('identity ')]
    img_line = next((l for l in lines if l.startswith('image ')), None)
    if not img_line or int(img_line.split()[2]) != img.stat().st_size: raise ValueError(f'{lst} is not a split of {img} (sizes differ)')
    ident = romfs_image.identity(img, title, version)
    lst.write_text('\n'.join(lines + [f'identity {ident}']) + '\n', newline='\n')
    return ident


def main():
    ap = argparse.ArgumentParser(description='Split a RomFS image into checked parts for transfer to a device')
    ap.add_argument('image', type=Path)
    ap.add_argument('--out', type=Path, help='the folder for the parts (default: <image>.parts)')
    ap.add_argument('--part-size', type=int, default=PART, help=f'bytes per part (default {PART:,}: 4 GiB less 1 MiB)')
    ap.add_argument('--title', default='', help='the game\'s title ID, named in the identity line')
    ap.add_argument('--version', default='', help='the game\'s version, named in the identity line')
    ap.add_argument('--identity-only', action='store_true', help='only add the identity line to an existing split')
    a = ap.parse_args()
    try:
        out = a.out or a.image.with_name(a.image.name + '.parts')
        if not a.identity_only:
            print(f'splitting {a.image} into {out}', flush=True)
            n = split(a.image, out, a.part_size, lambda s: print(s, flush=True))
        print(add_identity(a.image, out, a.title, a.version), flush=True)
    except (OSError, ValueError, RuntimeError) as e: sys.exit(f'[RED] {e}')
    if not a.identity_only:
        print(f'done: {n} parts and parts.sha256 in {out}; install them with the Android launcher, or on the device: sh join_romfs.sh <parts folder> <image path>')


if __name__ == '__main__':
    main()
