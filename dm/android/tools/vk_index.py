#!/usr/bin/env python3
# The shader index the Android host's Vulkan backend reads (dm/android/host/nvn_vk_draw.c): one text file in place of 16,888 <key>.res.json files plus the c1 segments,
# which only vk_spirv.py's jobs.json holds. Written next to the .spv files as index.txt. One block per shader:
#   S <key> <stage vs|ps|gs|cs> <frag_out_map> <c1 hex, or ->
#   C <binding> <constant buffer slot>                                        uniform buffers (slot 0 / 2 driver segments, 1 the shader's c1, 3-16 uniform slot - 3, 24 push constants)
#   B <binding> <cb_slot> <cb_offset in words> <flags>                         storage buffers: their address is at that c0 word (dmsh_abi.h DMSH_C0_SSBO_*)
#   T <binding> <type> <cbuf_slot> <handle> <array> <flags>                   textures (Ryujinx TextureDescriptor: handle = word offset of the handle in that constant buffer, -1 = c2)
#   I <binding> <type> <cbuf_slot> <handle> <array> <flags>                   images, the same
#   E                                                                          end of the block
# Usage: vk_index.py <spirv dir (vk_spirv.py output)>
import json, os, sys


def main():
    d = sys.argv[1]
    jobs = {j['key']: j for j in json.load(open(os.path.join(d, 'jobs.json'), encoding='utf-8'))}
    keys = sorted(f[:-9] for f in os.listdir(d) if f.endswith('.res.json'))
    out = []; no_c1 = 0
    for k in keys:
        r = json.load(open(os.path.join(d, k + '.res.json'), encoding='utf-8'))
        if not os.path.exists(os.path.join(d, k + '.spv')): continue
        c1 = jobs.get(k, {}).get('c1') or '-'
        if c1 == '-': no_c1 += 1
        out.append(f"S {k} {r['stage']} {r.get('frag_out_map', 0)} {c1}")
        out += [f"C {b['binding']} {b['slot']}" for b in r['cbufs']]
        out += [f"B {b['binding']} {b['cb_slot']} {b['cb_offset']} {b['flags']}" for b in r['sbufs']]
        for kind, key in (('T', 'textures'), ('I', 'images')):
            out += [f"{kind} {t['binding']} {t['type']} {t['cbuf_slot']} {t['handle']} {t['array']} {t['flags']}" for t in r[key]]
        out.append('E')
    p = os.path.join(d, 'index.txt')
    open(p, 'w', encoding='ascii', newline='\n').write('\n'.join(out) + '\n')
    print(f'{len(keys)} shaders, {no_c1} without a c1 segment -> {p} ({os.path.getsize(p) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
