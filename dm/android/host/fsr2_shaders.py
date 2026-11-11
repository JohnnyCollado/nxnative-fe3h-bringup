#!/usr/bin/env python3
# FSR 2's Vulkan shaders (dm/third_party/fsr2), compiled the way its CMake build does with AMD's FidelityFX_SC tool, but with the NDK's glslc: every permutation of each
# pass (the options below, and FFX_HALF 0/1 except for the luminance pyramid), with the same defines, into <out>/<pass>_permutations.h, the headers
# vk/shaders/ffx_fsr2_shaders_vk.cpp includes. Each header holds:
#   ffx_fsr2_<pass>_PermutationKey  a union of one bit per option (the .cpp sets them by name) and its index
#   g_ffx_fsr2_<pass>_PermutationInfo[]  the distinct SPIR-V blobs, each with its resources by kind (storage images, sampled images, uniform buffers): names and bindings
#   g_ffx_fsr2_<pass>_IndirectionTable[]  key index -> blob
# Resources are read back from each module: the set-1 variables (set 0 holds the backend's samplers), named as declared (cbFSR2, r_input_color_jittered, ...), which is
# what ffx_fsr2.cpp matches them by. Runs from build.py, which skips it while the headers are newer than the shaders and this script.
# Usage: fsr2_shaders.py <glslc> <out dir> [-j N]
import concurrent.futures as cf, hashlib, itertools, os, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'third_party', 'fsr2', 'src', 'ffx-fsr2-api', 'shaders')
PASSES = ['ffx_fsr2_tcr_autogen_pass', 'ffx_fsr2_autogen_reactive_pass', 'ffx_fsr2_accumulate_pass', 'ffx_fsr2_compute_luminance_pyramid_pass', 'ffx_fsr2_depth_clip_pass',
          'ffx_fsr2_lock_pass', 'ffx_fsr2_reconstruct_previous_depth_pass', 'ffx_fsr2_rcas_pass']
BASE = ['-DFFX_GLSL=1', '-DFFX_GPU=1', '-DFFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0', '-DFFX_FSR2_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0',
        '-DFFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1', '-DFFX_FSR2_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0', '-DFFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2']
OPTIONS = ['FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE', 'FFX_FSR2_OPTION_HDR_COLOR_INPUT', 'FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS',
           'FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS', 'FFX_FSR2_OPTION_INVERTED_DEPTH', 'FFX_FSR2_OPTION_APPLY_SHARPENING', 'FFX_HALF']   # bit order of the key

def compile_one(glslc, src, defs, opt='-Os'):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'o.spv')
        r = subprocess.run([glslc, '-fshader-stage=comp', '--target-env=vulkan1.1', opt, '-I', SRC] + BASE + defs + ['-o', out, src], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(f'{os.path.basename(src)} {" ".join(defs)}:\n{r.stderr[-2000:]}')
        return open(out, 'rb').read()

def reflect(spv, bnames=None):
    """(storage images, sampled images, uniform buffers), each a list of (name, binding), from the module's set-1 variables. The optimized modules carry no
    names: bnames maps each set-1 binding to its name, from the same permutation compiled with -g (each resource of a pass has a binding of its own)"""
    w = struct.unpack('<%dI' % (len(spv) // 4), spv)
    names, sets, binds, types, ptrs, variables, block = {}, {}, {}, {}, {}, [], set()
    i = 5
    while i < len(w):
        op, n = w[i] & 0xffff, w[i] >> 16
        a = w[i + 1:i + n]
        if op == 5: names[a[0]] = struct.pack('<%dI' % (n - 2), *a[1:]).split(b'\0')[0].decode()        # OpName
        elif op == 71:                                                                                     # OpDecorate
            if a[1] == 34: sets[a[0]] = a[2]
            elif a[1] == 33: binds[a[0]] = a[2]
            elif a[1] in (2, 3): block.add(a[0])                                                           # Block / BufferBlock
        elif op == 25: types[a[0]] = ('image', a[6])                                                       # OpTypeImage: Sampled 1 sampled, 2 storage
        elif op in (28, 29): types[a[0]] = ('array', a[1])                                                 # OpTypeArray / RuntimeArray
        elif op == 30: types[a[0]] = ('struct',)
        elif op == 26: types[a[0]] = ('sampler',)
        elif op == 32: ptrs[a[0]] = (a[1], a[2])                                                           # OpTypePointer: storage class, type
        elif op == 59: variables.append((a[1], a[0]))                                                      # OpVariable: id, pointer type
        i += n
    out = ([], [], [])
    for vid, pt in variables:
        if sets.get(vid) != 1 or vid not in binds: continue
        sc, t = ptrs[pt]
        while types.get(t, ('',))[0] == 'array': t = types[t][1]
        k = types.get(t, ('',))
        nm = (bnames or {}).get(binds[vid]) or names.get(vid)
        if not nm: raise RuntimeError(f'set 1 binding {binds[vid]} has no name')
        if k[0] == 'image': out[0 if k[1] == 2 else 1].append((nm, binds[vid]))
        elif k[0] == 'struct' and sc == 2 and t in block: out[2].append((nm, binds[vid]))
    return tuple(sorted(x, key=lambda e: e[1]) for x in out)

def header(name, perms, blobs):
    p = f'ffx_fsr2_{name[len("ffx_fsr2_"):]}' if name.startswith('ffx_fsr2_') else name
    L = [f'// Generated by dm/android/host/fsr2_shaders.py from dm/third_party/fsr2/src/ffx-fsr2-api/shaders/{name}.glsl: do not edit', '#pragma once', '#include <stdint.h>', '']
    L.append(f'typedef union {name}_PermutationKey {{\n    struct {{\n' + ''.join(f'        uint32_t {o} : 1;\n' for o in OPTIONS) + '    };\n    uint32_t index;\n} ' + f'{name}_PermutationKey;\n')
    L.append(f'typedef struct {name}_PermutationInfo {{\n    const uint32_t blobSize;\n    const unsigned char* blobData;\n'
             '    const uint32_t numStorageImageResources;\n    const char** storageImageResourceNames;\n    const uint32_t* storageImageResourceBindings;\n'
             '    const uint32_t numSampledImageResources;\n    const char** sampledImageResourceNames;\n    const uint32_t* sampledImageResourceBindings;\n'
             '    const uint32_t numUniformBufferResources;\n    const char** uniformBufferResourceNames;\n    const uint32_t* uniformBufferResourceBindings;\n'
             f'}} {name}_PermutationInfo;\n')
    infos = []
    for b, (spv, refl) in enumerate(blobs):
        tag = f'g_{name}_{b}'
        L.append(f'static const unsigned char {tag}_data[] = {{' + ','.join(str(x) for x in spv) + '};')
        fields = []
        for kind, lst in zip(('StorageImage', 'SampledImage', 'UniformBuffer'), refl):
            if lst:
                L.append(f'static const char* {tag}_{kind}Names[] = {{' + ', '.join(f'"{n}"' for n, _ in lst) + '};')
                L.append(f'static const uint32_t {tag}_{kind}Bindings[] = {{' + ', '.join(str(bd) for _, bd in lst) + '};')
                fields.append(f'{len(lst)}, {tag}_{kind}Names, {tag}_{kind}Bindings')
            else: fields.append('0, 0, 0')
        infos.append(f'    {{ {len(spv)}, {tag}_data, ' + ', '.join(fields) + ' },')
    L.append(f'\nstatic const {name}_PermutationInfo g_{name}_PermutationInfo[] = {{\n' + '\n'.join(infos) + '\n};\n')
    L.append(f'static const int32_t g_{name}_IndirectionTable[] = {{ ' + ', '.join(str(perms[i]) for i in range(1 << len(OPTIONS))) + ' };\n')
    return '\n'.join(L) + '\n'

def main():
    if len(sys.argv) < 3: sys.exit('usage: fsr2_shaders.py <glslc> <out dir> [-j N]')
    glslc, out = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[sys.argv.index('-j') + 1]) if '-j' in sys.argv else (os.cpu_count() or 8)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for name in PASSES:
        half_ok = name != 'ffx_fsr2_compute_luminance_pyramid_pass'   # as the CMake build: no 16-bit permutations for the luminance pyramid
        for bits in itertools.product((0, 1), repeat=len(OPTIONS)):
            if bits[-1] and not half_ok: continue
            jobs.append((name, bits))
    with cf.ThreadPoolExecutor(nj) as ex:
        spvs = list(ex.map(lambda j: compile_one(glslc, os.path.join(SRC, j[0] + '.glsl'), [f'-D{o}={v}' for o, v in zip(OPTIONS, j[1])]), jobs))
        dbg = list(ex.map(lambda j: compile_one(glslc, os.path.join(SRC, j[0] + '.glsl'), [f'-D{o}={v}' for o, v in zip(OPTIONS, j[1])], '-g'), jobs))
    bnames = []   # per job: set-1 binding -> name, from its unoptimized twin
    for d in dbg: r = reflect(d); bnames.append({b: n for lst in r for n, b in lst})
    total = 0
    for name in PASSES:
        blobs, by_hash, perms = [], {}, {}
        for (n, bits), spv, bn in zip(jobs, spvs, bnames):
            if n != name: continue
            h = hashlib.sha256(spv).digest()
            if h not in by_hash: by_hash[h] = len(blobs); blobs.append((spv, reflect(spv, bn)))
            perms[sum(b << k for k, b in enumerate(bits))] = by_hash[h]
        for idx in range(1 << len(OPTIONS)):   # FFX_HALF=1 keys of a pass without 16-bit permutations take the 32-bit blob
            if idx not in perms: perms[idx] = perms[idx & ~(1 << (len(OPTIONS) - 1))]
        open(os.path.join(out, name + '_permutations.h'), 'w', encoding='utf-8', newline='\n').write(header(name, perms, blobs))
        total += len(blobs)
    print(f'FSR 2 shaders: {len(jobs)} permutations compiled, {total} distinct, into {out}')

if __name__ == '__main__':
    main()
