#!/usr/bin/env python3
# Builds the host for Android (arm64): the dmhost command-line program that runs a game from /data/local/tmp over adb, compiled with the NDK's clang.
# What differs from the Mac build (dm/host/Makefile):
#   - dm/android/compat/include comes first on the include path: Linux forms of the Apple headers the host includes (mach/mach.h, CommonCrypto, ...), with their
#     implementations in dm/android/compat/dm_compat.c, and dm_android_compat.h force-included for the BSD libc extras
#   - the Objective-C files and the Apple-framework ones are left out (EXCLUDE); dm/android/host/*.c stands in for what they export (a front end with no window,
#     the graphics backend's null form, silent audio, the movie decoders Android has no counterpart for yet)
#   - the assembly (entry.S, nvn_entry.S, nvn_thunks.S) is Mach-O syntax: elf_asm() rewrites a copy for ELF (symbol names without the leading underscore,
#     :lo12: for @PAGEOFF, .hidden for .private_extern, .zerofill as a .bss block)
#   - dav1d's config.h is the Mac's with PREFIX 0 (ELF symbols) and HAVE_GETAUXVAL 1 (Linux CPU feature detection: the Odin 3's Snapdragon 8 Elite has no SVE)
#   - FSR 2 (dm/third_party/fsr2): its Vulkan shaders are compiled by fsr2_shaders.py with the NDK's glslc into gen-android/fsr2-shaders (skipped when the headers are
#     fresh), and its C++ sources build as their own group with fsr2_compat.h force-included (Microsoft checked-string stand-ins): vkd_dlss's temporal upscaler
# Rebuilds only what changed: each object's dependencies come from the compiler (-MD), and an object is rebuilt when any of them, or the flags, are newer.
# Usage: build.py [--game <title ID>] [-j N] [--clean]   (exit 1 on error). Output: dm/host/gen-android/dmhost
import argparse, concurrent.futures as cf, glob, hashlib, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DM = os.path.normpath(os.path.join(HERE, '..', '..'))
HOST = os.path.join(DM, 'host')
OUT = os.path.join(HOST, 'gen-android')
API = 35   # Android 15: the Odin 3
EXCLUDE = {   # Apple-framework sources with Android stand-ins in dm/android/host/android_stubs.c
    'host/hle_movie_avc.c',    # H.264 on VideoToolbox: no H.264 decoder yet (Fortune's Weave's movies are AV1, which dav1d decodes)
    'host/hle_uemalloc.c',     # Unreal's allocator on Apple malloc zones (Unreal games only)
}   # hle_audio.c is built: every output plays through its silent sink on Android (compat AudioToolbox.h fails the Core Audio calls if they are reached)


def ndk():
    """The newest NDK's root and its prebuilt host tag"""
    sdk = os.environ.get('ANDROID_HOME') or os.path.expandvars(r'%LOCALAPPDATA%\Android\Sdk')
    ndks = sorted(glob.glob(os.path.join(sdk, 'ndk', '*')), key=lambda p: [int(x) for x in re.findall(r'\d+', os.path.basename(p))])
    if not ndks: sys.exit(f'[RED] no Android NDK under {sdk}/ndk')
    return ndks[-1], 'windows-x86_64' if os.name == 'nt' else ('darwin-x86_64' if sys.platform == 'darwin' else 'linux-x86_64')


def ndk_bin():
    root, tag = ndk()
    b = os.path.join(root, 'toolchains', 'llvm', 'prebuilt', tag, 'bin')
    return os.path.join(b, 'clang' + ('.exe' if os.name == 'nt' else '')), os.path.join(b, 'llvm-ar' + ('.exe' if os.name == 'nt' else ''))


def elf_asm(src, dst):
    """A Mach-O assembly file rewritten for ELF (see the file header)"""
    s = open(src, encoding='utf-8').read()
    out = []
    for ln in s.split('\n'):
        m = re.match(r'(\s*)\.zerofill\s+[^,]+,[^,]+,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)(.*)', ln)
        if m:
            ind, sym, size, align, rest = m.groups()
            sym = sym[1:] if sym.startswith('_') else sym
            out.append(f'{ind}.pushsection .bss\n{ind}.p2align {align}\n{sym}:\n{ind}.zero {size}\n{ind}.popsection')
            continue
        if re.match(r'\s*\.subsections_via_symbols', ln): continue
        ln = re.sub(r'\.private_extern\b', '.hidden', ln)
        ln = re.sub(r'\.section\s+__TEXT\s*,\s*__text\b.*', '.text', ln)
        ln = re.sub(r'\.section\s+__TEXT\s*,\s*__(const|cstring)\b.*', '.section .rodata', ln)
        ln = re.sub(r'\.section\s+__DATA\s*,\s*__data\b.*', '.data', ln)
        ln = re.sub(r'\.section\s+__DATA\s*,\s*__bss\b.*', '.bss', ln)
        ln = re.sub(r'^(\s*)\.(cstring|const)\s*$', r'\1.section .rodata', ln)
        ln = re.sub(r'^(\s*)\.const_data\s*$', r'\1.section .data.rel.ro', ln)
        code, sep, comment = ln.partition('//')
        code = re.sub(r'\b_(\w+)@GOTPAGEOFF\b', r':got_lo12:\1', code)
        code = re.sub(r'\b_(\w+)@GOTPAGE\b', r':got:\1', code)
        code = re.sub(r'\b_(\w+)@PAGEOFF\b', r':lo12:\1', code)
        code = re.sub(r'\b_(\w+)@PAGE\b', r'\1', code)
        code = re.sub(r'(?<![\w.$])_([A-Za-z]\w*)', r'\1', code)   # Mach-O's C symbol prefix
        out.append(code + sep + comment)
    data = '\n'.join(out)
    if not os.path.exists(dst) or open(dst, encoding='utf-8').read() != data:
        open(dst, 'w', encoding='utf-8', newline='\n').write(data)


def dav1d_config(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    s = open(os.path.join(DM, 'third_party', 'dav1d', 'config', 'config.h'), encoding='utf-8').read()
    s = re.sub(r'#define PREFIX 1\n', '', s)   # dav1d's asm.S tests #ifdef PREFIX, so the define goes, not just its value
    s = re.sub(r'#define HAVE_GETAUXVAL 0', '#define HAVE_GETAUXVAL 1', s)
    s = '// Generated by dm/android/host/build.py from dm/third_party/dav1d/config/config.h: no PREFIX (ELF symbols), HAVE_GETAUXVAL 1 (Linux CPU feature detection)\n' + s
    p = os.path.join(dst_dir, 'config.h')
    if not os.path.exists(p) or open(p, encoding='utf-8').read() != s: open(p, 'w', encoding='utf-8', newline='\n').write(s)


def fsr2_shaders(out, py, j):
    """FSR 2's Vulkan shaders (dm/third_party/fsr2): every permutation of each pass compiled with the NDK's glslc into <out>/fsr2-shaders/<pass>_permutations.h,
    the blob tables vk/shaders/ffx_fsr2_shaders_vk.cpp includes (fsr2_shaders.py). Skipped while the headers are newer than the shaders and the script"""
    gen = os.path.join(out, 'fsr2-shaders'); os.makedirs(gen, exist_ok=True)
    script = os.path.join(HERE, 'fsr2_shaders.py')
    srcs = glob.glob(os.path.join(DM, 'third_party', 'fsr2', 'src', 'ffx-fsr2-api', 'shaders', '*.glsl'))
    passes = ('ffx_fsr2_tcr_autogen_pass', 'ffx_fsr2_autogen_reactive_pass', 'ffx_fsr2_accumulate_pass', 'ffx_fsr2_compute_luminance_pyramid_pass',
              'ffx_fsr2_depth_clip_pass', 'ffx_fsr2_lock_pass', 'ffx_fsr2_reconstruct_previous_depth_pass', 'ffx_fsr2_rcas_pass')
    heads = [os.path.join(gen, p + '_permutations.h') for p in passes]
    if all(os.path.exists(h) and os.path.getmtime(h) > max(os.path.getmtime(s) for s in srcs) and os.path.getmtime(h) > os.path.getmtime(script) for h in heads):
        print(f'FSR 2 shaders up to date ({len(heads)} headers)'); return gen
    root, tag = ndk()
    glslc = os.path.join(root, 'shader-tools', tag, 'glslc' + ('.exe' if os.name == 'nt' else ''))
    if not os.path.exists(glslc): sys.exit(f'[RED] no glslc in the NDK ({glslc})')
    r = subprocess.run([py, '-X', 'utf8', script, glslc, gen, '-j', str(j)])
    if r.returncode: sys.exit('[RED] fsr2_shaders.py failed')
    return gen


def deps_of(d):
    if not os.path.exists(d): return None
    txt = open(d, encoding='utf-8', errors='replace').read().replace('\\\n', ' ')
    m = re.search(r'\.o:\s', txt)   # the target ends in ".o:" (a Windows path has a drive letter colon before it)
    rest = txt[m.end():] if m else ''
    return [p.replace('\\ ', ' ') for p in re.split(r'(?<!\\)\s+', rest.strip()) if p]


def stale(obj, dep, flag_stamp):
    if not os.path.exists(obj): return True
    t = os.path.getmtime(obj)
    if os.path.getmtime(flag_stamp) > t: return True
    ds = deps_of(dep)
    if ds is None: return True
    return any(not os.path.exists(p) or os.path.getmtime(p) > t for p in ds)


def main():
    ap = argparse.ArgumentParser(description='Build the Android host (dmhost)')
    ap.add_argument('--game', default=os.environ.get('GAME', ''), help='the game profile (title ID)')
    ap.add_argument('-j', type=int, default=os.cpu_count() or 8)
    ap.add_argument('--clean', action='store_true')
    a = ap.parse_args()
    if not a.game: sys.exit('[RED] --game <title ID> is required (there is no default game, as for the Mac host)')
    if a.clean: shutil.rmtree(OUT, ignore_errors=True)
    cc, ar = ndk_bin()
    py = sys.executable
    os.makedirs(OUT, exist_ok=True)
    gen = os.path.join(OUT, a.game); os.makedirs(gen, exist_ok=True)
    r = subprocess.run([py, '-X', 'utf8', os.path.join(DM, 'games', 'game.py'), 'header', a.game, os.path.join(gen, 'game_profile.h')], capture_output=True, text=True)
    if r.returncode: sys.exit(f'[RED] game.py header: {r.stdout}{r.stderr}')
    dav1d_config(os.path.join(OUT, 'dav1d-config'))
    gen_fsr = fsr2_shaders(OUT, py, a.j)
    asm_dir = os.path.join(OUT, 'asm'); os.makedirs(asm_dir, exist_ok=True)
    asm = []
    for rel in ('host/entry.S', 'gfx/nvn/nvn_entry.S', 'gfx/nvn/nvn_thunks.S'):
        dst = os.path.join(asm_dir, os.path.basename(rel)); elf_asm(os.path.join(DM, rel), dst); asm.append(dst)

    target = [f'--target=aarch64-linux-android{API}']
    inc = [os.path.join(DM, 'android', 'compat', 'include'), os.path.join(DM, 'gfx', 'nvn', 'include'), os.path.join(DM, 'third_party', 'vulkan-headers', 'include'),
           gen, os.path.join(DM, 'third_party', 'libvpx'), os.path.join(DM, 'third_party', 'dav1d', 'include'), os.path.join(DM, 'third_party', 'stb_vorbis'), HOST,
           os.path.join(DM, 'gfx', 'nvn'), os.path.join(DM, 'gfx', 'tex'), os.path.join(HOST, 'gen', 'msl-macosx'),
           os.path.join(DM, 'third_party', 'volk'), os.path.join(DM, 'third_party', 'libadrenotools', 'include')]
    cflags = target + ['-O1', '-g', '-std=gnu11', '-fPIE', '-Wall', '-Wno-unused-function', '-Wno-unknown-pragmas', '-D_GNU_SOURCE', '-DDM_ANDROID=1', '-DVK_USE_PLATFORM_ANDROID_KHR=1',
                       '-include', 'dm_android_compat.h'] + ['-I' + p for p in inc]
    # third-party code keeps its own flags (as the Mac Makefile builds it), without the host's compat header
    vpx = os.path.join(DM, 'third_party', 'libvpx'); dav = os.path.join(DM, 'third_party', 'dav1d')
    vpx_flags = target + ['-O2', '-g', '-fPIE', '-I' + vpx, '-I' + os.path.join(vpx, 'config')]
    dav_flags = target + ['-O2', '-g', '-fPIE', '-fvisibility=hidden', '-fomit-frame-pointer', '-I' + os.path.join(OUT, 'dav1d-config'), '-I' + dav,
                          '-I' + os.path.join(dav, 'include'), '-I' + os.path.join(dav, 'config')]
    # volk (Vulkan through function pointers, so the driver can be chosen at run time: vk_driver.c) and libadrenotools' loader with its linker-namespace helper
    volk = os.path.join(DM, 'third_party', 'volk'); adt = os.path.join(DM, 'third_party', 'libadrenotools')
    volk_flags = target + ['-O2', '-g', '-fPIE', '-DVK_USE_PLATFORM_ANDROID_KHR=1', '-I' + os.path.join(DM, 'third_party', 'vulkan-headers', 'include')]
    adt_flags = target + ['-O2', '-g', '-fPIE', '-std=c++20', '-I' + os.path.join(adt, 'include'), '-I' + adt, '-I' + os.path.join(adt, 'lib', 'linkernsbypass')]
    adt_src = [os.path.join(adt, 'src', 'driver.cpp'), os.path.join(adt, 'lib', 'linkernsbypass', 'android_linker_ns.cpp'), os.path.join(adt, 'lib', 'linkernsbypass', 'elf_soname_patcher.cpp')]

    host_src = sorted(glob.glob(os.path.join(HOST, '*.c')))
    gfx_src = sorted(sum((glob.glob(os.path.join(DM, 'gfx', d, '*.c')) for d in ('', 'nvn', 'tex')), []))
    android_src = sorted(glob.glob(os.path.join(DM, 'android', 'host', '*.c'))) + [os.path.join(DM, 'android', 'compat', 'dm_compat.c')]
    rel = lambda p: os.path.relpath(p, DM).replace('\\', '/')
    srcs = [p for p in host_src + gfx_src if rel(p) not in EXCLUDE] + android_src
    vpx_src = sorted(p for p in glob.glob(os.path.join(vpx, '**', '*.c'), recursive=True))
    dav_src = sorted([p for p in glob.glob(os.path.join(dav, 'src', '*.c')) if not p.endswith('_tmpl.c')] + [os.path.join(dav, 'src', 'arm', 'cpu.c')] +
                     glob.glob(os.path.join(dav, 'config', 'tmpl', '*.c')) +
                     [p for p in glob.glob(os.path.join(dav, 'src', 'arm', '64', '*.S')) if not p.endswith('_tmpl.S') and os.path.basename(p) != 'util.S'])

    jobs = []   # (source, object, flags, group)
    def add(group, files, flags, root):
        stamp = os.path.join(OUT, f'flags-{group}.txt'); text = '\n'.join(flags)
        if not os.path.exists(stamp) or open(stamp, encoding='utf-8').read() != text: open(stamp, 'w', encoding='utf-8').write(text)
        for p in files:
            o = os.path.join(OUT, 'obj', group, os.path.relpath(p, root).replace('\\', '/').replace('/', '__') + '.o')
            jobs.append((p, o, flags, stamp))
    add('host', srcs, cflags, DM)
    add('asm', asm, target + ['-g'], asm_dir)
    add('vpx', vpx_src, vpx_flags, vpx)
    add('dav1d', dav_src, dav_flags, dav)
    add('volk', [os.path.join(volk, 'volk.c')], volk_flags, volk)
    add('adrenotools', adt_src, adt_flags, adt)
    # FSR 2 (dm/third_party/fsr2, written for Windows): its core and Vulkan backend, the shader blob table (gen_fsr) and the C++ glue (fsr2_glue.cpp) behind the
    # C interface vkd_dlss calls. fsr2_compat.h force-included stands in for the Microsoft C runtime's checked string functions; FFX_GCC drops its __declspec exports
    fsr = os.path.join(DM, 'third_party', 'fsr2', 'src', 'ffx-fsr2-api')
    fsr_flags = target + ['-O2', '-g', '-fPIE', '-std=c++17', '-D_GNU_SOURCE', '-DVK_USE_PLATFORM_ANDROID_KHR=1', '-DFFX_GCC=1', '-DFFX_BACKEND_VK=1', '-Wno-unknown-pragmas',
                          '-include', 'fsr2_compat.h', '-I' + HERE, '-I' + fsr, '-I' + gen_fsr, '-I' + os.path.join(DM, 'third_party', 'volk'),
                          '-I' + os.path.join(DM, 'third_party', 'vulkan-headers', 'include')]
    add('fsr2', [os.path.join(fsr, 'ffx_fsr2.cpp'), os.path.join(fsr, 'vk', 'ffx_fsr2_vk.cpp'), os.path.join(fsr, 'vk', 'shaders', 'ffx_fsr2_shaders_vk.cpp'),
                 os.path.join(HERE, 'fsr2_glue.cpp')], fsr_flags, DM)

    todo = [j for j in jobs if stale(j[1], j[1][:-2] + '.d', j[3])]
    print(f'{len(jobs)} objects, {len(todo)} to compile ({a.j} at a time)', flush=True)
    t0 = time.time()
    def compile_one(j):
        src, obj, flags, _ = j
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        r = subprocess.run([cc] + flags + ['-MD', '-MF', obj[:-2] + '.d', '-c', src, '-o', obj], capture_output=True, text=True, encoding='utf-8', errors='replace')
        return src, r.returncode, r.stderr
    failed = []
    with cf.ThreadPoolExecutor(a.j) as ex:
        for src, rc, err in ex.map(compile_one, todo):
            warn = [l for l in err.splitlines() if ': warning:' in l]
            if rc:
                failed.append(src)
                print(f'[RED] {rel(src) if src.startswith(DM) else src}', flush=True)
                for l in [l for l in err.splitlines() if ': error:' in l or ': fatal error:' in l][:8]: print('    ' + l.split(os.sep)[-1][:300], flush=True)
            elif warn and '/third_party/' not in src.replace('\\', '/'):
                print(f'  {len(warn)} warning(s) in {rel(src)}: {warn[0].split(": warning: ", 1)[-1][:160]}', flush=True)
    print(f'compiled in {time.time() - t0:.1f} s', flush=True)
    if failed: sys.exit(f'[RED] {len(failed)} file(s) failed to compile')

    exe = os.path.join(OUT, 'dmhost')
    objs = [j[1] for j in jobs]
    if os.path.exists(exe) and not todo and os.path.getmtime(exe) >= max(os.path.getmtime(o) for o in objs):
        print(f'up to date: {exe}'); app_libs(cc, target, exe); return
    rsp = os.path.join(OUT, 'link.rsp')
    open(rsp, 'w', encoding='utf-8').write('\n'.join('"' + o.replace('\\', '/') + '"' for o in objs))
    # no -lvulkan: the driver is opened at run time (vk_driver.c); clang++ with the C++ runtime linked in, for libadrenotools
    r = subprocess.run([cxx(cc)] + target + ['-fPIE', '-pie', '-g', '-static-libstdc++', '-o', exe, '@' + rsp, '-lm', '-ldl', '-llog', '-lnativewindow', '-landroid', '-laaudio', '-Wl,--no-undefined', '-Wl,--error-limit=0'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        und = sorted(set(re.findall(r'undefined symbol: (\S+)', r.stderr)))
        print(f'[RED] link failed: {len(und)} undefined symbol(s)', flush=True)
        for u in und: print('    ' + u)
        if not und: print(r.stderr[-3000:])
        sys.exit(1)
    print(f'linked {exe} ({os.path.getsize(exe) / 1048576:.1f} MB)')
    app_libs(cc, target, exe)


def cxx(cc):   # the NDK's clang++ next to its clang
    d, b = os.path.split(cc); return os.path.join(d, b.replace('clang', 'clang++', 1))


def adreno_hooks(cc, target):
    """libadrenotools' hook libraries (dm/third_party/libadrenotools/src/hook), which its loader finds in the app's native library folder (DM_HOOK_DIR): the custom-driver
    hook libmain_hook.so with its implementation libhook_impl.so, and the two optional hooks it may load (file redirection, GPU memory import). Built as its CMake does"""
    adt = os.path.join(DM, 'third_party', 'libadrenotools'); hk = os.path.join(adt, 'src', 'hook'); lnb = os.path.join(adt, 'lib', 'linkernsbypass')
    common = target + ['-shared', '-fPIC', '-O2', '-g', '-Wall', '-I' + os.path.join(adt, 'include'), '-I' + lnb]
    impl = os.path.join(OUT, 'libhook_impl.so'); out = [impl]
    jobs = [(impl, [cxx(cc)] + common + ['-std=c++20', '-fvisibility=hidden', '-static-libstdc++', '-o', impl, os.path.join(hk, 'hook_impl.cpp'), os.path.join(lnb, 'android_linker_ns.cpp'),
                                          os.path.join(lnb, 'elf_soname_patcher.cpp'), '-Wl,-soname,libhook_impl.so', '-landroid', '-ldl', '-llog'], [os.path.join(hk, 'hook_impl.cpp')])]
    for name in ('main_hook', 'file_redirect_hook', 'gsl_alloc_hook'):
        so = os.path.join(OUT, f'lib{name}.so'); out.append(so)
        jobs.append((so, [cc] + common + ['-o', so, os.path.join(hk, f'{name}.c'), '-Wl,-soname,' + os.path.basename(so), '-Wl,-z,global', impl, '-ldl', '-llog'], [os.path.join(hk, f'{name}.c')]))
    for so, cmd, srcs in jobs:
        deps = srcs + glob.glob(os.path.join(hk, '*.h')) + [os.path.abspath(__file__)]
        if os.path.exists(so) and os.path.getmtime(so) >= max(os.path.getmtime(d) for d in deps): continue
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode: print(r.stderr[-3000:]); sys.exit(f'[RED] {os.path.basename(so)} failed to build')
        print(f'built {so}')
    return out


def app_libs(cc, target, exe):
    """The launcher app's native libraries (dm/android/app/src/main/jniLibs/arm64-v8a, git-ignored): libdmfront.so (dm/android/front: the game's screen, input) and
    the host itself as libdmhost.so. The app runs the host from its native library folder, the one place Android lets an app execute a file it ships"""
    jni = os.path.join(DM, 'android', 'app', 'src', 'main', 'jniLibs', 'arm64-v8a'); os.makedirs(jni, exist_ok=True)
    src = os.path.join(DM, 'android', 'front', 'dmfront.c'); so = os.path.join(OUT, 'libdmfront.so')
    deps = [src, os.path.join(DM, 'android', 'host', 'dm_front_proto.h')]
    if not os.path.exists(so) or os.path.getmtime(so) < max(os.path.getmtime(d) for d in deps):
        r = subprocess.run([cc] + target + ['-shared', '-fPIC', '-O2', '-g', '-Wall', '-o', so, src, '-landroid', '-lnativewindow', '-lEGL', '-lGLESv2', '-llog', '-Wl,--no-undefined'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode: print(r.stderr[-3000:]); sys.exit('[RED] libdmfront.so failed to build')
        print(f'built {so}')
    hooks = adreno_hooks(cc, target)
    for a, b in [(exe, 'libdmhost.so'), (so, 'libdmfront.so')] + [(h, os.path.basename(h)) for h in hooks]:
        d = os.path.join(jni, b)
        if not os.path.exists(d) or os.path.getmtime(d) < os.path.getmtime(a): shutil.copy2(a, d); print(f'copied to the app: {d}')


if __name__ == '__main__':
    main()
