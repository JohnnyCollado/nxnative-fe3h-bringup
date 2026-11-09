#!/usr/bin/env python3
# Switch 2 (NVN2) shaders → SPIR-V only, for a Vulkan host (the Android port): the first stage of dmsh_build.py's route B without the Metal stages after it
# (spirv-cross MSL, metallib). The translator (dmsh_xlate, the Ryujinx shader translator) already writes SPIR-V for Vulkan (Program.cs: TargetApi.Vulkan); Metal is
# only what dmsh_build.py does with it afterwards. The MSL-side rewrites dmsh_build.py makes (depth convention entry points, NaN-exact min / max, padding, alpha and logic
# operations, vertex pulling) are not made here: a Vulkan host must give each its Vulkan form.
# The jobs are built as dmsh_build.py job_for builds them for an NVN2 record (Ampere code, arch sm86, c1 from dmsh_ampere, nvn2_cs for compute, the profile's answer table);
# Maxwell (NVN1) records are listed as skipped: their VertexA / B merge, transform feedback layout and early fragment tests live in dmsh_build.py.
# Not a chain tool (dmsh_chain.FILES): changing it changes no DMSH_CHAIN version.
# Usage: vk_spirv.py <package dir (shader_split.py output)> <output dir> [key ...]
# Environment: DM_GAME (required), XLATE (the translator: an executable, or a .dll run with `dotnet`; default the in-repo dmsh_xlate_bin/dmsh_xlate), J (parallel translator
#   processes, default the CPU count), XLATE_DEBUG (default 0 here: the optimizer on, see job_for) / XLATE_REDUCED (as in dmsh_build.py), DM_SHADER_TABLES (a folder holding shader_map.tsv and shader_facts.tsv to use
#   instead of the profile's).
# Writes <key>.spv and <key>.res.json per translated shader, jobs.json (every job), and summary.tsv (key, stage, result, message). Exit 1 when nothing translated.
# Vulkan form added here: StorageImageRead / WriteWithoutFormat on modules using a storage image of Unknown format (add_format_caps; 38 of Fortune's Weave's 16,888
# failed spirv-val without it). The device must support shaderStorageImageRead / WriteWithoutFormat.
import concurrent.futures as cf, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '../../games'))
import dmsh_ampere, dmsh_chain
import game as _game


def records(d):   # as dmsh_build.py records: a package file is BTCDBTCD-separated shader records
    r = [m.start() for m in re.finditer(b'BTCDBTCD', d)]
    return [d[x:(r[k + 1] if k + 1 < len(r) else len(d))] for k, x in enumerate(r)]


def tables():
    t = os.environ.get('DM_SHADER_TABLES')
    g = _game.load()
    return (os.path.join(t, 'shader_map.tsv') if t else str(g.table('shader_map.tsv')),
            os.path.join(t, 'shader_facts.tsv') if t else str(g.table('shader_facts.tsv')))


def load_map(path):   # key → (package file, record index), as dmsh_build.py loc
    loc = {}
    for ln in open(path, encoding='utf-8'):
        c = ln.rstrip('\r\n').split('\t')
        if len(c) > 4 and c[0].endswith('.bin'): loc.setdefault(c[3], (c[0], int(c[1])))
    return loc


def load_facts(path):   # as dmsh_facts.load
    if not os.path.exists(path): return {}
    return {k: json.loads(v) for k, v in (l.rstrip('\r\n').split('\t', 1) for l in open(path, encoding='utf-8') if l.strip() and not l.startswith('#'))}


def job_for(key, name, rec, out, facts):
    """dmsh_build.py job_for's NVN2 branch; None for a Maxwell record"""
    ci = rec.find(bytes.fromhex('34127698')); code = rec[0x38:ci]; stage = name[:2]
    if ci < 0 or not dmsh_ampere.is_nvn2(rec[ci:]): return None
    c1 = dmsh_ampere.c1_segment(rec[ci:], code)
    code = code[:dmsh_ampere.inst_len(rec[ci:])]
    extra = {'arch': 'sm86'}
    if stage == 'cs': extra['nvn2_cs'] = [rec[ci + 0xb1], rec[ci + 0xb2], rec[ci + 0xb3], int.from_bytes(rec[ci + 0xba:ci + 0xbc], 'little'), rec[ci + 0xa9] * 4]
    cp = os.path.join(out, 'code', key + '.bin'); open(cp, 'wb').write(code)
    j = {'key': key, 'stage': stage, 'code': cp, 'tex': {}, 'img': {}, 'attr': {}, 'fout': {}, 'c1': c1.hex(), **extra,
         # The translator's optimizer stays on here (debug off), unlike dmsh_build.py's Metal route, where the Metal compiler optimizes the MSL after it: the Adreno
         # driver's compiler handles the unoptimized SPIR-V badly. Fortune's Weave's light culling (35d12d6f98fa05e2, 32x18x16 groups) took 205 ms per dispatch as
         # translated in debug mode and tripped the GPU's watchdog (the device was lost at the first 3D frame); optimized, its frame runs in well under 30 ms
         'debug': os.environ.get('XLATE_DEBUG', '0') != '0', 'reduced': os.environ.get('XLATE_REDUCED', '1') == '1'}
    if key not in facts: j['no_facts'] = True
    else: j.update(facts[key])
    return j


OP_CAPABILITY, OP_TYPE_IMAGE, OP_IMAGE_READ, OP_IMAGE_WRITE = 17, 25, 98, 99
CAP_READ_WITHOUT_FORMAT, CAP_WRITE_WITHOUT_FORMAT = 55, 56


def add_format_caps(path):
    """Declares StorageImageRead / WriteWithoutFormat in a module that reads / writes a storage image of Unknown format (OpTypeImage Sampled 2, Format 0),
    which Vulkan requires (VUID-StandaloneSpirv-OpImageWrite-07112 and the read rule) and spirv-val checks. The translator leaves image formats Unknown when the
    answer table has none (NVN takes the format from the image bound at run time), and the Metal route never needed the declaration. → the capabilities added"""
    d = bytearray(open(path, 'rb').read())
    w = [int.from_bytes(d[i:i + 4], 'little') for i in range(0, len(d), 4)]
    if len(w) < 5 or w[0] != 0x07230203: return []
    caps, unknown_storage, results, reads, writes, cap_end = set(), set(), {}, set(), set(), 5
    i = 5
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff
        if n == 0: return []   # malformed: leave it to the validator
        if op == OP_CAPABILITY: caps.add(w[i + 1]); cap_end = i + n
        elif op == OP_TYPE_IMAGE and n >= 9 and w[i + 7] == 2 and w[i + 8] == 0: unknown_storage.add(w[i + 1])
        elif op in (OP_IMAGE_READ, OP_IMAGE_WRITE):
            (reads if op == OP_IMAGE_READ else writes).add(w[i + (3 if op == OP_IMAGE_READ else 1)])
        elif n >= 4 and op in (61, 100):   # OpLoad / OpImage: result id → its type, to find which image a read or write uses
            results[w[i + 2]] = w[i + 1]
        i += n
    def unknown(img_id): return results.get(img_id) in unknown_storage
    add = []
    if any(unknown(x) for x in reads) and CAP_READ_WITHOUT_FORMAT not in caps: add.append(CAP_READ_WITHOUT_FORMAT)
    if any(unknown(x) for x in writes) and CAP_WRITE_WITHOUT_FORMAT not in caps: add.append(CAP_WRITE_WITHOUT_FORMAT)
    if not add: return []
    w[cap_end:cap_end] = [w2 for c in add for w2 in (2 << 16 | OP_CAPABILITY, c)]
    open(path, 'wb').write(b''.join(x.to_bytes(4, 'little') for x in w))
    return add


def xlate_cmd():
    x = os.environ.get('XLATE', os.path.join(dmsh_chain.XLATE_DIR, 'dmsh_xlate'))
    return ['dotnet', x] if x.endswith('.dll') else [x]


def main():
    if len(sys.argv) < 3: sys.exit('usage: vk_spirv.py <package dir> <output dir> [key ...]')
    pkg, out, want = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    os.makedirs(os.path.join(out, 'code'), exist_ok=True)
    map_path, facts_path = tables()
    loc, facts = load_map(map_path), load_facts(facts_path)
    keys = sorted(k for k in loc if not want or k in want)
    packs, jobs, summary = {}, {}, {}
    for key in keys:
        name, k = loc[key]
        if name not in packs: packs[name] = records(open(os.path.join(pkg, name), 'rb').read())
        j = job_for(key, name, packs[name][k], out, facts)
        if j is None: summary[key] = (name[:2], 'skipped', 'Maxwell (NVN1) record: build it with dmsh_build.py')
        else: jobs[key] = j
    json.dump(list(jobs.values()), open(os.path.join(out, 'jobs.json'), 'w'), indent=0)
    nj = int(os.environ.get('J', str(os.cpu_count() or 8)))
    todo = list(jobs)
    chunks = [todo[i::nj] for i in range(nj) if todo[i::nj]]
    cmd = xlate_cmd()
    def run(ch):
        jp = os.path.join(out, 'code', f'job_{ch[0]}.json'); json.dump([jobs[k] for k in ch], open(jp, 'w'))
        return subprocess.run(cmd + [jp, out], capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(f'{len(keys)} shaders in the map, {len(jobs)} NVN2 jobs, {nj} translator processes', flush=True)
    with cf.ThreadPoolExecutor(nj) as ex: results = list(ex.map(run, chunks))
    errors = {}
    for r in results:
        for ln in r.stderr.splitlines():
            m = re.match(r'\[RED\] (\w+): (.*)', ln)
            if m: errors[m.group(1)] = m.group(2)[:200]
        if r.returncode and not any(re.match(r'\[RED\] \w+:', ln) for ln in r.stderr.splitlines()):
            print(f'[RED] the translator exited with {r.returncode}: {r.stderr.strip()[-400:]}', flush=True)
    for key, j in jobs.items():
        spv = os.path.join(out, key + '.spv')
        if key in errors: summary[key] = (j['stage'], 'failed', errors[key])
        elif os.path.exists(spv):
            notes = (['no facts (defaults)'] if j.get('no_facts') else []) + [{CAP_READ_WITHOUT_FORMAT: 'declared StorageImageReadWithoutFormat', CAP_WRITE_WITHOUT_FORMAT: 'declared StorageImageWriteWithoutFormat'}[c] for c in add_format_caps(spv)]
            summary[key] = (j['stage'], 'ok', '; '.join(notes))
        else: summary[key] = (j['stage'], 'failed', 'no spv')
    with open(os.path.join(out, 'summary.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('key\tstage\tresult\tmessage\n')
        for key in sorted(summary): f.write('\t'.join((key,) + summary[key]) + '\n')
    count = {}
    for st, res, _ in summary.values(): count[res] = count.get(res, 0) + 1
    print('result: ' + ', '.join(f'{v} {k}' for k, v in sorted(count.items())), flush=True)
    for key, (st, res, msg) in sorted(summary.items()):
        if res == 'failed': print(f'[RED] {key} ({st}): {msg}', flush=True)
    sys.exit(0 if count.get('ok') else 1)


if __name__ == '__main__':
    main()
