// The Android host's implementations of the Apple calls the compat headers declare (dm/android/compat/include): Mach memory calls on Linux, getsectiondata
// over the linker's __start_ / __stop_ brackets, SHA-256, clonefile as a recursive copy, and the BSD libc extras.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/getsect.h>
#include <sys/clonefile.h>
#include <CommonCrypto/CommonDigest.h>

uint64_t mach_absolute_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

// ---- Mach memory ----
// A copy within this process that reports an unmapped or unreadable page as KERN_INVALID_ADDRESS instead of faulting, as vm_read_overwrite does
kern_return_t vm_read_overwrite(vm_map_t task, vm_address_t addr, vm_size_t size, vm_address_t data, vm_size_t *out_size) {
    (void)task;
    struct iovec local = {(void *)data, size}, remote = {(void *)addr, size};
    ssize_t n = size ? process_vm_readv(getpid(), &local, 1, &remote, 1, 0) : 0;
    if (out_size) *out_size = n > 0 ? (vm_size_t)n : 0;
    return n == (ssize_t)size ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
// process_vm_writev honours page protection like vm_write: a read-only page fails (the cheats writer then writes through its alias, or not at all)
kern_return_t vm_write(vm_map_t task, vm_address_t addr, vm_offset_t data, mach_msg_type_number_t count) {
    (void)task;
    struct iovec local = {(void *)data, count}, remote = {(void *)addr, count};
    ssize_t n = count ? process_vm_writev(getpid(), &local, 1, &remote, 1, 0) : 0;
    return n == (ssize_t)count ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
static int prot_of(vm_prot_t p) { return (p & VM_PROT_READ ? PROT_READ : 0) | (p & VM_PROT_WRITE ? PROT_WRITE : 0) | (p & VM_PROT_EXECUTE ? PROT_EXEC : 0); }
kern_return_t vm_allocate(vm_map_t task, vm_address_t *addr, vm_size_t size, int flags) {
    (void)task;
    void *p = mmap(flags & VM_FLAGS_ANYWHERE ? NULL : (void *)*addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (flags & VM_FLAGS_ANYWHERE ? 0 : MAP_FIXED_NOREPLACE), -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    *addr = (vm_address_t)p; return KERN_SUCCESS;
}
kern_return_t vm_deallocate(vm_map_t task, vm_address_t addr, vm_size_t size) {
    (void)task;
    if (task == mach_task_self() && size && addr && !munmap((void *)addr, size)) return KERN_SUCCESS;
    // task_threads hands out a malloc'd array here (it never succeeds on Android, so nothing reaches this)
    return size ? KERN_INVALID_ADDRESS : KERN_SUCCESS;
}
kern_return_t vm_protect(vm_map_t task, vm_address_t addr, vm_size_t size, boolean_t set_max, vm_prot_t prot) {
    (void)task; (void)set_max;
    return mprotect((void *)addr, size, prot_of(prot)) ? KERN_PROTECTION_FAILURE : KERN_SUCCESS;
}
kern_return_t mach_vm_protect(vm_map_t task, mach_vm_address_t addr, mach_vm_size_t size, boolean_t set_max, vm_prot_t prot) { return vm_protect(task, (vm_address_t)addr, (vm_size_t)size, set_max, prot); }
kern_return_t mach_vm_deallocate(vm_map_t task, mach_vm_address_t addr, mach_vm_size_t size) { return vm_deallocate(task, (vm_address_t)addr, (vm_size_t)size); }
// The first mapping at or above *addr, from /proc/self/maps: its start, length and protection
kern_return_t vm_region_64(vm_map_t task, vm_address_t *addr, vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info, mach_msg_type_number_t *count, mach_port_t *object) {
    (void)task;
    if (flavor != VM_REGION_BASIC_INFO_64 || !count || *count < VM_REGION_BASIC_INFO_COUNT_64) return KERN_INVALID_ARGUMENT;
    FILE *f = fopen("/proc/self/maps", "re");
    if (!f) return KERN_FAILURE;
    char line[512]; kern_return_t r = KERN_INVALID_ADDRESS;
    while (fgets(line, sizeof line, f)) {
        unsigned long lo, hi, off; char perm[5];
        if (sscanf(line, "%lx-%lx %4s %lx", &lo, &hi, perm, &off) != 4 || hi <= *addr) continue;
        vm_region_basic_info_data_64_t *bi = (vm_region_basic_info_data_64_t *)info;
        memset(bi, 0, sizeof *bi);
        bi->protection = (perm[0] == 'r' ? VM_PROT_READ : 0) | (perm[1] == 'w' ? VM_PROT_WRITE : 0) | (perm[2] == 'x' ? VM_PROT_EXECUTE : 0);
        bi->max_protection = VM_PROT_ALL; bi->shared = perm[3] == 's'; bi->offset = off;
        *addr = lo; *size = hi - lo; *count = VM_REGION_BASIC_INFO_COUNT_64;
        if (object) *object = MACH_PORT_NULL;
        r = KERN_SUCCESS; break;
    }
    fclose(f); return r;
}
kern_return_t mach_vm_region(vm_map_t task, mach_vm_address_t *addr, mach_vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info, mach_msg_type_number_t *count, mach_port_t *object) {
    vm_address_t a = (vm_address_t)*addr; vm_size_t s = 0;
    kern_return_t r = vm_region_64(task, &a, &s, flavor, info, count, object);
    if (r == KERN_SUCCESS) { *addr = a; *size = s; }
    return r;
}
// A second mapping of private anonymous pages has no Linux form (mremap with old size 0 duplicates only shared mappings)
kern_return_t vm_remap(vm_map_t target, vm_address_t *target_addr, vm_size_t size, vm_address_t mask, int flags, vm_map_t src_task, vm_address_t src_addr,
                       boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit) {
    (void)target; (void)target_addr; (void)size; (void)mask; (void)flags; (void)src_task; (void)src_addr; (void)copy; (void)cur; (void)max; (void)inherit;
    return KERN_NOT_SUPPORTED;
}
kern_return_t mach_vm_remap(vm_map_t target, mach_vm_address_t *target_addr, mach_vm_size_t size, mach_vm_offset_t mask, int flags, vm_map_t src_task,
                            mach_vm_address_t src_addr, boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit) {
    (void)target; (void)target_addr; (void)size; (void)mask; (void)flags; (void)src_task; (void)src_addr; (void)copy; (void)cur; (void)max; (void)inherit;
    return KERN_NOT_SUPPORTED;
}

kern_return_t task_info(task_t task, task_flavor_t flavor, task_info_t info, mach_msg_type_number_t *count) {
    (void)task;
    if (flavor != TASK_VM_INFO || !count || *count < TASK_VM_INFO_COUNT) return KERN_NOT_SUPPORTED;
    FILE *f = fopen("/proc/self/statm", "re"); unsigned long long size = 0, res = 0;
    if (!f) return KERN_FAILURE;
    int ok = fscanf(f, "%llu %llu", &size, &res) == 2; fclose(f);
    if (!ok) return KERN_FAILURE;
    uint64_t pg = (uint64_t)getpagesize();
    task_vm_info_data_t *vi = (task_vm_info_data_t *)info; memset(vi, 0, sizeof *vi);
    vi->virtual_size = size * pg; vi->page_size = (integer_t)pg; vi->resident_size = res * pg; vi->phys_footprint = res * pg;
    *count = TASK_VM_INFO_COUNT; return KERN_SUCCESS;
}

// ---- Mach thread inspection: not available (the diagnostics that use it report nothing) ----
kern_return_t thread_suspend(thread_act_t t) { (void)t; return KERN_NOT_SUPPORTED; }
kern_return_t thread_resume(thread_act_t t) { (void)t; return KERN_NOT_SUPPORTED; }
kern_return_t thread_get_state(thread_act_t t, thread_flavor_t f, thread_state_t s, mach_msg_type_number_t *c) { (void)t; (void)f; (void)s; (void)c; return KERN_NOT_SUPPORTED; }
kern_return_t thread_info(thread_act_t t, thread_flavor_t f, thread_info_t i, mach_msg_type_number_t *c) { (void)t; (void)f; (void)i; (void)c; return KERN_NOT_SUPPORTED; }
kern_return_t task_threads(task_t task, thread_act_array_t *list, mach_msg_type_number_t *count) { (void)task; if (list) *list = NULL; if (count) *count = 0; return KERN_NOT_SUPPORTED; }

// ---- getsectiondata over the linker's section brackets ----
const struct mach_header_64 _mh_execute_header = {0xfeedfacf};
#define BRACKETS(s) extern const uint8_t __start_##s[] __attribute__((weak)), __stop_##s[] __attribute__((weak));
BRACKETS(swhle) BRACKETS(swforce) BRACKETS(swnvn)
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *segname, const char *sectname, unsigned long *size) {
    (void)mh; (void)segname;
    const uint8_t *b = NULL, *e = NULL;
    if (!strcmp(sectname, "swhle")) { b = __start_swhle; e = __stop_swhle; }
    else if (!strcmp(sectname, "swforce")) { b = __start_swforce; e = __stop_swforce; }
    else if (!strcmp(sectname, "swnvn")) { b = __start_swnvn; e = __stop_swnvn; }
    if (!b || e <= b) { if (size) *size = 0; return NULL; }
    if (size) *size = (unsigned long)(e - b);
    return (uint8_t *)b;
}

// ---- SHA-256 (FIPS 180-4) ----
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(CC_SHA256_CTX *c, const uint8_t *p) {
    uint32_t w[64], a, b, d, e, f, g, h, cc, t1, t2;
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) w[i] = w[i - 16] + (ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10));
    a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
    for (int i = 0; i < 64; i++) {
        t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}
int CC_SHA256_Init(CC_SHA256_CTX *c) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(c->h, iv, sizeof iv); c->len = 0; c->n = 0; return 1;
}
int CC_SHA256_Update(CC_SHA256_CTX *c, const void *data, CC_LONG len) {
    const uint8_t *p = data; c->len += len;
    while (len) {
        uint32_t k = 64 - c->n < len ? 64 - c->n : len;
        memcpy(c->buf + c->n, p, k); c->n += k; p += k; len -= k;
        if (c->n == 64) { sha_block(c, c->buf); c->n = 0; }
    }
    return 1;
}
int CC_SHA256_Final(unsigned char *md, CC_SHA256_CTX *c) {
    uint64_t bits = c->len * 8; uint8_t pad = 0x80, z = 0, l[8];
    CC_SHA256_Update(c, &pad, 1);
    while (c->n != 56) CC_SHA256_Update(c, &z, 1);
    for (int i = 0; i < 8; i++) l[i] = (uint8_t)(bits >> (56 - 8 * i));
    CC_SHA256_Update(c, l, 8);
    for (int i = 0; i < 8; i++) { md[4 * i] = (uint8_t)(c->h[i] >> 24); md[4 * i + 1] = (uint8_t)(c->h[i] >> 16); md[4 * i + 2] = (uint8_t)(c->h[i] >> 8); md[4 * i + 3] = (uint8_t)c->h[i]; }
    return 1;
}
unsigned char *CC_SHA256(const void *data, CC_LONG len, unsigned char *md) { CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, data, len); CC_SHA256_Final(md, &c); return md; }

// ---- clonefile: a recursive copy ----
static int rm_tree(const char *p) {
    struct stat st; if (lstat(p, &st)) return errno == ENOENT ? 0 : -1;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(p); if (!d) return -1;
        struct dirent *e; char c[4096];
        while ((e = readdir(d))) { if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue; snprintf(c, sizeof c, "%s/%s", p, e->d_name); rm_tree(c); }
        closedir(d); return rmdir(p);
    }
    return unlink(p);
}
static int copy_file(const char *src, const char *dst, mode_t mode) {
    int in = open(src, O_RDONLY | O_CLOEXEC); if (in < 0) return -1;
    int out = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode & 0777); if (out < 0) { int e = errno; close(in); errno = e; return -1; }
    char buf[1 << 16]; ssize_t n; int r = 0;
    while ((n = read(in, buf, sizeof buf)) > 0) for (ssize_t o = 0; o < n;) { ssize_t w = write(out, buf + o, (size_t)(n - o)); if (w < 0) { r = -1; break; } o += w; }
    if (n < 0) r = -1;
    int e = errno; close(in); if (close(out)) r = -1; errno = e; return r;
}
static int copy_tree(const char *src, const char *dst) {
    struct stat st; if (lstat(src, &st)) return -1;
    if (S_ISREG(st.st_mode)) return copy_file(src, dst, st.st_mode);
    if (S_ISLNK(st.st_mode)) { char t[4096]; ssize_t n = readlink(src, t, sizeof t - 1); if (n < 0) return -1; t[n] = 0; return symlink(t, dst); }
    if (!S_ISDIR(st.st_mode)) { errno = ENOTSUP; return -1; }
    if (mkdir(dst, st.st_mode & 0777)) return -1;
    DIR *d = opendir(src); if (!d) return -1;
    struct dirent *e; char a[4096], b[4096]; int r = 0;
    while (!r && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        snprintf(a, sizeof a, "%s/%s", src, e->d_name); snprintf(b, sizeof b, "%s/%s", dst, e->d_name);
        r = copy_tree(a, b);
    }
    int er = errno; closedir(d); errno = er; return r;
}
int clonefile(const char *src, const char *dst, unsigned int flags) {
    (void)flags;
    struct stat st;
    if (!lstat(dst, &st)) { errno = EEXIST; return -1; }
    if (!copy_tree(src, dst)) return 0;
    int e = errno; rm_tree(dst); errno = e; return -1;
}

// ---- CFDictionary with NULL callbacks: an address-keyed open-addressing table (keys are never NULL in the host's use) ----
#include <CoreFoundation/CoreFoundation.h>
struct dm_cfdict { const void **k, **v; size_t cap, n, used; };   // used: live entries + tombstones
static const void *const TOMB = (const void *)&TOMB;
static size_t slot_of(const void *k, size_t cap) { uintptr_t x = (uintptr_t)k; x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; return (size_t)x & (cap - 1); }
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex capacity, const void *kc, const void *vc) {
    (void)a; (void)capacity; (void)kc; (void)vc;
    struct dm_cfdict *d = calloc(1, sizeof *d); if (!d) return NULL;
    d->cap = 64; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v);
    if (!d->k || !d->v) { free(d->k); free(d->v); free(d); return NULL; }
    return d;
}
static size_t find(const struct dm_cfdict *d, const void *key) {   // its slot, or SIZE_MAX
    for (size_t i = slot_of(key, d->cap), n = 0; n < d->cap; i = (i + 1) & (d->cap - 1), n++) {
        if (!d->k[i]) return SIZE_MAX;
        if (d->k[i] == key) return i;
    }
    return SIZE_MAX;
}
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *key) { size_t i = d ? find(d, key) : SIZE_MAX; return i == SIZE_MAX ? NULL : d->v[i]; }
static void grow(struct dm_cfdict *d) {
    struct dm_cfdict o = *d;
    d->cap = o.n * 4 > o.cap ? o.cap * 2 : o.cap; d->n = d->used = 0;
    d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v);
    for (size_t i = 0; i < o.cap; i++) if (o.k[i] && o.k[i] != TOMB) CFDictionarySetValue(d, o.k[i], o.v[i]);
    free(o.k); free(o.v);
}
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *value) {
    size_t i = find(d, key);
    if (i != SIZE_MAX) { d->v[i] = value; return; }
    if ((d->used + 1) * 4 > d->cap * 3) grow(d);
    for (i = slot_of(key, d->cap); d->k[i] && d->k[i] != TOMB; i = (i + 1) & (d->cap - 1)) {}
    if (!d->k[i]) d->used++;
    d->k[i] = key; d->v[i] = value; d->n++;
}
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key) { size_t i = find(d, key); if (i != SIZE_MAX) { d->k[i] = TOMB; d->v[i] = NULL; d->n--; } }
CFIndex CFDictionaryGetCount(CFDictionaryRef d) { return d ? (CFIndex)d->n : 0; }
void CFRelease(CFTypeRef t) { struct dm_cfdict *d = (struct dm_cfdict *)t; if (d) { free(d->k); free(d->v); free(d); } }

// ---- BSD libc extras ----
int dm_getdtablesize(void) { struct rlimit rl; return getrlimit(RLIMIT_NOFILE, &rl) || rl.rlim_cur > 0x7fffffff ? 1024 : (int)rl.rlim_cur; }
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *)) {
    if (n < 2) return 0;
    char *a = base, *tmp = malloc(n * size); if (!tmp) { errno = ENOMEM; return -1; }
    for (size_t w = 1; w < n; w *= 2) {   // bottom-up merge sort: stable, O(n log n)
        for (size_t lo = 0; lo < n; lo += 2 * w) {
            size_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) { if (cmp(a + j * size, a + i * size) < 0) memcpy(tmp + k++ * size, a + j++ * size, size); else memcpy(tmp + k++ * size, a + i++ * size, size); }
            if (i < mid) { memcpy(tmp + k * size, a + i * size, (mid - i) * size); k += mid - i; }
            if (j < hi) { memcpy(tmp + k * size, a + j * size, (hi - j) * size); }
        }
        memcpy(a, tmp, n * size);
    }
    free(tmp); return 0;
}
void *pthread_get_stackaddr_np(pthread_t t) {   // Darwin returns the top (highest address) of the stack
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return NULL;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return (char *)lo + sz;
}
size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return 0;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return sz;
}
int pthread_cond_timedwait_relative_np(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel) {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec += rel->tv_sec; t.tv_nsec += rel->tv_nsec;
    if (t.tv_nsec >= 1000000000) { t.tv_sec++; t.tv_nsec -= 1000000000; }
    return pthread_cond_timedwait(c, m, &t);
}
