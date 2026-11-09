// Force-included into every Android host source (Makefile.android: -include dm_android_compat.h): the BSD / Darwin libc calls the host makes that bionic
// lacks or names differently. Implementations that need more than a line are in dm/android/compat/dm_compat.c.
#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/param.h>

#include <pthread/qos.h>   // Darwin's <pthread.h> brings the QoS classes with it; several host files use them without including this
#include <TargetConditionals.h>   // so do Darwin's system headers with TARGET_OS_IPHONE

// clock_gettime_nsec_np (Darwin): the clock in nanoseconds
static inline uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; if (clock_gettime(c, &t)) return 0; return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// getdtablesize (BSD): the descriptor table size, which is the open file limit
int dm_getdtablesize(void);
#define getdtablesize dm_getdtablesize
// mergesort (BSD): a stable sort; returns 0, or -1 with errno when it cannot allocate
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));
// F_GETPATH (Darwin fcntl): the path of an open descriptor. Bionic has no such command, so fcntl fails with EINVAL and the only caller, the open-file
// census (hle_fdstat.c, a diagnostic), counts that descriptor without a path
#define F_GETPATH 50
// pthread_get_stackaddr_np / pthread_get_stacksize_np (Darwin): the top of a thread's stack and its size, from pthread_getattr_np
void *pthread_get_stackaddr_np(pthread_t t);
size_t pthread_get_stacksize_np(pthread_t t);
// pthread_cond_timedwait_relative_np (Darwin): a wait for a relative time, as an absolute CLOCK_REALTIME wait (bionic's default condition clock)
int pthread_cond_timedwait_relative_np(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel);
// malloc_size / malloc_good_size (Darwin): the usable size of an allocation; the size an allocation of n would get
#include <malloc.h>
static inline size_t malloc_size(const void *p) { return p ? malloc_usable_size((void *)p) : 0; }
static inline size_t malloc_good_size(size_t n) { return n; }
// MAP_ANON is MAP_ANONYMOUS on Linux; bionic defines both
// arc4random (BSD) is in bionic. strlcpy / strlcat are in bionic.

// ---- Signal context registers (dm/host/dm.h DM_UC_SS / DM_UC_NS / DM_UC_ESR) ----
// Linux's struct sigcontext is {fault_address, regs[31], sp, pc, pstate, __reserved}: regs[0..28] are x0-x28, regs[29] fp, regs[30] lr, then sp, pc and pstate,
// the same order as Apple's arm64 thread state, so that structure laid over &regs[0] reads them. The SIMD state and the ESR are tagged records in __reserved.
#include <signal.h>
#include <sys/ucontext.h>
#include <asm/sigcontext.h>
struct __darwin_arm_thread_state64 { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; };
typedef struct __darwin_arm_thread_state64 _STRUCT_ARM_THREAD_STATE64;
struct __darwin_arm_neon_state64 { struct _aarch64_ctx __head; uint32_t __fpsr, __fpcr; __uint128_t __v[32]; };   // = struct fpsimd_context
typedef struct __darwin_arm_neon_state64 _STRUCT_ARM_NEON_STATE64;
_Static_assert(sizeof(struct __darwin_arm_neon_state64) == sizeof(struct fpsimd_context), "fpsimd_context layout");
static inline void *dm_uc_record(ucontext_t *uc, uint32_t magic) {
    for (uint8_t *p = (uint8_t *)uc->uc_mcontext.__reserved, *e = p + sizeof uc->uc_mcontext.__reserved; p + sizeof(struct _aarch64_ctx) <= e;) {
        struct _aarch64_ctx *h = (struct _aarch64_ctx *)p;
        if (!h->magic || !h->size) break;
        if (h->magic == magic) return h;
        p += h->size;
    }
    return 0;
}
static inline uint64_t dm_uc_esr(ucontext_t *uc) { struct esr_context *e = (struct esr_context *)dm_uc_record(uc, ESR_MAGIC); return e ? e->esr : 0; }
static struct __darwin_arm_neon_state64 dm_uc_no_neon;   // a zero state when the kernel gave no FPSIMD record (it always does on arm64)
static inline struct __darwin_arm_neon_state64 *dm_uc_neon(ucontext_t *uc) { void *r = dm_uc_record(uc, FPSIMD_MAGIC); return r ? (struct __darwin_arm_neon_state64 *)r : &dm_uc_no_neon; }
#define DM_UC_SS(uc) ((struct __darwin_arm_thread_state64 *)&(uc)->uc_mcontext.regs[0])
#define DM_UC_NS(uc) dm_uc_neon(uc)
#define DM_UC_ESR(uc) dm_uc_esr(uc)

// ---- Thread names: Darwin's pthread_setname_np names the calling thread ----
#define pthread_setname_np(name) pthread_setname_np(pthread_self(), (name))

// ---- Extended attributes: Darwin's calls take a position and options (XATTR_NOFOLLOW); Linux names user attributes "user.<name>" ----
#include <sys/xattr.h>
#include <errno.h>
#include <stdio.h>
#define XATTR_NOFOLLOW 0x0001
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
static inline const char *dm_xattr_name(const char *n, char *buf, size_t cap) { snprintf(buf, cap, "user.%s", n); return buf; }
static inline ssize_t dm_getxattr(const char *p, const char *n, void *v, size_t s, uint32_t pos, int opt) {
    char b[256]; (void)pos; dm_xattr_name(n, b, sizeof b); return opt & XATTR_NOFOLLOW ? lgetxattr(p, b, v, s) : getxattr(p, b, v, s); }
static inline int dm_setxattr(const char *p, const char *n, const void *v, size_t s, uint32_t pos, int opt) {
    char b[256]; (void)pos; dm_xattr_name(n, b, sizeof b); return opt & XATTR_NOFOLLOW ? lsetxattr(p, b, v, s, 0) : setxattr(p, b, v, s, 0); }
static inline int dm_removexattr(const char *p, const char *n, int opt) {
    char b[256]; dm_xattr_name(n, b, sizeof b); return opt & XATTR_NOFOLLOW ? lremovexattr(p, b) : removexattr(p, b); }
#define getxattr(p, n, v, s, pos, opt) dm_getxattr(p, n, v, s, pos, opt)
#define setxattr(p, n, v, s, pos, opt) dm_setxattr(p, n, v, s, pos, opt)
#define removexattr(p, n, opt) dm_removexattr(p, n, opt)

// ---- renamex_np(RENAME_SWAP): Linux's atomic exchange of two paths, renameat2(RENAME_EXCHANGE) ----
#include <stdio.h>
#define RENAME_SWAP 0x2
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)
#endif
static inline int renamex_np(const char *a, const char *b, unsigned flags) { return renameat2(AT_FDCWD, a, AT_FDCWD, b, flags & RENAME_SWAP ? RENAME_EXCHANGE : 0); }

// ---- struct stat: Darwin's st_<x>timespec are Linux's st_<x>tim ----
#define st_mtimespec st_mtim
#define st_atimespec st_atim
#define st_ctimespec st_ctim
