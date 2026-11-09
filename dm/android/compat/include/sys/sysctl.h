// Android stand-in for the BSD <sys/sysctl.h> the host reads: process start time (main.c) and the CPU's performance levels (hle_nvn.c). Both callers fall back
// when the call fails, which it always does here (ENOENT): the boot timeline starts at the first clock read, and the CPU counts as one performance level.
#pragma once
#include <errno.h>
#include <stddef.h>
#include <sys/time.h>
#define CTL_KERN 1
#define KERN_PROC 14
#define KERN_PROC_PID 1
struct kinfo_proc { struct { struct timeval p_starttime; } kp_proc; };
static inline int sysctl(int *name, unsigned namelen, void *old, size_t *oldlen, void *new_, size_t newlen) { (void)name; (void)namelen; (void)old; (void)oldlen; (void)new_; (void)newlen; errno = ENOENT; return -1; }
static inline int sysctlbyname(const char *name, void *old, size_t *oldlen, void *new_, size_t newlen) { (void)name; (void)old; (void)oldlen; (void)new_; (void)newlen; errno = ENOENT; return -1; }
