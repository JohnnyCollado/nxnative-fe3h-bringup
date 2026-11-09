// Android stand-in for <mach/mach_time.h>: mach_absolute_time counts CLOCK_MONOTONIC nanoseconds, so the timebase is 1/1 and the engine's conversions stay exact.
#pragma once
#include <stdint.h>
#include <time.h>
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t;
typedef mach_timebase_info_data_t *mach_timebase_info_t;
uint64_t mach_absolute_time(void);   // dm/android/compat/dm_compat.c: an out-of-line function, since some host files declare it themselves instead of including this
static inline uint64_t mach_continuous_time(void) { struct timespec t; clock_gettime(CLOCK_BOOTTIME, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static inline int mach_timebase_info(mach_timebase_info_t i) { i->numer = 1; i->denom = 1; return 0; }
// Sleep until mach_absolute_time() reaches the deadline (an absolute CLOCK_MONOTONIC time in nanoseconds), resuming after a signal as Mach does
#include <errno.h>
static inline int mach_wait_until(uint64_t deadline) {
    struct timespec t = {(time_t)(deadline / 1000000000ull), (long)(deadline % 1000000000ull)};
    int r; while ((r = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL)) == EINTR) {}
    return r ? 14 /* KERN_ABORTED */ : 0;
}
