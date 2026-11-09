// Android stand-in for <dispatch/dispatch.h>: semaphores and times only (hle_audio.c's release signalling). The queue calls take blocks, which Android's C has
// no runtime for, so they are not declared here: the files that use queues have an Android branch of their own (hle_lifecycle.c, hle_cheats.c).
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <semaphore.h>
typedef uint64_t dispatch_time_t;   // an absolute CLOCK_MONOTONIC time in nanoseconds, or DISPATCH_TIME_FOREVER
#define DISPATCH_TIME_NOW ((dispatch_time_t)0)
#define DISPATCH_TIME_FOREVER (~(dispatch_time_t)0)
#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000ull
#define NSEC_PER_MSEC 1000000ull
#define NSEC_PER_USEC 1000ull
#define USEC_PER_SEC 1000000ull
#endif
static inline dispatch_time_t dispatch_time(dispatch_time_t when, int64_t delta) {
    if (when == DISPATCH_TIME_FOREVER) return when;
    if (when == DISPATCH_TIME_NOW) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); when = (uint64_t)t.tv_sec * NSEC_PER_SEC + (uint64_t)t.tv_nsec; }
    return delta < 0 && (uint64_t)-delta > when ? 1 : when + (uint64_t)delta;
}
typedef struct dm_dispatch_sem { sem_t s; } *dispatch_semaphore_t;
static inline dispatch_semaphore_t dispatch_semaphore_create(long value) {
    dispatch_semaphore_t d = malloc(sizeof *d); if (!d) return NULL;
    if (sem_init(&d->s, 0, (unsigned)value)) { free(d); return NULL; }
    return d;
}
static inline long dispatch_semaphore_signal(dispatch_semaphore_t d) { sem_post(&d->s); return 0; }
// 0 when signalled, non-zero when the time passed first (as dispatch_semaphore_wait)
static inline long dispatch_semaphore_wait(dispatch_semaphore_t d, dispatch_time_t timeout) {
    if (timeout == DISPATCH_TIME_FOREVER) { while (sem_wait(&d->s)) if (errno != EINTR) return -1; return 0; }
    struct timespec t = {(time_t)(timeout / NSEC_PER_SEC), (long)(timeout % NSEC_PER_SEC)};
    while (sem_clockwait(&d->s, CLOCK_MONOTONIC, &t)) { if (errno == EINTR) continue; return -1; }
    return 0;
}
static inline void dispatch_release(void *o) { dispatch_semaphore_t d = o; if (d) { sem_destroy(&d->s); free(d); } }
