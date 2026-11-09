// Host lifecycle: graceful exit (SIGTERM; on iOS the App termination callback calls dm_host_shutdown).
// Pause audio, suspend graphics (wait for in-flight NVN calls to end and drain GPU work), then _exit(0); a 1.5 s fallback goes straight to _exit(0).
// The default action also exits in about 0.08 s (exit code 143); the value of this function is stopping background threads in order, exit code 0, and giving the iOS lifecycle callback an entry.
#include "hle.h"
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#include <pthread.h>
#include <stdatomic.h>
#include <signal.h>
#include <unistd.h>
void hlecount_flush(void); void dm_rec_finish(void);
void dm_host_shutdown(const char *why) {
    static _Atomic int once; if (atomic_exchange(&once, 1)) return;
    uint64_t t0 = now_ns(); LOG("received %s: graceful exit", why);
#ifdef __APPLE__
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 1500 * NSEC_PER_MSEC), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
        LOG("[RED] graceful exit not done in 1.5 seconds, exiting at once"); _exit(0); });
#else   // Android: no GCD; the fallback is a thread of its own
    { void *dm_term_fallback(void *); pthread_t th; if (!pthread_create(&th, NULL, dm_term_fallback, NULL)) pthread_detach(th); }
#endif
    hlecount_flush();                               // DM_HLECOUNT: _exit does not run atexit, write a copy here
    dm_rec_finish();                                // DM_REC: the held state and the buffered steps (hle_hid.c)
    dm_cheats_exit();                               // a clean exit: no safe mode for the cheats at the next launch (atexit does not run either)
    dm_audio_suspend(1);
    dm_gfx_suspend(1);
    LOG("graceful exit done (%.3f s)", (now_ns() - t0) / 1e9);
    _exit(0);
}
#ifdef __APPLE__
void dm_term_install(void) {
    signal(SIGTERM, SIG_IGN);                       // do not take the default action: hand over to the GCD signal source to wrap up on an ordinary thread
    static dispatch_source_t src;
    src = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
    dispatch_source_set_event_handler(src, ^{ dm_host_shutdown("SIGTERM"); });
    dispatch_resume(src);
}
#else
// Android: no GCD signal source. The SIGTERM handler writes one byte to a pipe (async-signal-safe), and a thread waiting on the pipe wraps up as the GCD handler does
void *dm_term_fallback(void *unused) { (void)unused; usleep(1500 * 1000); LOG("[RED] graceful exit not done in 1.5 seconds, exiting at once"); _exit(0); }
static int g_term_pipe[2] = {-1, -1};
static void on_term(int sig) { (void)sig; char c = 1; if (write(g_term_pipe[1], &c, 1) < 0) {} }
static void *term_thread(void *unused) {
    (void)unused; char c;
    while (read(g_term_pipe[0], &c, 1) < 0 && errno == EINTR) {}
    dm_host_shutdown("SIGTERM");
    return NULL;
}
void dm_term_install(void) {
    if (pipe(g_term_pipe)) { LOG("[RED] SIGTERM pipe: errno %d, SIGTERM keeps its default action", errno); return; }
    pthread_t th; if (pthread_create(&th, NULL, term_thread, NULL)) { LOG("[RED] SIGTERM thread could not start"); return; }
    pthread_detach(th);
    struct sigaction sa = {0}; sa.sa_handler = on_term; sa.sa_flags = SA_RESTART | SA_ONSTACK; sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, NULL);
}
#endif
