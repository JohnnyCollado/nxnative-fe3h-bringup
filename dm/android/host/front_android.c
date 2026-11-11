// The Android host's front end (dm/host/front.h) for the command-line dmhost: the guest runs on the calling thread, the app's SurfaceView is the screen, and the
// keyboard is the app's text box with the system keyboard (vkd_front_text, nvn_vk_front.c, over the screen link). With no app (no DM_FRONT_SOCKET), text input
// answers as cancelled (hle_swkbd.c: "no window, treated as cancel"; DM_KBD answers it instead).
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "front.h"

int vkd_front_text(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, int min_chars, char *out, int out_cap);   // nvn_vk_front.c
extern int g_swkbd_min_chars;   // hle_swkbd.c: this request's minimum length (-1: none), set just before the input call

void front_run(void (*guest)(void)) { guest(); }
void *front_metal_layer(void) { return NULL; }
int front_screen_max_fps(void) { return 60; }
int front_screen_variable_refresh(void) { return 0; }
void front_screen_match(int fps) { (void)fps; }
void front_progress(const char *title, long done, long total) { (void)title; (void)done; (void)total; }
void (*front_progress_hook)(const char *title, long done, long total);
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    return vkd_front_text(title, sub, guide, initial, max_chars, g_swkbd_min_chars, out, out_cap);
}

// Apple's per-thread performance counters (cycles and instructions, a private call): none on Android, so the CPU detail line reports no counts
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }
