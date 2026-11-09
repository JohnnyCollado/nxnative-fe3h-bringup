// The Android host's front end (dm/host/front.h) for the command-line dmhost: no window yet. The guest runs on the calling thread, as the Mac host does without a
// window; the system keyboard answers as cancelled (hle_swkbd.c: "no window, treated as cancel"; DM_KBD answers it instead). The launcher app's surface comes later.
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "front.h"

void front_run(void (*guest)(void)) { guest(); }
void *front_metal_layer(void) { return NULL; }
int front_screen_max_fps(void) { return 60; }
int front_screen_variable_refresh(void) { return 0; }
void front_screen_match(int fps) { (void)fps; }
void front_progress(const char *title, long done, long total) { (void)title; (void)done; (void)total; }
void (*front_progress_hook)(const char *title, long done, long total);
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    (void)title; (void)sub; (void)guide; (void)initial; (void)max_chars; (void)out; (void)out_cap; return -1; }

// Apple's per-thread performance counters (cycles and instructions, a private call): none on Android, so the CPU detail line reports no counts
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }
