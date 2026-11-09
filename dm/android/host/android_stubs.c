// What the Android host build leaves out (dm/android/host/build.py EXCLUDE) and what stands in for it:
// - H.264 movies (hle_movie_avc.c, VideoToolbox): no decoder yet. A movie decoder that gets H.264 logs one [RED] line and produces no pictures; Fortune's Weave's
//   movies are AV1 (dav1d, hle_movie_av1.c), so it does not need this. Android's MediaCodec is the planned decoder.
// - Unreal's allocator on Apple malloc zones (hle_uemalloc.c): only Unreal games use it, so the Android host reports that it owns nothing and installs nothing.
#include "dm.h"
#include "hle_movie_avc.h"

struct mavc { int red; };
mavc *mavc_new(void) { mavc *a = calloc(1, sizeof *a); if (!a) die("movie: H.264 decoder allocation failed"); return a; }
void mavc_free(mavc *a) { free(a); }
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts_us) {
    (void)p; (void)pts_us;
    if (n > 0 && !a->red++) LOG("[RED] movie: no H.264 decoder on Android yet: this movie shows no pictures");
    return n > 0 ? -1 : 0;
}
int mavc_take(mavc *a, int flush, uint8_t *nv12, int w, int h, int64_t *pts_us) { (void)a; (void)flush; (void)nv12; (void)w; (void)h; (void)pts_us; return 0; }
int mavc_queued(const mavc *a) { (void)a; return 0; }
void mavc_set_rgba(mavc *a) { (void)a; }
void mavc_drop(mavc *a) { (void)a; }

int dm_ue_heap_owns(uint64_t a, uint64_t n) { (void)a; (void)n; return 0; }
void dm_ue_malloc_install(void) {}
