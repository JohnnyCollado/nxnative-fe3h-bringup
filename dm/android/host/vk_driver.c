// Which Vulkan driver the host runs on: the system's, or a custom one the player installed (a Mesa Turnip build, or a repackaged Qualcomm driver), loaded through
// libadrenotools (dm/third_party/libadrenotools). The host calls Vulkan through volk (dm/third_party/volk): every vkXxx is a function pointer, filled in from the
// driver's vkGetInstanceProcAddr here, and from the instance and the device once they exist (nvn_vk_mem.c).
// Environment (set by the app, GameActivity.startHost):
//   DM_GPU_DRIVER  the custom driver's library file (in the app's internal storage: adrenotools refuses shared storage); unset or empty: the system driver
//   DM_HOOK_DIR    the folder holding adrenotools' hook libraries (libhook_impl.so, libmain_hook.so, ...): the app's native library folder, where the host itself lives
// A custom driver that does not load is reported and the system driver is used instead, so a bad driver package never leaves the game without a GPU.
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "volk.h"
#include <adrenotools/driver.h>

#define VLOG(...) do { flockfile(stderr); fprintf(stderr, "[dm] [vk] " __VA_ARGS__); fputc('\n', stderr); funlockfile(stderr); } while (0)

static char g_driver_desc[300] = "the system driver";
const char *vk_driver_desc(void) { return g_driver_desc; }

static void *open_custom(const char *path, const char *hooks) {
    const char *slash = strrchr(path, '/');
    if (!slash || !slash[1]) { VLOG("[RED] DM_GPU_DRIVER=%s is not a library file", path); return NULL; }
    char dir[PATH_MAX]; size_t n = (size_t)(slash - path) + 1;   // adrenotools joins the folder and the name as they are: the folder keeps its trailing '/'
    if (n >= sizeof dir) return NULL;
    memcpy(dir, path, n); dir[n] = 0;
    void *h = adrenotools_open_libvulkan(RTLD_NOW | RTLD_LOCAL, ADRENOTOOLS_DRIVER_CUSTOM, NULL, hooks, dir, slash + 1, NULL, NULL);
    if (!h) VLOG("[RED] the custom GPU driver %s did not load (adrenotools, hooks in %s): the system driver is used instead", path, hooks);
    return h;
}

int vk_driver_load(void) {
    const char *drv = getenv("DM_GPU_DRIVER"), *hooks = getenv("DM_HOOK_DIR");
    void *h = NULL;
    if (drv && *drv) {
        if (!hooks || !*hooks) VLOG("[RED] DM_GPU_DRIVER is set but DM_HOOK_DIR (adrenotools' hook libraries) is not: the system driver is used instead");
        else if ((h = open_custom(drv, hooks))) snprintf(g_driver_desc, sizeof g_driver_desc, "the custom driver %s", drv);
    }
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { VLOG("[RED] libvulkan.so did not load: %s", dlerror()); return 0; }
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(h, "vkGetInstanceProcAddr");
    if (!gipa) { VLOG("[RED] the Vulkan library has no vkGetInstanceProcAddr"); return 0; }
    volkInitializeCustom(gipa);
    VLOG("Vulkan through %s", g_driver_desc);
    return 1;
}
