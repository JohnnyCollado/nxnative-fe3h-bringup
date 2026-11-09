// Android stand-in for the parts of CoreFoundation the host's C files use:
// - a pointer-keyed dictionary (hle_nvn.c's per-program key table: CFDictionaryCreateMutable with NULL callbacks, so keys compare by address and nothing is
//   retained), implemented in dm/android/compat/dm_compat.c as an open-addressing hash table;
// - the app bundle (main.c reads the build commit from it): there is no bundle, so CFBundleGetMainBundle returns NULL and main.c logs "not recorded".
#pragma once
#include <stdint.h>
#include <stddef.h>
typedef const void *CFTypeRef;
typedef const struct dm_cfstr *CFStringRef;
typedef struct dm_cfdict *CFMutableDictionaryRef;
typedef const struct dm_cfdict *CFDictionaryRef;
typedef const struct dm_cfbundle *CFBundleRef;
typedef const void *CFAllocatorRef;
typedef long CFIndex;
typedef unsigned long CFTypeID;
typedef uint32_t CFStringEncoding;
typedef unsigned char Boolean;
#define kCFStringEncodingUTF8 0x08000100
#define CFSTR(s) ((CFStringRef)(s))

CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex capacity, const void *key_callbacks, const void *value_callbacks);
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *key);
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *value);
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key);
CFIndex CFDictionaryGetCount(CFDictionaryRef d);
void CFRelease(CFTypeRef t);

static inline CFBundleRef CFBundleGetMainBundle(void) { return NULL; }
static inline CFTypeRef CFBundleGetValueForInfoDictionaryKey(CFBundleRef b, CFStringRef key) { (void)b; (void)key; return NULL; }
static inline CFTypeID CFGetTypeID(CFTypeRef t) { (void)t; return 0; }
static inline CFTypeID CFStringGetTypeID(void) { return 7; }
static inline Boolean CFStringGetCString(CFStringRef s, char *buf, CFIndex size, CFStringEncoding e) { (void)s; (void)buf; (void)size; (void)e; return 0; }
