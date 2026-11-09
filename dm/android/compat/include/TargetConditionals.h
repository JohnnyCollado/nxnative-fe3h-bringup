// Android stand-in for Apple's TargetConditionals.h (dm/android/compat: the Android host build only puts this folder on the include path).
// Every Apple target is off, so the engine's iOS / Mac branches drop out and its other branch is taken.
#pragma once
#define TARGET_OS_MAC 0
#define TARGET_OS_OSX 0
#define TARGET_OS_IPHONE 0
#define TARGET_OS_IOS 0
#define TARGET_OS_SIMULATOR 0
#define TARGET_OS_TV 0
#define TARGET_OS_WATCH 0
