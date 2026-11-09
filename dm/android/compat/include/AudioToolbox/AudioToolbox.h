// Android stand-in for the Core Audio output calls in hle_audio.c, implemented on AAudio (dm/android/host/dm_aaudio.c): an output unit is an AAudio stream of the
// stream format hle_audio.c sets, whose data callback calls hle_audio.c's render callback. Real output is on when DM_AUDIO=1 (the app's GameActivity sets it; a run from
// adb keeps the silent sink). Device-change listeners and the Mac's buffer-size properties have no AAudio counterpart: listeners are accepted and never called, and
// property reads report kAudioUnitErr_NoConnection (hle_audio.c au_buffer then leaves the buffer size alone).
#pragma once
#include <stdint.h>
typedef int32_t OSStatus;
typedef uint32_t UInt32;
typedef int32_t SInt32;
typedef double Float64;
typedef uint32_t AudioUnitPropertyID, AudioUnitScope, AudioUnitElement, AudioUnitRenderActionFlags, AudioDeviceID, AudioFormatID, AudioFormatFlags;
typedef struct dm_audio_unit *AudioUnit, *AudioComponentInstance;
typedef struct dm_audio_component *AudioComponent;
#define noErr 0
#define kAudioUnitErr_NoConnection (-10876)
#define kAudioObjectUnknown 0
enum { kAudioUnitType_Output = 1, kAudioUnitSubType_RemoteIO = 2, kAudioUnitSubType_DefaultOutput = 3, kAudioUnitManufacturer_Apple = 4 };
enum { kAudioFormatLinearPCM = 0x6c70636d, kAudioFormatMPEG4AAC = 0x61616320 };
enum { kLinearPCMFormatFlagIsFloat = 1, kLinearPCMFormatFlagIsSignedInteger = 4, kLinearPCMFormatFlagIsPacked = 8, kAudioFormatFlagIsNonInterleaved = 32,
       kAudioFormatFlagsNativeEndian = 0 };
enum { kAudioUnitScope_Global = 0, kAudioUnitScope_Input = 1, kAudioUnitScope_Output = 2 };
enum { kAudioUnitProperty_StreamFormat = 8, kAudioUnitProperty_SetRenderCallback = 23, kAudioOutputUnitProperty_CurrentDevice = 2000,
       kAudioDevicePropertyBufferFrameSize = 0x6673697a, kAudioDevicePropertyBufferFrameSizeRange = 0x66737a23, kAudioOutputUnitProperty_EnableIO = 2003 };
typedef struct { Float64 mSampleRate; AudioFormatID mFormatID; AudioFormatFlags mFormatFlags; UInt32 mBytesPerPacket, mFramesPerPacket, mBytesPerFrame, mChannelsPerFrame, mBitsPerChannel, mReserved; } AudioStreamBasicDescription;
typedef struct { UInt32 componentType, componentSubType, componentManufacturer, componentFlags, componentFlagsMask; } AudioComponentDescription;
typedef struct { UInt32 mNumberChannels, mDataByteSize; void *mData; } AudioBuffer;
typedef struct { UInt32 mNumberBuffers; AudioBuffer mBuffers[1]; } AudioBufferList;
typedef struct { Float64 mSampleTime; uint64_t mHostTime; Float64 mRateScalar; uint64_t mWordClockTime; uint8_t mSMPTETime[24]; UInt32 mFlags, mReserved; } AudioTimeStamp;
typedef struct { Float64 mMinimum, mMaximum; } AudioValueRange;
typedef OSStatus (*AURenderCallback)(void *ctx, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts, UInt32 bus, UInt32 frames, AudioBufferList *data);
typedef struct { AURenderCallback inputProc; void *inputProcRefCon; } AURenderCallbackStruct;
typedef void (*AudioUnitPropertyListenerProc)(void *ctx, AudioUnit unit, AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement el);
AudioComponent AudioComponentFindNext(AudioComponent after, const AudioComponentDescription *d);
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *out);
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u);
OSStatus AudioUnitInitialize(AudioUnit u);
OSStatus AudioUnitUninitialize(AudioUnit u);
OSStatus AudioUnitGetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope s, AudioUnitElement e, void *data, UInt32 *size);
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope s, AudioUnitElement e, const void *data, UInt32 size);
OSStatus AudioUnitAddPropertyListener(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc p, void *ctx);
OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc p, void *ctx);
OSStatus AudioOutputUnitStart(AudioUnit u);
OSStatus AudioOutputUnitStop(AudioUnit u);
