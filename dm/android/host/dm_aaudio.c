// The Core Audio output unit calls hle_audio.c makes (dm/android/compat/include/AudioToolbox/AudioToolbox.h), on AAudio: one output unit is one AAudio stream, opened
// at AudioUnitInitialize with the stream format hle_audio.c set (48 kHz, 16-bit, the device's channels), low latency, whose data callback calls the render callback
// hle_audio.c set, as Core Audio's would. A stream the system disconnected (headphones, a route change) is opened again at the next start; hle_audio.c's watchdog
// notices a stream that stopped calling back and restarts it (Stop, Start), as it does for an AudioUnit that went quiet on iOS.
#include <aaudio/AAudio.h>
#include <stdlib.h>
#include <string.h>
#include "AudioToolbox/AudioToolbox.h"

struct dm_audio_unit { AAudioStream *s; AudioStreamBasicDescription fmt; AURenderCallbackStruct cb; int have_fmt, have_cb; };

static aaudio_data_callback_result_t data_cb(AAudioStream *s, void *user, void *data, int32_t frames) {
    (void)s;
    AudioUnit u = user;
    AudioBufferList bl = {1, {{u->fmt.mChannelsPerFrame, (UInt32)frames * u->fmt.mBytesPerFrame, data}}};
    AudioUnitRenderActionFlags f = 0; AudioTimeStamp ts; memset(&ts, 0, sizeof ts);
    if (u->cb.inputProc) u->cb.inputProc(u->cb.inputProcRefCon, &f, &ts, 0, (UInt32)frames, &bl);
    else memset(data, 0, (size_t)frames * u->fmt.mBytesPerFrame);
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}
static OSStatus open_stream(AudioUnit u) {
    if (u->s) { AAudioStream_close(u->s); u->s = NULL; }
    AAudioStreamBuilder *b = NULL; if (AAudio_createStreamBuilder(&b)) return kAudioUnitErr_NoConnection;
    AAudioStreamBuilder_setDirection(b, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setSampleRate(b, (int32_t)u->fmt.mSampleRate);
    AAudioStreamBuilder_setChannelCount(b, (int32_t)u->fmt.mChannelsPerFrame);
    AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
    AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setUsage(b, AAUDIO_USAGE_GAME); AAudioStreamBuilder_setContentType(b, AAUDIO_CONTENT_TYPE_MUSIC);
    AAudioStreamBuilder_setDataCallback(b, data_cb, u);
    aaudio_result_t r = AAudioStreamBuilder_openStream(b, &u->s);
    AAudioStreamBuilder_delete(b);
    if (r) { u->s = NULL; return (OSStatus)r; }
    return noErr;
}

AudioComponent AudioComponentFindNext(AudioComponent after, const AudioComponentDescription *d) { (void)d; return after ? NULL : (AudioComponent)1; }
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *out) { (void)c; *out = calloc(1, sizeof **out); return *out ? noErr : kAudioUnitErr_NoConnection; }
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u) { if (u) { if (u->s) AAudioStream_close(u->s); free(u); } return noErr; }
OSStatus AudioUnitInitialize(AudioUnit u) { return u && u->have_fmt ? open_stream(u) : kAudioUnitErr_NoConnection; }
OSStatus AudioUnitUninitialize(AudioUnit u) { if (u && u->s) { AAudioStream_close(u->s); u->s = NULL; } return noErr; }
OSStatus AudioUnitGetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope s, AudioUnitElement e, void *data, UInt32 *size) { (void)u; (void)id; (void)s; (void)e; (void)data; (void)size; return kAudioUnitErr_NoConnection; }
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope s, AudioUnitElement e, const void *data, UInt32 size) {
    (void)s; (void)e;
    if (id == kAudioUnitProperty_StreamFormat && size >= sizeof u->fmt) { memcpy(&u->fmt, data, sizeof u->fmt); u->have_fmt = 1; return noErr; }
    if (id == kAudioUnitProperty_SetRenderCallback && size >= sizeof u->cb) { memcpy(&u->cb, data, sizeof u->cb); u->have_cb = 1; return noErr; }
    return kAudioUnitErr_NoConnection;
}
OSStatus AudioUnitAddPropertyListener(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc p, void *ctx) { (void)u; (void)id; (void)p; (void)ctx; return noErr; }
OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc p, void *ctx) { (void)u; (void)id; (void)p; (void)ctx; return noErr; }
OSStatus AudioOutputUnitStart(AudioUnit u) {
    if (!u) return kAudioUnitErr_NoConnection;
    if (!u->s || AAudioStream_getState(u->s) == AAUDIO_STREAM_STATE_DISCONNECTED) { OSStatus r = open_stream(u); if (r) return r; }
    return AAudioStream_requestStart(u->s) ? kAudioUnitErr_NoConnection : noErr;
}
OSStatus AudioOutputUnitStop(AudioUnit u) { if (u && u->s) AAudioStream_requestStop(u->s); return noErr; }
