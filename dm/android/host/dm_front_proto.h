// The link between the Android host (dmhost, a process the app starts) and the app's screen (libdmfront.so in the app): one local socket, SOCK_SEQPACKET, in the
// abstract namespace under the name in DM_FRONT_SOCKET. The host draws each presented frame into one of a few Android hardware buffers it allocated and handed to the
// app once; the app shows them on its SurfaceView and hands each back when the next one is on screen. Controller input goes the other way.
//   host → app: DMF_BUFFERS (count, width, height), then per buffer DMF_BUFFER (index) followed by the buffer itself (AHardwareBuffer_sendHandleToUnixSocket)
//               DMF_FRAME (index): that buffer holds a finished frame
//               DMF_TEXT (dmf_text_msg): the game's keyboard (nn::swkbd) asks for text; the host waits for the answer (the game thread blocks on the spot)
//   app → host: DMF_RELEASE (index): the app no longer reads that buffer
//               DMF_INPUT: player 1's controller (NpadButton bits as dm_input.h, sticks -32767..32767, y up)
//               DMF_PAUSE (index 1 / 0): the app's pause menu opened / closed: the game goes to the background and back (dm_background, hle_oe.c)
//               DMF_OPMODE (index 0 handheld / 1 console): the console mode changes while the game runs (dm_set_opmode)
//               DMF_TEXT_DONE (dmf_text_msg): the user's answer (ok 1 and the text, or ok 0 = cancelled)
#pragma once
#include <stdint.h>

enum { DMF_BUFFERS = 1, DMF_BUFFER, DMF_FRAME, DMF_RELEASE, DMF_INPUT, DMF_PAUSE, DMF_OPMODE, DMF_TEXT, DMF_TEXT_DONE };
enum { DMF_MAX_BUFFERS = 4 };
typedef struct { uint32_t type, index, width, height; uint64_t buttons; int32_t lx, ly, rx, ry; } dmf_msg;
// hle_swkbd.c's display buffers: title 65 / subtitle 129 / hint 257 / initial 501 / text 1002 characters, UTF-8 at 3 bytes each + the terminating 0
enum { DMF_TEXT_TITLE = 65 * 3 + 1, DMF_TEXT_SUB = 129 * 3 + 1, DMF_TEXT_GUIDE = 257 * 3 + 1, DMF_TEXT_INITIAL = 501 * 3 + 1, DMF_TEXT_TEXT = 1002 * 3 + 1 };
typedef struct {
    uint32_t type;      // DMF_TEXT or DMF_TEXT_DONE
    uint32_t max_chars; // DMF_TEXT: the game's limit in UTF-16 units (0: none; the host truncates to it on the way back as well)
    int32_t min_chars;  // DMF_TEXT: the game's minimum length in UTF-16 units (-1: none)
    int32_t ok;         // DMF_TEXT_DONE: 1 accepted / 0 cancelled
    char title[DMF_TEXT_TITLE], sub[DMF_TEXT_SUB], guide[DMF_TEXT_GUIDE], initial[DMF_TEXT_INITIAL], text[DMF_TEXT_TEXT];
} dmf_text_msg;
