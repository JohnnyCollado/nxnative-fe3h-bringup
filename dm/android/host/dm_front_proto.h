// The link between the Android host (dmhost, a process the app starts) and the app's screen (libdmfront.so in the app): one local socket, SOCK_SEQPACKET, in the
// abstract namespace under the name in DM_FRONT_SOCKET. The host draws each presented frame into one of a few Android hardware buffers it allocated and handed to the
// app once; the app shows them on its SurfaceView and hands each back when the next one is on screen. Controller input goes the other way.
//   host → app: DMF_BUFFERS (count, width, height), then per buffer DMF_BUFFER (index) followed by the buffer itself (AHardwareBuffer_sendHandleToUnixSocket)
//               DMF_FRAME (index): that buffer holds a finished frame
//   app → host: DMF_RELEASE (index): the app no longer reads that buffer
//               DMF_INPUT: player 1's controller (NpadButton bits as dm_input.h, sticks -32767..32767, y up)
#pragma once
#include <stdint.h>

enum { DMF_BUFFERS = 1, DMF_BUFFER, DMF_FRAME, DMF_RELEASE, DMF_INPUT };
enum { DMF_MAX_BUFFERS = 4 };
typedef struct { uint32_t type, index, width, height; uint64_t buttons; int32_t lx, ly, rx, ry; } dmf_msg;
