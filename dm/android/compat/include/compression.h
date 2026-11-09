// Android stand-in for Apple's <compression.h>: only the GPU capture writer (nvn_capture.c, a diagnostic) compresses. Encoding and decoding report 0 bytes,
// which its callers treat as "could not compress": captures are written uncompressed or skipped on Android.
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef enum { COMPRESSION_LZ4 = 0x100, COMPRESSION_ZLIB = 0x205, COMPRESSION_LZMA = 0x306, COMPRESSION_LZ4_RAW = 0x101, COMPRESSION_LZFSE = 0x801 } compression_algorithm;
static inline size_t compression_encode_buffer(uint8_t *dst, size_t dst_size, const uint8_t *src, size_t src_size, void *scratch, compression_algorithm a) { (void)dst; (void)dst_size; (void)src; (void)src_size; (void)scratch; (void)a; return 0; }
static inline size_t compression_decode_buffer(uint8_t *dst, size_t dst_size, const uint8_t *src, size_t src_size, void *scratch, compression_algorithm a) { (void)dst; (void)dst_size; (void)src; (void)src_size; (void)scratch; (void)a; return 0; }
