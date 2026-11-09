package dev.nxnative.fortunesweave

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The identity of a RomFS image (DMROMFS1, dm/tools/romfs_image.py), the same as dm/host/romfs_identity.h and romfs_image.py --identity compute:
 * the image's size and an FNV-1a 64 hash of the little-endian 64-bit words of its 32-byte header followed by its index (zero-padded to a multiple of 8 bytes).
 * Only size and index decide a match; the line's title, version and build only name the image. The file data is never read, so a 27 GB image checks in a moment.
 */
data class RomfsIdentity(val size: Long, val index: Long) {
    companion object {
        private const val MAGIC = "DMROMFS1"
        private const val FNV_OFFSET = -0x340d631b7bdddcdbL   // 0xcbf29ce484222325
        private const val FNV_PRIME = 0x100000001b3L

        /** Size and index of an identity line ("DMROMFSID1 title=... size=<bytes> index=<16 hex digits>"), or null when it is not one. */
        fun parse(line: String): RomfsIdentity? {
            val parts = line.trim().split(Regex("\\s+"))
            if (parts.firstOrNull() != "DMROMFSID1") return null
            val f = parts.drop(1).mapNotNull { p -> p.indexOf('=').takeIf { it > 0 }?.let { p.substring(0, it) to p.substring(it + 1) } }.toMap()
            val size = f["size"]?.toLongOrNull() ?: return null
            val index = f["index"]?.takeIf { it.length == 16 }?.toULongOrNull(16)?.toLong() ?: return null
            return RomfsIdentity(size, index)
        }

        /** The identity of a complete image, or null for a missing, cut-off or invalid one (the checks of romfs_image.py's identity). */
        fun of(img: File): RomfsIdentity? {
            if (!img.isFile) return null
            RandomAccessFile(img, "r").use { f ->
                val size = f.length()
                if (size < 32) return null
                val head = ByteArray(32); f.readFully(head)
                if (String(head, 0, 8, Charsets.US_ASCII) != MAGIC) return null
                val hb = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN)
                val ioff = hb.getLong(8); val isz = hb.getLong(16)
                if (ioff < 32 || ioff > size || isz != size - ioff) return null
                var h = mix(FNV_OFFSET, head, 32)
                f.seek(ioff)
                val buf = ByteArray(8 shl 20)
                var left = isz
                var carry = 0   // bytes at the start of buf kept from the last read (an incomplete 64-bit word)
                while (left > 0) {
                    val want = minOf(left, (buf.size - carry).toLong()).toInt()
                    f.readFully(buf, carry, want)
                    left -= want
                    var n = carry + want
                    if (left == 0L && n % 8 != 0) { val pad = 8 - n % 8; buf.fill(0, n, n + pad); n += pad }
                    val whole = n - n % 8
                    h = mix(h, buf, whole)
                    carry = n - whole
                    System.arraycopy(buf, whole, buf, 0, carry)
                }
                return RomfsIdentity(size, h)
            }
        }

        private fun mix(start: Long, b: ByteArray, len: Int): Long {
            val bb = ByteBuffer.wrap(b, 0, len).order(ByteOrder.LITTLE_ENDIAN)
            var h = start
            while (bb.remaining() >= 8) h = (h xor bb.long) * FNV_PRIME
            return h
        }
    }
}
