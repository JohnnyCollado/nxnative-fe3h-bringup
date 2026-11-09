package dev.nxnative.fortunesweave

/**
 * parts.sha256, written by scripts/split_romfs.py: one line per part, in order ("<SHA-256>  <name> <size>"), then "image <SHA-256 of the whole image> <size>"
 * and "identity <the image's identity line>".
 */
class PartsManifest(val parts: List<Part>, val imageSize: Long, val identityLine: String?) {
    class Part(val name: String, val size: Long, val sha256: String, val offset: Long)

    val identity: RomfsIdentity? get() = identityLine?.let { RomfsIdentity.parse(it) }

    companion object {
        private val HEX64 = Regex("[0-9a-f]{64}")
        private val NAME = Regex("[A-Za-z0-9._-]+")

        /** Throws IllegalArgumentException, naming the line, for anything else. */
        fun parse(text: String): PartsManifest {
            val parts = mutableListOf<Part>()
            var imageSize = -1L
            var identity: String? = null
            var off = 0L
            for ((k, raw) in text.lines().withIndex()) {
                val line = raw.trim()
                if (line.isEmpty()) continue
                val t = line.split(Regex("\\s+"))
                when (t[0]) {
                    "image" -> imageSize = t.getOrNull(2)?.toLongOrNull() ?: bad(k, raw)
                    "identity" -> identity = line.removePrefix("identity").trim()
                    else -> {
                        if (t.size != 3 || !HEX64.matches(t[0]) || !NAME.matches(t[1])) bad(k, raw)
                        val size = t[2].toLongOrNull()?.takeIf { it > 0 } ?: bad(k, raw)
                        parts += Part(t[1], size, t[0], off)
                        off += size
                    }
                }
            }
            require(parts.isNotEmpty()) { "parts.sha256 lists no parts" }
            require(imageSize == off) { "parts.sha256: the parts add up to $off bytes, the image line says $imageSize" }
            if (identity != null) require(RomfsIdentity.parse(identity)?.size == imageSize) { "parts.sha256: the identity line does not name a $imageSize-byte image" }
            return PartsManifest(parts, imageSize, identity)
        }

        private fun bad(k: Int, line: String): Nothing = throw IllegalArgumentException("parts.sha256 line ${k + 1} is not a part, image or identity line: $line")
    }
}
