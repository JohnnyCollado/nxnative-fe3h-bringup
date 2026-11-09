package dev.nxnative.fortunesweave

import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.security.MessageDigest

/**
 * Installs the game's RomFS image from the parts scripts/split_romfs.py writes, pushed into [importDir] (adb push), into [dir]/romfs.img.
 *
 * Each part is appended to romfs.img.joining while its SHA-256 is computed in the same pass, so each byte is read once. A part that fails its check is cut back off
 * the image and deleted, and the installer waits for it to be pushed again. A part that passes is recorded in romfs.img.joined (the number of verified parts) and
 * deleted, so the device needs about one part of space beyond the image. Parts may still be arriving while it runs: a missing or still-growing part is waited for.
 * A stopped install (the app killed, the device restarted) resumes from the last verified part: the image is cut back to it. The finished image is checked against
 * the manifest's size and identity (romfs_identity.h: its index, not its 27 GB of data, which the parts' checks have covered), then renamed to romfs.img and its
 * identity line written to romfs.id. A start only reads romfs.id ([installed]): the install is one-time, and the data is not checked again.
 *
 * Plain JVM code (no Android API), so InstallerTest runs it on the build machine.
 */
class Installer(val dir: File, val importDir: File = File(dir, "import")) {
    val image = File(dir, "romfs.img")
    val idFile = File(dir, "romfs.id")
    private val work = File(dir, "romfs.img.joining")
    private val state = File(dir, "romfs.img.joined")
    private val saved = File(dir, "install.sha256")   // the manifest of the install under way, kept while import/ is emptied part by part
    private val incoming get() = File(importDir, "parts.sha256")

    enum class Phase { CHECKING, WAITING, COPYING, VERIFYING, DONE, FAILED, CANCELLED }

    class Progress(val phase: Phase, val text: String, val done: Long = 0, val total: Long = 0)

    /**
     * The identity line of the installed image, or null when there is none. The install is checked once, when it finishes (each part's SHA-256, then the image's
     * identity), and romfs.id is written only after that: a start reads romfs.id and compares the image's size from the file system, never the image's data.
     */
    fun installed(): String? {
        val line = idFile.takeIf { it.isFile }?.readText()?.trim() ?: return null
        val want = RomfsIdentity.parse(line) ?: return null
        return if (image.isFile && image.length() == want.size) line else null
    }

    /** Whether there is something to install: a pushed manifest for another image than the installed one, or an install under way. */
    fun pending(): Boolean {
        if (saved.isFile) return true
        val m = incoming.takeIf { it.isFile }?.let { runCatching { PartsManifest.parse(it.readText()) }.getOrNull() } ?: return false
        val have = installed()?.let { RomfsIdentity.parse(it) }
        return have == null || m.identity != have
    }

    /**
     * Runs the install to its end. [report] receives progress; [keepGoing] is asked while waiting and between chunks and stops the install (resumable) when it
     * returns false; [pause] waits between polls for a part (milliseconds).
     */
    fun install(report: (Progress) -> Unit, keepGoing: () -> Boolean, pause: (Long) -> Unit = { Thread.sleep(it) }): Progress {
        try {
            dir.mkdirs(); importDir.mkdirs()
            report(Progress(Phase.CHECKING, "Reading the parts list"))
            if (!pending()) {
                installed()?.let { return Progress(Phase.DONE, "Game data installed", image.length(), image.length()).also(report) }
                return Progress(Phase.WAITING, "No game data to install: push the parts and parts.sha256 into ${importDir.path}").also(report)
            }
            val m = manifest() ?: return Progress(Phase.WAITING, "No game data to install: push the parts and parts.sha256 into ${importDir.path}").also(report)
            var count = (state.takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull() ?: 0).coerceIn(0, m.parts.size)
            var off = m.parts.take(count).sumOf { it.size }
            if (!work.isFile || work.length() < off) { count = 0; off = 0; work.delete(); work.createNewFile(); writeState(0) }
            RandomAccessFile(work, "rw").use { out ->
                out.setLength(off)   // anything past the last verified part is from an interrupted copy
                var note: String? = null   // why a part is waited for again, shown while waiting (a bad copy)
                while (count < m.parts.size) {
                    val p = m.parts[count]
                    val src = File(importDir, p.name)
                    if (!waitFor(p, src, m, report, keepGoing, pause, note)) return cancelled(report)
                    note = null
                    val needed = p.size + (64L shl 20)
                    if (dir.usableSpace < needed) {
                        return fail(report, "Not enough free space for ${p.name}: ${gb(needed)} needed, ${gb(dir.usableSpace)} free")
                    }
                    when (copy(out, src, p, off, m, report, keepGoing)) {
                        null -> return cancelled(report)
                        false -> {   // a bad copy: cut it back off, delete it and wait for it again
                            out.setLength(off)
                            src.delete()
                            note = "${p.name} failed its check and was deleted: push it again"
                            continue
                        }
                        true -> {}
                    }
                    out.fd.sync()
                    count++; off += p.size
                    writeState(count)
                    src.delete()
                }
            }
            report(Progress(Phase.VERIFYING, "Checking the finished image", off, m.imageSize))
            if (work.length() != m.imageSize) return fail(report, "The joined image is ${work.length()} bytes, expected ${m.imageSize}", reset = true)
            val got = RomfsIdentity.of(work) ?: return fail(report, "The joined image is not a valid RomFS image", reset = true)
            val want = m.identity
            if (want != null && got != want) return fail(report, "The joined image is not the one parts.sha256 names: push the parts again", reset = true)
            image.delete()
            if (!work.renameTo(image)) return fail(report, "Could not rename ${work.name} to ${image.name}")
            idFile.writeText((m.identityLine ?: "DMROMFSID1 size=${got.size} index=%016x".format(got.index)) + "\n")
            state.delete(); saved.delete()
            File(importDir, "parts.sha256").delete(); File(importDir, "join_romfs.sh").delete()
            return Progress(Phase.DONE, "Game data installed (${gb(m.imageSize)})", m.imageSize, m.imageSize).also(report)
        } catch (e: Exception) {
            return fail(report, "Install stopped: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    /** The manifest to install from: the one of the install under way, unless a different one has been pushed (then that one, from the start). */
    private fun manifest(): PartsManifest? {
        val pushed = incoming.takeIf { it.isFile }?.readText()
        val current = saved.takeIf { it.isFile }?.readText()
        if (pushed != null && pushed != current) {
            val m = PartsManifest.parse(pushed)
            work.delete(); state.delete()
            if (image.isFile) { image.delete(); idFile.delete() }   // a new version replaces the old: its space is needed
            saved.writeText(pushed)
            return m
        }
        return current?.let { PartsManifest.parse(it) }
    }

    private fun waitFor(p: PartsManifest.Part, src: File, m: PartsManifest, report: (Progress) -> Unit, keepGoing: () -> Boolean, pause: (Long) -> Unit, note: String?): Boolean {
        var note = note
        var last = Long.MIN_VALUE   // not -1 (a missing part), so the first look always reports: a part not pushed yet was shown as "Reading the parts list"
        while (true) {
            if (!keepGoing()) return false
            val len = if (src.isFile) src.length() else -1L
            if (len == p.size) return true
            if (len > p.size) {
                src.delete()
                note = "${p.name} is larger than expected and was deleted: push it again"
                report(Progress(Phase.WAITING, note, p.offset, m.imageSize))
            } else if (len != last) {
                val text = when {
                    len < 0 -> note ?: "Waiting for ${p.name}"
                    else -> "Waiting for ${p.name} to finish copying (${gb(len)} of ${gb(p.size)})"
                }
                report(Progress(Phase.WAITING, text, p.offset, m.imageSize))
            }
            last = len
            pause(2000)
        }
    }

    /** Appends [src] at [off] while hashing it: true when its SHA-256 matches, false when not, null when stopped. */
    private fun copy(out: RandomAccessFile, src: File, p: PartsManifest.Part, off: Long, m: PartsManifest, report: (Progress) -> Unit, keepGoing: () -> Boolean): Boolean? {
        val md = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(8 shl 20)
        out.seek(off)
        var n = 0L
        var lastReport = 0L
        FileInputStream(src).use { inp ->
            while (true) {
                val r = inp.read(buf)
                if (r < 0) break
                md.update(buf, 0, r); out.write(buf, 0, r); n += r
                if (n - lastReport >= (64L shl 20)) {
                    lastReport = n
                    if (!keepGoing()) { out.setLength(off); return null }
                    report(Progress(Phase.COPYING, "Installing ${p.name}", off + n, m.imageSize))
                }
            }
        }
        if (n != p.size) return false
        return md.digest().joinToString("") { "%02x".format(it) } == p.sha256
    }

    private fun writeState(count: Int) {
        val tmp = File(dir, state.name + ".tmp")
        tmp.writeText("$count\n")
        if (!tmp.renameTo(state)) { state.delete(); tmp.renameTo(state) }
    }

    private fun cancelled(report: (Progress) -> Unit) = Progress(Phase.CANCELLED, "Install paused: it resumes from the last finished part").also(report)

    private fun fail(report: (Progress) -> Unit, text: String, reset: Boolean = false): Progress {
        if (reset) { work.delete(); state.delete(); saved.delete() }
        return Progress(Phase.FAILED, text).also(report)
    }

    companion object {
        fun gb(bytes: Long) = "%.2f GB".format(bytes / 1073741824.0)
    }
}
