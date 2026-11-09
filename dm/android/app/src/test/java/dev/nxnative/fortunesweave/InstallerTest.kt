package dev.nxnative.fortunesweave

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.security.MessageDigest

class InstallerTest {
    private lateinit var root: File
    private lateinit var img: ByteArray

    // The image romfs_image.py's identity was computed on: header, data of D bytes, an index of I bytes (crossing the 8 MiB read chunk, not a multiple of 8)
    private val D = 3000001
    private val I = 9 * (1 shl 20) + 5
    private val PY_IDENTITY = "DMROMFSID1 title=04007EF01F8AE000 version=131072 build=- size=12437222 index=454701013345126b"   // romfs_image.identity(...)
    private val PART = 4 * (1 shl 20)

    @Before fun setUp() {
        root = Files.createTempDirectory("installer").toFile()
        val b = ByteBuffer.allocate(32 + D + I).order(ByteOrder.LITTLE_ENDIAN)
        b.put("DMROMFS1".toByteArray()); b.putLong(32L + D); b.putLong(I.toLong()); b.putLong(7)
        for (i in 0 until D) b.put(((i * 31 + 7) and 0xFF).toByte())
        for (i in 0 until I) b.put(((i * 17 + 3) and 0xFF).toByte())
        img = b.array()
    }

    @After fun tearDown() { root.deleteRecursively() }

    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    /** What split_romfs.py writes: the parts and parts.sha256, into [dir]. Returns the parts' bytes. */
    private fun split(dir: File, identity: String? = PY_IDENTITY): List<ByteArray> {
        dir.mkdirs()
        val parts = (img.indices step PART).map { img.copyOfRange(it, minOf(it + PART, img.size)) }
        val lines = parts.mapIndexed { k, p -> "${sha(p)}  romfs.img.%03d ${p.size}".format(k) }.toMutableList()
        lines += "image ${sha(img)} ${img.size}"
        if (identity != null) lines += "identity $identity"
        parts.forEachIndexed { k, p -> File(dir, "romfs.img.%03d".format(k)).writeBytes(p) }
        File(dir, "parts.sha256").writeText(lines.joinToString("\n") + "\n")
        return parts
    }

    private fun run(inst: Installer, keepGoing: () -> Boolean = { true }): Installer.Progress {
        val seen = mutableListOf<Installer.Progress>()
        return inst.install({ seen += it }, keepGoing, pause = {})
    }

    @Test fun identityMatchesPython() {
        val f = File(root, "id.img"); f.writeBytes(img)
        assertEquals(RomfsIdentity.parse(PY_IDENTITY), RomfsIdentity.of(f))
    }

    @Test fun identityRefusesCutOffImage() {
        val f = File(root, "cut.img"); f.writeBytes(img.copyOf(img.size - 1))
        assertNull(RomfsIdentity.of(f))
    }

    @Test fun installsAndDeletesParts() {
        val inst = Installer(File(root, "game"))
        assertEquals(3, split(inst.importDir).size)   // 12,437,222 bytes in 4 MiB parts
        assertTrue(inst.pending())
        val end = run(inst)
        assertEquals(end.text, Installer.Phase.DONE, end.phase)
        assertArrayEquals(img, inst.image.readBytes())
        assertEquals(PY_IDENTITY, inst.installed())
        assertEquals(emptyList<String>(), inst.importDir.list()!!.toList())
        assertFalse(inst.pending())
        assertEquals(Installer.Phase.DONE, run(inst).phase)   // a second run finds it installed
    }

    @Test fun startDoesNotRecheckTheData() {
        val inst = Installer(File(root, "game"))
        split(inst.importDir)
        assertEquals(Installer.Phase.DONE, run(inst).phase)
        java.io.RandomAccessFile(inst.image, "rw").use { it.seek(img.size - 100L); it.write(ByteArray(100)) }   // index bytes changed, size the same
        assertEquals(PY_IDENTITY, inst.installed())   // one-time install: a start reads romfs.id and the size only
        java.io.RandomAccessFile(inst.image, "rw").use { it.setLength(img.size - 1L) }
        assertNull(inst.installed())   // a missing or cut-off image still shows as not installed
    }

    @Test fun badPartIsDeletedAndWaitedFor() {
        val inst = Installer(File(root, "game"))
        val parts = split(inst.importDir)
        val bad = File(inst.importDir, "romfs.img.001")
        bad.writeBytes(parts[1].copyOf().also { it[100] = (it[100] + 1).toByte() })
        var polls = 0
        val seen = mutableListOf<Installer.Progress>()
        val end = inst.install({ seen += it }, { !(bad.exists().not() && ++polls > 3) }, pause = {})   // stop once the bad part is gone and waited for
        assertEquals(Installer.Phase.CANCELLED, end.phase)
        // the reason stays on screen while waiting, not replaced by a plain "Waiting for"
        assertEquals("romfs.img.001 failed its check and was deleted: push it again", seen.last { it.phase == Installer.Phase.WAITING }.text)
        assertFalse(bad.exists())
        assertFalse(File(inst.importDir, "romfs.img.000").exists())   // the good part before it went in
        bad.writeBytes(parts[1])   // pushed again
        val done = run(inst)
        assertEquals(done.text, Installer.Phase.DONE, done.phase)
        assertArrayEquals(img, inst.image.readBytes())
    }

    @Test fun missingPartIsWaitedForThenResumes() {
        val inst = Installer(File(root, "game"))
        val parts = split(inst.importDir)
        File(inst.importDir, "romfs.img.002").delete()
        var polls = 0
        val seen = mutableListOf<Installer.Progress>()
        assertEquals(Installer.Phase.CANCELLED, inst.install({ seen += it }, { ++polls < 50 }, pause = {}).phase)
        assertTrue(seen.joinToString { it.text }, seen.any { it.phase == Installer.Phase.WAITING && it.text == "Waiting for romfs.img.002" })
        File(inst.importDir, "romfs.img.002").writeBytes(parts[2])
        val done = run(inst)
        assertEquals(done.text, Installer.Phase.DONE, done.phase)
        assertArrayEquals(img, inst.image.readBytes())
    }

    @Test fun resumeCutsOffAnInterruptedCopy() {
        val inst = Installer(File(root, "game"))
        val parts = split(inst.importDir)
        File(inst.importDir, "romfs.img.001").delete()
        var polls = 0
        run(inst) { ++polls < 20 }   // part 0 verified, waiting for part 1
        File(inst.dir, "romfs.img.joining").appendBytes(ByteArray(12345) { 9 })   // a copy of part 1 cut off by a kill
        File(inst.importDir, "romfs.img.001").writeBytes(parts[1])
        val done = run(inst)
        assertEquals(done.text, Installer.Phase.DONE, done.phase)
        assertArrayEquals(img, inst.image.readBytes())
    }

    @Test fun wrongIdentityFails() {
        val inst = Installer(File(root, "game"))
        split(inst.importDir, PY_IDENTITY.replace("454701013345126b", "0000000000000001"))
        val end = run(inst)
        assertEquals(Installer.Phase.FAILED, end.phase)
        assertNull(inst.installed())
        assertFalse(inst.image.exists())
    }

    @Test fun manifestRejectsMismatchedTotal() {
        val bad = "${"a".repeat(64)}  romfs.img.000 10\nimage ${"b".repeat(64)} 11\n"
        val e = runCatching { PartsManifest.parse(bad) }.exceptionOrNull()
        assertNotNull(e); assertTrue(e!!.message!!.contains("add up"))
    }
}
