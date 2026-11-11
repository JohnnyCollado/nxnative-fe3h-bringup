package dev.nxnative.fortunesweave

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

/**
 * Custom GPU drivers: packages in the format other Android emulators use (a zip holding the driver library and meta.json, whose "libraryName" names the library;
 * "name", "description", "author", "packageVersion", "driverVersion" and "minApi" describe it), installed into the app's internal storage: files/drivers/<folder>/.
 * The host loads the chosen one through libadrenotools (dm/android/host/vk_driver.c), which refuses drivers on shared storage.
 * Packages arrive two ways: the launcher's Import GPU driver button (a file picker), or zips pushed with adb into <game folder>/import/drivers/, imported when the
 * launcher opens and then deleted.
 */
object Drivers {
    class Driver(val folder: File, val name: String, val version: String, val library: File)

    fun root(context: Context) = File(context.filesDir, "drivers")
    fun pushDir(context: Context) = File(InstallService.gameDir(context), "import/drivers")

    /** The installed drivers, by name. */
    fun list(context: Context): List<Driver> =
        root(context).listFiles()?.filter { it.isDirectory }?.mapNotNull { read(it) }?.sortedBy { it.name.lowercase() } ?: emptyList()

    private fun read(folder: File): Driver? = try {
        val m = JSONObject(File(folder, "meta.json").readText())
        val lib = File(folder, m.getString("libraryName"))
        if (!lib.isFile) null else Driver(folder, m.optString("name", folder.name), m.optString("driverVersion", m.optString("packageVersion", "")), lib)
    } catch (e: Exception) { null }

    /** Installs a driver package from a zip stream; returns the driver, or throws with a reason a person can read. */
    fun install(context: Context, zip: InputStream): Driver {
        val tmp = File(root(context), ".incoming"); tmp.deleteRecursively(); tmp.mkdirs()
        try {
            ZipInputStream(zip.buffered()).use { z ->
                while (true) {
                    val e = z.nextEntry ?: break
                    if (e.isDirectory) continue
                    val name = e.name.substringAfterLast('/')   // the files of a package sit at its top level; any folders in the zip are dropped
                    if (name.isEmpty() || name.startsWith(".")) continue
                    File(tmp, name).outputStream().use { z.copyTo(it) }
                }
            }
            val metaFile = File(tmp, "meta.json")
            if (!metaFile.isFile) throw IllegalArgumentException("not a driver package: it has no meta.json")
            val m = JSONObject(metaFile.readText())
            val libName = m.optString("libraryName")
            if (libName.isEmpty() || !File(tmp, libName).isFile) throw IllegalArgumentException("meta.json names the library \"$libName\", which the package does not hold")
            val minApi = m.optInt("minApi", 0)
            if (minApi > android.os.Build.VERSION.SDK_INT) throw IllegalArgumentException("the driver needs Android API $minApi; this device has ${android.os.Build.VERSION.SDK_INT}")
            val label = m.optString("name", libName)
            val folder = File(root(context), (label + " " + m.optString("driverVersion", m.optString("packageVersion", ""))).trim().replace(Regex("[^A-Za-z0-9._-]+"), "_"))
            folder.deleteRecursively()
            if (!tmp.renameTo(folder)) throw IllegalStateException("could not move the driver into ${folder.path}")
            return read(folder) ?: throw IllegalStateException("the installed driver could not be read back")
        } finally { tmp.deleteRecursively() }
    }

    /** Deletes an installed driver's folder (only a folder directly inside [root]); throws with a reason a person can read. A setting that chose it falls back to the
     *  system driver (Settings.get). */
    fun remove(context: Context, d: Driver) {
        val root = root(context).canonicalFile; val f = d.folder.canonicalFile
        if (f.parentFile != root || f.name.startsWith(".")) throw IllegalArgumentException("${f.path} is not an installed driver's folder")
        if (!f.deleteRecursively()) throw IllegalStateException("could not delete ${f.path}")
    }

    /** Imports the zips pushed into [pushDir]; returns a line per package (installed, or why not). Runs off the UI thread. */
    fun importPushed(context: Context): List<String> {
        val zips = pushDir(context).listFiles()?.filter { it.isFile && it.name.endsWith(".zip", ignoreCase = true) } ?: return emptyList()
        return zips.map { f ->
            val line = try { val d = f.inputStream().use { install(context, it) }; "GPU driver installed: ${d.name} ${d.version}" } catch (e: Exception) { "${f.name}: ${e.message}" }
            f.delete(); line
        }
    }
}
