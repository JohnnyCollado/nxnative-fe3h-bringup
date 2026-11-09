package dev.nxnative.fortunesweave

import android.app.Activity
import android.os.Bundle
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.Toast
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The game. The host (libdmhost.so, a program the app ships as a native library: dm/android/host) runs as a process of its own, started here; it draws each frame into
 * hardware buffers it shares with libdmfront.so, which shows them on this activity's SurfaceView (dm/android/host/dm_front_proto.h). The controller goes the other way.
 * The game's files live in the app's folder ([InstallService.gameDir]): romfs.img (the launcher's install), patched/, spirv/ and font/ (the PC prep), run/ (saves, the log,
 * the pipeline cache) and mods/<name>/romfs (files that replace the game's own).
 * Buttons by position, as on a Switch: the right face button is A, the bottom one B (Android names them the other way round, after the Xbox layout).
 */
class GameActivity : Activity(), SurfaceHolder.Callback {
    companion object {
        init { System.loadLibrary("dmfront") }
        @JvmStatic external fun nativeStart(socketName: String): Boolean
        @JvmStatic external fun nativeSurface(surface: Surface?)
        @JvmStatic external fun nativeInput(buttons: Long, lx: Int, ly: Int, rx: Int, ry: Int)

        // nn::hid NpadButton bits (dm/host/dm_input.h)
        const val A = 1L shl 0; const val B = 1L shl 1; const val X = 1L shl 2; const val Y = 1L shl 3
        const val LS = 1L shl 4; const val RS = 1L shl 5; const val L = 1L shl 6; const val R = 1L shl 7
        const val ZL = 1L shl 8; const val ZR = 1L shl 9; const val PLUS = 1L shl 10; const val MINUS = 1L shl 11
        const val LEFT = 1L shl 12; const val UP = 1L shl 13; const val RIGHT = 1L shl 14; const val DOWN = 1L shl 15
    }

    private var host: Process? = null
    private var keys = 0L        // buttons held as keys
    private var axisButtons = 0L // triggers and the d-pad hat, from motion events
    private var lx = 0; private var ly = 0; private var rx = 0; private var ry = 0

    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val view = SurfaceView(this)
        setContentView(view)
        view.holder.addCallback(this)
        window.insetsController?.let { it.hide(WindowInsets.Type.systemBars()); it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE }
        val socket = "dmfront-" + android.os.Process.myPid()
        if (!nativeStart(socket)) { fail("The game's screen could not start"); return }
        startHost(socket)
    }

    private fun startHost(socket: String) {
        val dir = InstallService.gameDir(this)
        val run = File(dir, "run"); File(run, "save").mkdirs(); File(run, "temp").mkdirs(); File(dir, "mods").mkdirs()
        val missing = listOf("romfs.img", "patched", "spirv", "font").filter { !File(dir, it).exists() }
        if (missing.isNotEmpty()) { fail("Missing in ${dir.path}: ${missing.joinToString()}"); return }
        val exe = File(applicationInfo.nativeLibraryDir, "libdmhost.so")
        val pb = ProcessBuilder(exe.path, File(dir, "patched").path).directory(dir).redirectErrorStream(true).redirectOutput(File(run, "log.txt"))
        val env = pb.environment()
        env["DM_ROOT"] = run.path; env["DM_SAVE"] = File(run, "save").path; env["DM_TEMP"] = File(run, "temp").path
        env["DM_FONT_DIR"] = File(dir, "font").path; env["DM_ROMFS"] = File(dir, "romfs.img").path; env["DM_VK_SPIRV"] = File(dir, "spirv").path
        env["DM_FRONT_SOCKET"] = socket; env["DM_AUDIO"] = "1"   // real sound (AAudio, dm/android/host/dm_aaudio.c)
        if (File("/system/fonts/Roboto-Regular.ttf").exists()) env["DM_SHARED_FONT"] = "/system/fonts/Roboto-Regular.ttf"
        // mods: each mods/<name>/romfs, in name order, the first holding a file wins (dm/host/hle_fs.c rom_overlay_path)
        val mods = File(dir, "mods").listFiles()?.filter { File(it, "romfs").isDirectory }?.sortedBy { it.name }?.map { File(it, "romfs").path } ?: emptyList()
        if (mods.isNotEmpty()) env["DM_ROMFS_OVERLAY"] = mods.joinToString(":")
        host = try { pb.start() } catch (e: Exception) { fail("The game could not start: ${e.message}"); return }
        Thread {
            val code = host?.waitFor()
            runOnUiThread { if (!isFinishing) fail("The game stopped (exit $code); the log is in ${File(run, "log.txt").path}") }
        }.start()
    }

    private fun fail(why: String) { Toast.makeText(this, why, Toast.LENGTH_LONG).show(); finish() }

    override fun onDestroy() { host?.destroy(); host = null; nativeSurface(null); super.onDestroy() }

    override fun surfaceCreated(holder: SurfaceHolder) = nativeSurface(holder.surface)
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) = nativeSurface(holder.surface)
    override fun surfaceDestroyed(holder: SurfaceHolder) = nativeSurface(null)

    private fun keyBit(code: Int): Long = when (code) {
        KeyEvent.KEYCODE_BUTTON_B -> A; KeyEvent.KEYCODE_BUTTON_A -> B; KeyEvent.KEYCODE_BUTTON_Y -> X; KeyEvent.KEYCODE_BUTTON_X -> Y
        KeyEvent.KEYCODE_BUTTON_L1 -> L; KeyEvent.KEYCODE_BUTTON_R1 -> R; KeyEvent.KEYCODE_BUTTON_L2 -> ZL; KeyEvent.KEYCODE_BUTTON_R2 -> ZR
        KeyEvent.KEYCODE_BUTTON_THUMBL -> LS; KeyEvent.KEYCODE_BUTTON_THUMBR -> RS
        KeyEvent.KEYCODE_BUTTON_START -> PLUS; KeyEvent.KEYCODE_BUTTON_SELECT -> MINUS
        KeyEvent.KEYCODE_DPAD_LEFT -> LEFT; KeyEvent.KEYCODE_DPAD_UP -> UP; KeyEvent.KEYCODE_DPAD_RIGHT -> RIGHT; KeyEvent.KEYCODE_DPAD_DOWN -> DOWN
        else -> 0L
    }
    override fun dispatchKeyEvent(e: KeyEvent): Boolean {
        val bit = keyBit(e.keyCode)
        if (bit == 0L) return super.dispatchKeyEvent(e)   // Back and the rest: the system's
        keys = if (e.action == KeyEvent.ACTION_DOWN) keys or bit else if (e.action == KeyEvent.ACTION_UP) keys and bit.inv() else keys
        send(); return true
    }
    private fun stick(v: Float): Int = if (abs(v) < 0.08f) 0 else (v.coerceIn(-1f, 1f) * 32767).roundToInt()
    override fun dispatchGenericMotionEvent(e: MotionEvent): Boolean {
        if (e.source and InputDevice.SOURCE_JOYSTICK == 0 || e.action != MotionEvent.ACTION_MOVE) return super.dispatchGenericMotionEvent(e)
        lx = stick(e.getAxisValue(MotionEvent.AXIS_X)); ly = -stick(e.getAxisValue(MotionEvent.AXIS_Y))
        rx = stick(e.getAxisValue(MotionEvent.AXIS_Z)); ry = -stick(e.getAxisValue(MotionEvent.AXIS_RZ))
        var b = 0L
        if (maxOf(e.getAxisValue(MotionEvent.AXIS_LTRIGGER), e.getAxisValue(MotionEvent.AXIS_BRAKE)) > 0.5f) b = b or ZL
        if (maxOf(e.getAxisValue(MotionEvent.AXIS_RTRIGGER), e.getAxisValue(MotionEvent.AXIS_GAS)) > 0.5f) b = b or ZR
        val hx = e.getAxisValue(MotionEvent.AXIS_HAT_X); val hy = e.getAxisValue(MotionEvent.AXIS_HAT_Y)
        if (hx < -0.5f) b = b or LEFT; if (hx > 0.5f) b = b or RIGHT; if (hy < -0.5f) b = b or UP; if (hy > 0.5f) b = b or DOWN
        axisButtons = b
        send(); return true
    }
    private fun send() = nativeInput(keys or axisButtons, lx, ly, rx, ry)

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
    }
}
