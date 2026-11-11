package dev.nxnative.fortunesweave

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputFilter
import android.view.Gravity
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
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.graphics.drawable.GradientDrawable
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The game. The host (libdmhost.so, a program the app ships as a native library: dm/android/host) runs as a process of its own, started here; it draws each frame into
 * hardware buffers it shares with libdmfront.so, which shows them on this activity's SurfaceView (dm/android/host/dm_front_proto.h). The controller goes the other way,
 * and so does the game's keyboard (nn::swkbd): a text box with the system keyboard opens over the game, and the answer goes back the same link.
 * The game's files live in the app's folder ([InstallService.gameDir]): romfs.img (the launcher's install), patched/, spirv/ and font/ (the PC prep), run/ (saves, the log,
 * the pipeline cache) and mods/<name>/romfs (files that replace the game's own).
 * Buttons by position, as on a Switch: the right face button is A, the bottom one B (Android names them the other way round, after the Xbox layout).
 *
 * The pause menu: Back, or − and + held together for a second, opens it over the game, and leaving the app does too. While it is open the game is in the background
 * (DMF_PAUSE: focus out, sound off, the game's clock and graphics stopped) and the controller drives the menu: the d-pad or stick moves, A (right) picks, B (bottom)
 * or Back closes it. It holds the settings ([SettingsPanel]), Restart game (for the settings that apply at start) and Quit to the launcher.
 */
class GameActivity : Activity(), SurfaceHolder.Callback {
    companion object {
        init { System.loadLibrary("dmfront") }
        @JvmStatic external fun nativeSurface(surface: Surface?)
        @JvmStatic external fun nativeInput(buttons: Long, lx: Int, ly: Int, rx: Int, ry: Int)
        @JvmStatic external fun nativePause(on: Boolean)
        @JvmStatic external fun nativeOpmode(mode: Int)
        @JvmStatic external fun nativeFrames(): Long
        @JvmStatic external fun nativeTextDone(ok: Boolean, text: String)

        // nn::hid NpadButton bits (dm/host/dm_input.h)
        const val A = 1L shl 0; const val B = 1L shl 1; const val X = 1L shl 2; const val Y = 1L shl 3
        const val LS = 1L shl 4; const val RS = 1L shl 5; const val L = 1L shl 6; const val R = 1L shl 7
        const val ZL = 1L shl 8; const val ZR = 1L shl 9; const val PLUS = 1L shl 10; const val MINUS = 1L shl 11
        const val LEFT = 1L shl 12; const val UP = 1L shl 13; const val RIGHT = 1L shl 14; const val DOWN = 1L shl 15
        const val MENU_HOLD_MS = 1000L
    }
    private external fun nativeStart(socketName: String): Boolean

    // ---------- The game's keyboard ----------
    // nn::swkbd asks for text (dmfront.c's server thread calls onTextInput): the system keyboard types into a box shown over the game, and nativeTextDone carries the
    // answer back to the waiting game thread. The game truncates to its own limit again (UTF-16 units), so the box's filter only keeps the typing within it.
    private fun onTextInput(title: String, sub: String, guide: String, initial: String, maxChars: Int, minChars: Int) {
        ui.post { textDialog(title, sub, guide, initial, maxChars, minChars) }
    }

    private fun textDialog(title: String, sub: String, guide: String, initial: String, maxChars: Int, minChars: Int) {
        if (isFinishing) { nativeTextDone(false, ""); return }
        var answered = false
        fun answer(ok: Boolean, text: String) { if (!answered) { answered = true; nativeTextDone(ok, text) } }
        lateinit var dlg: AlertDialog
        lateinit var accept: () -> Unit
        val edit = EditText(this).apply {
            setText(initial)
            if (initial.isNotEmpty()) setSelection(initial.length)
            if (maxChars > 0) filters = arrayOf(InputFilter.LengthFilter(maxChars))
            imeOptions = EditorInfo.IME_ACTION_DONE
            setOnEditorActionListener { _, action, _ -> if (action == EditorInfo.IME_ACTION_DONE) { accept(); true } else false }
            minimumWidth = dp(300)
        }
        val hint = guide.ifEmpty { sub }
        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(8), dp(24), 0)
            if (hint.isNotEmpty()) addView(TextView(this@GameActivity).apply { text = hint; setTextColor(Color.WHITE); textSize = 14f })
            addView(edit)
        }
        dlg = AlertDialog.Builder(this)
            .setTitle(title.ifEmpty { "Enter text" })
            .setView(body)
            .setPositiveButton("OK", null)
            .setNegativeButton("Cancel", null)
            .setOnDismissListener { answer(false, "") }   // an answered dialog is already dismissed: the guard swallows this
            .create()
        accept = {
            val s = edit.text.toString()
            if (s.length >= minChars) { answer(true, s); dlg.dismiss() } else edit.error = "At least $minChars characters"
        }
        dlg.setOnShowListener {
            dlg.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener { accept() }
            dlg.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener { answer(false, ""); dlg.dismiss() }
            edit.requestFocus()
        }
        dlg.show()
    }
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private var host: Process? = null
    private lateinit var socket: String
    private lateinit var settings: Settings
    private var keys = 0L        // buttons held as keys
    private var axisButtons = 0L // triggers and the d-pad hat, from motion events
    private var lx = 0; private var ly = 0; private var rx = 0; private var ry = 0
    private val ui = Handler(Looper.getMainLooper())

    private lateinit var menu: View
    private lateinit var resume: Button
    private lateinit var restart: Button
    private lateinit var panel: SettingsPanel
    private lateinit var controls: ControlsPanel
    private lateinit var nav: MenuNav
    private lateinit var fps: TextView
    private var fpsFrames = 0L; private var fpsAt = 0L
    // the frame-rate counter: the host's finished frames over the last half second, while the setting is on and the game runs
    private val fpsTick = object : Runnable {
        override fun run() {
            val n = nativeFrames(); val t = System.nanoTime()
            if (fpsAt != 0L && t > fpsAt) fps.text = "%.0f FPS".format((n - fpsFrames) * 1e9 / (t - fpsAt))
            fpsFrames = n; fpsAt = t
            ui.postDelayed(this, 500)
        }
    }
    private var menuOpen = false
    private var restartArmed = false   // Restart asks for a second press: unsaved progress is lost
    private val holdOpen = Runnable { if (keys and (PLUS or MINUS) == (PLUS or MINUS)) openMenu() }

    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        settings = Settings(this)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val view = SurfaceView(this)
        view.holder.addCallback(this)
        menu = buildMenu()
        fps = TextView(this).apply { textSize = 14f; setTextColor(Color.WHITE); setBackgroundColor(Color.argb(0x80, 0, 0, 0)); setPadding(12, 4, 12, 4); typeface = Typeface.MONOSPACE }
        setContentView(FrameLayout(this).apply { addView(view); addView(fps, FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.START)); addView(menu) })
        showFps()
        window.insetsController?.let { it.hide(WindowInsets.Type.systemBars()); it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE }
        socket = "dmfront-" + android.os.Process.myPid()
        if (!nativeStart(socket)) { fail("The game's screen could not start"); return }
        startHost()
    }

    // ---------- The pause menu ----------
    private fun buildMenu(): View {
        fun dp(v: Int) = FeTheme.dp(this, v)
        fun button(label: String, onClick: (View) -> Unit) = Button(this).apply {
            text = label; FeTheme.button(this); setTextSize(android.util.TypedValue.COMPLEX_UNIT_DIP, 16f); setOnClickListener { onClick(it) }
        }
        val right = FocusBox(this)
        val hint = TextView(this).apply { setTextSize(android.util.TypedValue.COMPLEX_UNIT_DIP, 12f); setTextColor(FeTheme.MUTED) }
        nav = MenuNav(this, settings, right, MenuNav.placeholder(this, "Choose Settings or Controls"), hint, "D-pad move  ·  [A] select  ·  [B] resume")
        panel = SettingsPanel(this, settings) { c, v -> if (c == Settings.CONSOLE_MODE) nativeOpmode(v.toInt()); if (c == Settings.FPS_COUNTER) showFps() }.apply { onEditChange = { nav.update() } }
        val settingsPage = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(MenuNav.head(this@GameActivity, "SETTINGS")); addView(panel) }
        controls = ControlsPanel(this, settings).apply { onMappingChange = { nav.update() }; table.onEditChange = { nav.update() } }
        resume = button("Resume") { closeMenu() }
        restart = button("Restart game") { if (restartArmed) restartHost() else { restartArmed = true; restart.text = "Press again to restart" } }
        val buttons = listOf(
            resume,
            button("Settings") { v -> panel.refresh(); nav.open(v, settingsPage, !v.isInTouchMode) },
            button("Controls") { v -> controls.refresh(); nav.open(v, controls, !v.isInTouchMode) },
            restart,
            button("Quit to launcher") { finish() })
        val left = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(TextView(this@GameActivity).apply {
                text = if (FeTheme.game(this@GameActivity)) "PAUSED" else "Paused"; setTextSize(android.util.TypedValue.COMPLEX_UNIT_DIP, 26f); letterSpacing = if (FeTheme.game(this@GameActivity)) 0.3f else 0.08f; typeface = FeTheme.SERIF; setTextColor(if (FeTheme.game(this@GameActivity)) FeTheme.GOLD else FeTheme.BRIGHT); setPadding(0, 0, 0, dp(12))
                FeTheme.art(this@GameActivity, "fe_dia_small")?.let { d -> d.setBounds(0, 0, dp(20), dp(20)); setCompoundDrawables(d, null, null, null); compoundDrawablePadding = dp(12) }
            })
            addView(FocusBox(this@GameActivity).apply { addView(LinearLayout(this@GameActivity).apply {
                orientation = LinearLayout.VERTICAL
                for (b in buttons) addView(b, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(50)).apply { bottomMargin = dp(8) })
            }) })
            addView(MenuNav.note(this@GameActivity, "Hold − and + for a second to open this menu. Unsaved progress is lost on restart or quit."))
        }
        val columns = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(32), dp(18), dp(32), dp(26))
            addView(ScrollView(this@GameActivity).apply { isFillViewport = true; isVerticalScrollBarEnabled = false
                addView(LinearLayout(this@GameActivity).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_VERTICAL; addView(left) }) },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 0.8f))
            addView(right, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1.2f).apply { marginStart = dp(34) })
        }
        return FrameLayout(this).apply {
            // nearly opaque: the game is paused behind it and should not show through the menu
            background = if (FeTheme.game(this@GameActivity)) FeTheme.backdrop(this@GameActivity, Color.argb(0xF0, 0x02, 0x09, 0x0b)) else GradientDrawable().apply {
                gradientType = GradientDrawable.RADIAL_GRADIENT; gradientRadius = resources.displayMetrics.widthPixels * 0.75f; setGradientCenter(0.3f, 0.4f)
                colors = intArrayOf(Color.argb(0xF2, 0x2a, 0x23, 0x50), Color.argb(0xF2, 0x16, 0x12, 0x2c), Color.argb(0xF5, 0x0e, 0x0b, 0x1d))
            }
            visibility = View.GONE
            isClickable = true   // touches on the paused game do not reach it
            addView(columns)
            addView(hint, FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.START).apply { setMargins(dp(34), 0, 0, dp(6)) })
            viewTreeObserver.addOnGlobalFocusChangeListener { _, _ -> nav.update() }
        }
    }

    private fun showFps() {
        ui.removeCallbacks(fpsTick)
        val on = settings[Settings.FPS_COUNTER] == "1" && !menuOpen
        fps.visibility = if (on) View.VISIBLE else View.GONE
        if (on) { fps.text = "-- FPS"; fpsAt = 0L; ui.post(fpsTick) }
    }

    private fun openMenu() {
        if (menuOpen) return
        menuOpen = true; ui.removeCallbacks(holdOpen)
        keys = 0; axisButtons = 0; lx = 0; ly = 0; rx = 0; ry = 0; send()
        nativePause(true)
        restartArmed = false; restart.text = "Restart game"; panel.refresh(); controls.refresh(); nav.clear()
        menu.visibility = View.VISIBLE
        showFps()
        resume.requestFocus()
    }

    private fun closeMenu() {
        if (!menuOpen) return
        menuOpen = false
        menu.visibility = View.GONE
        nativePause(false)
        showFps()
    }

    // ---------- The host ----------
    private fun startHost() {
        val dir = InstallService.gameDir(this)
        val run = File(dir, "run"); File(run, "save").mkdirs(); File(run, "temp").mkdirs(); File(dir, "mods").mkdirs()
        val missing = listOf("romfs.img", "patched", "spirv", "font").filter { !File(dir, it).exists() }
        if (missing.isNotEmpty()) { fail("Missing in ${dir.path}: ${missing.joinToString()}"); return }
        // the logs of the three runs before stay beside the new one (log.1.txt the latest): a failed start (a driver that did not load) can still be read after the next
        for (i in 2 downTo 0) { val from = File(run, if (i == 0) "log.txt" else "log.$i.txt"); if (from.exists()) from.renameTo(File(run, "log.${i + 1}.txt")) }
        val exe = File(applicationInfo.nativeLibraryDir, "libdmhost.so")
        val pb = ProcessBuilder(exe.path, File(dir, "patched").path).directory(dir).redirectErrorStream(true).redirectOutput(File(run, "log.txt"))
        val env = pb.environment()
        env["DM_ROOT"] = run.path; env["DM_SAVE"] = File(run, "save").path; env["DM_TEMP"] = File(run, "temp").path
        env["DM_FONT_DIR"] = File(dir, "font").path; env["DM_ROMFS"] = File(dir, "romfs.img").path; env["DM_VK_SPIRV"] = File(dir, "spirv").path
        env["DM_FRONT_SOCKET"] = socket
        env.putAll(settings.env())   // console mode, language, sound (Settings.kt)
        // diagnostics: extra host environment from run/env.txt (KEY=VALUE lines, # comments), applied last so it can override. Lets DM_VK_* switches be set without a rebuild
        val envFile = File(run, "env.txt")
        if (envFile.exists()) for (line in envFile.readLines()) {
            val t = line.trim()
            if (t.isNotEmpty() && !t.startsWith("#") && t.contains('=')) {
                val k = t.substringBefore('=').trim(); val v = t.substringAfter('=').trim()
                if (k.isNotEmpty()) env[k] = v
            }
        }
        if (File("/system/fonts/Roboto-Regular.ttf").exists()) env["DM_SHARED_FONT"] = "/system/fonts/Roboto-Regular.ttf"
        // mods: each mods/<name>/romfs, in name order, the first holding a file wins (dm/host/hle_fs.c rom_overlay_path)
        val mods = File(dir, "mods").listFiles()?.filter { File(it, "romfs").isDirectory }?.sortedBy { it.name }?.map { File(it, "romfs").path } ?: emptyList()
        if (mods.isNotEmpty()) env["DM_ROMFS_OVERLAY"] = mods.joinToString(":")
        val p = try { pb.start() } catch (e: Exception) { fail("The game could not start: ${e.message}"); return }
        host = p
        Thread {
            val code = p.waitFor()
            runOnUiThread { if (!isFinishing && host === p) fail("The game stopped (exit $code); the log is in ${File(run, "log.txt").path}") }
        }.start()
    }

    /** Restart game: the host stops and a new one starts with the current settings; the screen takes the new host's buffers (dmfront.c g_gen). */
    private fun restartHost() {
        val old = host; host = null
        closeMenuQuietly()
        Thread {
            old?.destroy(); old?.waitFor()
            runOnUiThread { if (!isFinishing) startHost() }
        }.start()
    }
    private fun closeMenuQuietly() { menuOpen = false; menu.visibility = View.GONE; showFps() }   // the old host is going away: nothing to resume

    private fun fail(why: String) { Toast.makeText(this, why, Toast.LENGTH_LONG).show(); finish() }

    override fun onPause() { super.onPause(); if (host != null) openMenu() }   // leaving the app pauses the game; it waits in the menu on return
    override fun onDestroy() { ui.removeCallbacks(fpsTick); val p = host; host = null; p?.destroy(); nativeSurface(null); super.onDestroy() }

    override fun surfaceCreated(holder: SurfaceHolder) = nativeSurface(holder.surface)
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) = nativeSurface(holder.surface)
    override fun surfaceDestroyed(holder: SurfaceHolder) = nativeSurface(null)

    // ---------- Controller ----------
    private fun keyBit(code: Int): Long = when (code) {
        KeyEvent.KEYCODE_BUTTON_A, KeyEvent.KEYCODE_BUTTON_B, KeyEvent.KEYCODE_BUTTON_X, KeyEvent.KEYCODE_BUTTON_Y ->   // the button mapping settings (FeTheme.face)
            when (FeTheme.face(settings, code)) { FeTheme.Face.A -> A; FeTheme.Face.B -> B; FeTheme.Face.X -> X; FeTheme.Face.Y -> Y; null -> 0L }
        KeyEvent.KEYCODE_BUTTON_L1 -> L; KeyEvent.KEYCODE_BUTTON_R1 -> R; KeyEvent.KEYCODE_BUTTON_L2 -> ZL; KeyEvent.KEYCODE_BUTTON_R2 -> ZR
        KeyEvent.KEYCODE_BUTTON_THUMBL -> LS; KeyEvent.KEYCODE_BUTTON_THUMBR -> RS
        KeyEvent.KEYCODE_BUTTON_START -> PLUS; KeyEvent.KEYCODE_BUTTON_SELECT -> MINUS
        KeyEvent.KEYCODE_DPAD_LEFT -> LEFT; KeyEvent.KEYCODE_DPAD_UP -> UP; KeyEvent.KEYCODE_DPAD_RIGHT -> RIGHT; KeyEvent.KEYCODE_DPAD_DOWN -> DOWN
        else -> 0L
    }
    override fun dispatchKeyEvent(e: KeyEvent): Boolean {
        if (menuOpen) return menuKey(e)
        if (e.keyCode == KeyEvent.KEYCODE_BACK) { if (e.action == KeyEvent.ACTION_UP) openMenu(); return true }
        val bit = keyBit(e.keyCode)
        if (bit == 0L) return super.dispatchKeyEvent(e)
        keys = if (e.action == KeyEvent.ACTION_DOWN) keys or bit else if (e.action == KeyEvent.ACTION_UP) keys and bit.inv() else keys
        if (keys and (PLUS or MINUS) == (PLUS or MINUS)) { if (e.action == KeyEvent.ACTION_DOWN && e.repeatCount == 0) ui.postDelayed(holdOpen, MENU_HOLD_MS) } else ui.removeCallbacks(holdOpen)
        send(); return true
    }
    /** Keys while the menu is open (Panels.kt): the button sending the game's A acts on the focused item, the one sending B (or Back) steps back out, and from the
     *  left list resumes the game; the rest move focus. */
    private fun menuKey(e: KeyEvent): Boolean = when {
        FeTheme.isPick(settings, e.keyCode) -> { if (e.action == KeyEvent.ACTION_UP) currentFocus?.performClick(); true }
        FeTheme.isBack(settings, e.keyCode) || e.keyCode == KeyEvent.KEYCODE_BACK -> { if (e.action == KeyEvent.ACTION_UP && !nav.back()) closeMenu(); true }
        e.keyCode == KeyEvent.KEYCODE_BUTTON_START || e.keyCode == KeyEvent.KEYCODE_BUTTON_SELECT -> true   // the − / + still held from opening it
        else -> super.dispatchKeyEvent(e)
    }
    private fun stick(v: Float): Int = if (abs(v) < 0.08f) 0 else (v.coerceIn(-1f, 1f) * 32767).roundToInt()
    override fun dispatchGenericMotionEvent(e: MotionEvent): Boolean {
        if (menuOpen || e.source and InputDevice.SOURCE_JOYSTICK == 0 || e.action != MotionEvent.ACTION_MOVE) return super.dispatchGenericMotionEvent(e)
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
