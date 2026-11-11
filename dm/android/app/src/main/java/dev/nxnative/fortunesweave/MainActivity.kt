package dev.nxnative.fortunesweave

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.content.res.ColorStateList
import android.graphics.drawable.GradientDrawable
import android.view.KeyEvent
import android.widget.FrameLayout
import android.widget.GridLayout
import android.widget.ImageView
import android.util.TypedValue
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.app.AlertDialog
import android.widget.TextView
import android.widget.Toast
import android.content.Intent

/**
 * The launcher's screen: whether the game data is installed, the install's progress, and where to push the parts. An install starts by itself when parts.sha256
 * has been pushed into the import folder. Play starts the game (GameActivity) once the data is installed. Settings shows the same settings as the pause menu (SettingsPanel).
 * Settings, Controls and GPU drivers open on the right (Panels.kt). GPU drivers imports a driver package (Drivers.kt) chosen in a file picker, chooses or deletes one; packages pushed with adb into <game folder>/import/drivers/ are installed when
 * this screen opens.
 */
class MainActivity : Activity() {
    companion object { const val PICK_DRIVER = 2 }
    private lateinit var status: TextView
    private lateinit var detail: TextView
    private lateinit var bar: ProgressBar
    private lateinit var install: Button
    private lateinit var pause: Button
    private lateinit var play: Button
    private lateinit var panel: SettingsPanel
    private lateinit var controls: ControlsPanel
    private lateinit var drivers: DriversPanel
    private lateinit var nav: MenuNav
    private val ui = Handler(Looper.getMainLooper())
    private val tick = object : Runnable { override fun run() { refresh(); ui.postDelayed(this, 500) } }
    private var checked: String? = null   // the installed identity line (romfs.id; the data is not read again after the install)
    private var checkedOnce = false
    private val settings by lazy { Settings(this) }

    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        fun dp(v: Int) = FeTheme.dp(this, v)
        fun size(t: TextView, v: Float) = t.setTextSize(TypedValue.COMPLEX_UNIT_DIP, v)
        status = TextView(this).apply { size(this, 18f); typeface = FeTheme.SERIF; setTextColor(FeTheme.INK) }
        detail = TextView(this).apply { size(this, 13f); typeface = FeTheme.SERIF; setTextColor(FeTheme.INK_SOFT) }
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000; progressTintList = ColorStateList.valueOf(Color.rgb(0x7a, 0x63, 0xc6)); progressBackgroundTintList = ColorStateList.valueOf(Color.rgb(0xcf, 0xc6, 0xad))
        }
        // the right side: one panel at a time, chosen on the left (Panels.kt)
        val right = FocusBox(this)
        val hint = TextView(this).apply { size(this, 12f); setTextColor(FeTheme.MUTED) }
        nav = MenuNav(this, settings, right, MenuNav.placeholder(this, "Choose Settings, Controls or GPU drivers"), hint, "D-pad move  ·  [A] select")
        panel = SettingsPanel(this, settings) { _, _ -> }.apply { onEditChange = { nav.update() } }
        val settingsPage = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(MenuNav.head(this@MainActivity, "SETTINGS")); addView(panel) }
        controls = ControlsPanel(this, settings).apply { onMappingChange = { nav.update() }; table.onEditChange = { nav.update() } }
        drivers = DriversPanel(this, settings, { pickDriver() }, { confirmDelete(it) })
        install = Button(this).apply { text = "Install game data"; setOnClickListener { startInstall() } }
        pause = Button(this).apply { text = "Pause install"; setOnClickListener { InstallService.stop(this@MainActivity) } }
        play = Button(this).apply { text = "Play"; isEnabled = false; setOnClickListener { startActivity(android.content.Intent(this@MainActivity, GameActivity::class.java)) } }
        fun section(label: String, page: () -> View, before: () -> Unit = {}) = Button(this).apply {
            text = label; setOnClickListener { v -> before(); nav.open(v, page(), !v.isInTouchMode) }
        }
        val settingsButton = section("Settings", { settingsPage }) { panel.refresh() }
        val controlsButton = section("Controls", { controls }) { controls.refresh() }
        val driversButton = section("GPU drivers", { drivers }) { drivers.refresh() }
        // Play | Settings / GPU drivers | Controls / Install (or Pause): Controls sits under Settings, next to GPU drivers
        val all = listOf(play, settingsButton, driversButton, controlsButton, install, pause)
        for (b in all) { FeTheme.button(b); size(b, 16f) }

        val title = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            FeTheme.art(this@MainActivity, "ic_launcher", "mipmap")?.let { addView(ImageView(this@MainActivity).apply { setImageDrawable(it) }, LinearLayout.LayoutParams(dp(72), dp(72)).apply { marginEnd = dp(14) }) }
            addView(TextView(this@MainActivity).apply {
                text = "Fortune's Weave"; size(this, 22f); typeface = FeTheme.SERIF; setTextColor(FeTheme.BRIGHT); gravity = Gravity.CENTER_VERTICAL; maxLines = 1
                background = FeTheme.plaque(this@MainActivity, "dark"); FeTheme.plaquePadding(this)
            }, LinearLayout.LayoutParams(0, dp(56), 1f))
        }
        val statusBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_VERTICAL; minimumHeight = dp(76)
            background = FeTheme.plaque(this@MainActivity, "cream"); setPadding(dp(48), dp(10), dp(34), dp(10))
            addView(status); addView(detail); addView(bar, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(8)).apply { topMargin = dp(4) })
        }
        val grid = GridLayout(this).apply {
            columnCount = 2
            for (b in all) addView(b, GridLayout.LayoutParams(GridLayout.spec(GridLayout.UNDEFINED), GridLayout.spec(GridLayout.UNDEFINED, 1f)).apply { width = 0; height = dp(50); setMargins(dp(4), dp(4), dp(4), dp(4)) })
        }
        val left = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_VERTICAL
            addView(title)
            addView(statusBox, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(12) })
            addView(FocusBox(this@MainActivity).apply { addView(grid) }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(10) })
        }
        val columns = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(28), dp(16), dp(28), dp(26))
            addView(ScrollView(this@MainActivity).apply { isFillViewport = true; isVerticalScrollBarEnabled = false
                addView(LinearLayout(this@MainActivity).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_VERTICAL; addView(left) }) },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f))
            addView(right, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1.05f).apply { marginStart = dp(30) })
        }
        setContentView(FrameLayout(this).apply {
            background = GradientDrawable().apply {
                gradientType = GradientDrawable.RADIAL_GRADIENT; gradientRadius = resources.displayMetrics.widthPixels * 0.75f; setGradientCenter(0.3f, 0.4f)
                colors = intArrayOf(Color.rgb(0x2a, 0x23, 0x50), Color.rgb(0x16, 0x12, 0x2c), Color.rgb(0x0e, 0x0b, 0x1d))
            }
            addView(columns)
            addView(hint, FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.START).apply { setMargins(dp(32), 0, 0, dp(6)) })
        })
        window.decorView.viewTreeObserver.addOnGlobalFocusChangeListener { _, _ -> nav.update() }
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
    }

    // Controller (Panels.kt): the button sending the game's A acts on the focused item, the one sending B steps back out; in the left list it returns to Play
    override fun dispatchKeyEvent(e: KeyEvent): Boolean {
        if (FeTheme.isPick(settings, e.keyCode)) { if (e.action == KeyEvent.ACTION_UP) currentFocus?.performClick(); return true }
        if (FeTheme.isBack(settings, e.keyCode)) { if (e.action == KeyEvent.ACTION_UP && !nav.back()) (if (play.isEnabled) play else install).requestFocus(); return true }
        return super.dispatchKeyEvent(e)
    }

    override fun onResume() {
        super.onResume()
        checkedOnce = false
        Thread {   // file reads stay off the UI thread
            val inst = Installer(InstallService.gameDir(this))
            val have = inst.installed()
            val todo = inst.pending()
            val drivers = Drivers.importPushed(this)   // driver zips pushed with adb
            ui.post {
                checked = have; checkedOnce = true
                for (line in drivers) Toast.makeText(this, line, Toast.LENGTH_LONG).show()
                if (drivers.isNotEmpty()) { panel.refresh(); this.drivers.refresh() }
                if (todo && !InstallState.running) startInstall()
            }
        }.start()
        ui.post(tick)
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(tick)
    }

    private fun confirmDelete(d: Drivers.Driver) {
        AlertDialog.Builder(this).setTitle("Delete ${d.name} ${d.version}?".trim()).setMessage("The driver's files are removed from the app. A game set to use it uses the system driver.")
            .setPositiveButton("Delete") { _, _ ->
                val msg = try {
                    val s = Settings(this); val chosen = s[Settings.GPU_DRIVER] == d.folder.name
                    Drivers.remove(this, d)
                    if (chosen) s[Settings.GPU_DRIVER] = ""
                    "GPU driver deleted: ${d.name} ${d.version}" + if (chosen) ". The system driver is chosen now" else ""
                } catch (e: Exception) { "The driver was not deleted: ${e.message}" }
                Toast.makeText(this, msg, Toast.LENGTH_LONG).show(); panel.refresh(); drivers.refresh()
            }.setNegativeButton("Cancel", null).create().let { FeTheme.keys(it, settings).show() }
    }

    private fun pickDriver() {
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
            .putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"))
        startActivityForResult(i, PICK_DRIVER)
    }

    @Deprecated("Activity result API of the platform Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode != PICK_DRIVER || resultCode != RESULT_OK || uri == null) return
        Thread {
            val line = try {
                val d = contentResolver.openInputStream(uri)?.use { Drivers.install(this, it) } ?: throw IllegalStateException("the file could not be opened")
                Settings(this)[Settings.GPU_DRIVER] = d.folder.name   // a driver just imported is the one wanted
                "GPU driver installed and chosen: ${d.name} ${d.version}"
            } catch (e: Exception) { "The driver was not installed: ${e.message}" }
            ui.post { Toast.makeText(this, line, Toast.LENGTH_LONG).show(); panel.refresh(); drivers.refresh() }
        }.start()
    }

    private fun startInstall() {
        InstallState.progress = null
        InstallService.start(this)
    }

    private fun refresh() {
        val p = InstallState.progress
        val running = InstallState.running
        val importDir = Installer(InstallService.gameDir(this)).importDir.path
        when {
            running && p != null -> {
                status.text = when (p.phase) { Installer.Phase.WAITING -> "Waiting for game data"; Installer.Phase.VERIFYING -> "Checking game data"; else -> "Installing game data" }
                detail.text = p.text + if (p.total > 0) "\n${Installer.gb(p.done)} of ${Installer.gb(p.total)}" else ""
                bar.visibility = if (p.total > 0) View.VISIBLE else View.INVISIBLE
                if (p.total > 0) bar.progress = (p.done * 1000 / p.total).toInt()
            }
            running -> { status.text = "Starting the install"; detail.text = ""; bar.visibility = View.INVISIBLE }
            p != null && p.phase == Installer.Phase.DONE -> { status.text = "Game data installed"; detail.text = p.text; bar.visibility = View.INVISIBLE; checked = checked ?: "done" }
            p != null && (p.phase == Installer.Phase.FAILED || p.phase == Installer.Phase.CANCELLED) -> {
                status.text = if (p.phase == Installer.Phase.FAILED) "Install stopped" else "Install paused"
                detail.text = p.text + "\nParts go in: $importDir"
                bar.visibility = View.INVISIBLE
            }
            !checkedOnce -> { status.text = "Checking game data"; detail.text = ""; bar.visibility = View.INVISIBLE }
            checked != null -> { status.text = "Game data installed"; detail.text = checked; bar.visibility = View.INVISIBLE }
            else -> {
                status.text = "Game data not installed"
                detail.text = "From the PC, push the parts and parts.sha256 into:\n$importDir\nThe install starts by itself and can run while the parts are still copying."
                bar.visibility = View.INVISIBLE
            }
        }
        install.visibility = if (!running && checked == null) View.VISIBLE else View.GONE
        pause.visibility = if (running) View.VISIBLE else View.GONE
        play.isEnabled = checked != null && !running
    }
}
