package dev.nxnative.fortunesweave

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView

/**
 * The launcher's screen: whether the game data is installed, the install's progress, and where to push the parts. An install starts by itself when parts.sha256
 * has been pushed into the import folder. Play starts the game (GameActivity) once the data is installed.
 */
class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var detail: TextView
    private lateinit var bar: ProgressBar
    private lateinit var install: Button
    private lateinit var pause: Button
    private lateinit var play: Button
    private val ui = Handler(Looper.getMainLooper())
    private val tick = object : Runnable { override fun run() { refresh(); ui.postDelayed(this, 500) } }
    private var checked: String? = null   // the installed identity line (romfs.id; the data is not read again after the install)
    private var checkedOnce = false

    override fun onCreate(saved: Bundle?) {
        super.onCreate(saved)
        val pad = (24 * resources.displayMetrics.density).toInt()
        status = TextView(this).apply { textSize = 22f; typeface = Typeface.DEFAULT_BOLD; setTextColor(Color.WHITE) }
        detail = TextView(this).apply { textSize = 15f; setPadding(0, pad / 3, 0, pad / 2); setTextColor(Color.rgb(0xCC, 0xCC, 0xCC)) }
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 1000 }
        install = Button(this).apply { text = "Install game data"; setOnClickListener { startInstall() } }
        pause = Button(this).apply { text = "Pause install"; setOnClickListener { InstallService.stop(this@MainActivity) } }
        play = Button(this).apply { text = "Play"; isEnabled = false; setOnClickListener { startActivity(android.content.Intent(this@MainActivity, GameActivity::class.java)) } }
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; addView(install); addView(pause); addView(play) }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(TextView(this@MainActivity).apply { text = "Fire Emblem Fortune's Weave"; textSize = 28f; typeface = Typeface.DEFAULT_BOLD; setTextColor(Color.WHITE) })
            addView(status); addView(detail); addView(bar); addView(buttons)
        })
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
    }

    override fun onResume() {
        super.onResume()
        checkedOnce = false
        Thread {   // file reads stay off the UI thread
            val inst = Installer(InstallService.gameDir(this))
            val have = inst.installed()
            val todo = inst.pending()
            ui.post {
                checked = have; checkedOnce = true
                if (todo && !InstallState.running) startInstall()
            }
        }.start()
        ui.post(tick)
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(tick)
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
