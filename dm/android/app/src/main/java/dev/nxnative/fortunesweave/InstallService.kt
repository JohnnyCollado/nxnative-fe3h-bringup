package dev.nxnative.fortunesweave

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock

/** What the install is doing, shared with MainActivity (one process). */
object InstallState {
    @Volatile var progress: Installer.Progress? = null
    @Volatile var running = false
    @Volatile var stop = false
}

/**
 * Runs [Installer] as a foreground service, so the system keeps it alive with the screen off or the launcher in the background, and shows its progress in a
 * notification. A partial wake lock keeps the CPU awake for the copy. Started by MainActivity; ACTION_STOP pauses it (the install resumes from the last part).
 */
class InstallService : Service() {
    companion object {
        const val ACTION_STOP = "dev.nxnative.fortunesweave.STOP"
        private const val CHANNEL = "install"
        private const val ID = 1

        fun gameDir(ctx: Context) = ctx.getExternalFilesDir(null) ?: ctx.filesDir

        fun start(ctx: Context) = ctx.startForegroundService(Intent(ctx, InstallService::class.java))
        fun stop(ctx: Context) = ctx.startService(Intent(ctx, InstallService::class.java).setAction(ACTION_STOP))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            InstallState.stop = true
            if (!InstallState.running) stopSelf()
            return START_NOT_STICKY
        }
        channel()
        startForeground(ID, note("Preparing the install", 0, 0, ongoing = true), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        if (InstallState.running) return START_NOT_STICKY
        InstallState.running = true
        InstallState.stop = false
        Thread({ run() }, "install").start()
        return START_NOT_STICKY
    }

    private fun run() {
        val lock = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "nxnative:install")
        lock.acquire(6 * 3600 * 1000L)
        val nm = getSystemService(NotificationManager::class.java)
        var last = 0L
        var end: Installer.Progress? = null
        try {
            end = Installer(gameDir(this)).install(report = { p ->
                InstallState.progress = p
                val now = SystemClock.elapsedRealtime()
                if (now - last > 1000) { last = now; nm.notify(ID, note(p.text, p.done, p.total, ongoing = true)) }
            }, keepGoing = { !InstallState.stop })
        } finally {
            lock.release()
            InstallState.running = false
            stopForeground(STOP_FOREGROUND_REMOVE)
            end?.let { if (it.phase == Installer.Phase.DONE || it.phase == Installer.Phase.FAILED) nm.notify(ID + 1, note(it.text, 0, 0, ongoing = false)) }
            stopSelf()
        }
    }

    private fun channel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel(CHANNEL, "Game data install", NotificationManager.IMPORTANCE_LOW))
    }

    private fun note(text: String, done: Long, total: Long, ongoing: Boolean): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Fortune's Weave game data")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(ongoing)
            .setOnlyAlertOnce(true)
            .apply { if (total > 0) setProgress(1000, (done * 1000 / total).toInt(), false) }
            .build()
    }
}
