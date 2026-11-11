package dev.nxnative.fortunesweave

import android.content.Context
import android.content.SharedPreferences

/**
 * The player's settings, kept in the app's preferences and handed to the host as environment settings when the game starts (GameActivity.startHost). The pause menu
 * and the launcher show them through [SettingsPanel].
 * Console mode and the frame-rate counter apply at once (the host's dm_set_opmode, dm/host/hle_oe.c; the counter is the app's); the rest apply when the game next starts
 * ([Choice.live] false).
 */
class Settings(private val context: Context) {
    private val p: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    /** One setting: its choices as (value, label) (a function: the GPU drivers are whatever is installed), the value stored, and whether a change reaches a running game. */
    class Choice(val key: String, val title: String, val choices: (Context) -> List<Pair<String, String>>, val default: String, val live: Boolean)

    companion object {
        private fun fixed(vararg c: Pair<String, String>): (Context) -> List<Pair<String, String>> = { c.toList() }
        // DM_OPMODE: 1 console (docked: the Switch 2's TV output, its higher render resolution), 0 handheld. The profile's default is console (hle_oe.c dm_opmode)
        val CONSOLE_MODE = Choice("opmode", "Console mode", fixed("1" to "Docked", "0" to "Handheld"), "1", live = true)
        // DM_LANG: the language the game is told the console uses (hle_oe.c GetDesiredLanguage; default en-US). Codes as nn::settings::LanguageCode
        val LANGUAGE = Choice("lang", "Game language", fixed(
            "en-US" to "English (US)", "en-GB" to "English (UK)", "ja" to "Japanese", "fr" to "French", "fr-CA" to "French (Canada)", "de" to "German",
            "it" to "Italian", "es" to "Spanish", "es-419" to "Spanish (Latin America)", "nl" to "Dutch", "pt-BR" to "Portuguese (Brazil)", "ko" to "Korean",
            "zh-Hans" to "Chinese (Simplified)", "zh-Hant" to "Chinese (Traditional)"), "en-US", live = false)
        // DM_AUDIO: 1 sound through AAudio (dm/android/host/dm_aaudio.c), 0 silent
        val SOUND = Choice("audio", "Sound", fixed("1" to "On", "0" to "Off"), "1", live = false)
        // DM_GPU_DRIVER: "" the system driver, else an installed driver's folder name (Drivers.kt); loaded through libadrenotools (dm/android/host/vk_driver.c)
        val GPU_DRIVER = Choice("gpu_driver", "GPU driver", { c -> listOf("" to "System") + Drivers.list(c).map { it.folder.name to "${it.name} ${it.version}".trim() } }, "", live = false)
        // the frame-rate counter in the game screen's corner: the game's finished frames a second (GameActivity, dmfront.c nativeFrames); app-side only
        val FPS_COUNTER = Choice("fps", "Frame-rate counter", fixed("0" to "Off", "1" to "On"), "0", live = true)
        // DM_FPS_TARGET stays 0, the game's own frame rate (30 in gameplay, 60 in menus). The host can write 60 into the game's target (dm/host/hle_fpslock.c), but the game
        // steps its logic by a fixed amount a frame and the device does not hold 60 in gameplay (GPU 11 to 19 ms a frame, logs_mem3/log_warm1.txt and log_blackzoom.txt), so
        // that setting only ran the game slow: it is gone from the app (run/env.txt can still set DM_FPS_TARGET=60 for a test). 60 on screen comes from frame generation
        // DM_LSFG: Lossless Scaling's frame generation (dm/third_party/lsfg-eden, GPL-3.0-or-later) with the user's own Lossless.dll, imported on the Settings page;
        // one generated frame between each two real ones, so 30 fps shows as 60 (with about half a frame more delay)
        val FRAME_GEN = Choice("lsfg", "Frame rate", fixed("0" to "30 fps (game's own)", "1" to "60 fps (frame generation)"), "0", live = false)
        val ALL = listOf(CONSOLE_MODE, LANGUAGE, SOUND, GPU_DRIVER, FRAME_GEN, FPS_COUNTER)
        // Button mapping (FeTheme.face), app-side, live: which physical face button sends which of the game's. Nintendo: by position (right A, bottom B, top X, left Y);
        // Xbox and PlayStation: the bottom button (A / ✕) is the game's A, the right one B, the left X, the top Y. Then A/B and X/Y can be swapped. The launcher and the
        // pause menu follow it too (the button sending A picks, B goes back)
        val BUTTON_STYLE = Choice("button_style", "Button style", fixed("nintendo" to "Nintendo", "xbox" to "Xbox", "playstation" to "PlayStation"), "nintendo", live = true)
        val SWAP_AB = Choice("swap_ab", "Swap A / B", fixed("0" to "Off", "1" to "On"), "0", live = true)
        val SWAP_XY = Choice("swap_xy", "Swap X / Y", fixed("0" to "Off", "1" to "On"), "0", live = true)
        val MAPPING = listOf(BUTTON_STYLE, SWAP_AB, SWAP_XY)
    }

    /** The stored value, or the default when the stored one is no longer a choice (a driver removed since). */
    operator fun get(c: Choice): String = p.getString(c.key, c.default)?.takeIf { v -> c.choices(context).any { it.first == v } } ?: c.default
    operator fun set(c: Choice, v: String) { p.edit().putString(c.key, v).apply() }
    fun label(c: Choice): String = c.choices(context).firstOrNull { it.first == get(c) }?.second ?: get(c)
    /** The next choice after the current one, wrapping round (the menu's rows step through their choices). */
    fun step(c: Choice, by: Int = 1): String {
        val all = c.choices(context); val i = all.indexOfFirst { it.first == get(c) }.coerceAtLeast(0); val n = all.size
        return all[((i + by) % n + n) % n].first
    }

    /** The host's environment for these settings. */
    fun env(): Map<String, String> {
        val e = mutableMapOf("DM_OPMODE" to get(CONSOLE_MODE), "DM_LANG" to get(LANGUAGE), "DM_AUDIO" to get(SOUND), "DM_FPS_TARGET" to "0", "DM_LSFG" to get(FRAME_GEN))
        val drv = get(GPU_DRIVER)
        val d = if (drv.isEmpty()) null else Drivers.list(context).firstOrNull { it.folder.name == drv }
        if (d != null) { e["DM_GPU_DRIVER"] = d.library.path; e["DM_HOOK_DIR"] = context.applicationInfo.nativeLibraryDir }   // adrenotools' hooks sit beside the host
        return e
    }
}
