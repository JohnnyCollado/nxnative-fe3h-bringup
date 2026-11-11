package dev.nxnative.fortunesweave

import android.annotation.SuppressLint
import android.content.Context
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * The settings as rows, one per [Settings.Choice] in [choices], each a cream plaque with the setting's name on the left and its value on the right (FeTheme); name and
 * value share the row and end in "…" when they do not fit, so they never run into each other.
 * Controller (see Panels.kt): up and down move between rows (the focused one gets a gold frame); A steps into a row (purple, ◀ value ▶) and only then do left and
 * right change its value; A or B steps back out ([stopEdit]). Touch: a tap on a row's left third steps back, anywhere else forward.
 * Rows of settings that apply when the game next starts end in " *", and a note under them says so. [onChange] hears each change after it is stored.
 */
class SettingsPanel(context: Context, private val settings: Settings, private val choices: List<Settings.Choice> = Settings.ALL,
                    private val onChange: (Settings.Choice, String) -> Unit) : LinearLayout(context), Editable {
    private class Row(val view: LinearLayout, val name: TextView, val value: TextView)
    private val rows = mutableMapOf<Settings.Choice, Row>()
    private var edit: Settings.Choice? = null
    val firstRow: View
    override val editing get() = edit != null

    init {
        orientation = VERTICAL
        for (c in choices) {
            val r = row(c)
            rows[c] = r
            addView(r.view, LayoutParams(LayoutParams.MATCH_PARENT, FeTheme.dp(context, 46)).apply { bottomMargin = FeTheme.dp(context, 6) })
        }
        firstRow = rows.getValue(choices.first()).view
        if (choices.any { !it.live }) addView(TextView(context).apply {
            text = "* applies when the game next starts"; setTextSize(TypedValue.COMPLEX_UNIT_DIP, 12f); typeface = FeTheme.SERIF; setTextColor(FeTheme.MUTED); setPadding(4, 2, 4, 2)
        })
        refresh()
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun row(c: Settings.Choice): Row {
        fun label(g: Int) = TextView(context).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_DIP, 15f); typeface = FeTheme.SERIF; gravity = g or Gravity.CENTER_VERTICAL
            maxLines = 1; ellipsize = TextUtils.TruncateAt.END
        }
        val name = label(Gravity.START); val value = label(Gravity.END)
        var tapX = -1f
        val v = LinearLayout(context).apply {
            orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            isFocusable = true; isClickable = true
            addView(name, LayoutParams(0, LayoutParams.MATCH_PARENT, 1.15f))
            addView(value, LayoutParams(0, LayoutParams.MATCH_PARENT, 1f).apply { marginStart = FeTheme.dp(context, 8) })
            setOnTouchListener { _, e -> if (e.action == MotionEvent.ACTION_DOWN) tapX = e.x; false }
            setOnClickListener {
                if (tapX >= 0) { val back = tapX < width / 3f; tapX = -1f; change(c, if (back) -1 else 1) }   // touch: change at once
                else { val was = edit; edit = if (was == c) null else c; was?.let { show(it) }; show(c); (parent as? View)?.let { notifyNav() } }   // controller: step in or out
            }
            setOnKeyListener { _, code, e ->
                if (edit != c) false   // not stepped in: the d-pad only moves
                else when (code) {
                    KeyEvent.KEYCODE_DPAD_RIGHT -> { if (e.action == KeyEvent.ACTION_DOWN) change(c, 1); true }
                    KeyEvent.KEYCODE_DPAD_LEFT -> { if (e.action == KeyEvent.ACTION_DOWN) change(c, -1); true }
                    KeyEvent.KEYCODE_DPAD_UP, KeyEvent.KEYCODE_DPAD_DOWN -> true   // stay on the row being changed
                    else -> false
                }
            }
            setOnFocusChangeListener { _, has -> if (!has && edit == c) { edit = null }; show(c) }
        }
        return Row(v, name, value)
    }

    /** Hooks the hint line (MenuNav.update) to step in and out. */
    var onEditChange: () -> Unit = {}
    private fun notifyNav() = onEditChange()

    override fun stopEdit() { val c = edit ?: return; edit = null; show(c); notifyNav() }

    private fun change(c: Settings.Choice, by: Int) {
        val v = settings.step(c, by); settings[c] = v; show(c); onChange(c, v)
    }

    private fun show(c: Settings.Choice) {
        val r = rows[c] ?: return
        val on = edit == c
        r.view.background = if (on) FeTheme.framed(context, "purple") else FeTheme.states(context, "cream", "cream")
        FeTheme.plaquePadding(r.view)
        val ink = if (on) FeTheme.BRIGHT else FeTheme.INK
        r.name.setTextColor(ink); r.value.setTextColor(if (on) FeTheme.GOLD else ink)
        r.name.text = c.title + if (c.live) "" else " *"
        r.value.text = if (on) "◀  " + settings.label(c) + "  ▶" else settings.label(c)
    }

    fun refresh() { for (c in rows.keys) show(c) }
}
