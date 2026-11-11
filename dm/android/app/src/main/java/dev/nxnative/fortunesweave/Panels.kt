package dev.nxnative.fortunesweave

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * The launcher's and the pause menu's shared layout: a list of sections on the left and a panel on the right that shows the open one. The player steps in and out:
 *  - the left list: the d-pad moves between its buttons, A opens a section (the right panel shows it and focus moves into it), B leaves (the activity decides how);
 *  - a panel: the d-pad moves inside it only, A acts on the item (a setting row: steps into it, see [SettingsPanel]), B goes back to the left list;
 *  - a setting row stepped into: left and right change its value, A or B steps back out.
 * The d-pad never crosses between the left list and the panel by itself ([FocusBox]), so no button does two things at once. Touch: a tap opens a section or acts on
 * an item directly.
 */

/** A panel with a stepped-into state that B leaves first. */
interface Editable { val editing: Boolean; fun stopEdit() }

/** A container focus does not leave: the d-pad stops at its edges. */
class FocusBox(context: Context) : FrameLayout(context) {
    override fun focusSearch(focused: View?, direction: Int): View? {
        val next = super.focusSearch(focused, direction)
        return if (next != null && isInside(next)) next else focused
    }
    private fun isInside(v: View): Boolean { var p: Any? = v; while (p is View) { if (p === this) return true; p = p.parent }; return false }
}

/** Which section is open on the right, the way back, and the hint line. */
class MenuNav(private val context: Context, private val settings: Settings, private val right: FocusBox, private val placeholder: View, private val hint: TextView, private val leftHint: String) {
    private var button: View? = null
    private var panel: View? = null
    /** The launcher: B closes the open section (the right side goes back to its empty state, its button is no longer marked) and focus returns to that button.
     *  Off (the pause menu): B only moves focus from the panel back to its button, and the panel stays. */
    var backCloses = false

    init { clear() }

    /** Shows [p] for section [b]; with [focus] (a controller press), focus moves to its first item. */
    fun open(b: View, p: View, focus: Boolean) {
        button?.isSelected = false; b.isSelected = true; button = b
        if (panel !== p) {
            right.removeAllViews()
            right.addView(ScrollView(context).apply { isVerticalScrollBarEnabled = false; isFillViewport = true
                addView(LinearLayout(context).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_VERTICAL
                    (p.parent as? ViewGroup)?.removeView(p); addView(p) }) })
            panel = p
        }
        if (focus) p.post { first(p)?.requestFocus(); update() }
        update()
    }

    /** The empty right side (a menu that has just opened). */
    fun clear() {
        button?.isSelected = false; button = null; panel = null
        right.removeAllViews(); right.addView(placeholder); update()
    }

    /** B: leaves a row being changed, else goes from the panel back to its section's button. False when already in the left list (the activity's own back). */
    fun back(): Boolean {
        val p = panel ?: return false
        if (p is Editable && p.editing) { p.stopEdit(); update(); return true }
        if (backCloses) { val b = button; clear(); b?.requestFocus(); update(); return true }   // from the panel or from the left list with a section open
        if (p.hasFocus()) { button?.requestFocus(); update(); return true }
        return false
    }

    /** The hint line for where focus is. */
    fun update() {
        val p = panel
        val t = when {
            p is Editable && p.editing -> "◀ ▶ change  ·  [A] or [B] done"
            p != null && p.hasFocus() -> "D-pad move  ·  [A] select  ·  [B] back"
            else -> leftHint
        }
        hint.text = FeTheme.hint(context, settings, t, FeTheme.dp(context, 18))   // "[A]" / "[B]": the chosen style's icons (FeTheme.faceIcon)
    }

    private fun first(v: View): View? {
        if (v.isFocusable && v.visibility == View.VISIBLE && v !is ViewGroup) return v
        if (v is ViewGroup) { if (v.isFocusable && v.visibility == View.VISIBLE && v.descendantFocusability == ViewGroup.FOCUS_BLOCK_DESCENDANTS) return v
            for (i in 0 until v.childCount) first(v.getChildAt(i))?.let { return it }
            if (v.isFocusable && v.visibility == View.VISIBLE) return v }
        return null
    }

    companion object {
        /** A section heading: gold spaced capitals after a diamond. */
        fun head(c: Context, t: String) = TextView(c).apply {
            text = t; textSize = 15f; letterSpacing = 0.3f; typeface = FeTheme.SERIF; setTextColor(FeTheme.GOLD_DIM); setPadding(0, 0, 0, FeTheme.dp(c, 8))
            FeTheme.art(c, "fe_dia_small")?.let { d -> d.setBounds(0, 0, FeTheme.dp(c, 16), FeTheme.dp(c, 16)); setCompoundDrawables(d, null, null, null); compoundDrawablePadding = FeTheme.dp(c, 10) }
        }
        fun note(c: Context, t: String) = TextView(c).apply { text = t; textSize = 13f; typeface = FeTheme.SERIF; setTextColor(FeTheme.MUTED); setPadding(0, FeTheme.dp(c, 6), 0, 0) }
        /** The empty right side: a faint crest and what to do. */
        fun placeholder(c: Context, t: String) = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER
            FeTheme.art(c, "fe_crest")?.let { addView(ImageView(c).apply { setImageDrawable(it); alpha = 0.16f }, LinearLayout.LayoutParams(FeTheme.dp(c, 110), FeTheme.dp(c, 140))) }
            addView(TextView(c).apply { text = t; textSize = 16f; typeface = FeTheme.SERIF; setTextColor(FeTheme.MUTED); gravity = Gravity.CENTER; setPadding(0, FeTheme.dp(c, 10), 0, 0) })
        }
    }
}

/** Controls: the button mapping rows and the face-button diagram (at each position, the game's button it sends now; Confirm under A, Back under B). */
class ControlsPanel(context: Context, private val settings: Settings) : LinearLayout(context), Editable {
    private val diagram = FrameLayout(context)
    private val rows = SettingsPanel(context, settings, Settings.MAPPING) { _, _ -> drawFaces(); onMappingChange() }
    /** Hooks the hint line to a change of style or swap (its icons follow). */
    var onMappingChange: () -> Unit = {}
    val table: SettingsPanel get() = rows
    override val editing get() = rows.editing
    override fun stopEdit() = rows.stopEdit()

    init {
        orientation = VERTICAL
        addView(MenuNav.head(context, "CONTROLS")); addView(rows)
        addView(MenuNav.note(context, "On the Odin, as the game sees them"))
        addView(diagram, LayoutParams(FeTheme.dp(context, 200), FeTheme.dp(context, 150)).apply { gravity = Gravity.CENTER_HORIZONTAL; topMargin = FeTheme.dp(context, 4) })
        addView(MenuNav.note(context, "Button icons by Zacksly (CC BY 3.0 | zacksly.itch.io)").apply { setTextSize(android.util.TypedValue.COMPLEX_UNIT_DIP, 10f) })
        drawFaces()
    }

    fun refresh() { rows.refresh(); drawFaces() }

    private fun drawFaces() {
        fun dp(v: Int) = FeTheme.dp(context, v)
        diagram.removeAllViews()
        val pos = listOf(dp(83) to 0, dp(18) to dp(48), dp(148) to dp(48), dp(83) to dp(96))
        for ((i, face) in FeTheme.faces(settings).withIndex()) {
            val col = LinearLayout(context).apply {
                orientation = VERTICAL; gravity = Gravity.CENTER_HORIZONTAL
                val icon = FeTheme.faceIcon(context, settings, i)
                if (icon != null) addView(ImageView(context).apply { setImageDrawable(icon) }, LayoutParams(dp(36), dp(36)))
                else addView(TextView(context).apply {   // no icon pack in this build: a plain circle with the game's letter
                    text = face.name; textSize = 15f; typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER; setTextColor(FeTheme.BRIGHT)
                    background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.rgb(0x2b, 0x24, 0x40)); setStroke(dp(2), Color.rgb(0xb9, 0xab, 0x86)) }
                }, LayoutParams(dp(34), dp(34)))
                // under each: the game's button it sends, and what A and B do
                val what = when (face) { FeTheme.Face.A -> "A · Confirm"; FeTheme.Face.B -> "B · Back"; else -> face.name }
                addView(TextView(context).apply { text = what; setTextSize(android.util.TypedValue.COMPLEX_UNIT_DIP, 12f); typeface = FeTheme.SERIF; setTextColor(FeTheme.LIGHT) })
            }
            diagram.addView(col, FrameLayout.LayoutParams(dp(70), FrameLayout.LayoutParams.WRAP_CONTENT).apply { leftMargin = pos[i].first - dp(18); topMargin = pos[i].second })
        }
    }
}

/** GPU drivers: Import at the top, then the system driver and each installed one with a check (use it from the next start) and a trash can (delete, after
 *  [onDelete]'s confirmation). The driver in use has a gold check and "in use". */
class DriversPanel(context: Context, private val settings: Settings, private val onImport: () -> Unit, private val onDelete: (Drivers.Driver) -> Unit) : LinearLayout(context) {
    init { orientation = VERTICAL; refresh() }

    /** Rebuilds the list (after an import, a delete or a change of driver); focus stays on the same kind of button of the same driver when it is still there. */
    fun refresh(keep: String? = null) {
        fun dp(v: Int) = FeTheme.dp(context, v)
        removeAllViews()
        addView(MenuNav.head(context, "GPU DRIVERS"))
        addView(Button(context).apply { text = "Import a GPU driver…"; FeTheme.button(this); setOnClickListener { onImport() } },
            LayoutParams(LayoutParams.MATCH_PARENT, dp(56)).apply { bottomMargin = dp(10) })
        addView(TextView(context).apply { text = "Installed"; textSize = 14f; letterSpacing = 0.15f; typeface = FeTheme.SERIF; setTextColor(FeTheme.MUTED); setPadding(0, 0, 0, dp(4)) })
        val chosen = settings[Settings.GPU_DRIVER]
        var refocus: View? = null
        for (d in listOf<Drivers.Driver?>(null) + Drivers.list(context)) {
            val id = d?.folder?.name ?: ""; val inUse = id == chosen
            val row = LinearLayout(context).apply { orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            row.addView(LinearLayout(context).apply {
                orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; background = FeTheme.plaque(context, "cream"); FeTheme.plaquePadding(this)
                addView(TextView(context).apply { text = d?.name ?: "System driver"; textSize = 16f; typeface = FeTheme.SERIF; setTextColor(FeTheme.INK); maxLines = 1; ellipsize = android.text.TextUtils.TruncateAt.END })
                addView(TextView(context).apply { text = d?.version ?: ""; textSize = 13f; typeface = FeTheme.SERIF; setTextColor(FeTheme.INK_SOFT); setPadding(dp(8), 0, 0, 0); maxLines = 1 },
                    LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
                if (inUse) addView(TextView(context).apply { text = "in use"; textSize = 13f; typeface = FeTheme.SERIF; setTextColor(FeTheme.GOLD) })
            }, LayoutParams(0, dp(52), 1f))
            val use = FeTheme.iconButton(context, R.drawable.fe_ic_check, if (inUse) FeTheme.GOLD else FeTheme.LIGHT).apply {
                contentDescription = "Use ${d?.name ?: "the system driver"}"
                setOnClickListener { settings[Settings.GPU_DRIVER] = id; refresh("use:$id") }
            }
            row.addView(use, LayoutParams(dp(46), dp(46)).apply { marginStart = dp(8) })
            if (d != null) {
                val del = FeTheme.iconButton(context, R.drawable.fe_ic_trash).apply { contentDescription = "Delete ${d.name}"; setOnClickListener { onDelete(d) } }
                row.addView(del, LayoutParams(dp(46), dp(46)).apply { marginStart = dp(6) })
                if (keep == "del:$id") refocus = del
            } else row.addView(View(context), LayoutParams(dp(46), dp(46)).apply { marginStart = dp(6) })
            if (keep == "use:$id") refocus = use
            addView(row, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { bottomMargin = dp(6) })
        }
        addView(MenuNav.note(context, "✓ use this driver from the next start  ·  trash: delete it"))
        refocus?.let { v -> v.post { v.requestFocus() } }
    }
}
