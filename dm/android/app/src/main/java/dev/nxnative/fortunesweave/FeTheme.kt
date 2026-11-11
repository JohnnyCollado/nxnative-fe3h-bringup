package dev.nxnative.fortunesweave

import android.app.AlertDialog
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.graphics.drawable.StateListDrawable
import android.view.KeyEvent
import android.view.View
import android.widget.ListView
import android.widget.TextView

/**
 * The Fire Emblem look of the launcher and the pause menu: plaques behind buttons and setting rows, a gold frame on the one focused or pressed, serif type.
 *
 * Plaques are drawn like Godot's NinePatchRect in "tile fit" mode ([Plaque]): scaled evenly to the view's height, the end caps (eyelet, star, border) and the crest are
 * drawn as they are, and only a thin plain strip repeats to fill the width, so nothing is stretched out of proportion. The art is game art kept out of the repository:
 * app/art_res/ (git-ignored): drawable-nodpi/fe_<cream|dark|purple>_<l|f|c|r>.png (left cap, fill strip, crest, right cap), drawable-xhdpi/fe_dia_small.png and
 * fe_crest.png, the icon in mipmap-*. Without it every plaque is a plain shape in the same colours, so a fresh checkout builds and keeps the layout.
 *
 * Controller: the face buttons follow the button mapping settings ([Settings.BUTTON_STYLE], swaps): the button that sends the game's A picks, the one that sends B goes
 * back, in the launcher, dialogs and the pause menu alike; the d-pad and left stick move focus.
 */
object FeTheme {
    val INK = Color.rgb(0x3a, 0x33, 0x54)
    val INK_SOFT = Color.rgb(0x6b, 0x60, 0x80)
    val LIGHT = Color.rgb(0xe8, 0xe2, 0xf6)
    val BRIGHT = Color.rgb(0xff, 0xf8, 0xe6)
    val GOLD = Color.rgb(0xe9, 0xcf, 0x7c)
    val GOLD_DIM = Color.rgb(0xd9, 0xc4, 0x8a)
    val MUTED = Color.rgb(0x9a, 0x90, 0xb8)
    val SERIF: Typeface = Typeface.SERIF

    fun dp(c: Context, v: Int) = (v * c.resources.displayMetrics.density).toInt()

    /** An art drawable by name, or null when art_res is not in this build. */
    fun art(c: Context, name: String, type: String = "drawable"): Drawable? {
        val id = c.resources.getIdentifier(name, type, c.packageName)
        return if (id == 0) null else c.getDrawable(id)
    }

    private val bitmaps = HashMap<String, Bitmap?>()
    private fun bitmap(c: Context, name: String): Bitmap? = bitmaps.getOrPut(name) {
        val id = c.resources.getIdentifier(name, "drawable", c.packageName)
        if (id == 0) null else BitmapFactory.decodeResource(c.resources, id, BitmapFactory.Options().apply { inScaled = false })
    }

    /** A plaque: [left] cap, [fill] strip repeated, the [crest] once in the middle (if any), [right] cap; all scaled by the height alone. */
    class Plaque(private val left: Bitmap, private val fill: Bitmap, private val crest: Bitmap?, private val right: Bitmap) : Drawable() {
        private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        private val fillPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply { shader = BitmapShader(fill, Shader.TileMode.REPEAT, Shader.TileMode.CLAMP) }
        private val m = Matrix()
        override fun draw(canvas: Canvas) {
            val b = bounds; if (b.isEmpty) return
            val s = b.height().toFloat() / left.height
            fun w(bm: Bitmap) = bm.width * s
            val lw = w(left); val rw = w(right); val cw = crest?.let { w(it) } ?: 0f
            fun strip(x0: Float, x1: Float) {
                if (x1 <= x0) return
                m.setScale(s, b.height().toFloat() / fill.height); m.postTranslate(x0, b.top.toFloat()); fillPaint.shader.setLocalMatrix(m)
                canvas.drawRect(x0, b.top.toFloat(), x1 + 0.5f, b.bottom.toFloat(), fillPaint)
            }
            val l = b.left.toFloat(); val r = b.right.toFloat()
            if (crest != null && r - l >= lw + rw + cw) {
                val cx = (l + r - cw) / 2
                strip(l + lw - 0.5f, cx); strip(cx + cw - 0.5f, r - rw)
                canvas.drawBitmap(crest, null, RectF(cx, b.top.toFloat(), cx + cw, b.bottom.toFloat()), paint)
            } else strip(l + lw - 0.5f, r - rw)
            canvas.drawBitmap(left, null, RectF(l, b.top.toFloat(), l + lw, b.bottom.toFloat()), paint)
            canvas.drawBitmap(right, null, RectF(r - rw, b.top.toFloat(), r, b.bottom.toFloat()), paint)
        }
        override fun setAlpha(alpha: Int) { paint.alpha = alpha; fillPaint.alpha = alpha; invalidateSelf() }
        override fun setColorFilter(cf: ColorFilter?) { paint.colorFilter = cf; fillPaint.colorFilter = cf; invalidateSelf() }
        @Deprecated("Drawable API") override fun getOpacity() = PixelFormat.TRANSLUCENT
    }

    /** Fallback plaque colours (fill, edge). */
    private fun shape(c: Context, name: String): Drawable {
        val (fill, edge) = when (name) {
            "cream" -> Color.rgb(0xef, 0xe8, 0xd6) to Color.rgb(0xb9, 0xab, 0x86)
            "purple" -> Color.rgb(0x6a, 0x58, 0xb0) to Color.rgb(0xc9, 0xb7, 0x7a)
            else -> Color.rgb(0x3e, 0x35, 0x66) to Color.rgb(0x8a, 0x7a, 0x58)
        }
        return GradientDrawable().apply { setColor(fill); setStroke(dp(c, 2), edge); cornerRadius = dp(c, 3).toFloat() }
    }

    /** The plaque "cream", "dark" or "purple". */
    fun plaque(c: Context, name: String): Drawable {
        val l = bitmap(c, "fe_${name}_l"); val f = bitmap(c, "fe_${name}_f"); val r = bitmap(c, "fe_${name}_r")
        return if (l != null && f != null && r != null) Plaque(l, f, bitmap(c, "fe_${name}_c"), r) else shape(c, name)
    }

    /** The plaque with a frame over it: gold for focused or pressed (and a row being changed), dim gold for a section that is open. */
    fun framed(c: Context, name: String, color: Int = GOLD, width: Int = 3): Drawable {
        val edge = GradientDrawable().apply { setColor(Color.TRANSPARENT); setStroke(dp(c, width), color); cornerRadius = dp(c, 2).toFloat() }
        return LayerDrawable(arrayOf(plaque(c, name), edge))
    }

    /** Background for a button or a row: [normal] plaque, [on] plaque with a gold frame when focused or pressed, a dim gold frame when selected (the open section),
     *  faded when disabled. */
    fun states(c: Context, normal: String, on: String): StateListDrawable = StateListDrawable().apply {
        addState(intArrayOf(android.R.attr.state_pressed), framed(c, on))
        addState(intArrayOf(android.R.attr.state_focused), framed(c, on))
        addState(intArrayOf(android.R.attr.state_selected), framed(c, normal, Color.rgb(0x8a, 0x7a, 0x58), 2))
        addState(intArrayOf(-android.R.attr.state_enabled), plaque(c, normal).apply { alpha = 110 })
        addState(intArrayOf(), plaque(c, normal))
    }

    /** A small diamond icon button (the drivers' use and delete): the game's diamond frame, a gold ring when focused or pressed. [glyph] is a res/drawable vector. */
    fun iconButton(c: Context, glyph: Int, tint: Int = LIGHT): android.widget.ImageView = android.widget.ImageView(c).apply {
        fun base(): Drawable = art(c, "fe_dia_small") ?: GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.rgb(0x2b, 0x24, 0x40)); setStroke(dp(c, 2), Color.rgb(0xb9, 0xab, 0x86)) }
        fun ring(): Drawable = LayerDrawable(arrayOf(GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.argb(0x55, 0xe9, 0xcf, 0x7c)); setStroke(dp(c, 2), GOLD) }, base()))
        background = StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), ring()); addState(intArrayOf(android.R.attr.state_focused), ring()); addState(intArrayOf(), base())
        }
        setImageResource(glyph); imageTintList = ColorStateList.valueOf(tint)
        val p = dp(c, 12); setPadding(p, p, p, p); scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
        isFocusable = true; isClickable = true
    }

    fun textStates(normal: Int, on: Int = BRIGHT, disabled: Int = MUTED) = ColorStateList(
        arrayOf(intArrayOf(android.R.attr.state_pressed), intArrayOf(android.R.attr.state_focused), intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
        intArrayOf(on, on, disabled, normal))

    /** Space inside a plaque for its content: past the eyelet on the left, inside the border on the right. */
    fun plaquePadding(v: View) { val c = v.context; v.setPadding(dp(c, 48), 0, dp(c, 34), 0) }

    /** A launcher or pause-menu button: dark plaque, a diamond before its label, bright purple with a gold frame when focused or pressed. */
    fun button(v: TextView, normal: String = "dark", on: String = "purple") {
        val c = v.context
        v.background = states(c, normal, on)
        v.stateListAnimator = null
        v.isAllCaps = false; v.typeface = SERIF; v.textSize = 18f; v.maxLines = 1
        v.setTextColor(textStates(LIGHT))
        v.gravity = android.view.Gravity.START or android.view.Gravity.CENTER_VERTICAL
        v.minHeight = dp(c, 56); v.minimumHeight = dp(c, 56)
        plaquePadding(v)
        art(c, "fe_dia_small")?.let { d -> d.setBounds(0, 0, dp(c, 16), dp(c, 16)); v.setCompoundDrawables(d, null, null, null); v.compoundDrawablePadding = dp(c, 10) }
        v.isFocusable = true
    }

    // ---------- Button mapping ----------
    enum class Face { A, B, X, Y }

    /** The game's face button an Android key code sends under the mapping settings, or null for other keys. Nintendo: by position (the right button is A, the bottom
     *  B, the top X, the left Y; Android names buttons by the Xbox positions). Xbox and PlayStation: by the Xbox label (the bottom button, A or ✕, is the game's A).
     *  Then the swaps. */
    fun face(s: Settings, code: Int): Face? {
        val nintendo = s[Settings.BUTTON_STYLE] == "nintendo"
        var f = when (code) {
            KeyEvent.KEYCODE_BUTTON_A -> if (nintendo) Face.B else Face.A
            KeyEvent.KEYCODE_BUTTON_B -> if (nintendo) Face.A else Face.B
            KeyEvent.KEYCODE_BUTTON_X -> if (nintendo) Face.Y else Face.X
            KeyEvent.KEYCODE_BUTTON_Y -> if (nintendo) Face.X else Face.Y
            else -> return null
        }
        if (s[Settings.SWAP_AB] == "1") f = when (f) { Face.A -> Face.B; Face.B -> Face.A; else -> f }
        if (s[Settings.SWAP_XY] == "1") f = when (f) { Face.X -> Face.Y; Face.Y -> Face.X; else -> f }
        return f
    }

    /** The game's face button at each position (top, left, right, bottom) under the mapping settings: for the pause menu's diagram. */
    fun faces(s: Settings): List<Face> = listOf(KeyEvent.KEYCODE_BUTTON_Y, KeyEvent.KEYCODE_BUTTON_X, KeyEvent.KEYCODE_BUTTON_B, KeyEvent.KEYCODE_BUTTON_A).map { face(s, it)!! }

    /** The chosen style's icon for a position (0 top, 1 left, 2 right, 3 bottom, the order of [faces]): Zacksly's button icon packs (CC BY 3.0, zacksly.itch.io;
     *  res/drawable-nodpi/ctl_<ns|xb|ps>_face_*.png, unmodified, as PlayFieldPortal bundles them; credit in dm/android/ui-art-credit.txt and the Controls panel). */
    fun faceIcon(c: Context, s: Settings, pos: Int): Drawable? {
        val pack = when (s[Settings.BUTTON_STYLE]) { "xbox" -> "xb"; "playstation" -> "ps"; else -> "ns" }
        return art(c, "ctl_${pack}_face_" + listOf("north", "west", "east", "south")[pos])
    }
    /** The icon of the button that sends the game's [f] now (hint lines). */
    fun iconOf(c: Context, s: Settings, f: Face): Drawable? { val i = faces(s).indexOf(f); return if (i < 0) null else faceIcon(c, s, i) }
    /** [t] with "[A]" / "[B]" replaced by the icon of the button that sends that face now, sized to the text. */
    fun hint(c: Context, s: Settings, t: String, px: Int): CharSequence {
        val out = android.text.SpannableStringBuilder()
        var i = 0
        while (i < t.length) {
            val f = if (t.startsWith("[A]", i)) Face.A else if (t.startsWith("[B]", i)) Face.B else null
            val d = f?.let { iconOf(c, s, it) }
            if (f != null && d != null) {
                d.setBounds(0, 0, px, px); val at = out.length; out.append(" ")
                out.setSpan(android.text.style.ImageSpan(d, android.text.style.DynamicDrawableSpan.ALIGN_CENTER), at, at + 1, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE); i += 3
            } else if (f != null) { out.append(f.name); i += 3 } else { out.append(t[i]); i++ }
        }
        return out
    }

    fun isPick(s: Settings, code: Int) = face(s, code) == Face.A
    fun isBack(s: Settings, code: Int) = face(s, code) == Face.B

    /** Dialogs with the mapping: the button that sends A picks the focused item, the one that sends B closes. */
    fun keys(d: AlertDialog, s: Settings): AlertDialog {
        d.setOnKeyListener { _, code, e ->
            when {
                isPick(s, code) -> {
                    if (e.action == KeyEvent.ACTION_UP) {
                        val f: View? = d.currentFocus
                        if (f is ListView && f.selectedItemPosition >= 0) f.performItemClick(f.selectedView, f.selectedItemPosition, f.selectedItemId) else f?.performClick()
                    }
                    true
                }
                isBack(s, code) -> { if (e.action == KeyEvent.ACTION_UP) d.dismiss(); true }
                else -> false
            }
        }
        return d
    }
}
