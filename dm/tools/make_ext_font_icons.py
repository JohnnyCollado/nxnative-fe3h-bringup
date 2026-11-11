#!/usr/bin/env python3
# The "Nintendo extension" shared font (nintendo_ext.ttf, see make_ext_font.py) made from button icon pictures, one font per controller style, so the prompts a game
# draws with that font show the buttons the player holds. The pictures are Zacksly's button icon packs (white, solid, 128 px PNG; CC BY 3.0, dm/android/ui-art-credit.txt),
# under the Android resource names PlayFieldPortal gives them (ctl_<ns|xb|ps>_<button>.png). A game takes this font through the scalable font engine, so every picture
# is traced into an outline: the alpha channel's 50% line, by marching squares with the crossing interpolated, then thinned (Douglas-Peucker). Shapes are not redrawn.
# Which picture a code point gets follows the app's button mapping (FeTheme.kt, Settings.BUTTON_STYLE and the swaps): Nintendo by position (the game's A is the right
# button), Xbox and PlayStation by the Xbox label (the game's A is the bottom button); a swap shows the button that now sends it.
# Usage: make_ext_font_icons.py <icon folder> <ns|xb|ps> out.ttf [--swap-ab] [--swap-xy] [--sheet out.png]     needs fontTools (pip install fonttools)
import math, struct, sys, zlib, os
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

EM, ASC, DESC = 1000, 880, -120   # as make_ext_font.py: a round button spans -100 to 800 (centre 350) and is 900 wide in a 1000 advance
CY, FACE, MAXW = 350.0, 900.0, 1250.0   # the packs leave room around each picture (a face button is under half its 128 px): every picture is scaled by what makes the
                                        # style's face button 900 high, less where that would pass 900 high or 1250 wide, and centred; the advance is its width + 100

def png_alpha(path):
    # 8-bit RGBA, not interlaced (what the packs are): the alpha channel as rows of 0..1
    b = open(path, 'rb').read(); i = 8; idat = b''; w = h = 0
    while i < len(b):
        n = struct.unpack('>I', b[i:i + 4])[0]; t = b[i + 4:i + 8]; d = b[i + 8:i + 8 + n]
        if t == b'IHDR':
            w, h = struct.unpack('>II', d[:8])
            if d[8] != 8 or d[9] != 6 or d[12] != 0: sys.exit('%s: not 8-bit RGBA, not interlaced' % path)
        elif t == b'IDAT': idat += d
        i += 12 + n
    raw = zlib.decompress(idat); st = w * 4; prev = bytearray(st); o = 0; rows = []
    for _ in range(h):
        f = raw[o]; line = bytearray(raw[o + 1:o + 1 + st]); o += 1 + st
        for x in range(st):
            a = line[x - 4] if x >= 4 else 0; u = prev[x]; c = prev[x - 4] if x >= 4 else 0
            if f == 1: line[x] = (line[x] + a) & 255
            elif f == 2: line[x] = (line[x] + u) & 255
            elif f == 3: line[x] = (line[x] + (a + u) // 2) & 255
            elif f == 4:
                p = a + u - c; pa, pb, pc = abs(p - a), abs(p - u), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else u if pb <= pc else c)) & 255
        prev = line; rows.append([line[x * 4 + 3] / 255.0 for x in range(w)])
    return w, h, rows

# marching squares: corner bits TL 8, TR 4, BR 2, BL 1; each case's segments run with the inside on the right (clockwise around a shape as seen, holes the other way)
SEG = {1: ['LB'], 2: ['BR'], 3: ['LR'], 4: ['RT'], 5: ['LB', 'RT'], 6: ['BT'], 7: ['LT'], 8: ['TL'], 9: ['TB'], 10: ['TL', 'BR'], 11: ['TR'], 12: ['RL'], 13: ['RB'], 14: ['BL']}

def trace(path, level=0.5, eps=0.3):
    w, h, a = png_alpha(path)
    g = [[0.0] * (w + 2)] + [[0.0] + r + [0.0] for r in a] + [[0.0] * (w + 2)]   # a transparent border, so every shape closes
    def cross(v0, v1): return (level - v0) / (v1 - v0)
    def key(x, y, e): return ('h', x, y) if e == 'T' else ('h', x, y + 1) if e == 'B' else ('v', x, y) if e == 'L' else ('v', x + 1, y)
    def pos(k):
        t, x, y = k
        if t == 'h': return (x + cross(g[y][x], g[y][x + 1]) - 0.5, y - 0.5)   # grid index to pixel coordinates: the border is index 0, a pixel's centre is at +0.5
        return (x - 0.5, y + cross(g[y][x], g[y + 1][x]) - 0.5)
    nxt = {}
    for y in range(h + 1):
        for x in range(w + 1):
            c = (8 if g[y][x] >= level else 0) | (4 if g[y][x + 1] >= level else 0) | (2 if g[y + 1][x + 1] >= level else 0) | (1 if g[y + 1][x] >= level else 0)
            for s in SEG.get(c, []): nxt[key(x, y, s[0])] = key(x, y, s[1])
    loops = []
    while nxt:
        k0 = next(iter(nxt)); k = k0; pts = []
        while k in nxt:
            pts.append(pos(k)); k = nxt.pop(k)
        if k == k0 and len(pts) >= 3: loops.append(simplify(pts, eps))
    return w, h, [l for l in loops if len(l) >= 3]

def simplify(pts, eps):
    # Douglas-Peucker on the closed loop, split at its two farthest-apart points
    n = len(pts); i0 = 0; j0 = max(range(n), key=lambda j: (pts[j][0] - pts[0][0]) ** 2 + (pts[j][1] - pts[0][1]) ** 2)
    def dp(seg):
        keep = [seg[0]]; stack = [(0, len(seg) - 1)]; mark = {0, len(seg) - 1}
        while stack:
            a, b = stack.pop(); ax, ay = seg[a]; bx, by = seg[b]; dx, dy = bx - ax, by - ay; ln = math.hypot(dx, dy) or 1e-9; best = 0; bi = -1
            for i in range(a + 1, b):
                d = abs((seg[i][0] - ax) * dy - (seg[i][1] - ay) * dx) / ln
                if d > best: best, bi = d, i
            if best > eps: mark.add(bi); stack.append((a, bi)); stack.append((bi, b))
        return [seg[i] for i in sorted(mark)]
    a = dp(pts[i0:j0 + 1]); b = dp(pts[j0:] + [pts[0]])
    return a[:-1] + b[:-1]

def arrow(deg):
    # the stick direction arrows (U+E090..E097) have no picture in the packs: the shape make_ext_font.py draws
    p = [(-380, 105), (60, 105), (60, 310), (420, 0), (60, -310), (60, -105), (-380, -105)]; c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    q = [(500 + x * c - y * s, 350 + x * s + y * c) for x, y in p]
    if sum(q[i][0] * q[i - 1][1] - q[i - 1][0] * q[i][1] for i in range(len(q))) < 0: q = q[::-1]   # clockwise, as the traced shapes
    return [q]

def table(style, swap_ab, swap_xy):
    # game button -> the position of the physical button that sends it (FeTheme.face), then that position's picture in the style's pack
    pos = {'A': 'east', 'B': 'south', 'X': 'north', 'Y': 'west'} if style == 'ns' else {'A': 'south', 'B': 'east', 'X': 'west', 'Y': 'north'}
    if swap_ab: pos['A'], pos['B'] = pos['B'], pos['A']
    if swap_xy: pos['X'], pos['Y'] = pos['Y'], pos['X']
    t = [('btn' + b, [0xE0A0 + i, 0xE0E0 + i], '%s_face_%s' % (style, pos[b])) for i, b in enumerate('ABXY')]
    for i, (b, f) in enumerate([('L', 'bumper_left'), ('R', 'bumper_right'), ('ZL', 'trigger_left'), ('ZR', 'trigger_right')]): t.append(('btn' + b, [0xE0A4 + i, 0xE0E4 + i], '%s_%s' % (style, f)))
    t += [('btnSL', [0xE0A8, 0xE0E8], 'ns_joycon_sl'), ('btnSR', [0xE0A9, 0xE0E9], 'ns_joycon_sr')]   # Joy-Con rail buttons: only the Switch pack has them
    t += [('btnPlus', [0xE0B5, 0xE0EF], style + '_start'), ('btnMinus', [0xE0B6, 0xE0F0], style + '_select'), ('dpad', [0xE0EA], style + '_dpad_all')]
    t += [('dpad_' + d, [c, c + 0x40], '%s_dpad_%s' % (style, d)) for c, d in [(0xE0AB, 'right'), (0xE0AC, 'down'), (0xE0AD, 'up'), (0xE0AE, 'left')]]
    t += [('stick', [0xE100], style + '_stick_left'), ('stickL', [0xE101, 0xE0C1], style + '_stick_left'), ('stickR', [0xE102, 0xE0C2], style + '_stick_right')]
    return t

def build(icons, style, out, swap_ab=False, swap_xy=False):
    glyphs, adv, cmap = {}, {}, {}
    def add(name, cps, loops, a=1000):
        pen = TTGlyphPen(None)
        for l in loops:
            q = [(int(round(x)), int(round(y))) for x, y in l]; q = [p for i, p in enumerate(q) if p != q[i - 1]]
            if len(q) < 3: continue
            pen.moveTo(q[0])
            for p in q[1:]: pen.lineTo(p)
            pen.closePath()
        glyphs[name] = pen.glyph(); adv[name] = a
        for c in cps: cmap[c] = name
    glyphs['.notdef'] = TTGlyphPen(None).glyph(); adv['.notdef'] = 500
    def box(loops): xs = [x for l in loops for x, _ in l]; ys = [y for l in loops for _, y in l]; return min(xs), min(ys), max(xs), max(ys)
    x0, y0, x1, y1 = box(trace(os.path.join(icons, 'ctl_%s_face_south.png' % style))[2]); k0 = FACE / (y1 - y0)
    cache = {}
    for name, cps, pic in table(style, swap_ab, swap_xy):
        if pic not in cache:
            loops = trace(os.path.join(icons, 'ctl_%s.png' % pic))[2]; x0, y0, x1, y1 = box(loops); k = min(k0, FACE / (y1 - y0), MAXW / (x1 - x0))
            a = max(1000, int(round((x1 - x0) * k)) + 100); mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            cache[pic] = (a, [[(a / 2 + (x - mx) * k, CY - (y - my) * k) for x, y in l] for l in loops])
        add(name, cps, cache[pic][1], cache[pic][0])
    for c, (nm, deg) in zip(range(0xE090, 0xE098), [('right', 0), ('left', 180), ('up', 90), ('down', 270), ('rightup', 45), ('rightdown', 315), ('leftdown', 225), ('leftup', 135)]):
        add('arrow_' + nm, [c], arrow(deg))
    order = ['.notdef'] + sorted(n for n in glyphs if n != '.notdef')
    fb = FontBuilder(EM, isTTF=True); fb.setupGlyphOrder(order); fb.setupCharacterMap(cmap); fb.setupGlyf(glyphs)
    glyf = fb.font['glyf']
    for n in order: glyf[n].recalcBounds(glyf)
    fb.setupHorizontalMetrics({n: (adv[n], getattr(glyf[n], 'xMin', 0)) for n in order}); fb.setupHorizontalHeader(ascent=ASC, descent=DESC)
    fb.setupNameTable({'familyName': 'DM Nintendo Extension Stand-in (%s icons)' % style, 'styleName': 'Regular'})
    fb.setupOS2(sTypoAscender=ASC, sTypoDescender=DESC, sTypoLineGap=0, usWinAscent=ASC, usWinDescent=-DESC); fb.setupPost(); fb.save(out)
    print('%s: %d glyphs, %d code points, %d bytes' % (out, len(order), len(cmap), os.path.getsize(out)))

def sheet(fonts, out, cell=72):
    # what the fonts hold, drawn from their outlines (even-odd fill, 3x3 samples): a row per font, white on the game's dark plaque colour
    names = ['btnA', 'btnB', 'btnX', 'btnY', 'btnL', 'btnR', 'btnZL', 'btnZR', 'btnPlus', 'btnMinus', 'dpad', 'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right', 'stickL', 'stickR', 'btnSL', 'btnSR', 'arrow_right', 'arrow_leftup']
    W, H = cell * len(names), cell * len(fonts); img = bytearray(b'\x18\x20\x28' * (W * H)); S = 3; k = (cell - 8) / 1350.0
    for r, fp in enumerate(fonts):
        f = TTFont(fp); glyf = f['glyf']
        for ci, n in enumerate(names):
            g = glyf[n]
            if g.numberOfContours <= 0: continue
            co, ends, _ = g.getCoordinates(glyf); loops = []; s = 0
            for e in ends: loops.append([co[i] for i in range(s, e + 1)]); s = e + 1
            cov = [[0] * cell for _ in range(cell)]
            for sy in range(cell * S):
                fy = (sy + 0.5) / S; gy = (cell / 2 - fy) / k + 350   # the icons' centre line (350) at the cell's middle
                xs = []
                for l in loops:
                    for i in range(len(l)):
                        (x1, y1), (x2, y2) = l[i - 1], l[i]
                        if (y1 <= gy) != (y2 <= gy): xs.append(x1 + (gy - y1) * (x2 - x1) / (y2 - y1))
                xs.sort()
                for j in range(0, len(xs) - 1, 2):
                    a, b = (xs[j] - f['hmtx'][n][0] / 2) * k + cell / 2, (xs[j + 1] - f['hmtx'][n][0] / 2) * k + cell / 2
                    for sx in range(max(0, int(a * S)), min(cell * S, int(b * S) + 1)):
                        if a <= (sx + 0.5) / S < b: cov[sy // S][sx // S] += 1
            for y in range(cell):
                for x in range(cell):
                    c = cov[y][x] / (S * S)
                    if c: o = ((r * cell + y) * W + ci * cell + x) * 3; img[o:o + 3] = bytes(int(v + (255 - v) * c) for v in img[o:o + 3])
    raw = b''.join(b'\0' + bytes(img[y * W * 3:(y + 1) * W * 3]) for y in range(H))
    def ch(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    open(out, 'wb').write(b'\x89PNG\r\n\x1a\n' + ch(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) + ch(b'IDAT', zlib.compress(raw)) + ch(b'IEND', b''))
    print('%s: %d fonts, %d glyphs each' % (out, len(fonts), len(names)))

if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == '--sheet': sheet(a[2:], a[1]); sys.exit(0)
    if len(a) < 3 or a[1] not in ('ns', 'xb', 'ps'): sys.exit('usage: make_ext_font_icons.py <icon folder> <ns|xb|ps> out.ttf [--swap-ab] [--swap-xy]\n       make_ext_font_icons.py --sheet out.png font.ttf...')
    build(a[0], a[1], a[2], '--swap-ab' in a, '--swap-xy' in a)
