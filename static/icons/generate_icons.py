#!/usr/bin/env python3
"""
Generate app icons for Eenkaartjeleggen - Klaverjassen.

Usage (from repo root):
    pip install Pillow
    python static/icons/generate_icons.py

Output: static/icons/icon-192.png
                      icon-512.png
                      icon-maskable-512.png
                      icon-180.png
"""

import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ── Colours (from static/style.css) ───────────────────────────────────────────
BG     = (26,  58,  26, 255)   # #1a3a1a  table background
FELT   = (46, 125,  50, 255)   # #2e7d32  felt green
CARD   = (255, 254, 240, 255)  # #fffef0  card face
BLACK  = ( 17,  17,  17, 255)  # #111111  black suit
SHADOW = (  0,   0,   0,  55)

# ── Font search (for ♣ and corner "J") ────────────────────────────────────────
_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/seguisym.ttf",   # Segoe UI Symbol — best ♣ glyph
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Symbol.ttf",
    "/Library/Fonts/Arial.ttf",
]

def _load_font(size: int) -> tuple:
    """Return (font, is_truetype). Falls back to PIL default bitmap font."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size), True
        except Exception:
            continue
    return ImageFont.load_default(), False


def _text_center(draw, cx: int, cy: int, text: str, font, color):
    """Draw text centred at (cx, cy) for both TTF and bitmap fonts."""
    try:
        draw.text((cx, cy), text, font=font, fill=color, anchor="mm")
    except (TypeError, AttributeError):
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - w // 2, cy - h // 2), text, font=font, fill=color)


# ── Drawing helpers ────────────────────────────────────────────────────────────
def _rounded_rect(draw, x0, y0, x1, y1, r, fill):
    r = max(r, 0)
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r),
                   (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def _draw_club_primitive(draw, cx: int, cy: int, size: int, color):
    """Draw a ♣ symbol using circles and a rectangle (no font needed)."""
    r = int(size * 0.22)
    draw.ellipse([cx - r, cy - int(size * 0.16) - r,
                  cx + r, cy - int(size * 0.16) + r], fill=color)
    blx, bly = cx - int(size * 0.19), cy + int(size * 0.09)
    draw.ellipse([blx - r, bly - r, blx + r, bly + r], fill=color)
    brx, bry = cx + int(size * 0.19), cy + int(size * 0.09)
    draw.ellipse([brx - r, bry - r, brx + r, bry + r], fill=color)
    sw = int(size * 0.09)
    draw.rectangle([cx - sw, cy + int(size * 0.26),
                    cx + sw, cy + int(size * 0.46)], fill=color)
    bw = int(size * 0.27)
    draw.rectangle([cx - bw, cy + int(size * 0.40),
                    cx + bw, cy + int(size * 0.46)], fill=color)


# ── Icon renderer ──────────────────────────────────────────────────────────────
def generate_icon(size: int, maskable: bool = False) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    # Rounded square background
    bg_r = int(size * 0.18)
    _rounded_rect(draw, 0, 0, size, size, bg_r, BG)

    # Radial felt highlight
    steps = 24
    for i in range(steps, 0, -1):
        frac = i / steps
        grad_r = int(size * 0.52 * frac)
        alpha = int(60 * frac * (1 - frac) * 4)
        draw.ellipse([cx - grad_r, cy - grad_r, cx + grad_r, cy + grad_r],
                     fill=(46, 125, 50, alpha))

    scale = 0.76 if maskable else 1.0
    card_w = int(size * 0.60 * scale)
    card_h = int(size * 0.82 * scale)
    x0, y0 = cx - card_w // 2, cy - card_h // 2
    x1, y1 = x0 + card_w, y0 + card_h
    card_r = max(int(size * 0.04), 4)

    so = max(int(size * 0.025), 2)
    _rounded_rect(draw, x0 + so, y0 + so, x1 + so, y1 + so, card_r, SHADOW)
    _rounded_rect(draw, x0, y0, x1, y1, card_r, CARD)

    border = (210, 205, 185, 180)
    _rounded_rect(draw, x0 + 2, y0 + 2, x1 - 2, y1 - 2, card_r, border)
    _rounded_rect(draw, x0 + 4, y0 + 4, x1 - 4, y1 - 4, card_r, CARD)

    # Club symbol
    club_size = int(card_w * 0.68)
    sym_cy = cy - int(card_h * 0.04)
    font, is_ttf = _load_font(int(club_size * 0.88))
    if is_ttf:
        try:
            _text_center(draw, cx, sym_cy, "♣", font, BLACK)
        except Exception:
            _draw_club_primitive(draw, cx, sym_cy, club_size, BLACK)
    else:
        _draw_club_primitive(draw, cx, sym_cy, club_size, BLACK)

    # Corner J♣ marker
    corner_size = max(int(size * 0.075 * scale), 10)
    corner_font, corner_ttf = _load_font(corner_size)
    margin = int(size * 0.07 * scale)
    if corner_ttf and size >= 180:
        try:
            draw.text((x0 + margin, y0 + margin),
                      "J", font=corner_font, fill=BLACK, anchor="lt")
            draw.text((x0 + margin, y0 + margin + corner_size + 2),
                      "♣", font=corner_font, fill=BLACK, anchor="lt")
        except Exception:
            pass

    return img


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    specs = [
        ("icon-192.png",          192, False),
        ("icon-512.png",          512, False),
        ("icon-maskable-512.png", 512, True),
        ("icon-180.png",          180, False),
    ]

    print("Generating icons...")
    for filename, size, maskable in specs:
        img = generate_icon(size, maskable=maskable)
        img.save(os.path.join(output_dir, filename), "PNG")
        print(f"  OK  {filename:30s} ({size}x{size})")

    print(f"\nSaved to: {os.path.normpath(output_dir)}")


if __name__ == "__main__":
    main()
