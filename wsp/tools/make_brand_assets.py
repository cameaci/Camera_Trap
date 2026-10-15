#!/usr/bin/env python3
"""
Generate every WSP CameraTrap image asset from the WSP logo.

Source: wsp/brand/wsp-logo-red.svg (the WSP logo, WSP red #FF372F).
Needs Pillow and cairosvg:  pip install pillow cairosvg

    python wsp/tools/make_brand_assets.py

Writes the app icon (electron/build), the in-app logos, favicons and the
home / setup backgrounds (frontend/public). Re-run after changing the logo.
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
LOGO_SVG = REPO / "wsp" / "brand" / "wsp-logo-red.svg"
PUBLIC = REPO / "frontend" / "public"
ELECTRON_BUILD = REPO / "electron" / "build"

WSP_RED = (255, 55, 47)
INK = (20, 18, 18)          # WSP black
GREY05 = (243, 243, 243)
GREY10 = (232, 231, 231)
WHITE = (255, 255, 255)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def logo(width: int) -> Image.Image:
    """The WSP logo rendered at `width` pixels, transparent background."""
    png = cairosvg.svg2png(url=str(LOGO_SVG), output_width=width)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("No bold sans-serif font found")


def centered(canvas: Image.Image, item: Image.Image, dy: int = 0) -> None:
    x = (canvas.width - item.width) // 2
    y = (canvas.height - item.height) // 2 + dy
    canvas.alpha_composite(item, (x, y))


def app_icon(size: int) -> Image.Image:
    """WSP-red rounded tile with the logo in white: the Windows app icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=round(size * 0.18), fill=WSP_RED
    )
    mark = logo(round(size * 0.72))
    white = Image.new("RGBA", mark.size, (*WHITE, 0))
    white.putalpha(mark.getchannel("A"))
    centered(img, white, dy=round(size * 0.06))
    return img


def logo_mark(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    centered(img, logo(round(size * 0.86)))
    return img


def wordmark() -> Image.Image:
    """WSP logo with "CameraTrap" beside it, dark ink, transparent."""
    height = 632
    mark = logo(1040)                      # 150 x 71.3 aspect: ~494 px high
    text = "CameraTrap"
    f = font(300)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=f)
    gap = 90
    width = mark.width + gap + (right - left) + 40
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    img.alpha_composite(mark, (20, (height - mark.height) // 2))
    draw = ImageDraw.Draw(img)
    # Baseline of the text on the baseline of the logo's letters.
    text_y = (height - mark.height) // 2 + round(mark.height * 0.54) - bottom
    draw.text((20 + mark.width + gap - left, text_y), text, font=f, fill=INK)
    return img


def background(width: int, height: int) -> Image.Image:
    """Soft WSP grey with a large, faint red WSP mark: calm behind glass panels."""
    img = Image.new("RGB", (width, height), GREY05)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(height - 1, 1)
        c = tuple(round(a + (b - a) * t) for a, b in zip(GREY05, GREY10))
        draw.line((0, y, width, y), fill=c)
    mark = logo(round(width * 0.5))
    faint = Image.new("RGBA", mark.size, (0, 0, 0, 0))
    faint.paste((*WSP_RED, 26), (0, 0), mark)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    layer.alpha_composite(
        faint, (round(width * 0.46), round(height * 0.92) - mark.height)
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(2, width // 400)))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def main() -> None:
    (PUBLIC / "branding").mkdir(parents=True, exist_ok=True)
    (ELECTRON_BUILD / "icons").mkdir(parents=True, exist_ok=True)

    logo_mark().save(PUBLIC / "branding" / "logo-mark.png")
    wordmark().save(PUBLIC / "branding" / "logo-wordmark.png")

    icon_1024 = app_icon(1024)
    icon_1024.save(ELECTRON_BUILD / "icon.png")
    for size in (16, 32, 48, 64, 128, 256, 512, 1024):
        app_icon(size).save(ELECTRON_BUILD / "icons" / f"{size}x{size}.png")
    app_icon(256).save(
        ELECTRON_BUILD / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

    app_icon(64).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    app_icon(32).save(PUBLIC / "favicon-32x32.png")
    touch = Image.new("RGBA", (180, 180), WHITE)
    centered(touch, logo(140))
    touch.convert("RGB").save(PUBLIC / "apple-touch-icon.png")

    background(2560, 1706).save(PUBLIC / "home-background.webp", quality=90)
    background(1024, 683).save(PUBLIC / "setup-background.webp", quality=90)
    print("Brand assets written.")


if __name__ == "__main__":
    main()
