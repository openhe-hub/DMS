#!/usr/bin/env python3
"""Qualitative grid figure: Source | MimicMotion | DisPose+SIREN, 2x4 clips.

Rebuilds the mm_failures_grid.png protocol (originally an ad-hoc huawei session
script, frames hand-picked 2026-07-05) with the SIREN system as the third
panel. Frame indices are SOURCE-video frame numbers; MimicMotion raw outputs
carry one leading padding frame, so the MM panel reads frame n+1, while
DisPose/SIREN outputs are frame-for-frame aligned (internal reference frame is
dropped on save).

Inputs (relative to repo root):
  assets/example_data/sign_videos/hard27k_orig/{id}.mp4      source, 640x360
  outputs/sign_cmp_hard27k/raw/mimicmotion/{id}_hiya.mp4     MM raw, 576x576
  outputs/sign_siren_best/best/*_to_{id}_*.mp4               SIREN best-of-N, 576x576
Output: outputs/sign_cmp_hard27k/figs/mm_vs_siren_grid.png
"""
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/sign_cmp_hard27k"
PANEL = 576          # working panel size; tile is downscaled 2x at the end
PAD, MARGIN = 8, 8   # grid spacing (at final scale), matches original figure

# (clip id, source frame index, gloss). Clips drawn from the 15 hard27k pilot
# clips (12 usable: podia's SIREN mp4 is corrupt, 2 clips lack gloss labels).
# Frames picked 2026-07-11, v2 protocol: candidates = per-clip local minima of
# the SIREN video's frame-difference motion (holds -> no motion blur), then
# eyeballed on Source/MM/SIREN contact sheets for max "MM broken, SIREN crisp
# handshape" contrast at the SAME source instant.
SPECS = [
    ("0bsujxxpwd", 446, "vulcanise"),      # MM text burst; SIREN crisp V-hand
    ("03os6hy28y", 15, "deposit"),         # MM mangles double claw; SIREN bent fingers separated
    ("0gjpljgpdj", 33, "choosey"),         # MM fingers tangled; SIREN clean thumb-hands
    ("0byrxo0heb", 44, "open book"),       # MM finger smears + blue bg tint; SIREN spread-over-palm
    ("0db3uk2cqw", 161, "backlight"),      # MM hallucinated yellow object; SIREN point-at-palm
    ("0bcxsenqga", 117, "hump"),           # MM fuses fingers into slab; SIREN flat palm, fingers clear
    ("0ihmqp5iz6", 59, "turn off (tv)"),   # MM blob claw on wrist; SIREN L-hand on wrist
    ("0hwhrmyqqx", 98, "mobilisation"),    # MM claw blob; SIREN crisp OK-ring hand
]

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    sys.exit("no usable bold TTF font found")


def grab(video, frame_idx, vf, out_png):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", f"select=eq(n\\,{frame_idx}),{vf}", "-frames:v", "1",
         str(out_png)],
        check=True,
    )
    return Image.open(out_png).convert("RGB")


def chip(img, text, f):
    d = ImageDraw.Draw(img, "RGBA")
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    w, h = r - l, b - t
    x = (img.width - w) // 2
    d.rectangle([x - 8, 10, x + w + 8, 10 + h + 16], fill=(0, 0, 0, 140))
    d.text((x, 18 - t), text, font=f, fill="white")


def word_label(img, text, f):
    d = ImageDraw.Draw(img, "RGBA")
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    w, h = r - l, b - t
    y = img.height - h - 20
    d.rectangle([8, y - 6, 8 + w + 12, y + h + 6], fill=(0, 0, 0, 153))
    d.text((14 - l, y - t), text, font=f, fill="white")


def main():
    chip_f, word_f = font(28), font(40)
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (cid, fr, word) in enumerate(SPECS):
            src = ROOT / f"assets/example_data/sign_videos/hard27k_orig/{cid}.mp4"
            mm = OUT / f"raw/mimicmotion/{cid}_hiya.mp4"
            (siren,) = glob.glob(str(ROOT / f"outputs/sign_siren_best/best/*_to_{cid}_*.mp4"))
            sq = f"scale={PANEL}:{PANEL}"
            panels = [
                ("Source", grab(src, fr, f"crop=360:360:140:0,{sq}", f"{tmp}/{i}a.png")),
                ("MimicMotion", grab(mm, fr + 1, sq, f"{tmp}/{i}b.png")),
                ("DisPose+SIREN", grab(siren, fr, sq, f"{tmp}/{i}c.png")),
            ]
            tile = Image.new("RGB", (PANEL * 3, PANEL))
            for k, (label, img) in enumerate(panels):
                chip(img, label, chip_f)
                tile.paste(img, (k * PANEL, 0))
            word_label(tile, word, word_f)
            tiles.append(tile.resize((PANEL * 3 // 2, PANEL // 2), Image.LANCZOS))

    tw, th = tiles[0].size
    grid = Image.new(
        "RGB",
        (2 * tw + PAD + 2 * MARGIN, 4 * th + 3 * PAD + 2 * MARGIN),
        "white",
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 2)
        grid.paste(tile, (MARGIN + c * (tw + PAD), MARGIN + r * (th + PAD)))
    dst = OUT / "figs/mm_vs_siren_grid.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dst)
    print(dst, grid.size)


if __name__ == "__main__":
    main()
