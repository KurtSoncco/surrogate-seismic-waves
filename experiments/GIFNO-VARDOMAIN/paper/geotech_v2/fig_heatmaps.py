#!/usr/bin/env python3
"""500 m exemplars already on disk. This script does not run a forward.

The stored dipping and blind images are 500 m windows. Their captions carried
a median mid-band skill and a Pearson of linear |TF|. Those reductions are not
in the frozen ladder, so the caption is replaced with the selection rule.
The spatial fields are the stored image and are not redrawn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "02_tf_heatmaps_and_skill"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

CAPTIONS = {
    "dipping.png": (
        "500 m exemplar, dipping test run 217. Selection: the held-out dipping record with the "
        "largest center-station mid-band column residual. High-signal example, not a representative "
        "record and not wide-domain evidence. This panel is not a pooled score."
    ),
    "realistic.png": (
        "500 m exemplar, blind section man_0297. Selection: the blind section with the largest stored "
        "mid-band |g| in the ensemble dump. High-signal example, not a representative section and not "
        "wide-domain evidence. This panel is not a pooled score."
    ),
}


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _content_end(ink: np.ndarray) -> int:
    """Row where the bottom caption starts, so a second run does not stack captions."""

    gaps: list[tuple[int, int]] = []
    start = None
    for index, hit in enumerate(ink):
        if not hit and start is None:
            start = index
        if hit and start is not None:
            gaps.append((start, index - start))
            start = None
    if start is not None:
        gaps.append((start, len(ink) - start))
    bottom = [row for row, width in gaps if width >= 60 and row > 0.85 * len(ink)]
    if not bottom:
        return len(ink)
    return min(bottom) + 16


def _relabel(path: Path, caption: str) -> None:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    ink = (~np.all(rgb > 245, axis=2)).any(axis=1)
    image = image.crop((0, 0, image.width, _content_end(ink)))
    font_size = max(28, image.width // 90)
    font = ImageFont.truetype(str(FONT), font_size) if FONT.is_file() else ImageFont.load_default()
    scratch = ImageDraw.Draw(image)
    lines = _wrap(scratch, caption, font, image.width - 2 * font_size)
    line_h = int(font_size * 1.35)
    gap = 80
    footer_h = gap + line_h * len(lines) + line_h // 2
    canvas = Image.new("RGB", (image.width, image.height + footer_h), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + gap
    for line in lines:
        draw.text((font_size // 2, y), line, fill="black", font=font)
        y += line_h
    canvas.save(path)
    fig = plt.figure(figsize=(7.48, 7.48 * canvas.height / canvas.width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, caption in CAPTIONS.items():
        path = OUT / name
        if not path.is_file():
            raise SystemExit(f"missing stored exemplar {path}; a forward would be required to draw it")
        _relabel(path, caption)
        print(f"relabeled {path}")
    (OUT / "heatmap_summary.csv").write_text(
        "case,sample_id,width_m,selection_rule\n"
        "dipping,run_217,500,held-out dipping test record with the largest center-station mid-band column residual; high-signal exemplar\n"
        "realistic,man_0297,500,blind manifold section with the largest stored mid-band absolute g in the ensemble dump; high-signal exemplar\n"
    )
    (OUT / "selection_rule.txt").write_text(
        "Dipping test run 217: 500 m held-out window, the record with the largest center-station "
        "mid-band column residual. High-signal exemplar, not a representative record and not wide-domain evidence.\n"
        "Blind section man_0297: 500 m manifold section, the blind section with the largest stored mid-band "
        "|g| in the ensemble dump. High-signal exemplar, not a representative section and not wide-domain evidence.\n"
        "Pooled scores are in analysis/frozen/ladder.csv. They are not printed on these panels.\n"
    )


if __name__ == "__main__":
    main()
