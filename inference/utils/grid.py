"""Compose a 2D matrix of equal-length frame lists into a single labeled tiled mp4."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .video_io import write_video

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_grid_video(
    cells: list[list[list[Image.Image]]],
    out_path: Path,
    *,
    fps: float,
    col_labels: list[str],
    row_labels: list[str],
    title: str | None = None,
    col_sublabels: list[str] | None = None,
    cell_w: int = 320,
    cell_h: int = 184,
    title_h: int = 24,
    header_h: int = 48,
    label_w: int = 96,
    bg_color: tuple[int, int, int] = (16, 16, 16),
    text_color: tuple[int, int, int] = (240, 240, 240),
    sub_text_color: tuple[int, int, int] = (170, 170, 170),
) -> None:
    """Tile `cells[row][col]` into a single labeled mp4.

    Layout (top to bottom):
      title strip (title_h, optional)  -- global small title (model/steps/guidance)
      column header strip (header_h)   -- col_labels + optional col_sublabels (cond values)
      n_rows of cells, each cell_h tall, with row labels in the left label_w column.

    All cell frame lists must be equal length.
    """
    if not cells or not cells[0]:
        raise ValueError("cells must be a non-empty 2D matrix")
    n_rows = len(cells)
    n_cols = len(cells[0])
    if len(row_labels) != n_rows:
        raise ValueError(f"row_labels has {len(row_labels)} entries, need {n_rows}")
    if len(col_labels) != n_cols:
        raise ValueError(f"col_labels has {len(col_labels)} entries, need {n_cols}")
    if col_sublabels is not None and len(col_sublabels) != n_cols:
        raise ValueError(
            f"col_sublabels has {len(col_sublabels)} entries, need {n_cols}"
        )

    n_frames = len(cells[0][0])
    for r, row in enumerate(cells):
        if len(row) != n_cols:
            raise ValueError(f"row {r} has {len(row)} cells, expected {n_cols}")
        for c, frames in enumerate(row):
            if len(frames) != n_frames:
                raise ValueError(
                    f"cell [{r}][{c}] has {len(frames)} frames, expected {n_frames}"
                )

    title_strip_h = title_h if title else 0
    canvas_w = label_w + n_cols * cell_w
    canvas_h = title_strip_h + header_h + n_rows * cell_h

    overlay = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(overlay)

    title_font = _load_font(max(11, title_h - 8))
    header_font = _load_font(max(14, header_h // 2))
    sub_font = _load_font(max(11, header_h // 4))
    row_font = _load_font(max(12, min(18, label_w // 7)))

    if title:
        draw.text(
            (canvas_w // 2, title_strip_h // 2),
            title,
            fill=sub_text_color,
            font=title_font,
            anchor="mm",
        )

    for c, label in enumerate(col_labels):
        cx = label_w + c * cell_w + cell_w // 2
        sub = col_sublabels[c] if col_sublabels else None
        if sub:
            draw.text(
                (cx, title_strip_h + header_h // 2 - header_h // 5),
                label,
                fill=text_color,
                font=header_font,
                anchor="mm",
            )
            draw.text(
                (cx, title_strip_h + header_h // 2 + header_h // 4),
                sub,
                fill=sub_text_color,
                font=sub_font,
                anchor="mm",
            )
        else:
            draw.text(
                (cx, title_strip_h + header_h // 2),
                label,
                fill=text_color,
                font=header_font,
                anchor="mm",
            )

    for r, label in enumerate(row_labels):
        cx = label_w // 2
        cy = title_strip_h + header_h + r * cell_h + cell_h // 2
        draw.text((cx, cy), label, fill=text_color, font=row_font, anchor="mm")

    composed: list[Image.Image] = []
    for t in range(n_frames):
        canvas = overlay.copy()
        for r in range(n_rows):
            for c in range(n_cols):
                tile = cells[r][c][t]
                if tile.size != (cell_w, cell_h):
                    tile = tile.resize((cell_w, cell_h), Image.BICUBIC)
                canvas.paste(
                    tile,
                    (label_w + c * cell_w,
                     title_strip_h + header_h + r * cell_h),
                )
        composed.append(canvas)

    write_video(composed, out_path, fps=fps)
