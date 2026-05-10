#!/usr/bin/env python3
"""
Hậu xử lý manga: tách panel từ ảnh trang manga (scene_XX.png) và xuất JSON storyText theo panel.

Yêu cầu:
  - Trang manga phải là lưới panel chữ nhật (L→R, T→D) như pipeline đã prompt.
  - Có `scenes.json` có trường `panelCount` và `panels[][{panelNumber, storyText}]`.

Ví dụ:
  python postprocess_split_panels.py --input ./out6 --output ./out6/panels
  python postprocess_split_panels.py --input ./out6 --output ./out6/panels --grid 2x2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int


def _load_scenes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("scenes.json phải là mảng JSON.")
    return data


def _factor_grids(n: int) -> list[Grid]:
    out: list[Grid] = []
    for r in range(1, n + 1):
        if n % r == 0:
            out.append(Grid(rows=r, cols=n // r))
    return out


def _choose_grid(n_panels: int, w: int, h: int) -> Grid:
    """
    Chọn lưới rows×cols cho n_panels.
    Heuristic: tối ưu cell_aspect ~= 1.3 (panel hơi cao) và ưu tiên rows>=cols cho trang portrait.
    """
    candidates = _factor_grids(n_panels)
    if not candidates:
        return Grid(rows=1, cols=1)

    page_aspect = h / max(1, w)
    target_cell_aspect = 1.25 if page_aspect >= 1.2 else 0.9

    best = candidates[0]
    best_score = 1e9
    for g in candidates:
        cell_w = w / g.cols
        cell_h = h / g.rows
        cell_aspect = cell_h / max(1e-6, cell_w)
        score = abs(cell_aspect - target_cell_aspect) + 0.08 * abs(g.rows - g.cols)
        # Với trang dọc, ưu tiên nhiều hàng hơn hoặc bằng cột
        if page_aspect >= 1.2 and g.rows < g.cols:
            score += 0.2
        if score < best_score:
            best_score = score
            best = g
    return best


def _parse_grid_arg(s: str) -> Grid:
    s = s.strip().lower()
    if "x" not in s:
        raise ValueError("grid phải dạng RxC, ví dụ 2x3")
    r, c = s.split("x", 1)
    rows = int(r)
    cols = int(c)
    if rows < 1 or cols < 1:
        raise ValueError("grid rows/cols phải >= 1")
    return Grid(rows=rows, cols=cols)


def _crop_with_margin(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, margin_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x0 + margin_px)
    y0 = max(0, y0 + margin_px)
    x1 = min(w, x1 - margin_px)
    y1 = min(h, y1 - margin_px)
    if x1 <= x0 or y1 <= y0:
        return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()
    return img[y0:y1, x0:x1].copy()

def _trim_background_border(
    img_bgr: np.ndarray,
    *,
    sample_px: int = 8,
    tol: int = 18,
    min_keep_ratio: float = 0.35,
) -> np.ndarray:
    """
    Trim sát vùng panel bằng cách coi màu nền ở 4 góc là "background" (gutter),
    rồi lấy bounding box của pixels khác background.

    - sample_px: kích thước ô lấy mẫu ở mỗi góc
    - tol: ngưỡng khác màu (0-255), càng lớn càng ít trim
    - min_keep_ratio: nếu vùng giữ lại quá nhỏ so với ảnh gốc => bỏ trim để tránh crop lỗi
    """
    h, w = img_bgr.shape[:2]
    sp = max(2, min(sample_px, h // 4, w // 4))

    corners = [
        img_bgr[0:sp, 0:sp],  # TL
        img_bgr[0:sp, w - sp : w],  # TR
        img_bgr[h - sp : h, 0:sp],  # BL
        img_bgr[h - sp : h, w - sp : w],  # BR
    ]
    corner_means = np.stack(
        [np.mean(c.reshape(-1, 3), axis=0) for c in corners], axis=0
    ).astype(np.float32)

    # Nhiều trang có 2 màu nền (ví dụ: trên xanh, dưới trắng).
    # Dùng KMeans 2 cụm để lấy 1-2 màu background và coi pixel gần bất kỳ màu nào là "background".
    bg_colors = [corner_means.mean(axis=0)]
    if np.max(np.linalg.norm(corner_means - corner_means.mean(axis=0), axis=1)) > 25:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        _compactness, _labels, centers = cv2.kmeans(
            corner_means, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        bg_colors = [centers[0], centers[1]]

    img_f = img_bgr.astype(np.float32)
    diffs = []
    for bg in bg_colors:
        bgv = bg.reshape(1, 1, 3)
        diffs.append(np.max(np.abs(img_f - bgv), axis=2))
    diff = np.min(np.stack(diffs, axis=0), axis=0)  # gần 1 trong các bg => background
    mask = (diff > tol).astype(np.uint8) * 255

    # Khử nhiễu nhẹ
    k = max(3, (min(h, w) // 120) * 2 + 1)  # odd
    mask = cv2.medianBlur(mask, k)

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return img_bgr

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    keep_w = x1 - x0
    keep_h = y1 - y0
    if keep_w < 2 or keep_h < 2:
        return img_bgr

    if (keep_w * keep_h) < (min_keep_ratio * w * h):
        return img_bgr

    return img_bgr[y0:y1, x0:x1].copy()


def split_page_into_grid_panels(
    image_bgr: np.ndarray,
    grid: Grid,
    *,
    gutter_margin_ratio: float = 0.01,
    trim_fit: bool = True,
    trim_tol: int = 18,
) -> list[np.ndarray]:
    """
    Cắt ảnh theo lưới đều.
    - gutter_margin_ratio: cắt vào một chút để tránh dính gutter (tùy trang).
    - trim_fit: nếu bật, trim sát khung panel bằng màu nền 4 góc (auto-fit vào các góc).
    """
    h, w = image_bgr.shape[:2]
    margin_px = int(round(min(w, h) * gutter_margin_ratio))
    panels: list[np.ndarray] = []
    for r in range(grid.rows):
        y0 = int(round(r * h / grid.rows))
        y1 = int(round((r + 1) * h / grid.rows))
        for c in range(grid.cols):
            x0 = int(round(c * w / grid.cols))
            x1 = int(round((c + 1) * w / grid.cols))
            cell = _crop_with_margin(image_bgr, x0, y0, x1, y1, margin_px)
            if trim_fit:
                cell = _trim_background_border(cell, tol=trim_tol)
            panels.append(cell)
    return panels


def _panel_story_texts(scene_row: dict[str, Any], n_panels: int) -> list[str]:
    panels = scene_row.get("panels")
    if isinstance(panels, list) and panels:
        # sort by panelNumber
        ordered = []
        for p in panels:
            if not isinstance(p, dict):
                continue
            try:
                pn = int(p.get("panelNumber", 0))
            except (TypeError, ValueError):
                pn = 0
            ordered.append((pn, str(p.get("storyText") or "")))
        ordered.sort(key=lambda t: t[0])
        texts = [t[1] for t in ordered][:n_panels]
        if len(texts) == n_panels:
            return texts

    # Fallback: chia đều storyText cả trang
    whole = str(scene_row.get("storyText") or "").strip()
    if not whole:
        return [""] * n_panels
    step = max(1, math.floor(len(whole) / n_panels))
    out = []
    pos = 0
    for i in range(n_panels):
        if i == n_panels - 1:
            out.append(whole[pos:].strip())
        else:
            out.append(whole[pos:pos + step].strip())
            pos += step
    return out


def _find_page_image(input_dir: Path, page_number: int) -> Path:
    # ưu tiên png, sau đó jpg/webp
    stem = f"scene_{page_number:02d}"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = input_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Không tìm thấy ảnh cho page {page_number}: {stem}.(png/jpg/webp)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tách panel từ trang manga và xuất JSON storyText theo panel.")
    ap.add_argument("--input", type=Path, required=True, help="Thư mục chứa scenes.json và scene_XX.png")
    ap.add_argument("--output", type=Path, required=True, help="Thư mục output panels/")
    ap.add_argument("--grid", type=str, default=None, help="Ép lưới RxC cho mọi trang, ví dụ 2x3")
    ap.add_argument("--margin", type=float, default=0.01, help="Tỉ lệ margin cắt bỏ gutter (0.0-0.05).")
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="Tắt auto-fit (trim) để giữ nguyên crop theo lưới.",
    )
    ap.add_argument(
        "--fit-tol",
        type=int,
        default=18,
        help="Ngưỡng khác màu để auto-fit (càng lớn càng ít trim).",
    )
    args = ap.parse_args()

    input_dir = args.input.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_path = input_dir / "scenes.json"
    scenes = _load_scenes(scenes_path)

    forced_grid = _parse_grid_arg(args.grid) if args.grid else None

    index: list[dict[str, Any]] = []

    for scene in scenes:
        mode = scene.get("mode")
        if mode != "manga":
            continue
        page_number = int(scene.get("pageNumber"))
        panel_count = int(scene.get("panelCount", 1))
        if panel_count < 1:
            panel_count = 1

        img_path = _find_page_image(input_dir, page_number)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"OpenCV không đọc được ảnh: {img_path}")
        h, w = img.shape[:2]

        grid = forced_grid if forced_grid else _choose_grid(panel_count, w, h)
        if grid.rows * grid.cols != panel_count:
            # nếu panelCount không factor được, fallback 1×N
            grid = Grid(rows=1, cols=panel_count)

        panels = split_page_into_grid_panels(
            img,
            grid,
            gutter_margin_ratio=args.margin,
            trim_fit=not args.no_fit,
            trim_tol=args.fit_tol,
        )
        texts = _panel_story_texts(scene, len(panels))

        page_dir = out_dir / f"page_{page_number:02d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        panel_items: list[dict[str, Any]] = []
        for i, (panel_img, story_text) in enumerate(zip(panels, texts), start=1):
            panel_file = page_dir / f"panel_{i:02d}.png"
            ok = cv2.imwrite(str(panel_file), panel_img)
            if not ok:
                raise RuntimeError(f"Không ghi được file panel: {panel_file}")
            panel_items.append(
                {
                    "panelNumber": i,
                    "image": str(panel_file.relative_to(out_dir)).replace("\\", "/"),
                    "storyText": story_text,
                }
            )

        page_json = {
            "mode": "manga",
            "pageNumber": page_number,
            "panelCount": len(panel_items),
            "grid": {"rows": grid.rows, "cols": grid.cols},
            "sourceImage": str(img_path.relative_to(input_dir)).replace("\\", "/"),
            "panels": panel_items,
        }
        (page_dir / "panels.json").write_text(
            json.dumps(page_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index.append(
            {
                "pageNumber": page_number,
                "panelCount": len(panel_items),
                "grid": {"rows": grid.rows, "cols": grid.cols},
                "dir": str(page_dir.relative_to(out_dir)).replace("\\", "/"),
            }
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

