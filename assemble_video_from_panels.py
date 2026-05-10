#!/usr/bin/env python3
"""
Ghép panel (ảnh) + audio tương ứng thành 1 video xuyên suốt.

Input A — manga (postprocess_split_panels):
  - panels_root/index.json, page_XX/panels.json, page_XX/panel_YY.png ...
  - audio_root/index.json, page_XX/panel_YY.wav (tùy chọn)

Input B — review / review_two_pass + tts_ngoc_huyen_panels (layout scene_XX):
  - audio_root/index.json (wav: scene_NN.wav, image: ../scene_NN.png hoặc tương đối)
  - ảnh scene_XX.png thường ở thư mục cha của audio_root (hoặc --images-dir)

Output:
  - video .mp4

Yêu cầu:
  - ffmpeg có trong PATH

Ví dụ manga:
  python assemble_video_from_panels.py \\
    --mode panels \\
    --panels-root ./out6/panels_fit \\
    --audio-root ./out6/audio_ngoc_huyen \\
    --output ./out6/final.mp4

Ví dụ review:
  python assemble_video_from_panels.py \\
    --mode review \\
    --audio-root ./out_review2/audio_ngoc_huyen \\
    --output ./out_review2/final.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):  # type: ignore[misc,no-redef]
        return it


@dataclass(frozen=True)
class Item:
    page: int
    panel: int
    image_path: Path
    audio_path: Path | None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_scene_stem_image(images_dir: Path, page: int) -> Path | None:
    stem = f"scene_{int(page):02d}"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p.resolve()
    return None


def _review_items_from_audio_index(
    audio_root: Path,
    images_dir: Path | None,
) -> list[Item]:
    """
    Ghép video từ index.json do tts_ngoc_huyen_panels (layout review) sinh ra.
    Ảnh: trường image (đường dẫn tương đối so với audio_root) hoặc scene_XX.* trong images_dir.
    """
    idx_path = audio_root / "index.json"
    if not idx_path.is_file():
        return []
    data = _load_json(idx_path)
    if not isinstance(data, list):
        return []
    img_base = (images_dir or audio_root.parent).resolve()
    items: list[Item] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            page = int(row.get("pageNumber"))
            panel = int(row.get("panelNumber", 1))
        except Exception:
            continue
        wav_rel = row.get("wav")
        wav_path: Path | None = None
        if wav_rel:
            wp = (audio_root / str(wav_rel)).resolve()
            if wp.is_file():
                wav_path = wp

        img_path: Path | None = None
        img_rel = row.get("image")
        if img_rel:
            cand = (audio_root / str(img_rel)).resolve()
            if cand.is_file():
                img_path = cand
            else:
                cand2 = (img_base / Path(str(img_rel)).name).resolve()
                if cand2.is_file():
                    img_path = cand2
        if img_path is None:
            img_path = _resolve_scene_stem_image(img_base, page)
        if img_path is None:
            continue
        items.append(Item(page=page, panel=panel, image_path=img_path, audio_path=wav_path))

    items.sort(key=lambda x: (x.page, x.panel))
    return items


def _audio_index_map(audio_root: Path) -> dict[tuple[int, int], Path]:
    idx_path = audio_root / "index.json"
    data = _load_json(idx_path)
    out: dict[tuple[int, int], Path] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                p = int(row.get("pageNumber"))
                k = int(row.get("panelNumber"))
            except Exception:
                continue
            wav_rel = row.get("wav")
            if not wav_rel:
                continue
            wav_path = (audio_root / str(wav_rel)).resolve()
            if wav_path.is_file():
                out[(p, k)] = wav_path
    return out


def _panel_items(panels_root: Path, audio_root: Path | None) -> list[Item]:
    audio_map = _audio_index_map(audio_root) if audio_root else {}

    idx_path = panels_root / "index.json"
    if idx_path.is_file():
        idx = _load_json(idx_path)
        page_dirs = []
        if isinstance(idx, list):
            for row in idx:
                if isinstance(row, dict) and row.get("dir"):
                    page_dirs.append(panels_root / str(row["dir"]))
        else:
            page_dirs = []
    else:
        page_dirs = sorted([p for p in panels_root.glob("page_*") if p.is_dir()])

    items: list[Item] = []
    for page_dir in page_dirs:
        pj = page_dir / "panels.json"
        if not pj.is_file():
            continue
        data = _load_json(pj)
        try:
            page = int(data.get("pageNumber", int(page_dir.name.split("_")[-1])))
        except Exception:
            continue
        panels = data.get("panels") or []
        if not isinstance(panels, list):
            continue
        for p in panels:
            if not isinstance(p, dict):
                continue
            try:
                panel = int(p.get("panelNumber"))
            except Exception:
                continue
            img_rel = p.get("image")
            if not img_rel:
                continue
            img_path = (panels_root / str(img_rel)).resolve()
            if not img_path.is_file():
                # fallback: try in page_dir
                img_path = (page_dir / Path(str(img_rel)).name).resolve()
            if not img_path.is_file():
                continue
            wav = audio_map.get((page, panel))
            items.append(Item(page=page, panel=panel, image_path=img_path, audio_path=wav))

    items.sort(key=lambda x: (x.page, x.panel))
    return items


def _wav_duration_seconds(path: Path) -> float:
    try:
        if not path.is_file() or path.stat().st_size < 44:
            return 0.0
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n"
            + " ".join(cmd)
            + "\n\nstderr:\n"
            + p.stderr.decode("utf-8", "ignore")[:4000]
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ghép panel + audio thành video MP4 xuyên suốt (FFmpeg).")
    ap.add_argument(
        "--mode",
        choices=("auto", "panels", "review"),
        default="auto",
        help="panels: manga page_XX/panels.json. review: audio index + scene_XX ảnh. auto: thử panels trước, sau đó review.",
    )
    ap.add_argument(
        "--panels-root",
        type=Path,
        default=None,
        help="Thư mục manga panels (index.json + page_XX/). Không bắt buộc nếu chỉ dùng --mode review.",
    )
    ap.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Thư mục output TTS (index.json + wav). Bắt buộc với --mode review.",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Review: thư mục chứa scene_XX.png (mặc định: thư mục cha của --audio-root).",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--silence", type=float, default=1.2, help="Thời lượng mặc định nếu panel thiếu audio (giây).")
    ap.add_argument("--no-kenburns", action="store_true", help="Tắt zoom/pan nhẹ.")
    ap.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình (tqdm).")
    args = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Không tìm thấy ffmpeg trong PATH.")

    panels_root = args.panels_root.expanduser().resolve() if args.panels_root else None
    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "panels":
        if not panels_root:
            raise SystemExit("--mode panels cần --panels-root.")
        items = _panel_items(panels_root, audio_root)
    elif args.mode == "review":
        if not audio_root:
            raise SystemExit("--mode review cần --audio-root (có index.json).")
        items = _review_items_from_audio_index(audio_root, images_dir)
    else:
        items = []
        if panels_root:
            items = _panel_items(panels_root, audio_root)
        if not items and audio_root:
            items = _review_items_from_audio_index(audio_root, images_dir)
        if not panels_root and not audio_root:
            raise SystemExit("--mode auto cần ít nhất --panels-root hoặc --audio-root.")

    if not items:
        raise SystemExit(
            "Không tìm thấy đoạn nào để ghép. Kiểm tra --mode, panels_root/page_XX/panels.json "
            "hoặc audio_root/index.json + scene_XX.png (--images-dir nếu ảnh không nằm cạnh audio)."
        )

    with tempfile.TemporaryDirectory(prefix="panel_vid_") as td:
        tmp = Path(td)
        seg_paths: list[Path] = []

        iterable = items
        if not args.no_progress:
            iterable = tqdm(items, desc="FFmpeg segments", unit="seg", total=len(items))

        for idx, it in enumerate(iterable, start=1):
            seg = tmp / f"seg_{idx:05d}.mp4"
            seg_paths.append(seg)

            if it.audio_path and it.audio_path.is_file():
                aud = it.audio_path
                dur = _wav_duration_seconds(aud)
                if dur <= 0.01:
                    aud = None
            else:
                aud = None

            # Build video filter: scale/pad to fixed size + optional slow zoom.
            base_scale = (
                f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
                f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:black"
            )
            if args.no_kenburns:
                vf = base_scale
            else:
                # Gentle Ken Burns: zoom in to 1.06 over clip duration.
                # Use fps for zoompan d.
                vf = (
                    base_scale
                    + f",zoompan=z='min(zoom+0.0006,1.06)':d=1:fps={args.fps}"
                )

            if aud is None:
                # silence audio
                _run(
                    [
                        ffmpeg,
                        "-y",
                        "-loop",
                        "1",
                        "-t",
                        f"{args.silence:.3f}",
                        "-i",
                        str(it.image_path),
                        "-f",
                        "lavfi",
                        "-t",
                        f"{args.silence:.3f}",
                        "-i",
                        "anullsrc=channel_layout=mono:sample_rate=22050",
                        "-r",
                        str(args.fps),
                        "-vf",
                        vf,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                        str(seg),
                    ]
                )
            else:
                _run(
                    [
                        ffmpeg,
                        "-y",
                        "-loop",
                        "1",
                        "-i",
                        str(it.image_path),
                        "-i",
                        str(aud),
                        "-r",
                        str(args.fps),
                        "-vf",
                        vf,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                        str(seg),
                    ]
                )

        # Concat segments
        concat_list = tmp / "concat.txt"
        concat_list.write_text(
            "".join([f"file '{p.as_posix()}'\n" for p in seg_paths]),
            encoding="utf-8",
        )

        _run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(out_path),
            ]
        )


if __name__ == "__main__":
    main()

