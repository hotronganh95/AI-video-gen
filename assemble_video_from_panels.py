#!/usr/bin/env python3
"""
Ghép panel (ảnh) + audio tương ứng thành 1 video xuyên suốt.

Input A — manga (postprocess_split_panels):
  - panels_root/index.json, page_XX/panels.json, page_XX/panel_YY.png ...
  - audio_root/index.json, page_XX/panel_YY.wav (tùy chọn)

Input B — review / review_two_pass + tts_ngoc_huyen_panels (layout scene_XX):
  - audio_root/index.json (wav: scene_NN.wav, image: ../scenes/scene_NN.png hoặc ../scene_NN.png)
  - ảnh beat thường ở thư mục cha của audio_root trong scenes/ (hoặc --images-dir)

Input C — i2v (clip đã sinh + TTS):
  - index.json + scene_NN.wav trong audio_root (giống review), hoặc chỉ các file scene_NN.wav (quét tự động)
  - video_root/scene_NN.mp4 (hoặc .webm, .mov, …): timeline video được co giãn (setpts) để đúng độ dài từng WAV

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

Ví dụ i2v (audio + clip video, resample tốc độ video theo độ dài audio):
  python assemble_video_from_panels.py \\
    --mode i2v \\
    --audio-root ./out_review15/audio \\
    --video-root ./out_review15/i2v_scenes \\
    --output ./out_review15/final_i2v.mp4 \\
    --xfade-sec 0.45

  --xfade-sec > 0: mờ nối giữa các đoạn (ffmpeg mới: chồng lấn xfade; cũ: fade qua đen).
"""

from __future__ import annotations

import argparse
import json
import re
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


@dataclass(frozen=True)
class I2vItem:
    """Một scene: clip i2v + wav TTS (độ dài đích = audio)."""

    page: int
    panel: int
    video_path: Path
    audio_path: Path
    extra_video_sec: float = 0.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_scene_stem_image(images_dir: Path, page: int) -> Path | None:
    stem = f"scene_{int(page):02d}"
    for base in (images_dir / "scenes", images_dir):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = base / f"{stem}{ext}"
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


def _ffprobe_duration_seconds(path: Path) -> float:
    """Độ dài media (giây) qua ffprobe; ưu tiên stream video, sau đó format."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.is_file():
        return 0.0

    def _one(args: list[str]) -> float:
        p = subprocess.run(
            [ffprobe, "-v", "error", *args, "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p.returncode != 0:
            return 0.0
        txt = p.stdout.decode("utf-8", "ignore").strip().splitlines()
        if not txt:
            return 0.0
        try:
            v = float(txt[0].strip())
            return v if v > 1e-6 else 0.0
        except ValueError:
            return 0.0

    d = _one(["-select_streams", "v:0", "-show_entries", "stream=duration"])
    if d > 0:
        return d
    return _one(["-show_entries", "format=duration"])


def _resolve_scene_stem_video(video_root: Path, page: int) -> Path | None:
    stem = f"scene_{int(page):02d}"
    for ext in (".mp4", ".webm", ".mov", ".mkv", ".m4v"):
        p = video_root / f"{stem}{ext}"
        if p.is_file():
            return p.resolve()
    return None


def _i2v_items_from_audio_index(audio_root: Path, video_root: Path) -> list[I2vItem]:
    """
    Cùng index.json với review: mỗi dòng có pageNumber + wav.
    Clip video: scene_NN.* trong video_root.
    """
    idx_path = audio_root / "index.json"
    if not idx_path.is_file():
        return []
    data = _load_json(idx_path)
    if not isinstance(data, list):
        return []
    video_root = video_root.resolve()
    items: list[I2vItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            page = int(row.get("pageNumber"))
            panel = int(row.get("panelNumber", 1))
        except Exception:
            continue
        wav_rel = row.get("wav")
        if not wav_rel:
            continue
        wav_path = (audio_root / str(wav_rel)).resolve()
        if not wav_path.is_file():
            continue
        vid = _resolve_scene_stem_video(video_root, page)
        if vid is None:
            continue
        items.append(I2vItem(page=page, panel=panel, video_path=vid, audio_path=wav_path))

    items.sort(key=lambda x: (x.page, x.panel))
    return items


_RE_SCENE_WAV = re.compile(r"^scene_(\d+)\.wav$", re.IGNORECASE)


def _i2v_items_from_scene_wav_glob(audio_root: Path, video_root: Path) -> list[I2vItem]:
    """
    Khi không có index.json: ghép scene_NN.wav trong audio_root với scene_NN.* trong video_root.
    """
    audio_root = audio_root.resolve()
    video_root = video_root.resolve()
    items: list[I2vItem] = []
    for wav in sorted(audio_root.glob("scene_*.wav")):
        m = _RE_SCENE_WAV.match(wav.name)
        if not m:
            continue
        page = int(m.group(1))
        vid = _resolve_scene_stem_video(video_root, page)
        if vid is None:
            continue
        items.append(I2vItem(page=page, panel=1, video_path=vid, audio_path=wav.resolve()))
    items.sort(key=lambda x: (x.page, x.panel))
    return items


def _i2v_items_resolve(audio_root: Path, video_root: Path) -> list[I2vItem]:
    out = _i2v_items_from_audio_index(audio_root, video_root)
    if out:
        return out
    return _i2v_items_from_scene_wav_glob(audio_root, video_root)


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n"
            + " ".join(cmd)
            + "\n\nstderr:\n"
            + p.stderr.decode("utf-8", "ignore")[:4000]
        )


def _merge_concat_copy(seg_paths: list[Path], out_path: Path, *, ffmpeg: str, tmp_dir: Path) -> None:
    concat_list = tmp_dir / "concat.txt"
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


def _reencode_segment_fade_ends(
    src: Path,
    dst: Path,
    *,
    idx: int,
    n: int,
    d_req: float,
    ffmpeg: str,
) -> None:
    """
    Fade-in đầu đoạn (trừ đoạn đầu tiên) và fade-out cuối (trừ đoạn cuối) — mềm nối qua đen, tương thích FFmpeg cũ.
    """
    dur = _ffprobe_duration_seconds(src)
    if dur <= 0.05:
        raise RuntimeError(f"Segment quá ngắn hoặc không đọc được: {src}")
    has_in = idx > 0
    has_out = idx < n - 1
    d_in = min(d_req, dur * 0.42) if has_in else 0.0
    d_out = min(d_req, dur * 0.42) if has_out else 0.0
    while d_in + d_out > dur * 0.82 and (d_in > 0.03 or d_out > 0.03):
        d_in *= 0.9
        d_out *= 0.9
    vf_parts: list[str] = []
    af_parts: list[str] = []
    if d_in > 0.03:
        vf_parts.append(f"fade=t=in:st=0:d={d_in:.5f}")
        af_parts.append(f"afade=t=in:st=0:d={d_in:.5f}")
    if d_out > 0.03:
        st = max(dur - d_out, 0.0)
        vf_parts.append(f"fade=t=out:st={st:.5f}:d={d_out:.5f}")
        af_parts.append(f"afade=t=out:st={st:.5f}:d={d_out:.5f}")
    cmd = [ffmpeg, "-y", "-i", str(src)]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    if af_parts:
        cmd.extend(["-af", ",".join(af_parts)])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(dst),
        ]
    )
    _run(cmd)


def _merge_segments_fade_through_black(
    seg_paths: list[Path],
    out_path: Path,
    *,
    ffmpeg: str,
    xfade_sec: float,
    tmp_dir: Path,
) -> None:
    n = len(seg_paths)
    faded: list[Path] = []
    for idx, p in enumerate(seg_paths):
        fd = tmp_dir / f"faded_{idx:05d}.mp4"
        _reencode_segment_fade_ends(p, fd, idx=idx, n=n, d_req=xfade_sec, ffmpeg=ffmpeg)
        faded.append(fd)
    _merge_concat_copy(faded, out_path, ffmpeg=ffmpeg, tmp_dir=tmp_dir)


def _merge_segments_xfade_acrossfade(
    seg_paths: list[Path],
    out_path: Path,
    *,
    ffmpeg: str,
    xfade_sec: float,
    audio_join: str,
) -> None:
    n = len(seg_paths)
    durs = [_ffprobe_duration_seconds(p) for p in seg_paths]
    if any(d <= 0.05 for d in durs):
        raise RuntimeError(
            "Một hoặc nhiều segment có độ dài không hợp lệ; không thể dùng --xfade-sec."
        )

    cmd: list[str] = [ffmpeg, "-y"]
    for p in seg_paths:
        cmd.extend(["-i", str(p)])

    fc_parts: list[str] = []
    v_in = "0:v"
    a_in = "0:a"
    cum = durs[0]
    # Với audio_join="concat": cần loại bỏ tổng thời lượng overlap để audio không dài hơn video.
    # Cắt ở CUỐI đoạn trước (ít làm mất chữ đầu câu hơn cắt đầu đoạn sau).
    end_trims: list[float] = [0.0] * n
    for i in range(1, n):
        dlim = min(
            xfade_sec,
            max(durs[i - 1] * 0.45, 0.04),
            max(durs[i] * 0.45, 0.04),
        )
        if cum <= dlim + 0.02:
            dlim = max(min(cum * 0.35, durs[i] * 0.35), 0.04)
        off = max(cum - dlim, 0.0)
        v_out = f"vx{i}" if i < n - 1 else "vout"
        a_out = f"ax{i}" if i < n - 1 else "aout"
        end_trims[i - 1] = dlim
        fc_parts.append(
            f"[{v_in}][{i}:v]xfade=transition=fade:duration={dlim:.6f}:offset={off:.6f}[{v_out}]"
        )
        if audio_join == "acrossfade":
            fc_parts.append(
                f"[{a_in}][{i}:a]acrossfade=d={dlim:.6f}:c1=tri:c2=tri[{a_out}]"
            )
        v_in = v_out
        if audio_join == "acrossfade":
            a_in = a_out
        cum += durs[i] - dlim

    if audio_join == "concat":
        a_labels: list[str] = []
        for i in range(n):
            lbl = f"ac{i}"
            fc_parts.append(f"[{i}:a]asetpts=N/SR/TB[{lbl}]")
            a_labels.append(f"[{lbl}]")
        fc_parts.append(f"{''.join(a_labels)}concat=n={n}:v=0:a=1[aout]")

    cmd.extend(
        [
            "-filter_complex",
            ";".join(fc_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out_path),
        ]
    )
    _run(cmd)


def _merge_segments_xfade_video_only(
    seg_paths: list[Path],
    out_path: Path,
    *,
    ffmpeg: str,
    xfade_sec: float,
) -> None:
    n = len(seg_paths)
    durs = [_ffprobe_duration_seconds(p) for p in seg_paths]
    if any(d <= 0.05 for d in durs):
        raise RuntimeError(
            "Một hoặc nhiều segment có độ dài không hợp lệ; không thể dùng --xfade-sec."
        )

    cmd: list[str] = [ffmpeg, "-y"]
    for p in seg_paths:
        cmd.extend(["-i", str(p)])

    fc_parts: list[str] = []
    v_in = "0:v"
    cum = durs[0]
    for i in range(1, n):
        dlim = min(
            xfade_sec,
            max(durs[i - 1] * 0.45, 0.04),
            max(durs[i] * 0.45, 0.04),
        )
        if cum <= dlim + 0.02:
            dlim = max(min(cum * 0.35, durs[i] * 0.35), 0.04)
        off = max(cum - dlim, 0.0)
        v_out = f"vx{i}" if i < n - 1 else "vout"
        fc_parts.append(
            f"[{v_in}][{i}:v]xfade=transition=fade:duration={dlim:.6f}:offset={off:.6f}[{v_out}]"
        )
        v_in = v_out
        cum += durs[i] - dlim

    cmd.extend(
        [
            "-filter_complex",
            ";".join(fc_parts),
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )
    _run(cmd)


def _merge_segments_to_output(
    seg_paths: list[Path],
    out_path: Path,
    *,
    ffmpeg: str,
    xfade_sec: float,
    audio_join: str,
    tmp_dir: Path,
) -> None:
    """
    Ghép các đoạn MP4. xfade_sec <= 0: concat demuxer -c copy (cắt cứng).
    xfade_sec > 0: ưu tiên xfade+acrossfade (chồng lấn, cần ffmpeg có filter xfade);
    nếu không có (ffmpeg cũ): fade-in/out từng đoạn rồi concat (mềm hơn cắt cứng, hơi tối giữa cảnh).
    """
    n = len(seg_paths)
    if n == 0:
        raise RuntimeError("Không có segment để ghép.")
    if n == 1:
        shutil.copyfile(seg_paths[0], out_path)
        return

    if xfade_sec <= 0:
        _merge_concat_copy(seg_paths, out_path, ffmpeg=ffmpeg, tmp_dir=tmp_dir)
        return

    try:
        _merge_segments_xfade_acrossfade(
            seg_paths,
            out_path,
            ffmpeg=ffmpeg,
            xfade_sec=xfade_sec,
            audio_join=audio_join,
        )
    except RuntimeError as e:
        err = str(e).lower()
        if "xfade" not in err and "no such filter" not in err:
            raise
        _merge_segments_fade_through_black(
            seg_paths, out_path, ffmpeg=ffmpeg, xfade_sec=xfade_sec, tmp_dir=tmp_dir
        )


def _atempo_filter(speed: float) -> str:
    """
    FFmpeg atempo hỗ trợ 0.5..2.0 mỗi filter. Chia nhỏ nếu cần.
    """
    if speed <= 0:
        raise ValueError("audio_speed phải > 0")
    parts: list[str] = []
    s = float(speed)
    while s > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-6:
        parts.append(f"atempo={s:.6f}")
    return ",".join(parts)


def _xfade_overlaps(durs: list[float], xfade_sec: float) -> list[float]:
    """
    Tính overlap (dlim) giữa các đoạn theo cùng heuristic như khi ghép xfade.
    Trả về list độ dài n, trong đó overlap_after[i] là overlap giữa đoạn i và i+1.
    """
    n = len(durs)
    if n <= 1:
        return [0.0] * n
    xfade_sec = float(xfade_sec)
    overlap_after = [0.0] * n
    cum = float(durs[0])
    for i in range(1, n):
        dlim = min(
            xfade_sec,
            max(durs[i - 1] * 0.45, 0.04),
            max(durs[i] * 0.45, 0.04),
        )
        if cum <= dlim + 0.02:
            dlim = max(min(cum * 0.35, durs[i] * 0.35), 0.04)
        overlap_after[i - 1] = float(dlim)
        cum += float(durs[i]) - float(dlim)
    return overlap_after


def _vf_scale_to_output(width: int, height: int, frame_fit: str) -> str:
    """
    Chuỗi filter scale/crop/pad đưa khung hình về width x height.
    pad: giữ tỉ lệ, viền đen nếu lệch tỉ lệ.
    crop: giữ tỉ lệ, phóng to rồi cắt giữa — full khung, không viền.
    stretch: kéo vừa khung (có thể méo).
    keep: giữ nguyên khung hình đầu vào (không scale/crop/pad).
    """
    w, h = width, height
    if frame_fit == "keep" or w <= 0 or h <= 0:
        return ""
    if frame_fit == "stretch":
        return f"scale={w}:{h}:flags=bicubic"
    if frame_fit == "crop":
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        )
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _encode_i2v_segment(
    ffmpeg: str,
    it: I2vItem,
    seg_out: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frame_fit: str,
    crf: int,
    preset: str,
    audio_speed: float,
) -> None:
    """
    Co giãn timeline video (setpts) để độ dài khớp WAV; âm thanh giữ nguyên.
    ratio = dur_audio / dur_video  =>  PTS' = PTS * ratio.
    """
    aud_dur = _wav_duration_seconds(it.audio_path)
    vid_dur = _ffprobe_duration_seconds(it.video_path)
    if aud_dur <= 0.01:
        raise RuntimeError(f"WAV không hợp lệ hoặc độ dài ~0: {it.audio_path}")
    if vid_dur <= 0.01:
        raise RuntimeError(f"Không đọc được độ dài video (cần ffprobe): {it.video_path}")
    eff_aud_dur = (
        aud_dur / float(audio_speed)
        if audio_speed and abs(float(audio_speed) - 1.0) > 1e-9
        else aud_dur
    )
    target_vid_dur = max(eff_aud_dur + float(it.extra_video_sec), 0.04)
    ratio = target_vid_dur / vid_dur
    parts: list[str] = [f"setpts=PTS*{ratio:.10f}", "setsar=1"]
    if fps > 0:
        parts.append(f"fps={fps}")
    fit_vf = _vf_scale_to_output(width, height, frame_fit)
    if fit_vf:
        parts.append(fit_vf)
    vf = ",".join(parts)
    af = _atempo_filter(audio_speed) if abs(float(audio_speed) - 1.0) > 1e-9 else ""
    fc = ";".join([f"[0:v]{vf}[v]"] + ([f"[1:a]{af}[a]"] if af else []))
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(it.video_path),
            "-i",
            str(it.audio_path),
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            "-map",
            ("[a]" if af else "1:a"),
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(seg_out),
        ]
    )


def _encode_i2v_video_segment_only(
    ffmpeg: str,
    it: I2vItem,
    seg_out: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frame_fit: str,
    crf: int,
    preset: str,
    audio_speed: float,
) -> None:
    """
    Encode video-only segment cho i2v.
    Video được setpts để khớp (audio_duration/audio_speed + extra_video_sec).
    Âm thanh sẽ được ghép riêng ở bước cuối.
    """
    aud_dur = _wav_duration_seconds(it.audio_path)
    vid_dur = _ffprobe_duration_seconds(it.video_path)
    if aud_dur <= 0.01:
        raise RuntimeError(f"WAV không hợp lệ hoặc độ dài ~0: {it.audio_path}")
    if vid_dur <= 0.01:
        raise RuntimeError(f"Không đọc được độ dài video (cần ffprobe): {it.video_path}")

    eff_aud_dur = (
        aud_dur / float(audio_speed)
        if audio_speed and abs(float(audio_speed) - 1.0) > 1e-9
        else aud_dur
    )
    target_vid_dur = max(eff_aud_dur + float(it.extra_video_sec), 0.04)
    ratio = target_vid_dur / vid_dur

    parts: list[str] = [f"setpts=PTS*{ratio:.10f}", "setsar=1"]
    if fps > 0:
        parts.append(f"fps={fps}")
    fit_vf = _vf_scale_to_output(width, height, frame_fit)
    if fit_vf:
        parts.append(fit_vf)
    vf = ",".join(parts)

    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(it.video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            str(seg_out),
        ]
    )


def _encode_i2v_audio_concat_track(
    ffmpeg: str,
    items: list[I2vItem],
    out_audio: Path,
    *,
    audio_speed: float,
) -> None:
    """
    Tạo 1 track audio bằng cách concat thẳng từng WAV (theo thứ tự items),
    có thể tăng tốc bằng atempo.
    """
    cmd: list[str] = [ffmpeg, "-y"]
    for it in items:
        cmd.extend(["-i", str(it.audio_path)])
    fc_parts: list[str] = []
    a_labels: list[str] = []
    af = _atempo_filter(audio_speed) if abs(float(audio_speed) - 1.0) > 1e-9 else ""
    for i in range(len(items)):
        lbl = f"a{i}"
        if af:
            fc_parts.append(f"[{i}:a]{af},asetpts=N/SR/TB[{lbl}]")
        else:
            fc_parts.append(f"[{i}:a]asetpts=N/SR/TB[{lbl}]")
        a_labels.append(f"[{lbl}]")
    fc_parts.append(f"{''.join(a_labels)}concat=n={len(items)}:v=0:a=1[aout]")

    cmd.extend(
        [
            "-filter_complex",
            ";".join(fc_parts),
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            str(out_audio),
        ]
    )
    _run(cmd)


def _mux_video_audio(
    ffmpeg: str,
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    tmp_dir: Path,
) -> None:
    vdur = _ffprobe_duration_seconds(video_path)
    adur = _ffprobe_duration_seconds(audio_path)
    src_video = video_path

    # Nếu video ngắn hơn audio thì -shortest sẽ cắt mất đuôi audio.
    # Pad video bằng cách đứng hình frame cuối cho đủ thời lượng.
    if vdur > 0.01 and adur > 0.01 and vdur + 0.02 < adur:
        need = min(adur - vdur + 0.10, 30.0)
        padded = tmp_dir / "video_padded.mp4"
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={need:.6f}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(padded),
            ]
        )
        src_video = padded

    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src_video),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ghép panel + audio thành video MP4 xuyên suốt (FFmpeg).")
    ap.add_argument(
        "--mode",
        choices=("auto", "panels", "review", "i2v"),
        default="auto",
        help="panels: manga page_XX/panels.json. review: audio index + scene_XX ảnh. i2v: audio index + scene_NN clip trong --video-root (co giãn video theo độ dài WAV). auto: thử panels trước, sau đó review.",
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
        help="Review: thư mục chứa scenes/scene_XX.* hoặc scene_XX.* (mặc định: thư mục cha của --audio-root).",
    )
    ap.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="i2v: thư mục scene_NN.mp4… (khớp pageNumber trong index.json hoặc số trong scene_NN.wav).",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=0, help="FPS đầu ra. 0 = giữ nguyên FPS của video nguồn.")
    ap.add_argument("--width", type=int, default=0, help="Chiều rộng đầu ra. 0 = giữ nguyên kích thước video nguồn.")
    ap.add_argument("--height", type=int, default=0, help="Chiều cao đầu ra. 0 = giữ nguyên kích thước video nguồn.")
    ap.add_argument(
        "--frame-fit",
        choices=("keep", "pad", "crop", "stretch"),
        default="keep",
        help="Cách đưa khung hình về --width x --height: keep=giữ nguyên; pad=viền đen giữ tỉ lệ; crop=phóng to cắt mép full khung; stretch=kéo vừa khung (méo).",
    )
    ap.add_argument("--crf", type=int, default=18, help="Chất lượng x264 (nhỏ hơn = nét hơn; 18 ~ gần lossless).")
    ap.add_argument("--preset", type=str, default="medium", help="Preset x264 (nhanh/chậm). Ví dụ: veryfast, fast, medium, slow.")
    ap.add_argument(
        "--xfade-sec",
        type=float,
        default=0.0,
        help="Thời gian chồng lấn mờ khi ghép các đoạn (giây). 0 = nối copy cắt cứng; ví dụ 0.35–0.6 mượt hơn (encode lại, chậm hơn concat -c copy).",
    )
    ap.add_argument(
        "--audio-join",
        choices=("acrossfade", "concat"),
        default="acrossfade",
        help="Khi --xfade-sec>0: acrossfade=mượt âm thanh theo video; concat=nối thẳng (cắt cứng) nhưng vẫn khớp độ dài bằng cách cắt bỏ phần overlap.",
    )
    ap.add_argument(
        "--audio-speed",
        type=float,
        default=1.0,
        help="Tốc độ audio (áp dụng ở bước i2v segment). Ví dụ 1.2 = nhanh hơn 20%.",
    )
    ap.add_argument("--silence", type=float, default=1.2, help="Thời lượng mặc định nếu panel thiếu audio (giây).")
    ap.add_argument("--no-kenburns", action="store_true", help="Tắt zoom/pan nhẹ.")
    ap.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình (tqdm).")
    args = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Không tìm thấy ffmpeg trong PATH.")
    if args.mode == "i2v" and not shutil.which("ffprobe"):
        raise SystemExit("--mode i2v cần ffprobe trong PATH (cùng gói ffmpeg).")

    panels_root = args.panels_root.expanduser().resolve() if args.panels_root else None
    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    video_root = args.video_root.expanduser().resolve() if args.video_root else None
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: list[Item] = []
    i2v_work: list[I2vItem] | None = None

    if args.mode == "panels":
        if not panels_root:
            raise SystemExit("--mode panels cần --panels-root.")
        items = _panel_items(panels_root, audio_root)
    elif args.mode == "review":
        if not audio_root:
            raise SystemExit("--mode review cần --audio-root (có index.json).")
        items = _review_items_from_audio_index(audio_root, images_dir)
    elif args.mode == "i2v":
        if not audio_root:
            raise SystemExit("--mode i2v cần --audio-root (có index.json).")
        if not video_root:
            raise SystemExit("--mode i2v cần --video-root (scene_NN.mp4…).")
        i2v_work = _i2v_items_resolve(audio_root, video_root)
        if not i2v_work:
            raise SystemExit(
                "Không ghép được cặp nào (i2v). Cần trong audio_root: index.json + wav, "
                "hoặc các file scene_NN.wav; trong video_root: scene_NN.mp4 (hoặc .webm/.mov) "
                "cùng số scene."
            )
        # Với video xfade + audio concat thẳng: tổng video sẽ bị trừ đi overlap.
        # Để KHÔNG phải cắt audio, ta bơm thêm thời lượng cho từng clip video (trừ clip cuối)
        # sao cho sau khi xfade thì tổng thời lượng video khớp tổng audio.
        if float(args.xfade_sec) > 0 and str(args.audio_join) == "concat":
            base = [
                _wav_duration_seconds(it.audio_path) / float(args.audio_speed)
                if float(args.audio_speed) and abs(float(args.audio_speed) - 1.0) > 1e-9
                else _wav_duration_seconds(it.audio_path)
                for it in i2v_work
            ]
            # Lặp 2 vòng để overlap ổn định hơn (vì duration mục tiêu sẽ được cộng overlap).
            ov1 = _xfade_overlaps(base, float(args.xfade_sec))
            tgt1 = [max(b + o, 0.04) for b, o in zip(base, ov1)]
            ov2 = _xfade_overlaps(tgt1, float(args.xfade_sec))
            i2v_work = [
                I2vItem(
                    page=it.page,
                    panel=it.panel,
                    video_path=it.video_path,
                    audio_path=it.audio_path,
                    extra_video_sec=float(ov2[idx]),
                )
                for idx, it in enumerate(i2v_work)
            ]
    else:
        if panels_root:
            items = _panel_items(panels_root, audio_root)
        if not items and audio_root:
            items = _review_items_from_audio_index(audio_root, images_dir)
        if not panels_root and not audio_root:
            raise SystemExit("--mode auto cần ít nhất --panels-root hoặc --audio-root.")

    if i2v_work is None and not items:
        raise SystemExit(
            "Không tìm thấy đoạn nào để ghép. Kiểm tra --mode, panels_root/page_XX/panels.json "
            "hoặc audio_root/index.json + scenes/scene_XX.* (--images-dir nếu ảnh không nằm cạnh audio)."
        )

    with tempfile.TemporaryDirectory(prefix="panel_vid_") as td:
        tmp = Path(td)
        seg_paths: list[Path] = []

        if i2v_work is not None:
            iterable_i2v = (
                tqdm(i2v_work, desc="FFmpeg i2v segments", unit="seg", total=len(i2v_work))
                if not args.no_progress
                else i2v_work
            )
            for idx, it in enumerate(iterable_i2v, start=1):
                seg = tmp / f"seg_{idx:05d}.mp4"
                seg_paths.append(seg)
                # i2v: encode VIDEO only; audio sẽ ghép riêng để tránh lệch khi xfade.
                _encode_i2v_video_segment_only(
                    ffmpeg,
                    it,
                    seg,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    frame_fit=args.frame_fit,
                    crf=args.crf,
                    preset=args.preset,
                    audio_speed=float(args.audio_speed),
                )
        else:
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

                # Build video filter: scale/crop/pad to fixed size + optional slow zoom.
                base_scale = _vf_scale_to_output(args.width, args.height, args.frame_fit)
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

        if i2v_work is None:
            _merge_segments_to_output(
                seg_paths,
                out_path,
                ffmpeg=ffmpeg,
                xfade_sec=float(args.xfade_sec),
                audio_join=str(args.audio_join),
                tmp_dir=tmp,
            )
            return

        # i2v: ghép video và audio theo pipeline riêng
        audio_track = tmp / "audio_track.m4a"
        _encode_i2v_audio_concat_track(
            ffmpeg,
            i2v_work,
            audio_track,
            audio_speed=float(args.audio_speed),
        )

        if float(args.xfade_sec) <= 0:
            merged_video = tmp / "video_merged.mp4"
            _merge_concat_copy(seg_paths, merged_video, ffmpeg=ffmpeg, tmp_dir=tmp)
        else:
            merged_video = tmp / "video_xfade.mp4"
            _merge_segments_xfade_video_only(
                seg_paths,
                merged_video,
                ffmpeg=ffmpeg,
                xfade_sec=float(args.xfade_sec),
            )

        _mux_video_audio(ffmpeg, merged_video, audio_track, out_path, tmp_dir=tmp)


if __name__ == "__main__":
    main()

