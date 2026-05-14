#!/usr/bin/env python3
"""
Sinh giọng đọc (Piper TTS) cho từng panel.storyText hoặc từng beat storyText (review).

Input A — manga panels (postprocess_split_panels.py):
  panels_root/
    index.json
    page_01/panels.json
    page_01/panel_01.png ...

Input B — review_two_pass / scenes.json (cùng thư mục với scenes/scene_XX.* hoặc legacy scene_XX.*):
  python tts_ngoc_huyen_panels.py \
    --scenes-json ./out_r2/scenes.json \
    --model /path/to/ngoc_huyen_moi.onnx \
    --output ./out_r2/audio_ngoc_huyen \
    --cuda
  Layout mặc định review: scene_01.wav … (khớp scenes/scene_01.*). Manifest có image tương đối tới ảnh.
  --layout panels: giữ page_NN/panel_MM.wav (manga / assemble_video_from_panels cũ).
  GPU: pip install onnxruntime-gpu nvidia-cudnn-cu12; --cuda không dùng chung --force-cli.
  (Script tự thêm thư mục lib của gói nvidia-* vào LD_LIBRARY_PATH trước khi nạp ONNX — tránh lỗi libcudnn.so.9 trên WSL.)
  Song song: --workers N. Với --cuda chỉ 1 worker; muốn N tiến trình thì --force-cli --workers N (piper CLI).

Output:
  review: out_audio/scene_01.wav … + index.json
  panels: out_audio/page_01/panel_01.wav … + index.json

Yêu cầu model:
  - Piper ONNX model + config JSON cùng basename.
  - Ví dụ: ngoc_huyen_moi.onnx + ngoc_huyen_moi.onnx.json

Ví dụ panels:
  pip install -r requirements.txt
  python tts_ngoc_huyen_panels.py \
    --panels-root ./out6/panels_fit \
    --model /path/to/ngoc_huyen_moi.onnx \
    --output ./out6/audio_ngoc_huyen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


def _prepend_nvidia_pip_libs_to_ld_library_path() -> None:
    """
    Gói pip nvidia-cudnn-cu12 (và bạn CUDA) đặt .so dưới site-packages/nvidia/*/lib.
    Trên WSL/Linux, chỉ set LD_LIBRARY_PATH đôi khi vẫn không đủ khi ONNX dlopen plugin CUDA;
    thêm preload ctypes RTLD_GLOBAL cho chuỗi libcudart → cublas → cudnn.
    Gọi TRƯỚC khi import piper/onnxruntime.
    """
    try:
        import nvidia.cudnn as _nv_cudnn
    except ImportError:
        return
    root = Path(_nv_cudnn.__file__).resolve().parent.parent
    if root.name != "nvidia":
        return
    sub_libs = (
        "cudnn",
        "cublas",
        "cuda_runtime",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nvjitlink",
        "cuda_nvrtc",
        "nccl",
    )
    prefixes: list[str] = []
    for sub in sub_libs:
        lib_dir = root / sub / "lib"
        if lib_dir.is_dir():
            prefixes.append(str(lib_dir.resolve()))
    if not prefixes:
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    for pre in reversed(prefixes):
        if pre not in parts:
            parts.insert(0, pre)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    # Preload: đảm bảo libcudnn.so.9 được nạp trước libonnxruntime_providers_cuda.so
    import ctypes

    def _dlopen_chain(lib_dir: Path, pattern: str) -> None:
        if not lib_dir.is_dir():
            return
        for so in sorted(lib_dir.glob(pattern)):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue

    cuda_rt = root / "cuda_runtime" / "lib"
    cublas = root / "cublas" / "lib"
    cudnn = root / "cudnn" / "lib"
    _dlopen_chain(cuda_rt, "libcudart.so*")
    _dlopen_chain(cublas, "libcublasLt.so*")
    _dlopen_chain(cublas, "libcublas.so*")
    _dlopen_chain(cudnn, "libcudnn.so*")


_prepend_nvidia_pip_libs_to_ld_library_path()

from piper import PiperVoice

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it: Iterable[Any], **_kw: Any) -> Iterable[Any]:  # type: ignore[misc,no-redef]
        return it


def _load_piper_voice(model_path: Path, *, use_cuda: bool) -> PiperVoice:
    """
    Tải PiperVoice; bật GPU qua use_cuda (cần onnxruntime-gpu + CUDA).
    """
    mp = str(model_path)
    try:
        voice = PiperVoice.load(mp, use_cuda=use_cuda)
    except TypeError:
        # Piper cũ không có tham số use_cuda
        if use_cuda:
            raise SystemExit(
                "Phiên bản `piper` hiện tại không hỗ trợ use_cuda. "
                "Hãy nâng cấp: pip install -U piper-tts"
            ) from None
        voice = PiperVoice.load(mp)
    return voice


@dataclass
class PanelRef:
    page_number: int
    panel_number: int
    story_text: str
    rel_image: str | None = None
    outline_section_number: int | None = None
    narrative_plane: str | None = None


def _panelref_to_dict(r: PanelRef) -> dict[str, Any]:
    return {
        "page_number": r.page_number,
        "panel_number": r.panel_number,
        "story_text": r.story_text,
        "rel_image": r.rel_image,
        "outline_section_number": r.outline_section_number,
        "narrative_plane": r.narrative_plane,
    }


def _panelref_from_dict(d: dict[str, Any]) -> PanelRef:
    return PanelRef(
        page_number=int(d["page_number"]),
        panel_number=int(d["panel_number"]),
        story_text=str(d.get("story_text") or ""),
        rel_image=d.get("rel_image"),
        outline_section_number=d.get("outline_section_number"),
        narrative_plane=d.get("narrative_plane"),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_panels(panels_root: Path) -> list[PanelRef]:
    """
    Đọc toàn bộ panels.json theo từng page_XX/.
    """
    refs: list[PanelRef] = []
    index_path = panels_root / "index.json"
    if index_path.is_file():
        idx = _load_json(index_path)
        if isinstance(idx, list):
            page_dirs = [panels_root / str(row.get("dir")) for row in idx if isinstance(row, dict)]
        else:
            page_dirs = []
    else:
        page_dirs = sorted([p for p in panels_root.glob("page_*") if p.is_dir()])

    for page_dir in page_dirs:
        pj = page_dir / "panels.json"
        if not pj.is_file():
            continue
        data = _load_json(pj)
        page_number = int(data.get("pageNumber", int(page_dir.name.split("_")[-1])))
        panels = data.get("panels") or []
        if not isinstance(panels, list):
            continue
        for p in panels:
            if not isinstance(p, dict):
                continue
            panel_number = int(p.get("panelNumber", 0) or 0)
            story_text = str(p.get("storyText") or "").strip()
            rel_image = p.get("image")
            refs.append(
                PanelRef(
                    page_number=page_number,
                    panel_number=panel_number,
                    story_text=story_text,
                    rel_image=str(rel_image) if rel_image else None,
                    outline_section_number=None,
                    narrative_plane=None,
                )
            )

    refs.sort(key=lambda r: (r.page_number, r.panel_number))
    return refs


def _iter_scenes_json(scenes_path: Path) -> list[PanelRef]:
    """
    Đọc scenes.json từ pipeline (review/storybook/manga): mỗi phần tử = một beat/cảnh.
    Dùng pageNumber làm chỉ số trang; panelNumber cố định 1 (một ảnh/beat) để tương thích index ghép video.
    """
    data = _load_json(scenes_path)
    if not isinstance(data, list):
        raise ValueError("scenes.json phải là một mảng JSON.")
    refs: list[PanelRef] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            pn = int(row.get("pageNumber", 0))
        except (TypeError, ValueError):
            continue
        if pn < 1:
            continue
        story_text = str(row.get("storyText") or "").strip()
        raw_os = row.get("outlineSectionNumber")
        osn: int | None = None
        if raw_os is not None and str(raw_os).strip() != "":
            try:
                osn = int(raw_os)
            except (TypeError, ValueError):
                osn = None
        nplane = str(row.get("narrativePlane") or "").strip() or None
        refs.append(
            PanelRef(
                page_number=pn,
                panel_number=1,
                story_text=story_text,
                rel_image=None,
                outline_section_number=osn,
                narrative_plane=nplane,
            )
        )
    refs.sort(key=lambda r: (r.page_number, r.panel_number))
    return refs


def _review_scene_image_relpath(images_dir: Path, audio_out_dir: Path, page_number: int) -> str:
    """Đường dẫn ảnh minh họa (scenes/scene_XX.* hoặc legacy scene_XX.* ở root) tương đối tới audio_out_dir."""
    stem = f"scene_{int(page_number):02d}"
    images_dir = images_dir.resolve()
    audio_out_dir = audio_out_dir.resolve()
    sub = images_dir / "scenes"
    for base in (sub, images_dir):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = base / f"{stem}{ext}"
            if p.is_file():
                return os.path.relpath(p, audio_out_dir).replace("\\", "/")
    return os.path.relpath(sub / f"{stem}.png", audio_out_dir).replace("\\", "/")


def _wav_path_for_beat(out_dir: Path, r: PanelRef, *, layout: str) -> Path:
    if layout == "review":
        return out_dir / f"scene_{r.page_number:02d}.wav"
    page_dir = out_dir / f"page_{r.page_number:02d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    return page_dir / f"panel_{r.panel_number:02d}.wav"


def _manifest_row(
    r: PanelRef,
    wav_path: Path,
    out_dir: Path,
    dur: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pageNumber": r.page_number,
        "panelNumber": r.panel_number,
        "wav": str(wav_path.relative_to(out_dir)).replace("\\", "/"),
        "durationSec": dur,
        "storyText": r.story_text,
        "image": r.rel_image,
    }
    if r.outline_section_number is not None:
        row["outlineSectionNumber"] = r.outline_section_number
    if r.narrative_plane:
        row["narrativePlane"] = r.narrative_plane
    return row


def _wav_duration_seconds(wav_path: Path) -> float:
    # WAV có thể bị hỏng/0 bytes nếu synth fail giữa chừng.
    try:
        if not wav_path.is_file():
            return 0.0
        if wav_path.stat().st_size < 44:  # nhỏ hơn header WAV
            return 0.0
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except (EOFError, wave.Error):
        return 0.0


def _is_valid_wav(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 44:
            return False
        head = path.read_bytes()[:12]
        if not (head[:4] == b"RIFF" and head[8:12] == b"WAVE"):
            return False
        # Must have at least some frames; empty WAV is not useful.
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() > 0 and wf.getframerate() > 0
        except Exception:
            return False
    except Exception:
        return False


def _synthesize_compat(voice: PiperVoice, text: str, wf: wave.Wave_write, kwargs: dict[str, Any]) -> None:
    """
    PiperVoice.synthesize API khác nhau theo version/build.
    Chiến lược: thử gọi với kwargs, nếu TypeError báo unexpected keyword -> bỏ key đó và thử lại.
    """
    local = dict(kwargs)
    while True:
        try:
            voice.synthesize(text, wf, **local)
            return
        except TypeError as e:
            msg = str(e)
            # Example: got an unexpected keyword argument 'length_scale'
            m = None
            if "unexpected keyword argument" in msg:
                import re as _re

                m = _re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", msg)
            if m:
                bad = m.group(1)
                if bad in local:
                    local.pop(bad, None)
                    continue
            raise


def _synthesize_via_cli(piper_exe: str, model_path: Path, text: str, wav_path: Path) -> None:
    """
    Fallback cực bền: dùng piper CLI để xuất WAV chuẩn.
    """
    proc = subprocess.run(
        [piper_exe, "--model", str(model_path), "--output_file", str(wav_path)],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0 or not _is_valid_wav(wav_path):
        raise RuntimeError(
            "Piper CLI synth failed. "
            f"exit={proc.returncode} stderr={(proc.stderr.decode('utf-8','ignore')[:400])}"
        )


_mp_bundle: dict[str, Any] | None = None


def _tts_mp_initializer(
    model_path: str,
    use_cuda: bool,
    fallback_sample_rate: int,
    speaker: int | None,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
    sentence_silence: float,
) -> None:
    """Mỗi worker (chế độ API CPU) tải Piper một lần."""
    global _mp_bundle
    _prepend_nvidia_pip_libs_to_ld_library_path()
    voice = _load_piper_voice(Path(model_path), use_cuda=use_cuda)
    sr = getattr(getattr(voice, "config", None), "sample_rate", None)
    try:
        sr_i = int(sr) if sr else fallback_sample_rate
    except Exception:
        sr_i = fallback_sample_rate
    _mp_bundle = {
        "voice": voice,
        "sample_rate": sr_i,
        "model_path": model_path,
        "speaker": speaker,
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
        "sentence_silence": sentence_silence,
    }


def _synthesize_one_ref(
    r: PanelRef,
    wav_path: Path,
    out_dir: Path,
    *,
    voice: PiperVoice | None,
    sample_rate: int,
    model_path: Path,
    piper_cli: str | None,
    force_cli: bool,
    skip_existing: bool,
    speaker: int | None,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
    sentence_silence: float,
) -> dict[str, Any]:
    """Tạo một file WAV và trả về dòng manifest (dùng cho tuần tự và đa tiến trình)."""
    if skip_existing and wav_path.is_file():
        dur = _wav_duration_seconds(wav_path)
        if dur > 0.0:
            return _manifest_row(r, wav_path, out_dir, dur)

    text = r.story_text.strip()
    if not text:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * int(0.1 * 22050))
    elif force_cli:
        if not piper_cli:
            raise RuntimeError("Thiếu lệnh `piper` trong PATH.")
        _synthesize_via_cli(piper_cli, model_path, text, wav_path)
    else:
        if voice is None:
            raise RuntimeError("Thiếu PiperVoice (force_cli=False).")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            base_kwargs: dict[str, Any] = {
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w": noise_w,
                "sentence_silence": sentence_silence,
            }
            if speaker is not None:
                for speaker_key in ("speaker_id", "speaker", "speaker_idx"):
                    try:
                        _synthesize_compat(
                            voice,
                            text,
                            wf,
                            {**base_kwargs, speaker_key: speaker},
                        )
                        break
                    except TypeError:
                        continue
                else:
                    _synthesize_compat(voice, text, wf, base_kwargs)
            else:
                _synthesize_compat(voice, text, wf, base_kwargs)

    if not _is_valid_wav(wav_path):
        if not piper_cli:
            raise RuntimeError(
                f"WAV sinh ra không hợp lệ và không tìm thấy lệnh `piper` để fallback. File: {wav_path}"
            )
        _synthesize_via_cli(piper_cli, model_path, text, wav_path)

    dur = _wav_duration_seconds(wav_path)
    return _manifest_row(r, wav_path, out_dir, dur)


def _tts_mp_task_cli(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: mỗi job gọi piper CLI (song song tốt; không dùng GPU API)."""
    r = _panelref_from_dict(payload["ref_dict"])
    out_dir = Path(payload["out_dir"])
    wav_path = _wav_path_for_beat(out_dir, r, layout=str(payload["layout"]))
    return _synthesize_one_ref(
        r,
        wav_path,
        out_dir,
        voice=None,
        sample_rate=int(payload.get("sample_rate") or 22050),
        model_path=Path(payload["model_path"]),
        piper_cli=payload.get("piper_cli"),
        force_cli=True,
        skip_existing=bool(payload.get("skip_existing")),
        speaker=payload.get("speaker"),
        length_scale=float(payload.get("length_scale", 1.0)),
        noise_scale=float(payload.get("noise_scale", 0.667)),
        noise_w=float(payload.get("noise_w", 0.8)),
        sentence_silence=float(payload.get("sentence_silence", 0.25)),
    )


def _tts_mp_task_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: dùng PiperVoice đã tải trong initializer (CPU, đa tiến trình)."""
    global _mp_bundle
    if _mp_bundle is None:
        raise RuntimeError("Worker chưa khởi tạo (thiếu initializer).")
    b = _mp_bundle
    r = _panelref_from_dict(payload["ref_dict"])
    out_dir = Path(payload["out_dir"])
    wav_path = _wav_path_for_beat(out_dir, r, layout=str(payload["layout"]))
    return _synthesize_one_ref(
        r,
        wav_path,
        out_dir,
        voice=b["voice"],
        sample_rate=int(b["sample_rate"]),
        model_path=Path(b["model_path"]),
        piper_cli=payload.get("piper_cli"),
        force_cli=False,
        skip_existing=bool(payload.get("skip_existing")),
        speaker=b["speaker"],
        length_scale=float(b["length_scale"]),
        noise_scale=float(b["noise_scale"]),
        noise_w=float(b["noise_w"]),
        sentence_silence=float(b["sentence_silence"]),
    )


def _build_mp_payload(
    r: PanelRef,
    *,
    out_dir: Path,
    layout: str,
    model_path: Path,
    piper_cli: str | None,
    skip_existing: bool,
    sample_rate: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "ref_dict": _panelref_to_dict(r),
        "out_dir": str(out_dir),
        "layout": layout,
        "model_path": str(model_path),
        "piper_cli": piper_cli,
        "skip_existing": skip_existing,
        "sample_rate": sample_rate,
        "speaker": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh giọng Ngọc Huyền cho từng panel.storyText (Piper TTS).")
    ap.add_argument(
        "--panels-root",
        type=Path,
        default=None,
        help="Thư mục panels_* (có index.json + page_XX/panels.json). Bỏ qua nếu dùng --scenes-json.",
    )
    ap.add_argument(
        "--scenes-json",
        type=Path,
        default=None,
        help="File scenes.json (review_two_pass / plan). Mặc định --layout auto → scene_NN.wav.",
    )
    ap.add_argument(
        "--layout",
        choices=("auto", "review", "panels"),
        default="auto",
        help="auto: scenes.json → review (scene_XX.wav); panels-root → page_XN/panel_YM.wav.",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Thư mục chứa scenes/scene_XX.* hoặc scene_XX.* (review). Mặc định: thư mục chứa scenes.json.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Tắt thanh tiến trình (tqdm).",
    )
    ap.add_argument("--model", type=Path, required=True, help="Đường dẫn file .onnx (config phải là .onnx.json cùng basename).")
    ap.add_argument("--output", type=Path, required=True, help="Thư mục output audio.")
    ap.add_argument("--speaker", type=int, default=None, help="Chọn speaker_id nếu model multi-speaker (mặc định None).")
    ap.add_argument("--sentence-silence", type=float, default=0.25, help="Khoảng nghỉ giữa câu (giây).")
    ap.add_argument("--length-scale", type=float, default=1.0, help="Tốc độ: <1 nhanh hơn, >1 chậm hơn.")
    ap.add_argument("--noise-scale", type=float, default=0.667, help="Piper noise_scale.")
    ap.add_argument("--noise-w", type=float, default=0.8, help="Piper noise_w.")
    ap.add_argument("--skip-existing", action="store_true", help="Bỏ qua nếu wav đã tồn tại.")
    ap.add_argument(
        "--force-cli",
        action="store_true",
        help="Luôn dùng lệnh `piper` CLI để sinh WAV (ổn định nhất).",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--cuda",
        "--gpu",
        dest="use_cuda",
        action="store_true",
        help="Chạy inference ONNX trên GPU (CUDAExecutionProvider). Cần onnxruntime-gpu và stack NVIDIA.",
    )
    g.add_argument(
        "--cpu",
        dest="use_cuda",
        action="store_false",
        help="Ép chạy CPU (mặc định).",
    )
    ap.set_defaults(use_cuda=False)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số tiến trình song song. >1 với --force-cli: nhiều piper CLI. >1 không CUDA: nhiều Piper API (CPU, mỗi worker một model). "
        "Không dùng >1 worker với --cuda (API GPU) — sẽ tự hạ về 1.",
    )
    args = ap.parse_args()

    if args.scenes_json is None and args.panels_root is None:
        raise SystemExit("Cần --panels-root hoặc --scenes-json.")
    if args.scenes_json is not None and args.panels_root is not None:
        raise SystemExit("Chỉ dùng một trong hai: --panels-root hoặc --scenes-json.")

    layout = args.layout
    if layout == "auto":
        layout = "review" if args.scenes_json else "panels"
    if layout == "review" and not args.scenes_json:
        raise SystemExit("--layout review cần --scenes-json.")

    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Không tìm thấy model: {model_path}")

    workers = max(1, int(args.workers))
    if workers > 1 and args.use_cuda and not args.force_cli:
        print(
            "Lưu ý: --cuda + API Python chỉ an toàn với 1 tiến trình (tránh nhiều bản model trên GPU). "
            "Đang chạy --workers 1. Muốn song song: dùng --force-cli --workers N (piper CLI, thường CPU).",
            file=sys.stderr,
        )
        workers = 1

    piper_cli = shutil.which("piper")
    voice: PiperVoice | None = None
    sample_rate = 22050

    if args.force_cli:
        if not piper_cli:
            raise SystemExit(
                "Bạn bật --force-cli nhưng không tìm thấy lệnh `piper` trong PATH.\n"
                "Cách khắc phục (trong env hiện tại):\n"
                "  pip install piper-tts\n"
                "Sau đó mở terminal mới và chạy lại, hoặc đảm bảo `which piper` có kết quả."
            )
        if args.use_cuda:
            raise SystemExit(
                "Không thể dùng --cuda/--gpu cùng --force-cli (CLI piper không được script này truyền GPU). "
                "Bỏ --force-cli để dùng API Python + CUDA."
            )
    elif workers == 1:
        try:
            voice = _load_piper_voice(model_path, use_cuda=args.use_cuda)
        except Exception as e:
            if args.use_cuda:
                raise SystemExit(
                    "Không khởi tạo được Piper trên GPU.\n"
                    f"Chi tiết: {e}\n\n"
                    "Kiểm tra:\n"
                    "  - pip install onnxruntime-gpu  (thường cần gỡ onnxruntime CPU nếu xung đột)\n"
                    "  - WSL2: nvidia-smi chạy được; driver NVIDIA trên Windows + CUDA trong WSL\n"
                    "  - Thử lại không GPU: bỏ --cuda\n"
                    "Cảnh báo DRM / card0 trên WSL thường vô hại nếu CUDA vẫn hoạt động."
                ) from e
            raise
        sr = getattr(getattr(voice, "config", None), "sample_rate", None)
        vc = getattr(voice, "voice_config", None)
        if isinstance(vc, dict):
            aud = vc.get("audio")
            if isinstance(aud, dict):
                sr = sr or aud.get("sample_rate")
        try:
            sample_rate = int(sr) if sr else 22050
        except Exception:
            sample_rate = 22050
    else:
        # Đa tiến trình + API: model chỉ tải trong worker (CPU).
        if args.use_cuda:
            raise SystemExit("Lỗi logic: workers>1 + CUDA không được phép tới đây.")

    scenes_path: Path | None = None
    if args.scenes_json is not None:
        scenes_path = args.scenes_json.expanduser().resolve()
        if not scenes_path.is_file():
            raise SystemExit(f"Không tìm thấy scenes.json: {scenes_path}")
        refs = _iter_scenes_json(scenes_path)
        if not refs:
            raise SystemExit("scenes.json không có mục hợp lệ (pageNumber + storyText).")
    else:
        panels_root = args.panels_root.expanduser().resolve()
        refs = _iter_panels(panels_root)
        if not refs:
            raise SystemExit("Không tìm thấy panel nào (panels.json).")

    if layout == "review":
        img_root = (
            args.images_dir.expanduser().resolve()
            if args.images_dir is not None
            else (scenes_path.parent if scenes_path else out_dir)
        )
        refs = [
            replace(r, rel_image=_review_scene_image_relpath(img_root, out_dir, r.page_number))
            for r in refs
        ]

    manifest: list[dict[str, Any]] = []

    if workers == 1:
        iterable: Iterable[PanelRef] = refs
        if not args.no_progress:
            unit = "scene" if layout == "review" else "panel"
            iterable = tqdm(
                refs,
                desc="TTS Piper",
                unit=unit,
                total=len(refs),
            )

        for r in iterable:
            wav_path = _wav_path_for_beat(out_dir, r, layout=layout)
            manifest.append(
                _synthesize_one_ref(
                    r,
                    wav_path,
                    out_dir,
                    voice=voice,
                    sample_rate=sample_rate,
                    model_path=model_path,
                    piper_cli=piper_cli,
                    force_cli=args.force_cli,
                    skip_existing=args.skip_existing,
                    speaker=args.speaker,
                    length_scale=args.length_scale,
                    noise_scale=args.noise_scale,
                    noise_w=args.noise_w,
                    sentence_silence=args.sentence_silence,
                )
            )
    else:
        payloads = [
            _build_mp_payload(
                r,
                out_dir=out_dir,
                layout=layout,
                model_path=model_path,
                piper_cli=piper_cli,
                skip_existing=args.skip_existing,
                sample_rate=sample_rate,
                args=args,
            )
            for r in refs
        ]
        manifest = []
        unit = "scene" if layout == "review" else "panel"
        if args.force_cli:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_tts_mp_task_cli, p) for p in payloads]
                done_iter = as_completed(futs)
                if not args.no_progress:
                    done_iter = tqdm(done_iter, desc="TTS Piper (parallel CLI)", unit=unit, total=len(futs))
                for fut in done_iter:
                    manifest.append(fut.result())
        else:
            initargs = (
                str(model_path),
                False,
                sample_rate,
                args.speaker,
                args.length_scale,
                args.noise_scale,
                args.noise_w,
                args.sentence_silence,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_tts_mp_initializer,
                initargs=initargs,
            ) as ex:
                futs = [ex.submit(_tts_mp_task_api, p) for p in payloads]
                done_iter = as_completed(futs)
                if not args.no_progress:
                    done_iter = tqdm(done_iter, desc="TTS Piper (parallel CPU)", unit=unit, total=len(futs))
                for fut in done_iter:
                    manifest.append(fut.result())
        manifest.sort(key=lambda row: (int(row["pageNumber"]), int(row["panelNumber"])))

    (out_dir / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

