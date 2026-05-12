#!/usr/bin/env python3
"""
Tách truyện thành các cảnh rồi sinh ảnh từng cảnh bằng Gemini API.
Logic tương đương mangagen: /api/plan + /api/generate — storybook (một ảnh/cảnh), review (nhiều ảnh minh họa nhỏ cho cùng một đoạn truyện), hoặc manga (--mode manga: full page nhiều panel).

Xác thực — một trong hai cách:

  A) Gemini Developer API (AI Studio):
     export GOOGLE_API_KEY=...

  B) Vertex AI (GCP): không dùng API key; dùng Application Default Credentials.
     gcloud auth application-default login
     export GOOGLE_CLOUD_PROJECT=your-project-id
     export GOOGLE_CLOUD_LOCATION=us-central1   # hoặc region có Gemini
     python pipeline.py --vertex --story story.txt ...

Ví dụ (ảnh nhân vật có sẵn):
  export GOOGLE_API_KEY=...
  python pipeline.py --story story.txt --char "Linh=refs/linh.png" --char "An=refs/an.png" -o out/

Tự động hoàn toàn (không cần biết trước có bao nhiêu nhân vật):
  python pipeline.py --story story.txt --auto-characters -o out/

  Pass 1 + pass 2 refine (mặc định) giúp nhân vật bám truyện; --no-refine-characters để bỏ pass 2.
  auto_refs/characters_draft.json = kết quả pass 1; characters_extracted.json = sau refine + trước khi vẽ portrait.

  Chế độ manga (lưới panel chữ nhật, không chữ trên ảnh — thoại dùng TTS/ghép sau):
  python pipeline.py --story story.txt --mode manga --auto-characters -o out/

  Chế độ review truyện (một đoạn văn → nhiều minh họa riêng, ví dụ về nhà / đưa quà / thoại / hai người nói chuyện):
  python pipeline.py --story story.txt --mode review --auto-characters -o out_review/
  # Gợi ý: --scenes 4 để ép đúng 4 ảnh cho một đoạn ngắn.

  Chỉ sinh ảnh từ plan đã có (không gọi lại planner):
  python pipeline.py --story story.txt -o out_review/ --from-scenes out_review/scenes.json
  # Liên kết hình trong cùng section (review + có outlineSectionNumber): thêm --section-image-continuity
  # Tiếp tục giữa chừng: thêm --resume

  Review 2 bước (dàn ý trước, beat sau — tránh JSON quá dài bị cắt):
  python pipeline.py --story story.txt --mode review --review-two-pass --auto-characters -o out_r2/
  Mỗi beat trong scenes.json có outlineSectionNumber + outlineSectionSummary (đoạn dàn ý); present beat trong cùng section khóa bối cảnh/trang phục, flashback được tách.

  Tiếp tục khi API lỗi giữa chừng:
  - Review 2 bước: tự đọc out_r2/checkpoints/review_two_pass.json nếu trùng truyện (không cần cờ).
  - Plan được ghi dần: scenes.json (beat) + checkpoints/review_outline.json (dàn ý) mỗi khi lưu checkpoint.
  - Portrait / ảnh scene: thêm --resume (bỏ qua file đã có).
  - Làm lại plan từ đầu: --fresh-checkpoint (xóa checkpoint review 2 bước).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


# Mặc định giống hướng mangagen/.env-sample; đổi qua biến môi trường nếu model trên tài khoản bạn khác
DEFAULT_PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gemini-2.5-flash")
DEFAULT_IMAGE_MODEL = os.environ.get(
    "CREATOR_IMAGE_MODEL_FLASH", "gemini-2.5-flash-image"
)


@dataclass
class Scene:
    page_number: int
    story_segment: str
    start_anchor: str
    end_anchor: str
    page_content: str
    # Đoạn truyện gốc nguyên văn thuộc cảnh
    story_text: str = ""
    panel_count: int = 1
    suggested_references: list[str] = field(default_factory=list)
    # Manga: nội dung truyện gán từng panel (đọc TTS / phụ đề), reading order L→R T→D
    panels: list[dict[str, Any]] = field(default_factory=list)
    # Review: "present" = hiện thực; "flashback" = hồi tưởng / ký ức (khác xử lý hình ảnh)
    narrative_plane: str = "present"
    # Review 2-pass: beat thuộc đoạn dàn ý nào (đánh dấu section trong scenes.json)
    outline_section_number: int | None = None
    outline_section_summary: str = ""
    # WAN 2.1 I2V motion prompt: chuỗi tiếng Anh mô tả CHUYỂN ĐỘNG cho beat này
    # (không tả lại nhân vật/wardrobe — ảnh tĩnh đã khoá). Dùng để đẩy thẳng vào WAN 2.1.
    motion_prompt: str = ""


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _json_loads_llm(text: str) -> Any:
    """
    Parse JSON từ LLM: cho phép control char trong chuỗi (strict=False) và thử làm sạch khi vẫn lỗi.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Model trả JSON rỗng.")

    def _scrub_c0(s: str) -> str:
        return "".join(c if (ord(c) >= 32 or c in "\n\t") else " " for c in s)

    def _extract_first_json_array(s: str) -> str:
        """
        Planner đôi khi trả kèm giải thích; ta cố trích mảng JSON đầu tiên dạng [...].
        Nếu không thấy, trả nguyên chuỗi.
        """
        if not s:
            return s
        a = s.find("[")
        b = s.rfind("]")
        if a >= 0 and b > a:
            return s[a : b + 1].strip()
        return s

    def _remove_trailing_commas(s: str) -> str:
        # Xóa dấu phẩy thừa trước } hoặc ]
        return re.sub(r",(\s*[}\]])", r"\1", s)

    def _quote_unquoted_object_keys(s: str) -> str:
        # Chỉ quote key dạng bareword: { pageNumber: 1 } hoặc , pageNumber: 1
        # Tránh đụng vào string vì lookbehind giới hạn ký tự trước đó.
        return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', s)

    def _jsonish_to_pythonish(s: str) -> str:
        # Cho ast.literal_eval: đổi null/true/false sang None/True/False
        s = re.sub(r"\bnull\b", "None", s)
        s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
        s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
        return s

    def _escape_unescaped_quotes_in_json(s: str) -> str:
        """
        Sửa JSON do LLM trả khi values chứa quote tiếng Việt/Anh không được escape, ví dụ:
            "k": "Anh nói: "xin chào.""
        → "k": "Anh nói: \"xin chào.\""
        Heuristic theo trạng thái in_string. Khi đang in_string mà gặp `"`:
          - Nếu ký tự non-ws kế tiếp thuộc {',', ':', '}', ']', kết-thúc-text} → đóng string.
          - Ngược lại coi là quote embedded → escape thành `\\"`.
        """
        out: list[str] = []
        i = 0
        n = len(s)
        in_str = False
        while i < n:
            ch = s[i]
            if not in_str:
                out.append(ch)
                if ch == '"':
                    in_str = True
                i += 1
                continue
            # đang trong string
            if ch == "\\":
                out.append(ch)
                i += 1
                if i < n:
                    out.append(s[i])
                    i += 1
                continue
            if ch == '"':
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                nxt = s[j] if j < n else ""
                if nxt in (",", ":", "}", "]", ""):
                    out.append(ch)
                    in_str = False
                    i += 1
                    continue
                out.append("\\\"")
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    variants = [
        text,
        text.replace("\r\n", "\n").replace("\r", "\n"),
        _scrub_c0(text.replace("\r\n", "\n").replace("\r", "\n")),
    ]
    last_err: json.JSONDecodeError | None = None
    for v in variants:
        try:
            return json.loads(v, strict=False)
        except json.JSONDecodeError as e:
            last_err = e

    # Fallback: cố "sửa nhẹ" JSON5-ish / output lẫn text.
    salvage = _extract_first_json_array(variants[-1])
    salvage = salvage.replace("\ufeff", "").strip()
    salvage = _remove_trailing_commas(salvage)
    salvage = _quote_unquoted_object_keys(salvage)
    try:
        return json.loads(salvage, strict=False)
    except json.JSONDecodeError:
        pass

    # Fallback bổ sung: escape các dấu " trong tiếng Việt/thoại không được model escape.
    quote_repaired = _escape_unescaped_quotes_in_json(salvage)
    if quote_repaired != salvage:
        try:
            return json.loads(quote_repaired, strict=False)
        except json.JSONDecodeError:
            try:
                return json.loads(_remove_trailing_commas(quote_repaired), strict=False)
            except json.JSONDecodeError:
                pass

    # Fallback bổ sung: dùng json_repair (lib chuyên trị JSON LLM hỏng) nếu có.
    try:
        import json_repair as _json_repair  # type: ignore

        for candidate in (quote_repaired, salvage, text):
            try:
                obj = _json_repair.loads(candidate)
                if obj is not None and obj != "":
                    return obj
            except Exception:
                continue
    except ImportError:
        pass

    # Fallback cuối: dùng ast.literal_eval (chịu được dấu '...' và một số JSON-ish),
    # sau khi chuyển null/true/false. Ưu tiên bản đã sửa quote.
    for src in (quote_repaired, salvage):
        pyish = _jsonish_to_pythonish(src)
        try:
            return ast.literal_eval(pyish)
        except Exception:
            continue

    # Hết cách → dump raw để debug rồi raise.
    try:
        debug_dir = Path(os.environ.get("LLM_DEBUG_DIR") or "/tmp")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"llm_bad_json_{int(time.time())}.txt"
        debug_path.write_text(text or "", encoding="utf-8")
        debug_hint = f" Raw output đã ghi vào {debug_path}."
    except Exception:
        debug_hint = ""

    raise ValueError(
        f"Không parse được JSON từ model (JSONDecodeError: {last_err}). "
        "Planner có thể đã trả về JSON không hợp lệ (JSON5-ish). "
        "Gợi ý: thử chạy lại, hoặc đổi PLANNER_MODEL, hoặc dùng --scenes để làm output ổn định hơn."
        + debug_hint
    ) from last_err


def _verbatim_between_anchors(story: str, start_anchor: str, end_anchor: str) -> str:
    """
    Trích đoạn nguyên văn từ story từ vị trí start_anchor đến hết end_anchor (lần xuất hiện đầu tiên).
    Dùng làm fallback khi planner không trả storyText.

    Lưu ý: anchor từ planner đôi khi nén nhiều dòng/blank line thành một newline duy nhất,
    còn story gốc lại có blank line / nhiều khoảng trắng — nên match whitespace-tolerant
    (mọi cụm whitespace trong anchor coi như \\s+ trong story).
    """
    if not story or not start_anchor.strip() or not end_anchor.strip():
        return ""

    def _find_with_ws_tolerance(haystack: str, needle: str, from_idx: int = 0) -> tuple[int, int]:
        i = haystack.find(needle, from_idx)
        if i >= 0:
            return i, i + len(needle)
        parts = [p for p in re.split(r"\s+", needle.strip()) if p]
        if not parts:
            return -1, -1
        pattern = r"\s+".join(re.escape(p) for p in parts)
        m = re.search(pattern, haystack[from_idx:], flags=re.DOTALL)
        if m is None:
            return -1, -1
        return from_idx + m.start(), from_idx + m.end()

    s_start, _ = _find_with_ws_tolerance(story, start_anchor, 0)
    if s_start < 0:
        return ""
    _, e_end = _find_with_ws_tolerance(story, end_anchor, s_start)
    if e_end <= s_start:
        return ""
    return story[s_start:e_end]


def _even_split_story_text(text: str, n: int) -> list[str]:
    """Chia đoạn văn thành n phần gần bằng nhau (fallback khi planner không trả panels)."""
    text = (text or "").strip()
    if not text or n < 1:
        return []
    if n == 1:
        return [text]
    total = len(text)
    base = total // n
    out: list[str] = []
    pos = 0
    for i in range(n):
        if i == n - 1:
            out.append(text[pos:].strip())
        else:
            chunk = text[pos : pos + base].strip()
            out.append(chunk)
            pos += base
    return [x for x in out if x]


def _trim_adjacent_story_overlap(prev_text: str, curr_text: str) -> tuple[str, str]:
    """
    Loại trùng giữa hai beat liền kề: beat sau không được lặp lại nguyên đuôi của beat trước.
    Ví dụ planner hay làm: beat trước kết thúc bằng cả câu \"… vào nhấn thích.\",
    beat sau chỉ là \"tôi lặng lẽ vào nhấn thích.\" → cắt đuôi beat trước để không overlap.
    """
    prev_text = (prev_text or "").strip()
    curr_text = (curr_text or "").strip()
    if not prev_text or not curr_text:
        return prev_text, curr_text
    # Beat sau trùng hoàn toàn làm hậu tố của beat trước
    if len(curr_text) <= len(prev_text) and prev_text.endswith(curr_text):
        new_prev = prev_text[: len(prev_text) - len(curr_text)].rstrip()
        while new_prev and new_prev[-1] in ",，;:—-":
            new_prev = new_prev[:-1].rstrip()
        return new_prev, curr_text
    # Overlap biên: đuôi(prev) == đầu(curr)
    max_k = min(len(prev_text), len(curr_text))
    for k in range(max_k, 0, -1):
        if prev_text[-k:] == curr_text[:k]:
            new_prev = prev_text[:-k].rstrip()
            while new_prev and new_prev[-1] in ",，;:—-":
                new_prev = new_prev[:-1].rstrip()
            return new_prev, curr_text
    return prev_text, curr_text


def _dedupe_review_story_text_overlaps(scenes: list[Scene]) -> int:
    """
    Chuẩn hoá storyText giữa các beat liền kề (theo page_number).
    Trả về số cặp đã chỉnh (mỗi cặp tối đa 1 lần).
    """
    if len(scenes) < 2:
        return 0
    idx_order = sorted(range(len(scenes)), key=lambda i: scenes[i].page_number)
    n_edges = 0
    for a in range(len(idx_order) - 1):
        i, j = idx_order[a], idx_order[a + 1]
        before = scenes[i].story_text
        after = scenes[j].story_text
        new_before, new_after = _trim_adjacent_story_overlap(before, after)
        if new_before != before or new_after != after:
            n_edges += 1
            scenes[i].story_text = new_before
            scenes[j].story_text = new_after
    return n_edges


def _count_tts_syllables(text: str) -> int:
    """
    Đếm "tiếng" (≈ syllable) cho text Việt-Anh để ước lượng thời lượng TTS.
    Heuristic: đếm "từ" tách bằng khoảng trắng / dấu câu, cộng thêm với từ tiếng Anh dài >1 syllable.
    """
    if not text:
        return 0
    cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [t for t in cleaned.split() if t]
    return len(tokens)


# Tốc độ đọc audiobook tiếng Việt ~3 tiếng/giây.
_TTS_SYLL_PER_SEC = 3.0
_TTS_BEAT_TARGET_SEC = 5.0
_TTS_BEAT_MAX_SEC = 8.0
_TTS_BEAT_TARGET_SYLL = int(_TTS_SYLL_PER_SEC * _TTS_BEAT_TARGET_SEC)  # 15
_TTS_BEAT_MAX_SYLL = int(_TTS_SYLL_PER_SEC * _TTS_BEAT_MAX_SEC)  # 24
_LONG_BEAT_AUTO_SPLIT_SEC = 10.0
_MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS = 5
_MOTION_BACKFILL_MAX_NEIGHBOR_RADIUS = 20


def _tts_seconds_for_text(text: str) -> float:
    return _count_tts_syllables(text or "") / _TTS_SYLL_PER_SEC


def _long_beat_syllable_threshold(tts_sec: float) -> int:
    return max(1, int(_TTS_SYLL_PER_SEC * float(tts_sec)))


def _warn_long_story_text_beats(scenes: list[Scene]) -> int:
    """
    Cảnh báo các beat có storyText vượt ngưỡng thời lượng (>8s ≈ >24 tiếng).
    Trả số beat vi phạm.
    """
    n_warn = 0
    for sc in scenes:
        n = _count_tts_syllables(sc.story_text or "")
        if n <= _TTS_BEAT_MAX_SYLL:
            continue
        n_warn += 1
        approx = n / _TTS_SYLL_PER_SEC
        sec_lbl = (
            f" [section {sc.outline_section_number}]"
            if getattr(sc, "outline_section_number", None) is not None
            else ""
        )
        snippet = (sc.story_text or "")[:80].replace("\n", " ")
        print(
            f"  [WARN] beat {sc.page_number}{sec_lbl} dài {n} tiếng (~{approx:.1f}s, "
            f"trần {_TTS_BEAT_MAX_SEC:.0f}s) — nên tách: {snippet}…",
            file=sys.stderr,
        )
    if n_warn:
        print(
            f"  → Tổng {n_warn} beat vi phạm rule 3–5s (max 8s). Cân nhắc rerun --plan-only "
            f"hoặc chỉnh tay scenes.json.",
            file=sys.stderr,
        )
    return n_warn


def _label_from_ref_filename(fn: str) -> str:
    """
    Suy ra tên nhân vật từ tên file ref (pattern phổ biến: số thứ tự + gạch dưới + tên):
      "01_Protagonist_Name.png"     → "Protagonist Name"
      "02_Titled_Character.png"     → "Titled Character"
      "03_Role_Alias_RealName.png"  → "Role Alias RealName"
    """
    if not fn:
        return ""
    base = Path(fn).stem
    base = re.sub(r"^\d+[_\-\s]+", "", base)
    base = base.replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)
    return base


# WAN 2.1 I2V (image-to-video) motion-prompt rules — DRY block, inject vào cả 2 planner prompts
# (2-pass beats_prompt và 1-pass review prompt). Buộc planner sinh ra prompt chuyển động cụ thể,
# verb đo lường được, một take liên tục, đúng quy ước WAN 2.1.
_WAN_MOTION_PROMPT_RULES: str = """\
9. motionPrompt — REQUIRED FIELD, MUST appear in EVERY beat object (never omit, never leave empty).
   This is a WAN 2.1 image-to-video (I2V) prompt that animates THIS beat's still image into a short clip
   (~3–5 seconds, matching the storyText TTS length). ALWAYS write in ENGLISH regardless of source language;
   use ASCII apostrophes (not curly quotes). Length target: 70–120 words.

   WAN 2.1 I2V CONVENTIONS — STRICT (follow ALL):
   (a) Do NOT re-describe wardrobe / identity / setting / lighting / palette — the input still image already locks all of those. Describe ONLY what MOVES during the clip.
   (b) SUBJECT LABELS — WAN I2V reads identity from the still image, NOT from story kinship or TTS.
       Applies to ANY genre (literary, romance, thriller, mystery, sci-fi, fantasy, horror, historical,
       slice-of-life, comedy, children's, nonfiction adaptation). Use short ENGLISH on-screen archetype
       roles only. Derive labels from pageContent + WHO IS WHERE (SPEAKER / LISTENERS / who is in frame),
       plus visible costume/posture when helpful — never copy a fixed genre template.
       Pattern: the [visible figure] + optional [wardrobe cue from pageContent] (e.g. "the barista in
       a green apron", "the detective in a trench coat", "the student in a school blazer", "the pilot
       in a flight suit", "the official in dark blue robes", "the shopkeeper behind the counter").
       FORBIDDEN as subject labels: kinship / relation words from storyText or narrator POV
       (father, mother, sister, brother, daughter, son, wife, husband, uncle, aunt, elder sister,
       younger sister, phu nhan, a ty, my father, her mother, etc.); source-language proper names;
       filename stems; suggestedReferences filenames.
       If storyText uses kinship but pageContent shows a visible social role, use the on-screen role
       (e.g. parent behind the counter → "the shopkeeper in an apron", NOT "mom"; colleague at a
       desk → "the analyst in a rolled-sleeve shirt", NOT "my brother"; official before a monarch
       → "the official in robes", NOT "the father").
       Repeat the SAME role label for a figure across consecutive beats in a section when the still
       shows the same person.
   (c) MOTION ENERGY — readable on screen in 5s. Prefer a short ACTION CHAIN (2–4 linked beats in ONE continuous take), not a single vague mood. Match pageContent + storyText emotional register:
       • calm / dialogue → speaking rhythm, head bobs, hand gestures, eye contact shifts;
       • tension / confrontation → clenched jaw, tightened fists, held breath, sharp head turns, weight shifts;
       • rescue / chase / impact → strain, pull, half-steps, supporting another body, urgent forward motion;
       • grief / outburst → tears at the rim, shoulders shake once, lean forward, sharp mouth movement;
       • env-only → wind, rain, smoke, dust, foliage, traffic, screens, flames, particles (no invented people).
   (d) CHARACTER MOTION — CONCRETE kinetic verbs anchored to measurable body parts. For EVERY on-screen character named in pageContent / WHO IS WHERE, give at least ONE visible motion (not only camera + ambient). Include at least 2 of:
       • Body: "chest rises and falls with one slow breath", "shoulders rise then drop", "she takes one half-step forward", "his torso leans forward 5 degrees", "weight shifts onto the front foot".
       • Head: "chin tilts down 5 degrees", "head turns 10 degrees toward [role]", "single slow nod", "chin lifts a touch", "eyes snap up and lock with [role]".
       • Eyes / face: "blinks once slowly", "lashes lower then lift", "eyes flick to the right then back", "gaze locks toward [role]", "the corners of his mouth pull upward into a sharp smirk", "jaw clenches", "brow lifts", "tears well at the rims".
       • Mouth: "lips part about half a centimetre and close again", "lips part and close in a clear speaking rhythm — short pauses every two beats".
       • Hands / contact: "right hand lifts and points sharply", "open palm extends toward [role]", "fingertips curl tighter into sleeve fabric", "fingers wrap around the cup", "one hand steadies [role]'s shoulder".
   (e) INTERACTION — when 2+ characters share the frame, choreograph reciprocal motion (speaker + listener): e.g. the barista slides a cup forward while the customer reaches; the instructor points at the board while a student in the front row nods; the pilot flips a switch while the copilot watches the readout; the official in robes bows while the monarch in the background nods once. Name the role each motion belongs to.
   (f) CAMERA — ONE continuous camera path for the whole clip (no cuts). MANDATORY in every CHARACTER beat unless env-only. Match pageContent shot size / angle; make zoom OR reframing READABLE over ~5s. Combine up to TWO of {zoom/dolly, angle/height, lateral/track} in ONE phrase when the still supports it:
       • ZOOM / DOLLY: "slow dolly-in from medium-wide to medium", "slow push-in toward the face", "slow pull-back from medium to medium-wide", "micro-zoom-in (~10% over the clip)", "slow dolly-in tightening from waist-up to medium close-up".
       • ANGLE / HEIGHT: "eye-level", "slight low-angle (5–15 degrees)", "high-angle wide", "over-the-shoulder from behind [role]", "camera drifts from eye-level to a 10-degree low-angle", "tilt down 5 degrees toward the hands".
       • LATERAL / TRACK: "slow lateral pan-right", "slow truck-left following her movement", "slow quarter-orbit around the subject".
       • HOLD: "static eye-level hold" only when pageContent is already a tight emotional close-up and subject motion carries the shot — still name angle (eye-level / low-angle / high-angle).
       FORBIDDEN camera language: jump-cut zoom, whip-pan, 360 spin, "cuts to new angle", rack focus between subjects, montage, cross-dissolve.
   (g) Ambient motion if appropriate — match pageContent setting only: "rain streaks down the window", "neon signage flickers", "leaves flutter in the breeze", "distant traffic blurs past", "dust motes drift in a sunbeam", "steam rises from a mug", "embers and smoke swirl", "holographic UI elements pulse".
   (h) FORBIDDEN softeners on CHARACTER motion — WAN treats these as "frozen subject" and animates only camera + ambient: "subtle", "subtly", "barely", "imperceptible", "a fraction", "almost", "slight movement", "slightly", "gentle", "gently", "soft", "softly", "quietly", "faint", "faintest", "minimal", "minimally", "hushed", "delicate", "nearly motionless", "remains motionless", "stands still" (unless the SAME sentence also gives an explicit body-anchor verb with a measurable amount). Allowed: camera geometry ("slight low-angle", "micro push-in") and one short memory-grade / DoF clause per flashback beat (rule k).
   (i) ONE CONTINUOUS TAKE — WAN cannot cut, dissolve, or rack-focus across multiple subjects mid-clip. FORBIDDEN in motionPrompt: "montage", "cross-dissolve", "cuts to", "rack focus", "split screen", "then we cut to". Pick ONE primary action chain in ONE place; camera zoom/angle change must be one smooth path, not a new shot.
   (j) DIALOGUE beats (a character speaks): WAN cannot lipsync from text alone. Describe lips/mouth motion in clear speaking rhythm with body anchors (head bobs, breath, hand gesture), then APPEND at the end: "(lipsync polished in post)".
   (k) FLASHBACK beats: briefly carry the memory-grade register from pageContent into the motion clip (e.g. "cool desaturated memory grade with light grain", "sepia memory tone with vignette"). One short clause is enough.
   (l) ENV-ONLY beats (no characters in pageContent): describe ONLY environment motion (rain on glass, traffic blur, mist, dust, foliage, screens, neon, steam, embers, particles). Do NOT add the "Subject motion priority" suffix.
   (m) END every CHARACTER beat with this exact cue line: "Subject motion priority over camera; characters move clearly, not frozen." Then add duration + framerate: "5s, 24fps" (use "3s, 24fps" for very short story lines, or "6s, 24fps" for the rare beat near the 8s ceiling).
   (n) NO source-language proper nouns inside motionPrompt. NO rendered text. NO captions / subtitles / speech bubbles / signs. NO kinship/relation labels when a visual archetype role is available in pageContent (see rule b).

   STRUCTURE — write motionPrompt in this order, as ONE plain prose paragraph (no bullets):
     [subject action chain with concrete kinetic verbs + body anchors, plus interaction if multi-character] [one continuous camera path: zoom/dolly and/or angle/height and/or lateral track] [ambient motion if appropriate] [memory-grade tag if flashback] [Subject motion priority cue if a character is in frame] [duration + fps tag]

   GOOD EXAMPLE (historical dialogue close-up):
     "The official in dark blue robes slowly tilts his chin down 5 degrees in a deeper bow, holds for a beat, then lifts his chin a touch back up; the right corner of his mouth pulls upward into a sharp smirk while his eyes narrow with proud satisfaction; he inhales once, shoulders rising and falling; behind him the elderly monarch in soft-focus gives a clear single nod. Slow micro push-in, slight low-angle, medium close-up. Loose hair strands flutter in the breeze, leaves drift through the bokeh. Subject motion priority over camera; characters move clearly, not frozen. (lipsync polished in post) 5s, 24fps."

   GOOD EXAMPLE (modern office dialogue):
     "The manager in a charcoal blazer speaks with clear lip rhythm, chin dipping once per phrase; her right hand lifts an open palm toward the conference table, then lowers; the analyst in a rolled-sleeve shirt across the table nods once and leans forward 5 degrees. Slow push-in from medium to medium close-up, eye-level. Ceiling lights shimmer, blinds sway at the window. Subject motion priority over camera; characters move clearly, not frozen. (lipsync polished in post) 5s, 24fps."

   GOOD EXAMPLE (sci-fi cockpit interaction):
     "The pilot in a flight suit flips one switch on the console and tightens their grip on the yoke; the copilot beside them turns their head 15 degrees toward the warning panel and taps the screen once; both torsos rock with a short turbulence jolt. Slow lateral truck-right, eye-level medium shot. Instrument lights pulse, HUD reflections drift across the canopy. Subject motion priority over camera; characters move clearly, not frozen. 5s, 24fps."

   GOOD EXAMPLE (physical interaction / rescue):
     "The cloaked figure in worn fatigues strains to support the injured soldier, his arm slung over her shoulder; she pulls him one half-step away from the smoke, his head lolling; embers swirl, distant explosions glow at the horizon, dust billows past them. Handheld-feel medium shot, slight organic shake, eye-level, urgent forward motion. Cool desaturated war-memory grade, sparks and smoke flicker. Subject motion priority over camera; characters move clearly, not frozen. 5s, 24fps."

   GOOD EXAMPLE (eye-contact interaction + camera drift):
     "The student in a school blazer suddenly lifts her head; her eyes snap up and lock with the classmate's steady gaze across the aisle; he stares back unmoving; a held tense beat, neither blinking. Slow push-in from medium close-up to tight close-up while the angle drifts to eye-level, background falling into shallow DoF, only light hair sway. Subject motion priority over camera; characters move clearly, not frozen. 5s, 24fps."

   GOOD EXAMPLE (env-only):
     "Rain streaks down a city window while headlights smear into blurred moving bands outside; a ceiling lamp flickers once; condensation beads slide down the glass. Very slow push-in toward the window, eye-level interior wide. Cool blue night palette, drifting rain haze. 5s, 24fps."

   BAD EXAMPLE (wrong labels — any genre):
     "Dad's lips part in a calm speaking rhythm while Mom nods by the counter." / "The father's lips part while the emperor listens over his shoulder."
     → kinship or informal family labels from storyText are not on-screen visual roles; WAN mis-assigns motion. Replace with roles from pageContent/WHO IS WHERE for THIS still (e.g. "the barista in a green apron ... the customer in a hoodie", "the manager in a charcoal blazer ... the analyst across the table", or "the official in dark robes ... the elderly monarch over his shoulder" — match the image, do not copy a fixed template).

   BAD EXAMPLE (DO NOT emit anything like this):
     "She has a subtle smile, lips barely move, almost imperceptible breath. Slow rack focus from her to him then montage of memories." → multiple forbidden tokens; WAN will produce a frozen still.

HARD REMINDER — `motionPrompt` is FIELD #9 of the required JSON schema, NOT optional commentary.
Every beat object you emit MUST contain a non-empty `motionPrompt` string. If a beat is reflective or
mostly internal monologue with no obvious motion, default to a minimal valid prompt such as:
"The speaker in frame draws one slow breath; chest rises and falls once; lashes lower then lift in a single slow blink; chin tilts down half an inch then settles; fingertips curl once into sleeve fabric. Static eye-level hold, medium close-up. Ambient air movement around hair and collar. Subject motion priority over camera; characters move clearly, not frozen. 5s, 24fps."
NEVER omit `motionPrompt`. NEVER set it to null, empty string, "TBD", or a placeholder.
"""

# Planner beat-split rules — split by illustration rhythm (one still per visual beat), not grammar alone.
_VISUAL_RHYTHM_SPLIT_RULES: str = """\
VISUAL RHYTHM SPLIT — PLAN BY ILLUSTRATION BEATS, NOT GRAMMAR ALONE (CRITICAL):
- Split by what could be ONE still frame + ONE short motion clip, NOT by how many commas a sentence has.
- Before finalizing each beat, count DISTINCT visual beats inside its storyText:
  • a new physical action or gesture (mask on, carry, sit, feed, apply medicine, stand, turn);
  • a new place or staging (open road → abandoned hut interior → bedside);
  • a new time slice that changes the tableau (arrival vs multi-day vigil vs recovery / "pulled back from death's door");
  • a new prop or contact (bowl, medicine, bed, scroll);
  • a new dialogue turn or emotional beat.
  If you count 2+ DISTINCT visuals, SPLIT into 2+ beats with verbatim storyText slices.
- One long source sentence often becomes 2–4 beats when the narration walks through multiple illustrated moments.
- HARD CHECK: pageContent must describe ONLY what storyText covers for THAT beat. If pageContent shows bedside nursing but storyText also covers carrying someone into the hut and pulling them back from death's door, you MERGED too much — split storyText first, then write pageContent per slice.
- When the visual beat changes, choose a NEW shot / staging (wide arrival → medium care → close-up recovery), even inside flashback.
- Do NOT merge unlike visuals to save JSON size. Prefer MORE beats over one overloaded image.

Example D — one long sentence → 2–3 images (STRUCTURE only; anchors/storyText must be VERBATIM from the user's source):
  Source (fictional, any language): "She masked her face, carried him into an abandoned watchpost, shared the bed for three days and nights, fed him and changed his medicine, and fiercely pulled him back from the edge of death."
  WRONG → 1 beat / 1 image cramming mask + carry + multi-day vigil + recovery.
  CORRECT → often 2–3 beats:
    • Beat 1 (flashback): masked carry into the abandoned post — wide / medium, movement.
    • Beat 2 (flashback): bedside vigil — feeding water, applying medicine — medium close-up; "three days and nights" may stay in this beat only if it is ONE continuous nursing tableau.
    • Beat 3 (flashback): revival / edge-of-death — close-up on breath, eyes, or gripping hands — outcome beat.
  If arrival, vigil, and outcome are each drawable as separate moments, prefer 3 beats over 2.
"""


# Prefix kính ngữ / danh xưng / chức danh phổ biến — dùng để cắt prefix khi match label nhân vật
# với pageContent (vì pageContent có thể chỉ gọi tên thật). Bao quát đa ngôn ngữ / đa thể loại;
# lower-case, KHÔNG dấu chấm cuối; match xử lý ở dạng chuẩn hoá.
_HONORIFIC_PREFIXES: tuple[str, ...] = (
    # ── Cổ trang Trung-Việt: hoàng cung & quý tộc ──
    "thái tử", "thái thượng hoàng", "thái hậu", "hoàng đế", "hoàng thượng",
    "hoàng hậu", "hoàng phi", "thái phi", "quý phi", "đức phi", "thần phi",
    "đại hoàng tử", "nhị hoàng tử", "tam hoàng tử", "tứ hoàng tử", "ngũ hoàng tử",
    "lục hoàng tử", "thất hoàng tử", "bát hoàng tử", "cửu hoàng tử",
    "hoàng tử", "hoàng nữ", "công chúa", "trưởng công chúa", "đại trưởng công chúa",
    "quận chúa", "quận vương", "vương gia", "vương phi", "vương phu",
    "hầu gia", "hầu phu nhân", "phu nhân", "đại nhân", "lão phu nhân",
    "tướng quân", "đại tướng quân", "đô đốc", "tổng quản", "thống lĩnh",
    "quốc sư", "thừa tướng", "tể tướng", "thượng thư", "đại học sĩ",
    # ── Cổ trang Trung-Việt: võ lâm / tu tiên ──
    "chưởng môn", "trưởng lão", "lão tổ", "đại sư huynh", "nhị sư huynh", "tam sư huynh",
    "đại sư tỷ", "nhị sư tỷ", "tam sư tỷ",
    "sư phụ", "sư mẫu", "sư bá", "sư thúc", "sư cô", "sư huynh", "sư tỷ",
    "sư đệ", "sư muội", "tổ sư",
    "đạo trưởng", "phương trượng", "trụ trì", "đại đức", "thiền sư", "đạo nhân",
    # ── Gia đình & xưng hô cổ ──
    "a tỷ", "a huynh", "a đệ", "a muội", "đại tỷ", "đại huynh", "đại ca",
    "tiểu muội", "tiểu đệ", "tiểu muội muội",
    "tiểu thư", "đại tiểu thư", "nhị tiểu thư", "tam tiểu thư",
    "công tử", "đại công tử", "thiếu gia", "tiểu thiếu gia", "lão gia",
    "thế tử", "tiểu vương gia", "tiểu hầu gia",
    "phụ thân", "mẫu thân", "huynh trưởng", "tỷ tỷ", "muội muội", "đệ đệ", "ca ca",
    "di nương", "thứ mẫu", "kế mẫu",
    # ── Gia đình tiếng Việt hiện đại / cũ ──
    "ông", "bà", "cụ", "cố", "tổ", "bác", "chú", "thím", "cô", "dì", "cậu", "mợ",
    "anh", "chị", "em", "con", "cha", "mẹ", "ba", "má", "bố", "u",
    "ngoại", "nội", "ông ngoại", "bà ngoại", "ông nội", "bà nội",
    # ── Nghề / chức danh tiếng Việt ──
    "thầy", "cô giáo", "thầy giáo", "bác sĩ", "y tá", "điều dưỡng",
    "giám đốc", "tổng giám đốc", "phó giám đốc", "chủ tịch", "tổng thống", "thủ tướng",
    "trưởng phòng", "phó phòng", "trưởng ban", "trưởng nhóm",
    "kỹ sư", "luật sư", "giáo sư", "phó giáo sư", "tiến sĩ", "thạc sĩ", "cử nhân",
    "đại tá", "trung tá", "thiếu tá", "đại uý", "trung uý", "thiếu uý", "đại tướng",
    "trung tướng", "thiếu tướng", "đại sư", "linh mục", "giám mục", "hồng y",
    # ── Tiếng Anh: titles + ranks ──
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor",
    "sir", "lord", "lady", "dame", "madam", "madame", "ma'am", "rev", "reverend",
    "father", "mother", "brother", "sister", "uncle", "aunt", "auntie", "grandpa",
    "grandma", "grandfather", "grandmother",
    "captain", "lieutenant", "colonel", "general", "major", "sergeant", "corporal",
    "admiral", "commander", "private",
    "king", "queen", "prince", "princess", "duke", "duchess", "earl", "count",
    "countess", "baron", "baroness", "lord chancellor",
    "president", "senator", "governor", "mayor", "minister", "ambassador",
    "officer", "detective", "inspector", "agent", "judge",
    # ── Phương ngữ / dân dã ──
    "lão", "tiểu", "đại", "nhị", "tam", "tứ",  # các tiền tố lone-token cổ trang
    # ── Tiếng Pháp ──
    "monsieur", "madame", "mademoiselle", "m", "mme", "mlle",
    # ── Tiếng Đức / Tây Ban Nha ──
    "herr", "frau", "fräulein", "señor", "señora", "señorita", "don", "doña",
    # ── Hàn / Nhật (tiền tố hiếm; suffix thường nằm cuối) ──
    "oppa", "hyung", "noona", "unnie", "samchon", "ahjussi", "ahjumma",
)


def _label_match_variants(label: str) -> list[str]:
    """
    Sinh nhiều biến thể của 1 label nhân vật để match với pageContent (đa dạng cách gọi tên):
      - full label
      - label bỏ prefix kính ngữ / danh xưng / title (cổ trang, gia đình VN, nghề, English)
      - last 2 tokens
      - last 1 token nếu đủ dài (≥ 3 ký tự)
    Hàm trung tính với mọi thể loại truyện; danh sách prefix là set tĩnh ở module-level
    (`_HONORIFIC_PREFIXES`), có thể mở rộng khi gặp thể loại mới.
    """
    out: list[str] = []
    if not label:
        return out
    out.append(label)

    def _norm(s: str) -> str:
        return s.rstrip(".").lower()

    low = _norm(label)
    # Phần label sau khi cắt prefix (nếu có); dùng làm nền cho last-N-tokens để tránh "giáo Lan" / "sĩ Khoa".
    stripped = label
    for pre in sorted(_HONORIFIC_PREFIXES, key=len, reverse=True):
        if low.startswith(pre + " ") and len(label) > len(pre) + 1:
            cand = label[len(pre) + 1 :].strip()
            if cand:
                out.append(cand)
                stripped = cand
            break
        if "." in label:
            head_dot = label.split(".", 1)
            if len(head_dot) == 2 and _norm(head_dot[0]) == pre:
                rest = head_dot[1].strip()
                if rest:
                    out.append(rest)
                    stripped = rest
                    break

    tokens = [t for t in stripped.split() if t]
    if len(tokens) >= 2:
        out.append(" ".join(tokens[-2:]))
    if tokens and len(tokens[-1]) >= 3:
        out.append(tokens[-1])
    seen: set[str] = set()
    uniq: list[str] = []
    for v in out:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq


def _label_appears_in_text(label: str, text: str) -> bool:
    """
    Match label với text qua nhiều biến thể; case-insensitive, có dùng phiên bản đã loại dấu (NFD).
    """
    if not label or not text:
        return False
    text_low = text.lower()
    text_fold = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()
    for variant in _label_match_variants(label):
        v_low = variant.lower()
        if v_low and v_low in text_low:
            return True
        v_fold = (
            unicodedata.normalize("NFD", variant)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        if v_fold and v_fold in text_fold:
            return True
    return False


# Module-level character aliases map: filename → list of names (canonicalName + aliases) đã được
# LLM trích sẵn từ truyện cụ thể. Đây là nguồn ground truth tốt nhất cho match co-presence.
# Set bởi `_configure_character_aliases()` từ `main()`; đọc bởi `_propagate_copresent_cast_in_scenes`
# và `_aliases_for_ref_filename`.
_CHARACTER_ALIASES_MAP: dict[str, list[str]] = {}


def _alias_fingerprint(s: str) -> str:
    """
    Tạo dấu vân tay gọn để match flexible giữa filename và canonicalName:
    bỏ dấu (NFD strip) + lower-case + chỉ giữ alphanum + bỏ leading digits (số thứ tự).
    """
    if not s:
        return ""
    folded = (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    folded = re.sub(r"[^a-z0-9]+", "", folded)
    folded = re.sub(r"^\d+", "", folded)
    return folded


def _aliases_for_ref_filename(fn: str) -> list[str]:
    """
    Trả về danh sách 'cách gọi' của nhân vật cho một filename ref.
    Ưu tiên dữ liệu từ `_CHARACTER_ALIASES_MAP` (do LLM trích từ chính truyện);
    fallback chỉ dùng filename-derived label.

    Lookup theo thứ tự: filename đầy đủ → stem → label → fingerprint (bỏ dấu / số / ký tự đặc biệt).
    """
    if not fn:
        return []
    base = Path(fn).name
    if base in _CHARACTER_ALIASES_MAP:
        return list(_CHARACTER_ALIASES_MAP[base])
    stem = Path(fn).stem
    if stem in _CHARACTER_ALIASES_MAP:
        return list(_CHARACTER_ALIASES_MAP[stem])
    label = _label_from_ref_filename(fn)
    if label and label in _CHARACTER_ALIASES_MAP:
        return list(_CHARACTER_ALIASES_MAP[label])
    # Fingerprint lookup: tìm key có cùng dấu vân tay (bỏ qua ngoặc / dấu / ký tự đặc biệt).
    fp = _alias_fingerprint(stem) or _alias_fingerprint(label or "")
    if fp:
        for k, v in _CHARACTER_ALIASES_MAP.items():
            if _alias_fingerprint(k) == fp:
                return list(v)
    return [label] if label else []


def _character_appears_in_text(label_or_filename: str, text: str) -> bool:
    """
    Match một nhân vật (filename hoặc label) với text qua TOÀN BỘ aliases:
      • Aliases từ characters_extracted.json (LLM trích từ truyện) — nguồn ưu tiên.
      • Plus filename-derived label + variants (cắt prefix kính ngữ).
    Bỏ qua alias quá ngắn/quá generic (1-2 ký tự, hoặc đại từ ngôi như "ta", "hắn", "nàng",
    "y", "thị") để tránh false-positive.
    """
    if not text:
        return False
    names: list[str] = []
    seen: set[str] = set()
    for n in _aliases_for_ref_filename(label_or_filename):
        if not isinstance(n, str):
            continue
        s = n.strip()
        if not s:
            continue
        # Bỏ alias quá ngắn / đại từ chung dễ false-positive
        if len(s) < 3:
            continue
        if s.lower() in _GENERIC_ALIAS_BLACKLIST:
            continue
        if s.lower() not in seen:
            seen.add(s.lower())
            names.append(s)
    # Nếu aliases_map không có (fallback) → ít nhất dùng label-derived variants
    if not names:
        label = (
            _label_from_ref_filename(label_or_filename)
            if "/" in label_or_filename or "\\" in label_or_filename or "." in label_or_filename
            else label_or_filename
        )
        names = [label] if label else []

    for nm in names:
        if _label_appears_in_text(nm, text):
            return True
    return False


# Đại từ / xưng hô chung — dễ false-match nếu coi là "alias nhân vật" (vì xuất hiện khắp nơi).
_GENERIC_ALIAS_BLACKLIST: set[str] = {
    # Vietnamese pronouns / generic refs
    "ta", "tôi", "tao", "tớ", "mình", "ngươi", "mày", "nó", "y", "thị", "vị",
    "hắn", "nàng", "chàng", "em", "anh", "chị", "ông", "bà",
    "ai đó", "người ấy", "người đó", "kẻ kia", "người kia",
    # English pronouns
    "he", "she", "it", "they", "them", "him", "her", "his", "hers", "their",
    "i", "you", "we", "us", "me", "my", "the one", "someone",
    # Chinese-Vietnamese 3rd-person from xianxia (already covered above; kept here for clarity)
    "tiểu tử", "tiểu nha đầu", "lão phu", "tại hạ",
}


def _safe_filename_basename(name: str) -> str:
    """Bản nhẹ của _safe_filename, dùng để build alias key (không phụ thuộc fwd-decl của _safe_filename)."""
    s = re.sub(r"[^\w\-.\u00C0-\u1EF9 ]", "_", name).strip()
    s = re.sub(r"\s+", "_", s)
    return s or "unnamed"


def _configure_character_aliases(
    *,
    out_dir: Path | None = None,
    extracted_rows: list[dict[str, Any]] | None = None,
) -> None:
    """
    Nạp aliases nhân vật vào `_CHARACTER_ALIASES_MAP` để dùng cho match co-presence.

    Ưu tiên 1: nếu có `extracted_rows` (vừa chạy auto_build_character_refs xong) — dùng trực tiếp.
    Ưu tiên 2: đọc {out_dir}/auto_refs/characters_extracted.json nếu tồn tại.

    Map có 3 dạng key (mỗi nhân vật map cùng giá trị):
      • Tên file đầy đủ ("01_Linh_Lung.png") nếu suy ra được index từ vị trí trong rows.
      • Stem ("01_Linh_Lung").
      • Label dạng "Linh Lung" (canonicalName).
    Lý do: `_aliases_for_ref_filename` thử cả 3 trước khi fallback.
    """
    rows: list[dict[str, Any]] = []
    if extracted_rows:
        rows = [r for r in extracted_rows if isinstance(r, dict)]
    elif out_dir is not None:
        manifest = out_dir / "auto_refs" / "characters_extracted.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rows = [r for r in data if isinstance(r, dict)]
            except (json.JSONDecodeError, OSError):
                pass
    _CHARACTER_ALIASES_MAP.clear()
    if not rows:
        return

    new_map: dict[str, list[str]] = {}
    for idx, row in enumerate(rows, start=1):
        canonical = str(row.get("canonicalName") or "").strip()
        if not canonical:
            continue
        # Tên hiển thị + alias text. Bỏ ngoặc quanh tên thật (vd. "A tỷ (Cố Thiên Từ)").
        names: list[str] = [canonical]
        m = re.search(r"\(([^)]+)\)", canonical)
        if m:
            names.append(m.group(1).strip())
            names.append(re.sub(r"\s*\([^)]+\)", "", canonical).strip())
        for a in row.get("aliases") or []:
            if isinstance(a, str) and a.strip():
                names.append(a.strip())

        # Build key candidates: filename stem (số thứ tự + safe canonical) + canonical chính.
        safe = _safe_filename_basename(canonical)
        stems = [f"{idx:02d}_{safe}"]
        keys: list[str] = []
        for stem in stems:
            keys.append(stem)
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                keys.append(stem + ext)
        keys.append(canonical)

        for k in keys:
            new_map[k] = names
    _CHARACTER_ALIASES_MAP.update(new_map)


def _propagate_copresent_cast_in_scenes(scenes: list[Scene]) -> int:
    """
    CO-PRESENT CAST CLEANUP & PROPAGATION:

    Mục tiêu: trong một mạch beat liên tiếp cùng (outlineSectionNumber, narrativePlane),
    chuẩn hoá `suggestedReferences` thành đúng những nhân vật ĐANG VẬT LÝ CÓ MẶT trong
    không gian — bất kể có là người nói, người nghe yên lặng, hay đứng cạnh.

    Quy tắc chính (rất quan trọng):
      1) Ref được giữ / propagate KHI tên nhân vật xuất hiện trong `pageContent` (visual brief)
         của ÍT NHẤT một beat trong mạch. Đây là tín hiệu "nhân vật được mô tả trong khung".
      2) Ref bị LOẠI nếu chỉ xuất hiện trong `storyText` (thoại / lời kể) mà không có trong
         `pageContent` của bất kỳ beat nào trong mạch — coi như chỉ được nhắc đến chứ không
         có mặt vật lý (vd. nhân vật A nói về nhân vật B nhưng B không ở đó).
      3) An toàn: nếu loại bỏ làm cho 1 beat còn 0 ref, GIỮ LẠI bộ ref cũ của riêng beat đó.

    Mạch (group) bị cắt khi:
      • outline_section_number đổi, hoặc
      • narrative_plane đổi (present ↔ flashback).

    Trả về tổng số beat thay đổi (cộng add lẫn drop).
    """
    if not scenes:
        return 0

    def _ref_key(s: str) -> str:
        return (s or "").strip().lower()

    n_added = 0
    n_dropped = 0
    i = 0
    while i < len(scenes):
        head = scenes[i]
        sec = getattr(head, "outline_section_number", None)
        plane = getattr(head, "narrative_plane", None)
        j = i + 1
        while j < len(scenes):
            nxt = scenes[j]
            if (
                getattr(nxt, "outline_section_number", None) != sec
                or getattr(nxt, "narrative_plane", None) != plane
            ):
                break
            j += 1

        # Tập hợp toàn bộ ref đã được planner gán cho mạch này
        candidates: list[str] = []
        seen_c: set[str] = set()
        for k in range(i, j):
            for r in scenes[k].suggested_references or []:
                rk = _ref_key(r)
                if rk and rk not in seen_c:
                    seen_c.add(rk)
                    candidates.append(r)

        # Ghép pageContent của cả mạch để kiểm tra "vật lý có mặt"
        pc_blob = "\n".join((scenes[k].page_content or "") for k in range(i, j))

        # Phân loại ref: physically present vs mention-only.
        # Match mỗi ref qua TOÀN BỘ aliases (canonicalName + aliases LLM đã trích từ truyện)
        # — nguồn ground truth "thông minh"; fallback dùng filename label + variants.
        physical: list[str] = []
        mention_only: list[str] = []
        for r in candidates:
            if _character_appears_in_text(r, pc_blob):
                physical.append(r)
            else:
                mention_only.append(r)

        # Nếu mạch không có ai vật lý → giữ nguyên (planner có thể đúng theo cách khác)
        if not physical:
            i = j
            continue

        physical_keys = {_ref_key(r) for r in physical}

        for k in range(i, j):
            old = list(scenes[k].suggested_references or [])
            old_keys = {_ref_key(r) for r in old}

            # add: các ref vật lý có mặt thiếu khỏi beat này
            added = [r for r in physical if _ref_key(r) not in old_keys]
            # drop: các ref hiện có nhưng không thuộc nhóm vật lý (mention-only)
            dropped = [r for r in old if _ref_key(r) not in physical_keys]

            new_refs = [r for r in old if _ref_key(r) in physical_keys] + added

            # An toàn: không để beat còn 0 ref. Nếu drop → 0 thì rollback drop, chỉ thêm.
            if not new_refs:
                new_refs = list(old) + added
                dropped = []

            if added:
                n_added += 1
            if dropped:
                n_dropped += 1

            scenes[k].suggested_references = new_refs

        i = j

    if n_added or n_dropped:
        print(
            f"  [INFO] Co-present cast cleanup: thêm refs cho {n_added} beat, "
            f"loại refs chỉ-được-nhắc-trong-thoại khỏi {n_dropped} beat "
            "(giữ đúng những người vật lý có mặt trong cảnh).",
            file=sys.stderr,
        )
    return n_added + n_dropped


def _parse_panels_from_row(raw: Any, max_count: int) -> list[dict[str, Any]]:
    """Đọc mảng panels từ JSON planner; sắp xếp, cắt dư, đánh số 1..n (n <= max_count)."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for j, p in enumerate(raw):
        if not isinstance(p, dict):
            continue
        try:
            pn = int(p.get("panelNumber", j + 1))
        except (TypeError, ValueError):
            pn = j + 1
        st = str(p.get("storyText") or "").strip()
        out.append({"panelNumber": pn, "storyText": st})
    out.sort(key=lambda x: int(x.get("panelNumber", 0)))
    if len(out) > max_count:
        out = out[:max_count]
    for idx, item in enumerate(out, start=1):
        item["panelNumber"] = idx
    return out


def _asset_list_for_prompt(char_paths: dict[str, Path], style_paths: list[Path]) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for name, p in char_paths.items():
        assets.append({"name": p.name, "role": "character", "label": name})
    for p in style_paths:
        assets.append({"name": p.name, "role": "style"})
    return assets


def _parse_planner_json_array(
    raw: str,
    client: genai.Client,
    planner_model: str,
) -> list[Any]:
    """Parse JSON array từ planner; thử repair một lần nếu lỗi."""
    raw = _strip_json_fence((raw or "").strip())
    try:
        data = _json_loads_llm(raw)
    except ValueError:
        repair_prompt = f"""
You are a strict JSON repair tool.
Task: Convert the following text into a VALID JSON array of objects.

Rules:
- Output ONLY the JSON array, no markdown fences, no commentary.
- Preserve as much content as possible.
- If the text is truncated, close any open strings/objects/arrays and drop only the incomplete tail.
- Ensure all object keys are double-quoted.

TEXT TO REPAIR:
{raw}
"""
        repaired = client.models.generate_content(
            model=planner_model,
            contents=repair_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        ).text or ""
        repaired = _strip_json_fence(repaired)
        data = _json_loads_llm(repaired)
    if not isinstance(data, list):
        raise ValueError("Planner phải trả về một mảng JSON.")
    return data


def _story_fingerprint(story: str) -> str:
    """Hash ngắn để checkpoint khớp cùng bản truyện."""
    blob = str(len(story)) + "\n" + story[:24000]
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _is_transient_api_error(exc: Exception) -> bool:
    """Phát hiện lỗi tạm thời từ Gemini (503/429/500/timeout) để retry."""
    msg = str(exc) or ""
    upper = msg.upper()
    return (
        "503" in msg
        or "UNAVAILABLE" in upper
        or "429" in msg
        or "RESOURCE_EXHAUSTED" in upper
        or "DEADLINE_EXCEEDED" in upper
        or "500 " in msg
        or "INTERNAL" in upper
        or "TIMEOUT" in upper
    )


def _generate_with_transient_retry(
    fn,
    *,
    label: str,
    delays: tuple[int, ...] = (8, 20, 45, 90, 180),
):
    """
    Gọi `fn()` với retry-backoff cho lỗi server tạm thời (503/429/500/timeout).
    Trả response của fn(); nếu hết retry vẫn fail → trả None và in log.
    Lỗi non-transient được ném thẳng ra ngoài để dễ debug.
    """
    last_exc: Exception | None = None
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_api_error(exc):
                raise
            if attempt >= len(delays):
                break
            wait_s = delays[attempt]
            head = (str(exc).splitlines() or [""])[0][:160]
            print(
                f"  [WARN] {label}: lỗi tạm thời ({type(exc).__name__}: {head}). "
                f"Retry sau {wait_s}s (lần {attempt + 2}/{len(delays) + 1})...",
                file=sys.stderr,
            )
            time.sleep(wait_s)

    head = (str(last_exc).splitlines() or [""])[0][:200] if last_exc else "?"
    print(
        f"  [WARN] {label}: vẫn fail sau {len(delays) + 1} lần — bỏ qua. "
        f"Last error: {type(last_exc).__name__ if last_exc else '?'}: {head}",
        file=sys.stderr,
    )
    return None


# ---- Image-call throttling + retry (chống 429 RESOURCE_EXHAUSTED trên Vertex) ----
_IMAGE_MIN_INTERVAL_S: float = 0.0
_IMAGE_RETRY_DELAYS: tuple[int, ...] = (30, 60, 120, 240, 480)
_LAST_IMAGE_CALL_AT: float = 0.0


def _configure_image_pacing(
    *,
    min_interval_s: float | None = None,
    retry_delays: tuple[int, ...] | None = None,
) -> None:
    """Cấu hình throttle + retry cho mọi image-generation call. Gọi từ main()."""
    global _IMAGE_MIN_INTERVAL_S, _IMAGE_RETRY_DELAYS
    if min_interval_s is not None:
        _IMAGE_MIN_INTERVAL_S = max(0.0, float(min_interval_s))
    if retry_delays is not None and len(retry_delays) > 0:
        _IMAGE_RETRY_DELAYS = tuple(int(x) for x in retry_delays)


def _throttle_image_call() -> None:
    """Đảm bảo cách lần gọi image gần nhất ít nhất _IMAGE_MIN_INTERVAL_S giây."""
    global _LAST_IMAGE_CALL_AT
    if _IMAGE_MIN_INTERVAL_S <= 0:
        return
    elapsed = time.monotonic() - _LAST_IMAGE_CALL_AT
    wait = _IMAGE_MIN_INTERVAL_S - elapsed
    if wait > 0:
        time.sleep(wait)


def _call_image_model_with_retry(fn, *, label: str):
    """
    Gọi `fn()` (thường là `client.models.generate_content(...)` cho model ảnh) với:
      • Throttle giãn cách tối thiểu giữa 2 lần gọi image (chống burst → 429).
      • Retry-backoff cho lỗi tạm thời (429/503/500/timeout).
    Nếu hết retry vẫn fail → re-raise last exception (giữ nguyên hành vi hiện tại
    của các hàm image, vốn raise lên trên để code gọi tự xử lý).
    """
    global _LAST_IMAGE_CALL_AT
    delays = _IMAGE_RETRY_DELAYS
    last_exc: Exception | None = None
    for attempt in range(len(delays) + 1):
        _throttle_image_call()
        try:
            resp = fn()
            _LAST_IMAGE_CALL_AT = time.monotonic()
            return resp
        except Exception as exc:
            _LAST_IMAGE_CALL_AT = time.monotonic()
            last_exc = exc
            if not _is_transient_api_error(exc):
                raise
            if attempt >= len(delays):
                break
            wait_s = delays[attempt]
            head = (str(exc).splitlines() or [""])[0][:160]
            print(
                f"  [WARN] {label}: lỗi tạm thời ({type(exc).__name__}: {head}). "
                f"Retry sau {wait_s}s (lần {attempt + 2}/{len(delays) + 1})...",
                file=sys.stderr,
            )
            time.sleep(wait_s)

    assert last_exc is not None
    raise last_exc


def _chunk_text_for_fix(text: str, max_chars: int = 6000) -> list[tuple[int, int, str]]:
    """
    Chia truyện thành các chunk <= max_chars, ưu tiên cắt tại biên đoạn (\\n\\n+) rồi tới biên câu.
    Trả list (start_offset, end_offset, chunk_text). Ghép lại bằng cách concat trực tiếp các chunk_text
    sẽ ra đúng `text` gốc (không mất ký tự nào).
    """
    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]

    chunks: list[tuple[int, int, str]] = []
    pos = 0

    para_re = re.compile(r"\n{2,}")
    sent_re = re.compile(r"(?<=[\.\!\?…\?！。])\s+|(?<=\n)")

    while pos < n:
        end_target = min(n, pos + max_chars)
        if end_target >= n:
            chunks.append((pos, n, text[pos:n]))
            break

        # Ưu tiên 1: tìm biên đoạn (\n\n+) gần end_target nhất, trong cửa sổ [pos+max_chars/2 .. end_target+max_chars/4]
        window_start = pos + max_chars // 2
        window_end = min(n, end_target + max_chars // 4)
        cut = -1
        for m in para_re.finditer(text, window_start, window_end):
            cut = m.end()
        if cut < 0:
            # Ưu tiên 2: cắt tại biên câu trong cửa sổ
            best_sent_cut = -1
            for m in sent_re.finditer(text, window_start, window_end):
                if m.end() <= window_end:
                    best_sent_cut = m.end()
            if best_sent_cut > 0:
                cut = best_sent_cut
        if cut < 0:
            # Ưu tiên 3: cắt tại whitespace gần nhất trước end_target
            ws = text.rfind(" ", pos + max_chars // 2, end_target)
            cut = ws if ws > pos else end_target

        cut = max(cut, pos + 1)
        chunks.append((pos, cut, text[pos:cut]))
        pos = cut

    return chunks


def _build_content_fix_prompt(chunk: str, *, total_chunks: int, chunk_index: int) -> str:
    progress_note = (
        f"\nNote: Đây là CHUNK {chunk_index + 1}/{total_chunks} của một truyện dài. "
        "CHỈ sửa phần được cung cấp; KHÔNG được nhắc tới các phần khác, KHÔNG thêm câu nối, "
        "KHÔNG thêm/bớt dòng đầu/cuối."
        if total_chunks > 1
        else ""
    )
    return f"""Task: Sửa CHÍNH TẢ tiếng Việt và PHIÊN ÂM các từ tiếng nước ngoài sang cách đọc THUẦN VIỆT cho TTS, KHÔNG ĐƯỢC đổi nội dung câu chuyện.{progress_note}

INPUT STORY (giữ nguyên cấu trúc, không bịa thêm ý, không tóm tắt, không lược bỏ):
<<<
{chunk}
>>>

NHIỆM VỤ — CHỈ làm hai việc dưới đây, không làm gì khác:

1. SỬA LỖI CHÍNH TẢ tiếng Việt: chỉ đổi từ viết SAI chính tả → đúng chính tả; chỉ đổi từ viết tắt teen-code → từ chuẩn.
   • Ví dụ sửa: "ko" → "không", "vs" → "với", "đc" → "được", "j" → "gì", "ng" → "người", "wa" → "qua",
     "ckung" → "chung", "lik" → "thích", "trc" → "trước", "sv" → "sinh viên" (chỉ khi rõ context), "bik" → "biết",
     "h" → "giờ" (khi rõ là "8h sáng"), "iu" → "yêu", "z" → "vậy", "đg" → "đang".
   • Sửa lỗi đánh máy / dấu rõ ràng: "tôi yêu việt nam" → "tôi yêu Việt Nam" CHỈ khi danh từ riêng đã viết hoa nhầm; sai dấu hỏi/ngã rõ ràng theo từ điển ("sửa" vs "sữa": chọn theo nghĩa câu).
   • SỬA LỖI DẤU THEO NGỮ CẢNH (CRITICAL — kể cả khi CẢ 2 dạng đều là từ tiếng Việt hợp lệ về từ điển):
     Quy tắc: nếu một dấu trên chữ làm thay đổi NGHĨA / TỪ LOẠI, và dạng còn lại KHÔNG khớp vai trò ngữ pháp
     trong câu, BẮT BUỘC sửa về dạng đúng theo ngữ cảnh. Phổ biến trong văn bản convert từ truyện Trung sang Việt:
     - "Hắn" (đại từ ngôi 3 nam: he/him) ↔ "Hẳn" (trạng từ: chắc chắn, đích thị).
         • Nếu chủ ngữ của câu là "Hẳn/Hắn" + động từ (Hẳn nói / cười / đi / nhìn / quỳ / bước / nâng…) HOẶC "Hẳn/Hắn" là tân ngữ (gặp hẳn, theo hắn, nhìn hắn) → BẮT BUỘC là "Hắn".
         • Chỉ giữ "Hẳn" khi nó là TRẠNG TỪ bổ nghĩa cho động từ/tính từ ngay sau ("hẳn là", "hẳn không biết", "hẳn rồi"). Test: thay được bằng "chắc chắn" mà câu vẫn nghĩa → đúng là "Hẳn".
         • Ví dụ: "Hẳn trước mặt tất cả mọi người xin thánh chỉ ban hôn." → "Hắn trước mặt tất cả mọi người xin thánh chỉ ban hôn." (chủ ngữ là người).
         • Ví dụ: "Hẳn không biết chuyện này." → giữ nguyên (= "chắc chắn không biết").
     - "Cố" (họ: Cố Thiên Từ / Cố phủ; hoặc động từ: cố gắng, cố ý) ↔ "Cô" (cô gái / đại từ ngôi 2-3 nữ trẻ). Phân biệt theo vai trò trong cụm danh từ.
     - "Hằn" (vết hằn, ấn tượng) ↔ "Hắn" (đại từ ngôi 3) — nếu đi sau "thằng/tên/gã" hoặc đứng chủ ngữ → "Hắn".
     - "Nàng" (đại từ ngôi 3 nữ trong văn cổ trang) ↔ "Nãng" (gần như chắc chắn là LỖI; sửa về "Nàng").
     - "Chàng" (đại từ ngôi 3 nam trong văn cổ trang) ↔ "Chãng" (LỖI; sửa về "Chàng").
     - "Lão" (cụ già / tiền tố tôn xưng) ↔ "Láo" (hỗn xược).
     - "Vương" (vua / họ) ↔ "Vướng" (mắc kẹt).
     - "Mặt" (face) ↔ "Mạt" (cuối, tận) — chọn theo nghĩa câu.
     - "Vẫn" (still) ↔ "Vận" (vận chuyển / số phận) — chọn theo nghĩa câu.
     - "Ngẩng" (ngẩng đầu) ↔ "Ngẳng" (LỖI; sửa về "Ngẩng").
     - "Thái tử" (correct) ↔ "Thái tữ" / "Thái tử̛" (LỖI; sửa về "Thái tử").
     - "Tỷ" (tỷ tỷ — chị) ↔ "Tỉ" (tỉ lệ) — chọn theo nghĩa.
     Heuristic chung: đọc CẢ CÂU; xác định VAI TRÒ NGỮ PHÁP của chữ nghi vấn (chủ ngữ / tân ngữ / trạng từ / tính từ / danh từ riêng); chọn dạng dấu khớp NGHĨA câu. Nếu cả 2 dạng đều có thể đứng được nhưng truyện đang dùng nhất quán một đại từ ngôi 3 ("hắn" / "nàng" / "y"…) ở các câu khác → bám theo dạng nhất quán đó.
   • TUYỆT ĐỐI KHÔNG sửa: phương ngữ vùng miền ("đi mô", "biết răng", "chi rứa", "có chi", "mần chi"…), tiếng lóng cố ý của truyện, văn phong cổ trang ("nàng", "chàng", "ngươi", "thiếp", "phu nhân"), từ Hán-Việt.
   • TUYỆT ĐỐI KHÔNG đổi từ đồng nghĩa: "thấy" KHÔNG đổi thành "trông thấy"; "đẹp" KHÔNG đổi thành "xinh đẹp"; "nói" KHÔNG đổi thành "thốt".
   • TUYỆT ĐỐI KHÔNG đổi xưng hô: tôi/tớ/mình/em/chị/anh/chú/bác/mày/tao/ngươi/thiếp/chàng… giữ nguyên 100%.

2. PHIÊN ÂM TỪ TIẾNG NƯỚC NGOÀI sang chuỗi chữ quốc ngữ ĐỌC GẦN GIỐNG bản gốc, để TTS Việt phát âm đúng.
   • Mục tiêu: giọng đọc Việt phát âm như người Việt nói (không cố đọc theo tiếng Anh).
   • Ví dụ:
     - "Facebook" → "Phây-bút"
     - "Google" → "Gu-gồ"
     - "Instagram" → "In-xờ-ta-gờ-ram"
     - "TikTok" → "Tích-tốc"
     - "iPhone" → "Ai-phôn"
     - "Tesla" → "Tét-la"
     - "Tom" → "Tôm"
     - "Lily" → "Li-li"
     - "Paris" → "Pa-ri"
     - "WhatsApp" → "Goát-sáp"
     - "Wi-Fi" → "Goai-phai"
     - "router" → "rao-tơ"
     - "deadline" → "đét-lai"
     - "online" → "on-lai" (nếu cần phiên âm để TTS đọc đúng)
   • Số / đơn vị / công thức → đọc bằng tiếng Việt: "USD" → "đô-la Mỹ", "km/h" → "ki-lô-mét trên giờ",
     "iOS 17" → "Ai-ô-ét mười bảy", "5G" → "năm-gờ", "AI" → "Ây-ai".
   • TÊN HÁN-VIỆT / địa danh Trung-Việt / Hán tự đã phiên âm sẵn ("Tống Kỳ An", "Dương Châu", "phủ Hầu",
     "Lưu Bị", "Trường An"…) → GIỮ NGUYÊN, không phiên âm thêm.
   • Tên Việt thuần ("Nguyễn Văn A", "Hà Nội", "Sài Gòn"…) → GIỮ NGUYÊN.
   • Nếu một từ tiếng Anh đã VIỆT-HOÁ phổ thông và TTS Việt đọc tốt ("game", "fan", "show", "video", "email",
     "internet"), được giữ nguyên. CHỈ phiên âm những từ TTS Việt sẽ đọc sai/nuốt âm.

QUY TẮC TUYỆT ĐỐI (NEGATIVE — vi phạm bất kỳ điểm nào dưới đây = SAI):
- KHÔNG thêm câu, KHÔNG bớt câu, KHÔNG đổi thứ tự câu, KHÔNG tóm tắt, KHÔNG diễn giải.
- KHÔNG đổi đại từ / xưng hô / giới tính nhân vật.
- KHÔNG dịch nghĩa từ tiếng nước ngoài — chỉ phiên âm CÁCH ĐỌC.
- KHÔNG thêm ghi chú / chú thích / dấu ngoặc giải thích / footnote.
- KHÔNG thêm tiêu đề, đề mục, đánh số chương, dòng phân cách nếu bản gốc không có.
- BẢO TOÀN tuyệt đối: dấu câu, khoảng trắng, dấu xuống dòng, dòng trống — vị trí và số lượng đúng như bản gốc.

OUTPUT FORMAT:
- Trả DUY NHẤT toàn văn (chunk) đã chỉnh, KHÔNG bao bọc trong JSON, KHÔNG markdown fence (```), KHÔNG thêm dòng giải thích nào.
- Bắt đầu thẳng bằng nội dung gốc của chunk, kết thúc đúng tại nội dung gốc của chunk (không thêm dòng đầu/cuối).
"""


def _strip_fence(text: str) -> str:
    out = (text or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:\w+)?\s*\n?", "", out)
        out = re.sub(r"\n?```\s*$", "", out)
    return out


def _normalize_story_content(
    client: genai.Client,
    *,
    story: str,
    out_dir: Path,
    planner_model: str,
    regenerate: bool = False,
    chunk_chars: int = 6000,
) -> str:
    """
    Sửa lỗi chính tả tiếng Việt + chuyển từ TIẾNG NƯỚC NGOÀI sang phiên âm THUẦN VIỆT (cách đọc),
    KHÔNG đổi nội dung / cấu trúc / xưng hô / dấu câu / xuống dòng. Lưu vào `out_dir / 'content_fix.txt'`
    cùng metadata `out_dir / 'content_fix.meta.json'`. Trả truyện đã chỉnh.

    Truyện dài > `chunk_chars` ký tự được chia thành nhiều chunk, fix song song theo trình tự,
    có retry transient + fallback từng chunk (dùng nguyên gốc nếu chunk đó fail).
    """
    fix_path = out_dir / "content_fix.txt"
    meta_path = out_dir / "content_fix.meta.json"
    src_fp = _story_fingerprint(story)

    if not regenerate and fix_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_fp") == src_fp:
                cached = fix_path.read_text(encoding="utf-8")
                print(
                    f"Dùng lại bản fix nội dung đã có: {fix_path.name} "
                    f"({len(cached)} ký tự, source_fp={src_fp[:8]}).",
                    file=sys.stderr,
                )
                return cached
            print(
                f"{fix_path.name} thuộc bản truyện khác (source_fp khác) → tạo lại.",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"Đọc {meta_path.name} lỗi ({exc}) → tạo lại {fix_path.name}.",
                file=sys.stderr,
            )

    chunks = _chunk_text_for_fix(story, max_chars=chunk_chars)
    print(
        f"Sửa chính tả + phiên âm thuần Việt cho truyện ({len(story)} ký tự, "
        f"{len(chunks)} chunk × ~{chunk_chars} ký tự) → {fix_path.name}...",
        file=sys.stderr,
    )

    fixed_parts: list[str] = []
    n_failed = 0
    for i, (s, e, chunk) in enumerate(chunks):
        prompt = _build_content_fix_prompt(
            chunk, total_chunks=len(chunks), chunk_index=i
        )
        label = f"content-fix chunk {i + 1}/{len(chunks)} ({e - s} ký tự)"
        if len(chunks) > 1:
            print(
                f"  · Đang fix {label}...",
                file=sys.stderr,
            )

        def _call() -> Any:
            return client.models.generate_content(
                model=planner_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )

        response = _generate_with_transient_retry(_call, label=label)
        if response is None:
            print(
                f"    [WARN] {label}: dùng nguyên gốc cho chunk này.",
                file=sys.stderr,
            )
            fixed_parts.append(chunk)
            n_failed += 1
            continue

        fixed_chunk = _strip_fence(response.text or "")
        if not fixed_chunk:
            print(
                f"    [WARN] {label}: model trả rỗng → dùng nguyên gốc cho chunk này.",
                file=sys.stderr,
            )
            fixed_parts.append(chunk)
            n_failed += 1
            continue

        if len(fixed_chunk) < 0.6 * len(chunk) or len(fixed_chunk) > 1.6 * len(chunk):
            print(
                f"    [WARN] {label}: độ dài lệch quá nhiều "
                f"({len(chunk)} → {len(fixed_chunk)}); model có thể đã tóm tắt/lặp → dùng nguyên gốc.",
                file=sys.stderr,
            )
            fixed_parts.append(chunk)
            n_failed += 1
            continue

        fixed_parts.append(fixed_chunk)

    fixed = "".join(fixed_parts)
    if not fixed.strip():
        print("[WARN] content fix tổng kết rỗng — dùng truyện gốc.", file=sys.stderr)
        return story

    if len(fixed) < 0.6 * len(story) or len(fixed) > 1.6 * len(story):
        print(
            f"[WARN] content fix tổng kết khác độ dài quá nhiều ({len(story)} → {len(fixed)} ký tự); "
            "dùng truyện gốc.",
            file=sys.stderr,
        )
        return story

    _atomic_write_text(fix_path, fixed)
    fixed_fp = _story_fingerprint(fixed)
    _atomic_write_json(
        meta_path,
        {
            "version": 1,
            "source_fp": src_fp,
            "fixed_fp": fixed_fp,
            "planner_model": planner_model,
            "source_chars": len(story),
            "fixed_chars": len(fixed),
            "chunks_total": len(chunks),
            "chunks_failed_fallback": n_failed,
            "chunk_chars_target": chunk_chars,
        },
    )
    suffix_warn = (
        f" (CẢNH BÁO: {n_failed}/{len(chunks)} chunk dùng nguyên gốc do server lỗi — chạy lại sau để hoàn thiện)"
        if n_failed
        else ""
    )
    print(
        f"  -> {fix_path} ({len(fixed)} ký tự; source_fp={src_fp[:8]} → fixed_fp={fixed_fp[:8]}"
        f"; {len(chunks)} chunk){suffix_warn}.",
        file=sys.stderr,
    )
    return fixed


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_outline_section_meta(row: dict[str, Any]) -> tuple[int | None, str]:
    """Đọc outlineSectionNumber / outlineSectionSummary từ manifest (nếu có)."""
    raw = row.get("outlineSectionNumber")
    num: int | None = None
    if raw is not None and str(raw).strip() != "":
        try:
            num = int(raw)
        except (TypeError, ValueError):
            num = None
    summ = str(
        row.get("outlineSectionSummary")
        or row.get("outlineSectionTitle")
        or ""
    ).strip()
    return num, summ


def _parse_narrative_plane(row: dict[str, Any]) -> str:
    v = row.get("narrativePlane") or row.get("narrative_plane") or "present"
    s = str(v).strip().lower()
    if s in ("flashback", "memory", "recollection", "recall", "dream", "imagined"):
        return "flashback"
    return "present"


def _manifest_dict_to_scene(d: dict[str, Any]) -> Scene:
    try:
        pc = int(d.get("panelCount", 1))
    except (TypeError, ValueError):
        pc = 1
    panels_raw = d.get("panels")
    panels: list[dict[str, Any]] = list(panels_raw) if isinstance(panels_raw, list) else []
    osn, oss = _parse_outline_section_meta(d)
    return Scene(
        page_number=int(d.get("pageNumber", 0)),
        story_segment=str(d.get("storySegment", "")),
        start_anchor=str(d.get("startAnchor", "")),
        end_anchor=str(d.get("endAnchor", "")),
        page_content=str(d.get("pageContent", "")),
        story_text=str(d.get("storyText", "") or ""),
        panel_count=pc,
        suggested_references=list(d.get("suggestedReferences") or []),
        panels=panels,
        narrative_plane=_parse_narrative_plane(d),
        outline_section_number=osn,
        outline_section_summary=oss,
        motion_prompt=str(d.get("motionPrompt", "") or ""),
    )


def scene_to_manifest_dict(s: Scene, mode: str) -> dict[str, Any]:
    """Một dòng plan giống scenes.json / checkpoint (dùng chung cho mọi mode)."""
    item: dict[str, Any] = {
        "mode": mode,
        "pageNumber": s.page_number,
        "storySegment": s.story_segment,
        "startAnchor": s.start_anchor,
        "endAnchor": s.end_anchor,
        "storyText": s.story_text,
        "pageContent": s.page_content,
        "panelCount": s.panel_count,
        "suggestedReferences": s.suggested_references,
    }
    if mode == "manga":
        item["mangaPageStyle"] = (
            "rectangular_grid_left_to_right_top_to_bottom; no_on_image_text"
        )
        item["panels"] = s.panels
    if mode == "review":
        item["narrativePlane"] = s.narrative_plane or "present"
        if s.outline_section_number is not None:
            item["outlineSectionNumber"] = s.outline_section_number
        if (s.outline_section_summary or "").strip():
            item["outlineSectionSummary"] = s.outline_section_summary
        if (s.motion_prompt or "").strip():
            item["motionPrompt"] = s.motion_prompt
    return item


def _row_section_number(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("sectionNumber", fallback))
    except (TypeError, ValueError):
        return fallback


def _split_review_two_pass(
    client: genai.Client,
    story: str,
    *,
    char_paths: dict[str, Path],
    style_paths: list[Path],
    scene_count: int | None,
    planner_model: str,
    checkpoint_path: Path,
    plan_manifest_path: Path | None = None,
    bible_inline: bool = False,
    bible_path: Path | None = None,
    bible_regenerate: bool = False,
) -> list[Scene]:
    """
    Review: bước 1 = dàn ý (ít object, không pageContent dài); bước 2 = beat chi tiết từng đoạn.
    Ghi checkpoint sau outline và sau mỗi đoạn (để resume khi API lỗi).
    """
    story_fp = _story_fingerprint(story)
    asset_list = _asset_list_for_prompt(char_paths, style_paths)
    outline_clause = (
        f"EXACTLY {scene_count} contiguous outline sections"
        if scene_count
        else (
            "a moderate number of contiguous outline sections "
            "(roughly 8–25 depending on length—fewer, larger sections for very long stories)"
        )
    )
    exact_outline = (
        f"IMPORTANT: You MUST output EXACTLY {scene_count} outline sections covering the full story in order.\n"
        if scene_count
        else ""
    )
    outline_prompt = f"""
Task: Create a STORY OUTLINE for later illustration-beat planning (review / storyboard mode).
You are NOT creating final illustration beats yet—only contiguous prose sections.

Break the following story into {outline_clause} that cover the ENTIRE story from beginning to end.

Story (full text):
{story}

Available Assets in Library: {json.dumps(asset_list, ensure_ascii=False)}

{exact_outline}
Rules:
- Sections MUST be in chronological reading order, contiguous, no gaps, no overlaps.
- Genre-agnostic: outline boundaries depend ONLY on explicit prose cues (time jump, location change, POV/memory shift),
  not on story type (romance, thriller, sci-fi, children's, etc.).
- Each section must be a single contiguous excerpt from the story (verbatim span).
- Keep this output COMPACT: do NOT write pageContent / image prompts in this step.

SECTION BOUNDARY RULES (CRITICAL — DO NOT BREAK A CONTINUOUS SCENE):
- A "section" should correspond to ONE continuous on-screen scene = same location + same time-of-day window + same continuous chain of action / dialogue. If the story spends many paragraphs on one continuous scene, KEEP it as ONE section (do not split it just because it is long).
- You MAY start a new section ONLY when the prose shows at least one of these clear breaks in the PRESENT timeline:
  (a) explicit time jump in the present timeline (e.g. "the next morning", "three days later", "weeks passed", or any equivalent in the source language),
  (b) explicit location change in the present timeline (characters physically move to a different place and the prose stays there),
  (c) point-of-view / narrator shift in the PRESENT timeline (a brief recollection / dream is NOT a POV shift — see below).
  (d) major narrative beat change clearly separated by a paragraph break + present-timeline time/space gap (NOT just a new dialogue turn, NOT a quick memory).
- DO NOT split between adjacent sentences that share the same room, same conversation, same minute. Example BAD split: ending one section at "she handed him the divorce papers." and starting the next at "he flipped to the last page and signed it." — that is one continuous action and MUST stay in one section.
- FLASHBACK / RECOLLECTION INSIDE A SCENE IS NOT A SECTION BREAK (CRITICAL):
  • If a continuous present scene is interrupted by a brief flashback / recollection / dream / inner-monologue memory and then RETURNS to the same room / same conversation / same minute, the WHOLE thing (present-before + flashback + present-after) is ONE section.
  • The pass-2 planner will tag the flashback prose with narrativePlane="flashback" and the surrounding prose with narrativePlane="present" — both belong to the SAME outline section here.
  • Only treat the flashback as a SEPARATE section if the prose never returns to the original present scene (i.e. after the memory the story moves to a different place / time, never resuming the prior conversation).
- If wardrobe / setting must remain identical across the next prose, that prose belongs to the SAME section.
- Prefer fewer, larger sections over many small ones whenever the prose stays in one continuous scene.

For each section output:
1. sectionNumber: integer 1..N in order
2. storySegment: one short line summarizing the section
3. startAnchor: EXACT first 5–10 words as they appear in the story
4. endAnchor: EXACT last 5–10 words as they appear in the story
5. storyText: VERBATIM excerpt from the story from startAnchor through endAnchor inclusive

Return ONLY a JSON array of objects with keys:
"sectionNumber", "storySegment", "startAnchor", "endAnchor", "storyText".
"""

    def _section_sort_key(r: dict[str, Any]) -> int:
        try:
            return int(r.get("sectionNumber", 0))
        except (TypeError, ValueError):
            return 0

    def _persist(
        outline: list[dict[str, Any]],
        completed: set[int],
        scenes_acc: list[Scene],
    ) -> None:
        _atomic_write_json(
            checkpoint_path,
            {
                "version": 1,
                "story_fp": story_fp,
                "planner_model": planner_model,
                "scene_count": scene_count,
                "outline": outline,
                "completed_section_numbers": sorted(completed),
                "scenes": [scene_to_manifest_dict(s, "review") for s in scenes_acc],
            },
        )
        outline_sidecar = checkpoint_path.parent / "review_outline.json"
        _atomic_write_json(
            outline_sidecar,
            {
                "version": 1,
                "story_fp": story_fp,
                "planner_model": planner_model,
                "scene_count": scene_count,
                "completed_section_numbers": sorted(completed),
                "outline": outline,
            },
        )
        if plan_manifest_path is not None:
            _atomic_write_json(
                plan_manifest_path,
                [scene_to_manifest_dict(s, "review") for s in scenes_acc],
            )

    ck: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        try:
            ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ck = None
        if ck and ck.get("story_fp") != story_fp:
            raise SystemExit(
                f"Checkpoint {checkpoint_path} thuộc bản truyện khác. "
                "Dùng --fresh-checkpoint để xóa và chạy lại, hoặc trỏ -o đúng thư mục cũ."
            )
        if ck and ck.get("planner_model") != planner_model:
            print(
                f"Cảnh báo: planner_model trong checkpoint ({ck.get('planner_model')!r}) "
                f"khác lệnh hiện tại ({planner_model!r}).",
                file=sys.stderr,
            )

    # Bible inline state (cập nhật sau mỗi section beats xong, persist ngay).
    bible_state: dict[int, dict[str, Any]] = {}
    if bible_inline and bible_path is not None:
        bible_state = _load_section_designs_state(
            bible_path,
            story_fp,
            regenerate=bible_regenerate,
        )
        if bible_state:
            print(
                f"Bible inline: load {len(bible_state)} section đã có từ {bible_path.name}.",
                file=sys.stderr,
            )

    outline_sorted: list[dict[str, Any]]
    completed_nums: set[int]
    all_scenes: list[Scene]

    if ck and isinstance(ck.get("outline"), list) and ck["outline"]:
        outline_sorted = [r for r in ck["outline"] if isinstance(r, dict)]
        outline_sorted.sort(key=_section_sort_key)
        raw_done = ck.get("completed_section_numbers") or []
        completed_nums = {int(x) for x in raw_done}
        scenes_raw = ck.get("scenes") or []
        all_scenes = [_manifest_dict_to_scene(d) for d in scenes_raw if isinstance(d, dict)]
        n_fix = _dedupe_review_story_text_overlaps(all_scenes)
        if n_fix:
            print(
                f"Đã chuẩn hoá storyText khi load checkpoint ({n_fix} cặp overlap).",
                file=sys.stderr,
            )
            _persist(outline_sorted, completed_nums, all_scenes)
        _warn_long_story_text_beats(all_scenes)
        _propagate_copresent_cast_in_scenes(all_scenes)
        print(
            f"Tiếp tục từ checkpoint: {len(completed_nums)}/{len(outline_sorted)} đoạn dàn ý, "
            f"{len(all_scenes)} beat.",
            file=sys.stderr,
        )
    else:
        def _outline_call() -> Any:
            return client.models.generate_content(
                model=planner_model,
                contents=outline_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.35,
                ),
            )

        o_resp = _generate_with_transient_retry(
            _outline_call, label=f"review outline (pass 1, {planner_model})"
        )
        if o_resp is None:
            raise RuntimeError(
                "Outline review (pass 1) gọi LLM thất bại sau nhiều retry. "
                "Thử đổi --planner-model gemini-2.5-pro hoặc chạy lại sau."
            )
        outline_rows = _parse_planner_json_array(o_resp.text or "", client, planner_model)
        outline_sorted = sorted(
            [r for r in outline_rows if isinstance(r, dict)],
            key=_section_sort_key,
        )
        if not outline_sorted:
            raise ValueError("Outline review (pass 1) trả về rỗng.")
        completed_nums = set()
        all_scenes = []
        _persist(outline_sorted, completed_nums, all_scenes)
        print(
            f"Đã lưu dàn ý ({len(outline_sorted)} đoạn) → {checkpoint_path}",
            file=sys.stderr,
        )

    global_idx = len(all_scenes)

    # Bible inline: nếu bible đã load có alias (vd. "Dr. Ôn") khác với tên thật trong char_paths
    # thì sửa ngay để khớp với required_labels — tránh phải --regenerate-section-designs.
    if bible_inline and bible_path is not None and bible_state and all_scenes:
        sec_to_scenes_for_norm: dict[int, list[Scene]] = {}
        for s in all_scenes:
            if s.outline_section_number is not None:
                sec_to_scenes_for_norm.setdefault(int(s.outline_section_number), []).append(s)
        sec_to_required = {
            sn: _required_character_labels_for_section(sec_to_scenes_for_norm[sn], char_paths)
            for sn in bible_state.keys()
            if sn in sec_to_scenes_for_norm
        }
        if sec_to_required:
            _normalize_existing_bible_state_inplace(
                bible_path,
                existing=bible_state,
                sec_to_required=sec_to_required,
                char_paths=char_paths,
                story_fp=story_fp,
                planner_model=planner_model,
            )
            _repair_existing_bible_inplace(
                bible_path,
                client=client,
                planner_model=planner_model,
                existing=bible_state,
                sec_to_required=sec_to_required,
                char_paths=char_paths,
                story_fp=story_fp,
            )

    for sec_i, row in enumerate(outline_sorted, start=1):
        sec_num = _row_section_number(row, sec_i)
        if sec_num in completed_nums:
            continue

        start_a = str(row.get("startAnchor", ""))
        end_a = str(row.get("endAnchor", ""))
        raw_chunk = str(row.get("storyText") or "").strip()
        chunk = raw_chunk if raw_chunk else _verbatim_between_anchors(story, start_a, end_a)
        chunk = (chunk or "").strip()
        if not chunk:
            print(
                f"Cảnh báo outline section {sec_i}: không có đoạn văn; đánh dấu xong để không kẹt resume.",
                file=sys.stderr,
            )
            completed_nums.add(sec_num)
            _persist(outline_sorted, completed_nums, all_scenes)
            continue

        sec_summary_line = str(row.get("storySegment", "")).strip()

        prev_tail_block = ""
        if all_scenes:
            prev_present = next(
                (s for s in reversed(all_scenes) if (s.narrative_plane or "present") == "present"),
                None,
            )
            if prev_present is not None and prev_present.page_content:
                prev_tail_block = (
                    "\nPREVIOUS-SECTION TAIL (for continuity check, DO NOT repeat as a beat in this section):\n"
                    f"- previousPresentBeatPageContent: {prev_present.page_content[:1200]}\n"
                    "- If the FIRST beat of THIS section continues the SAME continuous scene as that previous beat "
                    "(same location, same time-of-day window, same conversation/action chain — check the SECTION EXCERPT "
                    "below against the previous beat's wording), you MUST copy verbatim the location, lighting, time-of-day, "
                    "and each character's outfit/hair/grooming wording from the previous beat into this section's present beats. "
                    "Treat them as one continuous scene that just happened to be cut into two outline sections.\n"
                    "- Otherwise (real time/place jump or flashback), describe the new setting/wardrobe naturally per the prose."
                )

        beats_prompt = f"""
Task: Act as a Story Review / Visual Reading Artist (NOT manga multi-panel pages).
Expand ONLY the following STORY SECTION into illustration beats. Each item is ONE standalone illustration.

SCOPE — GENRE & SOURCE LANGUAGE (apply to every story):
- Rules below apply to ANY genre: literary fiction, romance, thriller, mystery, sci-fi, fantasy, horror,
  historical, xianxia/wuxia, slice-of-life, children's, comedy, drama, nonfiction adaptation, etc.
- Anchors (startAnchor, endAnchor) and storyText MUST be VERBATIM from the user's source story —
  do NOT translate or paraphrase, regardless of source language (Vietnamese, English, Chinese, Japanese,
  Korean, French, Spanish, mixed scripts, etc.).
- Worked examples below illustrate STRUCTURE (how to split beats), not tone/era/plot — never copy their wording.

OUTLINE SECTION (this entire response is ONE section only):
- sectionIndex: {sec_i} of {len(outline_sorted)} (reading order).
- sectionLabel: {sec_summary_line!r}
- PRESENT-THREAD CONTINUITY (DEFINITION — applies to every "consecutive beats" rule below):
  • The "present thread" is the chain of beats with narrativePlane="present" inside this section, READ IN ORDER and IGNORING any flashback beats inserted between them. So if the section runs `present₁ → flashback → present₂ → present₃ → flashback → present₄`, the present thread is `present₁ → present₂ → present₃ → present₄`.
  • Whenever a rule says "consecutive beats", "consecutive present beats", "the previous beat", or "the next beat" for continuity (setting / wardrobe / camera variation / co-present cast), it means consecutive WITHIN THIS PRESENT THREAD — flashback beats are skipped for that comparison. Flashback beats only need to follow flashback rules (memory styling) and do not reset the present baseline.
  • DO NOT BREAK A CONTINUOUS PRESENT SCENE INTO TWO INDEPENDENT FRAGMENTS just because a flashback is inserted in the middle. After the flashback ends, the next present beat resumes the SAME continuous scene as the present beat before the flashback — same room, same conversation, same minute, same outfits.
- LOCKED REALITY RULE: For ALL beats with narrativePlane="present" in THIS section, you MUST use ONE shared, copy-pasted baseline for (1) exact location/setting, (2) time of day & lighting, (3) each visible character's outfit/hair/grooming — identical wording repeated in every present beat's pageContent. Do NOT drift location or costume across the present thread (flashbacks inserted between them do NOT reset this baseline). ONLY narrativePlane="flashback" beats may show a different place/era/outfit (memory styling). After each flashback, the next "present" beat MUST return to the SAME locked baseline as the present beat just before the flashback, unless the SECTION EXCERPT text explicitly moves to a new place or signals an outfit change.
- PER-CHARACTER WARDROBE LOCK: For EVERY named character who appears visually in a "present" beat, spell out their FULL outfit + hairstyle + footwear/accessories IN WORDS inside pageContent (repeat the SAME wording on each present beat where they appear — copy-paste). Never rely on vague phrases like "nice outfit"; always concrete colors, garment types, fabric cues.
- CROSS-SECTION CONTINUITY: see PREVIOUS-SECTION TAIL block below — if the first beat here continues the previous section's scene, copy its location/wardrobe wording verbatim instead of inventing new details.

SECTION SUMMARY (for your orientation): {sec_summary_line}
{prev_tail_block}
SECTION EXCERPT (verbatim — anchors MUST be exact substrings of this excerpt AND of the full story):
{chunk}

Available Assets in Library: {json.dumps(asset_list, ensure_ascii=False)}

HARD LIMIT: Output at most 120 beats for this section. If you need fewer to stay precise, that is OK.

DURATION RULE — EVERY BEAT MUST FIT 3–5 SECONDS OF NARRATION; NEVER EXCEED ~8 SECONDS (CRITICAL):
- The final product is a video where TTS reads each beat's storyText aloud. Each beat = ONE still image shown for as long as the narration of that storyText takes.
- Target spoken duration per beat: **3–5 seconds**. Hard ceiling: **~8 seconds**.
- Token budget per language (assume normal audiobook pace; the model should self-estimate before emitting a beat):
  • Vietnamese (~3 syllables/sec): 3s ≈ 8–10 syllables, 5s ≈ 12–15 syllables, 8s ≈ 20–24 syllables (ABSOLUTE CEILING).
  • English (~2.5 words/sec): 3s ≈ 7–8 words, 5s ≈ 12–13 words, 8s ≈ ~20 words (ABSOLUTE CEILING).
  • Chinese / Japanese (~4 chars/sec): 3s ≈ 12 chars, 5s ≈ 20 chars, 8s ≈ ~32 chars.
  • Other languages: estimate against your model's own fluent reading pace, keeping the same time targets.
- HARD CHECK before emitting a beat: estimate the spoken length of storyText. If it would exceed ~8s, you MUST split. If 6–8s, prefer to split further when natural boundaries exist.
- Splitting a long sentence:
  • Split at commas / semicolons / colons / em-dashes / coordinating conjunctions ("and", "but", "then", "so", "yet", "or"; or their equivalents in the source language).
  • Split between two independent clauses.
  • Split between a physical action and an inner thought.
  • A long sentence with 2–3 clauses → 2–3 separate beats.
- SHORT beats are encouraged (one clause, ~3–10 syllables/words). A beat longer than one sentence is a SIGN to split further.
- Direct dialogue: each short utterance / each speaker turn / each pause-marked phrase → one beat. A long line of dialogue must be split at commas or natural breath points.

{_VISUAL_RHYTHM_SPLIT_RULES}

GRANULARITY RULE — ONE BEAT = ONE FRAME = ONE MOMENT (CRITICAL):
- A beat is ONE freeze-frame illustration. It can show only ONE point in time, ONE place, ONE state of action.
- You MUST NOT cram multiple events / time points / locations into a single beat. If the prose moves through several moments, CREATE one beat per moment.
- Trigger a NEW beat whenever ANY of these appear in the prose (not exhaustive):
  • A time-jump cue in any language (e.g. "the next morning", "years later", "that afternoon", "before that", "later"; or equivalent markers in the source language).
  • A location change (e.g. "she stepped outside", "they boarded the shuttle", "entered the ward", or any equivalent).
  • A new action by ANY character (new gesture, prop pickup, turn, attack, embrace, sit/stand, etc.).
  • An EXPLICIT FLASHBACK / RECOLLECTION (not just past tense) — only when the prose clearly steps OUT of the current moment: e.g. "two years ago", "in my previous life", "back when she was a child", "I remembered that…", "in a dream", "earlier that month", "before the war", or a verbatim recalled memory inside a present scene. This kind of beat MUST be tagged narrativePlane="flashback".
  • A summarised journey / process (departure timing, long road, travel montage) → its own narrativePlane="flashback" "montage" beat, not glued to the arrival.
  • A new dialogue turn / new internal-monologue thought.
- HARD CHECK before emitting a beat: re-read its storyText and ask "Could a single still illustration realistically depict ALL of this in ONE frozen moment?" If no (it spans multiple times / places / events), SPLIT IT.
- Prefer SHORT, focused beats. A typical present-action beat is ONE short clause (~3–5 seconds). An exposition / flashback beat may be 1 clause or even a single phrase.
- DO NOT skip exposition prose by attaching it to the next present beat — render it as one or more flashback beats so the audience hears the narration over a memory image.

WORKED EXAMPLES — for STRUCTURE only (these are NEUTRAL FICTIONAL examples in English; the user's story may be in any language and your anchors / storyText must come VERBATIM from the user's source):

Example A — a long expository paragraph MUST split into MANY beats, not one:
Source paragraph (fictional neutral English): "My younger brother was the first person in our village to be admitted to university. Our family used to sell rice at the local market. Last year the school sent a letter, so Mother gathered money and we travelled to town. The dirt road ran along the river, and we cycled for three hours. We arrived early; the boarding room was not yet ready. Mother went to help in the kitchen while I sat on the porch and re-read the admission letter."

CORRECT split → ~10 beats (flashback for past/exposition; present for the now-moments):
  • flashback: brother's reputation / the family's old rice trade.
  • flashback: school letter + mother gathering money.
  • flashback: the cycling journey along the river (one or two beats depending on duration).
  • present: arriving early at the boarding room.
  • present: mother going into the kitchen.
  • present: narrator sitting on the porch reading the letter.

Example B — a present sentence with two clauses MUST become 2 beats:
Source (neutral English): "I rubbed my chin. Yes, the ages match."
CORRECT → 2 beats:
  • Beat (present) — "I rubbed my chin." (~4 words) → close-up of hand on chin.
  • Beat (present) — "Yes, the ages match." (~5 words) → close-up of satisfied expression.

Example C — a long action / dialogue line MUST split at natural breath points:
Source (neutral English): "The boy startled, raised his head, hurried to his feet, and greeted me politely."
CORRECT → 2–3 beats: "The boy startled, raised his head," / "hurried to his feet," / "and greeted me politely."

A WRONG output that puts an entire paragraph (or even a 2-clause sentence) into ONE beat is FORBIDDEN.

Split at natural visual boundaries (arrivals, gestures/props, dialogue turns, camera distance changes, flashbacks vs present).

NARRATIVE PLANE — every beat MUST set narrativePlane to exactly "present" or "flashback":
- "present" = the MAIN narrative timeline of the story — the events that the story is currently dramatising "now", regardless of grammatical tense. First-person retrospective narration (the narrator looking back in past tense) is STILL "present" as long as the beat dramatises the main story timeline. DEFAULT to "present" when in doubt.
- "flashback" = the prose explicitly steps OUT of the main timeline into a recollection, dream, memory, or earlier-era exposition. Trigger ONLY when there is a clear textual cue, e.g.: "X năm trước…", "kiếp trước…", "ngày xưa…", "hồi nhỏ…", "ta nhớ lại…", "trong giấc mơ…", "two years ago…", "in my previous life…", "earlier that month…", "back when…", "I remembered…", or a verbatim recalled memory inside an ongoing present scene. Such beats use a clearly different visual register (memory tint, softer edges, vignette, grain, faded palette). After a flashback, the next "present" beat MUST restore the previous reality baseline.
- DO NOT tag a beat as "flashback" just because the verb is past tense, the narration is first-person retrospective, or the prose mentions a fact that happened before the scene started. If the dramatised action is happening in the main story moment, it is "present".
- Consecutive "present" beats in this section MUST share the same setting (location, time of day, lighting) and the same per-character wardrobe until the prose explicitly moves to a new place or changes outfits.

For each beat:
1. storySegment: Short label for this beat.
2. startAnchor: EXACT first 5–10 words from the SECTION EXCERPT / story (must match exactly, in the source language).
3. endAnchor: EXACT last 5–10 words from the SECTION EXCERPT / story (must match exactly, in the source language).
4. pageContent: Detailed single-image prompt in ENGLISH (the visual brief is for the image model — write it in English even when the source story is in another language; do NOT include rendered text on the image). Start with "NARRATIVE: present reality —" or "NARRATIVE: flashback/memory —". Wardrobe is taken from context; references are identity-only. For present beats, reuse the SAME setting + wardrobe wording across consecutive present beats in this section unless the prose changes them.
   CAMERA / SHOT (REQUIRED per beat): include an explicit cinematic shot description tailored to THIS beat's emotion/action. Pick from: extreme close-up | close-up | medium close-up | medium | medium wide | wide | extreme wide; combined with one of: eye-level | low-angle | high-angle | over-the-shoulder | POV | top-down | slight Dutch tilt; plus staging (centered / left-third / right-third / through-doorway / foreground silhouette) and depth-of-field cue (shallow vs deep). VARY shot/angle/staging across consecutive beats IN THE SAME PRESENT THREAD (skip flashback beats when comparing — see PRESENT-THREAD CONTINUITY) — do NOT repeat the same framing for several present beats in a row; intimate dialogue → close-up/OTS, action → medium-wide/wide, isolation → wide, introspection → expressive close-up, discovery → POV/OTS to object. Flashback beats use their own memory framing and do not need to follow this present-thread variation rule.
   CO-PRESENT CAST IN pageContent (STRICT — MUST APPLY for every dialogue/group beat):
   • If the beat takes place inside a conversation / group scene with 2+ characters in the same location, pageContent MUST acknowledge ALL of them, even when the shot is a close-up of one.
   • For close-ups / OTS / extreme close-ups: keep the focused character as the main subject, AND describe the other co-present characters as part of the staging — e.g. "in the background, [X] stands at left, [Y] sits at the head of the table"; or "over [X]'s shoulder, [Y] is partially visible at frame-right"; or, if you truly need a tight isolated cut, append a final clause: "OFF-FRAME COMPANIONS (present in scene but outside frame): [X], [Y]".
   • NEVER describe an empty interior / empty hall / empty courtyard for a beat whose conversation has 2+ co-present people. The other characters' presence must be made readable to the illustrator (background, silhouette, edge of frame, OTS, or explicit off-frame note).
   • When the prose names a new speaker mid-conversation (e.g. a byline like "<Name> said slowly:" or any dialogue attribution), the cut may push the new speaker to foreground but the original co-present cast stays visible (background) or is named in OFF-FRAME COMPANIONS — they did not vanish.
   • Carry over wardrobe + setting wording for these co-present characters across consecutive beats; do not rebuild the setting from zero each beat.
5. storyText: Verbatim excerpt spanning startAnchor..endAnchor (in the source language).
   **NO OVERLAP between consecutive beats** in this JSON array: after beat N, beat N+1 must start at the NEXT unread word in the SECTION EXCERPT. If you split a sentence, end beat N before the first word of beat N+1 (e.g. end with a comma "..., " and start the next beat with the following clause); NEVER repeat the same clause/sentence in two beats.
   **Especially after narrativePlane changes** (flashback then present): do NOT copy the tail of the flashback beat's storyText into the next beat.
6. panelCount: ALWAYS 1.
7. suggestedReferences — PHYSICAL CO-PRESENCE RULE (STRICT — read carefully):
   This is a list of filenames from Available Assets for characters who are PHYSICALLY PRESENT in the beat's space — NOT for everyone who is merely mentioned.
   In pageContent, also write at the very end a single-line audit block:
     "WHO IS WHERE: SPEAKER → <name or 'narrator'>; LISTENERS PRESENT → <name>, <name>; OFF-FRAME but in scene → <name> (only if needed)."
   MUST INCLUDE (add the filename to suggestedReferences):
     • The SPEAKER of this beat (if they have a ref).
     • Anyone LISTENING / standing nearby / sitting in the same enclosed or open space (room, hallway, hall, courtyard, plaza, cabin, vehicle, hospital ward, cave, spaceship deck, etc.) — even silent and not the focal subject.
     • Anyone the prose explicitly describes as physically present (generic patterns: "<relative> stands behind me", "<authority figure> sits at the head of the table", "<rival> leans on the doorframe" — substitute the actual names from THIS story; do not copy these placeholders).
     • In a continuous conversation / interaction at THE SAME location: the SAME co-present cast must appear in refs for ALL beats in that thread — wide → medium → OTS → close-up → reaction shot → extreme close-up — until the prose explicitly says someone leaves or the scene moves to a different time/place.
   MUST EXCLUDE (do NOT add to suggestedReferences):
     • Characters merely MENTIONED in dialogue / narration but NOT physically present.
       Generic example: A says "you only want X to notice you" but X is not in the scene → DO NOT add X.
       Generic example: a present beat where the narrator briefly remembers character Y from the past but Y is not standing there → DO NOT add Y; convey the recollection as an expression cue on someone who IS present.
     • Characters who left in a previous scene/location.
     • Characters that exist only inside the inner-monologue of a present beat.
     • Unnamed crowds / extras without a portrait ref.
   If a present beat contains a brief recollection of someone who is not there → keep refs for the physically present cast only; render the recollection inside pageContent as "the protagonist's expression briefly recalls <memory subject>" WITHOUT drawing the absent person.
   Order: section-protagonist(s) first, then in narrative order of appearance.
   Pattern (replace placeholders with the right Available Assets filenames + character names from THIS story):
     Two characters A and B converse inside the same room; the wide shot and every close-up / ECU within that thread keep refs for BOTH because both are physically present.
     A subsequent beat whose dialogue mentions a third character C who is NOT in the room → refs are [A, B] only; do NOT add C.
8. narrativePlane: "present" or "flashback" (required).

{_WAN_MOTION_PROMPT_RULES}

Return ONLY a JSON array with keys (EVERY object MUST contain ALL 10 keys, including a non-empty `motionPrompt` string — see rule 9 above):
"pageNumber", "storySegment", "startAnchor", "endAnchor", "storyText", "pageContent", "panelCount", "suggestedReferences", "narrativePlane", "motionPrompt".
Number pageNumber starting at 1 within this section (it will be renumbered).
"""

        def _beats_call() -> Any:
            return client.models.generate_content(
                model=planner_model,
                contents=beats_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                ),
            )

        b_resp = _generate_with_transient_retry(
            _beats_call,
            label=f"review beats section {sec_i}/{len(outline_sorted)} ({planner_model})",
        )
        if b_resp is None:
            raise RuntimeError(
                f"Review beats section {sec_i}/{len(outline_sorted)} gọi LLM thất bại sau nhiều retry. "
                "Chạy lại lệnh với cùng --output để resume; hoặc đổi --planner-model gemini-2.5-pro."
            )
        beats_data = _parse_planner_json_array(b_resp.text or "", client, planner_model)
        chunk_scenes = _manifest_rows_to_scenes(beats_data, story, "review")
        outline_summary = sec_summary_line
        for s in chunk_scenes:
            global_idx += 1
            all_scenes.append(
                Scene(
                    page_number=global_idx,
                    story_segment=s.story_segment,
                    start_anchor=s.start_anchor,
                    end_anchor=s.end_anchor,
                    page_content=s.page_content,
                    story_text=s.story_text,
                    panel_count=s.panel_count,
                    suggested_references=s.suggested_references,
                    panels=s.panels,
                    narrative_plane=s.narrative_plane,
                    outline_section_number=sec_num,
                    outline_section_summary=outline_summary,
                )
            )
        completed_nums.add(sec_num)
        _persist(outline_sorted, completed_nums, all_scenes)
        print(
            f"Review 2-pass: section {sec_i}/{len(outline_sorted)} → {len(chunk_scenes)} beat "
            f"(tổng {global_idx}); đã lưu checkpoint.",
            file=sys.stderr,
        )

        # Bible inline: ngay sau khi section beats xong + persist, plan luôn bible cho section này.
        if bible_inline and bible_path is not None and sec_num not in bible_state:
            this_section_scenes = [
                s for s in all_scenes if s.outline_section_number == sec_num
            ]
            if this_section_scenes:
                obj, required_labels = _plan_one_section_wardrobe_bible(
                    client=client,
                    planner_model=planner_model,
                    sec_num=sec_num,
                    sec_scenes=this_section_scenes,
                    section_summary=outline_summary,
                    char_paths=char_paths,
                    prev_design=bible_state.get(sec_num - 1),
                )
                if obj is not None:
                    bible_state[sec_num] = obj
                    _persist_section_designs(
                        bible_path,
                        story_fp=story_fp,
                        planner_model=planner_model,
                        existing=bible_state,
                    )
                    n_chars = (
                        len((obj.get("characters") or {}).keys())
                        if isinstance(obj.get("characters"), dict)
                        else 0
                    )
                    if _wardrobe_bible_missing_details(obj, required_labels):
                        print(
                            f"  · Bible inline section {sec_num:02d}: lưu ({n_chars} character) — kiểm tra outfit/hair.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  · Bible inline section {sec_num:02d}: lưu ({n_chars} character, đủ outfit+hair).",
                            file=sys.stderr,
                        )

    n_fix = _dedupe_review_story_text_overlaps(all_scenes)
    if n_fix:
        print(
            f"Đã chuẩn hoá storyText toàn timeline ({n_fix} cặp overlap).",
            file=sys.stderr,
        )
        _persist(outline_sorted, completed_nums, all_scenes)
    _warn_long_story_text_beats(all_scenes)
    if _propagate_copresent_cast_in_scenes(all_scenes):
        _persist(outline_sorted, completed_nums, all_scenes)

    return all_scenes


def split_story_into_scenes(
    client: genai.Client,
    story: str,
    *,
    char_paths: dict[str, Path],
    style_paths: list[Path],
    scene_count: int | None,
    planner_model: str,
    app_mode: str = "storybook",
    review_two_pass: bool = False,
    review_checkpoint_path: Path | None = None,
    plan_manifest_path: Path | None = None,
    bible_inline: bool = False,
    bible_path: Path | None = None,
    bible_regenerate: bool = False,
) -> list[Scene]:
    """
    Tách truyện: storybook = một minh họa / section; review = nhiều beat hình ảnh nhỏ (mỗi beat một ảnh);
    manga = từng trang manga (nhiều panel), giống mangagen /api/plan.
    """
    if review_two_pass:
        if app_mode != "review":
            raise ValueError("review_two_pass chỉ hợp lệ khi app_mode == 'review'.")
        if review_checkpoint_path is None:
            raise ValueError(
                "review_two_pass cần review_checkpoint_path (vd. thư_mục_output/checkpoints/review_two_pass.json)."
            )
        return _split_review_two_pass(
            client,
            story,
            char_paths=char_paths,
            style_paths=style_paths,
            scene_count=scene_count,
            planner_model=planner_model,
            checkpoint_path=review_checkpoint_path,
            plan_manifest_path=plan_manifest_path,
            bible_inline=bible_inline,
            bible_path=bible_path,
            bible_regenerate=bible_regenerate,
        )

    asset_list = _asset_list_for_prompt(char_paths, style_paths)
    count_clause = (
        f"EXACTLY {scene_count} sections"
        if scene_count
        else (
            "a logical sequence of illustrative sections"
            if app_mode == "storybook"
            else (
                "a rich sequence of fine-grained illustration beats — each beat must be ~3–5 seconds of TTS narration (max 8s); split aggressively by visual rhythm (one still per drawable moment), one sentence often becomes 2–4 beats"
                if app_mode == "review"
                else "a logical sequence of manga pages"
            )
        )
    )
    exact_note = (
        f"IMPORTANT: You MUST generate EXACTLY {scene_count} sections. "
        f"Distribute the story content evenly across these {scene_count} sections.\n            "
        if scene_count
        else ""
    )
    # Review mode 1-pass: với rule 3–5s/beat (max 8s), một truyện vài phút đọc dễ vượt 100 beat.
    # Cap 200 để model vẫn ra JSON đầy đủ; nếu vẫn không đủ, dùng 2-pass (mặc định) thay vì 1-pass.
    review_cap_note = (
        ""
        if scene_count
        else (
            "\nHARD LIMIT: Output at most 200 beats total. "
            "If the story is too long to fit under this cap with the 3–5s/beat rule, "
            "prefer pruning low-information sentences over merging beats; ưu tiên giữ rule thời lượng.\n"
            if app_mode == "review"
            else ""
        )
    )

    if app_mode == "manga":
        prompt = f"""
Task: Act as a Manga Storyboard Artist and Scriptwriter (any genre: shōnen, shōjo, seinen, josei, 4-koma,
action, slice-of-life, fantasy, etc. — follow the USER story).
Break down the following story into {count_clause}. Each item is ONE manga PAGE (one exported image with multiple panels).

Story (full text):
{story}

Available Assets in Library: {json.dumps(asset_list, ensure_ascii=False)}

{exact_note}
For each PAGE, define:
1. storySegment: A short summary of what happens on this page (Short Text).
2. startAnchor: The EXACT first 5-10 words of this page's script segment as they appear in the original story. DO NOT CHANGE ANY WORDS.
3. endAnchor: The EXACT last 5-10 words of this page's script segment as they appear in the original story. DO NOT CHANGE ANY WORDS.
4. pageContent: A detailed IMAGE PROMPT for this ENTIRE PAGE. The final artwork is SILENT (no speech bubbles, no text). Describe:
   (a) GRID LAYOUT: Panels MUST form a clean RECTANGULAR grid only—rows stacked top-to-bottom, within each row panels left-to-right (Western reading order). Specify rows × columns if helpful (e.g. 2×3 for six panels). Equal straight gutters; rectangular panel frames only—NO diagonal panel cuts, NO full-page bleeds that break the grid, NO decorative bubble shapes as panels.
   (b) Story order: Number panels implicitly in reading order (1=left-top, then right, then next row) and describe what happens in each cell.
   (c) Per-panel: camera, action, background, and each character's pose, expression, wardrobe for THIS moment (identity vs references; outfits from story context).
   (d) Props and interactions—purely visual; do NOT plan any on-image typography.
5. storyText: The COMPLETE verbatim excerpt from the original story for this page—copy EXACTLY from the story text above. It MUST span from startAnchor through endAnchor inclusive.
6. panelCount: Integer 1-9 = number of panels on this page (must tile into a simple row/column grid).
7. panels: JSON array of EXACTLY panelCount objects, in READING ORDER (panel 1 = first cell top-left, then left-to-right, then next row). Each object MUST have:
   - "panelNumber": integer 1..panelCount (sequential).
   - "storyText": Prose from the story that this panel represents—prefer VERBATIM excerpts copied from the story text above; split the page naturally across panels. Together, all panel storyText should cover the same narrative span as storyText for this page without contradiction.
8. suggestedReferences: Filenames from Available Assets to use as references for this page.

Return ONLY a JSON array of objects with the keys:
"pageNumber", "storySegment", "startAnchor", "endAnchor", "storyText", "pageContent", "panelCount", "panels", "suggestedReferences".
"""
    elif app_mode == "review":
        prompt = f"""
Task: Act as a Story Review / Visual Reading Artist (NOT manga multi-panel pages).
Break down the following story into {count_clause}. Each item is ONE standalone illustration image (single full-bleed scene, no panel grid).

SCOPE — GENRE & SOURCE LANGUAGE:
- Apply to ANY genre (fiction or adapted nonfiction) and any setting (past/present/future, real or invented).
- Anchors and storyText MUST be VERBATIM from the user's source — never translated or paraphrased — regardless of source language (Vietnamese, English, Chinese, Japanese, Korean, French, Spanish, mixed scripts, etc.).

Goal: When the author reads the story slowly, almost every change of beat deserves its own picture — like a rich storyboard for audiobook or slide review.
Split at natural visual boundaries, including (examples):
- Arrivals / location changes / time-of-day shifts.
- A character begins speaking, presents a prop (e.g. gift box), or changes gesture.
- Camera could move closer for dialogue, then wider for a two-shot conversation.
- Flashbacks, memories, or "recalled earlier event" moments: give them THEIR OWN illustration with distinct lighting or subtle visual cue (softer edges, memory tint, faded palette) — then return to present for the next beat.

DURATION RULE — EVERY BEAT MUST FIT 3–5 SECONDS OF NARRATION; NEVER EXCEED ~8 SECONDS (CRITICAL):
- The final product is a video where TTS reads each beat's storyText. Each beat = ONE still image shown for as long as that storyText takes to be read aloud.
- Target spoken duration per beat: **3–5 seconds**. Hard ceiling: **~8 seconds**.
- Token budget per language (assume normal audiobook pace; estimate before emitting a beat):
  • Vietnamese (~3 syllables/sec): 3s ≈ 8–10 syllables, 5s ≈ 12–15, 8s ≈ 20–24 (ABSOLUTE CEILING).
  • English (~2.5 words/sec): 3s ≈ 7–8 words, 5s ≈ 12–13, 8s ≈ ~20 (ABSOLUTE CEILING).
  • Chinese / Japanese (~4 chars/sec): 3s ≈ 12 chars, 5s ≈ 20, 8s ≈ ~32.
  • Other languages: estimate against your model's own fluent reading pace, keeping the same time targets.
- HARD CHECK before emitting: estimate spoken length of storyText. If it would exceed ~8s, you MUST split. If 6–8s, prefer to split further when natural boundaries exist.
- Splitting a long sentence: split at commas / semicolons / colons / em-dashes / coordinating conjunctions ("and", "but", "then", "so", or their equivalents in the source language); split between two clauses; split between a physical action and an inner thought; a 2–3-clause sentence becomes 2–3 beats.
- SHORT beats are encouraged (one clause, ~3–10 syllables/words). A beat longer than one sentence is a SIGN to split further.
- Direct dialogue: each short utterance / each speaker turn / each pause-marked phrase → one beat.

{_VISUAL_RHYTHM_SPLIT_RULES}

GRANULARITY RULE — ONE BEAT = ONE FRAME = ONE MOMENT (CRITICAL):
- A beat is ONE freeze-frame illustration. It can show only ONE point in time, ONE place, ONE state of action.
- You MUST NOT cram multiple events / time points / locations into a single beat. If the prose moves through several moments, CREATE one beat per moment.
- Trigger a NEW beat whenever ANY of these appear in the prose:
  • A time-jump cue in any language (e.g. "the next day", "years later", "that afternoon", "before that", "later"; or its equivalent in the source language).
  • A location change (e.g. "she stepped outside", "they boarded the shuttle", "entered the ward", or any equivalent).
  • A new action by ANY character (gesture, prop pickup, turn, attack, embrace, sit/stand, etc.).
  • An EXPLICIT FLASHBACK / RECOLLECTION (not just past tense) — only when the prose clearly steps OUT of the current moment via a textual cue: e.g. "two years ago", "in my previous life", "back when…", "I remembered…", "in a dream", "earlier that month". Such a beat MUST be tagged narrativePlane="flashback".
  • A summarised journey / process (departure timing, long road, travel montage) → its own narrativePlane="flashback" montage beat.
  • A new dialogue turn / internal-monologue thought.
- HARD CHECK before emitting a beat: re-read its storyText and ask "Could a single still illustration realistically depict ALL of this in ONE frozen moment?" If no (multiple times / places / events), SPLIT IT.
- Prefer SHORT, focused beats — typically ONE short clause (~3–5 seconds of narration). A beat MAY be a single short clause / phrase.
- DO NOT skip exposition prose by attaching it to the next present beat — render it as one or more flashback beats so the audience hears the narration over a memory image.

WORKED EXAMPLES — for STRUCTURE only (NEUTRAL FICTIONAL examples in English; the user's source story may be in any language and your anchors / storyText must come VERBATIM from the user's source, not from these examples):

Example A — a long expository paragraph MUST split into MANY beats, not one:
Source (neutral English): "My younger brother was the first person in our village to be admitted to university. Our family used to sell rice at the local market. Last year the school sent a letter, so Mother gathered money and we travelled to town. The dirt road ran along the river, and we cycled for three hours. We arrived early; the boarding room was not yet ready. Mother went to help in the kitchen while I sat on the porch and re-read the admission letter."
CORRECT split → ~10 beats (mix of flashback for past/exposition and present for the now-moments).

Example B — a present sentence with two clauses MUST become 2 beats:
"I rubbed my chin." | "Yes, the ages match."

Example C — a long action / dialogue line MUST split at natural breath points:
"The boy startled, raised his head," | "hurried to his feet," | "and greeted me politely."

A WRONG output that puts an entire paragraph (or even a 2-clause sentence) into ONE beat is FORBIDDEN.

Prefer MORE beats rather than cramming several actions into one image. A single long paragraph will often become 8–15+ beats with the 3–5s rule.

Story (full text):
{story}

Available Assets in Library: {json.dumps(asset_list, ensure_ascii=False)}

{exact_note}
{review_cap_note}
NARRATIVE PLANE (critical — tag EVERY beat):
- Each beat MUST include narrativePlane: exactly "present" OR "flashback".
- "present" = the MAIN narrative timeline of the story — events the story is currently dramatising "now", regardless of grammatical tense. First-person retrospective narration (the narrator looking back in past tense) is STILL "present" as long as the beat dramatises the main story timeline. DEFAULT to "present" when in doubt; clear realistic image, "normal" lighting, no memory filter.
- "flashback" = the prose explicitly steps OUT of the main timeline into a recollection, dream, memory, or earlier-era exposition. Trigger ONLY with a clear textual cue — e.g. "X năm trước…", "kiếp trước…", "ngày xưa…", "hồi nhỏ…", "ta nhớ lại…", "trong giấc mơ…", "two years ago…", "in my previous life…", "back when…", "I remembered…" — or a verbatim recalled memory inside an ongoing present scene. Visuals clearly differ from the present (slightly reduced contrast, memory tint, soft edges, light grain, or shifted palette) so viewers can immediately distinguish it from adjacent present beats.
- DO NOT tag a beat as "flashback" just because the verb is past tense, the narration is first-person retrospective, or the prose mentions a fact that happened before the scene started. If the dramatised action is happening in the main story moment, it is "present".

PRESENT-THREAD CONTINUITY (DEFINITION — applies to every "consecutive beats" rule below):
- The "present thread" is the chain of beats with narrativePlane="present" READ IN ORDER and IGNORING flashback beats inserted between them. So in a sequence `present₁ → flashback → present₂ → present₃ → flashback → present₄`, the present thread is `present₁ → present₂ → present₃ → present₄`.
- Whenever a rule says "consecutive beats", "the previous beat", or "the next beat" for continuity (setting / wardrobe / camera variation / co-present cast), it means consecutive WITHIN THE PRESENT THREAD — flashback beats are skipped. Flashback beats only need to follow flashback rules (memory styling) and do not reset the present baseline.
- DO NOT BREAK A CONTINUOUS PRESENT SCENE INTO INDEPENDENT FRAGMENTS just because a flashback / recollection is inserted in the middle. After the flashback ends, the next present beat resumes the SAME continuous scene as the present beat just before the flashback — same room, same conversation, same minute, same outfits — unless the prose explicitly moves to a new place or changes outfits.

CONSISTENCY WITHIN A "present" THREAD (unless prose explicitly changes scene or outfit):
- Consecutive present beats (per PRESENT-THREAD CONTINUITY above) MUST keep the SAME setting (location, time of day, indoor/outdoor) and the SAME wardrobe per character — one continuous reality thread.
- A flashback beat in the middle does NOT reset this baseline. The present beat after the flashback restores the same setting + wardrobe as the present beat before the flashback.

For each beat, define:
1. storySegment: Short label for this beat (what the viewer should notice in this frame).
2. startAnchor: The EXACT first 5–10 words of this beat's excerpt as they appear in the original story (source language). DO NOT CHANGE ANY WORDS.
3. endAnchor: The EXACT last 5–10 words of this beat's excerpt as they appear in the original story (source language). DO NOT CHANGE ANY WORDS.
4. pageContent: A detailed IMAGE PROMPT for THIS single illustration only, written in ENGLISH (regardless of source language; it is a brief for the image model). Same layers as premium storybook art direction:
   (a) Setting, time, lighting, camera/composition (close-up / medium / wide as appropriate).
   (b) Every visible character: emotion and SPECIFIC facial expression + posture for THIS beat.
   (c) Wardrobe for THIS moment from story context; references are identity-only.
   (d) Key props and interactions.
   (e) CAMERA / SHOT (REQUIRED per beat): explicit cinematic shot tailored to this beat — pick shot size (extreme close-up | close-up | medium close-up | medium | medium wide | wide | extreme wide) + camera angle (eye-level | low-angle | high-angle | over-the-shoulder | POV | top-down | slight Dutch tilt) + staging (centered / left-third / right-third / through-doorway / foreground silhouette) + depth-of-field cue. VARY framing between consecutive beats IN THE SAME PRESENT THREAD (skip flashback beats when comparing — see PRESENT-THREAD CONTINUITY) so the visuals feel like film cuts, not the same shot repeated. Flashback beats use their own memory framing and do not need to follow this present-thread variation rule.
   (f) CO-PRESENT CAST (STRICT — MUST APPLY for every dialogue / group beat):
       • If the beat takes place inside a conversation / group scene with 2+ characters in the same location, pageContent MUST acknowledge ALL of them, even on a close-up of one.
       • For close-ups / OTS / extreme close-ups: keep the focused character as the main subject AND describe the other co-present characters as part of the staging — e.g. "in the background, [X] stands at left, [Y] sits at the head of the table"; "over [X]'s shoulder, [Y] is partially visible at frame-right"; or append a final clause: "OFF-FRAME COMPANIONS (present in scene but outside this frame): [X], [Y]".
       • NEVER describe an empty interior / empty hall / empty courtyard for a beat whose conversation has 2+ co-present people. Their presence must be made readable to the illustrator (background, silhouette, edge of frame, OTS, or explicit off-frame note).
       • When prose names a new speaker mid-conversation, the cut may push them to foreground, but the original co-present cast stays visible (background) or is named in OFF-FRAME COMPANIONS — they did not vanish.
   Start pageContent by stating explicitly: "NARRATIVE: present reality — …" OR "NARRATIVE: flashback/memory — …" then the visual brief. For present beats, repeat the SAME location + wardrobe keywords you use across consecutive present beats until the prose changes setting/outfit.
   NO dialogue text rendered in the image; NO captions; NO speech bubbles.
5. storyText: Verbatim excerpt from the original story for THIS beat ONLY (source language) — copy EXACTLY; it must span startAnchor through endAnchor.
   Consecutive beats MUST have **ZERO overlapping prose**: partition the story into non-overlapping spans only.
   Never duplicate the end of the previous beat's storyText at the start of the next beat (including flashback → present cuts).
6. panelCount: ALWAYS 1 (this mode never outputs multi-panel pages).
7. suggestedReferences — PHYSICAL CO-PRESENCE RULE (STRICT — read carefully):
   This is a list of filenames from Available Assets for characters PHYSICALLY PRESENT in the beat — NOT for everyone merely mentioned.
   In pageContent, also write at the very end a single-line audit block:
     "WHO IS WHERE: SPEAKER → <name or 'narrator'>; LISTENERS PRESENT → <name>, <name>; OFF-FRAME but in scene → <name> (only if needed)."
   MUST INCLUDE:
     • The SPEAKER of this beat (if they have a ref).
     • Anyone LISTENING / standing nearby / sitting in the same enclosed or open space (room, hall, hallway, courtyard, plaza, vehicle, operating room, ship deck, cave, etc.) — even silent and not the focal subject.
     • Anyone the prose explicitly describes as physically present.
     • In a continuous conversation / interaction at THE SAME location: the SAME co-present cast must appear in refs for ALL beats in that thread (wide → close-up → OTS → ECU) until someone leaves or the scene moves elsewhere.
   MUST EXCLUDE:
     • Characters merely MENTIONED in dialogue / narration but NOT physically present (generic example: a line refers to "a third party" who is not in the room → DO NOT add their ref).
     • Characters who left in the previous scene/location.
     • Characters who only exist inside an inner-monologue / brief recollection of a present beat (express the memory as a thought cue on someone PRESENT, do NOT draw the absent person).
     • Unnamed crowds / extras without a portrait ref.
   Pattern (replace placeholders with the right Available Assets filenames + character names from THIS story):
     - two characters physically in the same room → every beat in that conversation thread keeps refs for BOTH (wide through ECU).
     - a beat whose dialogue mentions a third character not in the frame → do NOT add a ref for that third character.
8. narrativePlane: "present" or "flashback" (required).

{_WAN_MOTION_PROMPT_RULES}

Return ONLY a JSON array of objects with the keys (EVERY object MUST contain ALL 10 keys, including a non-empty `motionPrompt` string — see rule 9 above):
"pageNumber", "storySegment", "startAnchor", "endAnchor", "storyText", "pageContent", "panelCount", "suggestedReferences", "narrativePlane", "motionPrompt".
"""
    else:
        prompt = f"""
Task: Act as an Illustrated Narrative Art Director (storybook mode — one hero image per section, any audience/genre).
Break down the following story into {count_clause}. Each section should represent a key visual moment
that can be captured in a SINGLE, high-fidelity illustration (not manga panels).

Story (full text):
{story}

Available Assets in Library: {json.dumps(asset_list, ensure_ascii=False)}

{exact_note}
For each section, define:
1. storySegment: A short, concise summary of the story text for this section (Short Text).
2. startAnchor: The EXACT first 5-10 words of this section as they appear in the original story. DO NOT CHANGE ANY WORDS.
3. endAnchor: The EXACT last 5-10 words of this section as they appear in the original story. DO NOT CHANGE ANY WORDS.
4. pageContent: A detailed IMAGE PROMPT for THIS scene only. It MUST include ALL of the following layers:
   (a) Setting, time of day, weather, lighting, and camera/composition.
   (b) For EVERY character who appears: their CURRENT emotional state matching THIS story beat (e.g. dread, relief, anger, wonder); describe SPECIFIC facial expression (eyes, brows, mouth) and body language/posture—avoid neutral or blank faces unless the story clearly calls for a numb/stoic moment.
   (c) WARDROBE FOR THIS SCENE: describe clothing, accessories, and grooming appropriate to THIS moment and setting (formal/casual/work/uniform/sleepwear/weather gear, etc.). Costumes CHANGE across scenes when the story context changes; do NOT assume everyone wears the same outfit as in any reference library image—references are for identity, not a locked costume.
   (d) Interactions between characters and key props if any.
   NO dialogue text in pageContent; NO captions.
5. storyText: The COMPLETE verbatim excerpt from the original story for this section ONLY—copy EXACTLY from the story text above (same characters, punctuation, line breaks). It MUST begin with startAnchor and end with endAnchor (include full sentences between them). Do not summarize or paraphrase.
6. panelCount: ALWAYS 1.
7. suggestedReferences: A list of filenames from the provided Available Assets that should guide the artist.

Return ONLY a JSON array of objects with the keys:
"pageNumber", "storySegment", "startAnchor", "endAnchor", "storyText", "pageContent", "panelCount", "suggestedReferences".
"""

    response = client.models.generate_content(
        model=planner_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )

    raw = response.text or ""
    data = _parse_planner_json_array(raw, client, planner_model)

    return _manifest_rows_to_scenes(data, story, app_mode)


def _manifest_rows_to_scenes(
    data: list[Any],
    story: str,
    app_mode: str,
) -> list[Scene]:
    """Chuyển mảng object planner/manifest (scenes.json) thành list Scene."""
    if not isinstance(data, list):
        raise ValueError("Plan phải là một mảng JSON các cảnh.")
    scenes: list[Scene] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        start_a = str(row.get("startAnchor", ""))
        end_a = str(row.get("endAnchor", ""))
        raw_story_text = str(row.get("storyText") or "").strip()
        fallback = _verbatim_between_anchors(story, start_a, end_a)
        resolved_text = raw_story_text if raw_story_text else fallback

        try:
            pc = int(row.get("panelCount", 1))
        except (TypeError, ValueError):
            pc = 1
        if app_mode == "manga":
            pc = max(1, min(9, pc))
        else:
            pc = 1

        panel_rows: list[dict[str, Any]] = []
        if app_mode == "manga":
            panel_rows = _parse_panels_from_row(row.get("panels"), pc)
            n_from_planner = len(panel_rows)
            if len(panel_rows) != pc:
                pg = row.get("pageNumber", i + 1)
                if resolved_text:
                    chunks = _even_split_story_text(resolved_text, pc)
                    panel_rows = [
                        {
                            "panelNumber": k + 1,
                            "storyText": chunks[k] if k < len(chunks) else "",
                        }
                        for k in range(pc)
                    ]
                    print(
                        f"Cảnh báo trang {pg}: panels từ planner ({n_from_planner}) ≠ panelCount {pc}; "
                        f"đã chia storyText thành {pc} phần.",
                        file=sys.stderr,
                    )
                else:
                    while len(panel_rows) < pc:
                        panel_rows.append(
                            {"panelNumber": len(panel_rows) + 1, "storyText": ""}
                        )
                    panel_rows = panel_rows[:pc]
                    for idx, item in enumerate(panel_rows, start=1):
                        item["panelNumber"] = idx
                    print(
                        f"Cảnh báo trang {pg}: thiếu storyText để chia; đã pad {pc} panel.",
                        file=sys.stderr,
                    )

        osn, oss = _parse_outline_section_meta(row)
        scenes.append(
            Scene(
                page_number=int(row.get("pageNumber", i + 1)),
                story_segment=str(row.get("storySegment", "")),
                start_anchor=start_a,
                end_anchor=end_a,
                page_content=str(row.get("pageContent", "")),
                story_text=resolved_text,
                panel_count=pc,
                suggested_references=list(row.get("suggestedReferences") or []),
                panels=panel_rows,
                narrative_plane=_parse_narrative_plane(row)
                if app_mode == "review"
                else "present",
                outline_section_number=osn if app_mode == "review" else None,
                outline_section_summary=oss if app_mode == "review" else "",
            )
        )
    if app_mode == "review":
        n_fix = _dedupe_review_story_text_overlaps(scenes)
        if n_fix:
            print(
                f"Đã chuẩn hoá storyText (overlap giữa {n_fix} cặp beat liền kề).",
                file=sys.stderr,
            )
        _warn_long_story_text_beats(scenes)
        _propagate_copresent_cast_in_scenes(scenes)
    return scenes


def _infer_render_mode_from_manifest(rows: list[Any], fallback: str) -> str:
    for row in rows:
        if isinstance(row, dict):
            m = row.get("mode")
            if m in ("storybook", "manga", "review"):
                return str(m)
    return fallback


def _scene_output_image_exists(out_dir: Path, page_number: int) -> bool:
    stem = f"scene_{int(page_number):02d}"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if (out_dir / f"{stem}{ext}").is_file():
            return True
    return False


def _first_existing_scene_image(out_dir: Path, page_number: int) -> Path | None:
    stem = f"scene_{int(page_number):02d}"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = out_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _pick_reference_paths(
    _scene: Scene,
    char_paths: dict[str, Path],
    style_paths: list[Path],
) -> list[tuple[str, Path]]:
    """
    Mỗi cảnh đều nhận toàn bộ ảnh nhân vật + style (cùng thứ tự), giống việc chọn ref trong mangagen
    để model luôn khóa ngoại hình nhân vật.
    """
    ordered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for label, p in char_paths.items():
        key = str(p.resolve())
        if key not in seen:
            ordered.append((label, p))
            seen.add(key)
    for p in style_paths:
        key = str(p.resolve())
        if key not in seen:
            ordered.append((p.stem, p))
            seen.add(key)
    return ordered


def _storybook_image_instruction(
    *,
    color_mode: str,
    aspect_ratio_key: str,
    art_style_key: str,
) -> str:
    aspect_ratio_map = {
        "square": "1:1 (Square)",
        "portrait": "2:3 (Portrait)",
        "landscape": "3:2 (Landscape)",
        "16:9": "16:9 (Widescreen)",
    }
    art_style_map = {
        "watercolor": "Watercolor painting with soft, blended colors and organic textures.",
        "oil_painting": "Rich oil painting with visible brushstrokes and deep colors.",
        "digital_illustration": "Modern digital illustration with clean lines and vibrant colors.",
        "anime": "Stylized anime illustration with expressive characters and dynamic compositions.",
        "storybook_classic": "Classic children's book illustration, warm and inviting with a hand-drawn quality.",
        "realistic": "Photorealistic digital art with high detail and accurate lighting.",
        "ghibli": (
            "Studio Ghibli inspired hand-drawn anime: soft watercolor-style painted backgrounds with rich naturalistic detail "
            "(grass, leaves, clouds, sky gradients, weathered wood, fabric folds); gentle 2D cel shading with limited tonal steps "
            "and warm, painterly lighting (golden-hour warmth, dappled sunlight, gentle rim light); subtly muted, harmonious color palette "
            "(pastel greens, sky blues, cream highlights, soft warm shadows). Characters: rounded soft features, expressive but calm faces, "
            "simple clean lineart, slightly stylized proportions in the spirit of Hayao Miyazaki / Studio Ghibli films "
            "(e.g. Spirited Away, My Neighbor Totoro, Howl's Moving Castle). Atmospheric, slow-cinema mood; tactile, lived-in environments; "
            "no plastic/3D-CGI look, no harsh airbrush gradients."
        ),
    }
    ar = aspect_ratio_map.get(aspect_ratio_key, "2:3 (Portrait)")
    style = art_style_map.get(art_style_key, art_style_map["storybook_classic"])
    color = "Full color." if color_mode == "color" else "Black and white with grayscale shading."
    return f"""Task: Acting as a professional narrative illustrator (any genre / audience: literary, romance, thriller, sci-fi, fantasy, slice-of-life, historical, children's, etc.), generate a SINGLE, high-fidelity illustration based on the story snippet and references provided.

VISUAL STYLE:
- Color: {color}
- Art Style: {style}
- Aspect Ratio: Generate the image in a {ar} aspect ratio.
- EXTREMELY IMPORTANT: Study and replicate the visual style from the provided 'Style' references if any are present.

IDENTITY vs COSTUME (critical):
- Character reference images are IDENTITY anchors only: keep the SAME person across scenes—face shape, jaw, eye spacing, nose/lips, skin tone, hair COLOR and STYLE, age, body type.
- Do NOT copy outfits, hats, jewelry, or uniforms from reference portraits if the Scene illustration brief specifies different wardrobe for this moment. Wardrobe MUST follow the text brief (period, weather, role, social context).
- Palette for clothing may vary by scene; keep skin/hair identity stable.

PERFORMANCE / EMOTION:
- Facial expressions and posture MUST match the narrative beat in the Scene illustration brief and story segment—show readable, specific emotion (eyes, eyebrows, mouth, tension in hands/shoulders). Avoid generic blank or stiff expressions unless the story demands it.
- Use staging and lighting to reinforce mood.

CINEMATIC FRAMING (treat each beat like a film cut):
- Choose shot size, camera angle, distance, and composition that BEST EXPRESS this beat's emotion and action — DO NOT default to one repeating framing.
- Vocabulary you should freely pick from per beat: extreme close-up, close-up, medium close-up, medium, medium wide, wide, extreme wide; eye-level, low-angle, high-angle, over-the-shoulder, POV, top-down, slight Dutch tilt; centered vs left/right-third staging; shallow vs deep depth-of-field; foreground silhouette / through-doorway framing.
- Match shot to story: intimate dialogue → close-up / OTS; physical action → medium-wide or wide; isolation / vastness → wide; introspection → expressive close-up; discovery → POV / OTS into object; reveal → push-in or pull-back framing.
- Across consecutive beats in the same setting, VARY at least one of: shot size, camera angle, or staging position — avoid copy-pasted compositions.

STRICT REQUIREMENTS:
- Output MUST be a high-quality SINGLE image. Absolutely NO multi-panel layouts.
- The image MUST contain ZERO readable text of any kind: no captions/subtitles, no dialogue balloons, no thought bubbles, no narration boxes, no SFX lettering, no titles, no logos with letters, no watermarks, no signatures, no UI overlays, and no signs with readable words.
- If any sign/label/phone screen/book appears in the scene, render it as abstract shapes/scribbles with NO legible characters.
- Focus on one cinematic moment that matches the emotional arc described.
- If you are being asked to call a tool, IGNORE that and instead directly output the image contents as your response."""


def _manga_fullpage_instruction(
    *,
    color_mode: str,
    aspect_ratio_key: str,
    panels: int,
    art_style_key: str,
) -> str:
    """Trang manga: lưới chữ nhật L→R, T→D; không chữ trên ảnh (tránh lỗi font tiếng Việt / glyph)."""
    ar_map = {
        "square": "1:1 (Square)",
        "portrait": "2:3 (Standard Manga Page)",
        "landscape": "3:2 (Wide)",
        "16:9": "16:9 (Widescreen)",
    }
    ar = ar_map.get(aspect_ratio_key, "2:3 (Standard Manga Page)")
    color = (
        "Full color digital illustration."
        if color_mode == "color"
        else "Traditional black and white manga with screen tones and ink textures."
    )
    style_note = {
        "watercolor": "Soft ink/wash manga hybrid.",
        "oil_painting": "Painterly manga rendering.",
        "digital_illustration": "Clean line art, crisp fills or cel shading.",
        "anime": "Expressive anime/manga eyes and dynamic line weight.",
        "storybook_classic": "Rounded, approachable manga-storybook hybrid.",
        "realistic": "Semi-realistic shading and anatomy within manga framing.",
        "ghibli": (
            "Studio Ghibli inspired hand-drawn anime within a manga page: soft watercolor-style painted backgrounds, "
            "gentle cel shading with warm painterly lighting, muted harmonious palette, rounded expressive characters "
            "in the spirit of Hayao Miyazaki / Studio Ghibli films."
        ),
    }.get(art_style_key, "Professional seinen/shonen-style manga clarity.")

    return f"""Task: Acting as a professional manga artist, generate a SINGLE high-fidelity manga PAGE image based on the story snippet and references provided.

VISUAL STYLE:
- {color}
- Line/render style: {style_note}
- Aspect ratio: compose the page with a {ar} overall proportion (one vertical page).

PANEL GRID (mandatory):
- Arrange exactly {panels} panels as a RECTANGULAR grid: complete rows from TOP to BOTTOM; within each row, panels read LEFT to RIGHT (Western order, NOT right-to-left Japanese).
- Each panel is a rectangle with straight edges; use consistent gutter width between cells.
- FORBIDDEN: diagonal panel splits, jagged "shattered" layouts, full-bleed panels that break the grid, speech-bubble-shaped frames, or overlapping panel borders that obscure reading order.
- Optional: slightly vary panel heights within a row only if the brief demands it, but keep a clear rectangular lattice (no chaos layouts).

TYPOGRAPHY (mandatory — silent page):
- The image MUST contain ZERO text of any kind: no dialogue balloons, thought bubbles, narration boxes, sound-effect (SFX) lettering, captions, titles, signs with readable words, logos with letters, watermarks, or subtitles.
- Do not render Vietnamese, Latin, CJK, or any alphabet—purely visual storytelling. (Spoken story will be added outside this image, e.g. TTS.)

IDENTITY vs PAGE CONTENT:
- Reference images lock facial likeness, hair, and body type for each named character.
- Wardrobe, props, poses, and what happens in each grid cell follow the PAGE BRIEF below—not the outfit seen in reference portraits.

REQUIREMENTS:
- Output ONE single image of the full page (not multiple files).
- DO NOT return JSON, markdown, or code—only the finished artwork.
- Ensure character likeness consistency with references.
- If asked to use tools, IGNORE and output the image directly."""


def _image_part_from_path(path: Path) -> types.Part:
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return types.Part.from_bytes(data=data, mime_type=mime)


def _extract_inline_image_from_response(response) -> tuple[bytes | None, str | None]:
    """
    Lấy ảnh inline đầu tiên từ phản hồi. candidate.content có thể là None khi bị safety / lỗi model.
    """
    for cand in response.candidates or []:
        cont = getattr(cand, "content", None)
        if cont is None:
            continue
        for part in cont.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                mime = getattr(inline, "mime_type", None) or "image/png"
                return inline.data, mime
    return None, None


def _response_debug_hint(response) -> str:
    """Chuỗi ngắn để log khi không có ảnh (finish_reason / prompt_feedback)."""
    hints: list[str] = []
    pf = getattr(response, "prompt_feedback", None)
    if pf is not None:
        hints.append(f"prompt_feedback={pf}")
    for i, cand in enumerate(response.candidates or []):
        fr = getattr(cand, "finish_reason", None)
        has_c = getattr(cand, "content", None) is not None
        si = getattr(cand, "safety_ratings", None)
        hints.append(f"cand[{i}] content={has_c} finish_reason={fr} safety={si}")
    return " | ".join(hints) if hints else "no debug metadata"


# --- Anatomy guard (extra limbs/fingers) + Text guard ---
# Lỗi artifact phổ biến của model sinh ảnh. Reuse cho cả scene và section keyframe.
_ANATOMY_AND_TEXT_GUARD = (
    "\n\nANATOMY GUARD (STRICT):\n"
    "- Human characters MUST have anatomically correct bodies.\n"
    "- Exactly TWO arms and TWO hands per person; no extra arms/hands.\n"
    "- Each visible hand has exactly five fingers (unless the brief explicitly says otherwise).\n"
    "- No duplicated limbs, no fused fingers, no extra fingers, no malformed hands.\n"
    "- If a hand is hard to render cleanly, pose it naturally or partially occlude it—DO NOT add extra limbs.\n"
    "\nTEXT GUARD (STRICT):\n"
    "- ZERO readable text in the image: no captions, subtitles, titles, logos with letters, watermarks, signatures, UI text, or readable signage.\n"
    "- The dialogue / voiceover / story text provided in the prompt is ONLY semantic context — it MUST NOT appear in the rendered image as printed letters, subtitles, speech bubbles, captions, or written words anywhere (including on phone screens, signs, papers, books, computer monitors).\n"
    "- Do NOT render Vietnamese, Latin, CJK, or any alphabet/glyphs at all in the image. Express dialogue/emotion only through facial expression, body language, gesture, eye contact, and staging.\n"
    "- If text-like elements must appear (e.g., phone/screen/sign), render as abstract shapes/scribbles with NO legible characters.\n"
    "\nNEGATIVE (avoid): extra arms, extra hands, extra fingers, duplicated limbs, mutated hands, malformed hands, deformed anatomy, "
    "text, letters, words, captions, subtitles, watermark, signature, logo.\n"
)


def generate_section_character_ref(
    client: genai.Client,
    *,
    section_num: int,
    char_label: str,
    char_portrait_path: Path,
    section_keyframe_path: Path | None,
    baseline_page_content: str,
    section_page_contents: list[str],
    section_summary: str,
    image_model: str,
    color_mode: str,
    art_style: str,
    section_design: dict[str, Any] | None = None,
) -> tuple[bytes, str | None]:
    """
    Sinh "section character reference" cho 1 nhân vật trong 1 section:
    Giữ **nhận diện khuôn mặt / thân hình** đúng portrait chuẩn trong auto_refs; chỉ thay **trang phục + tóc + grooming**
    theo SECTION WARDROBE BIBLE và mô tả beat của section (bối cảnh).

    PRIORITY ORDER cho wardrobe (cứng):
      1. SECTION WARDROBE BIBLE (text)     ← ground truth outfit/hair cho section.
      2. BASELINE TEXT + ADDITIONAL DESCRIPTIONS (pageContent).
      3. Portrait auto_refs (ảnh)        ← CHỈ identity (face/body), không copy outfit cũ.
      4. section_keyframe_path (ảnh)     ← chỉ palette/lighting/setting nếu có.
    """
    system_block = _portrait_instruction(
        color_mode=color_mode,
        aspect_ratio_key="square",
        art_style_key=art_style,
    )

    extras_block = ""
    if section_page_contents:
        unique = []
        for pc in section_page_contents:
            if pc and pc != baseline_page_content and pc not in unique:
                unique.append(pc)
        if unique:
            merged = "\n---\n".join(p[:800] for p in unique[:6])
            extras_block = (
                f"\n\nADDITIONAL DESCRIPTIONS of «{char_label}» in this section "
                "(supplementary canonical wording):\n"
                f"{merged}"
            )

    char_bible_block = ""
    setting_bible_block = ""
    if section_design and isinstance(section_design, dict):
        chars = section_design.get("characters") or {}
        info = chars.get(char_label) if isinstance(chars, dict) else None
        if isinstance(info, dict):
            outfit = (info.get("outfit") or "").strip()
            hair = (info.get("hair") or "").strip()
            footwear = (info.get("footwear") or "").strip()
            accessories = (info.get("accessories") or "").strip()
            grooming = (info.get("groomingNotes") or "").strip()
            lines = [
                f"- outfit: {outfit}" if outfit else "",
                f"- hair: {hair}" if hair else "",
                f"- footwear: {footwear}" if footwear else "",
                f"- accessories: {accessories}" if accessories else "",
                f"- grooming/expression: {grooming}" if grooming else "",
            ]
            joined = "\n".join(li for li in lines if li)
            if joined:
                char_bible_block = (
                    f"\n\n=== SECTION WARDROBE BIBLE for «{char_label}» (HIGHEST PRIORITY — render EXACTLY this) ===\n"
                    f"{joined}\n"
                    "=== END BIBLE ===\n"
                )
        setting = section_design.get("setting") or {}
        if isinstance(setting, dict) and setting:
            sline = "; ".join(
                f"{k}: {(setting.get(k) or '').strip()}"
                for k in ("location", "timeOfDay", "lighting", "weather", "ambientPalette")
                if (setting.get(k) or "").strip()
            )
            if sline:
                setting_bible_block = (
                    f"\n\nSECTION SETTING BIBLE (use ONLY for ambient palette + light direction; the character must be in front of a soft, blurred suggestion of this setting): {sline}."
                )

    text_a = (
        f"{system_block}"
        f"\n\nSECTION {section_num} — CHARACTER WARDROBE SHEET for «{char_label}».\n"
        "Purpose: a SINGLE-CHARACTER reference image showing this character WEARING THE EXACT WARDROBE + HAIR + GROOMING + "
        "ACCESSORIES specified by the SECTION WARDROBE BIBLE for this character. This sheet will be reused as a reference "
        "for every downstream beat in this section.\n"
        f"{char_bible_block}"
        "\nWARDROBE SOURCE PRIORITY (STRICT — do NOT deviate):\n"
        "1. **The SECTION WARDROBE BIBLE block above for this character is the GROUND TRUTH.** Render its outfit, hair, footwear, accessories, grooming EXACTLY.\n"
        "2. BASELINE TEXT and ADDITIONAL DESCRIPTIONS below: secondary support, MUST not contradict the BIBLE.\n"
        "3. The IDENTITY portrait below is ONLY for facial likeness + hair color/length + body type. **DO NOT copy the outfit from the portrait** if it differs from the BIBLE.\n"
        "4. The SECTION ENVIRONMENT KEYFRAME image (if attached) is ONLY for **lighting tone + color palette + setting hint** — it contains NO people. **DO NOT copy any outfit from it** (it has none).\n"
        "\nCOMPOSITION REQUIREMENTS:\n"
        "- Solo character only — no other people in frame.\n"
        "- Full-body, head to feet, facing camera at slight 3/4 angle. Whole outfit must be visible (top, bottom, shoes if relevant).\n"
        "- Neutral standing pose, arms relaxed at sides or slightly forward. Calm, lightly engaged expression. NOT a climactic action.\n"
        "- Background: minimal/clean — a soft, slightly blurred suggestion of the section's setting (same color palette + lighting tone as the keyframe / setting bible), with NO other characters and NO complex props.\n"
        "- This is a wardrobe/identity SHEET, not a story moment. NO text on the image.\n"
        f"\nSECTION SUMMARY: {section_summary or '(none)'}"
        f"{setting_bible_block}"
        f"\n\nBASELINE TEXT (supplementary, MUST not contradict the BIBLE above):\n"
        f"{baseline_page_content}"
        f"{extras_block}"
    )

    bind = (
        f"\n\nREMINDER: Output is a SOLO full-body wardrobe sheet of «{char_label}». "
        "Priority: SECTION WARDROBE BIBLE > baseline text > portrait > keyframe. "
        "If anything disagrees with the BIBLE on wardrobe/hair, **BIBLE WINS**."
    )

    parts: list[types.Part] = [
        types.Part.from_text(text=text_a),
        types.Part.from_text(text=_ANATOMY_AND_TEXT_GUARD),
    ]

    parts.append(
        types.Part.from_text(
            text=(
                f"\n\nCANONICAL IDENTITY portrait for «{char_label}» from auto_refs (file {char_portrait_path.name}): "
                "this is the **ground-truth face + body type** for this character across the whole project. "
                "Use it ONLY for facial likeness, proportions, skin tone, hair color/length. "
                "**DO NOT copy the outfit from this portrait** — the outfit MUST match the SECTION WARDROBE BIBLE above."
            )
        )
    )
    parts.append(_image_part_from_path(char_portrait_path))

    if section_keyframe_path is not None and section_keyframe_path.is_file():
        parts.append(
            types.Part.from_text(
                text=(
                    "\n\nSECTION ENVIRONMENT KEYFRAME (empty-set establishing still, NO people) — "
                    "use ONLY for color temperature, light direction, ambient palette, and rough setting feel. "
                    "**DO NOT copy any outfit from this image** (there is no character in it); outfits come from the SECTION WARDROBE BIBLE."
                )
            )
        )
        parts.append(_image_part_from_path(section_keyframe_path))

    parts.append(types.Part.from_text(text=bind))

    response = _call_image_model_with_retry(
        lambda: client.models.generate_content(
            model=image_model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.4,
            ),
        ),
        label=f"section_char_ref(section={section_num}, char={char_label})",
    )

    img_bytes, img_mime = _extract_inline_image_from_response(response)
    if img_bytes is not None:
        return img_bytes, img_mime

    text_fallback = response.text or ""
    raise RuntimeError(
        f"Section character ref ({char_label}, section {section_num}): model không trả về ảnh. "
        f"Lời thoại: {text_fallback[:400]} | {_response_debug_hint(response)}"
    )


def generate_section_keyframe(
    client: genai.Client,
    *,
    section_num: int,
    section_summary: str,
    baseline_page_content: str,
    extra_present_excerpts: list[str],
    ref_paths: list[tuple[str, Path]],
    image_model: str,
    color_mode: str,
    aspect_ratio: str,
    art_style: str,
    section_design: dict[str, Any] | None = None,
    ref_kind: str = "portrait",
) -> tuple[bytes, str | None]:
    """
    Sinh "section environment keyframe" = production establishing still **CHỈ bối cảnh** cho 1 outline section.
    KHÔNG vẽ nhân vật trong ảnh này. Mục đích: lock location + props + lighting + palette để các beat trong
    section bám theo. Wardrobe nhân vật được khoá riêng qua các wardrobe sheet (section_XX__<tên>.*).

    `ref_paths` / `ref_kind` được giữ trong signature cho tương thích, nhưng KHÔNG đính kèm cho prompt
    (giúp model không "lén" vẽ nhân vật vào establishing shot).
    """
    _ = (ref_paths, ref_kind)  # signature compat, không dùng
    system_block = _storybook_image_instruction(
        color_mode=color_mode,
        aspect_ratio_key=aspect_ratio,
        art_style_key=art_style,
    )

    bible_block = _format_wardrobe_bible_for_prompt(section_design)
    setting_only_lines: list[str] = []
    if isinstance(section_design, dict):
        setting = section_design.get("setting") or {}
        if isinstance(setting, dict) and setting:
            for k in ("location", "timeOfDay", "lighting", "weather", "ambientPalette"):
                v = (setting.get(k) or "").strip()
                if v:
                    setting_only_lines.append(f"- {k}: {v}")
    setting_only_block = (
        "\n\nSECTION SETTING BIBLE (HIGHEST PRIORITY ground truth — copy location, time-of-day, lighting, "
        "weather, ambient palette EXACTLY):\n" + "\n".join(setting_only_lines)
        if setting_only_lines
        else (
            "\n\nSECTION WARDROBE BIBLE (use ONLY the 'setting' block; ignore character wardrobe — this image "
            "MUST NOT contain people):\n" + bible_block
            if bible_block
            else ""
        )
    )

    extras_block = ""
    if extra_present_excerpts:
        merged = "\n---\n".join(e[:600] for e in extra_present_excerpts[:4])
        extras_block = (
            "\n\nADDITIONAL PRESENT BEATS DESCRIPTIONS in this section "
            "(use ONLY for environment, props, time-of-day clues — IGNORE all character actions):\n"
            f"{merged}"
        )

    text_a = (
        f"{system_block}"
        f"\n\nSECTION {section_num} ENVIRONMENT KEYFRAME — production establishing still (NO PEOPLE).\n"
        "Purpose: a SINGLE empty-set reference image that LOCKS the visual baseline (location, props, lighting, palette) "
        "for the ENTIRE outline section. Every present-reality beat downstream will use this as a setting anchor.\n"
        "**HARD CONSTRAINT — NO HUMANS IN THIS IMAGE**:\n"
        "- Do NOT depict any person, body part, silhouette, shadow of a person, reflection of a person, mannequin, "
        "or implied character. The frame is an empty room / empty environment.\n"
        "- Personal belongings (a chart on the wall, a stethoscope on a desk, a coat on a chair) ARE allowed if the "
        "section text mentions them as props.\n"
        f"\nSECTION SUMMARY:\n{section_summary or '(none)'}"
        f"{setting_only_block}"
        f"\n\nSCENE BACKDROP DESCRIPTION (for environment, NOT for posing characters):\n"
        f"{baseline_page_content}"
        f"{extras_block}"
        "\n\nCOMPOSITION REQUIREMENTS:\n"
        "- Wide / medium-wide environment establishing shot. No characters in frame at all.\n"
        "- Show the COMPLETE setting clearly: room/location with key props, full background readable.\n"
        "- Lighting and palette per the BIBLE; consistent across the whole frame.\n"
        "- Production-design quality (like a film set still or game environment art); NO text, NO captions."
    )

    parts: list[types.Part] = [
        types.Part.from_text(text=text_a),
        types.Part.from_text(text=_ANATOMY_AND_TEXT_GUARD),
        types.Part.from_text(
            text=(
                "\n\nFINAL REMINDER: NO PEOPLE in the output image. This is an environment / location plate "
                "used as a setting anchor; characters will be added in downstream beat illustrations."
            )
        ),
    ]

    response = _call_image_model_with_retry(
        lambda: client.models.generate_content(
            model=image_model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.45,
            ),
        ),
        label=f"section_env_keyframe(section={section_num})",
    )

    img_bytes, img_mime = _extract_inline_image_from_response(response)
    if img_bytes is not None:
        return img_bytes, img_mime

    text_fallback = response.text or ""
    raise RuntimeError(
        "Section environment keyframe: model không trả về ảnh. Kiểm tra image_model / quota / safety. "
        f"Lời thoại: {text_fallback[:400]} | {_response_debug_hint(response)}"
    )


def generate_scene_image(
    client: genai.Client,
    scene: Scene,
    story_context: str,
    *,
    ref_paths: list[tuple[str, Path]],
    image_model: str,
    color_mode: str,
    aspect_ratio: str,
    art_style: str,
    app_mode: str = "storybook",
    text_density: str = "dialog_fx",
    continuity_image_path: Path | None = None,
    section_anchor_image_path: Path | None = None,
    section_design: dict[str, Any] | None = None,
) -> tuple[bytes, str | None]:
    """
    Sinh một ảnh cho cảnh/trang: storybook = một minh họa; manga = một trang nhiều panel (full page).
    ref_paths: [(nhãn, đường dẫn ảnh), ...].
    """
    if app_mode == "manga":
        system_block = _manga_fullpage_instruction(
            color_mode=color_mode,
            aspect_ratio_key=aspect_ratio,
            panels=scene.panel_count,
            art_style_key=art_style,
        )
        text_a = system_block + "\n\nManga page brief (rectangular grid + story beats, silent page):\n" + scene.page_content
        ref_caption = (
            "Identity reference for «{label}» ({fname}): match face, hair, body type; "
            "poses/outfits per grid cell follow the page brief. No text on the final page."
        )
        bind = (
            "\n\nREMINDER: Rectangular L→R, T→D grid only; SILENT PAGE (ZERO text). "
            "No speech bubbles, no narration boxes, no SFX lettering, no captions/subtitles, no logos/watermarks/signatures, "
            "and no readable words on signs/screens. Likeness from references; action/wardrobe from brief."
        )
    else:
        system_block = _storybook_image_instruction(
            color_mode=color_mode,
            aspect_ratio_key=aspect_ratio,
            art_style_key=art_style,
        )
        text_a = system_block + "\n\nScene illustration brief:\n" + scene.page_content
        ref_caption = (
            "Identity reference for character named «{label}» (file {fname}): "
            "use for facial likeness and hair/body type; outfit in THIS scene is defined in the text brief, "
            "not necessarily in this image."
        )
        bind = (
            "\n\nREMINDER: Wardrobe, props, pose, and expression for THIS frame come ONLY from the illustration brief "
            "and segment above—reference portraits below are for face/hair/body identity, not a fixed costume."
        )

    anatomy_guard = _ANATOMY_AND_TEXT_GUARD

    verbatim = (scene.story_text or "").strip()
    verbatim_block = (
        "\n\nVOICEOVER / NARRATION TEXT (audio-only, will be added later as TTS — DO NOT render any of these "
        "words, letters, or punctuation as visible text on the image; do NOT draw subtitles, captions, speech "
        "bubbles, callouts, or signs containing this text). This is provided ONLY so you understand the emotional "
        "and contextual meaning of the moment, NOT as text to draw:\n"
        f"<<<\n{verbatim}\n>>>\n"
        if verbatim
        else ""
    )
    plane = (getattr(scene, "narrative_plane", None) or "present").strip().lower()
    plane_note = ""
    if app_mode == "review":
        if plane == "flashback":
            plane_note = (
                "\n\nNARRATIVE PLANE: FLASHBACK / MEMORY — Render as recalled time (not crisp present-day reality): "
                "softer contrast or memory color grade, optional gentle vignette/edge softness, slight grain OK. "
                "Setting and wardrobe follow the recalled moment in the brief, not the present thread."
            )
        else:
            plane_note = (
                "\n\nNARRATIVE PLANE: PRESENT REALITY — Sharp, solid environment; match the brief's locked location, "
                "time/lighting, and character wardrobe for this ongoing real-time thread. No dreamy flashback filter."
            )

    outline_note = ""
    if app_mode == "review" and getattr(scene, "outline_section_number", None) is not None:
        osumm = (getattr(scene, "outline_section_summary", None) or "").strip()
        outline_note = (
            f"\n\nOUTLINE SECTION #{scene.outline_section_number}"
            + (f" ({osumm})" if osumm else "")
            + ": within the same section, all PRESENT-reality beats share the same baseline setting + wardrobe "
            "from the brief; only FLASHBACK beats may use a different visual register."
        )

    bible_note = ""
    if app_mode == "review" and section_design and (getattr(scene, "narrative_plane", None) or "present").strip().lower() == "present":
        bible_text = _format_wardrobe_bible_for_beat(
            section_design,
            list(getattr(scene, "suggested_references", []) or []),
        )
        if bible_text:
            bible_note = (
                "\n\nSECTION WARDROBE BIBLE (HIGHEST PRIORITY for this present-reality beat — render outfit, hair, accessories EXACTLY as specified; "
                "use this lighting + palette + location). If the brief above contradicts the bible on outfit/hair, **the BIBLE wins**:\n"
                f"{bible_text}"
            )

    text_b = (
        f"{verbatim_block}"
        f"{plane_note}"
        f"{outline_note}"
        f"{bible_note}"
        f"\n\nStory summary (this segment): {scene.story_segment}\n"
        f"Narrative window (±2 beats around this beat — context only for tone/continuity; "
        f"the line marked with → is THIS beat. DO NOT render any of these words as text on the image):\n"
        f"{story_context[:2000]}"
    )
    parts: list[types.Part] = [
        types.Part.from_text(text=text_a),
        types.Part.from_text(text=text_b),
        types.Part.from_text(text=anatomy_guard),
        types.Part.from_text(text=bind),
    ]

    section_anchor_image_path = (
        section_anchor_image_path
        if section_anchor_image_path is not None and section_anchor_image_path.is_file()
        else None
    )
    use_continuity = (
        continuity_image_path is not None
        and continuity_image_path.is_file()
        and app_mode != "manga"
    )
    use_anchor = (
        section_anchor_image_path is not None
        and app_mode != "manga"
        and (
            continuity_image_path is None
            or section_anchor_image_path.resolve() != continuity_image_path.resolve()
        )
    )

    if use_anchor or use_continuity:
        parts.append(
            types.Part.from_text(
                text=(
                    "\n\nIN-SECTION CONTINUITY LOCK (HIGHEST PRIORITY for present-reality beats):\n"
                    "- Within the same outline section, every 'present' beat MUST share the same setting, location, time-of-day / lighting, "
                    "palette, and per-character wardrobe / hair / accessories — IDENTICAL pixels-of-truth where possible.\n"
                    "- A flashback beat inserted between two present beats does NOT reset this baseline: the present beat AFTER the flashback "
                    "must restore the SAME continuous scene (same room, same minute, same outfits) as the present beat BEFORE the flashback.\n"
                    "- DO NOT change clothing, hairstyle, carried items, interior color/style, time of day, or weather — UNLESS this beat's brief explicitly says so.\n"
                    "- Wardrobe + hair come from the Wardrobe Bible (text) and the per-character wardrobe-sheet refs below; "
                    "face + identity come from the portrait / wardrobe-sheet refs.\n"
                    "\nCAMERA / FRAMING FREEDOM (encouraged to vary between beats):\n"
                    "- You ARE ALLOWED and ENCOURAGED to change: shot size (extreme close-up / close-up / medium close-up / medium / "
                    "medium wide / wide / extreme wide), camera angle (eye-level / low-angle / high-angle / over-the-shoulder / "
                    "slight Dutch tilt / top-down / POV), apparent focal length (35mm wide vs 85mm compressed), camera distance, "
                    "the characters' position and eyeline within the frame, staging (left-third / center / right-third), "
                    "depth-of-field (foreground/background blur), pose and expression.\n"
                    "- Choose the shot to fit this beat's **drama + emotion + action**: intimate dialogue → close-up / medium close-up / OTS; "
                    "physical action → medium-wide or wide; isolation / vast space → wide; introspection / distant gaze → expressive close-up; "
                    "discovery → POV or OTS to the object.\n"
                    "- AVOID repeating the previous beat's exact angle, framing, layout, and character placement — change AT LEAST ONE of: "
                    "shot size, camera angle, or staging position. Each image is a NEW cut of the story, not a copy-paste of the previous one.\n"
                    "- What MUST stay the same: setting / location / props / lighting tone / palette / wardrobe / hair / character identity. "
                    "What SHOULD change per the beat brief: shot, angle, distance, pose, expression, eyeline, framing, composition."
                )
            )
        )

    if use_anchor:
        parts.append(
            types.Part.from_text(
                text=(
                    "\n\nSECTION ENVIRONMENT KEYFRAME (establishing still — SETTING ONLY, NO characters) — "
                    "this is the MANDATORY anchor for location, time-of-day, lighting, palette, and props for the ENTIRE section. "
                    "Copy exactly: the room/location structure, props, light direction and warmth, ambient palette. "
                    "**DO NOT copy the camera angle / shot size / framing** from this image — the keyframe is a wide establishing shot; "
                    "this beat can (and should) use a DIFFERENT angle / distance to tell the story. "
                    "This image contains NO characters — do not try to 'read' wardrobe from it; wardrobe comes from the Wardrobe Bible (text) "
                    "and the per-character wardrobe-sheet refs below."
                )
            )
        )
        parts.append(_image_part_from_path(section_anchor_image_path))

    if use_continuity:
        parts.append(
            types.Part.from_text(
                text=(
                    "\n\nMOST RECENT PRESENT BEAT (same outline section) — this is the previous PRESENT beat in the present-thread "
                    "(any flashback beats between it and this beat are skipped on purpose: the present scene is continuous across them). "
                    "Because it shares the same setting and cast, use it as a REFERENCE for:\n"
                    "  • Setting / lighting / palette: must match exactly (alongside the environment anchor).\n"
                    "  • Wardrobe + hair + grooming + accessories: copy precisely from this image (do NOT drift).\n"
                    "  • Character identity: same faces, same builds, same ages — do NOT replace any person.\n"
                    "**DO NOT copy the layout / shot size / camera angle / staging** of this image. This is a NEW CUT: "
                    "change the shot size, camera angle, distance, character placement, eyeline, or pose so this beat has its own "
                    "cinematic rhythm matching the new brief."
                )
            )
        )
        parts.append(_image_part_from_path(continuity_image_path))

    if len(ref_paths) >= 2 and app_mode != "manga":
        cast_names = ", ".join(f"«{lbl}»" for lbl, _ in ref_paths)
        parts.append(
            types.Part.from_text(
                text=(
                    "\n\nCO-PRESENT CAST CONTEXT (READ THIS BEFORE DRAWING — MANDATORY):\n"
                    f"The following characters are ALL physically present in this scene at the same place/time: {cast_names}. "
                    "They are part of the same conversation / situation — NONE of them has left.\n"
                    "How to use these references when STAGING:\n"
                    "  • If the brief asks for wide / medium / medium-wide → draw ALL of the above characters with the correct identity + wardrobe from the refs.\n"
                    "  • If the brief asks for close-up / extreme close-up / OTS on one character → keep that character as the main subject, "
                    "BUT the other co-present characters MUST still be readable inside the frame in AT LEAST one of these forms:\n"
                    "      - over-the-shoulder silhouette at the edge of frame,\n"
                    "      - background figures (soft, depth-of-field) standing/sitting in their correct positions,\n"
                    "      - a foreground silhouette (shoulder/arm/back) framing the shot,\n"
                    "      - or a distinctive piece of their wardrobe / accessory clipping into the frame's edge.\n"
                    "  • NEVER render an empty shared space (room, hall, courtyard, plaza, street…) with only 1 character "
                    "when the CO-PRESENT CAST list has 2+ people — that breaks the story's logic (as if the others had vanished). "
                    "There MUST be a visual signal of co-presence.\n"
                    "  • If the brief contains an 'OFF-FRAME COMPANIONS' line listing names → those listed names may genuinely sit outside the frame, "
                    "but every other character NOT listed in OFF-FRAME COMPANIONS MUST appear inside the frame using one of the methods above.\n"
                    "  • Each character's identity / face / build comes from their portrait reference (attached below); "
                    "wardrobe comes from the Wardrobe Bible + brief. Do NOT swap characters, do NOT merge two characters into one."
                )
            )
        )

    for label, p in ref_paths:
        parts.append(
            types.Part.from_text(
                text="\n\n"
                + ref_caption.format(label=label, fname=p.name)
            )
        )
        parts.append(_image_part_from_path(p))

    response = _call_image_model_with_retry(
        lambda: client.models.generate_content(
            model=image_model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.6,
            ),
        ),
        label=f"scene_image(scene={getattr(scene, 'page_number', '?')})",
    )

    img_bytes, img_mime = _extract_inline_image_from_response(response)
    if img_bytes is not None:
        return img_bytes, img_mime

    text_fallback = response.text or ""
    raise RuntimeError(
        "Model không trả về ảnh. Thử đổi CREATOR_IMAGE_MODEL_FLASH hoặc kiểm tra quota / safety. "
        f"Lời thoại: {text_fallback[:400]} | {_response_debug_hint(response)}"
    )


def _story_excerpt_for_llm(story: str, max_chars: int = 120000) -> str:
    """Giữ đầu + đuôi truyện nếu quá dài để refine không vượt context."""
    if len(story) <= max_chars:
        return story
    half = (max_chars // 2) - 80
    return (
        story[:half]
        + "\n\n[... đoạn giữa truyện đã lược bớt do độ dài ...]\n\n"
        + story[-half:]
    )


def extract_characters_from_story(
    client: genai.Client,
    story: str,
    *,
    planner_model: str,
    max_characters: int,
) -> list[dict[str, str]]:
    """
    Pass 1: phân tích truyện — nhân vật cần thiết kế nhất quán + mô tả neo vào văn bản.
    """
    prompt = f"""
You are a casting and continuity supervisor for illustrated fiction. Read the ENTIRE story carefully.

Story:
{story}

Return ONLY a JSON array (no markdown, no commentary). Each object MUST use these keys:

- "canonicalName": The clearest name or title readers use for this person (match story spelling).
- "aliases": Array of other strings that refer to the SAME person in this story (nicknames, titles, shortened names). Empty array [] if none.
- "evidenceQuote": ONE short verbatim quote (20-90 characters if possible) copied EXACTLY from the story above that proves this character appears—must be findable as a substring of the story text. If the snippet contains dialogue with double quotes ("), REPLACE those inner double quotes with single quotes (') OR with curly quotes (“ ”) BEFORE putting it into the JSON value, so that the surrounding JSON string is not broken.
- "visualDescription": 4-8 sentences for IMAGE CONSISTENCY. Include ONLY:
  • Approximate age or age band AS IMPLIED by the text (if unclear, say "age not specified").
  • Build/height relative cues IF present.
  • Face: shape, eyes, brows, nose/mouth if described or strongly implied; otherwise neutral professional guess that does NOT contradict the text.
  • Hair: color, length, style AS IN TEXT or "not specified".
  • Skin tone ONLY if supported by text or safely generic without stereotype.
  • Any scars, glasses, facial hair, voice-related visuals IF mentioned.
  • Typical posture or energy IF implied.
  • End with: "Scene-specific clothing will vary; default neutral/simple clothing for reference portrait only."
- "importance": "main" | "supporting" | "minor".

STRICT RULES:
1. Do NOT invent contradicting biography; when the story is silent on a trait, say it is not specified rather than inventing exotic traits.
2. EXCLUDE: pure locations, organizations without a single recurring visual persona, objects, animals unless they are illustrated as characters, nameless crowd extras.
3. INCLUDE: every named person who appears on-page more than a passing mention OR who drives a plot beat visually.
4. Merge the same person under ONE canonicalName (do not list duplicates).
5. Maximum {max_characters} characters total—prioritize main, then supporting; omit the least important minors if over limit.
6. If no suitable characters, return [].

Language of strings: match the story language for names and evidenceQuote.

JSON SYNTAX: Valid JSON only.
- Do not put raw line breaks inside string values.
- Do NOT put raw unescaped double quotes (") inside string values. If a quote-mark appears inside a value, either escape it as \\" OR rewrite it as a single quote ' or curly quote “ ”.
- Double-check evidenceQuote: it MUST be a single, well-formed JSON string with NO inner unescaped " characters.
"""

    response = client.models.generate_content(
        model=planner_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    raw = _strip_json_fence(response.text or "")
    data = _json_loads_llm(raw)
    if not isinstance(data, list):
        raise ValueError("Trích nhân vật phải trả về một mảng JSON.")
    return data[:max_characters]


def refine_character_list(
    client: genai.Client,
    story: str,
    draft: list[dict],
    *,
    planner_model: str,
    max_characters: int,
) -> list[dict]:
    """
    Pass 2: đối chiếu lại truyện — gộp trùng, bỏ nhầm, sửa mô tả không bám văn.
    """
    if not draft:
        return []

    story_body = _story_excerpt_for_llm(story)
    draft_json = json.dumps(draft, ensure_ascii=False, indent=2)

    prompt = f"""
You are a strict continuity editor. You MUST compare the DRAFT character list to the FULL STORY and output a corrected list.

FULL STORY:
\"\"\"
{story_body}
\"\"\"

DRAFT LIST (JSON):
{draft_json}

Tasks:
1. DELETE entries that are not animate speaking/thinking characters (places, props-only names, abstract titles with no embodied scenes, duplicate merges already missing).
2. MERGE duplicates: same individual under one canonicalName; union all aliases.
3. ADD any major named on-page character who is clearly in the story but MISSING from the draft (with correct evidenceQuote and careful visualDescription).
4. VERIFY each evidenceQuote appears verbatim (as substring) in the story; if not, replace with a correct short verbatim quote from the story or remove that character if they do not appear.
5. REWRITE visualDescription so every trait is either directly supported by the story OR explicitly marked "not specified in text". Remove fanciful inventions.
6. Cap the final list at {max_characters} entries: keep main first, then supporting, then minor.

Return ONLY a JSON array of objects with keys:
canonicalName, aliases (array), evidenceQuote, visualDescription, importance.

Do not add markdown or explanations. String values must follow standard JSON escaping (no raw control characters inside strings).
"""

    response = client.models.generate_content(
        model=planner_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.15,
        ),
    )

    raw = _strip_json_fence(response.text or "")
    data = _json_loads_llm(raw)
    if not isinstance(data, list):
        raise ValueError("Refine nhân vật phải trả về một mảng JSON.")
    return data[:max_characters]


def _row_character_name(row: dict) -> str:
    return str(row.get("canonicalName") or row.get("name") or "").strip()


def _dedupe_rows_by_canonical(rows: list[dict]) -> list[dict]:
    """An toàn phía client nếu model vẫn trả trùng canonicalName (so theo casefold)."""
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        name = _row_character_name(row)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _character_row_fingerprints(row: dict) -> set[str]:
    """
    Bộ dấu vân tay của một character row, gồm fingerprint của canonicalName + mọi alias
    (loại bỏ generic aliases như "bà", "ông ấy", "nàng"…). Dùng để dedup giữa các row mà
    LLM có thể đã đặt canonicalName khác nhau (vd. "Cố phu nhân" vs "Phu nhân") nhưng cùng người.
    """
    out: set[str] = set()
    name = _row_character_name(row)
    if name:
        fp = _alias_fingerprint(name)
        if fp and len(fp) >= 3:
            out.add(fp)
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        for a in aliases:
            if not isinstance(a, str):
                continue
            s = a.strip()
            if not s or len(s) < 3:
                continue
            if s.lower() in _GENERIC_ALIAS_BLACKLIST:
                continue
            fp = _alias_fingerprint(s)
            if fp and len(fp) >= 3:
                out.add(fp)
    return out


def _merge_character_rows(rows: list[dict]) -> list[dict]:
    """
    Dedup + merge danh sách character rows một cách thông minh:
    - Match qua fingerprint của canonicalName + aliases (bỏ generic).
    - Khi trùng: giữ row đầu (đã có), gộp aliases mới vào (giữ thứ tự, không trùng).
    - Bảo toàn thứ tự xuất hiện. Phù hợp khi gộp existing_rows + new_rows hoặc dedup output LLM.
    """
    canonical_map: list[set[str]] = []  # parallel to merged: fingerprints owned by group
    merged: list[dict] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _row_character_name(row):
            continue
        fps = _character_row_fingerprints(row)
        if not fps:
            merged.append(dict(row))
            canonical_map.append(set())
            continue

        match_idx = -1
        for i, owned in enumerate(canonical_map):
            if owned and (owned & fps):
                match_idx = i
                break

        if match_idx < 0:
            merged.append(dict(row))
            canonical_map.append(set(fps))
            continue

        target = merged[match_idx]
        owned = canonical_map[match_idx]
        # Gộp aliases mới (giữ thứ tự, dedup theo casefold), thêm canonicalName của row mới
        # vào aliases nếu khác canonical hiện tại.
        existing_aliases = target.get("aliases")
        if not isinstance(existing_aliases, list):
            existing_aliases = []
        seen_lower = {str(a).strip().lower() for a in existing_aliases if isinstance(a, str)}
        seen_lower.add(str(target.get("canonicalName") or "").strip().lower())

        def _maybe_add(s: str) -> None:
            ss = s.strip()
            if not ss:
                return
            kk = ss.lower()
            if kk in seen_lower:
                return
            seen_lower.add(kk)
            existing_aliases.append(ss)

        new_canon = _row_character_name(row)
        if new_canon:
            _maybe_add(new_canon)
        new_aliases = row.get("aliases")
        if isinstance(new_aliases, list):
            for a in new_aliases:
                if isinstance(a, str):
                    _maybe_add(a)

        target["aliases"] = existing_aliases
        owned |= fps
        canonical_map[match_idx] = owned

    return merged


def _safe_filename(name: str) -> str:
    s = re.sub(r"[^\w\u00C0-\u024F\-]+", "_", name.strip())[:80]
    return s or "character"


def _portrait_instruction(
    *,
    color_mode: str,
    aspect_ratio_key: str,
    art_style_key: str,
) -> str:
    aspect_ratio_map = {
        "square": "1:1 (Square)",
        "portrait": "2:3 (Portrait)",
    }
    art_style_map = {
        "watercolor": "Watercolor painting with soft, blended colors and organic textures.",
        "oil_painting": "Rich oil painting with visible brushstrokes and deep colors.",
        "digital_illustration": "Modern digital illustration with clean lines and vibrant colors.",
        "anime": "Stylized anime illustration with expressive characters.",
        "storybook_classic": "Classic children's book illustration, warm and inviting.",
        "realistic": "Photorealistic digital art with high detail and accurate lighting.",
        "ghibli": (
            "Studio Ghibli inspired hand-drawn anime portrait: soft 2D cel shading, warm painterly lighting, "
            "muted harmonious pastel palette, rounded gentle features, simple clean lineart, "
            "in the spirit of Hayao Miyazaki / Studio Ghibli character art."
        ),
    }
    ar = aspect_ratio_map.get(aspect_ratio_key, "1:1 (Square)")
    style = art_style_map.get(art_style_key, art_style_map["storybook_classic"])
    color = "Full color." if color_mode == "color" else "Black and white with grayscale shading."
    return f"""Task: Generate ONE character reference illustration for an illustrated narrative project (any genre / audience).

VISUAL STYLE:
- Color: {color}
- Art Style: {style}
- Aspect Ratio: {ar}

REQUIREMENTS:
- Single character only, neutral or very simple background (no narrative scene, no other people).
- Camera: waist-up or three-quarter preferred; face MUST occupy a clear, readable portion of the frame (sharp eyes, nose, mouth).
- Lighting: soft even studio-like light so facial structure is unambiguous for reuse across scenes.
- Match EVERY identity trait in the written description below; do not substitute a different age, gender presentation, or ethnicity than described.
- Use simple or neutral everyday clothing—this sheet locks FACE/HAIR/BODY identity only; story illustrations will use context-appropriate costumes per scene.
- Neutral, calm facial expression unless the written description explicitly asks for another mood.
- NO text, captions, labels, or typography on the image.
- Output ONE high-quality image only.
- If asked to use tools, ignore and output the image directly."""


def generate_character_portrait(
    client: genai.Client,
    *,
    name: str,
    visual_description: str,
    image_model: str,
    color_mode: str,
    art_style: str,
    evidence_quote: str = "",
    aliases_line: str = "",
) -> tuple[bytes, str | None]:
    """Một ảnh reference cho một nhân vật (không cần ảnh đầu vào)."""
    instr = _portrait_instruction(
        color_mode=color_mode,
        aspect_ratio_key="square",
        art_style_key=art_style,
    )
    extra = ""
    if aliases_line:
        extra += f"\nAliases in story: {aliases_line}"
    if evidence_quote.strip():
        extra += (
            f'\nStory snippet this character MUST remain consistent with '
            f'(identity/presence—not costume): "{evidence_quote.strip()}"'
        )
    body = (
        f"{instr}\n\nPRIMARY NAME (label): {name}\n"
        f"{extra}\n\nVISUAL IDENTITY — render exactly:\n{visual_description}"
    )
    parts: list[types.Part] = [types.Part.from_text(text=body)]

    last_err: RuntimeError | None = None
    for attempt, temp in enumerate((0.42, 0.58), start=1):
        response = _call_image_model_with_retry(
            lambda temp=temp: client.models.generate_content(
                model=image_model,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    temperature=temp,
                ),
            ),
            label=f"character_portrait({name!r}, attempt={attempt})",
        )

        img_bytes, img_mime = _extract_inline_image_from_response(response)
        if img_bytes is not None:
            return img_bytes, img_mime

        msg = (response.text or "")[:400]
        dbg = _response_debug_hint(response)
        last_err = RuntimeError(
            f"Không sinh được ảnh portrait (lần {attempt}). Text: {msg} | {dbg}"
        )

    raise last_err or RuntimeError("Không sinh được ảnh portrait nhân vật.")


def auto_build_character_refs(
    client: genai.Client,
    story: str,
    *,
    out_dir: Path,
    planner_model: str,
    image_model: str,
    color_mode: str,
    art_style: str,
    max_characters: int,
    refine_characters: bool = True,
    generate_portraits: bool = True,
    resume: bool = False,
    regenerate: bool = False,
) -> tuple[dict[str, Path], list[dict]]:
    """
    Trích nhân vật + (tuỳ chọn) sinh ảnh reference trong out_dir/auto_refs/.

    Caching: nếu out_dir/auto_refs/characters_extracted.json đã tồn tại và có ≥1 nhân vật hợp lệ,
    BỎ QUA hai lần gọi LLM (pass 1 extract + pass 2 refine) và dùng nguyên danh sách cũ. Đặt
    `regenerate=True` (hoặc CLI `--regenerate-characters`) để ép trích lại.

    Trả về (char_paths, danh sách nhân vật cuối cùng).
    """
    ref_dir = out_dir / "auto_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Nếu đã có characters_extracted.json, giữ nguyên danh sách cũ và chỉ bổ sung nhân vật mới.
    # Đồng thời tự dedup file cũ qua aliases (LLM lần này có thể trả canonical khác lần trước,
    # vd. "Cố phu nhân" và "Phu nhân" cùng có alias "Minh Nguyệt quận chúa" → gộp lại 1 entry).
    existing_manifest = ref_dir / "characters_extracted.json"
    existing_rows: list[dict] = []
    if existing_manifest.is_file():
        try:
            raw = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing_rows = [r for r in raw if isinstance(r, dict)]
        except Exception:
            existing_rows = []
        if existing_rows:
            n_before = len(existing_rows)
            existing_rows = _merge_character_rows(existing_rows)
            if len(existing_rows) < n_before:
                print(
                    f"characters_extracted.json: gộp {n_before - len(existing_rows)} entry trùng "
                    f"qua aliases ({n_before} → {len(existing_rows)} nhân vật).",
                    file=sys.stderr,
                )

    # CACHE HIT: characters_extracted.json đã có ≥1 nhân vật hợp lệ → bỏ hẳn 2 lần gọi LLM
    # (extract pass 1 + refine pass 2). Tiết kiệm thời gian/chi phí khi rerun pipeline.
    # Ép trích lại bằng `regenerate=True` (CLI: --regenerate-characters).
    def _row_is_valid(r: dict) -> bool:
        name = _row_character_name(r) or ""
        return bool(name.strip())

    valid_existing = [r for r in existing_rows if _row_is_valid(r)]
    if valid_existing and not regenerate:
        print(
            f"Đã có {existing_manifest} với {len(valid_existing)} nhân vật → "
            "BỎ QUA pass 1 (extract) + pass 2 (refine). "
            "Dùng --regenerate-characters để ép trích lại.",
            file=sys.stderr,
        )
        # Ghi lại file (đã dedup) cho nhất quán.
        existing_manifest.write_text(
            json.dumps(valid_existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows = valid_existing
    else:
        if regenerate and existing_rows:
            print(
                "--regenerate-characters: ép trích lại từ truyện dù characters_extracted.json đã có.",
                file=sys.stderr,
            )
        rows = extract_characters_from_story(
            client,
            story,
            planner_model=planner_model,
            max_characters=max_characters,
        )
        (ref_dir / "characters_draft.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if refine_characters and rows:
            try:
                rows = refine_character_list(
                    client,
                    story,
                    rows,
                    planner_model=planner_model,
                    max_characters=max_characters,
                )
                print(
                    "Đã chạy pass 2 (refine): đối chiếu truyện, gộp trùng, làm sạch mô tả.",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"Cảnh báo: refine nhân vật thất bại ({exc}); dùng kết quả pass 1.",
                    file=sys.stderr,
                )

        rows = _merge_character_rows(rows)

        if existing_rows:
            # Gộp existing + new theo fingerprint (canonicalName + aliases): nếu LLM lần này trả tên
            # khác (vd. "Cố phu nhân" lần trước, "Phu nhân" lần này) nhưng có alias chung →
            # KHÔNG tạo entry mới, mà bổ sung alias còn thiếu vào entry cũ.
            n_old = len(existing_rows)
            merged_all = _merge_character_rows(existing_rows + rows)
            n_added = len(merged_all) - n_old
            if n_added > 0:
                print(
                    f"Tự động bổ sung {n_added} nhân vật mới vào characters_extracted.json (giữ nguyên danh sách cũ).",
                    file=sys.stderr,
                )
            rows = merged_all

    char_paths: dict[str, Path] = {}
    if not generate_portraits:
        manifest = ref_dir / "characters_extracted.json"
        manifest.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Vẫn cần char_paths cho planner (bible) dù không vẽ portrait:
        #  - nếu portrait đã tồn tại từ run trước → dùng path thật
        #  - nếu chưa có → để placeholder Path ở ref_dir (file chưa tồn tại)
        # Bible/planner chỉ dùng .keys(); render sau này khi có portrait sẽ tự nhận đúng path.
        for i, row in enumerate(rows, start=1):
            name = _row_character_name(row) or f"character_{i}"
            existing_files = sorted(
                p
                for p in ref_dir.glob(f"{i:02d}_*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            if existing_files:
                char_paths[name] = existing_files[0]
            else:
                char_paths[name] = ref_dir / f"{i:02d}_{_safe_filename(name)}.png"
        if char_paths:
            print(
                f"--plan-only: nạp {len(char_paths)} nhân vật từ characters_extracted.json "
                f"(không sinh portrait; bible dùng tên ngay).",
                file=sys.stderr,
            )
        return char_paths, rows

    ext_for_mime = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }

    for i, row in enumerate(rows, start=1):
        name = _row_character_name(row) or f"character_{i}"
        desc = str(row.get("visualDescription", "")).strip()
        if not desc:
            continue
        aliases_raw = row.get("aliases")
        if isinstance(aliases_raw, list):
            aliases_line = ", ".join(
                str(a).strip() for a in aliases_raw if str(a).strip()
            )
        else:
            aliases_line = ""
        evidence = str(row.get("evidenceQuote", "") or "").strip()

        if resume:
            existing = sorted(
                (
                    p
                    for p in ref_dir.glob(f"{i:02d}_*")
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                ),
                key=lambda p: p.name,
            )
            if existing:
                path = existing[0]
                char_paths[name] = path
                print(
                    f"Resume: giữ ảnh reference {name!r} → {path}",
                    file=sys.stderr,
                )
                continue

        print(f"Đang sinh ảnh reference cho nhân vật: {name!r}...", file=sys.stderr)
        try:
            data, mime = generate_character_portrait(
                client,
                name=name,
                visual_description=desc,
                image_model=image_model,
                color_mode=color_mode,
                art_style=art_style,
                evidence_quote=evidence,
                aliases_line=aliases_line,
            )
        except RuntimeError as exc:
            msg = str(exc)
            # Một số danh xưng/nhân vật (vd. "đứa bé") dễ bị safety chặn dù prompt lành.
            # Khi resume/render lại scene, tốt hơn là bỏ qua ref này thay vì crash toàn pipeline.
            if "IMAGE_SAFETY" in msg or "FinishReason.IMAGE_SAFETY" in msg:
                print(
                    f"Cảnh báo: portrait bị safety chặn cho {name!r}; bỏ qua ref này và tiếp tục. ({msg[:200]})",
                    file=sys.stderr,
                )
                continue
            raise
        ext = ext_for_mime.get(mime or "", ".png")
        fname = f"{i:02d}_{_safe_filename(name)}{ext}"
        path = ref_dir / fname
        path.write_bytes(data)
        char_paths[name] = path
        print(f"  -> {path}", file=sys.stderr)

    manifest = ref_dir / "characters_extracted.json"
    manifest.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return char_paths, rows


def _parse_char_args(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Cần định dạng Tên=đường/dẫn/ảnh.png, nhận được: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        p = Path(path.strip()).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Không tìm thấy file nhân vật: {p}")
        out[name] = p
    return out


def _parse_optional_backfill_page_numbers(spec: str | None) -> frozenset[int] | None:
    """
  Parse pageNumber filter cho backfill motionPrompt.
  Ví dụ: "1,3,5-8,12" → {1,3,5,6,7,8,12}. None / rỗng → không lọc (toàn bộ beat).
  """
    if spec is None or not str(spec).strip():
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError as exc:
                raise ValueError(f"đoạn range không hợp lệ: {token!r}") from exc
            if start <= 0 or end <= 0:
                raise ValueError(f"pageNumber phải > 0: {token!r}")
            if start > end:
                raise ValueError(f"range phải tăng dần: {token!r}")
            out.update(range(start, end + 1))
            continue
        try:
            pn = int(token)
        except ValueError as exc:
            raise ValueError(f"pageNumber không hợp lệ: {token!r}") from exc
        if pn <= 0:
            raise ValueError(f"pageNumber phải > 0: {token!r}")
        out.add(pn)
    if not out:
        raise ValueError("danh sách pageNumber rỗng")
    return frozenset(out)


def _motion_backfill_neighbor_radius_from_args(args: Any) -> int:
    """±N beat lân cận đưa vào ngữ cảnh backfill motionPrompt (mặc định 5, tối đa 20)."""
    raw = getattr(args, "backfill_motion_neighbor_beats", _MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS
    return max(0, min(n, _MOTION_BACKFILL_MAX_NEIGHBOR_RADIUS))


def _gather_section_detection_texts(sec_scenes: list[Scene]) -> list[str]:
    """
    Gộp văn bản để dò tên nhân vật: ưu tiên beat present, kèm cả storyText (TTS / prose).
    Nếu section chỉ có flashback thì dùng mọi beat trong section.
    """
    texts: list[str] = []
    present = [
        s
        for s in sec_scenes
        if (getattr(s, "narrative_plane", None) or "present").strip().lower() == "present"
    ]
    src = present if present else sec_scenes
    for s in src:
        if s.page_content:
            texts.append(s.page_content)
        if s.story_text:
            texts.append(s.story_text)
    return texts


def _bible_normalize_targets(
    required_labels: list[str],
    char_paths: dict[str, Path],
) -> list[str]:
    """Tập key chuẩn để rename alias: subset section hoặc toàn cast nếu subset rỗng."""
    if required_labels:
        return list(dict.fromkeys(required_labels))
    return sorted(char_paths.keys())


def _required_character_labels_for_section(
    sec_scenes: list[Scene],
    char_paths: dict[str, Path],
) -> list[str]:
    """
    Nhân vật cần có trong bible **cho đúng section này** — chỉ người xuất hiện trong beat/ prose section.
    Không ép mọi section phải có đủ toàn bộ cast.
    """
    texts = _gather_section_detection_texts(sec_scenes)
    return _detect_section_characters(texts, char_paths)


def _wardrobe_bible_missing_details(obj: dict[str, Any], required_names: list[str]) -> list[str]:
    """
    Trả về danh sách lỗi validation.
    - Nếu required_names không rỗng: bắt buộc đủ các tên đó (đã dò được trong section).
    - Nếu rỗng: chỉ kiểm tra các entry planner đã thêm — không ép cast đầy đủ.
    """
    issues: list[str] = []
    chars = obj.get("characters") if isinstance(obj.get("characters"), dict) else {}
    if required_names:
        for name in required_names:
            if name not in chars:
                issues.append(f"thiếu nhân vật «{name}» trong characters")
                continue
            info = chars.get(name)
            if not isinstance(info, dict):
                issues.append(f"«{name}» không phải object")
                continue
            if not str(info.get("outfit", "")).strip():
                issues.append(f"«{name}».outfit trống hoặc quá mơ hồ")
            if not str(info.get("hair", "")).strip():
                issues.append(f"«{name}».hair trống")
        return issues
    for name, info in chars.items():
        if not isinstance(info, dict):
            issues.append(f"«{name}» không phải object")
            continue
        if not str(info.get("outfit", "")).strip():
            issues.append(f"«{name}».outfit trống hoặc quá mơ hồ")
        if not str(info.get("hair", "")).strip():
            issues.append(f"«{name}».hair trống")
    return issues


_GENERIC_ROLE_TOKENS = {
    "dr", "doctor", "mr", "mrs", "ms", "miss",
    "the", "a", "an",
    "husband", "wife", "spouse",
    "man", "woman", "girl", "boy", "child", "kid",
    "young", "old", "older", "younger",
    "pregnant", "patient", "doctor's",
    "professional", "medical",
    "captain", "pilot",
}


def _bible_key_tokens(s: str) -> set[str]:
    """Lower-case ASCII tokens (đã bỏ dấu) — bỏ qua các chữ generic."""
    if not s:
        return set()
    norm = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    toks = {t.lower() for t in re.findall(r"[a-zA-Z]+", norm) if len(t) >= 2}
    return toks - _GENERIC_ROLE_TOKENS


def _normalize_bible_character_keys(
    obj: dict[str, Any],
    canonical_targets: list[str],
) -> list[tuple[str, str]]:
    """
    Đổi key trong obj["characters"] từ alias sang tên chuẩn trong canonical_targets (subset section hoặc toàn cast).
    Trả về list (alias_cũ, tên_mới) đã được map.
    """
    chars = obj.get("characters")
    if not isinstance(chars, dict) or not canonical_targets:
        return []

    req_tokens = {name: _bible_key_tokens(name) for name in canonical_targets}

    new_chars: dict[str, Any] = {}
    used: set[str] = set()
    leftover: list[tuple[str, Any]] = []
    mapping: list[tuple[str, str]] = []

    for k, v in chars.items():
        if k in canonical_targets and k not in used:
            new_chars[k] = v
            used.add(k)
            continue
        k_tok = _bible_key_tokens(k)
        if k_tok:
            scored: list[tuple[float, str]] = []
            for n in canonical_targets:
                if n in used:
                    continue
                inter = len(req_tokens[n] & k_tok)
                if not inter:
                    continue
                jaccard = inter / max(1, len(req_tokens[n] | k_tok))
                scored.append((jaccard, n))
            if scored:
                scored.sort(key=lambda x: (-x[0], x[1]))
                top_score = scored[0][0]
                top_names = [n for sc, n in scored if sc == top_score]
                if top_score >= 0.4 and len(top_names) == 1:
                    new_chars[top_names[0]] = v
                    used.add(top_names[0])
                    mapping.append((k, top_names[0]))
                    continue
        leftover.append((k, v))

    remaining_required = [n for n in canonical_targets if n not in used]
    if leftover and remaining_required and len(leftover) == len(remaining_required):
        for (k_old, v), name in zip(leftover, remaining_required):
            new_chars[name] = v
            used.add(name)
            mapping.append((k_old, name))
    else:
        for k_old, v in leftover:
            new_chars[k_old] = v

    obj["characters"] = new_chars
    return mapping


def _repair_section_wardrobe_bible(
    client: genai.Client,
    planner_model: str,
    sec_num: int,
    issues: list[str],
    partial: dict[str, Any],
    required_names: list[str],
) -> dict[str, Any] | None:
    """Một lần gọi nữa để sửa JSON thiếu trang phục/tóc."""
    if required_names:
        naming_rules = """
REQUIRED: the "characters" object MUST contain an entry for EVERY name in this list, with **non-empty, specific** "outfit" and "hair" for each:
""" + json.dumps(required_names, ensure_ascii=False) + """

**Strict naming**: each JSON key under "characters" MUST be **byte-identical** to one of the names above (preserve exact spelling, diacritics, and script as listed — Vietnamese, English, Chinese, Japanese, Korean, mixed, etc.). DO NOT translate them, DO NOT use generic English role labels such as "Dr. X", "The Husband", "The Pregnant Woman", "Young Woman", "Doctor", "Medical Professional", "Pilot", etc. If the previous JSON used such aliases, RENAME each key to the matching canonical name above. You may keep extra side-characters under separate keys, but every name in the MANDATORY list MUST appear with the EXACT spelling.
"""
    else:
        naming_rules = """
This section uses **subset mode**: only characters who actually appear in this section should appear under "characters".
For EVERY entry you keep under "characters", you MUST provide **non-empty, specific** "outfit" and "hair".
Remove any entries for cast members who do not appear in this section.
**Strict naming**: JSON keys MUST use exact canonical spellings (preserve diacritics and script as in the cast list, regardless of language); no generic English role labels such as "Dr. X", "The Husband", "Young Woman", "Doctor", etc.
"""

    repair = f"""
Task: Fix a SECTION WARDROBE BIBLE JSON object. The first response was incomplete.

Section number: {sec_num}
Validation errors (MUST be fixed):
{json.dumps(issues, ensure_ascii=False, indent=2)}
{naming_rules}
Current (possibly incomplete) JSON to fix and return as a single complete object:
{json.dumps(partial, ensure_ascii=False, indent=2)}

Return ONLY the corrected full JSON object (same schema as before). No markdown. No commentary.
"""
    try:
        resp = client.models.generate_content(
            model=planner_model,
            contents=repair,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        raw = resp.text or ""
        fixed = _json_loads_llm(_strip_json_fence(raw))
        if not isinstance(fixed, dict):
            return None
        fixed["sectionNumber"] = int(sec_num)
        return fixed
    except (ValueError, json.JSONDecodeError, TypeError):
        return None


def _warn_section_design_wardrobe_gaps(
    designs: dict[int, dict[str, Any]],
    sec_map: dict[int, list[Scene]],
    cp: dict[str, Path],
) -> None:
    for sn in sorted(designs.keys()):
        row = designs.get(sn)
        if not isinstance(row, dict):
            continue
        req = _required_character_labels_for_section(sec_map.get(sn) or [], cp)
        miss = _wardrobe_bible_missing_details(row, req)
        if miss:
            print(
                f"  [WARN] section_designs.json section {sn:02d}: bible chưa đủ trang phục/tóc — {miss}. "
                "Sửa tay trong checkpoints/section_designs.json hoặc chạy --regenerate-section-designs.",
                file=sys.stderr,
            )


def _bible_keys_for_cast(
    design: dict[str, Any] | None,
    cast_filenames: list[str],
) -> set[str]:
    """
    Map suggestedReferences (filenames như "06_Đại_tướng_quân.png") sang bible character keys
    (canonical names như "Đại tướng quân") qua `_CHARACTER_ALIASES_MAP` + fingerprint.

    Trả về set các bible keys khớp được. Set rỗng = không khớp được (caller sẽ fallback).
    """
    if not design or not isinstance(design, dict):
        return set()
    chars = design.get("characters") or {}
    if not isinstance(chars, dict) or not chars:
        return set()
    bible_keys = list(chars.keys())
    bible_fps: dict[str, str] = {}
    for k in bible_keys:
        fp = _alias_fingerprint(k)
        if fp:
            bible_fps[k] = fp

    matched: set[str] = set()
    for fn in cast_filenames or []:
        if not fn:
            continue
        aliases: list[str] = []
        seen: set[str] = set()
        for a in _aliases_for_ref_filename(fn):
            if isinstance(a, str) and a.strip():
                key = a.strip().casefold()
                if key not in seen:
                    seen.add(key)
                    aliases.append(a.strip())
        label = _label_from_ref_filename(fn)
        if label and label.casefold() not in seen:
            aliases.append(label)

        found = False
        for a in aliases:
            for k in bible_keys:
                if k.casefold() == a.casefold():
                    matched.add(k)
                    found = True
                    break
            if found:
                break
        if found:
            continue
        for a in aliases:
            fp = _alias_fingerprint(a)
            if not fp:
                continue
            for k, kfp in bible_fps.items():
                if kfp == fp:
                    matched.add(k)
                    found = True
                    break
            if found:
                break
    return matched


def _format_wardrobe_bible_for_beat(
    design: dict[str, Any] | None,
    cast_filenames: list[str],
) -> str:
    """
    Wardrobe bible cho 1 beat: giữ nguyên setting + lọc characters theo cast của beat.

    Fallback giữ full bible khi không an toàn để lọc:
      • design rỗng / không có setting hoặc characters,
      • cast rỗng,
      • section roster ≤ 2 (lọc không tiết kiệm gì),
      • mapping suggestedReferences → bible keys quá nghèo (matched < cast - 1).
    """
    if not design or not isinstance(design, dict):
        return ""
    full_text = _format_wardrobe_bible_for_prompt(design)
    chars = design.get("characters") or {}
    if not isinstance(chars, dict) or not chars:
        return full_text
    cast_real = [fn for fn in (cast_filenames or []) if fn]
    if not cast_real:
        return full_text
    if len(chars) <= 2:
        return full_text
    matched = _bible_keys_for_cast(design, cast_real)
    if not matched:
        return full_text
    if len(matched) < max(1, len(cast_real) - 1):
        return full_text
    filtered_design = {
        "setting": design.get("setting") or {},
        "characters": {k: chars[k] for k in chars if k in matched},
    }
    return _format_wardrobe_bible_for_prompt(filtered_design)


def _format_wardrobe_bible_for_prompt(design: dict[str, Any] | None) -> str:
    """Format JSON wardrobe bible thành block plain-text gọn để chèn vào prompt render."""
    if not design or not isinstance(design, dict):
        return ""
    lines: list[str] = []
    setting = design.get("setting") or {}
    if isinstance(setting, dict) and setting:
        loc = (setting.get("location") or "").strip()
        tod = (setting.get("timeOfDay") or "").strip()
        light = (setting.get("lighting") or "").strip()
        weather = (setting.get("weather") or "").strip()
        palette = (setting.get("ambientPalette") or "").strip()
        sline = "; ".join(
            p
            for p in [
                f"location: {loc}" if loc else "",
                f"time-of-day: {tod}" if tod else "",
                f"lighting: {light}" if light else "",
                f"weather: {weather}" if weather else "",
                f"palette: {palette}" if palette else "",
            ]
            if p
        )
        if sline:
            lines.append(f"Setting: {sline}.")
    chars = design.get("characters") or {}
    if isinstance(chars, dict):
        for name, info in chars.items():
            if not isinstance(info, dict):
                continue
            outfit = (info.get("outfit") or "").strip()
            hair = (info.get("hair") or "").strip()
            footwear = (info.get("footwear") or "").strip()
            accessories = (info.get("accessories") or "").strip()
            grooming = (info.get("groomingNotes") or "").strip()
            parts = [
                f"outfit: {outfit}" if outfit else "",
                f"hair: {hair}" if hair else "",
                f"footwear: {footwear}" if footwear else "",
                f"accessories: {accessories}" if accessories else "",
                f"grooming: {grooming}" if grooming else "",
            ]
            joined = "; ".join(p for p in parts if p)
            if joined:
                lines.append(f"«{name}»: {joined}.")
    return "\n".join(lines)


def _bible_path_for_out_dir(out_dir: Path) -> Path:
    return out_dir / "checkpoints" / "section_designs.json"


def _load_section_designs_state(
    bible_path: Path,
    story_fp: str,
    regenerate: bool = False,
) -> dict[int, dict[str, Any]]:
    """Đọc bible cũ. Bỏ qua nếu khác story_fp hoặc regenerate."""
    existing: dict[int, dict[str, Any]] = {}
    if regenerate or not bible_path.is_file():
        return existing
    try:
        meta = json.loads(bible_path.read_text(encoding="utf-8")) or {}
    except json.JSONDecodeError:
        return existing
    if meta.get("story_fp") and meta.get("story_fp") != story_fp:
        print(
            f"Cảnh báo: {bible_path.name} thuộc bản truyện khác (story_fp khác). "
            "Sẽ plan lại các section còn thiếu trên fingerprint hiện tại.",
            file=sys.stderr,
        )
        return existing
    for row in meta.get("sections") or []:
        if isinstance(row, dict) and isinstance(row.get("sectionNumber"), int):
            existing[int(row["sectionNumber"])] = row
    return existing


def _normalize_existing_bible_state_inplace(
    bible_path: Path,
    *,
    existing: dict[int, dict[str, Any]],
    sec_to_required: dict[int, list[str]],
    char_paths: dict[str, Path],
    story_fp: str,
    planner_model: str,
) -> bool:
    """
    Sửa alias key (vd. 'Dr. Ôn' → 'Ôn Đường') trên bible đã load. Persist nếu có thay đổi.
    Dùng khi resume: tránh bắt user phải --regenerate-section-designs nếu chỉ sai key.
    Trả True nếu có sửa.
    """
    changed = False
    for sec_num, design in existing.items():
        if sec_num not in sec_to_required:
            continue
        required = sec_to_required[sec_num]
        targets = _bible_normalize_targets(required, char_paths)
        if not targets:
            continue
        mapping = _normalize_bible_character_keys(design, targets)
        if mapping:
            changed = True
            pretty = ", ".join(f"{a!s} → {b!s}" for a, b in mapping)
            print(
                f"  · Section {sec_num:02d}: alias trong bible cũ đã đổi ({pretty}).",
                file=sys.stderr,
            )
    if changed:
        _persist_section_designs(
            bible_path,
            story_fp=story_fp,
            planner_model=planner_model,
            existing=existing,
        )
    return changed


def _repair_existing_bible_inplace(
    bible_path: Path,
    *,
    client: genai.Client,
    planner_model: str,
    existing: dict[int, dict[str, Any]],
    sec_to_required: dict[int, list[str]],
    char_paths: dict[str, Path],
    story_fp: str,
) -> bool:
    """
    Sau khi normalize alias mà bible cũ VẪN thiếu nhân vật (vd. 'The Husband' không
    map được vì token bị generic-strip), gọi LLM repair từng section và persist
    sau mỗi section. Tránh bắt user phải --regenerate-section-designs.
    Trả True nếu có ít nhất 1 section được sửa.
    """
    any_change = False
    for sec_num in sorted(existing.keys()):
        design = existing.get(sec_num)
        if not isinstance(design, dict):
            continue
        if sec_num not in sec_to_required:
            continue
        required = sec_to_required[sec_num]
        issues = _wardrobe_bible_missing_details(design, required)
        if not issues:
            continue
        print(
            f"  · Section {sec_num:02d}: bible cũ vẫn thiếu {len(issues)} mục — gọi repair LLM...",
            file=sys.stderr,
        )
        fixed = _repair_section_wardrobe_bible(
            client,
            planner_model,
            sec_num,
            issues,
            design,
            required,
        )
        if fixed is None:
            print(
                f"  [WARN] Section {sec_num:02d}: repair bible thất bại.",
                file=sys.stderr,
            )
            continue
        targets = _bible_normalize_targets(required, char_paths)
        aliases_after = _normalize_bible_character_keys(fixed, targets)
        if aliases_after:
            pretty = ", ".join(f"{a!s} → {b!s}" for a, b in aliases_after)
            print(
                f"  · Section {sec_num:02d}: đổi alias sau repair load ({pretty}).",
                file=sys.stderr,
            )
        existing[sec_num] = fixed
        any_change = True
        _persist_section_designs(
            bible_path,
            story_fp=story_fp,
            planner_model=planner_model,
            existing=existing,
        )
        issues_after = _wardrobe_bible_missing_details(fixed, required)
        if issues_after:
            print(
                f"  [WARN] Section {sec_num:02d}: sau repair vẫn còn thiếu — {issues_after[:3]}{'...' if len(issues_after) > 3 else ''}.",
                file=sys.stderr,
            )
        else:
            print(
                f"  · Section {sec_num:02d}: bible đã đầy đủ sau repair.",
                file=sys.stderr,
            )
    return any_change


def _persist_section_designs(
    bible_path: Path,
    *,
    story_fp: str,
    planner_model: str,
    existing: dict[int, dict[str, Any]],
) -> None:
    bible_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [existing[k] for k in sorted(existing.keys())]
    _atomic_write_json(
        bible_path,
        {
            "version": 1,
            "story_fp": story_fp,
            "planner_model": planner_model,
            "sections": rows,
        },
    )


def _plan_one_section_wardrobe_bible(
    *,
    client: genai.Client,
    planner_model: str,
    sec_num: int,
    sec_scenes: list[Scene],
    section_summary: str,
    char_paths: dict[str, Path],
    prev_design: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Plan bible cho **một** section. Trả (obj | None, required_labels).
    Tự gọi repair khi outfit/hair còn thiếu.
    """
    present_scenes = [
        s
        for s in sec_scenes
        if (getattr(s, "narrative_plane", None) or "present").strip().lower() == "present"
    ]
    story_chunks: list[str] = []
    for s in sec_scenes:
        t = (s.story_text or "").strip()
        if t and (not story_chunks or story_chunks[-1] != t):
            story_chunks.append(t)
    section_excerpt = "\n".join(story_chunks)[:6000]
    page_blob = "\n".join(
        (s.page_content or "")[:1500] for s in present_scenes[:8]
    )[:6000]
    required_labels = _required_character_labels_for_section(sec_scenes, char_paths)
    cast_list = json.dumps(sorted(char_paths.keys()), ensure_ascii=False)
    norm_targets = _bible_normalize_targets(required_labels, char_paths)

    prev_block = ""
    if prev_design:
        prev_block = (
            "\n\nPREVIOUS SECTION DESIGN (for continuity check; only change wardrobe/setting if "
            "the prose explicitly indicates a change of place, day, outfit etc.):\n"
            f"{json.dumps(prev_design, ensure_ascii=False, indent=2)[:2500]}"
        )

    if required_labels:
        mandatory_block = f"""
**MANDATORY CHARACTER LIST** — you MUST output a "characters" object with **one entry per name below**, using these EXACT display names as JSON keys (no omissions, no aliases):
{json.dumps(required_labels, ensure_ascii=False)}

**Strict naming**: each JSON key under "characters" MUST be **byte-identical** to one of the names above (preserve exact spelling, diacritics, and script as in the cast list — Vietnamese, English, or mixed). DO NOT translate them, DO NOT use English roles such as "Dr. X", "The Husband", "The Pregnant Woman", "Young Woman", "Doctor", "Pilot", etc. Use ONLY the canonical names listed above.

For EACH of those names you MUST fill **non-empty, concrete** "outfit" and "hair" strings (minimum). Also fill "footwear", "accessories", "groomingNotes" (use "none" only if truly nothing applies).
"""
        schema_hint = '"<EXACT name from MANDATORY list>"'
        hard_tail = """
Hard rules:
- Be **specific** (color, fabric, length, sleeves, hairstyle). Avoid vague words like "casual outfit" or "nice dress".
- Wardrobe MUST match the section's situation (home / hospital / office / evening gown event / maternity / sleepwear, etc.).
- For present-reality continuity across sections: reuse the PREVIOUS SECTION DESIGN wardrobe when prose does not say anyone changed clothes or moved to a new day/place.
- You CANNOT skip any name from the MANDATORY list. You CANNOT leave "outfit" or "hair" empty.
- Output ONLY a single JSON object (no markdown fences, no extra commentary).
"""
    else:
        mandatory_block = f"""
**AVAILABLE CAST** (canonical JSON keys — use ONLY these spellings when you include a character):
{cast_list}

Include a "characters" entry **ONLY** for cast members who **actually appear** in this section's illustration beats or prose (SECTION STORY EXCERPT / PRESENT-BEAT BRIEFS below). Do NOT list the entire cast unless every one of them truly appears in **this** section.

If nobody from the cast appears in this section, return "characters": {{}}.

**Strict naming**: each JSON key you output MUST be **byte-identical** to one of the AVAILABLE CAST names (preserve exact spelling, diacritics, and script). DO NOT use English role labels such as "Dr. X", "The Husband", "Young Woman", "Doctor", "Medical Professional", "Pilot", etc.

For each character you DO include, fill **non-empty, concrete** "outfit" and "hair" (minimum), plus "footwear", "accessories", "groomingNotes" (use "none" only if truly nothing applies).
"""
        schema_hint = '"<EXACT name from AVAILABLE CAST, only if they appear in this section>"'
        hard_tail = """
Hard rules:
- Be **specific** (color, fabric, length, sleeves, hairstyle). Avoid vague words like "casual outfit" or "nice dress".
- Wardrobe MUST match the section's situation (home / hospital / office / evening gown event / maternity / sleepwear, etc.).
- For present-reality continuity across sections: reuse the PREVIOUS SECTION DESIGN wardrobe when prose does not say anyone changed clothes or moved to a new day/place.
- Do NOT add wardrobe entries for cast members who do not appear in this section.
- For every character you DO include, you CANNOT leave "outfit" or "hair" empty.
- Output ONLY a single JSON object (no markdown fences, no extra commentary).
"""

    prompt = f"""
Task: Author a SECTION WARDROBE BIBLE for an illustration pipeline (any genre or era: contemporary, historical,
period drama, sci-fi, fantasy, medical/legal procedural, children's — follow the USER'S prose only).

Each section in the story has its own canonical setting + per-character wardrobe that MUST stay consistent across every illustration of that section.

You are designing section {sec_num}.

Section label / summary: {section_summary!r}
{mandatory_block}
SECTION STORY EXCERPT (verbatim prose for this section):
\"\"\"
{section_excerpt}
\"\"\"

PRESENT-BEAT IMAGE BRIEFS already drafted (consolidate inconsistent wording into ONE canonical wardrobe per character for this section):
\"\"\"
{page_blob}
\"\"\"

{prev_block}

Output a SINGLE JSON object describing this section. Schema:

{{
  "sectionNumber": int,
  "setting": {{
    "location": str,
    "timeOfDay": str,
    "lighting": str,
    "weather": str,
    "ambientPalette": str
  }},
  "characters": {{
    {schema_hint}: {{
      "outfit": str,
      "hair": str,
      "footwear": str,
      "accessories": str,
      "groomingNotes": str
    }}
  }}
}}
{hard_tail}
"""

    try:
        resp = client.models.generate_content(
            model=planner_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        raw = resp.text or ""
        obj = _json_loads_llm(_strip_json_fence(raw))
        if not isinstance(obj, dict):
            raise ValueError("Section design response không phải JSON object.")
        obj["sectionNumber"] = int(sec_num)
    except (ValueError, json.JSONDecodeError) as exc:
        print(
            f"  [WARN] Section {sec_num:02d}: plan bible lỗi ({exc}).",
            file=sys.stderr,
        )
        return None, required_labels

    aliases_fixed = _normalize_bible_character_keys(obj, norm_targets)
    if aliases_fixed:
        pretty = ", ".join(f"{a!s} → {b!s}" for a, b in aliases_fixed)
        print(
            f"  · Section {sec_num:02d}: đổi alias trong bible ({pretty}).",
            file=sys.stderr,
        )

    issues = _wardrobe_bible_missing_details(obj, required_labels)
    if issues:
        fixed = _repair_section_wardrobe_bible(
            client,
            planner_model,
            sec_num,
            issues,
            obj,
            required_labels,
        )
        if fixed is not None:
            aliases_fixed2 = _normalize_bible_character_keys(fixed, norm_targets)
            if aliases_fixed2:
                pretty = ", ".join(f"{a!s} → {b!s}" for a, b in aliases_fixed2)
                print(
                    f"  · Section {sec_num:02d}: đổi alias sau repair ({pretty}).",
                    file=sys.stderr,
                )
            issues2 = _wardrobe_bible_missing_details(fixed, required_labels)
            if not issues2:
                obj = fixed
                print(
                    f"  · Section {sec_num:02d}: bible đã repair sau validation.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [WARN] Section {sec_num:02d}: bible sau repair vẫn thiếu — {issues2}.",
                    file=sys.stderr,
                )
        else:
            print(
                f"  [WARN] Section {sec_num:02d}: repair bible thất bại; giữ bản gốc thiếu trang phục.",
                file=sys.stderr,
            )

    return obj, required_labels


def _plan_section_designs(
    *,
    client: genai.Client,
    story: str,
    scenes: list[Scene],
    char_paths: dict[str, Path],
    planner_model: str,
    out_dir: Path,
    regenerate: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Sinh "Wardrobe Bible" cho mỗi outline section: setting + per-character wardrobe.
    Lưu/đọc từ checkpoints/section_designs.json để bạn có thể chỉnh tay trước khi render ảnh.
    Resume-friendly: chỉ plan section chưa có. Persist sau MỖI section xong.
    """
    bible_path = _bible_path_for_out_dir(out_dir)
    bible_path.parent.mkdir(parents=True, exist_ok=True)
    story_fp = _story_fingerprint(story)

    existing = _load_section_designs_state(bible_path, story_fp, regenerate=regenerate)

    section_to_scenes: dict[int, list[Scene]] = {}
    section_summary: dict[int, str] = {}
    for sc in scenes:
        sec = getattr(sc, "outline_section_number", None)
        if sec is None:
            continue
        sec_int = int(sec)
        section_to_scenes.setdefault(sec_int, []).append(sc)
        if sec_int not in section_summary and sc.outline_section_summary:
            section_summary[sec_int] = sc.outline_section_summary

    sorted_sections = sorted(section_to_scenes.keys())
    if not sorted_sections:
        return existing

    # Sửa alias key cho các section đã có sẵn (vd. "Dr. Ôn" → "Ôn Đường"),
    # rồi gọi LLM repair nếu sau normalize vẫn thiếu nhân vật.
    if existing:
        sec_to_required = {
            sn: _required_character_labels_for_section(section_to_scenes[sn], char_paths)
            for sn in sorted_sections
            if sn in existing
        }
        _normalize_existing_bible_state_inplace(
            bible_path,
            existing=existing,
            sec_to_required=sec_to_required,
            char_paths=char_paths,
            story_fp=story_fp,
            planner_model=planner_model,
        )
        _repair_existing_bible_inplace(
            bible_path,
            client=client,
            planner_model=planner_model,
            existing=existing,
            sec_to_required=sec_to_required,
            char_paths=char_paths,
            story_fp=story_fp,
        )

    todo = [s for s in sorted_sections if s not in existing]
    if not todo:
        print(
            f"Section designs: đã có đủ {len(existing)}/{len(sorted_sections)} bible (reuse).",
            file=sys.stderr,
        )
        _warn_section_design_wardrobe_gaps(existing, section_to_scenes, char_paths)
        return existing

    print(
        f"Section designs: cần plan {len(todo)}/{len(sorted_sections)} bible "
        f"(lưu vào {bible_path}, có thể chỉnh tay trước khi render).",
        file=sys.stderr,
    )

    for sec_num in todo:
        obj, required_labels = _plan_one_section_wardrobe_bible(
            client=client,
            planner_model=planner_model,
            sec_num=sec_num,
            sec_scenes=section_to_scenes[sec_num],
            section_summary=section_summary.get(sec_num, ""),
            char_paths=char_paths,
            prev_design=existing.get(sec_num - 1),
        )
        if obj is None:
            continue
        existing[sec_num] = obj
        _persist_section_designs(
            bible_path,
            story_fp=story_fp,
            planner_model=planner_model,
            existing=existing,
        )
        n_chars = len((obj.get("characters") or {}).keys()) if isinstance(obj.get("characters"), dict) else 0
        if _wardrobe_bible_missing_details(obj, required_labels):
            print(
                f"  · Section {sec_num:02d}: bible đã lưu ({n_chars} character) — **kiểm tra outfit/hair** trong file.",
                file=sys.stderr,
            )
        else:
            print(
                f"  · Section {sec_num:02d}: bible đã lưu ({n_chars} character, đủ outfit+hair).",
                file=sys.stderr,
            )

    _warn_section_design_wardrobe_gaps(existing, section_to_scenes, char_paths)
    return existing


def _existing_section_keyframe(out_dir: Path, sec_num: int) -> Path | None:
    kdir = out_dir / "section_keyframes"
    if not kdir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = kdir / f"section_{sec_num:02d}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _build_section_designs(scenes: list[Scene]) -> dict[int, dict[str, Any]]:
    """Gom mỗi outline section: summary, baseline pageContent (beat present đầu), extras, suggestedReferences union."""
    designs: dict[int, dict[str, Any]] = {}
    for sc in scenes:
        sec = getattr(sc, "outline_section_number", None)
        if sec is None:
            continue
        plane = (getattr(sc, "narrative_plane", None) or "present").strip().lower()
        if plane != "present":
            continue
        d = designs.setdefault(
            int(sec),
            {
                "section_summary": getattr(sc, "outline_section_summary", None) or "",
                "first_page_content": "",
                "extra_page_contents": [],
                "all_page_contents": [],
                "suggested_refs": [],
            },
        )
        if not d["first_page_content"] and sc.page_content:
            d["first_page_content"] = sc.page_content
        elif sc.page_content and len(d["extra_page_contents"]) < 4:
            d["extra_page_contents"].append(sc.page_content)
        if sc.page_content:
            d["all_page_contents"].append(sc.page_content)
        for ref in (sc.suggested_references or []):
            if ref and ref not in d["suggested_refs"]:
                d["suggested_refs"].append(ref)
    return designs


_VN_NAME_STOPWORDS = frozenset(
    {
        "của",
        "và",
        "cho",
        "với",
        "trong",
        "một",
        "người",
        "được",
        "các",
        "những",
        "the",
        "a",
        "an",
    }
)


def _fold_for_char_match(s: str) -> str:
    """Chuẩn hoá để so khớp tên (bỏ dấu, lower ASCII)."""
    if not s:
        return ""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def _detect_section_characters(
    page_contents: list[str],
    char_paths: dict[str, Path],
) -> list[str]:
    """
    Quét pageContent / storyText của section để dò nhân vật (key của char_paths) có thực sự xuất hiện.
    So khớp có bỏ dấu; giữ thứ tự key trong char_paths.
    """
    if not char_paths or not page_contents:
        return []
    blob_fold = _fold_for_char_match("\n".join(page_contents))
    found: list[str] = []
    for label in char_paths.keys():
        needle = (label or "").strip()
        if not needle:
            continue
        if _fold_for_char_match(needle) in blob_fold:
            found.append(label)
            continue
        tokens = [
            t
            for t in re.split(r"\s+", needle)
            if len(t) >= 2 and t.lower() not in _VN_NAME_STOPWORDS
        ]
        if tokens and all(_fold_for_char_match(t) in blob_fold for t in tokens):
            found.append(label)
    return found


def _resolve_keyframe_ref_paths(
    ref_filenames: list[str],
    char_paths: dict[str, Path],
) -> list[tuple[str, Path]]:
    """Map filenames trong suggested_references về (label, Path) bằng cách tra trong char_paths."""
    if not ref_filenames or not char_paths:
        return []
    by_filename: dict[str, tuple[str, Path]] = {}
    for label, p in char_paths.items():
        try:
            by_filename[p.name] = (label, p)
        except Exception:
            continue
    resolved: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for fn in ref_filenames:
        item = by_filename.get(fn)
        if item is None:
            continue
        if item[1].name in seen:
            continue
        seen.add(item[1].name)
        resolved.append(item)
    return resolved


def _existing_section_char_ref(
    out_dir: Path,
    sec_num: int,
    char_label: str,
) -> Path | None:
    kdir = out_dir / "section_keyframes"
    if not kdir.is_dir():
        return None
    safe = _safe_filename(char_label)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = kdir / f"section_{sec_num:02d}__{safe}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _render_one_section_keyframes(
    *,
    client: genai.Client,
    sec_num: int,
    sec_scenes: list[Scene],
    section_design: dict[str, Any] | None,
    out_dir: Path,
    char_paths: dict[str, Path],
    image_model: str,
    color_mode: str,
    aspect_ratio: str,
    art_style: str,
    ext_for_mime: dict[str, str],
    regenerate: bool,
    render_character_refs: bool,
) -> tuple[Path | None, dict[str, Path]]:
    """
    Render wardrobe sheets + section ENVIRONMENT keyframe cho **một** section.
    Trả (keyframe_path, {char_label: sheet_path}).
    Thứ tự cố định:
      1. Wardrobe sheet cho từng nhân vật (dựa trên auto_refs portrait + Wardrobe Bible).
      2. Environment keyframe (ESTABLISHING STILL — chỉ bối cảnh / props / ánh sáng, KHÔNG có nhân vật).
    Section keyframe sẽ làm anchor bối cảnh cho beat đầu của section; các beat sau dùng beat-trước làm anchor.
    """
    kdir = out_dir / "section_keyframes"
    kdir.mkdir(parents=True, exist_ok=True)

    # Lấy thông tin section từ scene list
    designs = _build_section_designs(sec_scenes)
    d = designs.get(sec_num) or {
        "section_summary": "",
        "first_page_content": "",
        "extra_page_contents": [],
        "all_page_contents": [],
    }
    baseline = (d.get("first_page_content") or "").strip()
    all_pcs: list[str] = d.get("all_page_contents") or []
    sec_summary = d.get("section_summary") or ""

    # Dò nhân vật
    detection_src = _gather_section_detection_texts(sec_scenes)
    detected_labels = _detect_section_characters(detection_src, char_paths)
    bible_labels: list[str] = []
    if section_design and isinstance(section_design.get("characters"), dict):
        bible_labels = [k for k in section_design["characters"] if k in char_paths]
    labels_for_section = detected_labels if detected_labels else bible_labels
    used_bible_fallback = bool(not detected_labels and bible_labels)

    sec_char_map: dict[str, Path] = {}

    # Bước 1: per-character wardrobe sheet
    if render_character_refs and char_paths and baseline and labels_for_section:
        if used_bible_fallback:
            print(
                f"  - Section {sec_num:02d}: dò từ bible {len(labels_for_section)} nhân vật để render wardrobe sheet trước.",
                file=sys.stderr,
            )
        else:
            print(
                f"  - Section {sec_num:02d}: render {len(labels_for_section)} wardrobe sheet trước rồi mới dựng environment keyframe.",
                file=sys.stderr,
            )
        for char_label in labels_for_section:
            portrait = char_paths.get(char_label)
            if portrait is None or not portrait.is_file():
                continue

            cpath: Path | None = None
            if not regenerate:
                cpath = _existing_section_char_ref(out_dir, sec_num, char_label)

            if cpath is not None:
                print(
                    f"    · Section {sec_num:02d}/{char_label}: đã có {cpath.name}, dùng lại.",
                    file=sys.stderr,
                )
                sec_char_map[char_label] = cpath
                continue

            print(
                f"    · Section {sec_num:02d}/{char_label}: render character ref...",
                file=sys.stderr,
            )
            try:
                cdata, cmime = generate_section_character_ref(
                    client,
                    section_num=sec_num,
                    char_label=char_label,
                    char_portrait_path=portrait,
                    section_keyframe_path=None,
                    baseline_page_content=baseline,
                    section_page_contents=all_pcs,
                    section_summary=sec_summary,
                    image_model=image_model,
                    color_mode=color_mode,
                    art_style=art_style,
                    section_design=section_design,
                )
            except RuntimeError as exc:
                print(
                    f"      [WARN] section {sec_num:02d}/{char_label} lỗi: {exc}; bỏ qua, sẽ dùng portrait gốc.",
                    file=sys.stderr,
                )
                continue

            cext = ext_for_mime.get(cmime or "", ".png")
            cpath = kdir / f"section_{sec_num:02d}__{_safe_filename(char_label)}{cext}"
            cpath.write_bytes(cdata)
            print(f"      -> {cpath}", file=sys.stderr)
            sec_char_map[char_label] = cpath
    elif render_character_refs and not labels_for_section and char_paths and baseline:
        print(
            f"    (Section {sec_num:02d}: không có nhân vật để wardrobe sheet — bỏ qua.)",
            file=sys.stderr,
        )

    # Bước 2: section ENVIRONMENT keyframe (NO PEOPLE)
    kpath: Path | None = None
    if not regenerate:
        kpath = _existing_section_keyframe(out_dir, sec_num)

    if kpath is not None:
        print(f"  - Section {sec_num:02d}: đã có {kpath.name}, dùng lại.", file=sys.stderr)
        return kpath, sec_char_map

    if not baseline:
        print(
            f"  - Section {sec_num:02d}: không có pageContent baseline, bỏ qua keyframe.",
            file=sys.stderr,
        )
        return None, sec_char_map

    print(
        f"  - Section {sec_num:02d}: render environment keyframe (no people)...",
        file=sys.stderr,
    )
    try:
        data, mime = generate_section_keyframe(
            client,
            section_num=sec_num,
            section_summary=sec_summary,
            baseline_page_content=baseline,
            extra_present_excerpts=d.get("extra_page_contents") or [],
            ref_paths=[],
            image_model=image_model,
            color_mode=color_mode,
            aspect_ratio=aspect_ratio,
            art_style=art_style,
            section_design=section_design,
            ref_kind="none",
        )
    except RuntimeError as exc:
        print(
            f"    [WARN] section {sec_num:02d} keyframe lỗi: {exc}; bỏ qua, sẽ dùng anchor mặc định (beat đầu).",
            file=sys.stderr,
        )
        return None, sec_char_map

    ext = ext_for_mime.get(mime or "", ".png")
    kpath = kdir / f"section_{sec_num:02d}{ext}"
    kpath.write_bytes(data)
    print(f"    -> {kpath}", file=sys.stderr)
    return kpath, sec_char_map


def _render_section_keyframes(
    *,
    client: genai.Client,
    scenes: list[Scene],
    out_dir: Path,
    char_paths: dict[str, Path],
    style_paths: list[Path],
    image_model: str,
    color_mode: str,
    aspect_ratio: str,
    art_style: str,
    ext_for_mime: dict[str, str],
    regenerate: bool,
    render_character_refs: bool,
    section_designs: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[int, Path], dict[int, dict[str, Path]]]:
    """
    Pre-render keyframes cho mỗi outline section:
      - 1 ảnh establishing still (section_XX.<ext>)
      - (tùy chọn) 1 ảnh wardrobe sheet cho mỗi nhân vật xuất hiện trong section (section_XX__<tên>.<ext>)
    Trả (section_keyframe_paths, per_section_char_paths) với:
      - section_keyframe_paths[sec_num] = Path establishing still
      - per_section_char_paths[sec_num][char_label] = Path wardrobe sheet
    """
    designs = _build_section_designs(scenes)
    if not designs:
        print("Không có outline section nào có present beat; bỏ qua section keyframes.", file=sys.stderr)
        return {}, {}

    kdir = out_dir / "section_keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    char_paths_by_section: dict[int, dict[str, Path]] = {}

    sorted_sections = sorted(designs.keys())
    print(
        f"Section keyframes: {len(sorted_sections)} section(s) cần render "
        f"(skip nếu đã tồn tại trong {kdir})...",
        file=sys.stderr,
    )

    for sec_num in sorted_sections:
        sec_scenes = [
            s
            for s in scenes
            if getattr(s, "outline_section_number", None) is not None
            and int(s.outline_section_number) == sec_num
        ]
        kpath, sec_char_map = _render_one_section_keyframes(
            client=client,
            sec_num=sec_num,
            sec_scenes=sec_scenes,
            section_design=(section_designs or {}).get(sec_num),
            out_dir=out_dir,
            char_paths=char_paths,
            image_model=image_model,
            color_mode=color_mode,
            aspect_ratio=aspect_ratio,
            art_style=art_style,
            ext_for_mime=ext_for_mime,
            regenerate=regenerate,
            render_character_refs=render_character_refs,
        )
        if sec_char_map:
            char_paths_by_section[sec_num] = sec_char_map
        if kpath is not None:
            paths[sec_num] = kpath

    print(
        f"Section keyframes hoàn tất: {len(paths)}/{len(sorted_sections)} section"
        + (
            f"; per-section character refs: {sum(len(v) for v in char_paths_by_section.values())} ảnh."
            if render_character_refs
            else "."
        ),
        file=sys.stderr,
    )
    return paths, char_paths_by_section


def _compact_neighbor_for_split_context(row: dict[str, Any]) -> dict[str, Any]:
    story_text = (row.get("storyText") or "").strip()
    return {
        "pageNumber": row.get("pageNumber"),
        "narrativePlane": row.get("narrativePlane") or "present",
        "storySegment": (row.get("storySegment") or "").strip(),
        "storyText": story_text[:220],
        "pageContent": (row.get("pageContent") or "").strip()[:320],
    }


def _renumber_review_manifest_rows(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            row["pageNumber"] = i


def _persist_review_manifest_rows(
    *,
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    manifest_path: Path | None = None,
) -> None:
    if checkpoint_path.is_file():
        try:
            ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ck = None
        if isinstance(ck, dict):
            ck["scenes"] = rows
            _atomic_write_json(checkpoint_path, ck)
    if manifest_path is not None:
        _atomic_write_json(manifest_path, rows)


def _merge_split_beat_row(template: dict[str, Any], piece: dict[str, Any]) -> dict[str, Any]:
    refs = piece.get("suggestedReferences")
    if not isinstance(refs, list) or not refs:
        refs = template.get("suggestedReferences") or []
    row: dict[str, Any] = {
        "mode": template.get("mode", "review"),
        "storySegment": str(piece.get("storySegment") or template.get("storySegment") or "").strip(),
        "startAnchor": str(piece.get("startAnchor") or "").strip(),
        "endAnchor": str(piece.get("endAnchor") or "").strip(),
        "storyText": str(piece.get("storyText") or "").strip(),
        "pageContent": str(piece.get("pageContent") or "").strip(),
        "panelCount": 1,
        "suggestedReferences": list(refs),
        "narrativePlane": _parse_narrative_plane(piece)
        if piece.get("narrativePlane") is not None
        else _parse_narrative_plane(template),
    }
    if template.get("outlineSectionNumber") is not None:
        row["outlineSectionNumber"] = template.get("outlineSectionNumber")
    summ = (template.get("outlineSectionSummary") or "").strip()
    if summ:
        row["outlineSectionSummary"] = summ
    return row


def _split_long_review_beat_with_llm(
    *,
    client: Any,
    planner_model: str,
    target_row: dict[str, Any],
    neighbors_before: list[dict[str, Any]],
    neighbors_after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_compact = {
        "pageNumber": target_row.get("pageNumber"),
        "narrativePlane": target_row.get("narrativePlane") or "present",
        "outlineSectionNumber": target_row.get("outlineSectionNumber"),
        "outlineSectionSummary": (target_row.get("outlineSectionSummary") or "").strip(),
        "storySegment": (target_row.get("storySegment") or "").strip(),
        "startAnchor": (target_row.get("startAnchor") or "").strip(),
        "endAnchor": (target_row.get("endAnchor") or "").strip(),
        "storyText": (target_row.get("storyText") or "").strip(),
        "pageContent": (target_row.get("pageContent") or "").strip(),
        "suggestedReferences": target_row.get("suggestedReferences") or [],
    }
    prompt = f"""\
You are re-splitting ONE overly long review illustration beat into 2–4 SHORTER beats.

GOAL:
- The TARGET beat's storyText is too long for one still image + one TTS clip.
- Split by VISUAL RHYTHM: each output beat = ONE drawable frozen moment (one still + one motion clip).
- Keep anchors/storyText VERBATIM from the TARGET beat's storyText only (source language unchanged).
- Do NOT borrow or duplicate text from neighbor beats.

{_VISUAL_RHYTHM_SPLIT_RULES}

DURATION (each output beat):
- Target 3–5 seconds of narration; never exceed ~8 seconds.
- Vietnamese: ~8–10 syllables / 3s, ~12–15 / 5s, ~20–24 / 8s (ABSOLUTE CEILING).

NEIGHBORS BEFORE (read-only context — up to 2 beats; DO NOT copy their text into output):
{json.dumps(neighbors_before, ensure_ascii=False, indent=2)}

TARGET BEAT (split THIS beat only):
{json.dumps(target_compact, ensure_ascii=False, indent=2)}

NEIGHBORS AFTER (read-only context — up to 2 beats; DO NOT copy their text into output):
{json.dumps(neighbors_after, ensure_ascii=False, indent=2)}

Task:
- Replace the TARGET beat with 2–4 beats in narrative order.
- Concatenated storyText of all output beats MUST cover the TARGET storyText exactly once, in order, with NO gaps and NO duplicated clauses.
- startAnchor/endAnchor for each output beat MUST be exact substrings of the TARGET storyText.
- pageContent: ENGLISH visual brief for THAT slice only; vary shot/staging across the split when the visual beat changes.
- suggestedReferences: usually same as TARGET unless a slice clearly drops a co-present character.
- narrativePlane: usually same as TARGET unless a slice is clearly a different memory register.

Return ONLY a JSON array (2–4 objects). Each object MUST have EXACTLY these keys:
"storySegment", "startAnchor", "endAnchor", "storyText", "pageContent", "panelCount", "suggestedReferences", "narrativePlane".
Do NOT include pageNumber or motionPrompt.
"""
    def _call_split() -> Any:
        return client.models.generate_content(
            model=planner_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.35,
                response_mime_type="application/json",
            ),
        )

    resp = _generate_with_transient_retry(
        _call_split,
        label=f"split long beat pageNumber={target_row.get('pageNumber')}",
    )
    if resp is None:
        return []
    try:
        parsed = _parse_planner_json_array(resp.text or "", client, planner_model)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(parsed, list) or len(parsed) < 2:
        return []
    out: list[dict[str, Any]] = []
    for piece in parsed:
        if not isinstance(piece, dict):
            continue
        merged = _merge_split_beat_row(target_row, piece)
        if not (merged.get("storyText") or "").strip() or not (merged.get("pageContent") or "").strip():
            continue
        out.append(merged)
    return out if len(out) >= 2 else []


def split_long_review_beats_in_checkpoint(
    *,
    client: Any,
    checkpoint_path: Path,
    manifest_path: Path,
    planner_model: str,
    tts_sec_threshold: float = _LONG_BEAT_AUTO_SPLIT_SEC,
) -> int:
    """
    Tự động tách các beat review có storyText dài hơn `tts_sec_threshold` (mặc định 10s),
    dùng ngữ cảnh ±2 beat lân cận, đánh số lại pageNumber, ghi checkpoint + scenes.json,
    rồi backfill motionPrompt cho beat mới/thiếu.

    Trả về số beat gốc đã được tách thành công.
    """
    if not checkpoint_path.is_file():
        return 0
    try:
        ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    if not isinstance(ck, dict):
        return 0
    raw_rows = ck.get("scenes")
    if not isinstance(raw_rows, list) or not raw_rows:
        return 0

    rows: list[dict[str, Any]] = [r for r in raw_rows if isinstance(r, dict)]
    if not rows:
        return 0

    rows.sort(key=lambda r: int(r.get("pageNumber") or 0))
    syll_threshold = _long_beat_syllable_threshold(tts_sec_threshold)
    n_split_sources = 0
    idx = 0
    split_attempts_at_idx = 0
    while idx < len(rows):
        row = rows[idx]
        story_text = (row.get("storyText") or "").strip()
        if not story_text:
            idx += 1
            split_attempts_at_idx = 0
            continue
        if _count_tts_syllables(story_text) <= syll_threshold:
            idx += 1
            split_attempts_at_idx = 0
            continue
        if split_attempts_at_idx >= 3:
            print(
                f"  [WARN] beat {row.get('pageNumber')}: vẫn dài sau 3 lần tách; bỏ qua.",
                file=sys.stderr,
            )
            idx += 1
            split_attempts_at_idx = 0
            continue

        pn = row.get("pageNumber")
        approx_s = _tts_seconds_for_text(story_text)
        print(
            f"Auto-split beat {pn}: ~{approx_s:.1f}s TTS (> {tts_sec_threshold:.0f}s) — "
            f"tách theo nhịp hình ảnh (context ±2 beat)...",
            file=sys.stderr,
        )
        neighbors_before = [
            _compact_neighbor_for_split_context(rows[j])
            for j in range(max(0, idx - 2), idx)
        ]
        neighbors_after = [
            _compact_neighbor_for_split_context(rows[j])
            for j in range(idx + 1, min(len(rows), idx + 3))
        ]
        new_rows = _split_long_review_beat_with_llm(
            client=client,
            planner_model=planner_model,
            target_row=row,
            neighbors_before=neighbors_before,
            neighbors_after=neighbors_after,
        )
        if len(new_rows) < 2:
            print(
                f"  [WARN] beat {pn}: planner không trả split hợp lệ; giữ nguyên.",
                file=sys.stderr,
            )
            idx += 1
            split_attempts_at_idx = 0
            continue

        insert_at = idx
        rows[insert_at : insert_at + 1] = new_rows
        n_split_sources += 1
        split_attempts_at_idx += 1
        _renumber_review_manifest_rows(rows)
        scene_objs = [_manifest_dict_to_scene(r) for r in rows]
        _dedupe_review_story_text_overlaps(scene_objs)
        rows = [scene_to_manifest_dict(s, "review") for s in scene_objs]
        _persist_review_manifest_rows(
            rows=rows,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )
        print(
            f"  ✓ beat {pn} → {len(new_rows)} beat (tổng {len(rows)}); đã lưu checkpoint + scenes.json.",
            file=sys.stderr,
        )
        still_long_at: int | None = None
        for j in range(insert_at, insert_at + len(new_rows)):
            if _count_tts_syllables((rows[j].get("storyText") or "").strip()) > syll_threshold:
                still_long_at = j
                break
        if still_long_at is None:
            idx = insert_at + len(new_rows)
            split_attempts_at_idx = 0
        else:
            idx = still_long_at

    if n_split_sources > 0:
        backfill_motion_prompts_in_scenes_json(
            client=client,
            scenes_path=manifest_path,
            planner_model=planner_model,
            batch_size=10,
            force=False,
            review_checkpoint_path=checkpoint_path,
        )
    return n_split_sources


def _sync_motion_prompts_to_review_checkpoint(
    *, scenes_path: Path, checkpoint_path: Path
) -> int:
    """
    Sau khi backfill `motionPrompt` vào scenes.json, sync field này vào field `scenes` của
    checkpoint `review_two_pass.json` để lần resume kế tiếp:
      • planner load lại scenes từ checkpoint vẫn có motionPrompt sẵn,
      • _persist khi ghi đè scenes.json không làm mất motionPrompt,
      • auto-backfill ở cuối main() thấy 0 beat thiếu → skip (không chạy lại vô ích).

    Trả về số entry trong checkpoint đã được cập nhật. Bỏ qua mọi lỗi nhẹ (best-effort).
    """
    if not checkpoint_path.is_file():
        return 0
    try:
        scenes_rows = json.loads(scenes_path.read_text(encoding="utf-8"))
        ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    if not isinstance(scenes_rows, list) or not isinstance(ck, dict):
        return 0
    ck_scenes = ck.get("scenes")
    if not isinstance(ck_scenes, list):
        return 0

    pn_to_mp: dict[int, str] = {}
    for r in scenes_rows:
        if not isinstance(r, dict):
            continue
        pn = r.get("pageNumber")
        mp = (r.get("motionPrompt") or "").strip()
        if pn is None or not mp:
            continue
        try:
            pn_to_mp[int(pn)] = mp
        except (TypeError, ValueError):
            continue

    n_updated = 0
    for r in ck_scenes:
        if not isinstance(r, dict):
            continue
        pn = r.get("pageNumber")
        if pn is None:
            continue
        try:
            key = int(pn)
        except (TypeError, ValueError):
            continue
        new_mp = pn_to_mp.get(key, "")
        old_mp = (r.get("motionPrompt") or "").strip()
        if new_mp and new_mp != old_mp:
            r["motionPrompt"] = new_mp
            n_updated += 1

    if n_updated > 0:
        try:
            _atomic_write_json(checkpoint_path, ck)
            print(
                f"  [sync] đồng bộ {n_updated} motionPrompt vào checkpoint "
                f"{checkpoint_path.name} → lần resume kế tiếp sẽ giữ nguyên.",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [WARN] sync checkpoint thất bại ({exc}); scenes.json đã ok, bỏ qua.",
                file=sys.stderr,
            )
            return 0
    return n_updated


def backfill_motion_prompts_in_scenes_json(
    *,
    client: Any,
    scenes_path: Path,
    planner_model: str,
    batch_size: int = 10,
    force: bool = False,
    review_checkpoint_path: Path | None = None,
    only_page_numbers: frozenset[int] | None = None,
    neighbor_radius: int = _MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS,
) -> int:
    """
    Đọc scenes.json sẵn có, với mỗi beat thiếu/empty `motionPrompt` gọi planner sinh prompt
    WAN 2.1 I2V theo `_WAN_MOTION_PROMPT_RULES`, rồi ghi đè ngược lại scenes.json.

    Dùng khi planner chạy trước đó (Gemini 2.5-flash) đã bỏ field `motionPrompt`, để khỏi
    phải re-run toàn bộ planner / re-generate ảnh.

    Khi `review_checkpoint_path` được cung cấp (vd. out_dir/checkpoints/review_two_pass.json):
    sau mỗi batch lưu xong cũng sync motionPrompt vào checkpoint, để lần planner-resume kế tiếp
    không bị mất kết quả backfill.

    `only_page_numbers`: nếu set, chỉ xử lý beat có `pageNumber` nằm trong tập này; None = toàn bộ.
    `neighbor_radius`: mỗi beat trong batch đọc thêm ±N beat lân cận (ngoài batch) làm ngữ cảnh read-only.

    Trả về số beat đã backfill thành công.
    """
    neighbor_radius = max(0, min(int(neighbor_radius), _MOTION_BACKFILL_MAX_NEIGHBOR_RADIUS))
    raw_text = scenes_path.read_text(encoding="utf-8")
    rows = json.loads(raw_text)
    if not isinstance(rows, list):
        raise SystemExit(f"scenes.json phải là JSON array: {scenes_path}")

    if only_page_numbers:
        present_pages = {
            int(r.get("pageNumber"))
            for r in rows
            if isinstance(r, dict) and isinstance(r.get("pageNumber"), int)
        }
        missing_pages = sorted(only_page_numbers - present_pages)
        if missing_pages:
            preview = ", ".join(str(p) for p in missing_pages[:12])
            tail = "..." if len(missing_pages) > 12 else ""
            print(
                f"  [WARN] backfill motionPrompt: không có pageNumber trong scenes.json: "
                f"{preview}{tail}",
                file=sys.stderr,
            )

    targets: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        pn = row.get("pageNumber")
        if only_page_numbers is not None:
            if not isinstance(pn, int) or pn not in only_page_numbers:
                continue
        existing = (row.get("motionPrompt") or "").strip()
        if force or not existing:
            targets.append((idx, row))

    if not targets:
        if only_page_numbers:
            scope = ", ".join(str(p) for p in sorted(only_page_numbers))
            print(
                f"motionPrompt: không có beat nào cần backfill trong phạm vi pageNumber "
                f"[{scope}] của {scenes_path}"
                f"{' (đã đủ hoặc không khớp pageNumber)' if not force else ''}.",
                file=sys.stderr,
            )
        else:
            print(
                f"motionPrompt đã đủ cho toàn bộ {len(rows)} beat trong {scenes_path}. "
                "Không cần backfill.",
                file=sys.stderr,
            )
        return 0

    if only_page_numbers:
        in_scope = sum(
            1
            for r in rows
            if isinstance(r, dict)
            and isinstance(r.get("pageNumber"), int)
            and r["pageNumber"] in only_page_numbers
        )
        n_already = max(0, in_scope - len(targets))
    else:
        n_already = len(rows) - len(targets)
    n_total_batches = (len(targets) + batch_size - 1) // batch_size
    resume_note = (
        f" (skip {n_already} beat đã có motionPrompt → resume)" if n_already > 0 and not force else ""
    )
    scope_note = ""
    if only_page_numbers:
        scope_note = (
            f", chỉ pageNumber [{', '.join(str(p) for p in sorted(only_page_numbers))}]"
        )
    print(
        f"Backfill motionPrompt: {len(targets)}/{len(rows)} beat cần điền"
        f"{resume_note}{scope_note}; batch={batch_size}, n_batches={n_total_batches}, "
        f"context=±{neighbor_radius} beat, model={planner_model}.",
        file=sys.stderr,
    )

    def _neighbor_summary(idx: int) -> dict[str, Any] | None:
        """Compact context row cho adjacency awareness — KHÔNG yêu cầu motionPrompt."""
        if idx < 0 or idx >= len(rows):
            return None
        nb = rows[idx]
        if not isinstance(nb, dict):
            return None
        story_text_raw = (nb.get("storyText", "") or "").strip()
        page_content_raw = (nb.get("pageContent", "") or "").strip()
        return {
            "pageNumber": nb.get("pageNumber"),
            "narrativePlane": nb.get("narrativePlane") or "present",
            "storyText": story_text_raw[:200],  # truncate dài để tiết kiệm token
            "pageContent": page_content_raw[:280],
        }

    n_filled = 0
    for batch_idx, start in enumerate(range(0, len(targets), batch_size), start=1):
        chunk = targets[start : start + batch_size]
        batch_row_indices = {idx for idx, _ in chunk}

        # (B) outlineSectionSummary đưa vào mỗi target để model có tonal context cho cả section.
        target_items: list[dict[str, Any]] = []
        for _, row in chunk:
            target_items.append(
                {
                    "pageNumber": row.get("pageNumber"),
                    "outlineSectionNumber": row.get("outlineSectionNumber"),
                    "outlineSectionSummary": (row.get("outlineSectionSummary", "") or "").strip(),
                    "narrativePlane": row.get("narrativePlane") or "present",
                    "storyText": row.get("storyText", "") or "",
                    "pageContent": row.get("pageContent", "") or "",
                    "suggestedReferences": row.get("suggestedReferences", []) or [],
                }
            )

        # (A) Neighbor window ±neighbor_radius: với mỗi target lấy N beat trước/sau nếu CHƯA nằm
        # trong batch. Chỉ giữ pageNumber + narrativePlane + storyText/pageContent (truncated)
        # — KHÔNG yêu cầu motionPrompt. Cửa sổ rộng hơn giúp nhịp section + nhãn vai ổn định.
        neighbor_items: list[dict[str, Any]] = []
        seen_neighbor_pages: set[Any] = set()
        for tgt_idx, _ in chunk:
            for off in range(-neighbor_radius, neighbor_radius + 1):
                if off == 0:
                    continue
                n_idx = tgt_idx + off
                if n_idx in batch_row_indices:
                    continue
                summary = _neighbor_summary(n_idx)
                if summary is None:
                    continue
                pn_key = summary.get("pageNumber")
                if pn_key in seen_neighbor_pages:
                    continue
                seen_neighbor_pages.add(pn_key)
                neighbor_items.append(summary)
        # Sắp xếp theo pageNumber để LLM đọc tuần tự.
        neighbor_items.sort(key=lambda x: x.get("pageNumber") if isinstance(x.get("pageNumber"), int) else 0)

        if neighbor_items:
            neighbor_block = (
                f"ADJACENT CONTEXT BEATS — up to {neighbor_radius} beats before and {neighbor_radius} "
                "beats after each target (read-only — for tonal / adjacency awareness only; DO NOT "
                "generate motionPrompt for these, they are NOT in the batch). Use them to understand "
                f"the local motion arc across a ~{max(1, neighbor_radius * 2 + 1)}-beat window and to "
                "keep on-screen role labels consistent with nearby pageContent / WHO IS WHERE:\n"
                + json.dumps(neighbor_items, ensure_ascii=False, indent=2)
                + "\n\n"
            )
        else:
            neighbor_block = ""

        prompt = f"""\
You are filling in the MISSING field `motionPrompt` (WAN 2.1 image-to-video prompt) for a batch
of already-planned story beats. Each beat already has a fixed `pageContent` (a static visual
description of the still image) and `storyText` (the narration TTS). You only produce `motionPrompt`.

Write motion that is READABLE on screen in ~5s: chained actions, clear facial expression, and
reciprocal interaction when multiple roles share the frame — match the energy of strong I2V
reference prompts (clear gestures, eye-lock, strain/pull, clenched jaw, sharp smirk), not vague
ambient-only drift. Include ONE continuous camera path with visible zoom/dolly and/or angle/height
drift when pageContent framing allows (no jump cuts).
Label every mover from pageContent WHO IS WHERE + visible on-screen archetype for THAT beat's genre
and setting — never kinship from storyText alone (father, mother, sister, a tỷ) and never
source-language proper names.

Each batch item also carries `outlineSectionSummary` — the section-level tonal arc the beat belongs
to. Use it to keep motion intensity / mood consistent with the section's emotional register
(e.g. climactic section → stronger kinetic verbs; reflective section → quieter body anchors).

{_WAN_MOTION_PROMPT_RULES}

{neighbor_block}BATCH BEATS (generate motionPrompt for EACH of these — read all fields for context, especially pageContent + narrativePlane + outlineSectionSummary):
{json.dumps(target_items, ensure_ascii=False, indent=2)}

Task: For EACH batch beat (NOT for neighbor context beats), produce ONE WAN 2.1 motion prompt that animates THAT beat's still image.
Where adjacent context is provided (up to {neighbor_radius} beats before and {neighbor_radius} beats after the batch), make the motion of each batch beat fit naturally into the local arc:
  • Read the preceding context beats (if any) to understand the kinetic state the character is COMING FROM (e.g. running → slowing; weeping → recovering; tense silence → breaking).
  • Read the following context beats (if any) to understand the kinetic state the character is HEADING INTO (e.g. about to stand up → end with weight shift; about to flinch → end with held breath).
  • Reuse the same on-screen role labels already established in nearby pageContent / WHO IS WHERE (do not fall back to kinship terms from storyText).
  • Match breath cadence, head orientation, hand position, and energy level so consecutive clips can be cut/lipsynced together without jarring discontinuity.

Output MUST be a JSON array of the SAME length and SAME order as the BATCH BEATS array, where each object has EXACTLY two keys:
  - "pageNumber" (int, matching the input pageNumber)
  - "motionPrompt" (non-empty string, 70-120 words, following ALL rules above)

Return ONLY the JSON array. No markdown fences. No commentary. No prose. No keys other than pageNumber + motionPrompt.
"""
        first_pn_for_log = target_items[0].get("pageNumber")

        def _call_backfill() -> Any:
            return client.models.generate_content(
                model=planner_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    response_mime_type="application/json",
                ),
            )

        try:
            resp = _generate_with_transient_retry(
                _call_backfill,
                label=f"backfill motionPrompt batch {batch_idx}/{n_total_batches} (start pageNumber={first_pn_for_log})",
            )
        except Exception as exc:  # noqa: BLE001
            # Non-transient (vd. quota / 400) → re-raise đã handle, ở đây bắt lỗi ngoài.
            print(
                f"  [WARN] batch bắt đầu pageNumber={first_pn_for_log} lỗi không-transient: {exc}; bỏ qua batch.",
                file=sys.stderr,
            )
            continue
        if resp is None:
            # Đã retry hết delays nhưng vẫn fail transient → skip, beat trong batch sẽ được retry
            # ở lần chạy resume kế tiếp (vẫn nằm trong targets vì chưa có motionPrompt).
            print(
                f"  [WARN] batch bắt đầu pageNumber={first_pn_for_log}: fail sau toàn bộ retry; "
                "bỏ qua, lần resume kế tiếp sẽ thử lại tự động.",
                file=sys.stderr,
            )
            continue

        text = _strip_json_fence(getattr(resp, "text", None) or "")
        try:
            parsed = _json_loads_llm(text)
        except Exception as exc:  # noqa: BLE001
            first_pn = target_items[0].get("pageNumber")
            print(
                f"  [WARN] batch bắt đầu pageNumber={first_pn} không parse được JSON ({exc}); bỏ qua batch.",
                file=sys.stderr,
            )
            continue

        if not isinstance(parsed, list):
            first_pn = target_items[0].get("pageNumber")
            print(
                f"  [WARN] batch bắt đầu pageNumber={first_pn}: planner trả không phải array; bỏ qua.",
                file=sys.stderr,
            )
            continue

        by_page: dict[int, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            pn_raw = item.get("pageNumber")
            mp_raw = item.get("motionPrompt")
            if pn_raw is None or not isinstance(mp_raw, str):
                continue
            try:
                by_page[int(pn_raw)] = mp_raw.strip()
            except (TypeError, ValueError):
                continue

        batch_filled = 0
        for _, row in chunk:
            pn_val = row.get("pageNumber")
            try:
                key = int(pn_val) if pn_val is not None else None
            except (TypeError, ValueError):
                key = None
            mp = by_page.get(key, "") if key is not None else ""
            if mp:
                row["motionPrompt"] = mp
                n_filled += 1
                batch_filled += 1
                preview = re.sub(r"\s+", " ", mp)[:90]
                print(f"  ✓ beat {pn_val}: {preview}…", file=sys.stderr)
            else:
                print(
                    f"  [WARN] beat {pn_val}: planner không trả motionPrompt; giữ nguyên (sẽ thử lại ở lần sau).",
                    file=sys.stderr,
                )

        # Persist sau MỖI batch để khỏi mất tiến độ nếu crash. Resume tự nhiên: lần chạy tiếp
        # theo sẽ thấy beat đã có motionPrompt → skip (logic ở đầu hàm). Chỉ ghi khi batch có
        # thay đổi thực để giảm I/O.
        if batch_filled > 0:
            try:
                _atomic_write_json(scenes_path, rows)
                print(
                    f"  [checkpoint] batch {batch_idx}/{n_total_batches}: lưu {n_filled}/{len(targets)} "
                    f"motionPrompt vào {scenes_path.name} (resume-safe).",
                    file=sys.stderr,
                )
                # Sync ngược vào review_two_pass.json để lần planner-resume sau không xoá mất.
                if review_checkpoint_path is not None:
                    _sync_motion_prompts_to_review_checkpoint(
                        scenes_path=scenes_path,
                        checkpoint_path=review_checkpoint_path,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [WARN] batch {batch_idx}/{n_total_batches}: lưu checkpoint thất bại ({exc}); "
                    "tiến độ trong memory chưa bị mất, vẫn tiếp tục.",
                    file=sys.stderr,
                )

    # Cuối cùng: đảm bảo file đã sync (nếu batch cuối không có thay đổi vẫn cần flush một lần).
    _atomic_write_json(scenes_path, rows)
    # Final sync vào checkpoint (no-op nếu đã sync đủ ở các batch trước).
    if review_checkpoint_path is not None:
        _sync_motion_prompts_to_review_checkpoint(
            scenes_path=scenes_path,
            checkpoint_path=review_checkpoint_path,
        )
    print(
        f"Hoàn tất: ghi {n_filled}/{len(targets)} motionPrompt mới vào {scenes_path}.",
        file=sys.stderr,
    )
    return n_filled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tách truyện thành cảnh + sinh ảnh (Gemini), consistency nhân vật qua reference."
    )
    parser.add_argument("--story", type=Path, help="File .txt chứa toàn bộ truyện")
    parser.add_argument(
        "--story-stdin",
        action="store_true",
        help="Đọc truyện từ stdin thay vì --story",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Dùng Vertex AI thay cho Gemini Developer API (AI Studio). Cần project + region "
        "(--gcp-project / --gcp-location hoặc GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION). "
        "Xác thực: gcloud auth application-default login hoặc GOOGLE_APPLICATION_CREDENTIALS (service account).",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        metavar="ID",
        help="GCP project id khi --vertex (mặc định: biến GOOGLE_CLOUD_PROJECT).",
    )
    parser.add_argument(
        "--gcp-location",
        type=str,
        default=None,
        metavar="REGION",
        help="Vertex region, ví dụ us-central1, asia-southeast1 (mặc định: GOOGLE_CLOUD_LOCATION hoặc us-central1).",
    )
    parser.add_argument(
        "--mode",
        choices=("storybook", "manga", "review"),
        default="storybook",
        help="manga: trang full nhiều panel. review: nhiều ảnh minh họa nhỏ liên tiếp (đọc/review truyện). storybook: một minh họa mỗi đoạn lớn.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Chỉ chạy planner và ghi scenes.json (không sinh ảnh).",
    )
    parser.add_argument(
        "--from-scenes",
        type=Path,
        default=None,
        metavar="PATH",
        help="Đọc plan có sẵn (thường là scenes.json), bỏ qua planner. Vẫn cần --story làm ngữ cảnh khi sinh ảnh.",
    )
    parser.add_argument(
        "--save-plan",
        action="store_true",
        help="Khi dùng --from-scenes: vẫn ghi đè scenes.json trong -o (mặc định không ghi để giữ chỉnh sửa tay).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua pageNumber nếu đã có scene_XX.png/jpg/webp trong thư mục -o.",
    )
    parser.add_argument(
        "--review-two-pass",
        action="store_true",
        help="Chỉ với --mode review: planner 2 lần — dàn ý toàn truyện rồi tách beat theo từng đoạn (JSON ngắn hơn, ít bị cắt cụt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua portrait nhân vật đã có (auto_refs/NN_*); bỏ qua scene_XX đã render (như --skip-existing).",
    )
    parser.add_argument(
        "--fresh-checkpoint",
        action="store_true",
        help="Chỉ với --review-two-pass: xóa checkpoints/review_two_pass.json trước khi plan (chạy lại từ đầu).",
    )
    parser.add_argument(
        "--section-image-continuity",
        action="store_true",
        help="Mode review: ảnh beat present vừa trước (cùng outlineSectionNumber) được gửi kèm API để đồng bộ bối cảnh/trang phục. Cần scenes.json có outlineSectionNumber.",
    )
    parser.add_argument(
        "--continuity-across-sections",
        action="store_true",
        help="Bật cùng --section-image-continuity: giữ anchor + previous-beat continuity XUYÊN qua boundary outline section nếu cả hai beat đều là present (chỉ reset khi gặp flashback). Hữu ích khi planner cắt nhầm 1 mạch hành động liên tục thành 2 section.",
    )
    parser.add_argument(
        "--section-keyframe",
        action="store_true",
        help="Mode review: với mỗi outline section sinh 1 ảnh 'environment establishing still' (CHỈ bối cảnh / props / ánh sáng, KHÔNG vẽ nhân vật) + per-character wardrobe sheets. Beat đầu của section dùng environment làm anchor; các beat sau dùng beat-trước làm anchor.",
    )
    parser.add_argument(
        "--regenerate-keyframes",
        action="store_true",
        help="Buộc render lại section keyframes ngay cả khi đã tồn tại trong section_keyframes/.",
    )
    parser.add_argument(
        "--section-character-refs",
        dest="section_character_refs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mặc định BẬT khi đi cùng --section-keyframe: với mỗi section sinh thêm 1 'wardrobe sheet' cho từng nhân vật xuất hiện. Dùng --no-section-character-refs để tắt.",
    )
    parser.add_argument(
        "--regenerate-section-designs",
        action="store_true",
        help="Plan lại section_designs.json (Wardrobe Bible) cho mỗi section dù đã có file. Không xoá ảnh keyframes/scenes.",
    )
    parser.add_argument(
        "--fix-content",
        dest="fix_content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trước khi planning/render, gọi LLM sửa CHÍNH TẢ tiếng Việt và phiên âm các từ tiếng nước ngoài "
        "sang cách đọc THUẦN VIỆT (không đổi nội dung). Lưu ra <output>/content_fix.txt và dùng làm bản truyện "
        "nguồn cho mọi bước sau (planner / TTS / render). Mặc định BẬT; tắt bằng --no-fix-content.",
    )
    parser.add_argument(
        "--regenerate-content-fix",
        action="store_true",
        help="Buộc chạy lại bước fix nội dung kể cả khi <output>/content_fix.txt đã có và khớp source_fp.",
    )
    parser.add_argument(
        "--fix-content-chunk-chars",
        type=int,
        default=6000,
        metavar="N",
        help="Truyện dài hơn N ký tự sẽ được CHIA thành các chunk ~N ký tự (cắt tại biên đoạn / câu) "
        "để fix tuần tự, retry độc lập từng chunk. Mặc định 6000. Giảm nếu hay 503/timeout.",
    )
    parser.add_argument(
        "--no-character-portraits",
        action="store_true",
        help="Với --auto-characters: chỉ trích/refine danh sách nhân vật (JSON), không sinh ảnh portrait reference.",
    )
    parser.add_argument(
        "--text-density",
        choices=("minimal", "dialog", "dialog_fx", "dialog_fx_narration", "full"),
        default="dialog_fx",
        help="Hiện không dùng: trang manga luôn không chữ trên ảnh (đọc truyện ghép bằng TTS sau).",
    )
    parser.add_argument(
        "--auto-characters",
        action="store_true",
        help="Tự đọc truyện, liệt kê nhân vật (JSON), sinh ảnh reference vào output/auto_refs/, "
        "rồi dùng làm ref cho toàn bộ cảnh. Không cần --char.",
    )
    parser.add_argument(
        "--max-auto-characters",
        type=int,
        default=16,
        metavar="N",
        help="Giới hạn số nhân vật khi --auto-characters (tránh quá nhiều lần gọi API ảnh).",
    )
    parser.add_argument(
        "--no-refine-characters",
        action="store_true",
        help="Với --auto-characters: bỏ pass 2 refine (tiết kiệm 1 lần gọi text API; độ chính xác thường giảm).",
    )
    parser.add_argument(
        "--regenerate-characters",
        action="store_true",
        help=(
            "Với --auto-characters: ÉP trích lại nhân vật từ truyện kể cả khi "
            "<output>/auto_refs/characters_extracted.json đã có. Mặc định: nếu file đã có ≥1 nhân vật "
            "hợp lệ thì bỏ hẳn pass 1 (extract) + pass 2 (refine) để tiết kiệm thời gian/chi phí."
        ),
    )
    parser.add_argument(
        "--char",
        action="append",
        default=[],
        metavar="Tên=path.png",
        help="Ảnh nhân vật có sẵn (lặp lại cho nhiều người). Khi dùng kèm --auto-characters, "
        "ảnh bạn chỉ định ghi đè nhân vật trùng tên.",
    )
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        type=Path,
        help="Ảnh style (lặp lại được). Tùy chọn.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("story_output"),
        help="Thư mục lưu scenes.json và scene_XX.png",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,
        metavar="N",
        help="Số cảnh cố định (optional)",
    )
    parser.add_argument(
        "--planner-model",
        default=DEFAULT_PLANNER_MODEL,
        help=f"Mặc định: {DEFAULT_PLANNER_MODEL}",
    )
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Mặc định: {DEFAULT_IMAGE_MODEL}",
    )
    parser.add_argument(
        "--image-min-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Giãn cách tối thiểu (giây) giữa 2 lần gọi image model, để tránh 429 "
            "RESOURCE_EXHAUSTED trên Vertex. Mặc định: 8s khi --vertex, 0s khi dùng AI Studio. "
            "Đặt 0 để tắt throttle."
        ),
    )
    parser.add_argument(
        "--image-retry-delays",
        type=str,
        default=None,
        metavar="S1,S2,...",
        help=(
            "Backoff cho image model khi gặp 429/503 (giây, comma-separated). "
            "Mặc định: '30,60,120,240,480' khi --vertex, '8,20,45,90,180' khi AI Studio."
        ),
    )
    parser.add_argument(
        "--color",
        choices=("color", "bw"),
        default="color",
        help="Màu hoặc đen trắng",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=("square", "portrait", "landscape", "16:9"),
        default="portrait",
    )
    parser.add_argument(
        "--backfill-motion-prompts",
        action="store_true",
        help=(
            "Đọc <output>/scenes.json đã có, với mỗi beat thiếu/empty motionPrompt thì gọi planner "
            "sinh prompt WAN 2.1 I2V rồi ghi đè scenes.json. Không cần --story. "
            "Dùng khi planner trước đó (Gemini 2.5-flash) bỏ qua field motionPrompt, để khỏi re-run "
            "toàn bộ planner / render lại ảnh."
        ),
    )
    parser.add_argument(
        "--force-backfill-motion-prompts",
        action="store_true",
        help="Đi kèm --backfill-motion-prompts: ép overwrite cả những beat đã có motionPrompt.",
    )
    parser.add_argument(
        "--backfill-motion-pages",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Tùy chọn: chỉ backfill motionPrompt cho các pageNumber trong scenes.json. "
            "Định dạng: 1,3,5-8,12. Không chỉ rõ thì xử lý toàn bộ beat (hoặc mọi beat thiếu "
            "khi auto-backfill). Dùng với --backfill-motion-prompts."
        ),
    )
    parser.add_argument(
        "--backfill-batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Số beat / batch khi backfill motionPrompt (mặc định 10). Giảm nếu hay 429/timeout.",
    )
    parser.add_argument(
        "--backfill-motion-neighbor-beats",
        type=int,
        default=_MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS,
        metavar="N",
        help=(
            "Backfill motionPrompt: số beat lân cận mỗi phía (±N) đưa vào ngữ cảnh read-only "
            f"(pageContent/storyText rút gọn). Mặc định {_MOTION_BACKFILL_DEFAULT_NEIGHBOR_RADIUS}; "
            f"tối đa {_MOTION_BACKFILL_MAX_NEIGHBOR_RADIUS}. Tăng (vd. 10) giúp nhịp section và "
            "nhãn vai ổn định hơn nhưng tốn token — có thể cần giảm --backfill-batch-size."
        ),
    )
    parser.add_argument(
        "--auto-split-long-beats",
        dest="auto_split_long_beats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mode review + resume checkpoint: trước khi planner chạy tiếp, tự tìm beat có storyText "
            "dài hơn --split-long-beat-sec, tách theo nhịp hình ảnh (context ±2 beat), đánh số lại "
            "pageNumber, ghi checkpoint/scenes.json, rồi backfill motionPrompt. "
            "Tắt bằng --no-auto-split-long-beats."
        ),
    )
    parser.add_argument(
        "--split-long-beat-sec",
        type=float,
        default=_LONG_BEAT_AUTO_SPLIT_SEC,
        metavar="SECONDS",
        help="Ngưỡng TTS (giây) để auto-split beat dài khi resume (mặc định 10).",
    )
    parser.add_argument(
        "--auto-backfill-motion-prompts",
        dest="auto_backfill_motion_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mode review: sau khi planner ghi scenes.json, nếu beat nào thiếu motionPrompt "
            "thì TỰ động gọi backfill ngay (cùng cơ chế với --backfill-motion-prompts). "
            "Mặc định BẬT vì Gemini 2.5-flash thường drop field này. "
            "Tắt bằng --no-auto-backfill-motion-prompts."
        ),
    )
    parser.add_argument(
        "--art-style",
        choices=(
            "watercolor",
            "oil_painting",
            "digital_illustration",
            "anime",
            "storybook_classic",
            "realistic",
            "ghibli",
        ),
        default="storybook_classic",
        help=(
            "ghibli = Studio Ghibli style (hand-drawn anime, watercolor-painted backgrounds, "
            "Miyazaki-esque mood). Áp dụng cho cả portrait nhân vật và scene/beat."
        ),
    )

    args = parser.parse_args()
    use_vertex = bool(getattr(args, "vertex", False))
    if use_vertex:
        project = (args.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
        location = (
            args.gcp_location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or "us-central1"
        ).strip()
        if not project:
            parser.error(
                "Chế độ --vertex cần --gcp-project hoặc biến môi trường GOOGLE_CLOUD_PROJECT."
            )
        client = genai.Client(vertexai=True, project=project, location=location)
        print(
            f"Vertex AI: project={project!r}, location={location!r} (ADC / service account).",
            file=sys.stderr,
        )
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Thiếu GOOGLE_API_KEY (hoặc dùng --vertex với GCP).", file=sys.stderr)
            raise SystemExit(1)
        client = genai.Client(api_key=api_key)

    if args.image_min_interval is not None:
        _img_min_interval = float(args.image_min_interval)
    else:
        _img_min_interval = 8.0 if use_vertex else 0.0
    if args.image_retry_delays is not None:
        try:
            _img_delays = tuple(
                int(x) for x in str(args.image_retry_delays).split(",") if x.strip()
            )
            if not _img_delays:
                raise ValueError("empty")
        except Exception:
            parser.error(
                "Tham số --image-retry-delays phải là chuỗi số nguyên cách nhau bởi dấu phẩy, "
                "ví dụ '30,60,120,240,480'."
            )
    else:
        _img_delays = (30, 60, 120, 240, 480) if use_vertex else (8, 20, 45, 90, 180)
    _configure_image_pacing(min_interval_s=_img_min_interval, retry_delays=_img_delays)
    print(
        f"Image pacing: min_interval={_img_min_interval}s, "
        f"retry_delays={list(_img_delays)} (vertex={use_vertex}).",
        file=sys.stderr,
    )

    try:
        backfill_only_page_numbers = _parse_optional_backfill_page_numbers(
            getattr(args, "backfill_motion_pages", None)
        )
    except ValueError as exc:
        parser.error(f"--backfill-motion-pages: {exc}")

    backfill_neighbor_radius = _motion_backfill_neighbor_radius_from_args(args)

    # Backfill motionPrompt cho scenes.json đã có — chạy nhanh, không cần --story.
    if getattr(args, "backfill_motion_prompts", False):
        out_dir_bf = args.output.expanduser().resolve()
        scenes_path_bf = out_dir_bf / "scenes.json"
        if not scenes_path_bf.is_file():
            parser.error(
                f"--backfill-motion-prompts cần {scenes_path_bf} đã tồn tại. "
                "Chạy planner trước, hoặc trỏ -o đúng thư mục output."
            )
        ck_for_bf = out_dir_bf / "checkpoints" / "review_two_pass.json"
        backfill_motion_prompts_in_scenes_json(
            client=client,
            scenes_path=scenes_path_bf,
            planner_model=args.planner_model,
            batch_size=int(getattr(args, "backfill_batch_size", 10) or 10),
            force=bool(getattr(args, "force_backfill_motion_prompts", False)),
            review_checkpoint_path=ck_for_bf if ck_for_bf.is_file() else None,
            only_page_numbers=backfill_only_page_numbers,
            neighbor_radius=backfill_neighbor_radius,
        )
        return

    if args.story_stdin:
        story = sys.stdin.read()
    elif args.story:
        story = args.story.read_text(encoding="utf-8")
    else:
        parser.error("Cần --story hoặc --story-stdin")

    if args.plan_only and args.from_scenes:
        parser.error("Không dùng --plan-only cùng --from-scenes.")
    if args.review_two_pass and args.mode != "review":
        parser.error("--review-two-pass chỉ dùng với --mode review.")
    if args.from_scenes and args.review_two_pass:
        parser.error("--review-two-pass không dùng cùng --from-scenes.")
    if args.fresh_checkpoint and not args.review_two_pass:
        parser.error("--fresh-checkpoint chỉ dùng với --review-two-pass.")

    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    review_ck = out_dir / "checkpoints" / "review_two_pass.json"
    review_outline_sidecar = out_dir / "checkpoints" / "review_outline.json"
    if args.fresh_checkpoint:
        if review_ck.is_file():
            review_ck.unlink()
            print(
                f"Đã xóa checkpoint review 2 bước: {review_ck}",
                file=sys.stderr,
            )
        if review_outline_sidecar.is_file():
            review_outline_sidecar.unlink()
            print(
                f"Đã xóa dàn ý tạm: {review_outline_sidecar}",
                file=sys.stderr,
            )

    if (
        args.mode == "review"
        and args.review_two_pass
        and bool(getattr(args, "auto_split_long_beats", True))
        and review_ck.is_file()
        and not args.fresh_checkpoint
    ):
        try:
            n_split = split_long_review_beats_in_checkpoint(
                client=client,
                checkpoint_path=review_ck,
                manifest_path=out_dir / "scenes.json",
                planner_model=args.planner_model,
                tts_sec_threshold=float(
                    getattr(args, "split_long_beat_sec", _LONG_BEAT_AUTO_SPLIT_SEC)
                ),
            )
            if n_split > 0:
                print(
                    f"Auto-split beat dài: đã tách {n_split} beat gốc (> "
                    f"{float(getattr(args, 'split_long_beat_sec', _LONG_BEAT_AUTO_SPLIT_SEC)):.0f}s TTS).",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [WARN] auto-split beat dài thất bại ({exc}); planner vẫn tiếp tục.",
                file=sys.stderr,
            )

    # Pre-resume backfill: nếu checkpoint review_two_pass đã có sẵn N beat (vd. đang dở 10/19 section),
    # fill motionPrompt cho N beat đó NGAY trước khi planner chạy tiếp các section còn lại. Nếu lần
    # planner kế tiếp crash, 399 beat đã có vẫn an toàn (motionPrompt đã lưu cả ở scenes.json lẫn
    # checkpoint nhờ cơ chế sync). Cũng giảm gánh nặng cho auto-backfill cuối main(): chỉ phải xử lý
    # số beat planner mới sinh, không phải toàn bộ truyện.
    if (
        args.mode == "review"
        and args.review_two_pass
        and bool(getattr(args, "auto_backfill_motion_prompts", True))
        and review_ck.is_file()
        and not args.fresh_checkpoint
    ):
        try:
            _ck_raw = json.loads(review_ck.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _ck_raw = None
        _ck_scenes = (
            _ck_raw.get("scenes") if isinstance(_ck_raw, dict) else None
        ) or []
        if isinstance(_ck_scenes, list) and _ck_scenes:
            _n_missing = sum(
                1
                for r in _ck_scenes
                if isinstance(r, dict) and not (r.get("motionPrompt") or "").strip()
            )
            if _n_missing > 0:
                print(
                    f"Pre-resume backfill: checkpoint hiện có {len(_ck_scenes)} beat, "
                    f"{_n_missing} thiếu motionPrompt. Fill ngay TRƯỚC khi planner chạy tiếp "
                    "các section còn lại (an toàn nếu crash giữa chừng)...",
                    file=sys.stderr,
                )
                _scenes_pre_path = out_dir / "scenes.json"
                # Mirror checkpoint.scenes → scenes.json để backfill có file để đọc/ghi.
                # Planner sẽ overwrite lại scenes.json sau mỗi section nhưng đã giữ motionPrompt
                # nhờ _manifest_dict_to_scene → scene_to_manifest_dict bảo toàn field này.
                try:
                    _atomic_write_json(_scenes_pre_path, _ck_scenes)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  [WARN] không ghi được {_scenes_pre_path} ({exc}); bỏ qua pre-resume backfill.",
                        file=sys.stderr,
                    )
                else:
                    try:
                        backfill_motion_prompts_in_scenes_json(
                            client=client,
                            scenes_path=_scenes_pre_path,
                            planner_model=args.planner_model,
                            batch_size=int(getattr(args, "backfill_batch_size", 10) or 10),
                            force=False,
                            review_checkpoint_path=review_ck,
                            only_page_numbers=backfill_only_page_numbers,
                            neighbor_radius=backfill_neighbor_radius,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  [WARN] pre-resume backfill thất bại ({exc}); planner vẫn tiếp tục bình thường.",
                            file=sys.stderr,
                        )

    manual_char_paths = _parse_char_args(args.char)

    if args.fix_content:
        story = _normalize_story_content(
            client,
            story=story,
            out_dir=out_dir,
            planner_model=args.planner_model,
            regenerate=bool(getattr(args, "regenerate_content_fix", False)),
            chunk_chars=int(getattr(args, "fix_content_chunk_chars", 6000)),
        )
    else:
        print(
            "Bỏ qua bước fix chính tả/phiên âm (--no-fix-content). Dùng truyện gốc cho mọi bước sau.",
            file=sys.stderr,
        )

    if args.auto_characters:
        print("Chế độ --auto-characters: trích nhân vật + sinh ảnh reference...", file=sys.stderr)
        gen_portraits = (not args.no_character_portraits) and (not args.plan_only)
        if args.plan_only and not args.no_character_portraits:
            print(
                "Ghi chú: --plan-only sẽ tự tắt sinh ảnh portrait nhân vật để tiết kiệm thời gian/chi phí. "
                "Dùng lại không có --plan-only để sinh portrait khi render ảnh.",
                file=sys.stderr,
            )
        auto_paths, _extracted_rows = auto_build_character_refs(
            client,
            story,
            out_dir=out_dir,
            planner_model=args.planner_model,
            image_model=args.image_model,
            color_mode=args.color,
            art_style=args.art_style,
            max_characters=args.max_auto_characters,
            refine_characters=not args.no_refine_characters,
            generate_portraits=gen_portraits,
            resume=args.resume,
            regenerate=bool(getattr(args, "regenerate_characters", False)),
        )
        _configure_character_aliases(extracted_rows=_extracted_rows)
        if _CHARACTER_ALIASES_MAP:
            n_chars = sum(
                1
                for k in _CHARACTER_ALIASES_MAP
                if not k.endswith((".png", ".jpg", ".jpeg", ".webp")) and "_" not in k
            )
            print(
                f"Character aliases: nạp ~{n_chars} nhân vật vào map "
                "(dùng cho check co-presence trong pageContent).",
                file=sys.stderr,
            )
        char_paths = {**auto_paths, **manual_char_paths}
        if not char_paths:
            print(
                "Cảnh báo: không trích được nhân vật và không có --char; cảnh sẽ vẽ không có ref ảnh.",
                file=sys.stderr,
            )
    else:
        char_paths = dict(manual_char_paths)
        if not char_paths:
            print(
                "Cảnh báo: không có --char; consistency nhân vật sẽ kém. Dùng --auto-characters để tự trích + sinh ref.",
                file=sys.stderr,
            )

    if not _CHARACTER_ALIASES_MAP:
        _configure_character_aliases(out_dir=out_dir)
        if _CHARACTER_ALIASES_MAP:
            print(
                f"Character aliases: nạp từ {out_dir}/auto_refs/characters_extracted.json "
                "(dùng cho check co-presence trong pageContent).",
                file=sys.stderr,
            )

    style_paths = []
    for sp in args.style:
        r = sp.expanduser().resolve()
        if not r.is_file():
            raise SystemExit(f"Không tìm thấy ảnh style: {r}")
        style_paths.append(r)

    render_mode = args.mode
    if args.from_scenes:
        plan_path = args.from_scenes.expanduser().resolve()
        if not plan_path.is_file():
            raise SystemExit(f"Không tìm thấy file plan: {plan_path}")
        try:
            plan_raw = plan_path.read_text(encoding="utf-8")
            plan_data = json.loads(plan_raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON plan không hợp lệ ({plan_path}): {exc}") from exc
        if not isinstance(plan_data, list):
            raise SystemExit(f"Plan phải là mảng JSON: {plan_path}")
        render_mode = _infer_render_mode_from_manifest(plan_data, args.mode)
        if render_mode != args.mode:
            print(
                f"Ghi chú: sinh ảnh theo mode={render_mode!r} lấy từ plan "
                f"(khác --mode {args.mode!r}).",
                file=sys.stderr,
            )
        scenes = _manifest_rows_to_scenes(plan_data, story, render_mode)
        print(
            f"Đã load {len(scenes)} cảnh từ {plan_path} (bỏ qua planner).",
            file=sys.stderr,
        )
        if render_mode == "review":
            _propagate_copresent_cast_in_scenes(scenes)
    else:
        _plan_label = {
            "manga": "trang manga",
            "review": "các beat minh họa (review)",
            "storybook": "cảnh storybook",
        }[args.mode]
        if args.mode == "review" and args.review_two_pass:
            _plan_label = "các beat minh họa (review, planner 2 bước)"
        print(f"Đang tách truyện thành {_plan_label}...", file=sys.stderr)
        bible_inline_enabled = bool(
            args.review_two_pass
            and args.mode == "review"
            and getattr(args, "section_image_continuity", False)
            and getattr(args, "section_keyframe", False)
        )
        if bible_inline_enabled:
            print(
                "Bible xen kẽ: BẬT — section_designs.json sẽ cập nhật sau MỖI section beats xong.",
                file=sys.stderr,
            )
        scenes = split_story_into_scenes(
            client,
            story,
            char_paths=char_paths,
            style_paths=style_paths,
            scene_count=args.scenes,
            planner_model=args.planner_model,
            app_mode=args.mode,
            review_two_pass=args.review_two_pass,
            review_checkpoint_path=review_ck if args.review_two_pass else None,
            plan_manifest_path=out_dir / "scenes.json" if args.review_two_pass else None,
            bible_inline=bible_inline_enabled,
            bible_path=_bible_path_for_out_dir(out_dir) if bible_inline_enabled else None,
            bible_regenerate=bool(getattr(args, "regenerate_section_designs", False)),
        )

    manifest_path = out_dir / "scenes.json"
    if args.from_scenes and not args.save_plan:
        print(
            f"Không ghi đè {manifest_path} (dùng --save-plan nếu muốn cập nhật).",
            file=sys.stderr,
        )
    else:
        _atomic_write_json(
            manifest_path,
            [scene_to_manifest_dict(s, render_mode) for s in scenes],
        )
        print(f"Đã ghi {manifest_path} ({len(scenes)} cảnh).", file=sys.stderr)

    # Auto-backfill motionPrompt: Gemini 2.5-flash thường drop field này dù prompt đã ép.
    # Chạy LUÔN cho mode review — kể cả khi resume từ --from-scenes (backfill chỉ THÊM motionPrompt
    # cho beat đang rỗng, KHÔNG ghi đè field có sẵn nên an toàn). Tắt bằng --no-auto-backfill-motion-prompts.
    # Target path:
    #   - Mặc định: out_dir/scenes.json (planner vừa ghi).
    #   - Với --from-scenes mà KHÔNG có --save-plan: out_dir/scenes.json chưa được ghi,
    #     ta backfill thẳng vào file plan nguồn (args.from_scenes) — đây là file user đang đọc.
    auto_bf_enabled = (
        render_mode == "review"
        and bool(getattr(args, "auto_backfill_motion_prompts", True))
    )
    if auto_bf_enabled:
        if args.from_scenes and not args.save_plan:
            bf_target_path = args.from_scenes.expanduser().resolve()
            bf_target_label = f"file plan nguồn {bf_target_path}"
        else:
            bf_target_path = manifest_path
            bf_target_label = str(manifest_path)

        if not bf_target_path.is_file():
            print(
                f"Auto-backfill motionPrompt: bỏ qua — không tìm thấy {bf_target_path}.",
                file=sys.stderr,
            )
        else:
            n_missing = sum(
                1 for s in scenes if not (getattr(s, "motion_prompt", "") or "").strip()
            )
            if n_missing > 0:
                print(
                    f"Auto-backfill motionPrompt: {n_missing}/{len(scenes)} beat thiếu — gọi backfill "
                    f"ngay vào {bf_target_label} (tắt: --no-auto-backfill-motion-prompts).",
                    file=sys.stderr,
                )
                try:
                    backfill_motion_prompts_in_scenes_json(
                        client=client,
                        scenes_path=bf_target_path,
                        planner_model=args.planner_model,
                        batch_size=int(getattr(args, "backfill_batch_size", 10) or 10),
                        force=False,
                        review_checkpoint_path=(
                            review_ck
                            if (bf_target_path == manifest_path and review_ck.is_file())
                            else None
                        ),
                        only_page_numbers=backfill_only_page_numbers,
                        neighbor_radius=backfill_neighbor_radius,
                    )
                    # Cập nhật motion_prompt vào scenes in-memory (phòng bước sau cần dùng).
                    try:
                        reloaded = json.loads(bf_target_path.read_text(encoding="utf-8"))
                    except Exception:
                        reloaded = None
                    if isinstance(reloaded, list):
                        pn_to_mp: dict[int, str] = {}
                        for r in reloaded:
                            if not isinstance(r, dict):
                                continue
                            pn = r.get("pageNumber")
                            mp = (r.get("motionPrompt") or "").strip()
                            if pn is None or not mp:
                                continue
                            try:
                                pn_to_mp[int(pn)] = mp
                            except (TypeError, ValueError):
                                continue
                        for s in scenes:
                            if (getattr(s, "motion_prompt", "") or "").strip():
                                continue
                            try:
                                v = pn_to_mp.get(int(s.page_number), "")
                            except (TypeError, ValueError):
                                v = ""
                            if v:
                                s.motion_prompt = v
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"Cảnh báo: auto-backfill motionPrompt thất bại ({exc}); tiếp tục.",
                        file=sys.stderr,
                    )

    if args.plan_only:
        # Khi đã yêu cầu --section-keyframe (cùng --section-image-continuity, mode review),
        # plan luôn Wardrobe Bible — thuần text, không tốn quota ảnh — đúng tinh thần --plan-only.
        if (
            render_mode == "review"
            and getattr(args, "section_image_continuity", False)
            and getattr(args, "section_keyframe", False)
        ):
            print(
                "Chế độ --plan-only + --section-keyframe: plan thêm Wardrobe Bible (text-only)...",
                file=sys.stderr,
            )
            try:
                _plan_section_designs(
                    client=client,
                    story=story,
                    scenes=scenes,
                    char_paths=char_paths,
                    planner_model=args.planner_model,
                    out_dir=out_dir,
                    regenerate=bool(getattr(args, "regenerate_section_designs", False)),
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [WARN] plan section_designs lỗi: {exc}; bỏ qua.",
                    file=sys.stderr,
                )
        print(
            "Chế độ --plan-only: đã dừng sau khi tạo plan (không render ảnh).",
            file=sys.stderr,
        )
        return

    ext_for_mime = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }

    section_image_continuity = args.section_image_continuity and render_mode == "review"
    if args.section_image_continuity and render_mode != "review":
        print(
            "Ghi chú: --section-image-continuity chỉ dùng với plan mode=review; đã bỏ qua.",
            file=sys.stderr,
        )
    cross_section = bool(getattr(args, "continuity_across_sections", False)) and section_image_continuity
    if getattr(args, "continuity_across_sections", False) and not section_image_continuity:
        print(
            "Ghi chú: --continuity-across-sections cần đi kèm --section-image-continuity (mode review); đã bỏ qua.",
            file=sys.stderr,
        )

    use_section_keyframe = (
        bool(getattr(args, "section_keyframe", False))
        and section_image_continuity
    )
    if bool(getattr(args, "section_keyframe", False)) and not section_image_continuity:
        print(
            "Ghi chú: --section-keyframe cần đi kèm --section-image-continuity (mode review); đã bỏ qua.",
            file=sys.stderr,
        )

    use_section_char_refs = (
        bool(getattr(args, "section_character_refs", False))
        and use_section_keyframe
    )
    if bool(getattr(args, "section_character_refs", False)) and not use_section_keyframe:
        print(
            "Ghi chú: --section-character-refs cần đi kèm --section-keyframe; đã bỏ qua.",
            file=sys.stderr,
        )

    section_keyframe_paths: dict[int, Path] = {}
    section_char_ref_paths: dict[int, dict[str, Path]] = {}
    section_designs: dict[int, dict[str, Any]] = {}
    if use_section_keyframe:
        section_designs = _plan_section_designs(
            client=client,
            story=story,
            scenes=scenes,
            char_paths=char_paths,
            planner_model=args.planner_model,
            out_dir=out_dir,
            regenerate=bool(getattr(args, "regenerate_section_designs", False)),
        )
        # Bible đã sẵn sàng; render keyframes/sheets chuyển sang interleave just-in-time TRONG main loop
        # (xem khối "Just-in-time" bên dưới) để mỗi section: bible → wardrobe sheets → keyframe → scenes
        # rồi mới sang section kế. Nếu user chỉ muốn pre-render keyframes (không render scene), gọi
        # _render_section_keyframes(...) thủ công.

    last_present_path: Path | None = None
    last_present_scene: Scene | None = None
    # Per-section anchor (mặc định). Khi cross_section bật, dùng key "_run_<n>" theo run liên tục present
    # (run reset khi gặp flashback). Khi cross_section tắt, key = sec_n như cũ.
    section_anchor_paths: dict[object, Path] = {}
    present_run_id = 0  # tăng mỗi khi gặp flashback (reset run continuity)
    rendered_section_keyframes: set[int] = set()  # các section đã chạy _render_one_section_keyframes

    def _is_present(sc: Scene) -> bool:
        return (getattr(sc, "narrative_plane", None) or "present").strip().lower() == "present"

    def _anchor_key(sec_n_value: int | None) -> object:
        if cross_section:
            return f"_run_{present_run_id}"
        return sec_n_value

    def _ensure_section_keyframes(sec_n_value: int | None) -> None:
        """Just-in-time: trước scene đầu tiên của section, render bible→sheets→keyframe."""
        if not use_section_keyframe or sec_n_value is None:
            return
        sec_int = int(sec_n_value)
        if sec_int in rendered_section_keyframes:
            return
        rendered_section_keyframes.add(sec_int)
        sec_scenes_only = [
            s
            for s in scenes
            if getattr(s, "outline_section_number", None) is not None
            and int(s.outline_section_number) == sec_int
        ]
        print(
            f"== Section {sec_int:02d}: chuẩn bị wardrobe sheets + keyframe trước khi render scenes ==",
            file=sys.stderr,
        )
        kpath, sec_char_map = _render_one_section_keyframes(
            client=client,
            sec_num=sec_int,
            sec_scenes=sec_scenes_only,
            section_design=section_designs.get(sec_int),
            out_dir=out_dir,
            char_paths=char_paths,
            image_model=args.image_model,
            color_mode=args.color,
            aspect_ratio=args.aspect_ratio,
            art_style=args.art_style,
            ext_for_mime=ext_for_mime,
            regenerate=bool(getattr(args, "regenerate_keyframes", False)),
            render_character_refs=use_section_char_refs,
        )
        if kpath is not None:
            section_keyframe_paths[sec_int] = kpath
        if sec_char_map:
            section_char_ref_paths[sec_int] = sec_char_map

    for i, scene in enumerate(scenes):
        idx = scene.page_number
        label = (
            "trang"
            if render_mode == "manga"
            else ("beat" if render_mode == "review" else "cảnh")
        )
        sec_n = getattr(scene, "outline_section_number", None)
        scene_is_present = _is_present(scene)

        # Just-in-time: render keyframe + sheets cho section này (nếu chưa làm).
        _ensure_section_keyframes(sec_n)

        # Reset run continuity khi gặp flashback (cả 2 chế độ).
        if not scene_is_present:
            present_run_id += 1
            last_present_path = None
            last_present_scene = None

        if (args.skip_existing or args.resume) and _scene_output_image_exists(out_dir, idx):
            print(
                f"Bỏ qua {label} {idx} (đã có ảnh scene_{idx:02d}.*).",
                file=sys.stderr,
            )
            ex = _first_existing_scene_image(out_dir, idx)
            if section_image_continuity and ex is not None and scene_is_present:
                last_present_path = ex
                last_present_scene = scene
                key = _anchor_key(sec_n)
                if key is not None and key not in section_anchor_paths:
                    section_anchor_paths[key] = ex
            continue

        continuity_path: Path | None = None
        anchor_path: Path | None = None
        if section_image_continuity and scene_is_present:
            prev_sec = (
                getattr(last_present_scene, "outline_section_number", None)
                if last_present_scene
                else None
            )
            same_section = prev_sec is not None and sec_n is not None and prev_sec == sec_n
            if (
                last_present_path is not None
                and last_present_path.is_file()
                and (cross_section or same_section)
            ):
                continuity_path = last_present_path

            if use_section_keyframe and sec_n is not None:
                kf = section_keyframe_paths.get(sec_n)
                if kf is not None and kf.is_file():
                    anchor_path = kf

            if anchor_path is None:
                key = _anchor_key(sec_n)
                if key is not None:
                    anc = section_anchor_paths.get(key)
                    if anc is not None and anc.is_file():
                        anchor_path = anc

        # Nếu section có per-character wardrobe sheet → ưu tiên dùng thay cho portrait gốc.
        effective_char_paths = char_paths
        n_overrides = 0
        if use_section_char_refs and sec_n is not None:
            sec_overrides = section_char_ref_paths.get(int(sec_n)) or {}
            if sec_overrides:
                merged = dict(char_paths)
                for lbl, p in sec_overrides.items():
                    if p.is_file():
                        merged[lbl] = p
                        n_overrides += 1
                effective_char_paths = merged

        ref_for_scene = _pick_reference_paths(scene, effective_char_paths, style_paths)
        extras = []
        if anchor_path is not None:
            extras.append(f"anchor: {anchor_path.name}")
        if continuity_path is not None and (
            anchor_path is None or continuity_path.resolve() != anchor_path.resolve()
        ):
            extras.append(f"prev: {continuity_path.name}")
        if n_overrides:
            extras.append(f"section-char-refs: {n_overrides}")
        if extras:
            print(
                f"Đang render {label} {idx} (kèm {', '.join(extras)})...",
                file=sys.stderr,
            )
        else:
            print(f"Đang render {label} {idx}...", file=sys.stderr)
        scene_section_design = section_designs.get(int(sec_n)) if (sec_n is not None and section_designs) else None

        # Narrative window: ±2 beat lân cận thay cho việc nhồi cả truyện gốc.
        _w_lo = max(0, i - 2)
        _w_hi = min(len(scenes), i + 3)
        _window_lines: list[str] = []
        for j in range(_w_lo, _w_hi):
            _stxt = (getattr(scenes[j], "story_text", "") or "").strip()
            if not _stxt:
                continue
            _marker = "→" if j == i else " "
            _pn = getattr(scenes[j], "page_number", j + 1)
            _window_lines.append(f"{_marker} [#{_pn}] {_stxt}")
        narrative_window = "\n".join(_window_lines)

        data, mime = generate_scene_image(
            client,
            scene,
            narrative_window,
            ref_paths=ref_for_scene,
            image_model=args.image_model,
            color_mode=args.color,
            aspect_ratio=args.aspect_ratio,
            art_style=args.art_style,
            app_mode=render_mode,
            text_density=args.text_density,
            continuity_image_path=continuity_path,
            section_anchor_image_path=anchor_path,
            section_design=scene_section_design,
        )
        ext = ext_for_mime.get(mime or "", ".png")
        img_path = out_dir / f"scene_{idx:02d}{ext}"
        img_path.write_bytes(data)
        print(f"  -> {img_path}", file=sys.stderr)

        if section_image_continuity and scene_is_present:
            last_present_path = img_path
            last_present_scene = scene
            key = _anchor_key(sec_n)
            if key is not None and key not in section_anchor_paths:
                section_anchor_paths[key] = img_path

    print("Xong.", file=sys.stderr)


if __name__ == "__main__":
    main()
