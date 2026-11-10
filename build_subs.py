"""
Build a readable SRT from WhisperX word-level alignment.

This replaces `resegment_srt.py` for the aligned path. The difference matters:
resegment receives a block of text with only a start and an end, so it has to
*estimate* where each sentence begins by counting characters. Here every word
carries its own timestamp, so cue boundaries are **measured**, not guessed.

Rules applied, in order of preference when a cue must end:
  1. sentence-final punctuation
  2. clause punctuation (, ; :)
  3. the longest silence inside the candidate cue
  4. the character / duration limit

Usage:
  python build_subs.py --align _work/audio.json --out "video subs.es.srt"
  python build_subs.py --align a.json --out o.srt --max-chars 42 --max-sec 6
"""
from __future__ import annotations

# =============================================================================
# CONFIG — edit these (CLI flags override)
# =============================================================================
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_SEC = 6.0
MIN_CUE_SEC = 1.0
MAX_CPS = 18.0          # if a cue exceeds this, it is split further
PAUSE_SEC = 0.40        # silence that ends an utterance
# Keep the cue on screen after the last word, when the silence allows it. This
# is what makes dense speech readable: the start stays glued to the audio (the
# part the eye notices) while the text gets extra time to be read.
LEAD_OUT_SEC = 0.60
MIN_GAP_SEC = 0.08
# =============================================================================

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SENTENCE_END = re.compile(r"[.!?…。！？]$")
CLAUSE_END = re.compile(r"[,;:，；：]$")
CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    text: str


def load_words(path: Path) -> tuple[list[Word], str]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    words: list[Word] = []
    for seg in data.get("segments", []):
        seg_words = seg.get("words", [])
        for w in seg_words:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            if "start" in w and "end" in w:
                words.append(Word(float(w["start"]), float(w["end"]), text))
            elif words:
                # The aligner could not place this token (usually punctuation or
                # a number). Attaching it to the previous word keeps the text
                # complete without inventing a timestamp.
                words[-1].text += text if CJK.search(text) else f" {text}"
    words.sort(key=lambda w: w.start)
    return words, str(data.get("language") or "")


def join_words(words: list[Word]) -> str:
    out = ""
    for w in words:
        if not out:
            out = w.text
        elif CJK.search(w.text[:1]) or CJK.search(out[-1:]):
            out += w.text
        else:
            out += f" {w.text}"
    return out


def crosses(breaks: list[float], a: Word | Cue, b: Word | Cue) -> bool:
    """Is there a hard break (e.g. a change of speaker) between the middle of `a` and
    the middle of `b`? Midpoints, because alignment often stretches a word over the
    silence after it. Pure."""
    lo, hi = (a.start + a.end) / 2, (b.start + b.end) / 2
    return any(lo < t <= hi for t in breaks)


def load_breaks(path: Path | None) -> list[float]:
    """--breaks: a JSON list of seconds where a cue must end (dubbing: speaker turns)."""
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(float(t) for t in (data.get("breaks", []) if isinstance(data, dict) else data))


def split_utterances(words: list[Word], pause: float, breaks: list[float] | None = None) -> list[list[Word]]:
    breaks = breaks or []
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (w.start - cur[-1].end > pause or crosses(breaks, cur[-1], w)):
            groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def choose_break(words: list[Word], max_chars: int) -> int:
    """Index (exclusive) where the cue should end. Never returns 0."""
    # Longest prefix that fits the character budget.
    limit = len(words)
    for i in range(1, len(words) + 1):
        if len(join_words(words[:i])) > max_chars:
            limit = i - 1
            break
    limit = max(1, limit)

    # 1) sentence end inside the fitting prefix, preferring the latest one
    for i in range(limit, 0, -1):
        if SENTENCE_END.search(words[i - 1].text):
            return i
    # 2) clause punctuation, but only in the second half so cues are not tiny
    for i in range(limit, max(1, limit // 2), -1):
        if CLAUSE_END.search(words[i - 1].text):
            return i
    # 3) the longest silence inside the prefix, if it is a real pause
    best_gap, best_i = 0.0, 0
    for i in range(1, limit):
        gap = words[i].start - words[i - 1].end
        if gap > best_gap:
            best_gap, best_i = gap, i
    if best_gap >= 0.25 and best_i:
        return best_i
    # 4) plain character limit
    return limit


def fits_on_screen(text: str, max_chars: int, max_lines: int) -> bool:
    """True if the text can be wrapped without any line overflowing."""
    return all(len(ln) <= max_chars for ln in wrap_balanced(text, max_chars, max_lines).split("\n"))


def pack_utterance(
    words: list[Word], *, max_chars: int, max_lines: int, max_sec: float
) -> list[Cue]:
    """Split an utterance into cues.

    Note there is no reading-speed rule here. CPS is set by how fast the person
    talks, and splitting a cue does not slow anyone down: each half keeps the
    same ratio. Density is handled later by holding the cue on screen into the
    following silence (see ``polish``).
    """
    cues: list[Cue] = []
    rest = words
    line_chars = max_chars // max_lines
    while rest:
        cut = choose_break(rest, max_chars)
        while cut > 1 and rest[cut - 1].end - rest[0].start > max_sec:
            cut -= 1
        # A long word can make a 2-line wrap impossible even inside budget.
        while cut > 1 and not fits_on_screen(join_words(rest[:cut]), line_chars, max_lines):
            cut -= 1
        chunk = rest[:cut]
        cues.append(Cue(chunk[0].start, chunk[-1].end, join_words(chunk)))
        rest = rest[cut:]
    return cues


def merge_short(
    cues: list[Cue], *, max_chars: int, max_lines: int, max_sec: float, pause: float,
    breaks: list[float] | None = None,
) -> list[Cue]:
    """Glue tiny cues to their neighbour when they still fit.

    A sentence like "Ya está." would otherwise get its own flash of half a
    second, which is unreadable.
    """
    out: list[Cue] = []
    for c in cues:
        if out:
            prev = out[-1]
            joined = f"{prev.text} {c.text}"
            fits = (
                len(joined) <= max_chars
                and c.end - prev.start <= max_sec
                and fits_on_screen(joined, max_chars // max_lines, max_lines)
            )
            close = c.start - prev.end <= pause and not crosses(breaks or [], prev, c)
            too_short = (prev.end - prev.start) < MIN_CUE_SEC or (c.end - c.start) < MIN_CUE_SEC
            if fits and close and too_short:
                out[-1] = Cue(prev.start, c.end, joined)
                continue
        out.append(c)
    return out


def wrap_balanced(text: str, max_chars: int, max_lines: int) -> str:
    """Split into at most max_lines, keeping line lengths similar."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    if len(words) < 2:
        return text

    best_split, best_cost = None, None
    for i in range(1, len(words)):
        a = " ".join(words[:i])
        b = " ".join(words[i:])
        if len(a) > max_chars or len(b) > max_chars * (max_lines - 1):
            continue
        cost = abs(len(a) - len(b))
        if best_cost is None or cost < best_cost:
            best_cost, best_split = cost, (a, b)

    if best_split is None:
        # Cannot fit: greedy wrap and accept the overflow on the last line.
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and len(trial) > max_chars:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return "\n".join(lines[:max_lines] if len(lines) <= max_lines
                         else lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])])

    a, b = best_split
    if len(b) > max_chars and max_lines > 2:
        return a + "\n" + wrap_balanced(b, max_chars, max_lines - 1)
    return f"{a}\n{b}"


def polish(cues: list[Cue], *, min_sec: float, lead_out: float, min_gap: float) -> list[Cue]:
    """Extend short cues into the following silence and remove overlaps."""
    out: list[Cue] = []
    for i, c in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < len(cues) else c.end + lead_out + min_sec
        end = c.end + lead_out
        if end - c.start < min_sec:
            end = c.start + min_sec
        end = min(end, next_start - min_gap)
        if end <= c.start:
            end = c.start + 0.05
        out.append(Cue(c.start, end, c.text))
    return out


def write_srt(path: Path, cues: list[Cue], max_chars: int, max_lines: int) -> None:
    lines: list[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(c.start)} --> {sec_to_ts(c.end)}")
        lines.append(wrap_balanced(c.text, max_chars, max_lines))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def sec_to_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build an SRT from word-level alignment")
    p.add_argument("--align", required=True, type=Path, help="WhisperX JSON with word timings")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_LINE)
    p.add_argument("--max-lines", type=int, default=MAX_LINES)
    p.add_argument("--max-sec", type=float, default=MAX_CUE_SEC)
    p.add_argument("--min-sec", type=float, default=MIN_CUE_SEC)
    p.add_argument("--max-cps", type=float, default=MAX_CPS)
    p.add_argument("--pause", type=float, default=PAUSE_SEC)
    p.add_argument("--breaks", type=Path, default=None,
                   help="JSON list of seconds where a cue must end (e.g. speaker changes for dubbing)")
    args = p.parse_args(argv)

    if not args.align.is_file():
        print(f"[ERROR] no existe: {args.align}", file=sys.stderr)
        return 1

    words, language = load_words(args.align)
    if not words:
        print("[ERROR] el JSON no trae tiempos por palabra. Se genero con "
              "--no_align o con --task translate (que desactiva la alineacion).",
              file=sys.stderr)
        return 1

    budget = args.max_chars * args.max_lines
    breaks = load_breaks(args.breaks)
    cues: list[Cue] = []
    for utt in split_utterances(words, args.pause, breaks):
        cues.extend(
            pack_utterance(utt, max_chars=budget, max_lines=args.max_lines, max_sec=args.max_sec)
        )
    cues = merge_short(
        cues,
        max_chars=budget,
        max_lines=args.max_lines,
        max_sec=args.max_sec,
        pause=args.pause,
        breaks=breaks,
    )
    cues = polish(cues, min_sec=args.min_sec, lead_out=LEAD_OUT_SEC, min_gap=MIN_GAP_SEC)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_srt(args.out, cues, args.max_chars, args.max_lines)

    durs = sorted(c.end - c.start for c in cues)
    print(f"idioma        : {language or '?'}")
    print(f"palabras      : {len(words)}")
    print(f"cues          : {len(cues)}")
    if durs:
        print(f"duracion cue  : mediana {durs[len(durs)//2]:.2f}s | max {durs[-1]:.2f}s")
    print(f"escrito       : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
