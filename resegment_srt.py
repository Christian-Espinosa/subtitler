"""
Resegment an SRT for readable on-screen subtitles (YouTube / Bilibili style).

Splits dense cues by sentence, enforces max chars/line, CPS, and cue duration.
Merges consecutive near-duplicate / identical texts.

Usage:
  python resegment_srt.py --in "video subs.en.srt"
  python resegment_srt.py --in "video subs.en.srt" --out "video subs.en.srt"
"""
from __future__ import annotations

# =============================================================================
# CONFIG — edit these (CLI flags override)
# =============================================================================
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CPS = 18.0          # characters per second (reading speed)
MIN_CUE_SEC = 1.0
MAX_CUE_SEC = 6.0
MIN_GAP_SEC = 0.08      # tiny gap between adjacent cues
# =============================================================================

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SRT_BLOCK = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n"
    r"(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])")
CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


def normalize_path(raw: str | Path) -> Path:
    s = str(raw).strip().strip('"').strip("'").strip()
    return Path(s).expanduser()


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cues: list[Cue] = []
    for m in SRT_BLOCK.finditer(raw.strip() + "\n\n"):
        start = ts_to_sec(m.group(2))
        end = ts_to_sec(m.group(3))
        text = re.sub(r"\s+", " ", m.group(4).replace("\n", " ")).strip()
        if end > start and text:
            cues.append(Cue(start, end, text))
    return cues


def wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    """Greedy word wrap into up to max_lines; overflow stays on last line if needed."""
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if cur and len(trial) > max_chars and len(lines) < max_lines - 1:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    # If still too many lines, collapse extras into last allowed line
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = head + [tail]
    return "\n".join(lines)


def split_units(text: str) -> list[str]:
    """Prefer sentence breaks; fall back to clauses; else whole text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in CLAUSE_SPLIT.split(text) if p.strip()] or [text]
    # Further split very long units by words
    out: list[str] = []
    max_chunk = MAX_CHARS_PER_LINE * MAX_LINES
    for p in parts:
        if len(p) <= max_chunk * 1.4:
            out.append(p)
            continue
        words = p.split()
        buf: list[str] = []
        for w in words:
            trial = (" ".join(buf + [w])).strip()
            if buf and len(trial) > max_chunk:
                out.append(" ".join(buf))
                buf = [w]
            else:
                buf.append(w)
        if buf:
            out.append(" ".join(buf))
    return out or [text]


def needed_duration(text: str, cps: float, min_sec: float, max_sec: float) -> float:
    chars = len(re.sub(r"\s+", "", text))
    if cps <= 0:
        return min_sec
    d = chars / cps
    return max(min_sec, min(max_sec, d))


def merge_identical(cues: list[Cue]) -> list[Cue]:
    if not cues:
        return []
    out: list[Cue] = []
    cur = Cue(cues[0].start, cues[0].end, cues[0].text)

    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.replace("\n", " ")).strip().lower()

    for c in cues[1:]:
        same = norm(c.text) == norm(cur.text)
        near = (
            len(norm(c.text)) > 40
            and (
                norm(c.text).startswith(norm(cur.text)[:60])
                or norm(cur.text).startswith(norm(c.text)[:60])
            )
        )
        gap_ok = c.start <= cur.end + 0.35
        if (same or near) and gap_ok:
            # Prefer longer text; extend end
            if len(c.text) > len(cur.text):
                cur.text = c.text
            cur.end = max(cur.end, c.end)
        else:
            out.append(cur)
            cur = Cue(c.start, c.end, c.text)
    out.append(cur)
    return out


def resegment_cue(
    cue: Cue,
    *,
    max_chars: int,
    max_lines: int,
    max_cps: float,
    min_sec: float,
    max_sec: float,
    min_gap: float,
) -> list[Cue]:
    units = split_units(cue.text)
    if not units:
        return []

    # Weight durations by character count
    weights = [max(1, len(re.sub(r"\s+", "", u))) for u in units]
    total_w = sum(weights) or 1
    span = max(cue.end - cue.start, min_sec * len(units))

    # Ideal durations from CPS, then scale to fit [cue.start, cue.end] if possible
    ideals = [needed_duration(u, max_cps, min_sec, max_sec) for u in units]
    ideal_sum = sum(ideals)
    if ideal_sum <= span + 1e-6:
        durs = ideals
        # distribute leftover proportionally
        leftover = span - ideal_sum
        durs = [d + leftover * (w / total_w) for d, w in zip(durs, weights)]
    else:
        # Need more time than available: pack by weight into original span
        durs = [span * (w / total_w) for w in weights]
        # Soft floor: don't go below ~0.6s unless unavoidable
        floor = min(0.6, span / len(units))
        durs = [max(floor, d) for d in durs]
        s = sum(durs)
        durs = [d * (span / s) for d in durs]

    out: list[Cue] = []
    t = cue.start
    for i, (unit, dur) in enumerate(zip(units, durs)):
        start = t
        end = cue.end if i == len(units) - 1 else min(cue.end, start + dur)
        if end - start < 0.05:
            end = min(cue.end, start + 0.05)
        wrapped = wrap_lines(unit, max_chars, max_lines)
        if wrapped.strip():
            out.append(Cue(start, end, wrapped))
        t = end + min_gap
        if t >= cue.end and i < len(units) - 1:
            # squeeze remaining into last available slice
            t = max(start, cue.end - 0.05 * (len(units) - i))
    return out


def resegment(
    cues: list[Cue],
    *,
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES,
    max_cps: float = MAX_CPS,
    min_sec: float = MIN_CUE_SEC,
    max_sec: float = MAX_CUE_SEC,
    min_gap: float = MIN_GAP_SEC,
) -> list[Cue]:
    merged = merge_identical(cues)
    out: list[Cue] = []
    for c in merged:
        plain = c.text.replace("\n", " ")
        chars = len(re.sub(r"\s+", "", plain))
        dur = c.end - c.start
        cps = chars / dur if dur > 0 else 999.0
        needs_split = (
            cps > max_cps
            or dur > max_sec
            or len(plain) > max_chars * max_lines
            or plain.count(" ") > 25
        )
        if needs_split:
            out.extend(
                resegment_cue(
                    c,
                    max_chars=max_chars,
                    max_lines=max_lines,
                    max_cps=max_cps,
                    min_sec=min_sec,
                    max_sec=max_sec,
                    min_gap=min_gap,
                )
            )
        else:
            out.append(Cue(c.start, c.end, wrap_lines(plain, max_chars, max_lines)))

    # Fix overlaps / order
    out.sort(key=lambda x: (x.start, x.end))
    fixed: list[Cue] = []
    for c in out:
        if fixed and c.start < fixed[-1].end:
            c = Cue(fixed[-1].end + min_gap, max(c.end, fixed[-1].end + min_gap + 0.2), c.text)
        if c.end > c.start and c.text.strip():
            fixed.append(c)
    return merge_identical(fixed)


def write_srt(path: Path, cues: Iterable[Cue]) -> None:
    lines: list[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(c.start)} --> {sec_to_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resegment SRT for readable on-screen timing")
    p.add_argument("--in", dest="inp", required=True, help="Input SRT")
    p.add_argument("--out", default=None, help="Output SRT (default: overwrite --in)")
    p.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_LINE)
    p.add_argument("--max-lines", type=int, default=MAX_LINES)
    p.add_argument("--max-cps", type=float, default=MAX_CPS)
    p.add_argument("--min-sec", type=float, default=MIN_CUE_SEC)
    p.add_argument("--max-sec", type=float, default=MAX_CUE_SEC)
    args = p.parse_args(argv)

    src = normalize_path(args.inp)
    if not src.is_file():
        print(f"[ERROR] file not found: {src}", file=sys.stderr)
        return 1
    dest = normalize_path(args.out) if args.out else src

    cues = parse_srt(src)
    print(f"Input cues: {len(cues)}", flush=True)
    out = resegment(
        cues,
        max_chars=args.max_chars,
        max_lines=args.max_lines,
        max_cps=args.max_cps,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
    )
    write_srt(dest, out)
    # quick stats
    if out:
        durs = [c.end - c.start for c in out]
        words = [len(c.text.replace("\n", " ").split()) for c in out]
        print(
            f"Output cues: {len(out)}  "
            f"dur med={sorted(durs)[len(durs)//2]:.1f}s  "
            f"words med={sorted(words)[len(words)//2]}",
            flush=True,
        )
    print(f"Wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
