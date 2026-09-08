"""
Measure subtitle quality of an SRT (and, optionally, the arbiter decisions).

Reports the things that make subtitles unreadable or out of sync:
  - cue duration vs MAX_CUE_SEC
  - characters per line vs MAX_CHARS_PER_LINE
  - reading speed (CPS) vs MAX_CPS
  - overlaps and negative gaps
  - how the arbiter distributed winners and how much screen time each pass owns

Usage:
  python analyze_srt.py --in "video subs.en.srt"
  python analyze_srt.py --in "video subs.en.srt" --arbiter "video._arbiter.json"
  python analyze_srt.py --in a.srt --compare b.srt
"""
from __future__ import annotations

# =============================================================================
# CONFIG — thresholds used to flag problems (match resegment_srt.py)
# =============================================================================
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CPS = 18.0
MIN_CUE_SEC = 1.0
MAX_CUE_SEC = 6.0
# =============================================================================

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SRT_BLOCK = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n"
    r"(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def lines(self) -> list[str]:
        return [ln for ln in self.text.split("\n") if ln.strip()]

    @property
    def chars(self) -> int:
        return len(re.sub(r"\s+", "", self.text))

    @property
    def cps(self) -> float:
        return self.chars / self.dur if self.dur > 0 else 999.0


def ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cues: list[Cue] = []
    for m in SRT_BLOCK.finditer(raw.strip() + "\n\n"):
        text = m.group(4).strip()
        if text:
            cues.append(Cue(ts_to_sec(m.group(2)), ts_to_sec(m.group(3)), text))
    return cues


def pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:5.1f}%" if total else "  n/a"


def quantiles(values: list[float]) -> tuple[float, float, float, float]:
    """min, median, p90, max."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    v = sorted(values)
    return v[0], v[len(v) // 2], v[int(len(v) * 0.9)], v[-1]


def report(cues: list[Cue], label: str) -> dict[str, float]:
    if not cues:
        print(f"[{label}] sin cues")
        return {}

    n = len(cues)
    durs = [c.dur for c in cues]
    lo, med, p90, hi = quantiles(durs)
    screen_time = sum(durs)

    long_cues = [c for c in cues if c.dur > MAX_CUE_SEC]
    long_time = sum(c.dur for c in long_cues)
    short_cues = [c for c in cues if c.dur < MIN_CUE_SEC]
    fast_cues = [c for c in cues if c.cps > MAX_CPS]
    wide_lines = [ln for c in cues for ln in c.lines if len(ln) > MAX_CHARS_PER_LINE]
    total_lines = sum(len(c.lines) for c in cues)
    many_lines = [c for c in cues if len(c.lines) > MAX_LINES]

    overlaps = 0
    gaps: list[float] = []
    for prev, cur in zip(cues, cues[1:]):
        gap = cur.start - prev.end
        if gap < 0:
            overlaps += 1
        else:
            gaps.append(gap)

    print(f"\n===== {label} =====")
    print(f"cues                    : {n}")
    print(f"tiempo en pantalla      : {screen_time / 60:.1f} min")
    print(f"duracion cue (s)        : min {lo:.2f} | mediana {med:.2f} | p90 {p90:.2f} | max {hi:.2f}")
    print(f"cues > {MAX_CUE_SEC:.0f}s              : {len(long_cues):4d}  {pct(len(long_cues), n)}"
          f"   (ocupan {pct(int(long_time), int(screen_time))} del tiempo en pantalla)")
    print(f"cues < {MIN_CUE_SEC:.0f}s              : {len(short_cues):4d}  {pct(len(short_cues), n)}")
    print(f"cues > {MAX_CPS:.0f} CPS           : {len(fast_cues):4d}  {pct(len(fast_cues), n)}")
    print(f"lineas > {MAX_CHARS_PER_LINE} caracteres  : {len(wide_lines):4d}  {pct(len(wide_lines), total_lines)}"
          f"   (de {total_lines} lineas)")
    print(f"cues con > {MAX_LINES} lineas     : {len(many_lines):4d}  {pct(len(many_lines), n)}")
    print(f"solapes con el anterior : {overlaps:4d}  {pct(overlaps, n)}")
    if gaps:
        _, gmed, _, gmax = quantiles(gaps)
        print(f"hueco entre cues (s)    : mediana {gmed:.2f} | max {gmax:.2f}")

    if long_cues:
        print(f"\n  Los 5 cues mas largos:")
        for c in sorted(long_cues, key=lambda x: -x.dur)[:5]:
            flat = c.text.replace("\n", " / ")
            print(f"   {c.start:7.1f}s  {c.dur:5.1f}s  {c.cps:4.1f} CPS  {flat[:70]}")

    return {
        "cues": n,
        "med_dur": med,
        "p90_dur": p90,
        "max_dur": hi,
        "long_pct": 100.0 * len(long_cues) / n,
        "wide_pct": 100.0 * len(wide_lines) / total_lines if total_lines else 0.0,
        "fast_pct": 100.0 * len(fast_cues) / n,
        "overlaps": overlaps,
    }


def report_arbiter(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    data = raw.get("decisions") if isinstance(raw, dict) else raw
    if not isinstance(data, list) or not data:
        print(f"\n[arbitro] {path.name}: formato inesperado")
        return

    winners: dict[str, int] = {}
    time_by_winner: dict[str, float] = {}
    spans: list[float] = []
    switches = 0
    prev_winner = None
    covered_skips = 0.0

    for d in data:
        w = str(d.get("winner", "?"))
        span = float(d.get("end", 0.0)) - float(d.get("start", 0.0))
        probe = float(d.get("probe_end", 0.0)) - float(d.get("probe_start", 0.0))
        winners[w] = winners.get(w, 0) + 1
        time_by_winner[w] = time_by_winner.get(w, 0.0) + span
        spans.append(span)
        if span > probe:
            covered_skips += span - probe
        if prev_winner is not None and w != prev_winner:
            switches += 1
        prev_winner = w

    total = sum(winners.values())
    lo, med, p90, hi = quantiles(spans)
    print(f"\n===== ARBITRO ({path.name}) =====")
    print(f"decisiones              : {total}")
    print(f"cambios de pasada       : {switches}  {pct(switches, total)} de las decisiones")
    print(f"span del cue ganador (s): min {lo:.2f} | mediana {med:.2f} | p90 {p90:.2f} | max {hi:.2f}")
    print(f"tiempo saltado por 'covered_until': {covered_skips / 60:.1f} min")
    print("\n  reparto de victorias:")
    for w in sorted(winners, key=lambda k: -winners[k]):
        print(f"   {w:>3} : {winners[w]:4d} cues  {pct(winners[w], total)}"
              f"   {time_by_winner[w] / 60:5.1f} min de audio")

    # Sesgo de longitud: la perplejidad por token baja cuando el texto es largo
    # (mas contexto para predecir). Si el ganador es casi siempre el candidato
    # mas largo, el arbitro no esta midiendo calidad, esta midiendo longitud.
    longest_wins = 0
    comparable = 0
    win_len: list[float] = []
    lose_len: list[float] = []
    for d in data:
        cands = {k: v for k, v in (d.get("candidates") or {}).items() if v}
        if len(cands) < 2:
            continue
        comparable += 1
        w = str(d.get("winner", "?"))
        longest = max(cands, key=lambda k: len(cands[k]))
        if w == longest:
            longest_wins += 1
        for k, txt in cands.items():
            (win_len if k == w else lose_len).append(len(txt))

    if comparable:
        _, wmed, _, _ = quantiles(win_len)
        _, lmed, _, _ = quantiles(lose_len)
        print(f"\n  sesgo de longitud (en {comparable} decisiones con >1 candidato):")
        print(f"   gana el candidato MAS LARGO : {longest_wins:4d}  {pct(longest_wins, comparable)}")
        print(f"   longitud mediana ganador    : {wmed:6.0f} caracteres")
        print(f"   longitud mediana perdedores : {lmed:6.0f} caracteres")
        if wmed > lmed * 1.3:
            print("   -> el arbitro premia longitud, no fidelidad")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure SRT readability and timing sanity")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--compare", type=Path, default=None, help="Second SRT to compare against")
    p.add_argument("--arbiter", type=Path, default=None, help="_arbiter.json from the pipeline")
    args = p.parse_args(argv)

    if not args.inp.is_file():
        print(f"[ERROR] no existe: {args.inp}", file=sys.stderr)
        return 1

    a = report(parse_srt(args.inp), args.inp.name)
    if args.compare and args.compare.is_file():
        b = report(parse_srt(args.compare), args.compare.name)
        print("\n===== COMPARATIVA =====")
        print(f"{'metrica':<24}{args.inp.name[:22]:>24}{args.compare.name[:22]:>24}")
        for key, fmt in (
            ("cues", "{:.0f}"),
            ("med_dur", "{:.2f}s"),
            ("p90_dur", "{:.2f}s"),
            ("max_dur", "{:.2f}s"),
            ("long_pct", "{:.1f}%"),
            ("wide_pct", "{:.1f}%"),
            ("fast_pct", "{:.1f}%"),
            ("overlaps", "{:.0f}"),
        ):
            print(f"{key:<24}{fmt.format(a.get(key, 0)):>24}{fmt.format(b.get(key, 0)):>24}")

    if args.arbiter and args.arbiter.is_file():
        report_arbiter(args.arbiter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
