"""
Merge 3 English SRTs (from WhisperX --task translate with different --language)
into one English SRT by picking the lowest-perplexity hypothesis per time window.

Usage:
  python arbitrate_en_srt.py ^
    --es path\\from_es.srt --en path\\from_en.srt --zh path\\from_zh.srt ^
    --out path\\subs.en.srt [--window 3.0] [--model distilgpt2] [--json decisions.json]
"""
from __future__ import annotations

# =============================================================================
# CONFIG — edit these defaults (CLI flags override)
# =============================================================================
DEFAULT_WINDOW_SEC = 3.0
DEFAULT_FLUENCY_MODEL = "distilgpt2"  # hub id; local snapshot preferred if present
# =============================================================================

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SRT_BLOCK = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n"
    r"(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


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


@dataclass
class Cue:
    start: float
    end: float
    text: str


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


def write_srt(path: Path, cues: Iterable[Cue]) -> None:
    lines: list[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(c.start)} --> {sec_to_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cue_at_midpoint(cues: list[Cue], w0: float, w1: float) -> Cue | None:
    """Return the cue whose midpoint falls in [w0, w1), or None.

    No overlap fallback: long Whisper cues must not be re-scored in every
    3s slice they touch (that produced identical paragraphs × dozens of windows).
    """
    for c in cues:
        cmid = (c.start + c.end) / 2.0
        if w0 <= cmid < w1:
            return c
    return None


class FluencyScorer:
    def __init__(self, model_id: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        resolved = model_id
        # Hub id → prefer complete local snapshot (China / offline)
        if not Path(model_id).is_dir():
            hub = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"
            snaps = hub / "snapshots"
            if snaps.is_dir():
                for s in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if (s / "config.json").exists() and (
                        (s / "model.safetensors").exists() or (s / "pytorch_model.bin").exists()
                    ):
                        resolved = str(s)
                        break
        local_only = Path(resolved).is_dir()
        kw = {"local_files_only": True} if local_only else {}
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(resolved, **kw).to(device)
        self.model.eval()
        self.resolved = resolved
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def perplexity(self, text: str) -> float:
        text = text.strip()
        if not text:
            return float("inf")
        # Cap length so one window cannot blow VRAM/time
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = enc.input_ids.to(self.device)
        if input_ids.size(1) < 2:
            return float("inf")
        out = self.model(input_ids, labels=input_ids)
        return float(torch.exp(out.loss).item())


def merge_windows(choices: list[tuple[float, float, str, str]]) -> list[Cue]:
    """choices: (start, end, text, source_lang). Merge adjacent same-source similar text."""
    if not choices:
        return []
    merged: list[Cue] = []
    cur_s, cur_e, cur_t, cur_src = choices[0]
    for s, e, t, src in choices[1:]:
        same = src == cur_src and t == cur_t
        gap_ok = s <= cur_e + 0.05
        if same and gap_ok:
            cur_e = max(cur_e, e)
        else:
            if cur_t.strip():
                merged.append(Cue(cur_s, cur_e, cur_t))
            cur_s, cur_e, cur_t, cur_src = s, e, t, src
    if cur_t.strip():
        merged.append(Cue(cur_s, cur_e, cur_t))
    return merged


def arbitrate(
    cues_es: list[Cue],
    cues_en: list[Cue],
    cues_zh: list[Cue],
    window: float,
    scorer: FluencyScorer,
) -> tuple[list[Cue], list[dict]]:
    """Pick one hypothesis per speech chunk (cue midpoint in window).

    Output SRT timings come from the **winning cue** (Whisper bounds), not from
    the 3s probe window. After a win, skip until that cue ends so the same
    paragraph is not decided again.
    """
    bags = {"es": cues_es, "en": cues_en, "zh": cues_zh}
    end = 0.0
    for bag in bags.values():
        if bag:
            end = max(end, bag[-1].end)
    if end <= 0:
        return [], []

    decisions: list[dict] = []
    choices: list[tuple[float, float, str, str]] = []
    t = 0.0
    covered_until = -1.0
    while t < end - 1e-6:
        w0, w1 = t, min(t + window, end)
        # Already emitted a longer cue that covers this probe window
        if w0 + 1e-6 < covered_until:
            t = w1
            continue

        cue_map = {lang: cue_at_midpoint(bag, w0, w1) for lang, bag in bags.items()}
        cands = {lang: (c.text if c else "") for lang, c in cue_map.items()}
        scores = {lang: scorer.perplexity(txt) for lang, txt in cands.items()}
        nonempty = {lang: s for lang, s in scores.items() if cands[lang]}
        if not nonempty:
            t = w1
            continue

        winner = min(nonempty, key=nonempty.get)
        win_cue = cue_map[winner]
        assert win_cue is not None
        text = win_cue.text
        # Keep original Whisper timing so resegment has a real span to split
        choices.append((win_cue.start, win_cue.end, text, winner))
        decisions.append(
            {
                "probe_start": w0,
                "probe_end": w1,
                "start": win_cue.start,
                "end": win_cue.end,
                "winner": winner,
                "text": text,
                "scores": scores,
                "candidates": cands,
            }
        )
        covered_until = max(covered_until, win_cue.end)
        t = w1

    return merge_windows(choices), decisions


# Default: prefer local HF snapshot if present (offline-friendly in China)
_DEFAULT_DISTIL = Path.home() / ".cache/huggingface/hub/models--distilgpt2/snapshots"
def _default_model() -> str:
    if _DEFAULT_DISTIL.is_dir():
        snaps = sorted(_DEFAULT_DISTIL.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for s in snaps:
            if (s / "config.json").exists() and (
                (s / "model.safetensors").exists() or (s / "pytorch_model.bin").exists()
            ):
                return str(s)
    return "distilgpt2"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arbitrate 3 English SRTs by LM perplexity")
    p.add_argument("--es", required=True, type=Path, help="SRT from --language es + translate")
    p.add_argument("--en", required=True, type=Path, help="SRT from --language en + translate")
    p.add_argument("--zh", required=True, type=Path, help="SRT from --language zh + translate")
    p.add_argument("--out", required=True, type=Path, help="Output English master SRT")
    p.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC, help="Window seconds")
    p.add_argument(
        "--model",
        default=None,
        help="HF causal LM id or local path (default: local distilgpt2 or CONFIG)",
    )
    p.add_argument("--json", type=Path, default=None, help="Optional decisions JSON path")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = p.parse_args(argv)
    if not args.model:
        args.model = _default_model()
        if args.model == "distilgpt2" and DEFAULT_FLUENCY_MODEL != "distilgpt2":
            args.model = DEFAULT_FLUENCY_MODEL

    def _np(path: Path) -> Path:
        s = str(path).strip().strip('"').strip("'").strip()
        return Path(s)

    args.es, args.en, args.zh, args.out = _np(args.es), _np(args.en), _np(args.zh), _np(args.out)
    if args.json:
        args.json = _np(args.json)

    for path in (args.es, args.en, args.zh):
        if not path.is_file():
            print(f"[ERROR] missing SRT: {path}", file=sys.stderr)
            return 1

    print(f"Loading fluency model: {args.model} ...")
    scorer = FluencyScorer(args.model, device=args.device)
    print(f"Device: {scorer.device}  resolved: {scorer.resolved}")

    cues_es = parse_srt(args.es)
    cues_en = parse_srt(args.en)
    cues_zh = parse_srt(args.zh)
    print(f"Cues: es={len(cues_es)} en={len(cues_en)} zh={len(cues_zh)} window={args.window}s")

    merged, decisions = arbitrate(cues_es, cues_en, cues_zh, args.window, scorer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_srt(args.out, merged)
    print(f"Wrote {args.out} ({len(merged)} cues)")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "window": args.window,
                    "model": args.model,
                    "counts": {"es": len(cues_es), "en": len(cues_en), "zh": len(cues_zh)},
                    "winner_tally": {
                        lang: sum(1 for d in decisions if d["winner"] == lang)
                        for lang in ("es", "en", "zh")
                    },
                    "decisions": decisions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
