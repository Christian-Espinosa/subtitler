"""
Translate an SRT to other languages, timings untouched.

Default translator: an LLM that reads the whole video and keeps each line inside a
duration budget (llm_translate.py; NVIDIA free tier, DeepSeek Flash as fallback;
needs NVIDIA_API_KEY or DEEPSEEK_API_KEY).

`--translator nllb` uses NLLB-200 locally instead. Its weights are licensed
CC-BY-NC-4.0 (non-commercial): opt-in only, for personal use.

Usage:
  python translate_srt.py --in "video subs.es.srt" --src es --langs en,zh
  python translate_srt.py --in "video subs.en.srt" --langs es --translator nllb
"""
from __future__ import annotations

# =============================================================================
# CONFIG — edit these defaults (CLI flags override)
# =============================================================================
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
# Default targets if --langs not passed
DEFAULT_TARGET_LANGS = ["es"]  # pipeline default is EN+ES; pass --langs es,zh for Chinese too
BATCH_SIZE = 16
MAX_LENGTH = 256
DEVICE = None  # None = auto (cuda if available)
# Soft wrap for on-screen (timings stay synced with EN master)
WRAP_MAX_CHARS = 42
WRAP_MAX_LINES = 2
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

# ISO short → NLLB-200 code
NLLB_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",  # Simplified Chinese (Bilibili / mainland)
}

_HUB_NAME = re.compile(r"^models--(.+)$")

# Chinese (and Japanese) are written without spaces between words: splitting
# them "by word" gave one token per sentence, so a sentence that spanned three
# cues landed whole on the first one and the other two were left empty
# (measured 2026-11-09). There the unit is the character, and a character is
# about two Latin letters wide on screen.
CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_ZH_PUNCT = str.maketrans({",": "，", ";": "；", ":": "：", "?": "？", "!": "！"})


def is_cjk(text: str) -> bool:
    return bool(CJK.search(text or ""))


def tokens(text: str) -> list[str]:
    """Words, or characters for CJK text."""
    text = " ".join((text or "").split())
    if is_cjk(text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def join_tokens(toks: list[str], cjk: bool) -> str:
    return ("" if cjk else " ").join(toks)


def line_limit(text: str, max_chars: int) -> int:
    return max(8, max_chars // 2) if is_cjk(text) else max_chars


def zh_punctuation(text: str) -> str:
    """NLLB writes ASCII punctuation in Chinese; on screen it should be full-width."""
    text = text.translate(_ZH_PUNCT)
    return re.sub(r"(?<=[\u4e00-\u9fff])\.(?=\s|$)", "。", text)


def normalize_path(raw: str | Path) -> Path:
    s = str(raw).strip().strip('"').strip("'").strip()
    return Path(s).expanduser()


@dataclass
class Cue:
    start_ts: str
    end_ts: str
    text: str


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cues: list[Cue] = []
    for m in SRT_BLOCK.finditer(raw.strip() + "\n\n"):
        text = re.sub(r"\s+", " ", m.group(4).replace("\n", " ")).strip()
        if text:
            cues.append(Cue(m.group(2).replace(".", ","), m.group(3).replace(".", ","), text))
    return cues


def wrap_lines(text: str, max_chars: int = WRAP_MAX_CHARS, max_lines: int = WRAP_MAX_LINES) -> str:
    """Wrap into at most max_lines, keeping the lines balanced.

    Balanced instead of greedy on purpose: a greedy wrap fills the first line to
    the brim and dumps whatever is left on the last one, which is how lines end
    up longer than the limit. Translations grow the text (Spanish -> English can
    add 20%), so this matters more here than in the source track.
    """
    text = " ".join(text.split())
    cjk = is_cjk(text)
    max_chars = line_limit(text, max_chars)
    if len(text) <= max_chars:
        return text
    words = tokens(text)
    if len(words) < 2:
        return text
    sep = "" if cjk else " "

    best, best_cost = None, None
    for i in range(1, len(words)):
        head = sep.join(words[:i])
        tail = sep.join(words[i:])
        if len(head) > max_chars:
            break
        if len(tail) > max_chars and max_lines <= 2:
            continue
        cost = abs(len(head) - len(tail))
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, (head, tail)

    if best is None:
        # No split fits: keep the first line legal and let the rest overflow.
        head, tail = words[0], sep.join(words[1:])
        for i in range(2, len(words)):
            candidate = sep.join(words[:i])
            if len(candidate) > max_chars:
                break
            head, tail = candidate, sep.join(words[i:])
        return f"{head}\n{tail}"

    head, tail = best
    if len(tail) > max_chars and max_lines > 2:
        return head + "\n" + wrap_lines(tail, max_chars, max_lines - 1)
    return f"{head}\n{tail}"


SENTENCE_END = re.compile(r"[.!?…。！？][\"')\]]?$")


def group_sentences(cues: list[Cue], max_cues: int = 6) -> list[list[int]]:
    """Group consecutive cues that form one sentence.

    A cue is a piece of screen, not a unit of meaning: "vamos a por" and "el
    siguiente" are two cues of the same sentence. Translating them separately
    gives broken output, because the translator never sees the whole sentence.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, c in enumerate(cues):
        cur.append(i)
        text = c.text.strip()
        if SENTENCE_END.search(text) or len(cur) >= max_cues:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def redistribute(source: list[str], translation: str) -> list[str]:
    """Spread a translated sentence over the cues it came from.

    Each cue keeps its own timing; only the text is shared out, in proportion to
    how much of the original sentence that cue held.
    """
    if len(source) == 1:
        return [translation]
    translation = " ".join(translation.split())
    if not translation:
        return source

    cjk = is_cjk(translation)
    total = sum(len(s) for s in source) or 1
    words = tokens(translation)
    if len(words) < len(source):
        # Not enough words to go around: put everything in the first cue.
        return [translation] + [""] * (len(source) - 1)

    # Un cue de una sola palabra parpadea y no se lee: si hay palabras de sobra,
    # cada trozo se lleva al menos dos.
    floor = 2 if len(words) >= 2 * len(source) else 1

    out: list[str] = []
    used = 0
    for i, s in enumerate(source):
        if i == len(source) - 1:
            out.append(join_tokens(words[used:], cjk))
            break
        remaining_cues = len(source) - i - 1
        share = len(s) / total
        take = max(floor, round(share * len(words)))
        take = min(take, len(words) - used - remaining_cues * floor)
        take = max(1, take)
        out.append(join_tokens(words[used : used + take], cjk))
        used += take
    return out


def ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole, ms = whole + 1, 0
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


def split_overflow(cues: list[Cue], max_chars: int, max_lines: int) -> list[Cue]:
    """Split cues whose translation no longer fits on screen.

    Translating grows the text, so a cue that fitted in the spoken language can
    overflow in the target one. Here the time is divided by character count,
    which is an estimate — but inside a cue of a few seconds, not inside a
    30-second block, so the error stays under a couple of tenths.
    """
    out: list[Cue] = []
    for c in cues:
        text = " ".join(c.text.split())
        cjk = is_cjk(text)
        budget = line_limit(text, max_chars) * max_lines
        if len(text) <= budget:
            out.append(c)
            continue

        parts = max(2, -(-len(text) // budget))
        words = tokens(text)
        target = len(text) / parts
        chunks: list[str] = []
        cur: list[str] = []
        for w in words:
            cur.append(w)
            if len(join_tokens(cur, cjk)) >= target and len(chunks) < parts - 1:
                chunks.append(join_tokens(cur, cjk))
                cur = []
        if cur:
            chunks.append(join_tokens(cur, cjk))

        start, end = ts_to_sec(c.start_ts), ts_to_sec(c.end_ts)
        total = sum(len(ch) for ch in chunks) or 1
        t = start
        for i, ch in enumerate(chunks):
            share = (end - start) * len(ch) / total
            piece_end = end if i == len(chunks) - 1 else min(end, t + share)
            out.append(Cue(sec_to_ts(t), sec_to_ts(piece_end), ch))
            t = piece_end
    return out


def write_srt(path: Path, cues: Iterable[Cue]) -> None:
    lines: list[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{c.start_ts} --> {c.end_ts}")
        lines.append(wrap_lines(c.text))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_model(model_id: str) -> str:
    """Prefer local HF snapshot if complete (offline / China-friendly)."""
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    # facebook/nllb-200-distilled-600M → models--facebook--nllb-200-distilled-600M
    hub_dir = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"
    snaps = hub_dir / "snapshots"
    if snaps.is_dir():
        for s in sorted(snaps.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if (s / "config.json").exists() and (
                (s / "model.safetensors").exists()
                or (s / "pytorch_model.bin").exists()
                or list(s.glob("model-*.safetensors"))
            ):
                return str(s)
    return model_id


class NllbTranslator:
    def __init__(self, model_id: str, device: str | None = None):
        import torch  # heavy: only when a model is actually loaded
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        resolved = resolve_model(model_id)
        local_only = Path(resolved).is_dir()
        kw = {"local_files_only": True} if local_only else {}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading NLLB: {resolved} (local_only={local_only}) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, **kw)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(resolved, **kw).to(device)
        self.model.eval()
        self.resolved = resolved

    def translate_batch(self, texts: list[str], src: str, tgt: str) -> list[str]:
        with self._torch.inference_mode():
            return self._translate_batch(texts, src, tgt)

    def _translate_batch(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if not texts:
            return []
        src_code = NLLB_CODES[src]
        tgt_code = NLLB_CODES[tgt]
        self.tokenizer.src_lang = src_code
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_code)
        out: list[str] = []
        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for bi, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
            chunk = texts[i : i + BATCH_SIZE]
            print(f"  batch {bi}/{n_batches} ({len(chunk)} cues)...", flush=True)
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LENGTH,
                num_beams=4,
            )
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            out.extend(t.strip() for t in decoded)
        return out


def translate_file(
    src_srt: Path,
    targets: list[str],
    out_dir: Path | None,
    model_id: str,
    device: str | None,
    source: str = "en",
    translator: str = "llm",
) -> list[Path]:
    cues = parse_srt(src_srt)
    if not cues:
        print(f"[ERROR] no cues in {src_srt}", file=sys.stderr)
        return []

    out_dir = out_dir or src_srt.parent
    # "viaje subs.en.srt" → stem base "viaje subs" → "viaje subs.es.srt"
    name = src_srt.name
    if name.endswith(f" subs.{source}.srt"):
        base = name[: -len(f" subs.{source}.srt")]
        def out_name(lang: str) -> str:
            return f"{base} subs.{lang}.srt"
    elif name.endswith(f".{source}.srt"):
        base = name[: -len(f".{source}.srt")]
        def out_name(lang: str) -> str:
            return f"{base}.{lang}.srt"
    else:
        stem = src_srt.stem
        def out_name(lang: str) -> str:
            return f"{stem}.{lang}.srt"

    need = [t for t in targets if t != source]
    written: list[Path] = []
    if not need:
        return written

    if translator == "llm":
        return _translate_llm(cues, source, need, out_dir, out_name)

    print("NLLB-200 weights are CC-BY-NC-4.0: non-commercial use only.", flush=True)
    tr = NllbTranslator(model_id, device=device)
    groups = group_sentences(cues)
    texts = [" ".join(cues[i].text for i in g) for g in groups]
    for lang in need:
        if lang not in NLLB_CODES:
            print(f"[ERROR] unsupported lang: {lang}", file=sys.stderr)
            continue
        print(f"Translating {NLLB_CODES[source]} -> {lang} ({NLLB_CODES[lang]}), "
              f"{len(cues)} cues en {len(texts)} frases ...", flush=True)
        translated = tr.translate_batch(texts, source, lang)
        if lang == "zh":
            translated = [zh_punctuation(t) for t in translated]

        out_cues: list[Cue] = []
        for g, whole in zip(groups, translated):
            pieces = redistribute([cues[i].text for i in g], whole)
            for idx, piece in zip(g, pieces):
                c = cues[idx]
                if piece.strip():
                    out_cues.append(Cue(c.start_ts, c.end_ts, piece.strip()))
        out_cues = split_overflow(out_cues, WRAP_MAX_CHARS, WRAP_MAX_LINES)
        dest = out_dir / out_name(lang)
        write_srt(dest, out_cues)
        print(f"Wrote {dest}", flush=True)
        written.append(dest)
    return written


def _translate_llm(cues: list[Cue], source: str, need: list[str], out_dir: Path, out_name) -> list[Path]:
    import llm_translate

    written: list[Path] = []
    spans = [(ts_to_sec(c.start_ts), ts_to_sec(c.end_ts), c.text) for c in cues]
    for lang in need:
        report = llm_translate.Report()
        print(f"Translating {source} -> {lang} with an LLM, {len(cues)} cues ...", flush=True)
        texts = llm_translate.translate_cues(spans, source, lang, report=report)
        if lang == "zh":
            texts = [zh_punctuation(t) for t in texts]
        out_cues = [Cue(c.start_ts, c.end_ts, t) for c, t in zip(cues, texts) if t.strip()]
        out_cues = split_overflow(out_cues, WRAP_MAX_CHARS, WRAP_MAX_LINES)
        dest = out_dir / out_name(lang)
        write_srt(dest, out_cues)
        print(f"Wrote {dest}  (provider: {', '.join(dict.fromkeys(report.provider_used))}; "
              f"rewritten shorter: {report.rewritten}; still over budget: {report.over_budget})", flush=True)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Translate an SRT (LLM by default, NLLB opt-in)")
    p.add_argument("--in", dest="inp", required=True, help="English master SRT")
    p.add_argument(
        "--langs",
        default=",".join(DEFAULT_TARGET_LANGS),
        help="Comma targets: es,zh (default from CONFIG)",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--model", default=NLLB_MODEL)
    p.add_argument("--device", default=DEVICE)
    p.add_argument(
        "--src",
        default="en",
        help="Language of the input SRT (en|es|zh). The aligned pipeline "
             "transcribes in the spoken language, so this is often 'es'.",
    )
    p.add_argument("--translator", default="llm", choices=["llm", "nllb"],
                   help="llm (default) or nllb (local, CC-BY-NC: personal use only)")
    args = p.parse_args(argv)

    src = normalize_path(args.inp)
    if not src.is_file():
        print(f"[ERROR] file not found: {src}", file=sys.stderr)
        return 1
    source = args.src.strip().lower()
    if args.translator == "nllb" and source not in NLLB_CODES:
        print(f"[ERROR] unsupported --src: {source}", file=sys.stderr)
        return 1
    langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    out_dir = normalize_path(args.out_dir) if args.out_dir else None
    written = translate_file(src, langs, out_dir, args.model, args.device, source, args.translator)
    return 0 if written or not [l for l in langs if l != source] else 1


if __name__ == "__main__":
    raise SystemExit(main())
