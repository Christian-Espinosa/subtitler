"""
Translate subtitle cues with an LLM that sees the whole video, inside a duration budget.

The default translator of the subtitler (NLLB-200 is CC-BY-NC: opt-in only, see
translate_srt.py --translator nllb). Also the translation module that dubbing
(ResolveDubber) imports: one module, two users.

Why an LLM and not a sentence translator: it reads the whole transcript first, so it
adapts expressions instead of translating them word by word ("Muy buenas gente" is
"Hey everyone", not "Very good people"), and it can be told how long each line may
be. That budget is what makes the text readable as a subtitle and speakable in the
time of the original cue when it is dubbed (isochrony).

Budget per cue: max_chars = 0.9 x duration x CPS (17 for Latin scripts, 8 for
Chinese/Japanese). A translation over 1.15 x its budget is sent back once to be
rewritten shorter, keeping the meaning.

Providers (OpenAI-compatible chat API, standard library only), tried in order:
  1. NVIDIA NIM  nvidia/nemotron-3-ultra-550b-a55b   NVIDIA_API_KEY   (free tier)
  2. DeepSeek    deepseek-flash                      DEEPSEEK_API_KEY
Each answer must be the JSON asked for, with every cue id: otherwise it is retried,
then the next provider is used. Nothing is invented: if no provider answers, it fails.

Usage as a library:
  from llm_translate import translate_cues
  out = translate_cues([(0.0, 1.2, "Hola a todos"), ...], src="es", tgt="en")
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

CPS_LATIN = 17.0
CPS_CJK = 8.0
BUDGET_FACTOR = 0.9
REWRITE_OVER = 1.15
CHUNK = 40                 # cues per request
CONTEXT_CHARS = 12000      # how much of the whole transcript goes as context
TIMEOUT = 120
RETRIES = 2

LANG_NAMES = {"es": "Spanish", "en": "English", "zh": "Simplified Chinese", "fr": "French",
              "de": "German", "it": "Italian", "pt": "Portuguese (Brazil)", "ja": "Japanese", "ko": "Korean",
              "hi": "Hindi", "ar": "Modern Standard Arabic", "id": "Indonesian"}
CJK = re.compile(r"[぀-ヿ一-鿿]")

# The intention of each line, for the dubbing voice (rate / pitch / expressiveness).
# The per-cue reference audio carries the rest of the emotion.
TONES = ("neutral", "happy", "excited", "serious", "sad", "angry", "whisper", "question")


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str

    def key(self) -> str:
        return os.environ.get(self.key_env, "").strip()


PROVIDERS = [
    Provider("nvidia", "https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3-ultra-550b-a55b", "NVIDIA_API_KEY"),
    Provider("deepseek", "https://api.deepseek.com", "deepseek-flash", "DEEPSEEK_API_KEY"),
]


@dataclass
class Report:
    """What happened, for the log and for dub_report.json."""
    provider_used: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    rewritten: int = 0
    over_budget: int = 0
    tones: dict[int, str] = field(default_factory=dict)   # cue index (0-based) -> tone


def cps_for(lang: str) -> float:
    return CPS_CJK if lang in ("zh", "ja") else CPS_LATIN


def budget(start: float, end: float, lang: str) -> int:
    """Max characters the translation of a cue may take on screen / in the mouth."""
    return max(4, int(BUDGET_FACTOR * max(0.0, end - start) * cps_for(lang)))


def _post(provider: Provider, messages: list[dict], timeout: float = TIMEOUT) -> str:
    body = json.dumps({"model": provider.model, "messages": messages, "temperature": 0.3,
                       "max_tokens": 8000}).encode("utf-8")
    req = urllib.request.Request(provider.base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {provider.key()}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"] or "")


def parse_tones(raw: str) -> dict[int, str]:
    """The optional "tone" of each cue; unknown or missing ones are left out. Pure."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    cues = data.get("cues") if isinstance(data, dict) else None
    out: dict[int, str] = {}
    for c in cues if isinstance(cues, list) else []:
        if not isinstance(c, dict):
            continue
        tone = str(c.get("tone") or "").strip().lower()
        try:
            if tone in TONES:
                out[int(c.get("i"))] = tone
        except (TypeError, ValueError):
            continue
    return out


def parse_answer(raw: str, ids: Sequence[int]) -> dict[int, str] | None:
    """The JSON the model must return: {"cues": [{"i": 1, "text": "..."}]}, every id once,
    no empty text. Anything else is None (and the call is retried). Pure."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("cues"), list):
        return None
    out: dict[int, str] = {}
    for c in data["cues"]:
        if not isinstance(c, dict):
            return None
        try:
            i = int(c.get("i"))
        except (TypeError, ValueError):
            return None
        text = " ".join(str(c.get("text") or "").split())
        if not text:
            return None
        out[i] = text
    return out if set(out) == set(ids) else None


SYSTEM = (
    "You translate video subtitles that will also be spoken by a dubbing voice. Translate the "
    "MEANING the way a native speaker would say it, adapting idioms and greetings; keep names, "
    "numbers and the speaker's tone. Each cue has max_chars: the translation must fit in it, so "
    "prefer shorter natural wording over literal completeness. Never merge, split or reorder cues. "
    "Also give the tone the line is said with, one of: " + ", ".join(TONES) + ". "
    "Answer ONLY with JSON: {\"cues\": [{\"i\": <id>, \"text\": \"<translation>\", \"tone\": \"<tone>\"}]} "
    "with every id."
)


def _user(cues: list[dict], src: str, tgt: str, context: str, shorter: bool) -> str:
    task = ("These translations are too long for their time. Rewrite each one SHORTER, within "
            "max_chars, keeping the meaning." if shorter else
            f"Translate each cue from {LANG_NAMES.get(src, src)} to {LANG_NAMES.get(tgt, tgt)}.")
    return (f"WHOLE VIDEO TRANSCRIPT (context only, do not translate it all):\n{context}\n\n"
            f"{task}\nCUES:\n{json.dumps({'cues': cues}, ensure_ascii=False)}")


Chat = Callable[[Provider, list[dict]], str]


def _ask(cues: list[dict], src: str, tgt: str, context: str, report: Report, *, shorter: bool = False,
         chat: Chat = _post, providers: Sequence[Provider] = PROVIDERS, sleep=time.sleep) -> dict[int, str]:
    ids = [c["i"] for c in cues]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user(cues, src, tgt, context, shorter)}]
    usable = [p for p in providers if p.key()]
    if not usable:
        raise RuntimeError("no translation provider configured: set NVIDIA_API_KEY (free) or DEEPSEEK_API_KEY")
    for p in usable:
        for attempt in range(RETRIES + 1):
            raw = ""
            try:
                raw = chat(p, messages)
                got = parse_answer(raw, ids)
            except Exception as exc:  # noqa: BLE001 - network/HTTP: retry, then next provider
                report.failures.append(f"{p.name}: {type(exc).__name__}: {str(exc)[:120]}")
                got = None
            if got is not None:
                report.provider_used.append(p.name)
                if not shorter:
                    report.tones.update({i - 1: t for i, t in parse_tones(raw).items()})
                return got
            if attempt < RETRIES:
                report.failures.append(f"{p.name}: answer not valid JSON for these cues (attempt {attempt + 1})")
                sleep(1.5 * (attempt + 1))
    raise RuntimeError("no provider returned a valid translation: " + "; ".join(report.failures[-4:]))


def translate_cues(cues: Iterable[tuple[float, float, str]], src: str, tgt: str, *,
                   chat: Chat = _post, providers: Sequence[Provider] = PROVIDERS,
                   report: Report | None = None, sleep=time.sleep) -> list[str]:
    """One translation per cue, same order, timings untouched."""
    cues = list(cues)
    report = report if report is not None else Report()
    context = " ".join(t for _, _, t in cues)[:CONTEXT_CHARS]
    items = [{"i": n + 1, "text": t, "max_chars": budget(a, b, tgt)} for n, (a, b, t) in enumerate(cues)]
    result: dict[int, str] = {}
    for k in range(0, len(items), CHUNK):
        part = items[k:k + CHUNK]
        result.update(_ask(part, src, tgt, context, report, chat=chat, providers=providers, sleep=sleep))
        long = [{"i": c["i"], "text": result[c["i"]], "max_chars": c["max_chars"]}
                for c in part if len(result[c["i"]]) > REWRITE_OVER * c["max_chars"]]
        if long:
            report.rewritten += len(long)
            try:
                result.update(_ask(long, tgt, tgt, context, report, shorter=True,
                                   chat=chat, providers=providers, sleep=sleep))
            except RuntimeError as exc:  # keep the longer version and say so, never drop text
                report.failures.append(f"rewrite shorter failed, kept the long version: {exc}"[:200])
    report.over_budget = sum(1 for c in items if len(result[c["i"]]) > REWRITE_OVER * c["max_chars"])
    return [result[c["i"]] for c in items]
