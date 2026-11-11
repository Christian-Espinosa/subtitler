# Subtitler — manual

Local tool (your PC, GPU) that writes **SRT** subtitles ready for YouTube / Bilibili.

It is aimed at videos that mix **Spanish, English and/or Chinese**. Default output:

- an **English** master (`subs.en.srt`)
- a **Spanish** track (`subs.es.srt`)

Transcription never leaves the machine: models download once, then run from cache.


**Setup notes (2026-11).**
- *Python and mirrors.* The `.bat` files find Python 3.12 by themselves (`_python.bat`). To force one, or to use a
  Hugging Face / pip mirror (e.g. in mainland China), copy `subtitler.local.example.bat` to `subtitler.local.bat`
  (not committed) and set `PY312` / `HF_ENDPOINT`. Without a mirror, if huggingface.co does not answer, the aligned
  pipeline uses the models already downloaded.
- *Translation.* `pipeline_aligned.py` translates with an LLM by default (`llm_translate.py`: whole-video context,
  each line kept within the time of its cue; needs `NVIDIA_API_KEY`, free, or `DEEPSEEK_API_KEY`).
  `--translator nllb` uses NLLB-200 locally instead; its weights are CC-BY-NC-4.0 (non-commercial).
- *Spoken language.* Transcription goes through `transcribe.py` (WhisperX Python API), because `python -m whisperx`
  3.8 writes the wrong language in its JSON when it auto-detects.
- *Tests.* `python -m unittest discover -s tests -t .` (no models needed).

License: MIT (`LICENSE`). Third-party models keep their own licenses.

---

## Why this repo for subtitles

Plenty of apps do “Whisper → SRT → translate” (for example [WhisperSubTranslate](https://github.com/Blue-B/WhisperSubTranslate): whisper.cpp plus Hy-MT2 / Gemini / DeepL). This repo ships the same product — SRT for YouTube / Bilibili — but the file is **better timed and more readable**, which is what you see after you publish.

### What those apps usually do

1. Transcribe with **whisper.cpp** (or plain Whisper). Each cue is an audio *segment*, not a phrase measured word by word.
2. Translate the text (sometimes with an LLM that sounds more natural than NLLB).
3. Write the SRT using those segment clocks.

That looks fine on a short, single-language clip. On a long video, fast speech, or mixed languages, the clock drifts: text appears over silence, runs ahead inside a paragraph, then snaps back at the next one. WhisperSubTranslate added “Sync” models because of that drift. Here the same failure was measured and the pipeline was changed ([§3-bis](#3-bis-why-the-aligned-pipeline-measured)).

### What this repo does differently

Timing and cue cuts are fixed **before** translation, and translation **does not touch the clocks**.

```text
video → ffmpeg
      → WhisperX  --task transcribe  + word-level alignment
      → build_subs.py   cuts on real word boundaries
      → translate_srt.py  (NLLB)  same clocks, text only
```

| Piece | Why it shows up on screen |
|-------|---------------------------|
| **WhisperX, not whisper.cpp** | WhisperX pins every word to the audio. whisper.cpp gives one start/end per segment; everything inside is a guess. |
| **`--task transcribe`, not `translate`** | `--task translate` in WhisperX **turns alignment off**. This repo’s old pipeline did that; cues came out ~24 s long and resegment guessed the inner cuts by character count. That is why they ran ahead. |
| **`build_subs.py`** | Closes cues on sentence ends, commas, or the longest **measured** silence. YouTube-ish limits: ~42 characters, 2 lines, 1–6 s. |
| **Translate last** | NLLB (or whatever translator you swap in) gets an already-cut SRT and copies the timestamps. Output language does not move the clock. |
| **Measure, don’t guess** | `analyze_srt.py` and `measure_drift.py` report cues that start in silence, long lines, and CPS spikes. |

Measured 2026-09-04 on a 23 min Spanish video, same machine, aligned pipeline vs the old “translate inside Whisper” flow:

| | 3-pass + arbiter | This repo (aligned) |
|---|---|---|
| Cues that start **in silence** | 26.5% | **0%** |
| Longest cue | 10.75 s | **5.98 s** (cap 6 s) |
| Lines > 42 characters | 9.3% | **0.6%** |
| Subtitles on screen vs real speech | 17.6 min / 14.9 min of speech | 15.1 min / 14.9 min |
| Runtime | ~25 min | **~1 min** |

The 26.5% silence figure is the one you notice while watching: text on screen when nobody is talking. Segment + Sync apps try to patch that. Here the cue **starts when the word starts**.

### Where this repo does not win

- **No GUI.** You launch a `.bat`. If you do not want paths or flags, an Electron app is easier.
- **NLLB is more literal** than Gemini or DeepL (“Muy buenas gente” → “Very good people”). In return the text never goes to an API and the clocks do not move. For LLM style, change `translate_srt.py`; you do not need to change the ASR.
- **One language per video** on the aligned flow. WhisperX aligns with one language model. Truly mixed cuts (es/en/zh in the same take) are still a gap (§7).
- **No dubbing.** SRT only. Spoken translation is a different program.

For **getting subtitles** — the SRT you upload to YouTube — this repo beats a generic Whisper + translator because the clock comes from the audio, not from an estimate, and translation cannot break it. The rest of this manual is how to run it and how to measure it.

---

## 1. Glossary (terms you will see in logs and code)

| Term | What it means here |
|------|--------------------|
| **SRT** | Subtitle format: index, start/end clock, text. The only format this repo writes. |
| **Cue** | One SRT block: “from 00:00:03 to 00:00:08, this text”. |
| **ASR** | *Automatic Speech Recognition*: audio to text. Done by **Whisper** via **WhisperX**. |
| **Whisper / WhisperX** | OpenAI’s Whisper plus the WhisperX wrapper (time alignment, VAD, CLI). The runtime is `python -m whisperx`; you do not need the `whisperX/` folder. |
| **`--task transcribe`** | Writes in the **spoken language** (or whatever you force with `--language`). |
| **`--task translate`** | Whisper **only translates to English**. Spanish or Chinese audio still yields an English SRT. Intentional: the 3 passes are compared in one language. |
| **`--language es\|en\|zh`** | **Assumes** the whole file is that language. Whisper does not switch mid-video. That is why the old flow ran 3 passes. |
| **Pass** | One WhisperX run with a forced language. Intermediates: `[name]._pass_en_from_es.srt` (etc.). |
| **Hypothesis** | Each pass proposes a different English text for the same span. The arbiter picks one. |
| **Arbiter** | `arbitrate_en_srt.py`: compares the 3 hypotheses and keeps the most “natural English”. |
| **Perplexity (PPL)** | “How surprised is an English language model” by a text. **Lower = more fluent**. It does not measure fidelity to the audio. |
| **`distilgpt2`** | Tiny GPT-2 used only to compute that perplexity. It does not write subtitles. |
| **Window / probe** | Time slice (default 3 s) where the arbiter looks for a cue whose **midpoint** falls inside. |
| **Midpoint** | Temporal middle of a cue. Avoids re-scoring the same long paragraph in every 3 s window. |
| **Resegment** | Split dense cues (e.g. 20 s of text) into short on-screen cues. |
| **CPS** | *Characters per second*: reading speed. ~18 is a typical YouTube cap. |
| **NLLB** | Meta’s *No Language Left Behind*: EN → ES or ZH. **Does not** listen to audio; it translates the English SRT. |
| **Hugging Face (HF)** | Model host. Typical cache: `%USERPROFILE%\.cache\huggingface\hub\`. |
| **`HF_ENDPOINT`** | Alternate Hugging Face hub URL, for networks where huggingface.co is blocked (set it in `subtitler.local.bat`). |
| **CUDA / `float16`** | NVIDIA GPU math. `float16` saves VRAM (RTX 3060 Ti). |
| **`batch_size`** | How many audio chunks Whisper processes at once. Drop it (e.g. 4) if you OOM. |
| **VAD** | *Voice Activity Detection*: where speech is. WhisperX uses Pyannote. |
| **FFmpeg / ffprobe** | Extract audio and durations. Must be on the Windows PATH. |
| **torchcodec / triton** | Common Windows warnings. **They do not stop** the pipeline if an SRT is written. |
| **Diarization** | Speaker split. **Off** (needs an HF token). |

---

## 2. How to run

### Aligned pipeline (recommended since 2026-09)

Double-click `subtitular_alineado.bat` (it asks for a path) or:

```bat
subtitular_alineado.bat "D:\Videos\mi_video.mp4"
subtitular_alineado.bat "video.mp4" --language es --langs es,en
```

Transcribes in the spoken language, **aligns word by word**, then translates. This is the one that fixes early-running cues and oversized blocks: see [§3-bis](#3-bis-why-the-aligned-pipeline-measured). Takes **~1 minute per 20 of video**, versus ~25 for the 3-pass pipeline.

**Output:** `[name] subs.es.srt` (spoken language) and `[name] subs.en.srt` (translation).

### 3-pass + arbiter pipeline (legacy)

Still there, **not recommended**: it measures worse on everything (§3-bis).

```bat
subtitular_en_master.bat "D:\Videos\mi_video.mp4"
```

Quotes are stripped if you paste a path from Explorer.

**Output, same folder as the video:**

| File | What it is |
|------|------------|
| `[name] subs.en.srt` | English master, already resegmented for the screen |
| `[name] subs.es.srt` | Spanish (NLLB translation of the English) |
| `[name]._arbiter.json` | Debug: which pass won each cue |

`._pass_en_from_*.srt` files are **deleted** at the end. Keep them with `--keep-passes`.

### Simple mode (one Whisper pass)

If the video is **almost all one language** and you do not need the arbiter:

```bat
subtitular.bat "D:\Videos\mi_video.mp4"
subtitular.bat "video.mp4" --language es
```

Output: `[name] subs.srt`.

### Pipeline flags (passed through to the `.bat`)

```bat
subtitular_en_master.bat "video.mp4" --langs en,es,zh
subtitular_en_master.bat "video.mp4" --keep-passes
subtitular_en_master.bat "video.mp4" --skip-resegment
subtitular_en_master.bat "video.mp4" --skip-nllb
subtitular_en_master.bat "video.mp4" --window 3
subtitular_en_master.bat "video.mp4" --skip-whisperx
```

`--skip-whisperx` reuses existing `._pass_*.srt` files (you need `--keep-passes` on a previous run).

---

## 3. Pipeline flow (step by step)

The orchestrator is `pipeline_en_master.py`. `subtitular_en_master.bat` launches it.

```text
video / audio
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEPS 1–3  WhisperX × 3                                 │
 │   each: --task translate --language es | en | zh        │
 │   all 3 SRTs are ENGLISH                                │
 │   files: [stem]._pass_en_from_es.srt  (etc.)            │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 4  Arbiter (distilgpt2)                            │
 │   picks 1 hypothesis per cue (lowest PPL)               │
 │   writes [stem] subs.en.srt  (cues still long)          │
 │   writes [stem]._arbiter.json                           │
 │   deletes _pass_*  (unless --keep-passes)               │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 5  Resegment                                       │
 │   splits sentences, 2 lines, ~42 chars, ~18 CPS, 1–6 s  │
 │   overwrites the same subs.en.srt                       │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 6  NLLB                                            │
 │   translates each cue EN → ES (same clocks)             │
 │   writes [stem] subs.es.srt                             │
 └─────────────────────────────────────────────────────────┘
```

**Why 3 passes.** Whisper **locks one language for the whole file**. If you force `es` and then speak Chinese, that pass usually fails. The pass whose language matches the speech produces coherent English; the others produce junk or calques. The arbiter picks span by span.

> This idea has a bug you only see by measuring: the arbiter rewards the most *fluent* English, not the most *faithful*, and a hallucination always sounds fluent. See §3-bis.

**What the pipeline does not do.** It does not fix ASR errors (“breakfast” vs “ayuno”). The arbiter only chooses among three texts. NLLB only translates the English already chosen.

The console prints `PASO x/y`, elapsed time, and a rough **ETA** (not exact; first run with downloads is slower).

---

## 3-bis. Why the aligned pipeline (measured)

Diagnosis on 2026-09-04 on `Microfono 1.0.mp4` (23 min, Spanish), comparing the real 3-pass output to the aligned one.

### The root cause: `--task translate` disables alignment

In `whisperX/whisperx/transcribe.py`:

```python
if task == "translate":
    # translation cannot be aligned
    no_align = True
```

The 3-pass pipeline uses `--task translate` on all three. That **turns off forced alignment**, which is the whole point of WhisperX over Whisper. Knock-on effects:

1. Without alignment, WhisperX emits **one cue per 30 s VAD chunk** (`chunk_size=30` in `asr.py`). Measured: cues the arbiter picked had a **median of 23.6 s**.
2. The arbiter compares whole 24 s paragraphs by perplexity, and after a win it skips later windows (`covered_until`). On this video that **skipped 15.4 minutes** of decisions.
3. `resegment_srt.py` gets 24 s blocks with only a start and an end, so it **guesses** each sentence boundary by counting characters. Of 309 final cues, only about 50 had boundaries taken from the audio; the rest were counts.

That is why subtitles ran ahead inside a paragraph and snapped back at the next one.

### The arbiter rewards fluency, not fidelity

On a video that is **100% Spanish**, the winning pass was **Chinese: 64%** (32 of 50 decisions), versus 18% for Spanish. Extreme case:

```text
269
00:17:17,303 --> 00:17:21,110
Leading Starring Qin Liying,
```

Invented Chinese TV credits over a stretch **with no voice**, left on screen while nobody spoke. Whisper hallucinates when you force a language it does not hear, and `distilgpt2` scores that hallucination as good English, because it is.

### Numbers, before and after

English track, same video:

| Metric | 3-pass + arbiter | Aligned | |
|---|---|---|---|
| Longest cue | 10.75 s | **5.98 s** | cap was 6 s |
| Cues > 6 s | 7.1% (15% of time) | **0%** | |
| Lines > 42 characters | 9.3% | **0.6%** | |
| Cues that start **in silence** | 26.5% | **0%** | |
| Subtitles on screen | 17.6 min | **15.1 min** | there are 14.9 min of speech |
| Runtime | ~25 min | **1 min 10 s** | |

Two honest caveats:

- The aligned **0%** figures are internally consistent: cues are built from the same alignment they are scored against. What is independent is the old 26.5%, and that it put 17.6 min of subtitles on 14.9 min of speech.
- The aligned flow has **more dense cues** (13% over 18 CPS, versus 2%). That is not a real regression: the old flow lowered CPS by leaving text on screen through silences. Here the cue lasts as long as the phrase plus 0.6 s of courtesy.

### New flow

```text
video → ffmpeg (16 kHz mono)
      → WhisperX  --task transcribe  (alignment ON)  → JSON with per-word times
      → build_subs.py    cuts on real word boundaries
      → translate_srt.py NLLB, translates text and does NOT touch clocks
```

Translation is last on purpose: output language never conditions sync. Style does change: Whisper-translating writes more natural English than NLLB, which is more literal (“Muy buenas gente” → “Very good people”). In return it says what was said: where the old flow wrote *"Go."*, the audio was *"¡Fuego!"*.

---

## 4. What each file does

### `subtitular_en_master.bat`

Entry point. Checks Python 3.12, asks for or takes a path, strips quotes, calls `pipeline_en_master.py`.

| Variable | Default | What for |
|----------|---------|----------|
| `PY312` | found by `_python.bat` | Set it in `subtitler.local.bat` to force one |
| `PIPE` | `pipeline_en_master.py` next to the `.bat` | Leave it |
| `EXTRA_ARGS` | empty | Flags **always** appended (e.g. `--langs en,es,zh`) |
| `HF_ENDPOINT` | unset (official hub) | Mirror, in `subtitler.local.bat` |

### `subtitular.bat`

One WhisperX call (`transcribe` implicit, not translate). Renames `[stem].srt` → `[stem] subs.srt`.

| Variable | Default | What for |
|----------|---------|----------|
| `PY312` | found by `_python.bat` | Same as above |
| `WHISPER_MODEL` | `large-v2` | ASR model |
| `BATCH_SIZE` | `8` | VRAM; lower if OOM |
| `COMPUTE_TYPE` | `float16` | GPU precision |
| `DEFAULT_LANGUAGE` | empty | Auto-detect; or `es` / `en` / `zh` |
| `HF_ENDPOINT` | mirror | Same as above |

### `pipeline_en_master.py`

Orchestrates the 4 programs. **Main CONFIG** if you want to change defaults.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PYTHON` | Python 3.12 | Interpreter for WhisperX and child scripts |
| `HF_ENDPOINT` | environment | Hugging Face mirror (unset = official hub) |
| `WHISPER_MODEL` | `large-v2` | ASR size (`large-v3` is heavier) |
| `WHISPER_BATCH_SIZE` | `8` | GPU batches |
| `WHISPER_COMPUTE_TYPE` | `float16` | Compute type |
| `WHISPER_PASS_LANGS` | `es, en, zh` | Forced audio languages (3 hypotheses) |
| `WHISPER_ETA_RTF` | `0.30` | ETA only: 0.30 ≈ 30% of video duration per pass |
| `ARBITER_WINDOW_SEC` | `3.0` | Arbiter probe size (s) |
| `FLUENCY_MODEL` | `distilgpt2` | Perplexity LM |
| `ARBITER_ETA_SEC_PER_MIN` | `8.0` | ETA only |
| `OUTPUT_SUB_LANGS` | `en, es` | Final tracks. Add `"zh"` for Chinese |
| `RESEGMENT` | `True` | If `False`, do not split cues for the screen |
| `RESEGMENT_MAX_CHARS` | `42` | Max characters per line |
| `RESEGMENT_MAX_LINES` | `2` | Max lines per cue |
| `RESEGMENT_MAX_CPS` | `18.0` | Reading cap |
| `RESEGMENT_MIN_SEC` | `1.0` | Minimum cue duration |
| `RESEGMENT_MAX_SEC` | `6.0` | Target maximum duration |
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Translator |
| `NLLB_ETA_SEC_PER_MIN` | `12.0` | ETA only |
| `KEEP_PASSES` | `False` | Keep `_pass_*.srt` |

CLI flags (`--langs`, `--window`, …) **override** these defaults for that run.

### `pipeline_aligned.py` + `subtitular_alineado.bat`

Orchestrator of the recommended flow. **CONFIG** for the new pipeline.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PYTHON` | Python 3.12 | Interpreter |
| `WHISPER_MODEL` | `large-v2` | ASR |
| `WHISPER_BATCH_SIZE` | `8` | Drop to 4 if VRAM is tight |
| `SPOKEN_LANGUAGE` | empty | Spoken language; empty = detect |
| `OUTPUT_SUB_LANGS` | `es, en` | Tracks to write |
| `MAX_CHARS_PER_LINE` / `MAX_LINES` | `42` / `2` | Screen limits |
| `MAX_CUE_SEC` / `MIN_CUE_SEC` | `6.0` / `1.0` | Cue duration |
| `KEEP_WORK` | `False` | Keep `_work/` (audio + alignment JSON) |

Useful flags: `--language es`, `--langs es,en,zh`, `--skip-translate`, `--keep-work`, and `--skip-whisperx` to rebuild subtitles without transcribing again (needs `--keep-work` from a previous run).

### `build_subs.py`

Turns word-level alignment into readable cues. Replaces `resegment_srt.py` on the aligned flow. The difference is fundamental: every word already has a timestamp, so cuts are **measured**, not estimated.

Preference order for closing a cue: sentence end → comma or semicolon → longest silence inside the cue → character limit.

```bat
python build_subs.py --align _work/video.json --out "video subs.es.srt"
```

No CPS rule on purpose: splitting a cue does not change how fast anyone talks. Density is handled by leaving the cue on screen 0.6 s after the last word, if silence allows.

### `analyze_srt.py`

Measures an SRT: durations, long lines, CPS, overlaps, and, if you pass `_arbiter.json`, how the arbiter split the wins.

```bat
python analyze_srt.py --in "video subs.en.srt" --compare "otro.srt"
python analyze_srt.py --in "video subs.en.srt" --arbiter "video._arbiter.json"
```

### `measure_drift.py`

Checks whether an SRT is in sync, using word-level alignment as the reference. Reports how many cues appear **before anyone speaks** and lists the worst.

```bat
python measure_drift.py --srt "video subs.en.srt" --align _work/video.json
```

### `arbitrate_en_srt.py`

Reads 3 English SRTs. Walks time in windows. If a cue’s **midpoint** falls in the window, it scores the 3 hypotheses with `distilgpt2` and the **lowest PPL** wins. Writes the master SRT with **Whisper’s original clocks** (long cues). Does not decide again until that cue ends.

Standalone:

```bat
python arbitrate_en_srt.py --es from_es.srt --en from_en.srt --zh from_zh.srt --out "video subs.en.srt" --json decisions.json
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `DEFAULT_WINDOW_SEC` | `3.0` | Probe |
| `DEFAULT_FLUENCY_MODEL` | `distilgpt2` | Prefers a local snapshot in the HF cache |

The JSON includes `probe_start` / `probe_end` (window), `start` / `end` (winning cue), `winner`, `scores`, `candidates`.

### `resegment_srt.py`

Splits the English master so it can be read: sentences, wrap to 2 lines, minimum gaps between cues.

```bat
python resegment_srt.py --in "video subs.en.srt"
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAX_CHARS_PER_LINE` | `42` | Line width |
| `MAX_LINES` | `2` | Lines per cue |
| `MAX_CPS` | `18.0` | Reading speed |
| `MIN_CUE_SEC` | `1.0` | No shorter than this (unless forced) |
| `MAX_CUE_SEC` | `6.0` | Try not to exceed this |
| `MIN_GAP_SEC` | `0.08` | Gap between adjacent cues |

**Do not** run it on the ES file after NLLB: that would split clocks and desync EN/ES. ES only *wraps* lines (see below).

### `translate_srt.py`

NLLB: each English cue → Spanish (and/or Chinese). **Same timestamps** as EN. Then wraps lines (`WRAP_*`) without changing clocks.

```bat
python translate_srt.py --in "video subs.en.srt" --langs es
python translate_srt.py --in "video subs.en.srt" --langs es,zh
```

If you edit EN by hand, rerun **only** this script to regenerate ES.

| Variable | Default | Meaning |
|----------|---------|---------|
| `NLLB_MODEL` | `nllb-200-distilled-600M` | Translation model |
| `DEFAULT_TARGET_LANGS` | `es` | If you omit `--langs` |
| `BATCH_SIZE` | `16` | Cues per GPU batch |
| `MAX_LENGTH` | `256` | Max generated tokens |
| `DEVICE` | auto CUDA | `cuda` or `cpu` |
| `WRAP_MAX_CHARS` | `42` | Visual wrap ES/ZH |
| `WRAP_MAX_LINES` | `2` | Lines |

Internal NLLB codes: `en` → `eng_Latn`, `es` → `spa_Latn`, `zh` → `zho_Hans` (simplified).

### `safe_downloader/`

Template for large model downloads with a watchdog (flaky network / China). See [`safe_downloader/README.md`](safe_downloader/README.md).

Typical scripts: `install_distilgpt2.ps1`, `install_nllb.ps1`, `install_faster_whisper_large_v2.ps1`.

### `whisperX/`

Reference clone of WhisperX. **What actually runs** is the `pip` package (`python -m whisperx`). You do not need this folder to subtitle.

---

## 5. Requirements

| Piece | Notes |
|-------|--------|
| Python **3.12** | WhisperX does not support 3.14 |
| `whisperx` + PyTorch **CUDA** | In that Python |
| FFmpeg and ffprobe | On PATH |
| GPU | Tested on RTX 3060 Ti |
| Models | ASR `large-v2`, `distilgpt2`, NLLB-600M (HF cache) |

If `huggingface.co` does not connect, set `HF_ENDPOINT` to a mirror (for example `https://hf-mirror.com`) in `subtitler.local.bat`.

---

## 6. Repo layout

```text
subtitler/
  subtitular_alineado.bat     recommended entry
  pipeline_aligned.py         aligned-flow orchestrator
  build_subs.py               word alignment → cues
  analyze_srt.py              measures SRT readability
  measure_drift.py            measures sync against alignment
  translate_srt.py            NLLB, --src anything → ES/EN/ZH

  subtitular_en_master.bat    legacy pipeline entry
  subtitular.bat              1 Whisper pass
  pipeline_en_master.py       legacy orchestrator
  arbitrate_en_srt.py         picks among 3 EN SRTs
  resegment_srt.py            readable cues (legacy flow)

  safe_downloader/            model downloads
  whisperX/                   reference clone
  README.md                   this manual
```

Do not leave `_test_*` or `.log` files in the root (they are in `.gitignore`).

---

## 7. Known quality issues

Usable as a **draft**. Review before you publish.

### Aligned pipeline (the recommended one)

**One language per video.** WhisperX aligns with one model per language, so the aligned flow assumes the video is mostly one language. Truly mixed videos would need per-segment language detection and a matching aligner per group; **that is not built**.

**NLLB is literal.** “Muy buenas gente” → “Very good people”. The English Whisper wrote while translating sounded better, even when it did not say what was said. If you want both, LLM translation (DeepSeek) is the natural next step.

**Dense cues.** ~13% go over 18 CPS when speech is fast. That is real: the cue lasts as long as the phrase. Loosen it by raising `LEAD_OUT_SEC` in `build_subs.py`.

**ASR.** Still mangles proper names (“Ches” → “Chess”). Fix by editing the spoken-language SRT and rerunning only `translate_srt.py`.

### Legacy pipeline (3 passes)

**Arbiter.** It does not measure fidelity, it measures fluency. On a Spanish video it picked the **Chinese pass 64%** of the time, and kept invented Chinese TV credits over silent stretches (§3-bis).

**Resegment.** The 2nd line sometimes exceeds 42 characters. Some dense cues exceed 6 s. Inevitable: it splits 24 s blocks by estimating from character counts.

**Sync.** 26.5% of cues appeared when nobody was talking.

**Console warnings.** `triton not found`, `torchcodec` / FFmpeg DLL, Lightning `upgrade_checkpoint`, TF32: **noise** if Whisper keeps progressing and an SRT appears at the end.

**Recommended review flow**

1. Full pipeline.
2. Edit `subs.en.srt` (key terms).
3. Regenerate ES: `python translate_srt.py --in "... subs.en.srt" --langs es`.
4. Upload to YouTube / Bilibili.

---

## 8. Later tweaks (shortcuts)

| I want… | Where |
|---------|--------|
| Chinese in the output too | `--langs es,en,zh` |
| Spoken language only | `--skip-translate` |
| Rebuild subtitles without transcribing again | `--keep-work` once, then `--skip-whisperx` |
| Longer or shorter cues | `--max-sec` / `--max-chars`, or the CONFIG in `build_subs.py` |
| Text to linger on screen | `LEAD_OUT_SEC` in `build_subs.py` |
| To know if an SRT is good | `analyze_srt.py` and `measure_drift.py` |
| Shorter/longer subtitles (legacy flow) | `RESEGMENT_*` in the pipeline (or `resegment_srt.py`) |
| Keep the 3 Whisper passes | `KEEP_PASSES = True` or `--keep-passes` |
| OOM / not enough VRAM | `WHISPER_BATCH_SIZE = 4` |
| Python in another path | `PY312` in the `.bat` files **and** `PYTHON` in `pipeline_en_master.py` |
| Official HF hub | empty `HF_ENDPOINT=` in `.bat` and pipeline |
| Re-translate after editing EN | `translate_srt.py` only |
| Re-arbitrate without Whisper | `--keep-passes` once, then `--skip-whisperx` |

Defaults live in the `# CONFIG` block at the **top** of each file (no need to hunt through the code).


## Vertical videos (9:16)

`python burn_subs.py --video clip.mp4 --srt "clip subs.es.srt"` burns the subtitles into a
1080x1920 MP4 for Shorts / Reels / TikTok: the picture is padded over a blurred copy of itself
(`--mode pad`, default) or cropped (`--mode crop`), and the lines sit above the apps' buttons.
