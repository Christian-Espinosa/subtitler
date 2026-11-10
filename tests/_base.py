import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def align_json(language: str = "es") -> dict:
    """A tiny WhisperX alignment: two sentences, word timings, one unplaced token."""
    words = [("Hola", 0.0, 0.4), ("a", 0.45, 0.5), ("todos.", 0.55, 1.0),
             ("Hoy", 1.6, 1.8), ("probamos", 1.85, 2.3), ("los", 2.35, 2.5), ("subtítulos.", 2.55, 3.2)]
    segs = [{"start": 0.0, "end": 3.2, "text": "Hola a todos. Hoy probamos los subtítulos.",
             "words": [{"word": w, "start": a, "end": b} for w, a, b in words] + [{"word": "!"}]}]
    return {"segments": segs, "word_segments": [], "language": language}
