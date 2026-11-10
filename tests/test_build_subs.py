import json
import tempfile
import unittest
from pathlib import Path

from tests._base import align_json

import build_subs


class BuildSubs(unittest.TestCase):
    def test_cues_come_from_word_timings_and_keep_the_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Path(tmp) / "a.json"
            out = Path(tmp) / "a.srt"
            j.write_text(json.dumps(align_json("es")), encoding="utf-8")
            words, lang = build_subs.load_words(j)
            self.assertEqual(lang, "es")
            self.assertEqual(words[0].start, 0.0)
            self.assertTrue(words[-1].text.endswith("!"))  # unplaced token glued, no invented time
            self.assertEqual(build_subs.main(["--align", str(j), "--out", str(out)]), 0)
            srt = out.read_text(encoding="utf-8")
        self.assertIn("00:00:00,000 -->", srt)
        self.assertIn("Hola a todos.", srt)
        self.assertIn("subtítulos.", srt)


class Breaks(unittest.TestCase):
    def test_a_speaker_change_ends_the_cue_even_without_a_pause(self):
        """Dubbing: two speakers back to back, the aligner glued the silence to the
        last word, so only the break says where one ends and the other starts."""
        W = build_subs.Word
        words = [W(0.0, 0.4, "Hola,"), W(0.45, 0.9, "¿qué"), W(0.95, 1.5, "tal?"),
                 W(1.55, 1.9, "Muy"), W(1.95, 2.4, "bien.")]
        self.assertEqual(len(build_subs.split_utterances(words, 0.4)), 1)
        parts = build_subs.split_utterances(words, 0.4, [1.53])
        self.assertEqual([build_subs.join_words(p) for p in parts], ["Hola, ¿qué tal?", "Muy bien."])

    def test_short_cues_are_not_glued_across_a_break(self):
        C = build_subs.Cue
        cues = [C(0.0, 0.5, "Sí."), C(0.6, 1.0, "No.")]
        kw = dict(max_chars=84, max_lines=2, max_sec=6.0, pause=0.4)
        self.assertEqual(len(build_subs.merge_short(cues, **kw)), 1)
        self.assertEqual(len(build_subs.merge_short(cues, **kw, breaks=[0.55])), 2)

    def test_breaks_file_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_text("[2.0, 1.0]")
            b.write_text('{"breaks": [3.5]}')
            self.assertEqual(build_subs.load_breaks(a), [1.0, 2.0])
            self.assertEqual(build_subs.load_breaks(b), [3.5])
            self.assertEqual(build_subs.load_breaks(None), [])


if __name__ == "__main__":
    unittest.main()
