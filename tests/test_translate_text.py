"""The text side of translate_srt.py: no model needed."""
import unittest

from tests._base import ROOT  # noqa: F401  (puts the repo on sys.path)

import translate_srt as t


class Chinese(unittest.TestCase):
    def test_a_sentence_over_three_cues_is_shared_out_not_dumped_on_the_first(self):
        # Before: Chinese has no spaces, so it was "one word" and cue 2 and 3 came out empty.
        src = ["Hoy vamos a probar", "los subtítulos", "en tres idiomas."]
        out = t.redistribute(src, "今天我们要用三种语言测试字幕。")
        self.assertEqual(len(out), 3)
        self.assertTrue(all(p for p in out), out)
        self.assertEqual("".join(out), "今天我们要用三种语言测试字幕。")
        self.assertNotIn(" ", "".join(out))

    def test_spanish_still_splits_by_words(self):
        out = t.redistribute(["Hello everyone", "and welcome"], "Hola a todos y bienvenidos")
        self.assertEqual(" ".join(out), "Hola a todos y bienvenidos")
        self.assertTrue(all(" " not in p[:1] for p in out))

    def test_chinese_lines_wrap_by_character_at_half_the_width(self):
        text = "这是一个非常长的中文句子，用来测试字幕在屏幕上是否会被正确地分成两行。"
        wrapped = t.wrap_lines(text, 42, 2)
        lines = wrapped.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(x) <= 21 for x in lines), lines)
        self.assertEqual("".join(lines), text)

    def test_a_chinese_cue_too_long_for_the_screen_is_split_in_time(self):
        cue = t.Cue("00:00:00,000", "00:00:06,000", "这" * 60)
        out = t.split_overflow([cue], 42, 2)
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(out[0].start_ts, "00:00:00,000")
        self.assertEqual(out[-1].end_ts, "00:00:06,000")
        self.assertEqual("".join(c.text for c in out), "这" * 60)

    def test_full_width_punctuation(self):
        self.assertEqual(t.zh_punctuation("您好,大家!今天我们测试."), "您好，大家！今天我们测试。")
        self.assertEqual(t.zh_punctuation("版本 3.8 很好"), "版本 3.8 很好")


if __name__ == "__main__":
    unittest.main()
