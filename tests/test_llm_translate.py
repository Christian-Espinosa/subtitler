"""The LLM translator with a fake model: no network, no key."""
import json
import unittest

from tests._base import ROOT  # noqa: F401

import llm_translate as lt


def provider(name, key="k"):
    p = lt.Provider(name, "http://x", "m", "")
    p.key = lambda: key
    return p


def answer(mapping):
    return json.dumps({"cues": [{"i": i, "text": t} for i, t in mapping.items()]}, ensure_ascii=False)


CUES = [(0.0, 1.2, "Muy buenas gente"), (1.3, 4.0, "Hoy vamos a probar los subtítulos.")]


class Budget(unittest.TestCase):
    def test_budget_follows_duration_and_script(self):
        self.assertEqual(lt.budget(0.0, 2.0, "en"), 30)    # 0.9 * 2 s * 17
        self.assertEqual(lt.budget(0.0, 2.0, "zh"), 14)    # 0.9 * 2 s * 8
        self.assertEqual(lt.budget(1.0, 1.0, "en"), 4)     # never zero


class Parse(unittest.TestCase):
    def test_valid_with_reasoning_and_prose_around(self):
        raw = "<think>hmm {\"no\": 1}</think> Here: " + answer({1: "Hey everyone", 2: "Today we test the subs."})
        self.assertEqual(lt.parse_answer(raw, [1, 2])[1], "Hey everyone")

    def test_missing_id_empty_text_or_garbage_is_rejected(self):
        self.assertIsNone(lt.parse_answer(answer({1: "Hey"}), [1, 2]))
        self.assertIsNone(lt.parse_answer(answer({1: "Hey", 2: "  "}), [1, 2]))
        self.assertIsNone(lt.parse_answer("Sorry, I cannot.", [1]))
        self.assertIsNone(lt.parse_answer('{"cues": "x"}', [1]))


class Translate(unittest.TestCase):
    def test_bad_json_from_nvidia_falls_back_to_the_next_provider(self):
        calls = []

        def chat(p, messages):
            calls.append(p.name)
            return "not json" if p.name == "nvidia" else answer({1: "Hey everyone", 2: "Today we test the subtitles."})

        rep = lt.Report()
        out = lt.translate_cues(CUES, "es", "en", chat=chat, providers=[provider("nvidia"), provider("deepseek")],
                                report=rep, sleep=lambda s: None)
        self.assertEqual(out, ["Hey everyone", "Today we test the subtitles."])
        self.assertEqual(calls, ["nvidia"] * 3 + ["deepseek"])
        self.assertEqual(rep.provider_used, ["deepseek"])

    def test_too_long_is_rewritten_shorter_once(self):
        long = "Good evening to each and every one of you, dear people watching"
        seen = []

        def chat(p, messages):
            user = messages[-1]["content"]
            seen.append("SHORTER" in user.upper() and "rewrite" in user.lower())
            return answer({1: "Hey all"}) if seen[-1] else answer({1: long, 2: "Today we test the subtitles."})

        rep = lt.Report()
        out = lt.translate_cues(CUES, "es", "en", chat=chat, providers=[provider("nvidia")], report=rep,
                                sleep=lambda s: None)
        self.assertEqual(out[0], "Hey all")
        self.assertEqual(rep.rewritten, 1)
        self.assertEqual(rep.over_budget, 0)
        self.assertEqual(seen, [False, True])

    def test_whole_video_goes_as_context(self):
        got = {}

        def chat(p, messages):
            got["user"] = messages[-1]["content"]
            return answer({1: "Hey everyone", 2: "Today we test the subtitles."})

        lt.translate_cues(CUES, "es", "en", chat=chat, providers=[provider("nvidia")], sleep=lambda s: None)
        self.assertIn("Muy buenas gente Hoy vamos a probar", got["user"])
        self.assertIn('"max_chars": 18', got["user"])

    def test_no_key_says_so(self):
        with self.assertRaises(RuntimeError) as e:
            lt.translate_cues(CUES, "es", "en", providers=[provider("nvidia", key="")])
        self.assertIn("NVIDIA_API_KEY", str(e.exception))


class Tone(unittest.TestCase):
    def test_tone_per_cue_is_kept_and_unknown_ones_dropped(self):
        def chat(p, messages):
            return json.dumps({"cues": [{"i": 1, "text": "Hey everyone!", "tone": "Happy"},
                                        {"i": 2, "text": "Today we test the subs.", "tone": "furious-ish"}]})

        rep = lt.Report()
        lt.translate_cues(CUES, "es", "en", chat=chat, providers=[provider("nvidia")], report=rep,
                          sleep=lambda s: None)
        self.assertEqual(rep.tones, {0: "happy"})

    def test_no_tone_is_fine(self):
        self.assertEqual(lt.parse_tones(answer({1: "Hey"})), {})


if __name__ == "__main__":
    unittest.main()
