"""Pipeline setup: no path to a particular PC, mirror only if asked, real language."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._base import ROOT, align_json

import pipeline_aligned
import transcribe


class NoPrivatePaths(unittest.TestCase):
    def test_no_pc_path_or_forced_mirror_in_the_code(self):
        for f in list(ROOT.glob("*.py")) + list(ROOT.glob("*.bat")) + list((ROOT / "safe_downloader").glob("*.ps1")):
            text = f.read_text(encoding="utf-8", errors="replace")
            self.assertNotIn("chres", text, f.name)
            self.assertNotIn("--trusted-host", text, f.name)
        for f in list(ROOT.glob("*.py")) + list(ROOT.glob("*.bat")):
            self.assertNotIn('"https://hf-mirror.com"', f.read_text(encoding="utf-8", errors="replace"), f.name)
        self.assertEqual(pipeline_aligned.PYTHON, __import__("sys").executable)


class Hub(unittest.TestCase):
    def test_reachable_hub_is_used(self):
        env = {}
        self.assertIn("huggingface.co", pipeline_aligned.ensure_hub(env, reachable=lambda: True))
        self.assertNotIn("HF_HUB_OFFLINE", env)

    def test_blocked_hub_uses_the_local_cache_not_a_mirror(self):
        env = {}
        self.assertIn("offline", pipeline_aligned.ensure_hub(env, reachable=lambda: False))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("HF_ENDPOINT", env)

    def test_an_explicit_mirror_wins(self):
        env = {"HF_ENDPOINT": "https://example.org"}
        self.assertIn("example.org", pipeline_aligned.ensure_hub(env, reachable=lambda: False))
        self.assertNotIn("HF_HUB_OFFLINE", env)


class Language(unittest.TestCase):
    def test_the_json_keeps_the_detected_language(self):
        """whisperx 3.8's CLI wrote "en" for a Spanish clip; transcribe.py writes what was detected."""
        self.assertEqual(transcribe.result_json({"segments": [1]}, "es")["language"], "es")

    def test_cpu_gets_int8_and_small(self):
        self.assertEqual(transcribe.pick_compute_type("cpu"), "int8")
        self.assertEqual(transcribe.pick_compute_type("cuda"), "float16")
        with mock.patch.object(transcribe, "pick_device", return_value="cpu"):
            self.assertEqual(pipeline_aligned.resolve_model("auto", "auto"), "small")

    def test_tracks_are_named_by_the_detected_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            media = tmp / "clip.mp4"
            media.write_bytes(b"not really a video")
            (tmp / "_work").mkdir()
            (tmp / "_work" / "clip.json").write_text(json.dumps(align_json("es")), encoding="utf-8")
            with mock.patch.object(pipeline_aligned, "hub_reachable", return_value=True):
                rc = pipeline_aligned.main([str(media), "--skip-whisperx", "--skip-translate", "--langs", "es,en"])
            self.assertEqual(rc, 0)
            self.assertTrue((tmp / "clip subs.es.srt").is_file())


if __name__ == "__main__":
    unittest.main()
