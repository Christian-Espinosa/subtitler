import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._base import ROOT  # noqa: F401

import burn_subs as b

SRT = "1\n00:00:00,000 --> 00:00:01,500\nHola a todos\n\n2\n00:00:01,600 --> 00:00:02,000\n{raro} línea\nsegunda\n"


class Ass(unittest.TestCase):
    def test_canvas_is_the_real_9x16_size_and_times_convert(self):
        ass = b.srt_to_ass(SRT)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("PlayResY: 1920", ass)
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,Hola a todos", ass)
        self.assertIn("(raro) línea\\Nsegunda", ass)  # braces would be ASS override codes

    def test_speaker_colour_becomes_an_ass_override(self):
        ass = b.srt_to_ass('1\n00:00:00,000 --> 00:00:01,000\n<font color="#FFD23F">Hola <i>tú</i></font>\n')
        self.assertIn("{\\c&H3FD2FF&}Hola tú{\\r}", ass)

    def test_chinese_gets_a_cjk_font(self):
        font = b.srt_to_ass("1\n00:00:00,000 --> 00:00:01,000\n大家好\n").split("Style: Default,")[1].split(",")[0]
        self.assertIn(font, ("Noto Sans CJK SC", "Microsoft YaHei"))

    def test_filter_modes(self):
        self.assertIn("crop=1080:1920", b.video_filter("subs.ass", "crop"))
        self.assertIn("boxblur", b.video_filter("subs.ass", "pad"))
        self.assertTrue(b.video_filter("subs.ass").endswith("[v0]ass=subs.ass[v]"))
        with self.assertRaises(ValueError):
            b.video_filter("subs.ass", "stretch")

    def test_output_name_carries_the_language(self):
        self.assertEqual(b.default_out(Path("/v/clip.mp4"), Path("/v/clip subs.es.srt")).name, "clip 9x16.es.mp4")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class RealBurn(unittest.TestCase):
    def test_two_second_clip_comes_out_1080x1920_with_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = tmp / "clip con espacios.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=640x360:d=2",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
                            "-c:v", "libx264", "-c:a", "aac", str(clip)], check=True)
            srt = tmp / "clip con espacios subs.es.srt"
            srt.write_text(SRT, encoding="utf-8")
            out = b.burn(clip, srt)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                                    "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout
        self.assertIn("1080,1920", probe)
        self.assertIn("audio", probe)


if __name__ == "__main__":
    unittest.main()
