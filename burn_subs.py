"""
Burn an SRT into a vertical 9:16 video (Shorts / Reels / TikTok / Douyin).

The SRT is turned into an ASS file whose canvas is the real output size
(1080x1920), so font size and margins are pixels, not libass' 288-line guess.
ffmpeg then reframes the video and burns the subtitles in one pass.

Reframe modes:
  pad   (default) the whole picture, scaled to the width, over a blurred copy
        of itself: nothing is cut, works for any source.
  crop  fill the screen by cutting the sides: best when the subject is centred.

Usage:
  python burn_subs.py --video clip.mp4 --srt "clip subs.es.srt"
  python burn_subs.py --video clip.mp4 --srt "clip subs.zh.srt" --mode crop --out short.mp4
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FONT_SIZE = 64          # px on the 1080x1920 canvas
MARGIN_V = 380          # px from the bottom: above the app buttons of Shorts/Reels
OUTLINE = 4

SRT_BLOCK = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)
CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def default_font(text: str) -> str:
    """A font that has the glyphs. libass falls back through fontconfig anyway."""
    if CJK.search(text):
        return "Microsoft YaHei" if os.name == "nt" else "Noto Sans CJK SC"
    return "Arial"


def srt_to_ass_time(ts: str) -> str:
    h, m, rest = ts.replace(",", ".").split(":")
    s, ms = rest.split(".")
    return f"{int(h)}:{m}:{s}.{ms[:2]}"


def parse_srt(text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in SRT_BLOCK.finditer(text.replace("\r\n", "\n").strip() + "\n\n"):
        body = m.group(4).strip()
        if body:
            out.append((m.group(2), m.group(3), body))
    return out


FONT = re.compile(r'<font\s+color="?#([0-9A-Fa-f]{6})"?\s*>(.*?)</font>', re.DOTALL | re.IGNORECASE)
TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def ass_colour(hex6: str) -> str:
    """#RRGGBB -> ASS &HBBGGRR& (ASS is blue-green-red)."""
    r, g, b = hex6[0:2], hex6[2:4], hex6[4:6]
    return f"&H{b}{g}{r}&".upper()


def ass_escape(text: str) -> str:
    """SRT text -> ASS. `<font color="#RRGGBB">` (speaker colours) becomes an ASS colour
    override; any other tag is dropped; braces would be override codes, so they become ()."""
    t = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    t = FONT.sub(lambda m: "{\\c" + ass_colour(m.group(1)) + "}" + m.group(2) + "{\\r}", t)
    return TAG.sub("", t).replace("\n", "\\N")


def srt_to_ass(srt_text: str, *, font: str = "", size: int = FONT_SIZE, margin_v: int = MARGIN_V,
               width: int = WIDTH, height: int = HEIGHT) -> str:
    cues = parse_srt(srt_text)
    font = font or default_font(" ".join(c[2] for c in cues))
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{OUTLINE},1,2,60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [f"Dialogue: 0,{srt_to_ass_time(a)},{srt_to_ass_time(b)},Default,,0,0,0,,{ass_escape(t)}"
             for a, b, t in cues]
    return head + "\n".join(lines) + "\n"


def video_filter(ass_name: str, mode: str = "pad", width: int = WIDTH, height: int = HEIGHT) -> str:
    """The ffmpeg -filter_complex. `ass_name` is relative: ffmpeg runs in its folder,
    so Windows paths (`C:\\...`) never have to be escaped inside the filter."""
    if mode == "crop":
        frame = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                 f"crop={width}:{height},setsar=1[v0]")
    elif mode == "pad":
        frame = (f"[0:v]split=2[bg][fg];"
                 f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
                 f"boxblur=20:2[bgb];"
                 f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
                 f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    else:
        raise ValueError(f"unknown mode: {mode} (pad|crop)")
    return f"{frame};[v0]ass={ass_name}[v]"


def ffmpeg_cmd(video: Path, ass_name: str, out: Path, mode: str = "pad") -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-i", str(video.resolve()),
            "-filter_complex", video_filter(ass_name, mode),
            "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            str(out.resolve())]


def default_out(video: Path, srt: Path) -> Path:
    m = re.search(r"\.([a-z]{2,3})\.srt$", srt.name)
    lang = f".{m.group(1)}" if m else ""
    return video.with_name(f"{video.stem} 9x16{lang}.mp4")


def burn(video: Path, srt: Path, out: Path | None = None, *, mode: str = "pad", font: str = "",
         size: int = FONT_SIZE) -> Path:
    out = out or default_out(video, srt)
    ass = srt_to_ass(srt.read_text(encoding="utf-8-sig", errors="replace"), font=font, size=size)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "subs.ass").write_text(ass, encoding="utf-8")
        proc = subprocess.run(ffmpeg_cmd(video, "subs.ass", out, mode), cwd=tmp)
    if proc.returncode != 0 or not out.is_file():
        raise SystemExit(f"[ERROR] ffmpeg could not burn the subtitles into {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Burn an SRT into a 9:16 video")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--srt", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--mode", default="pad", choices=["pad", "crop"])
    p.add_argument("--font", default="", help="Font name (default: Arial, or a CJK font for Chinese)")
    p.add_argument("--size", type=int, default=FONT_SIZE)
    args = p.parse_args(argv)
    for f in (args.video, args.srt):
        if not f.is_file():
            print(f"[ERROR] not found: {f}", file=sys.stderr)
            return 1
    out = burn(args.video, args.srt, args.out, mode=args.mode, font=args.font, size=args.size)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
