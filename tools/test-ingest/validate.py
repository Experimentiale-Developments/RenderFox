#!/usr/bin/env python3
"""Check what the dummy ingest received against YouTube's ingest spec.

Reads the dummy's API for liveness and its newest recording for the stream itself,
then prints one line per check. Exit status 0 only if every check passed.

    validate.py [--segment FILE] [--api http://127.0.0.1:9996] [--seconds 60]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PATH = "live2/rf-dummy-key"
FPS = 24  # what Render Fox sends; YouTube takes 24 to 60
HERE = Path(__file__).resolve().parent
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {detail}")


def api(base, route):
    with urllib.request.urlopen(f"{base}{route}", timeout=5) as r:
        return json.load(r)


def probe(args):
    out = subprocess.run(["ffprobe", "-v", "error", "-of", "json"] + args, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def newest_segment():
    segments = sorted((HERE / "recordings").rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return segments[-1] if segments else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=Path)
    ap.add_argument("--api", default="http://127.0.0.1:9996")
    ap.add_argument("--seconds", type=float, default=60, help="how much of the segment's end to analyse")
    args = ap.parse_args()

    # Liveness: the path is being published and bytes keep arriving at the right rate.
    try:
        p1 = api(args.api, f"/v3/paths/get/{PATH}")
        time.sleep(5)
        p2 = api(args.api, f"/v3/paths/get/{PATH}")
        rate = (p2["bytesReceived"] - p1["bytesReceived"]) * 8 / 5 / 1e6
        check("publisher connected", p2.get("ready", False), f"source {p2.get('source', {}).get('type')}")
        check("tracks", sorted(p2.get("tracks", [])) == ["H264", "MPEG-4 Audio"], ", ".join(p2.get("tracks", [])))
        check("arrival rate", 9.0 <= rate <= 11.5, f"{rate:.2f} Mbit/s over 5 s (10.13 expected)")
    except OSError as e:
        check("dummy API reachable", False, str(e))

    segment = args.segment or newest_segment()
    if not segment:
        check("recording", False, "no segments recorded")
        return
    print(f"      segment {segment}")
    info = probe(["-show_streams", "-show_format", str(segment)])
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if not video or not audio:
        check("streams", False, "missing video or audio")
        return

    check("video codec", video["codec_name"] == "h264", video["codec_name"])
    check("profile", video.get("profile") in ("High", "Main"), f"{video.get('profile')} (CABAC, B-frames allowed)")
    check("resolution", (video["width"], video["height"]) == (1920, 1080), f"{video['width']}x{video['height']}")
    check("pixel format", video.get("pix_fmt") == "yuv420p", video.get("pix_fmt"))
    check("level", video.get("level", 0) <= 42, str(video.get("level")))
    check("progressive", video.get("field_order", "progressive") in ("progressive", "unknown"),
          video.get("field_order", "progressive"))
    check("colour", video.get("color_space") == "bt709", f"{video.get('color_space')}/{video.get('color_range')}")

    duration = float(info["format"]["duration"])
    start = max(0.0, duration - args.seconds)
    vp = probe(["-select_streams", "v:0", "-read_intervals", f"{start}%", "-show_entries",
                "packet=pts_time,dts_time,size,flags", str(segment)])["packets"]
    apk = probe(["-select_streams", "a:0", "-read_intervals", f"{start}%", "-show_entries",
                 "packet=pts_time,size", str(segment)])["packets"]
    pts = [float(p["pts_time"]) for p in vp]
    span = max(pts) - min(pts) + 1 / FPS
    check("frame rate", abs(len(vp) / span - FPS) < 0.1, f"{len(vp) / span:.3f} fps over {span:.1f} s ({FPS} target)")
    deltas = sorted(b - a for a, b in zip(sorted(pts), sorted(pts)[1:]))
    irregular = sum(1 for d in deltas if abs(d - 1 / FPS) > 0.002)
    check("constant frame rate", irregular == 0, f"{irregular} irregular frame intervals")
    reordered = sum(1 for p in vp if p.get("dts_time") and float(p["pts_time"]) > float(p["dts_time"]) + 1e-6)
    check("B-frames", reordered > 0, f"{reordered} reordered frames")

    keys = [float(p["pts_time"]) for p in vp if "K" in p["flags"]]
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    check("keyframe interval", bool(gaps) and all(abs(g - 2) < 0.05 for g in gaps),
          f"{len(keys)} keyframes, gaps {min(gaps, default=0):.2f}..{max(gaps, default=0):.2f} s (2 s, never over 4)")

    vrate = sum(int(p["size"]) for p in vp) * 8 / span / 1000
    check("video bitrate (CBR)", 9000 <= vrate <= 11000, f"{vrate:.0f} kbit/s (10000 target)")
    # CBR means steady, not just the right average: no 2 s window far off the target.
    windows = {}
    for p in vp:
        windows.setdefault(int(float(p["pts_time"]) // 2), 0)
        windows[int(float(p["pts_time"]) // 2)] += int(p["size"]) * 8 / 2 / 1000
    inner = list(windows.values())[1:-1]
    check("bitrate steadiness", bool(inner) and min(inner) > 6250 and max(inner) < 15000,
          f"2 s windows {min(inner, default=0):.0f}..{max(inner, default=0):.0f} kbit/s")

    check("audio codec", audio["codec_name"] == "aac" and audio.get("profile") == "LC",
          f"{audio['codec_name']} {audio.get('profile')}")
    check("sample rate", audio["sample_rate"] == "44100", audio["sample_rate"])
    check("channels", audio["channels"] == 2, str(audio["channels"]))
    apts = [float(p["pts_time"]) for p in apk]
    aspan = max(apts) - min(apts) + 1024 / 44100
    arate = sum(int(p["size"]) for p in apk) * 8 / aspan / 1000
    check("audio bitrate", 110 <= arate <= 140, f"{arate:.0f} kbit/s (128 target)")
    drift = (max(apts) + 1024 / 44100) - (max(pts) + 1 / FPS)
    check("audio/video alignment", abs(drift) < 0.5, f"ends {drift * 1000:+.0f} ms apart")

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
    sys.exit(0 if results and all(results) else 1)
