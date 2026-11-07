#!/usr/bin/env python3
"""Render Fox radio: plays the music folder forever as AAC (ADTS) on stdout.

The music never changes while it plays, so each track is encoded to AAC once (stereo,
44.1 kHz, 128 kbit/s, as docs/youtube-ingest.md asks) into a cache beside the music and
the stream copies those frames: nothing is decoded or encoded while the stream runs.
Output is paced against the engine's frame clock, so the encoder receives audio at
exactly the rate it receives video. The title of the track that is audible right now is
written to the now playing file at the moment its first frame is due, which is what the
overlay card shows.

Order is a shuffle bag: every track plays once before any repeats, and a new bag never
opens with the track that closed the last one. The folder is rescanned for every bag, so
new music is picked up without a restart (and encoded, at low priority, the first time
it comes up). If there is no playable music the radio sends silence rather than stopping,
because the encoder treats the end of the audio as the end of the stream.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

RATE = 44100
SAMPLES = 1024               # per AAC frame, about 23 ms
LEAD = 0.5                   # seconds of audio kept ahead of the clock
BITRATE = "128k"
EXTENSIONS = {".flac", ".ogg", ".opus", ".wav", ".mp3", ".m4a"}


def log(msg):
    print(f"radio: {msg}", file=sys.stderr, flush=True)


def title_of(path):
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=title", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=20, check=True)
        tags = json.loads(probe.stdout).get("format", {}).get("tags", {})
        title = next((v for k, v in tags.items() if k.lower() == "title"), "").strip()
        if title:
            return title
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        log(f"no tags for {path.name}: {e}")
    stem = path.stem.lstrip("0123456789 _-.")
    return stem.replace("_", " ").replace("-", " ").strip() or path.stem


def wait_for_epoch(path, since, timeout=900):
    """The engine writes its frame clock's zero once it starts emitting frames. That may
    be before this interpreter is even up, so a file is judged by the zero it holds: a
    leftover from an earlier run holds one long past."""
    while time.time() < since + timeout:
        try:
            epoch = float(path.read_text().strip())
            if epoch >= since - 30:
                return epoch
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.05)
    log(f"no frame clock in {path} after {timeout} s, giving up")
    sys.exit(1)


def adts_frames(data):
    """The ADTS frames of an AAC stream, each with its header."""
    at = 0
    while at + 7 <= len(data):
        if data[at] != 0xFF or data[at + 1] & 0xF6 != 0xF0:
            raise ValueError(f"no ADTS sync at byte {at}")
        length = ((data[at + 3] & 3) << 11) | (data[at + 4] << 3) | (data[at + 5] >> 5)
        if length < 7 or at + length > len(data):
            raise ValueError(f"bad ADTS frame length at byte {at}")
        yield data[at:at + length]
        at += length


class Clock:
    def __init__(self, epoch):
        self.epoch = epoch
        self.samples = 0
        self.pending = []    # (due, title) not yet shown

    def due(self):
        return self.epoch + self.samples / RATE

    def announce(self, title):
        self.pending.append((self.due(), title))

    def write(self, out, frames):
        """AAC frames, sent once they are due less the lead."""
        wait = self.due() - LEAD - time.time()
        if wait > 0:
            time.sleep(wait)
        out.write(b"".join(frames))
        self.samples += SAMPLES * len(frames)
        self.flush_titles()

    def flush_titles(self):
        now = time.time()
        while self.pending and self.pending[0][0] <= now + 0.05:
            _, title = self.pending.pop(0)
            write_now_playing(title)


NOW_PLAYING = None


def write_now_playing(title):
    tmp = NOW_PLAYING.with_suffix(".tmp")
    tmp.write_text(title + "\n")
    os.replace(tmp, NOW_PLAYING)
    log(f"now playing: {title or '(nothing)'}")


class Library:
    """Each track as AAC frames, encoded once into `cache` (named by the track's size and
    time, so an edited file is encoded again) and read from there afterwards."""

    def __init__(self, cache, gain_db):
        self.cache = cache
        self.gain_db = gain_db
        cache.mkdir(parents=True, exist_ok=True)
        self.silence = adts_frames_list(self.encode_silence())

    def path(self, track):
        st = track.stat()
        gain = f"-g{self.gain_db:+g}" if self.gain_db else ""
        return self.cache / f"{track.stem}-{st.st_size}-{int(st.st_mtime)}{gain}.aac"

    def ready(self, track):
        return self.path(track).exists()

    def encode(self, track):
        """Encodes the track at the lowest priority; True when its AAC is there."""
        out = self.path(track)
        if out.exists():
            return True
        tmp = out.with_suffix(".part")
        cmd = ["nice", "-n", "19", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(track), "-map", "0:a:0", "-vn"]
        if self.gain_db:
            cmd += ["-af", f"volume={self.gain_db}dB"]
        cmd += ["-c:a", "aac", "-b:a", BITRATE, "-ar", str(RATE), "-ac", "2", "-f", "adts", str(tmp)]
        if subprocess.run(cmd).returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            log(f"cannot encode {track.name}")
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, out)
        return True

    def frames(self, track):
        return adts_frames_list(self.path(track).read_bytes())

    def encode_silence(self):
        out = self.cache / f"silence-{BITRATE}.aac"
        if not out.exists():
            tmp = out.with_suffix(".part")
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"anullsrc=r={RATE}:cl=stereo",
                            "-t", "2", "-c:a", "aac", "-b:a", BITRATE, "-f", "adts", str(tmp)], check=True)
            os.replace(tmp, out)
        return out.read_bytes()


def adts_frames_list(data):
    try:
        return list(adts_frames(data))
    except ValueError as e:
        log(f"unreadable AAC: {e}")
        return []


def silence(clock, out, lib, seconds):
    # The middle of the silent encode: its first frames carry the encoder's start-up.
    frames = lib.silence[4:-2] or lib.silence
    n = int(seconds * RATE / SAMPLES)
    i = 0
    while i < n:
        chunk = [frames[(i + k) % len(frames)] for k in range(min(2, n - i))]
        clock.write(out, chunk)
        i += len(chunk)


def play(clock, out, frames):
    for i in range(0, len(frames), 2):
        clock.write(out, frames[i:i + 2])


def scan(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())


def prepare(lib, tracks, titles):
    """Titles and AAC for every track, in the background: a track change must not wait
    for ffprobe or an encode, or the encoder starves and the picture freezes."""
    for track in tracks:
        if track not in titles:
            titles[track] = title_of(track)
        lib.encode(track)


def main():
    global NOW_PLAYING
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True, type=Path)
    ap.add_argument("--now-playing", required=True, type=Path)
    ap.add_argument("--cache", type=Path, help="where the tracks' AAC is kept (default: .aac-cache in the music folder)")
    ap.add_argument("--epoch-file", type=Path, help="wait for the engine's frame clock; without it, start now")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds of silence between tracks")
    ap.add_argument("--gain", type=float, default=0.0, help="dB applied to every track")
    args = ap.parse_args()
    NOW_PLAYING = args.now_playing
    NOW_PLAYING.parent.mkdir(parents=True, exist_ok=True)
    write_now_playing("")
    lib = Library(args.cache or args.music / ".aac-cache", args.gain)

    started = time.time()
    epoch = wait_for_epoch(args.epoch_file, started) if args.epoch_file else started
    clock = Clock(epoch)
    out = open(sys.stdout.fileno(), "wb", buffering=0, closefd=False)
    titles = {}
    last = None
    preparing = None
    try:
        silence(clock, out, lib, 1.5)
        while True:
            bag = scan(args.music)
            if not bag:
                clock.announce("")
                silence(clock, out, lib, 10)
                continue
            random.shuffle(bag)
            if len(bag) > 1 and bag[0] == last:
                bag.append(bag.pop(0))
            if preparing is None or not preparing.is_alive():
                preparing = threading.Thread(target=prepare, args=(lib, bag, titles), daemon=True)
                preparing.start()
            for track in bag:
                if not lib.ready(track):
                    # Not encoded yet (new music): later in the bag, or skipped this time.
                    continue
                frames = lib.frames(track)
                if last is not None:
                    clock.announce("")
                    silence(clock, out, lib, args.gap)
                if track not in titles:
                    titles[track] = title_of(track)
                if not frames:
                    titles.pop(track, None)
                    log(f"{track.name} produced no audio, skipping")
                    continue
                clock.announce(titles[track])
                play(clock, out, frames)
                last = track
            if last is None:
                # Nothing encoded yet: the first track, now, rather than silence for long.
                clock.announce("")
                silence(clock, out, lib, 2)
    except BrokenPipeError:
        # The encoder went away; the service restarts the whole pipeline. Point stdout
        # at /dev/null so the interpreter's exit flush does not complain about it too.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()
