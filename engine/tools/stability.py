#!/usr/bin/env python3
"""How steady and how sharp a rendered sequence is.

    stability.py frames.gray [...]     luma frames from `rfengine --bench N --dump frames.gray`
    stability.py clip.mp4 [...]        or a recorded video (decoded with ffmpeg)
    --size WxH                          the size of .gray dumps, 1920x1080 by default
    --cycle N                           frames per full render (staticEvery), 7 by default

jitter      frame-to-frame motion of tiles (64 px at 720p, scaled with the height), minus
            their smooth motion: how much the picture shakes, in 720p pixels so numbers
            compare across sizes. The orbit itself moves well under a pixel a frame, so
            anything near a pixel is visible as wobble. Rendering every frame measures
            about 0.03.
sharpness   mean luma gradient in the middle of the frame. Only comparable between runs
            of the same angle.
pulse       spread of sharpness across the full/reprojected cycle, in percent.
            A few percent is invisible; the reprojection bug that looked like heat haze
            measured 0.58 px jitter and a 5 % pulse.

Needs numpy. Skips the first 20 frames while the temporal history fills.
"""
import subprocess
import sys

import numpy as np

def frames(path, size):
    if path.endswith(".gray"):
        w, h = size
        raw = np.fromfile(path, np.uint8)
    else:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                "-of", "csv=p=0", path], capture_output=True, check=True, text=True).stdout
        w, h = map(int, probe.strip().split(","))
        raw = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "150", "-f", "rawvideo",
                                            "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout, np.uint8)
    return raw.reshape(-1, h, w).astype(np.float32)[20:]


def shift(a, b):
    """Sub-pixel translation from b to a by phase correlation."""
    tile = a.shape[0]
    window = np.hanning(tile)[:, None] * np.hanning(tile)[None]
    r = np.fft.fft2(a * window) * np.conj(np.fft.fft2(b * window))
    r = np.fft.ifft2(r / (np.abs(r) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    out = []
    for axis, c in ((1, x), (0, y)):
        lo = r[(y - (axis == 0)) % tile, (x - (axis == 1)) % tile]
        hi = r[(y + (axis == 0)) % tile, (x + (axis == 1)) % tile]
        v = c + (lo - hi) / (2 * (lo - 2 * r[y, x] + hi) + 1e-9)
        out.append(v - tile if v > tile / 2 else v)
    return out


def report(path, size, cycle):
    f = frames(path, size)
    h, w = f.shape[1:]
    k = h / 720  # the layout below was chosen at 720p
    tile = 2 * round(32 * k)
    jitter = []
    for ty in range(round(200 * k), round(640 * k), round(110 * k)):
        for tx in range(round(80 * k), round(1200 * k), round(150 * k)):
            d = np.array([shift(f[i + 1, ty:ty + tile, tx:tx + tile], f[i, ty:ty + tile, tx:tx + tile])
                          for i in range(len(f) - 1)]) / k
            smooth = np.array([np.convolve(d[:, c], np.ones(9) / 9, mode="same") for c in range(2)]).T
            jitter.append(np.sqrt(((d - smooth)[9:-9] ** 2).mean()))
    mid = f[:, round(150 * k):round(650 * k), round(100 * k):round(1180 * k)]
    sharp = np.array([np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean() for g in mid])
    phase = np.array([sharp[p::cycle].mean() for p in range(cycle)])
    print(f"{path}: jitter {np.mean(jitter):.3f} px (worst tile {np.max(jitter):.3f}), "
          f"sharpness {sharp.mean():.2f}, pulse {100 * (phase.max() - phase.min()) / phase.mean():.1f} %")


if __name__ == "__main__":
    args, size, cycle = sys.argv[1:], (1920, 1080), 7
    if "--size" in args:
        i = args.index("--size")
        size = tuple(map(int, args[i + 1].split("x")))
        del args[i:i + 2]
    if "--cycle" in args:
        i = args.index("--cycle")
        cycle = int(args[i + 1])
        del args[i:i + 2]
    for p in args:
        report(p, size, cycle)
