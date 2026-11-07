#!/usr/bin/env python3
"""CPU and memory of the render box's processes, sampled from /proc.

    measure.py [--every S] [--for S] [--csv out.csv] [--quiet]

Every S seconds (default 10) prints, per role, CPU as percent of one core over the
interval and resident memory (PSS, so shared pages are split fairly), then the box's
total as percent of all its cores. Roles are found by command line, so it measures the
real services on the box as well as bench/stack.sh:

    engine    rfengine
    radio     radio.py and the ffmpeg decoding the current track
    mux       the ffmpeg in encode.sh (copies video, encodes AAC)
    relay     MediaMTX with relay.yml
    uplink    ffmpeg copying the relay to an ingest (both of them)
    watchdog  relay/watchdog.py

Anything else (the fake server, the stand-in ingest) is not counted. At the end, the
mean over the whole run, and the peak of each.
"""
import argparse
import os
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")
CORES = os.cpu_count()
ROLES = ["engine", "radio", "mux", "relay", "uplink", "watchdog"]


def role(cmd):
    if not cmd:
        return None
    exe = os.path.basename(cmd[0])
    joined = " ".join(cmd)
    if exe == "rfengine" or "rfengine" in exe:
        return "engine"
    if "radio.py" in joined:
        return "radio"
    if "watchdog.py" in joined:
        return "watchdog"
    if exe == "mediamtx":
        return "relay" if "relay.yml" in joined else None
    if exe == "ffmpeg":
        if "-map 0:a:0" in joined:
            return "radio"
        if "audio.pipe" in joined:
            return "mux"
        if "rtmp://127.0.0.1:1935/fm" in joined and "-c copy" in joined:
            return "uplink"
    return None


def snapshot():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            r = role([c.decode(errors="replace") for c in cmd if c])
            if r is None:
                continue
            stat = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rindex(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
            pss = 0
            try:
                for line in open(f"/proc/{pid}/smaps_rollup"):
                    if line.startswith("Pss:"):
                        pss = int(line.split()[1])
                        break
            except OSError:
                pass
            out[int(pid)] = (r, ticks, pss)
        except (OSError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--for", dest="duration", type=float, default=0)
    ap.add_argument("--csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    csv = open(a.csv, "w") if a.csv else None
    if csv:
        csv.write("t," + ",".join(f"{r}_cpu,{r}_mb" for r in ROLES) + ",box_cpu,box_mb\n")
    start = time.time()
    last, last_t = snapshot(), time.time()
    sums = {r: [0.0, 0.0] for r in ROLES + ["box"]}
    peaks = {r: [0.0, 0.0] for r in ROLES + ["box"]}
    n = 0
    try:
        while not a.duration or time.time() - start < a.duration:
            time.sleep(a.every)
            now, now_t = snapshot(), time.time()
            dt = now_t - last_t
            cpu = {r: 0.0 for r in ROLES}
            mem = {r: 0.0 for r in ROLES}
            for pid, (r, ticks, pss) in now.items():
                before = last.get(pid)
                if before and before[0] == r:
                    cpu[r] += (ticks - before[1]) / TICK / dt * 100
                else:
                    cpu[r] += 0  # started this interval: counted from the next
                mem[r] += pss / 1024
            cpu["box"] = sum(cpu[r] for r in ROLES) / CORES
            mem["box"] = sum(mem[r] for r in ROLES)
            n += 1
            for r in ROLES + ["box"]:
                sums[r][0] += cpu[r]
                sums[r][1] += mem[r]
                peaks[r][0] = max(peaks[r][0], cpu[r])
                peaks[r][1] = max(peaks[r][1], mem[r])
            if not a.quiet:
                parts = " ".join(f"{r} {cpu[r]:5.1f}% {mem[r]:5.0f}M" for r in ROLES if mem[r] > 0 or cpu[r] > 0)
                print(f"{now_t - start:7.0f}s  {parts}  | box {cpu['box']:5.1f}% of {CORES} cores, {mem['box']:5.0f} MB", flush=True)
            if csv:
                csv.write(f"{now_t - start:.1f}," + ",".join(f"{cpu[r]:.2f},{mem[r]:.1f}" for r in ROLES) + f",{cpu['box']:.2f},{mem['box']:.1f}\n")
                csv.flush()
            last, last_t = now, now_t
    except KeyboardInterrupt:
        pass
    if n:
        print("mean   " + " ".join(f"{r} {sums[r][0] / n:5.1f}% {sums[r][1] / n:5.0f}M" for r in ROLES)
              + f"  | box {sums['box'][0] / n:5.1f}%, {sums['box'][1] / n:5.0f} MB")
        print("peak   " + " ".join(f"{r} {peaks[r][0]:5.1f}% {peaks[r][1]:5.0f}M" for r in ROLES)
              + f"  | box {peaks['box'][0]:5.1f}%, {peaks['box'][1]:5.0f} MB")


if __name__ == "__main__":
    main()
