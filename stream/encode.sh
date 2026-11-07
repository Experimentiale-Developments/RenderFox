#!/bin/bash
# Render Fox encoder: the engine's H.264 (in FLV) and the radio's AAC in, one H.264 +
# AAC stream out to the relay beside it (relay/). The engine encodes the video itself,
# with the patched x264 it links (x264/), because it replays an unchanged orbit's
# pictures rather than render and encode them again (engine/src/stream.rs); ffmpeg
# passes the video through and only puts the audio beside it.
# The three processes live and die together: as soon as any of them exits the others are
# stopped and systemd starts the pipeline again. (With plain pipes a dead engine would
# leave ffmpeg encoding audio alone, forever.) The uplink to YouTube is a separate
# service reading from the relay, so a blip between here and the ingest never restarts the
# renderer. A relay that is down does: ffmpeg cannot publish, and the pipeline restarts
# every 5 s until it is back.
#
# Encoder settings follow docs/youtube-ingest.md: the video's are the engine's (CBR,
# 2 s closed GOP, High profile, 2 B-frames), the audio AAC-LC stereo 128 kbps at 44.1 kHz,
# which the radio sends ready made (each track is encoded once, then copied): ffmpeg
# encodes nothing, it only puts the two together.
set -uo pipefail

HERE=$(dirname "$(readlink -f "$0")")
RF=${RF_HOME:-$(dirname "$HERE")}
CONFIG=${RF_CONFIG:-$RF/renderfox.json}
RELAY=${RF_RELAY:-rtmp://127.0.0.1:1935/fm}
MUSIC=${RF_MUSIC:-$RF/music}
RUN=$RF/run

mkdir -p "$RUN"
rm -f "$RUN/stream-epoch" "$RUN/video.pipe" "$RUN/audio.pipe"
mkfifo -m 600 "$RUN/video.pipe" "$RUN/audio.pipe"

"$RF/engine/target/release/rfengine" --config "$CONFIG" --stream --epoch-file "$RUN/stream-epoch" ${RF_ENGINE_ARGS:-} > "$RUN/video.pipe" &
engine=$!
python3 "$HERE/radio.py" --music "$MUSIC" --cache "$RF/cache/aac" --now-playing "$RUN/nowplaying.txt" \
    --epoch-file "$RUN/stream-epoch" > "$RUN/audio.pipe" &
radio=$!
# One malloc arena and freed memory handed back sooner: about 4 MB less, at the same CPU.
MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=131072 nice -n 5 ffmpeg -hide_banner -loglevel warning -nostdin -nostats \
    -f flv -thread_queue_size 64 -i "$RUN/video.pipe" \
    -f aac -thread_queue_size 256 -i "$RUN/audio.pipe" \
    -map 0:v -map 1:a -c:v copy -c:a copy \
    -f flv -flvflags no_duration_filesize "$RELAY" &
encoder=$!

wait -n -p ended "$engine" "$radio" "$encoder"
status=$?
case $ended in
    "$engine") who=engine ;;
    "$radio") who=radio ;;
    *) who=ffmpeg ;;
esac
kill "$engine" "$radio" "$encoder" 2>/dev/null
# A process blocked opening a FIFO whose other end never comes (the engine died before
# ffmpeg opened the audio, say) does not stop on SIGTERM, and waiting for it would leave
# the service up with no stream: what is left after five seconds is killed outright.
alive() { for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
for _ in $(seq 50); do
    alive "$engine" "$radio" "$encoder" || break
    sleep 0.1
done
kill -KILL "$engine" "$radio" "$encoder" 2>/dev/null
wait
echo "encode: $who exited with status $status, stopped the rest" >&2
exit 1
