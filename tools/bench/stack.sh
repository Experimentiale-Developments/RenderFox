#!/bin/bash
# The whole render box on one machine, for measuring: the fake game server
# (fakeserver.py) feeding the engine, encode.sh (engine + radio + ffmpeg) publishing to
# the relay (MediaMTX with relay/relay.yml), and two uplinks (relay/uplink.sh, primary and
# backup) sending it over RTMPS, with certificate checks, to a local stand-in for YouTube
# that does not record. Everything runs until this script is stopped (Ctrl-C or SIGTERM).
#
#   bench/stack.sh DIR [fakeserver options]
#
# DIR is a scratch directory; it gets renderfox.json (from stream/renderfox.json, pointed
# at the checkout's assets/ and fonts/, which setup.sh fetches), a key, test music if it
# has none, and the logs. Set RF_ENGINE to run another engine binary, RF_NO_UPLINK=1 to
# leave the uplinks out. Needs Python's cryptography and numpy.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DIR=$(mkdir -p "$1" && cd "$1" && pwd)
shift
cd "$DIR"
MEDIAMTX=${MEDIAMTX:-$REPO/deps/mediamtx}
[ -x "$MEDIAMTX" ] || { echo "no MediaMTX at $MEDIAMTX (setup.sh pins the release)" >&2; exit 1; }

[ -f bench.key ] || python3 "$HERE/fakeserver.py" key bench.key >/dev/null
PUB=$(python3 -c '
import sys
from base64 import b64encode
from cryptography.hazmat.primitives import serialization
k = serialization.load_der_private_key(open("bench.key", "rb").read(), password=None)
print(b64encode(k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode())')
if [ ! -f renderfox.json ]; then
    python3 - "$PUB" "$REPO" <<'EOF'
import json, re, sys
pub, repo = sys.argv[1:]
text = open(f"{repo}/stream/renderfox.json").read()
text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', lambda m: m.group(0) if m.group(0)[0] == '"' else " ", text, flags=re.S)
c = json.loads(text)
c["assets"] = f"{repo}/assets/client"
c["observerKey"] = pub
c["overlay"]["fonts"] = f"{repo}/fonts"
c["overlay"]["nowPlaying"] = "rf/run/nowplaying.txt"
json.dump(c, open("renderfox.json", "w"), indent=1)
EOF
fi
if [ ! -d music ] || [ -z "$(ls music)" ]; then
    mkdir -p music
    for i in 1 2 3; do
        ffmpeg -v error -y -f lavfi -i "sine=f=$((180 + i * 60)):d=150,volume=0.3[a];anoisesrc=d=150:c=pink:a=0.08[b];[a][b]amix=inputs=2,aformat=channel_layouts=stereo" \
            -ar 44100 -sample_fmt s16 -metadata title="Test Piece $i" "music/0$i-test.flac"
    done
fi
# The engine binary encode.sh runs is $RF/engine/target/release/rfengine.
mkdir -p rf/engine/target/release rf/run logs
ln -sf "${RF_ENGINE:-$REPO/engine/target/release/rfengine}" rf/engine/target/release/rfengine
rm -rf rf/run/replay

# A YouTube stand-in: RTMPS with a self-signed certificate for localhost, nothing recorded.
if [ ! -f dummy.crt ]; then
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3650 \
        -keyout dummy.key -out dummy.crt -subj /CN=localhost -addext subjectAltName=DNS:localhost 2>/dev/null
fi
# Primary and backup land on two paths, as they land on two hosts at YouTube.
sed -e 's/^    record: yes/    record: no/' -e 's/^logLevel: info/logLevel: warn/' \
    -e 's|path: live2/rf-dummy-key$|path: ~^live2/rf-dummy-key|' -e 's|^  live2/rf-dummy-key:|  "~^live2/rf-dummy-key":|' \
    "$REPO/tools/test-ingest/mediamtx.yml" > ingest.yml
cat > uplink.env <<EOF
PRIMARY_INGEST_URL=rtmps://localhost:1936/live2/rf-dummy-key
BACKUP_INGEST_URL=rtmps://localhost:1936/live2/rf-dummy-key-backup
CA_FILE=$DIR/dummy.crt
EOF

echo $$ > stack.pid
pids=()
cleanup() {
    trap - INT TERM EXIT
    kill "${pids[@]}" 2>/dev/null
    pkill -P $$ 2>/dev/null
    # An ffmpeg blocked opening a FIFO whose writer is gone does not stop on SIGTERM:
    # whatever of this process group is left after a few seconds is killed outright.
    local group
    group=$(ps -o pgid= $$ | tr -d ' ')
    ( sleep 5; kill -KILL -- "-$group" 2>/dev/null ) &
    wait 2>/dev/null
}
trap cleanup INT TERM EXIT

"$MEDIAMTX" "$REPO/relay/relay.yml" > logs/relay.log 2>&1 &
pids+=($!)
"$MEDIAMTX" ingest.yml > logs/ingest.log 2>&1 &
pids+=($!)
python3 "$HERE/fakeserver.py" run --key bench.key "$@" > logs/fakeserver.log 2>&1 &
pids+=($!)
sleep 1
RF_HOME=$DIR/rf RF_CONFIG=$DIR/renderfox.json RF_MUSIC=$DIR/music RF_RELAY=rtmp://127.0.0.1:1935/fm \
    "$REPO/stream/encode.sh" > logs/encode.log 2>&1 &
pids+=($!)
if [ -z "${RF_NO_UPLINK:-}" ]; then
    for role in primary backup; do
        (while true; do RF_RELAY_HOME=$DIR "$REPO/relay/uplink.sh" $role; sleep 1; done) > "logs/uplink-$role.log" 2>&1 &
        pids+=($!)
    done
fi
wait
