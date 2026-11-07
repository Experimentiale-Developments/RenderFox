#!/bin/sh
# Prepares an Ubuntu or Debian machine to run Render Fox from this checkout. Safe to run
# again. Run as the user the stream is to run as (not root); sudo is asked for packages
# and the services only.
#
#   ./setup.sh                 everything for streaming
#   ./setup.sh --test-ingest   also the stand-in for YouTube, for testing without going live
set -eu

RF=$(dirname "$(readlink -f "$0")")
# The Minecraft version whose client assets the engine draws with: the game server's.
MC_VERSION=${MC_VERSION:-26.2}
INTER_URL=https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
INTER_SHA256=9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e
MEDIAMTX_URL=https://github.com/bluenviron/mediamtx/releases/download/v1.21.0/mediamtx_v1.21.0_linux_amd64.tar.gz
MEDIAMTX_SHA256=e02e34c3337a35f20ac9e5aa31524566108964e6e37dbc46cf8292169f6c792b
# rustup only; the compiler version itself is pinned by engine/rust-toolchain.toml.
RUSTUP_URL=https://static.rust-lang.org/rustup/archive/1.29.1/x86_64-unknown-linux-gnu/rustup-init
RUSTUP_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71

[ "$(id -u)" -ne 0 ] || { echo "run setup.sh as the user the stream runs as, not root" >&2; exit 1; }
[ "$(uname -m)" = x86_64 ] || { echo "Render Fox runs on x86-64 only (its SIMD is AVX2 or SSE2)" >&2; exit 1; }

# build-essential and nasm build x264 (nasm for its assembly, without which it is several
# times slower); libavcodec-dev and pkg-config let the engine decode its own recordings.
sudo apt-get install -y build-essential nasm git ffmpeg libavcodec-dev pkg-config python3 curl unzip
mkdir -p "$RF/assets" "$RF/fonts" "$RF/deps" "$RF/run" "$RF/music" "$RF/cache"
scratch=$RF/scratch_setup
mkdir -p "$scratch"

# Minecraft's client assets (textures, models, blockstates, biome colours), straight from
# Mojang and checked against the version manifest. Fetched here, never redistributed.
if [ ! -d "$RF/assets/client/assets/minecraft" ]; then
    cd "$RF/assets"
    curl -fsSL https://piston-meta.mojang.com/mc/game/version_manifest_v2.json -o manifest.json
    set -- $(python3 -c 'import json,sys; v=[v for v in json.load(open("manifest.json"))["versions"] if v["id"]==sys.argv[1]][0]; print(v["url"], v["sha1"])' "$MC_VERSION")
    curl -fsSL "$1" -o "$MC_VERSION.json"
    echo "$2  $MC_VERSION.json" | sha1sum -c -
    set -- $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["downloads"]["client"]; print(d["url"], d["sha1"])' "$MC_VERSION.json")
    curl -fsSL "$1" -o "$scratch/client.jar"
    echo "$2  $scratch/client.jar" | sha1sum -c -
    unzip -q -o "$scratch/client.jar" 'assets/*' 'data/*' -d client
fi

if [ ! -f "$RF/fonts/Inter-SemiBold.ttf" ]; then
    curl -fsSL "$INTER_URL" -o "$scratch/inter.zip"
    echo "$INTER_SHA256  $scratch/inter.zip" | sha256sum -c -
    unzip -q -o -j "$scratch/inter.zip" extras/ttf/Inter-SemiBold.ttf extras/ttf/Inter-Medium.ttf LICENSE.txt -d "$RF/fonts"
fi

if [ ! -x "$RF/deps/mediamtx" ]; then
    curl -fsSL "$MEDIAMTX_URL" -o "$scratch/mediamtx.tar.gz"
    echo "$MEDIAMTX_SHA256  $scratch/mediamtx.tar.gz" | sha256sum -c -
    tar -xzf "$scratch/mediamtx.tar.gz" -C "$RF/deps" mediamtx
fi
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    curl -fsSL "$RUSTUP_URL" -o "$scratch/rustup-init"
    echo "$RUSTUP_SHA256  $scratch/rustup-init" | sha256sum -c -
    chmod +x "$scratch/rustup-init"
    "$scratch/rustup-init" -y --profile minimal --default-toolchain none --no-modify-path
fi
rm -rf "$scratch"

"$RF/x264/build.sh"
(cd "$RF/engine" && "$HOME/.cargo/bin/cargo" build --release --locked)
[ -f "$RF/renderfox.json" ] || cp "$RF/stream/renderfox.json" "$RF/renderfox.json"
sudo sh "$RF/install.sh" "$@"
cat <<DONE

Render Fox is installed. Next:
  1. renderfox youtube          give it your stream key (or: renderfox twitch, renderfox rtmp URL)
  2. put music in $RF/music     any FLAC, Ogg, Opus, WAV, MP3 or M4A; silence until you do
  3. edit $RF/renderfox.json    the overlay text and the orbit's centre over your spawn
  4. renderfox start
  5. renderfox pair             then run the /renderfox pair command it prints on your server
DONE
