#!/usr/bin/env bash
# Builds the patched x264 as a static library into build/, which the engine links
# (engine/build.rs). 8-bit 4:2:0 only, no command line tool; the assembly needs nasm.
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/prepare.sh"
command -v nasm >/dev/null || { echo "x264's assembly needs nasm (apt install nasm)" >&2; exit 1; }

cd "$DIR/third_party/x264"
./configure --prefix="$DIR/build" --enable-static --enable-pic --disable-cli --disable-opencl \
    --bit-depth=8 --chroma-format=420 \
    --disable-avs --disable-swscale --disable-lavf --disable-ffms --disable-gpac --disable-lsmash >/dev/null
make -j"$(nproc)" >/dev/null
# build/ may be a link into resources/: empty it rather than replace it.
mkdir -p "$DIR/build"
find "$DIR/build/" -mindepth 1 -delete
make install-lib-static >/dev/null
echo "==> $(grep -m1 X264_POINTVER config.h 2>/dev/null || grep -m1 X264_POINTVER x264_config.h) built into build/"
