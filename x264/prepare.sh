#!/usr/bin/env bash
# Clones the pinned x264 into third_party/x264 and applies patches/. Safe to rerun: it
# resets the checkout first, so a half-applied or hand-edited tree is repaired rather
# than built on top of.
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/third_party/x264"
# shellcheck disable=SC1091
. "$DIR/x264.pin"

if [ ! -d "$SRC/.git" ]; then
  echo "==> cloning x264 $X264_BRANCH"
  rm -rf "$SRC"
  mkdir -p "$(dirname "$SRC")"
  git -c advice.detachedHead=false clone --quiet --branch "$X264_BRANCH" https://code.videolan.org/videolan/x264.git "$SRC"
fi

cd "$SRC"
git fetch --quiet origin "$X264_COMMIT" 2>/dev/null || true
git -c advice.detachedHead=false checkout --quiet --force "$X264_COMMIT"

# A branch moves; the commit cannot. This is the check that makes the pin real.
have="$(git rev-parse HEAD)"
if [ "$have" != "$X264_COMMIT" ]; then
  echo "x264 checkout is at $have, expected $X264_COMMIT. Refusing to build." >&2
  exit 1
fi

git checkout --quiet -- .
git clean --quiet -fdx

for p in "$DIR"/patches/*.patch; do
  echo "==> applying $(basename "$p")"
  git apply --check "$p"
  git apply "$p"
done
echo "==> x264 ready at third_party/x264 with $(ls "$DIR"/patches/*.patch | wc -l) patches"
