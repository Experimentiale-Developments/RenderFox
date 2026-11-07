# x264 for the Render Fox engine

The engine encodes the stream itself, with this x264: upstream's `stable` branch at
the commit Ubuntu ships as libx264 0.165.3222 (`x264.pin`), plus one patch
(`patches/0001-region.patch`) and six that build on it. `build.sh` clones it into `third_party/` (ignored by git),
checks the commit, applies the patches and builds a static library into `build/`, which
`engine/build.rs` links. It needs nasm for the assembly; without it x264 is several
times slower, which a four-core render box cannot afford. Never edit `third_party/x264` by hand:
`prepare.sh` resets it on every build. A change to x264 is a change to the patch.

## Why a patch

The stream replays an unchanged orbit's encoded pictures instead of rendering and
encoding it again (`engine/src/replay.rs`). One thing on screen never repeats: the
now playing card, which follows the music. So a replayed picture keeps the recorded
slices everywhere but around the card, whose slices come from a second encode of the
same frame with the card as it is now. For the decoder to rebuild both parts exactly as
their own encoders did, the two parts must never predict from each other. Stock x264
cannot promise that; the patch adds a **region**, `x264_param_t.region`, a rectangle of
macroblocks that:

- is coded in slices of its own (a slice ends where a row enters or leaves it);
- is never predicted from outside, nor anything outside from inside it. The motion
  search is bounded per macroblock so that no read, including the 6-tap filter's reach
  of 2 pixels before and 3 after, crosses the edge. Macroblocks outside keep off it on
  whichever side leaves them most room. Where the fractional bounds would shut out the
  zero vector, next to the edge, that axis is searched in whole pixels, which may come
  right up to the edge, so static content there can still be skipped;
- has skip and direct vectors, which are derived rather than searched, checked against
  the same bounds (fraction-aware) and not used where they would cross;
- has no deblocking across its slice edges (`disable_deblocking_filter_idc` 2, as x264
  already does for sliced threads).

Six more patches build on it:

- `0002-region-only.patch`: `region.b_only`, for the encoder that only recodes the
  region. Macroblocks outside it are neither analysed nor encoded (their slices come out
  empty, and are thrown away anyway), rows its predictions cannot reach are not filtered,
  and only its rows of the input are copied or measured for adaptive quantisation. A frame
  whose quantiser is forced keeps it on every row (no row-level VBV), so its slices never
  depend on what was skipped. With `rc-lookahead=0` the now playing card costs 4.5 ms a
  frame to recode at 1080p instead of 15.8.
- `0003-bands.patch`: `region.i_bands` cuts the frame into horizontal bands, each coded
  apart from the others as the region is (bounded motion, slices of its own, no
  deblocking across), the region's rows being one band. Zones are numbered 0 for the
  region and 1 + k for band k, and `x264_picture_t.i_region_zones` says which zones a
  region-only encoder encodes (changed only on an IDR). The stream draws players over a
  recorded group by recoding just their bands. Measured on freshly rendered 1080p at 10
  Mbit/s (luma PSNR): no region 39.55 dB, the card's region 39.36, the card and cuts at
  rows 19/32/45/58 39.00. Cuts through the sky cost nothing; through busy ground about
  0.1 dB each.
- `0004-recorded-quantisers.patch`: with `region.b_only`, a macroblock's
  `quant_offsets` entry replaces adaptive quantisation's offset instead of adding to it
  (`X264_QUANT_OFFSET_AQ` leaves it to adaptive quantisation). The engine recodes bands
  from the decoded recording at the quantiser each macroblock was recorded at (libavcodec
  exports them), with small deadzones so the levels already quantised to survive: two
  busy bands at 10 Mbit/s came back at 38.23 dB for the recording's 38.63 at the same
  bits, against 34.39 dB with the encoder's own adaptive quantisation.
- `0005-window.patch`: `region.i_mb_offset`, `b_top_closed` and `b_bottom_closed`,
  for a region-only encoder given just the rows it encodes (a window) rather than whole
  frames it would mostly hold and never encode: slices are numbered as in the full frame,
  and the window's edges, where they are not the frame's, are kept to as band cuts are.
  At 1080p the encoder recoding the card holds 10 macroblock rows instead of 68.
- `0006-copy.patch`: `x264_picture_t.copy_index`/`copy_pool`, macroblocks to write as
  they were coded before (`x264_copy_mb_t`: type, intra modes, quantiser, levels and the
  vectors the decoder found), predicted and reconstructed here as a decoder does. A skip
  or direct macroblock whose vectors would come out otherwise is written with them
  explicitly. Frames with copies are not weighted. The vertical vector limits are now
  worked out at a row's first analysed macroblock rather than its first one. The engine's
  test reads a recording's slices (h264syn.rs), recodes two bands with every macroblock
  given back, and gets the recording's slices byte for byte.
- `0007-region-lean.patch`: region-only encoders keep no half-resolution frames: every
  frame's type and quantiser is given, so the lookahead's frame costs and weight analysis
  are never used. Recoding two 1080p bands went from 17.7 to about 11 ms a frame, and the
  encoder's memory from 78 to 57 MB at full frame size.

The first patch also lets the caller choose each IDR's `idr_pic_id`
(`x264_picture_t.i_idr_pic_id`): slices of one picture must agree on it, and the
engine uses the frame's place in the orbit, which is the same in both encodes.

A side of the region not on the frame's edge needs it at least 3 macroblocks deep.

## What it costs, what proves it

Measured on 8 s of the stream at 1080p, CRF 21, region 0,58 to 47,68 (the card):

| | kbit/s |
|---|---|
| no region | 7963 |
| region: slices and deblocking only | 8068 (+1.3 %) |
| region, first version of the bounds | 9676 (+21.5 %) |
| bounds by the side with most room, whole-pixel edges | 8227 (+3.3 %) |

At the stream's constant 10 Mbit/s that is about 3 % of the bits, spent along the
card's edges. Encoding speed is unchanged.

`cargo test --release -- --ignored spliced` in `engine/` encodes a clip whole, then
recodes the region with other content following the first encode's frame types and
quantisers, splices the two, decodes all three with ffmpeg and requires the spliced
decode to match the first encode outside the region and the second inside it, to the
bit. It covers regions on the left edge, in the middle and in a corner, with B-frames,
B-references, IDRs and two frame threads, and real footage at 1080p
(`RF_SPLICE_CLIP=clip.yuv`, raw yuv420p). Every leak found while writing the patch (the
zero vector tried unclipped in the search and in B-frame bi-prediction) failed it first.
