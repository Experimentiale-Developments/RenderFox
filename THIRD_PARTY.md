# Third-party software

What Render Fox builds on, links, runs or downloads. Nothing here is copied into the
repository except the x264 patches, which are Render Fox's own changes to x264.

## Linked into the engine

| Component | Version | License | How |
|---|---|---|---|
| [x264](https://code.videolan.org/videolan/x264) | `stable` at `b35605ac` (0.165.3222) | GPL-2.0-or-later | Cloned at the pinned commit by `x264/prepare.sh`, patched with `x264/patches/`, linked statically. The patches are under x264's license. |
| libavcodec, libavutil ([FFmpeg](https://ffmpeg.org)) | the system's | LGPL-2.1-or-later (GPL as Debian and Ubuntu build it) | Linked dynamically; the engine decodes its own recordings with it. |
| [ab_glyph](https://crates.io/crates/ab_glyph), ab_glyph_rasterizer | 0.2.32, 0.1.10 | Apache-2.0 | The overlay's text. |
| [ed25519-dalek](https://crates.io/crates/ed25519-dalek) | 3.0.0 | BSD-3-Clause | Checking the game server's signatures. |
| [sha2](https://crates.io/crates/sha2) | 0.11.0 | MIT OR Apache-2.0 | The pairing proof's HMAC. |
| [miniz_oxide](https://crates.io/crates/miniz_oxide) | 0.9.1 | MIT OR Zlib OR Apache-2.0 | Inflating terrain from the feed. |
| [png](https://crates.io/crates/png) | 0.18.1 | MIT OR Apache-2.0 | Reading Minecraft's textures, writing snapshots. |
| [serde_json](https://crates.io/crates/serde_json) | 1.0.151 | MIT OR Apache-2.0 | Reading the config and Minecraft's models. |

Those crates pull in about 35 more (curve25519-dalek, digest, flate2, ttf-parser, serde,
libc and others), all under MIT, Apache-2.0, BSD, Zlib or Unlicense terms; `Cargo.lock`
pins every one. All of it is compatible with the GPL-3.0, Render Fox's license.

## Run beside the engine

| Component | Version | License | How |
|---|---|---|---|
| [FFmpeg](https://ffmpeg.org) (`ffmpeg`) | the system's | LGPL/GPL | Muxes video and audio, encodes each music track to AAC once, and copies the stream to the ingest. |
| [MediaMTX](https://github.com/bluenviron/mediamtx) | 1.21.0 | MIT | The local relay, and the test ingest. Downloaded by `setup.sh`, checked by SHA-256. |
| [Rust](https://www.rust-lang.org) and rustup | 1.98.1, rustup 1.29.1 | MIT OR Apache-2.0 | Building the engine. |
| Python 3 | the system's | PSF | `radio.py`, `watchdog.py`, the `renderfox` tool (standard library only). |

## Downloaded on your machine, never redistributed

| What | From | Terms |
|---|---|---|
| Minecraft's client assets | Mojang's version manifest (`piston-meta.mojang.com`), checked by Mojang's SHA-1 | [Minecraft EULA](https://www.minecraft.net/eula). Fetched by `setup.sh` for your own use; not part of Render Fox. |
| Player skins and capes | `textures.minecraft.net`, at runtime | Mojang's. Cached locally. |
| [Inter](https://rsms.me/inter/) 4.1, the overlay's font | GitHub release, checked by SHA-256 | SIL Open Font License 1.1 |

## The plugin

| Component | Version | License | How |
|---|---|---|---|
| [Paper API](https://papermc.io) | 26.2.build.123-stable | GPL-3.0 (Bukkit API) and MIT | Compiled against, provided by the server at runtime. |
| JUnit | 4.13.2 | EPL-1.0 | Tests only. |

The plugin's cryptography (Ed25519, HMAC-SHA256) is Java's own.

## Development tools only

`tools/bench/` and `engine/tools/feedtool.py` use Python's
[cryptography](https://cryptography.io) (Apache-2.0 OR BSD-3-Clause) and
[NumPy](https://numpy.org) (BSD-3-Clause), installed separately.

Render Fox is not affiliated with Mojang, Microsoft, YouTube, Twitch or PaperMC.
