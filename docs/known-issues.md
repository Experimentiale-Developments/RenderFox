# Known issues

Problems that are known and not fixed yet. Close an entry by deleting it once the fix is
in.

## Player skins are cached forever

Every distinct skin a player shows up with stays in memory until the engine restarts:

- `Entities.skins` (`engine/src/entities.rs`) keeps one entry per skin key, and each
  downloaded skin registers a texture with its mips in `Assets`, about 22 KB each.
- The downloader's `requested` set only grows, so a skin that failed to download is never
  retried.
- The texture table has 65536 slots (`MAX_TEXTURES`, `engine/src/assets.rs`), shared with
  the block textures, and is never freed.

At a few hundred players this is a few MB and harmless. Over months of uptime on a busy
server, or with players changing skins often, memory creeps up; once the table fills, the
engine stops with "texture table full" and systemd restarts it (about 10 s of lost
stream).

Fix: bound the skin cache (least recently seen skins evicted past a fixed count, say 512),
give evicted textures' slots back to `Assets` for reuse, and let `requested` forget keys
so a failed download can be retried later. Verify with a test server cycling through many
skins and the engine's memory sampled for an hour.

## Only players and mannequins

Mobs, items, item frames, minecarts and other entities are not drawn, and held items are
not shown.

## One fixed daytime, no weather

The sun never moves and it never rains. A day cycle would have to run on orbit time (see
[architecture.md](architecture.md), "The orbit") to keep replay working.
