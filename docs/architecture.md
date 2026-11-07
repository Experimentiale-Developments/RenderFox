# Architecture

How Render Fox turns a game server into a 1080p24 stream on four CPU cores. Timings are
for a four-core Haswell-class CPU with no GPU, the minimum Render Fox is designed for.

## The pieces

| Process | What |
|---|---|
| The RenderFox plugin, on the game server | Mirrors blocks and players around one point to the engine ([protocol.md](protocol.md)). Read-only. |
| `rfengine` (`engine/`) | Keeps a copy of that world, lights and meshes it, renders the orbit, draws the overlay, and encodes H.264 itself with a patched x264, writing FLV to a pipe. Or, for an unchanged part of the orbit, sends what it recorded the last time round. |
| `radio.py` (`stream/`) | Plays the music folder as AAC, paced to the engine's frame clock, and writes the title of what is audible to the now playing file. |
| ffmpeg in `encode.sh` | Puts the engine's video and the radio's audio together, copying both, and publishes to the relay. |
| MediaMTX (`relay/relay.yml`) | A relay on 127.0.0.1 only. It decouples the encoder from the ingest: a dropped connection to YouTube restarts an uplink, never the renderer and its recording. |
| `uplink.sh` (one or two) | ffmpeg copying the relay to the ingest over RTMPS, unchanged. |
| `watchdog.py` | Watches the relay's API and restarts an uplink systemd has given up on. |

The engine, the radio and ffmpeg live and die together (`encode.sh`): when any of them
exits, the others are stopped and systemd starts all three again.

## The renderer

A software rasterizer purpose-built for Minecraft geometry, in Rust (`engine/src/render.rs`
and around it). Every quad in view becomes a primitive with its plane equations over pixel
coordinates, rasterised into a visibility buffer (the nearest primitive per pixel); each
pixel is then shaded once. The per-pixel passes run on eight-lane SIMD types (`wide.rs`):
AVX2 and FMA where the CPU has them, two SSE2 halves otherwise. The code is safe Rust
except where it talks to the hardware or the OS: the SIMD types, the job pool handing a
borrowed closure to its threads (`jobs.rs`), and three POSIX calls (`sys.rs`). Parallel
passes write disjoint rows through `parts.rs`.

What it draws:

- The real client assets: blockstates, multipart models, element rotation, uvlock,
  animated textures, biome tints from the colormaps.
- Vanilla sky and block light, smooth lighting and ambient occlusion.
- Sun shadows from two cascaded shadow maps, updated incrementally when blocks change;
  cloud shadows; players cast silhouette shadows.
- A sky gradient and blocky clouds, fog, water with a fresnel reflection and a sun glint,
  bloom, ACES tonemapping.
- Players and mannequins with their real skins (outer layers included), capes, armour,
  walking and swinging, elytra folded, spread while gliding and swept back in a dive, a
  gliding player lying along its flight and rolling into turns. Skins come from
  `textures.minecraft.net` only and are cached. Held items are not drawn; other entity
  types are left out rather than drawn wrong.
- Name tags as the game draws them (its pixel font, white on a quarter-opaque box), enlarged
  so a font pixel is at least two screen pixels. A tag dims once its player has been mostly
  hidden for 1.5 s and clears after 0.5 s in view; hidden is measured by walking the
  world's blocks from 20 points over the body towards the camera, counting only opaque
  full blocks and leaves.

### Frames

Every frame is rendered natively at the output size with temporal antialiasing; nothing
is upscaled. The static world is rendered in full every `staticEvery`th frame (7) and
reprojected in between; players are drawn every frame. On such a CPU a full render takes
about two frame slots, and the reprojected frames catch up: 33 to 37 ms of rendering plus
2.4 ms of overlay and colour conversion a frame against a 41.7 ms budget, with the engine
rendering up to `aheadFrames` (4) ahead so only the average has to fit.

- **Reprojection** finds the exact sub-pixel source point and samples it with a clamped
  Catmull-Rom filter. Taking the nearest pixel made regions hold still and then jump a
  pixel at different moments, which looked like heat haze.
- **Antialiasing** jitters where the world is sampled but always rebuilds colour at pixel
  centres, and reads the history back with a 12-tap Catmull-Rom. Pixels away from depth
  edges and players take each new full render in gently (6 % a frame, unclamped), so a
  refresh moves them a few percent rather than all at once and shadows do not shimmer;
  edges and players keep a fast clamped blend so nothing trails. The settled picture is
  kept on the full render's own grid and resampled once per refresh, not every frame.
  Mip levels are rounded as GPUs pick them, which calms sparkle in the distance.
- **Thin things near the camera** (bamboo above all) slide against the ground behind
  them as the camera moves. Where a nearer surface is within 8 pixels along a row, the
  warp also searches holding that nearer depth, as in a depth test, and where it cannot
  settle, holding the farthest depth nearby; ground a stalk has just uncovered is filled
  from the nearest pixel that shows it.
- **Bloom** on a full render is taken from the settled picture the next frames show, not
  the fresh render, so a sun glint on water does not flash once per refresh.
- A block that changes makes the tiles where it or its shadow shows take the new render
  at once, like the tiles around players.

### The orbit

Always daytime, the sun fixed (`look.sunElevation`, `sunAzimuth`). The camera's angle is
derived from the wall clock, so a restart resumes the orbit where it was. The orbit is
planned, not just followed: where something stands on the circle above eye height, the
camera pulls in towards the centre rather than climbing over it, easing in and out over
about a minute (at most 0.35 m of radius per metre travelled), and climbs only where even
45 % of the radius is blocked. The plan is redone every 20 s and blended in over 5 s, so
a new build on the path moves the camera instead of cutting it. The engine logs the plan
(`orbit: pulled in over ...`).

Everything the renderer animates runs on orbit time, so frame N of one lap is frame N of
the next: the clouds drift in a circle and the water's waves complete whole cycles per
lap. That is what lets replay work. A new animation keyed to wall time would jump
wherever replay and rendering meet.

## The feed

From a placed block to the stream:

1. The plugin notices the change through its event and sends the sections within a block
   of it at the end of the same server tick. Changes without an event (`/setblock`,
   `/fill`, WorldEdit, other plugins) are found by the sweep instead: about 20 s near the
   centre, a few minutes at the edge.
2. The engine's world thread wakes on arrival and, at normal priority, relights only the
   31-block square light can change, remeshes the sections the change touches and redraws
   its shadow: 26 to 43 ms on average. Light further out settles over the next moments at
   low priority. Changes that keep coming (a redstone clock in view) are taken together
   at most ten times a second.
3. The new meshes, shadow and a note of the change reach the renderer together. The next
   full render is brought forward (never two in a row, and never more than one in
   `staticEvery` on average), and the tiles where the block and its shadow show take it at
   once.
4. Frames are rendered up to `aheadFrames` slots before they go out.

Altogether 310 to 385 ms on average from arrival to the frame going out to the encoder,
the longest 520. The `stream:` line in the journal reports it every minute
(`N edits out X ms after arriving`). The encoder then holds each picture for 18 frames,
0.75 s (10 of rate-control lookahead, 3 of B-frame decisions, 1 for that thread, 4 of
slack), and YouTube adds its own delay, set per stream.

**Players** are sent every tick (20 a second) in a lane that goes out ahead of terrain,
and placed on the server's own tick timeline rather than by when they arrived, so a burst
of chunks does not bunch their steps together. They are posed (`aheadFrames` + 1) / fps
+ 75 ms behind the feed, about 0.28 s, which keeps a position on either side of every
frame; if the feed stalls, a player carries on at its last speed for up to a tenth of a
second rather than freezing.

## Replay

Most of the time nothing near the centre changes, and the orbit shows the same frames
every lap. Rendering and encoding them again takes all four cores. So the engine records
the orbit as it streams it and, once a part of it has been shown unchanged, sends the
recorded pictures instead: nothing rendered, nothing encoded, about 36 % of one core
instead of 300 %. Anything that changes the picture brings rendering back straight away.

- **The timeline.** The stream's epoch is put on a 2 s boundary and a lap is a whole
  number of 2 s groups of pictures (360 s: 180 groups of 48 frames), so each group starts
  at the same point of the orbit every time round and is recorded, and replayed, as a
  unit.
- **What counts as a change: only what a recording could show** (`engine/src/sight.rs`).
  Each recorded group keeps what its frames could see: the camera of every full frame
  rendered while it was recorded, and the farthest surface in each 32x32 pixel tile. The
  world worker reports every change it publishes (each remeshed section as a box, each
  redrawn shadow as the stretch the sun's rays reach from the changed blocks to the
  ground), and a change throws away only the groups whose frames it would show in. A block
  placed behind the camera, or someone digging underground, leaves the rest of the orbit
  replayable. A group is not replayed while a player (or a name tag, which shows through
  blocks) could be in its frames, or where the camera's path has changed.
- **Natural changes wait** (`replay.natureSeconds`, 1800). Sheep eat grass, crops grow,
  leaves decay, and each such change near the centre shows from nearly every point of the
  orbit: a few a minute kept the whole orbit rendering. Changes more than 10 blocks from
  any player are held back and shown together every 30 minutes; what players do shows at
  once, and so does anything held next to it.
- **Recording** happens as a side effect of streaming: every picture the encoder makes is
  kept, and a whole group goes into memory (about 2.5 MB at 10 Mbit/s, 450 MB for a lap)
  once the renderer has run `warmSeconds` from a cold start and `settleSeconds` have passed
  since a player or change was last in view.
- **Stopping.** When a player comes into view mid-replay, or a change the group shows
  arrives, the render thread picks the first point where the pictures sent so far form a
  decodable prefix (at most 3 frames on) and renders from the next frame, an IDR opening a
  fresh encoder. A lamp placed during replay was on screen 0.21 s after the command.
- **Handing over warm.** Where the next group will have to be rendered, the renderer
  starts 1.5 s before the end of the replayed one, in the background, so it takes over with
  its antialiasing settled.
- **The now playing card** is the one thing that never repeats: the music does not follow
  the orbit. The patched x264 (`x264/`) encodes a rectangle around the card (macroblocks
  0,58 to 47,68 at 1080p, 752x152 pixels in the bottom left) in slices of its own that
  never predict across its edge. While replaying, a second encoder recodes only that
  rectangle: the recorded scene under the card, from the region store, with the card as it
  is now drawn over it, following the recorded picture's frame type and quantiser. Its
  slices replace the recorded ones inside the rectangle; the decoder cannot tell the
  difference, which `cargo test --release -- --ignored spliced` checks to the bit. The
  region costs about 3 % more bits on every picture, spent along its edges.
- **Players over a recording.** The same machinery lets the engine recode horizontal bands
  of a recorded picture with players drawn in, rather than rendering the whole frame
  (`x264/README.md`, patches 0003 to 0007).
- **The region store** (`replay.store`, 1.48 GB at 1080p) holds the rendered scene under
  the card for every frame of the orbit, before the card is drawn.
- **Persistence.** Recordings are written to disk beside the region store, with a hash of
  every world section they show. After a restart the engine compares the world it receives
  with those hashes, spoils only the recordings a change made while it was down could
  show, and resumes replaying, as long as the engine binary and every setting that alters
  the picture are the same.

The `stream:` minute line shows which it is doing: frames rendered and why (a player in
view, not recorded, the camera moved, partway into a group), frames drawn in the
background, frames replayed and how many of the groups are recorded. A `replay:` line
names where a change was each time one throws recordings away. `replay.enabled: false`
turns it all off; every frame is then rendered and encoded.

## Memory

The engine's resident memory is about 700 to 800 MB while it renders, flat over hours.
The ffmpeg processes take about 55 to 60 MB each. `systemctl status renderfox-encode`
reports the service's cgroup memory, which includes the page cache from reading and
writing the recordings (about 1.7 GB); the kernel reclaims that whenever it needs to, so
look at `free -h` (the available column) for what is really free.
