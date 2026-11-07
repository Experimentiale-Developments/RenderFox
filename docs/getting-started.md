# Getting started

From an empty machine to a live stream. Two machines take part: the **render box**,
which runs the engine and sends the stream, and the **game server**, which runs Paper
with the RenderFox plugin. They can be the same machine only if it has cores to spare:
rendering live takes about four.

## 1. The render box

Any x86-64 machine with four cores that have AVX2 and FMA (Intel Haswell or newer, AMD
Zen or newer) running Ubuntu 26.04, which Render Fox is developed and run on (another
recent Debian-based system with a current ffmpeg should do, but is untested). A VM works, as long as
its CPU type passes AVX2 through: `host` on Proxmox or KVM. Check with

```sh
grep -o -w -m1 avx2 /proc/cpuinfo    # prints avx2, or nothing
```

Without AVX2 the engine still builds and runs, on SSE2, but at a fraction of the speed:
fine to try things, far too slow to stream. A binary built on one CPU refuses to start on
a CPU lacking what it was built for, so build on the machine that runs it.

Give the box a fixed address (a DHCP reservation on the router, or a static address),
because the game server dials it by that address.

## 2. Install

As the user the stream should run as (not root; it needs sudo once, for packages and
services):

```sh
git clone https://github.com/Experimentiale-Developments/RenderFox.git ~/renderfox
cd ~/renderfox
./setup.sh
```

`setup.sh` installs the packages (`build-essential`, `nasm`, `ffmpeg`, `libavcodec-dev`,
`pkg-config`, `python3`, `git`, `curl`, `unzip`), then downloads and checks:

- Minecraft's client assets for the version in `MC_VERSION` (26.2 unless you set it),
  straight from Mojang, checked against Mojang's own hashes. Match it to the game
  server's version: `MC_VERSION=26.2 ./setup.sh`.
- The Inter font for the overlay, MediaMTX for the relay and rustup, each pinned by
  SHA-256.
- x264 at a pinned commit, which it patches and builds, then the engine itself (a few
  minutes).

It copies `stream/renderfox.json` to `renderfox.json`, installs the services, links
`renderfox` into `/usr/local/bin`, and enables the services on boot. It is safe to run
again; it skips what is already there.

## 3. Where the stream goes

For YouTube: in YouTube Studio, **Go live**, **Stream**. Use the default persistent stream
key (or create one under Stream settings), and turn **Auto-start** on, so the broadcast
starts by itself whenever the stream arrives and resumes after a drop. Then on the render
box:

```sh
renderfox youtube
```

and paste the key when asked. It is not echoed, not logged, and kept in
`relay/uplink.env`, readable by the stream's user only. YouTube gets the stream twice,
on its primary and backup ingest, so one dropped connection never ends the broadcast.
For Twitch or another service, see [outputs.md](outputs.md).

## 4. Music

Put audio files in `~/renderfox/music`: FLAC, Ogg, Opus, WAV, MP3 or M4A. They play as a
shuffle bag (every track once before any repeats), and the card shows each file's `title`
tag, or its file name without leading numbers. The folder is rescanned for every round,
so new music needs no restart. Each track is encoded to AAC once, the first time it comes
up, and cached.

Use music you own or have licensed for streaming: YouTube matches streams against its
catalogue and can take a stream down. Mastering everything to the same loudness (say,
-16 LUFS integrated) keeps the stream from jumping in level between tracks. With no music
the stream carries silence.

## 5. The look

Edit `~/renderfox/renderfox.json` ([configuration.md](configuration.md) has every key).
The ones to set now:

- `overlay`: `station` (top left of the banner), `invite` (beside it), `joinLabel` and
  `address` (the card in the bottom right, the address being what viewers should type to
  join).
- `orbit.center`: the block the camera circles, `[x, y, z]`, usually your spawn at about
  head height. `orbit.radius` and `orbit.height` set how far out and how high it flies.
- In the plugin's `config.yml` (step 7), `center` should be the same place.

## 6. Start it

```sh
renderfox start
renderfox status
renderfox logs        # the engine's journal: a "stream:" line every minute
```

YouTube Studio should show the stream arriving within half a minute: an empty sky, since
no server feeds the engine yet.

## 7. The game server

Build the plugin, or take `RenderFox-<version>.jar` from the releases:

```sh
cd ~/renderfox/plugin && mvn package    # needs Java 25 and Maven
```

Put the jar in the server's `plugins/` folder and restart the server. On first start the
plugin makes this server's key pair in `plugins/RenderFox/` and waits to be paired.

## 8. Pair

On the render box:

```sh
renderfox pair
```

It prints a one-time code and the exact command for the server, with this machine's
address:

```
/renderfox pair 192.168.1.50:25590 7KQ2M-X9PRT
```

Run it in the server's console or as an operator (permission `renderfox.admin`). The
server answers "Paired with the Render Fox engine", connects, and sends the area around
spawn; within a few seconds the stream shows your world. The engine remembers the server
across restarts, and refuses any other server from now on. [pairing.md](pairing.md)
explains what happens underneath and how to move the stream to another server.

The game server must reach TCP port 25590 on the render box. Nothing has to be opened on
the game server: it dials out.

## 9. Check it

- `renderfox status`: every service active, the server paired, the relay receiving and
  one uplink reading per ingest (two for YouTube).
- `/renderfox` on the server: connected, how much it has sent.
- The `stream:` line in `renderfox logs`: frames rendered or replayed, and how many were
  repeated. A few repeats right after a start are normal; many all the time mean the
  CPU cannot keep up (try 720p, or fewer other programs on the box).
- After the engine has seen a whole orbit unchanged (six minutes by default) it starts
  replaying, and its CPU use drops to a fraction of a core while nobody is in view.

## Where to go from here

- [operations.md](operations.md): what runs, logs, what failures look like, upgrading.
- [outputs.md](outputs.md): Twitch and other ingests.
- [architecture.md](architecture.md): how the renderer and replay work.
