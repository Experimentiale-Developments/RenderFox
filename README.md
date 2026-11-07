# Render Fox

**Render Fox turns your Minecraft server into a live stream that runs 24/7 on YouTube or
Twitch, and it does not need a graphics card.**

In plain words:

1. You add a small plugin to your Paper server.
2. You run Render Fox on any Linux computer with a decent four-core processor (the
   "render box" in these docs).
3. The plugin tells Render Fox what is happening around your spawn: the blocks, and the
   players walking around.
4. Render Fox draws it as a slow camera flying in a circle around your spawn, with
   shader-like lighting, shadows, water and clouds. It adds your music, a small card with
   the song that is playing, and a card with your server's address so viewers can join.
5. It sends that to YouTube (or Twitch, or any other streaming site) in 1080p, day and
   night, by itself.

Players who join your server show up on the stream, with their own skins and name tags.
Blocks they place or break are on their way to YouTube about a second later (YouTube adds a
few seconds of its own before viewers see them).

## Features

- **No graphics card needed.** Everything is drawn by the processor, with a renderer
  written from scratch for Minecraft's blocks.
- **Looks like a shader pack.** Smooth lighting, sun and cloud shadows, reflective water,
  fog, bloom and antialiasing, at 1920x1080 and 24 frames a second, using the game's own
  textures and models.
- **Real players.** Skins, capes, armour, elytra, walking and swinging, and name tags.
- **Cheap when nothing happens.** When nothing a viewer could see has changed, Render Fox
  replays what it already recorded instead of drawing it again, and uses a fraction of
  one core. As soon as a player walks into view or a block changes, it draws live again.
- **Easy to connect.** Render Fox shows a one-time code; you type
  `/renderfox pair <address> <code>` on your server and they are linked. It streams only
  that server and refuses any other.
- **Easy to point at YouTube.** `renderfox youtube` asks for your stream key and does the
  rest; Twitch and any RTMP site work the same way.
- **Stays on air.** If the game server restarts, the stream keeps showing the last
  picture and catches up when the server is back. If the connection to YouTube drops, it
  reconnects by itself, and YouTube's backup ingest carries the stream meanwhile.
- **Safe for your world.** The plugin only reads. It never changes anything, and never
  shows spectators, vanished staff or invisible players.

## How it works

```
game server                          render box
+--------------------+   TCP 25590   +--------------------------------------------------+
| Paper + RenderFox  | ------------> | rfengine: world mirror, lighting, CPU renderer,  |
| plugin (read-only) |  paired, then |   overlay, patched x264: H.264 1080p24 in FLV,   |
+--------------------+  blocks and   |   or an unchanged orbit replayed from memory     |
                        players      |     |                                            |
                                     |     v            radio.py: your music as AAC,    |
                                     | ffmpeg: copies video, adds audio <-- paced to it |
                                     |     |                                            |
                                     |     v                                            |
                                     | MediaMTX relay on 127.0.0.1:1935/fm              |
                                     |     |                    |                       |
                                     |     v                    v                       |
                                     | uplink (primary)     uplink (backup)             |
                                     +-----|--------------------|-----------------------+
                                           v                    v
                                 rtmps://a.rtmps.youtube.com   rtmps://b.rtmps.youtube.com
```

The plugin dials the engine and sends the area around spawn, then every change as it
happens. The engine (`rfengine`) keeps its own copy of that world, renders the camera's
orbit and encodes it to H.264 itself. `radio.py` plays your music, ffmpeg puts the two
together, and a local relay hands the stream to one or two uplinks that send it to the
streaming site. [docs/architecture.md](docs/architecture.md) explains each part in depth.

## Requirements

| | |
|---|---|
| Render box | x86-64 Linux (Ubuntu or Debian), 4 cores with AVX2 and FMA (Intel Haswell or newer, AMD Zen), 4 GB of RAM (6 to be comfortable), 5 GB of disk plus your music (2 GB of it is the replay recording). In a VM, give it a CPU type that passes AVX2 through (`host` on Proxmox/KVM). No GPU. |
| Game server | Paper 26.2 or newer, Java 25. It dials the render box on TCP 25590, so nothing new opens on the game server. |
| Uplink | About 11 Mbit/s up, steady, for 1080p at 10 Mbit/s. |
| Output | A YouTube channel with live streaming enabled (or Twitch, or any RTMP/RTMPS ingest). |

At 1080p24 rendering live, the engine and x264 use nearly all of four Haswell cores. A
faster CPU has room to spare; a slower one can run 720p (`width`, `height` in
`renderfox.json`).

## Quick start

On the render box, as the user the stream should run as:

```sh
git clone https://github.com/Experimentiale-Developments/RenderFox.git ~/renderfox
cd ~/renderfox
./setup.sh                    # packages, assets, fonts, Rust, x264, the engine, the services
renderfox youtube             # paste your stream key; it is not shown or logged
cp ~/my-music/*.flac music/   # any FLAC, Ogg, Opus, WAV, MP3 or M4A
nano renderfox.json           # the overlay text, and the orbit's centre over your spawn
renderfox start
renderfox pair                # prints the command to run on the game server
```

On the game server, put `RenderFox-<version>.jar` (from the releases, or
`cd plugin && mvn package`) in `plugins/`, restart, and run what `renderfox pair` printed
from the console or as an operator:

```
/renderfox pair 192.168.1.50:25590 7KQ2M-X9PRT
```

`renderfox status` then shows the server feeding the engine and the uplinks sending.
[docs/getting-started.md](docs/getting-started.md) walks through all of it, YouTube
Studio's side included.

## Repository

| Path | What |
|---|---|
| `engine/` | The renderer and encoder, `rfengine`, in Rust. |
| `x264/` | x264 pinned to a commit, with the patches that make replay and the spliced card possible. |
| `plugin/` | The Paper plugin that feeds the engine (Maven). |
| `stream/` | `encode.sh` (engine, radio and ffmpeg together), `radio.py` (the music), the `renderfox.json` template. |
| `relay/` | The local MediaMTX relay, the uplinks, the watchdog, a TCP keepalive tweak. |
| `systemd/` | The services, as templates `install.sh` fills in for this checkout. |
| `renderfox` | The command-line tool for the render box: status, pairing, outputs. |
| `setup.sh`, `install.sh` | First-time setup, and installing or refreshing the services. |
| `tools/` | For development: a fake game server and a whole-stack bench, a stand-in for YouTube's ingest with a validator, a test plugin with walking mannequins. |
| `docs/` | Everything else, below. |

## Documentation

- [Getting started](docs/getting-started.md): from an empty machine to live.
- [Pairing](docs/pairing.md): how the server and engine trust each other, moving to another server.
- [Outputs](docs/outputs.md): YouTube, Twitch, any RTMP ingest, stream keys.
- [Configuration](docs/configuration.md): every key of `renderfox.json`, the plugin's `config.yml` and commands.
- [Operations](docs/operations.md): services, logs, what failures look like, upgrades, memory.
- [Upgrading without going off air](docs/live-migration.md): swapping the engine on a live YouTube stream.
- [Architecture](docs/architecture.md): the renderer, the feed and replay, in depth.
- [Protocol](docs/protocol.md): the plugin to engine wire format.
- [YouTube ingest](docs/youtube-ingest.md): what YouTube accepts and why the encoder is set up as it is.
- [x264 patches](x264/README.md): the region, bands and macroblock copies.
- [Development](docs/development.md): building, tests, benchmarks.
- [Known issues](docs/known-issues.md).

## Limits

- Players and mannequins are drawn; mobs, items and other entities are not.
- One fixed daytime, no weather.
- The engine mirrors a 224-block radius around one point; terrain beyond fades into fog.
- One game server per engine, one engine per server.
- x86-64 only.

## License

Render Fox is free software under the [GNU General Public License v3.0 or later](LICENSE).
The engine links x264, which is GPL-2.0-or-later, and the patches in `x264/patches/` are
under x264's own license. [THIRD_PARTY.md](THIRD_PARTY.md) lists everything Render Fox
builds on or downloads. Minecraft's assets are fetched from Mojang on your machine and
never redistributed. Render Fox is not affiliated with Mojang or Microsoft.
