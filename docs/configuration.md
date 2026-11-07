# Configuration

## renderfox.json (the engine)

At the top of the checkout, made by `setup.sh` from `stream/renderfox.json`, and never
overwritten by it. It is JSON that may carry `//` and `/* */` comments. Relative paths
count from the file's own folder. The engine reads it at start: change it, then
`renderfox restart`.

Changing anything that alters the picture (size, look, orbit, overlay text, encoder), or
running another engine build, throws away the replay recording, so the next orbit renders
in full before replay resumes.

| Key | Default in the template | What |
|---|---|---|
| `assets` | `assets/client` | Minecraft's client assets, unpacked from the client jar (`setup.sh` fetches them). |
| `cache` | `cache` | Downloaded skins and capes (`cache/skins`). |
| `port` | 25590 | TCP port the engine listens on for the game server, on every interface. |
| `observerKey` | `""` | A server's public key, to trust it without pairing ([pairing.md](pairing.md)). Leave empty to pair. |
| `pairing.keyFile` | `observer.pub` | Where pairing saves the paired server's key. |
| `pairing.codeFile` | `run/pairing.code` | Where an unpaired engine writes its code for `renderfox pair`. |
| `width`, `height` | 1920, 1080 | Output size. 1280x720 needs well under half the CPU. |
| `fps` | 24 | Frames a second. At 1080p, 24 is what four Haswell cores sustain. |
| `threads` | 4 | Rendering threads, the CPU's core count. |
| `staticEvery` | 7 | The world is rendered in full every Nth frame and reprojected in between; players are drawn every frame. Lower is sharper in motion and costs more. |
| `aheadFrames` | 4 | Frames rendered ahead of the one going out: room to absorb a slow frame, at one frame of delay each. 1 to 16. |
| `fancyLeavesRadius` | 96 | Leaves within this many blocks of the orbit's centre are drawn with their cut-outs, beyond it as solid blocks. |
| `plantSway` | false | Grass and flowers sway. |

**`encoder`**

| Key | Default | What |
|---|---|---|
| `bitrateKbps` | 10000 | Constant bitrate. YouTube recommends 10000 at 1080p, 4000 at 720p; Twitch takes 6000 at most. |
| `preset` | `superfast` | x264 preset. Slower presets look better and cost CPU the renderer needs. |
| `threads` | 2 | x264's threads. |

The rest of the encoder (CBR with NAL HRD, a closed 2 s GOP, High profile, two B-frames)
is fixed to what YouTube asks for: [youtube-ingest.md](youtube-ingest.md).

**`replay`** ([architecture.md](architecture.md), "Replay")

| Key | Default | What |
|---|---|---|
| `enabled` | true | Replay an unchanged orbit instead of rendering it again. Off renders every frame. |
| `store` | `run/replay/region.bin` | The scene under the now playing card for every frame of the orbit, about 1.5 GB at 1080p. The recorded pictures are kept beside it. |
| `settleSeconds` | 4 | Seconds after a change or a player before frames may be recorded, so antialiasing has settled. |
| `warmSeconds` | 1.5 | Seconds the renderer runs from cold before its frames are recorded, and ahead of the end of a replay so it takes over warm. |
| `natureSeconds` | 1800 | Changes no player made (grass eaten, crops growing) more than 10 blocks from any player are held back and shown together this often, so they do not keep the whole orbit rendering. 0 shows them at once. |

**`orbit`**

| Key | Default | What |
|---|---|---|
| `center` | `[0, 77, 0]` | The point the camera circles, `[x, y, z]` in blocks. Match the plugin's `center`. |
| `radius` | 55 | Distance from the centre, in blocks. Where something on the circle stands in the way, the camera pulls in rather than climbing. |
| `height` | 22 | Height above the centre. |
| `lookHeight` | 0 | Where it looks, above the centre. |
| `periodSeconds` | 360 | One lap, in seconds. Replay needs a whole number of 2 s groups of frames in a lap (an even number of seconds); otherwise every frame is rendered. |
| `fov` | 70 | Vertical field of view, degrees. |

The camera's angle follows the wall clock, so a restart resumes the orbit where it was.

**`overlay`**

| Key | Default | What |
|---|---|---|
| `fonts` | `fonts` | A folder with `Inter-SemiBold.ttf` and `Inter-Medium.ttf`. Empty: no overlay at all. |
| `nowPlaying` | `run/nowplaying.txt` | The file the radio writes the current title to. |
| `station` | `RENDER FOX` | Top banner, left. |
| `invite` | `Join the server to see yourself in the live stream` | Top banner, right. |
| `joinLabel` | `JOIN US` | Bottom right card, the small line. |
| `address` | `play.example.com` | Bottom right card, large: what viewers type to join. |

**`look`**

| Key | Default | What |
|---|---|---|
| `sunElevation`, `sunAzimuth` | 36, 118 | Sun position in degrees. Always daytime. |
| `sunColor`, `sunIntensity` | `[1.0, 0.86, 0.7]`, 3.3 | Sunlight. |
| `skyZenith`, `skyHorizon` | `[0.1, 0.26, 0.75]`, `[0.5, 0.66, 0.9]` | Sky gradient, linear RGB. |
| `ambient`, `ambientIntensity` | `[0.42, 0.55, 0.85]`, 0.85 | Skylight colour and strength. |
| `fogStart`, `fogEnd` | 140, 250 | Fog distances in blocks. |
| `exposure`, `saturation` | 0.95, 1.0 | Grading. |

### Environment

`stream/encode.sh` and the engine read a few variables, for testing:

| Variable | What |
|---|---|
| `RF_HOME` | The checkout (default: the folder above `stream/`). |
| `RF_CONFIG` | The config (default `$RF_HOME/renderfox.json`). |
| `RF_MUSIC` | The music folder (default `$RF_HOME/music`). |
| `RF_RELAY` | Where the encoder publishes (default `rtmp://127.0.0.1:1935/fm`). |
| `RF_ENGINE_ARGS` | Extra engine arguments. |
| `RF_VERIFY=dir` | Also renders every replayed frame and compares it with the recording, saving the worst to `dir`. Testing only: it renders everything. |
| `RF_KEEP_ANY_BUILD=1` | Keeps recordings across engine builds. Only for changes that cannot alter the picture. |

## config.yml (the plugin)

In `plugins/RenderFox/` on the game server. `/renderfox reload` applies changes.

| Key | Default | What |
|---|---|---|
| `engine` | `""` | The engine as `host:port`, set by `/renderfox pair`. Empty leaves the plugin idle. |
| `world` | `world` | The world to mirror. |
| `center.x`, `center.z` | 0, 0 | The middle of the mirrored area. Match the engine's `orbit.center`. |
| `radius` | 224 | How far around it to mirror, 16 to 512 blocks. The engine's fog is set for 224. |
| `chunks-per-tick` | 6 | Chunks captured per tick while the first snapshot loads. |
| `sweep-chunks-per-second` | 6 | Chunks re-checked each second for changes no event reports (WorldEdit, `/fill`). |

The plugin's key pair sits beside it: `identity.key` and `identity.pub`.

### Commands

All need `renderfox.admin` (operators by default) and work from the console.

| Command | What |
|---|---|
| `/renderfox` | Connected or not, chunks watched and pending, bytes sent, this server's key. |
| `/renderfox pair <host[:port]> <code>` | Pair with an engine ([pairing.md](pairing.md)). |
| `/renderfox unpair` | Stop feeding the engine, and forget its address. |
| `/renderfox key` | This server's public key. |
| `/renderfox reload` | Re-read `config.yml` and reconnect. |

### What it sends

Blocks within `radius` of `center`, as a snapshot on every connection and then as they
change: placing, breaking, doors, levers, buttons, pressure plates, redstone, pistons,
fluids, growth, fire, explosions, at the end of the tick they happen in. A slow sweep
catches what no event reports. Players and mannequins in the area, twenty times a second,
with their names, skins and equipment. Never spectators, players hidden by a vanish
plugin (the `vanished` metadata), or invisible entities. It reads the world only.
