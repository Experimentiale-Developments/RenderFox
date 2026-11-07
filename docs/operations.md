# Operations

## Services

`install.sh` writes these from `systemd/*.service.in` for the checkout it sits in, run as
the checkout's owner (the watchdog as root, to restart uplinks).

| Unit | What |
|---|---|
| `renderfox-encode` | `stream/encode.sh`: engine, radio and ffmpeg, publishing to the relay. If any of them exits, all three stop and systemd restarts them after 5 s. |
| `renderfox-relay` | MediaMTX on 127.0.0.1:1935 (RTMP) and 127.0.0.1:9997 (its API); nothing on the network. |
| `renderfox-uplink` | `relay/uplink.sh primary`: the relay to the ingest. |
| `renderfox-uplink-backup` | `relay/uplink.sh backup`: the relay to the backup ingest; enabled only when the output has one (YouTube). |
| `renderfox-watchdog` | `relay/watchdog.py`: restarts an uplink that is not running when it should be, logs a stalled encoder or a choked uplink. |
| `renderfox-test-ingest` | Testing only (`install.sh --test-ingest`): a stand-in for YouTube on `rtmps://localhost:1936`. |

```sh
renderfox status               # all of it at a glance
renderfox start | stop | restart
renderfox logs                 # the encoder's journal; also: relay, uplink, backup, watchdog
curl -s 127.0.0.1:9997/v3/paths/get/fm    # the relay: bytes in, readers
```

The engine writes a `stream:` line to the journal every minute: frames rendered,
replayed, repeated or sent late, the bitrate, edits shown and how long they took, and
whether the feed is connected. Repeated frames are frames the encoder had to send twice
because the engine was late; a few after a start are normal.

## What failures look like

| What happens | What the stream shows |
|---|---|
| The game server restarts or goes away | The last known world, music playing. It resyncs by itself when the server is back (a full snapshot takes seconds). |
| The engine, radio or ffmpeg dies | systemd restarts the encoder within about 15 s; the ingest sees a short drop. Recordings on disk survive, so replay resumes as soon as the world is in. |
| One ingest connection drops | With YouTube's backup, nothing: YouTube plays the other while this uplink reconnects within seconds. Without a backup, a short drop. |
| The network goes away for longer | The uplinks retry every second; the renderer keeps running. YouTube may end the broadcast after a long outage and start a new one when the stream returns (with Auto-start on). |
| The render box reboots | Everything comes back on boot. |

The guardrails against a stalled connection: uplinks time out reads and writes after 5 s
(`-rw_timeout`) and restart after 1 s; `relay/99-renderfox-tcp.conf` (installed into
`/etc/sysctl.d`) has the kernel notice a dead TCP connection within about 19 s; the
watchdog restarts an uplink systemd has given up on.

## Testing the renderer without the stream

Only one engine can take the game server's feed, so stop the encoder first
(`renderfox stop`), then from the checkout:

```sh
E=engine/target/release/rfengine
$E --snap still.png --angle 214                    # one frame at an orbit angle
$E --bench 90 --angle 100 --dump frames.gray       # timing, and every frame's luma
$E --bench 96 --angle 160 --rgb frames.rgb         # every frame as raw RGB, 6 MB each
python3 engine/tools/stability.py frames.gray      # jitter, sharpness, the refresh pulse
```

The engine waits for the paired server's snapshot before rendering. `--realtime` keeps
each benched frame in its slot of the wall clock: players are posed on the engine's clock,
so a bench that runs ahead shows them frozen. Writing raw frames at 24 a second is more
than many disks take, and the stalls look exactly like players stuttering, so judge
players without `--rgb`. Judge the renderer from `--dump`, not from an encoded `.mp4`:
x264's own frame types add a sharpness rhythm of their own.

## Upgrading

```sh
cd ~/renderfox
git pull
./x264/build.sh                    # only if x264/ changed
(cd engine && ~/.cargo/bin/cargo build --release --locked)
sudo ./install.sh                  # only if systemd/ changed
renderfox restart
```

Build at low priority (`nice -n 19`) if the stream is live: the build competes with the
renderer. A new engine build starts without its predecessor's recordings, so the first
lap renders in full. To swap the engine without the broadcast seeing a drop, see
[live-migration.md](live-migration.md).

The plugin: replace the jar in `plugins/` and restart the server. The pairing survives.
The engine and plugin check each other's protocol version and say which one to update.

## Memory and disk

The engine uses about 700 to 800 MB while rendering. The replay recording is about 2 GB on
disk at 1080p (`run/replay/`), and the kernel's page cache over it counts towards the
service's memory in `systemctl status`, though it is reclaimable: `free -h` shows what is
really available. Skins are cached in `cache/skins`, the encoded music in `cache/aac`.

## Files on the render box

All inside the checkout, none in git:

| Path | What |
|---|---|
| `renderfox.json` | The engine's config. |
| `observer.pub` | The paired server's public key. |
| `relay/uplink.env` | Where the stream goes, with the stream key (mode 0600). |
| `music/` | Your music. |
| `assets/`, `fonts/`, `deps/` | Fetched by `setup.sh`. |
| `run/` | FIFOs, the stream epoch, the now playing title, the pairing code, the replay recording. |
| `cache/` | Skins and encoded music. |
