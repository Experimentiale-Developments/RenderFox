# Development

## Building

```sh
./x264/build.sh                                   # clones x264 at the pinned commit, patches, builds
cd engine && cargo build --release --locked       # the engine; Rust is pinned by rust-toolchain.toml
cd plugin && mvn package                          # the plugin, into plugin/target/
cd tools/testactors && mvn package                # the test plugin
```

The engine needs `nasm` (for x264), `libavcodec-dev` and `pkg-config` (it decodes its own
recordings), and a C compiler. It is built for the CPU it is built on
(`engine/.cargo/config.toml`, `target-cpu=native`).

Never edit `x264/third_party/x264` by hand: `prepare.sh` resets it on every build, and a
change to x264 is a change to the patches. Never link the system's libx264 instead: without
the patches the spliced card is not exact.

## Tests

```sh
cd engine
cargo test --release                                 # unit tests
cargo test --release -- --ignored spliced            # the x264 splice, bit for bit (needs ffmpeg)
cargo test --release -- --ignored copied syntax      # macroblock copies (needs libavcodec)
RF_ASSETS=../assets/client cargo test --release -- --ignored   # everything, with Mojang's assets
cd ../plugin && mvn test
```

Some ignored tests are benchmarks or need a clip of real footage (`RF_SPLICE_CLIP`, raw
1080p yuv420p); their `ignore` message says what they need. The pairing proof has the same
test vector in `engine/src/net.rs` and `plugin/.../PairingTest.java`: change one, change
both.

## Without a game server

- `engine/tools/feedtool.py` speaks the observer's side of the protocol: `key` makes a key
  pair, `pair` pairs an engine with it, `record` stands in for the engine and records a real
  server's snapshot, `replay` feeds that recording to an engine. Two engine builds fed the
  same recording see byte-identical worlds, for comparing them frame for frame.
- `tools/bench/fakeserver.py` generates a world (hills, a lake, forests, houses, a tower for
  the orbit to steer around), speaks the protocol and plays a scenario: players walking,
  blocks placed and broken, or nothing.
- `tools/bench/stack.sh DIR` runs the whole render box on one machine: the fake server, the
  encoder, the relay, both uplinks over RTMPS to a local stand-in ingest.
  `tools/bench/measure.py` samples every role's CPU and memory from `/proc`.

These need Python's `cryptography` and `numpy`.

## With a game server

`tools/testactors/` is a Paper plugin for test servers only: `/actors start` spawns
mannequins walking around spawn and a builder raising and removing an arch, so moving
players and terrain updates can be watched end to end. Never install it on a real server.

`tools/test-ingest/` stands in for YouTube's ingest; [outputs.md](outputs.md) shows how to
point the stream at it and check what arrives with `validate.py`. `validate.py` checks the
newest recording by default, which is still being written; give it a finished one with
`--segment` before believing a failure at its end.

## Rules for changes

- New crates, Maven dependencies or versions: each version must have been published at
  least 7 days before it is added, pinned exactly (`=x.y.z` in `Cargo.toml`), with
  `Cargo.lock` committed. Downloads in `setup.sh` are pinned by hash.
- Anything the renderer animates runs on orbit time, never wall time, or replay and
  rendering stop meeting seamlessly.
- The plugin only ever reads the world, and never sends spectators, vanished or invisible
  players.
- Check a whole lap, not chosen angles, before judging a rendering change.
