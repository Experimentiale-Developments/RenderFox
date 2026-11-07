# RenderFox plugin

The Paper plugin that feeds a Render Fox engine: it mirrors the blocks and players around
one point of one world to the engine, and never changes the world.

```sh
mvn package        # Java 25, Maven; the jar lands in target/RenderFox-<version>.jar
```

Drop the jar in the server's `plugins/` folder (Paper 26.2 or newer), restart, and pair
it with `/renderfox pair <host[:port]> <code>`, the code coming from `renderfox pair` on
the render box. [../docs/pairing.md](../docs/pairing.md) explains pairing,
[../docs/configuration.md](../docs/configuration.md) the `config.yml` and commands.

| Class | What |
|---|---|
| `RenderFoxPlugin` | Config, the `/renderfox` command, and the block events that mark terrain to resend. |
| `ChunkFeed` | The snapshot, the changed sections, and the sweep that catches changes no event reports. |
| `EntityFeed` | Players and mannequins twenty times a second; names, skins and equipment when they change. |
| `Link` | The connection to the engine: handshake, pairing, two send lanes, reconnects. |
| `Identity` | This server's Ed25519 key pair. |
| `Pairing` | The PAIR message's proof and signature. |
| `Wire`, `Address` | Framing, and parsing `host:port`. |
