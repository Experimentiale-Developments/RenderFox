# Security

Please report vulnerabilities privately, through GitHub's "Report a vulnerability" on the
repository's Security tab, not in a public issue.

What Render Fox protects, and what it does not, is described in
[docs/pairing.md](docs/pairing.md) and [docs/protocol.md](docs/protocol.md): the engine
streams only the server it is paired with, but the feed after the handshake is neither
encrypted nor signed, so keep the feed port reachable from the game server only, or
tunnel it.

Stream keys live in `relay/uplink.env` on the render box (mode 0600, never in git). If
one leaks, reset it on the streaming platform and run `renderfox youtube` (or `twitch`,
`rtmp`) again.
