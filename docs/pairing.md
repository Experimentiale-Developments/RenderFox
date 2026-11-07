# Pairing

An engine streams exactly one game server, and only a server it is paired with can feed
it. Pairing is how the two meet: the engine shows a one-time code, and the server proves
it knows the code.

## Pairing a server

1. Start the stream on the render box (`renderfox start`). While unpaired, the engine
   makes a code like `7KQ2M-X9PRT`, logs it and writes it to `run/pairing.code`.
2. `renderfox pair` shows it with the command to run, for example:

   ```
   /renderfox pair 192.168.1.50:25590 7KQ2M-X9PRT
   ```

3. Run that on the game server, in the console or as an operator with
   `renderfox.admin`. The address is however the game server reaches the render box;
   the port is the engine's `port` (25590 unless changed), and may be left out when it
   is 25590. Case and dashes in the code do not matter, and `O`, `I` and `L` are read as
   `0`, `1` and `1`.
4. The server answers "Paired with the Render Fox engine at ...", saves the address as
   `engine` in `plugins/RenderFox/config.yml`, reconnects and sends its snapshot. The
   engine saves the server's public key to `observer.pub`.

From then on the engine accepts that server and refuses any other, across restarts of
either. If pairing fails, the server says why and goes back to what it did before:

| The server says | Why |
|---|---|
| refused to pair: tried to pair with the wrong code | The code is wrong. After five wrong codes the engine makes a new one: run `renderfox pair` again. |
| refused to pair: ... already paired with another server | Run `renderfox unpair` on the render box first. |
| engine unreachable | The address or port is wrong, or a firewall blocks TCP 25590 to the render box. |
| took the connection but sent no challenge | Another server is connected right now; the engine serves one at a time. |

## Moving the stream to another server

On the render box, `renderfox unpair` forgets the paired server, restarts the encoder and
shows a new code; pair the new server with it. On the old server, `/renderfox unpair`
stops it from trying to connect (otherwise it keeps retrying quietly, and is refused).

Never copy `plugins/RenderFox/identity.key` from one server to another: whoever holds it
can feed the stream. A copy of the server (a staging clone, a backup restored elsewhere)
that brings the key along would be accepted too, so leave `identity.key` out of copies,
or delete it there and let that copy make its own.

## What it proves

Each server has its own Ed25519 key pair, made by the plugin on first start:
`identity.key` (PKCS#8 DER, readable by the server's user only) and `identity.pub` (the
public key in base64), in `plugins/RenderFox/`. Every connection starts with a fresh
random challenge from the engine.

- To pair, the server sends its public key, an HMAC of the challenge and the key with the
  code as its key, and a signature over all of it. The code itself never crosses the
  network, a recorded request is useless on the next connection, and the signature shows
  the server holds the private half of the key it pairs.
- Every connection after that, the server signs the challenge with the same key, and the
  engine checks it against the one it saved.

What it does not do: messages after the handshake are not signed or encrypted. Someone
who can intercept the network between the two could watch the feed (it carries what the
stream shows anyway) or inject into it. On a LAN that is usually acceptable; across the
internet, run the feed through a tunnel such as WireGuard or SSH, and keep port 25590
closed to everyone but the game server. [protocol.md](protocol.md) has the details.

## Pinning a key by hand

`observerKey` in `renderfox.json` takes precedence over pairing: set it to a server's
public key (from `/renderfox key`, or `identity.pub`) and restart the encoder, and the
engine trusts that key with no pairing at all. `renderfox unpair` refuses to run while
it is set.

To have a server's key before its first start (for example to configure the engine in
advance), make the pair yourself in the format the plugin reads:

```sh
cd plugins/RenderFox
umask 077
openssl genpkey -algorithm ed25519 -outform DER -out identity.key
openssl pkey -inform DER -in identity.key -pubout -outform DER | tail -c 32 | base64 -w0 > identity.pub
```

and give both files to the server's user.

## Making a new key

If a server's key may have leaked, delete `identity.key` and `identity.pub` from
`plugins/RenderFox/`, restart the server (it makes a new pair), run `renderfox unpair` on
the render box and pair again.
