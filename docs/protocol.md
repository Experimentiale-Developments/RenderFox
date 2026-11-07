# Observer to engine protocol (version 2)

The observer plugin on the game server dials the engine (default port 25590) and
streams the area around spawn. After the handshake nothing flows back: the engine only
ever closes the connection. All integers are big endian (Java `DataOutputStream`),
strings are Java modified UTF-8 with a 16-bit length (`writeUTF`).

## Who may connect

Exactly one game server feeds the engine: the one it is paired with. The plugin makes
an Ed25519 key pair on first start (`plugins/RenderFox/identity.key`, PKCS#8 DER, mode
0600, which never leaves that folder, and `identity.pub`, the raw public key in base64).
The engine trusts one public key: `observerKey` in `renderfox.json` if set, otherwise the
key file pairing writes (`pairing.keyFile`, `observer.pub` by default).

```
engine                                   plugin
  CHALLENGE  protocol, 32 random bytes  ->
                                        <-  HELLO  ..., its public key, signature
  ACCEPT (or close the connection)      ->
                                        <-  SECTIONS, ENTITIES, ...
```

The signature covers the ASCII bytes `RenderFox observer hello v2`, then the nonce, then
the hello's body, so a recorded hello is worthless on the next connection. The engine
checks the key against the one it trusts before the signature (with `verify_strict`),
and logs a refusal with the key it was offered, at most every ten minutes per reason. A
peer has 5 s to answer the challenge; the plugin gives the engine 10 s for each step.
It serves one connection at a time, so a second server waits until the first is gone
and is then refused.

### Pairing

An engine that trusts no key is unpaired. At start it makes a one-time code of ten
characters from Crockford's base32 (`0-9`, `A-Z` without `I`, `L`, `O`, `U`), shown as
`XXXXX-XXXXX`, logs it and writes it to `pairing.codeFile` (`run/pairing.code`, mode
0600). The plugin answers a challenge with PAIR instead of HELLO:

```
engine                                   plugin
  CHALLENGE  protocol, 32 random bytes  ->
                                        <-  PAIR  protocol, key, proof, signature
  PAIRED, or REFUSED with a reason      ->
  (closes; the plugin reconnects and says HELLO as above)
```

- `proof` is HMAC-SHA256 keyed with the code (upper case, separators dropped, `O` read
  as `0` and `I`, `L` as `1`) over `RenderFox pair v1`, the nonce and the key. The code
  itself never crosses the network, and a recorded PAIR is worthless on another
  connection.
- `signature` is the key's Ed25519 signature over `RenderFox pair v1`, the nonce, the
  key and the proof, so the server proves it holds the private half of what it pairs.
- On success the engine writes the key to the key file (written beside and renamed
  over), forgets the code and answers PAIRED. Pairing again with the key already paired
  answers PAIRED too, so a plugin that lost the answer can retry.
- A wrong code answers REFUSED; after five, the engine makes a new code. An engine that
  is already paired with another key refuses every PAIR until `renderfox unpair` removes
  the key file.

What this does not cover: the messages after the handshake are not signed, so someone
able to take over a TCP connection on the network between the two could inject into it,
and the plugin does not check that it is talking to the real engine (it sends only what
the stream shows anyway). Keep the feed port reachable from the game server only, or
send it through a tunnel (WireGuard, SSH) when the two are far apart.

## Framing

```
int32  length      bytes that follow, type byte included
byte   type
byte[] payload     length - 1 bytes
```

| type | name          | payload                                   | notes |
|-----:|---------------|-------------------------------------------|-------|
| 1    | HELLO         | plain                                     | always the first message |
| 2    | SECTIONS      | **deflated** (zlib, `Deflater.BEST_SPEED`) | one chunk column's changed sections |
| 3    | SNAPSHOT_DONE | empty                                     | the initial area has been sent once |
| 4    | ENTITIES      | plain                                     | 20 times a second, may be dropped under backlog |
| 5    | ENTITY_INFO   | plain                                     | when an entity first appears or its look changes |
| 6    | PING          | empty                                     | after 5 s with nothing else to send |
| 7    | CHALLENGE     | plain, engine to plugin                   | the first message on a connection |
| 8    | ACCEPT        | empty, engine to plugin                   | the hello was accepted |
| 9    | PAIR          | plain                                     | instead of HELLO, to pair |
| 10   | PAIRED        | empty, engine to plugin                   | paired; the engine closes |
| 11   | REFUSED       | UTF reason, engine to plugin              | pairing refused; the engine closes |

## CHALLENGE

```
int32  protocol     2
byte   nonce[32]    fresh from /dev/urandom for every connection
```

## HELLO

```
int32  protocol     2
byte   key[32]      the server's Ed25519 public key; must be the one the engine trusts
UTF    world        e.g. minecraft:overworld
int32  minY, maxY   world height range
int32  centerX, centerZ, radius   area in blocks
int32  entityHz     entity updates per second
byte   signature[64]  Ed25519 over "RenderFox observer hello v2" + nonce + everything above
```

## PAIR

```
int32  protocol     2
byte   key[32]      the server's Ed25519 public key
byte   proof[32]    HMAC-SHA256(code, "RenderFox pair v1" + nonce + key)
byte   signature[64]  Ed25519 over "RenderFox pair v1" + nonce + key + proof
```

## SECTIONS (after inflating)

```
int32  chunkX, chunkZ
int16  count
section[count]:
    byte   sy                 section Y index (blockY >> 4)
    int16  paletteSize        0 means the section is empty (all air), nothing follows
    UTF    blockData[paletteSize]   e.g. minecraft:oak_stairs[facing=east,half=bottom,...]
    indices[4096]             byte each, or int16 each if paletteSize > 256;
                              order (y * 16 + z) * 16 + x
    byte   biomeCount
    UTF    biome[biomeCount]  e.g. minecraft:plains
    byte   cells[64]          4x4x4 biome cells, same order
```

A section is only sent when its content hash changed since it was last sent on this
connection, so a reconnect resends everything and the engine ignores what it
already has.

Changes go out at the end of the server tick they happen in (Paper's
`ServerTickEndEvent`). A block event names the sections within one block of it,
neighbours included, and only those are encoded again; the rest of the column is not
touched. Events cover players placing, breaking and using blocks (doors, levers,
buttons, pressure plates), redstone power changes, pistons, fluids, growth, fire and
explosions. What fires no event (`/setblock`, `/fill`, WorldEdit, other plugins) is
found by the sweep, which takes six chunks a second: half from the 4-chunk core the
camera circles (about 20 s to go round it), a third from the 9-chunk inner ring, the
rest from the whole area.

## ENTITIES

Players and mannequins only, the entities the engine draws, twenty times a second.

```
int64  frame, worldTick   the engine places positions on the tick timeline
int16  count
entity[count]:
    int32  id
    double x, y, z
    float  yaw, pitch, bodyYaw
    byte   pose        org.bukkit.entity.Pose ordinal
    byte   flags       1 sneaking, 2 sprinting, 4 swing main, 8 swing off hand,
                       16 hurt, 32 burning, 64 on ground
```

An entity missing from a frame has left the area. Spectators, vanished players and
invisible entities are never sent.

## ENTITY_INFO

```
int32  id
UTF    type        e.g. minecraft:player
UTF    name
int32  skinLength, byte[skinLength]   UTF-8: the base64 "textures" property, or
                                      "uuid:<uuid>" when there is none
byte   gearCount
UTF    gear[gearCount]   item keys: main hand, off hand, head, chest, legs, feet
                         ("" for empty)
```

## Flow control

Terrain is never dropped: if the plugin's unsent backlog passes 96 MB it closes the
connection and the reconnect resynchronises. Entity frames and entity info have a lane
of their own that is always sent first, so a player never waits behind a burst of
terrain; entity frames are skipped only if that lane itself backs up past 1 MB. If the
socket stops accepting data for 20 s the plugin drops it. Reconnects back off from 1 s
to 10 s.
