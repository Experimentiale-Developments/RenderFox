#!/usr/bin/env python3
"""A stand-in for the game server and its observer plugin, for measuring the engine.

It generates a world around spawn (hills, a lake, forests, flowers, a spawn platform,
houses, and a tall white tower southwest of spawn for the orbit to steer around), speaks the observer protocol
(docs/protocol.md, version 2) to the engine, and then plays a scenario: players walking
about spawn at 20 Hz, blocks placed and broken, or nothing at all. Deterministic for a
given seed, so two engines see the same world and the same players.

    fakeserver.py key bench.key                       make a key; prints its public half
    fakeserver.py run --key bench.key [--port 25590] [scenario options]

Scenario options:
    --players N          players walking near spawn (default 0)
    --players-at T       seconds after the snapshot when they join (default 0)
    --players-for T      seconds they stay (default: forever)
    --edits-per-min N    blocks placed and broken near spawn by an invisible builder
    --edits-at T         seconds after the snapshot when edits start
    --seed N             world and players
    --radius R           area radius in blocks (default 224, as the plugin's default)

Runs until killed. Needs numpy and cryptography.
"""
import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
import zlib
from base64 import b64encode

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HELLO, SECTIONS, SNAPSHOT_DONE, ENTITIES, ENTITY_INFO, PING, CHALLENGE, ACCEPT = 1, 2, 3, 4, 5, 6, 7, 8
PROTOCOL = 2
CONTEXT = b"RenderFox observer hello v2"
MIN_Y, MAX_Y = -64, 320
SEA = 62

# ---------------------------------------------------------------------------------------
# Blocks: a palette of full block state strings, as Paper's BlockData.getAsString() gives.

STATES = [
    "minecraft:air",
    "minecraft:stone", "minecraft:deepslate[axis=y]", "minecraft:dirt", "minecraft:grass_block[snowy=false]",
    "minecraft:sand", "minecraft:gravel", "minecraft:water[level=0]", "minecraft:coal_ore", "minecraft:iron_ore",
    "minecraft:oak_log[axis=y]", "minecraft:oak_leaves[distance=1,persistent=false,waterlogged=false]",
    "minecraft:birch_log[axis=y]", "minecraft:birch_leaves[distance=1,persistent=false,waterlogged=false]",
    "minecraft:spruce_log[axis=y]", "minecraft:spruce_leaves[distance=1,persistent=false,waterlogged=false]",
    "minecraft:short_grass", "minecraft:tall_grass[half=lower]", "minecraft:tall_grass[half=upper]",
    "minecraft:poppy", "minecraft:dandelion", "minecraft:cornflower", "minecraft:oxeye_daisy", "minecraft:fern",
    "minecraft:stone_bricks", "minecraft:polished_andesite", "minecraft:smooth_stone",
    "minecraft:stone_brick_slab[type=bottom,waterlogged=false]",
    "minecraft:lantern[hanging=false,waterlogged=false]",
    "minecraft:oak_planks", "minecraft:cobblestone", "minecraft:glass", "minecraft:spruce_planks",
    "minecraft:white_concrete", "minecraft:light_gray_concrete", "minecraft:bricks", "minecraft:oak_slab[type=bottom,waterlogged=false]",
    "minecraft:dirt_path", "minecraft:lily_pad", "minecraft:sugar_cane", "minecraft:glowstone",
    "minecraft:stripped_oak_log[axis=y]", "minecraft:hay_block[axis=y]", "minecraft:cyan_wool", "minecraft:gold_block",
    "minecraft:redstone_block", "minecraft:sea_lantern", "minecraft:moss_block", "minecraft:mossy_cobblestone",
]
for f in ("north", "south", "east", "west"):
    STATES.append(f"minecraft:stone_brick_stairs[facing={f},half=bottom,shape=straight,waterlogged=false]")
    STATES.append(f"minecraft:oak_stairs[facing={f},half=bottom,shape=straight,waterlogged=false]")
ID = {}
for i, s in enumerate(STATES):
    ID.setdefault(s.split("[")[0].removeprefix("minecraft:"), i)  # "oak_log": its first state
    ID[s.removeprefix("minecraft:")] = i                          # "tall_grass[half=upper]"
    ID[s] = i                                                       # the full string


def B(name):
    return ID[name]


# ---------------------------------------------------------------------------------------
# Terrain


def value_noise(rng, size, cell):
    """Smooth 2D value noise over size x size, features about `cell` blocks across."""
    n = size // cell + 3
    grid = rng.random((n, n)).astype(np.float32)
    coords = np.arange(size, dtype=np.float32) / cell
    i = coords.astype(int)
    f = coords - i
    f = f * f * (3 - 2 * f)
    a = grid[i][:, i]
    b = grid[i + 1][:, i]
    c = grid[i][:, i + 1]
    d = grid[i + 1][:, i + 1]
    fy = f[:, None]
    fx = f[None, :]
    return (a * (1 - fy) + b * fy) * (1 - fx) + (c * (1 - fy) + d * fy) * fx


class World:
    def __init__(self, seed, radius):
        self.radius = radius
        self.c0 = -((radius + 15) // 16)  # first chunk
        self.c1 = (radius + 15) // 16      # one past the last chunk
        self.size = (self.c1 - self.c0) * 16
        self.x0 = self.c0 * 16
        rng = np.random.default_rng(seed)
        s = self.size
        h = (value_noise(rng, s, 64) * 14 + value_noise(rng, s, 24) * 6 + value_noise(rng, s, 9) * 2) + 56
        xs = np.arange(s) + self.x0
        X, Z = np.meshgrid(xs, xs)  # [z][x]
        d = np.hypot(X, Z)
        # Spawn's ground is levelled to just under the platform; a lake lies east of it.
        k = np.clip(1.6 - d / 25, 0, 1)
        h = h * (1 - k) + 72 * k
        lake = np.hypot(X - 70, Z + 25)
        h -= np.clip(1 - lake / 30, 0, 1) * 14
        self.height = np.round(h).astype(int)
        self.forest = value_noise(rng, s, 48)
        self.flowers = value_noise(rng, s, 12)
        self.rng = rng
        # Columns: blocks[z][x][y - MIN_Y]
        self.blocks = np.zeros((s, s, MAX_Y - MIN_Y), np.uint8)
        self.generate()

    def col(self, x, z):
        return self.blocks[z - self.x0, x - self.x0]

    def set(self, x, y, z, b):
        zi, xi = z - self.x0, x - self.x0
        if 0 <= zi < self.size and 0 <= xi < self.size and MIN_Y <= y < MAX_Y:
            self.blocks[zi, xi, y - MIN_Y] = b

    def get(self, x, y, z):
        zi, xi = z - self.x0, x - self.x0
        if 0 <= zi < self.size and 0 <= xi < self.size and MIN_Y <= y < MAX_Y:
            return self.blocks[zi, xi, y - MIN_Y]
        return 0

    def top(self, x, z):
        return int(self.height[z - self.x0, x - self.x0])

    def generate(self):
        s, rng = self.size, self.rng
        b = self.blocks
        Y = np.broadcast_to(np.arange(MIN_Y, MAX_Y)[None, None, :], b.shape)
        H = np.broadcast_to(self.height[:, :, None], b.shape)
        b[:] = 0
        b[Y < 0] = B("deepslate")
        b[(Y >= 0) & (Y < H - 3)] = B("stone")
        b[(Y >= H - 3) & (Y < H)] = B("dirt")
        top = Y == H
        under = H < SEA
        beach = H <= SEA + 1
        flowers = np.broadcast_to(self.flowers[:, :, None], b.shape)
        b[top & ~beach] = B("grass_block")
        b[top & beach] = B("sand")
        b[top & under & (flowers > 0.6)] = B("gravel")
        b[(Y > H) & (Y <= SEA)] = B("water")
        # A few ores in the stone, as caves show them (mostly hidden).
        ore = rng.random(b.shape) < 0.004
        b[ore & (b == B("stone"))] = B("coal_ore")
        # Plants and trees on grass.
        xs = np.arange(s) + self.x0
        for zi in range(s):
            for xi in range(s):
                x, z = xs[xi], xs[zi]
                h = self.height[zi, xi]
                if b[zi, xi, h - MIN_Y] != B("grass_block"):
                    if b[zi, xi, h - MIN_Y] == B("sand") and h == SEA and rng.random() < 0.04:
                        for k in range(1, 3 + int(rng.random() * 2)):
                            b[zi, xi, h + k - MIN_Y] = B("sugar_cane")
                    elif h == SEA - 1 and rng.random() < 0.03:
                        b[zi, xi, SEA + 1 - MIN_Y] = B("lily_pad")
                    continue
                if math.hypot(x, z) < 26:
                    continue  # spawn's own ground stays clear
                r = rng.random()
                f = self.forest[zi, xi]
                if r < 0.004 + 0.05 * max(0.0, f - 0.4):
                    self.tree(x, h + 1, z, "birch" if f > 0.7 else ("spruce" if f < 0.5 else "oak"))
                elif r < 0.25:
                    fl = self.flowers[zi, xi]
                    if fl > 0.72 and r < 0.12:
                        b[zi, xi, h + 1 - MIN_Y] = B(["poppy", "dandelion", "cornflower", "oxeye_daisy"][int(r * 1000) % 4])
                    elif r < 0.05:
                        b[zi, xi, h + 1 - MIN_Y] = B("tall_grass[half=lower]")
                        b[zi, xi, h + 2 - MIN_Y] = B("tall_grass[half=upper]")
                    elif r < 0.07:
                        b[zi, xi, h + 1 - MIN_Y] = B("fern")
                    else:
                        b[zi, xi, h + 1 - MIN_Y] = B("short_grass")
        self.spawn()
        self.houses()
        self.tower()

    def tree(self, x, y, z, kind):
        log, leaves = B(f"{kind}_log"), B(f"{kind}_leaves")
        rng = self.rng
        trunk = 4 + int(rng.random() * 3) + (3 if kind == "spruce" else 0)
        for k in range(trunk):
            self.set(x, y + k, z, log)
        top = y + trunk
        if kind == "spruce":
            for dy in range(-6, 2):
                r = max(0, (1 - dy) // 2 + (1 if dy < -2 else 0))
                for dz in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dx) + abs(dz) <= r + 1 and self.get(x + dx, top + dy, z + dz) == 0:
                            self.set(x + dx, top + dy, z + dz, leaves)
            return
        for dy in range(-3, 2):
            r = 2 if dy < 0 else 1
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) == r and abs(dz) == r and (dy == 1 or rng.random() < 0.5):
                        continue
                    if self.get(x + dx, top + dy, z + dz) == 0:
                        self.set(x + dx, top + dy, z + dz, leaves)

    def spawn(self):
        """A stone platform at spawn with steps, a fountain, lanterns and benches."""
        y = 76
        for z in range(-12, 13):
            for x in range(-12, 13):
                d = max(abs(x), abs(z))
                for yy in range(self.top(x, z) + 1, y):
                    self.set(x, yy, z, B("stone_bricks"))
                for yy in range(y, y + 6):
                    self.set(x, yy, z, 0)
                self.set(x, y, z, B("polished_andesite") if (x + z) % 2 else B("smooth_stone"))
                if d == 12:
                    self.set(x, y, z, B("stone_bricks"))
                    if (x + z) % 6 == 0:
                        self.set(x, y + 1, z, B("lantern"))
        # Steps down each side.
        for f, (dx, dz) in (("south", (0, 1)), ("north", (0, -1)), ("east", (1, 0)), ("west", (-1, 0))):
            for k in range(1, 8):
                for w in range(-2, 3):
                    x = dx * (12 + k) + (w if dx == 0 else 0)
                    z = dz * (12 + k) + (w if dz == 0 else 0)
                    yy = y - k
                    if yy <= self.top(x, z):
                        break
                    opp = {"south": "north", "north": "south", "east": "west", "west": "east"}[f]
                    self.set(x, yy, z, B(f"minecraft:stone_brick_stairs[facing={opp},half=bottom,shape=straight,waterlogged=false]"))
                    for down in range(self.top(x, z) + 1, yy):
                        self.set(x, down, z, B("stone_bricks"))
        # Fountain.
        for z in range(-3, 4):
            for x in range(-3, 4):
                if max(abs(x), abs(z)) == 3:
                    self.set(x, y + 1, z, B("stone_brick_slab"))
                else:
                    self.set(x, y, z, B("water"))
        self.set(0, y + 1, 0, B("stone_bricks"))
        self.set(0, y + 2, 0, B("sea_lantern"))
        # Benches and planters.
        for x in (-8, 8):
            for z in range(-3, 4):
                self.set(x, y + 1, z, B(f"minecraft:oak_stairs[facing={'east' if x < 0 else 'west'},half=bottom,shape=straight,waterlogged=false]"))
        for x, z in ((-9, -9), (9, -9), (-9, 9), (9, 9)):
            self.set(x, y + 1, z, B("moss_block"))
            self.set(x, y + 2, z, B("oak_leaves"))

    def house(self, cx, cz, w, d, h, wall, roof):
        base = max(self.top(cx + dx, cz + dz) for dx in range(w) for dz in range(d)) + 1
        for dz in range(d):
            for dx in range(w):
                x, z = cx + dx, cz + dz
                for yy in range(self.top(x, z) + 1, base):
                    self.set(x, yy, z, B("cobblestone"))
                self.set(x, base - 1, z, B("cobblestone"))
                edge = dx in (0, w - 1) or dz in (0, d - 1)
                corner = dx in (0, w - 1) and dz in (0, d - 1)
                for yy in range(base, base + h):
                    if corner:
                        self.set(x, yy, z, B("stripped_oak_log"))
                    elif edge:
                        window = yy == base + 1 and (dx + dz) % 3 == 1
                        self.set(x, yy, z, B("glass") if window else B(wall))
                    else:
                        self.set(x, yy, z, 0)
                self.set(x, base + h, z, B(roof))
        return base

    def houses(self):
        for (x, z, w, d) in ((-40, 20, 7, 6), (30, 35, 6, 8), (-25, -45, 8, 6), (45, -10, 6, 6), (5, 55, 7, 7)):
            self.house(x, z, w, d, 4, "oak_planks" if x < 0 else "spruce_planks", "oak_slab")
        # A hay farm and a wool tent for colour.
        for x in range(20, 26):
            for z in range(-30, -26):
                self.set(x, self.top(x, z) + 1, z, B("hay_block"))
        for x in range(-30, -26):
            for z in range(40, 44):
                self.set(x, self.top(x, z) + 1, z, B("cyan_wool"))

    def tower(self):
        """A ~90 block white tower at 200-228 degrees of the orbit, for the camera to pull in around."""
        cx, cz = -48, -28
        for z in range(cz - 4, cz + 4):
            for x in range(cx - 4, cx + 4):
                edge = x in (cx - 4, cx + 3) or z in (cz - 4, cz + 3)
                for y in range(self.top(x, z) + 1, 168):
                    if edge:
                        self.set(x, y, z, B("white_concrete") if y % 12 else B("light_gray_concrete"))
                    elif y % 24 == 0:
                        self.set(x, y, z, B("white_concrete"))

    def section(self, cx, sy, cz):
        """Encoded section: palette and indices, None when all air."""
        xi, zi = cx * 16 - self.x0, cz * 16 - self.x0
        y0 = sy * 16 - MIN_Y
        cube = self.blocks[zi:zi + 16, xi:xi + 16, y0:y0 + 16]  # [z][x][y]
        if not cube.any():
            return None
        ordered = np.transpose(cube, (2, 0, 1)).reshape(-1)  # (y * 16 + z) * 16 + x
        used, idx = np.unique(ordered, return_inverse=True)
        out = bytearray()
        out += struct.pack(">bh", sy, len(used))
        for u in used:
            s = STATES[u].encode()
            out += struct.pack(">H", len(s)) + s
        out += idx.astype(np.uint8).tobytes()
        biome = b"minecraft:forest" if self.forest[zi + 8, xi + 8] > 0.55 else b"minecraft:plains"
        out += struct.pack(">BH", 1, len(biome)) + biome + bytes(64)
        return bytes(out)

    def column(self, cx, cz, only=None):
        parts = []
        for sy in range(MIN_Y >> 4, MAX_Y >> 4):
            if only is not None and sy not in only:
                continue
            enc = self.section(cx, sy, cz)
            if enc is None:
                if only is not None:
                    parts.append(struct.pack(">bh", sy, 0))
                continue
            parts.append(enc)
        if not parts:
            return None
        body = struct.pack(">iih", cx, cz, len(parts)) + b"".join(parts)
        return zlib.compress(body, 1)


# ---------------------------------------------------------------------------------------
# Protocol


def frame(kind, payload=b""):
    return struct.pack(">IB", len(payload) + 1, kind) + payload


def utf(s):
    b = s.encode()
    return struct.pack(">H", len(b)) + b


class Link:
    def __init__(self, port, key, radius):
        self.lock = threading.Lock()
        self.sock = None
        self.port, self.key, self.radius = port, key, radius

    def connect(self):
        while True:
            try:
                s = socket.create_connection(("127.0.0.1", self.port))
                break
            except OSError:
                time.sleep(0.5)
        head = recv_exact(s, 5)
        (n,) = struct.unpack(">I", head[:4])
        body = recv_exact(s, n - 1)
        if head[4] != CHALLENGE:
            sys.exit("no challenge")
        nonce = body[4:36]
        pub = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        hello = struct.pack(">i", PROTOCOL) + pub + utf("minecraft:overworld") + struct.pack(">iiiiii", MIN_Y, MAX_Y, 0, 0, self.radius, 20)
        s.sendall(frame(HELLO, hello + self.key.sign(CONTEXT + nonce + hello)))
        ack = recv_exact(s, 5)
        if ack[4] != ACCEPT:
            sys.exit("refused: the engine's observerKey must be " + b64encode(pub).decode())
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def send(self, data):
        with self.lock:
            self.sock.sendall(data)


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("engine closed")
        buf += c
    return buf


# ---------------------------------------------------------------------------------------
# Scenario


class Player:
    NAMES = ["Steve_Builds", "alexandra", "fox_walker", "PixelMage", "dirt_enjoyer", "notch_fan", "Riga2026", "creeperkeeper"]

    def __init__(self, i, world, rng):
        self.id = 1000 + i
        self.name = self.NAMES[i % len(self.NAMES)]
        self.world = world
        self.rng = rng
        self.phase = rng.random() * math.tau
        self.radius = 5 + rng.random() * 25
        self.speed = 0.8 + rng.random() * 2.5  # m/s
        self.wander = rng.random() * 0.6

    def at(self, t):
        a = self.phase + t * self.speed / self.radius
        r = self.radius * (1 + self.wander * math.sin(t * 0.21 + self.phase))
        x, z = math.cos(a) * r, math.sin(a) * r
        y = self.ground(x, z)
        dx, dz = -math.sin(a), math.cos(a)
        yaw = math.degrees(math.atan2(-dx, dz))
        return x, y, z, yaw

    def ground(self, x, z):
        bx, bz = math.floor(x), math.floor(z)
        for y in range(160, MIN_Y, -1):
            b = self.world.get(bx, y, bz)
            if b and STATES[b] not in ("minecraft:short_grass", "minecraft:tall_grass[half=lower]", "minecraft:tall_grass[half=upper]",
                                       "minecraft:poppy", "minecraft:dandelion", "minecraft:cornflower", "minecraft:oxeye_daisy",
                                       "minecraft:fern", "minecraft:water[level=0]", "minecraft:lily_pad"):
                return y + 1.0
        return 80.0

    def info(self):
        gear = ["minecraft:diamond_sword", "", "", "minecraft:iron_chestplate" if self.id % 2 else "", "", ""]
        skin = f"uuid:00000000-0000-0000-0000-{self.id:012d}".encode()
        out = struct.pack(">i", self.id) + utf("minecraft:player") + utf(self.name) + struct.pack(">i", len(skin)) + skin
        out += struct.pack(">B", len(gear)) + b"".join(utf(g) for g in gear)
        return out


def run(args):
    key = serialization.load_der_private_key(open(args.key, "rb").read(), password=None)
    t0 = time.time()
    world = World(args.seed, args.radius)
    log(f"world generated in {time.time() - t0:.1f}s")
    t0 = time.time()
    columns = []
    for cz in range(world.c0, world.c1):
        for cx in range(world.c0, world.c1):
            if math.hypot(cx * 16 + 8, cz * 16 + 8) > args.radius + 12:
                continue
            c = world.column(cx, cz)
            if c:
                columns.append(((cx * 16 + 8) ** 2 + (cz * 16 + 8) ** 2, c))
    columns.sort(key=lambda c: c[0])
    log(f"{len(columns)} columns encoded in {time.time() - t0:.1f}s, {sum(len(c[1]) for c in columns) / 1e6:.1f} MB")
    while True:
        link = Link(args.port, key, args.radius)
        try:
            link.connect()
            log("connected")
            serve(link, world, columns, args)
        except (ConnectionError, OSError) as e:
            log(f"link lost ({e}), reconnecting")
            time.sleep(1)


def serve(link, world, columns, args):
    for _, c in columns:
        link.send(frame(SECTIONS, c))
    link.send(frame(SNAPSHOT_DONE))
    start = time.time()
    log("snapshot sent")
    rng = np.random.default_rng(args.seed + 1)
    players = [Player(i, world, rng) for i in range(args.players)]
    joined = False
    tick = 0
    next_ping = time.time() + 5
    edit_at = start + args.edits_at
    placed = []
    edit_rng = np.random.default_rng(args.seed + 2)
    while True:
        now = time.time()
        t = now - start
        tick += 1
        present = players and t >= args.players_at and (args.players_for <= 0 or t < args.players_at + args.players_for)
        if present and not joined:
            for p in players:
                link.send(frame(ENTITY_INFO, p.info()))
            joined = True
            log(f"{len(players)} players joined")
        if joined and not present:
            joined = False
            log("players left")
        if not present:
            # As the observer plugin does: a frame every tick, empty when nobody is about.
            link.send(frame(ENTITIES, struct.pack(">qqh", tick, tick, 0)))
            next_ping = now + 5
        if present:
            body = bytearray(struct.pack(">qqh", tick, tick, len(players)))
            for p in players:
                x, y, z, yaw = p.at(t)
                flags = 64 | (4 if int(t * 2 + p.id) % 7 == 0 else 0)
                body += struct.pack(">idddfffbb", p.id, x, y, z, yaw, 0.0, yaw, 0, flags)
            link.send(frame(ENTITIES, bytes(body)))
            next_ping = now + 5
        if args.edits_per_min > 0 and now >= edit_at:
            edit_at = now + 60.0 / args.edits_per_min
            edit(link, world, edit_rng, placed)
            next_ping = now + 5
        if now >= next_ping:
            link.send(frame(PING))
            next_ping = now + 5
        time.sleep(max(0.0, start + tick * 0.05 - time.time()))


def edit(link, world, rng, placed):
    """A builder near spawn: places a block, or takes one it placed earlier away."""
    if placed and (len(placed) > 30 or rng.random() < 0.4):
        x, y, z, old = placed.pop(int(rng.random() * len(placed)))
        world.set(x, y, z, old)
    else:
        x, z = int(rng.integers(-20, 21)), int(rng.integers(-20, 21))
        y = world.top(x, z) + 1
        while world.get(x, y, z) not in (0,) and y < 150:
            y += 1
        placed.append((x, y, z, world.get(x, y, z)))
        world.set(x, y, z, B(["gold_block", "glowstone", "bricks", "redstone_block", "cyan_wool"][int(rng.integers(5))]))
    cx, cz, sy = x >> 4, z >> 4, y >> 4
    # The plugin sends the sections within a block of the edit.
    for ncx in {(x - 1) >> 4, cx, (x + 1) >> 4}:
        for ncz in {(z - 1) >> 4, cz, (z + 1) >> 4}:
            only = {(y - 1) >> 4, sy, (y + 1) >> 4}
            c = world.column(ncx, ncz, only)
            if c:
                link.send(frame(SECTIONS, c))


def log(msg):
    print(f"fakeserver: {msg}", file=sys.stderr, flush=True)


def keygen(path):
    key = Ed25519PrivateKey.generate()
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(der)
    print(b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "key":
        keygen(sys.argv[2])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--key", required=True)
    ap.add_argument("--port", type=int, default=25590)
    ap.add_argument("--players", type=int, default=0)
    ap.add_argument("--players-at", type=float, default=0)
    ap.add_argument("--players-for", type=float, default=0)
    ap.add_argument("--edits-per-min", type=float, default=0)
    ap.add_argument("--edits-at", type=float, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--radius", type=int, default=224)
    run(ap.parse_args())
