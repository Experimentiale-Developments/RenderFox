package net.experimentiale.renderfox.observer;

import org.bukkit.Chunk;
import org.bukkit.ChunkSnapshot;
import org.bukkit.World;
import org.bukkit.block.Biome;
import org.bukkit.block.data.BlockData;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Keeps the engine's copy of the terrain around spawn identical to the server's.
 *
 * Three sources feed the same path (snapshot on the main thread, encode and diff on
 * the encoder thread): the full area once per connection, chunks named by block
 * events, and a slow round-robin sweep. Events give immediacy; the sweep is what
 * catches the changes no event announces, such as WorldEdit, so correctness never
 * depends on the event list being complete.
 *
 * An event names the sections around its block, and only those are encoded again: a
 * placed block reaches the engine at the end of the tick it was placed in, as one small
 * section rather than a whole column.
 */
final class ChunkFeed {
    private static final int MAX_LOADS_IN_FLIGHT = 8;
    private static final long ALL_SECTIONS = -1L;

    private final World world;
    private final Link link;
    private final int chunksPerTick;
    private final int sweepEveryTicks;
    private final long[] area;
    private final Set<Long> areaSet = new HashSet<>();
    private final int minSection;
    private final int maxSection;

    private final ArrayDeque<Long> initial = new ArrayDeque<>();
    /** Chunk key to a bit per section (from the lowest), or ALL_SECTIONS. */
    private final LinkedHashMap<Long, Long> dirty = new LinkedHashMap<>();
    private int loadsInFlight;
    private int sweepCursor;
    private int innerCursor;
    private int innerCount;
    private int coreCursor;
    private int coreCount;
    private long ticks;
    private boolean snapshotting;
    private int generation;

    /** Encoder thread only. Section key to hash of what the engine last received. */
    private final Map<Long, Long> sentHashes = new HashMap<>();
    private final ExecutorService encoder = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().name("RenderFox-encoder").daemon().factory());

    ChunkFeed(World world, Link link, int centerX, int centerZ, int radius, int chunksPerTick, int sweepPerSecond) {
        this.world = world;
        this.link = link;
        this.chunksPerTick = chunksPerTick;
        this.sweepEveryTicks = Math.max(1, 20 / Math.max(1, sweepPerSecond));
        this.minSection = world.getMinHeight() >> 4;
        this.maxSection = (world.getMaxHeight() - 1) >> 4;
        int r = (radius + 15) >> 4;
        int ccx = centerX >> 4;
        int ccz = centerZ >> 4;
        List<Long> keys = new ArrayList<>();
        for (int dz = -r; dz <= r; dz++) {
            for (int dx = -r; dx <= r; dx++) {
                if (dx * dx + dz * dz <= (r + 1) * (r + 1)) keys.add(key(ccx + dx, ccz + dz));
            }
        }
        // Nearest first, so the part of the world the camera actually looks at arrives first.
        keys.sort((a, b) -> Integer.compare(dist(a, ccx, ccz), dist(b, ccx, ccz)));
        this.area = keys.stream().mapToLong(Long::longValue).toArray();
        areaSet.addAll(keys);
        // The camera frames the middle of the area; those chunks get swept far more often,
        // and the ones it circles most of all.
        for (long k : area) {
            if (dist(k, ccx, ccz) <= 9 * 9) innerCount++;
            if (dist(k, ccx, ccz) <= 4 * 4) coreCount++;
        }
    }

    int areaChunks() {
        return area.length;
    }

    boolean snapshotting() {
        return snapshotting;
    }

    int pending() {
        return initial.size() + dirty.size() + loadsInFlight;
    }

    /** Main thread. A fresh connection knows nothing, so everything goes again. */
    void resync() {
        generation++;
        initial.clear();
        dirty.clear();
        loadsInFlight = 0;
        for (long k : area) initial.add(k);
        snapshotting = true;
        encoder.execute(sentHashes::clear);
    }

    /**
     * Main thread. The block's neighbours count too: placing or breaking one reshapes the
     * fences, stairs, panes and walls next to it without an event of their own.
     */
    void markDirty(int blockX, int blockY, int blockZ) {
        long sections = sections((blockY - 1) >> 4, (blockY + 1) >> 4);
        for (int cz = (blockZ - 1) >> 4; cz <= (blockZ + 1) >> 4; cz++) {
            for (int cx = (blockX - 1) >> 4; cx <= (blockX + 1) >> 4; cx++) {
                long k = key(cx, cz);
                if (areaSet.contains(k)) dirty.merge(k, sections, (a, b) -> a | b);
            }
        }
    }

    private long sections(int from, int to) {
        if (maxSection - minSection >= 64) return ALL_SECTIONS;
        long bits = 0;
        for (int sy = Math.max(from, minSection); sy <= Math.min(to, maxSection); sy++) bits |= 1L << (sy - minSection);
        return bits;
    }

    /** Main thread, at the end of every tick, so edits made during it go out at once. */
    void tick() {
        ticks++;
        if (!link.connected()) return;
        int budget = chunksPerTick;
        while (budget > 0 && !dirty.isEmpty()) {
            Map.Entry<Long, Long> first = dirty.pollFirstEntry();
            long k = first.getKey();
            if (world.isChunkLoaded(cx(k), cz(k))) {
                capture(world.getChunkAt(cx(k), cz(k)), first.getValue());
                budget--;
            }
        }
        while (budget > 0 && !initial.isEmpty() && loadsInFlight < MAX_LOADS_IN_FLIGHT) {
            long k = initial.removeFirst();
            budget--;
            if (world.isChunkLoaded(cx(k), cz(k))) {
                capture(world.getChunkAt(cx(k), cz(k)), ALL_SECTIONS);
                continue;
            }
            loadsInFlight++;
            int gen = generation;
            world.getChunkAtAsync(cx(k), cz(k), false, chunk -> {
                if (gen != generation) return;
                loadsInFlight--;
                if (chunk != null) capture(chunk, ALL_SECTIONS);
            });
        }
        if (snapshotting && initial.isEmpty() && loadsInFlight == 0) {
            snapshotting = false;
            encoder.execute(() -> link.sendTerrain(Wire.message(Wire.SNAPSHOT_DONE, out -> {})));
        }
        if (!snapshotting && ticks % sweepEveryTicks == 0) {
            // Of every six chunks swept (area is sorted nearest first), three come from the
            // core the camera circles, two from the inner ring, one from the whole area:
            // at six a second, the core is gone over every twenty seconds or so.
            long k = switch ((int) (ticks / sweepEveryTicks % 6)) {
                case 0, 2, 4 -> area[coreCursor++ % coreCount];
                case 1, 3 -> area[innerCursor++ % innerCount];
                default -> area[sweepCursor++ % area.length];
            };
            if (world.isChunkLoaded(cx(k), cz(k))) capture(world.getChunkAt(cx(k), cz(k)), ALL_SECTIONS);
        }
    }

    void shutdown() {
        encoder.shutdownNow();
    }

    private void capture(Chunk chunk, long sections) {
        ChunkSnapshot snapshot = chunk.getChunkSnapshot(false, true, false);
        encoder.execute(() -> encode(snapshot, sections));
    }

    private void encode(ChunkSnapshot snapshot, long sections) {
        List<byte[]> changed = new ArrayList<>();
        for (int sy = minSection; sy <= maxSection; sy++) {
            if (sections != ALL_SECTIONS && (sections & 1L << (sy - minSection)) == 0) continue;
            byte[] section = snapshot.isSectionEmpty(sy - minSection) ? null : section(snapshot, sy);
            long hash = section == null ? 0 : fnv(section);
            long k = sectionKey(snapshot.getX(), sy, snapshot.getZ());
            Long previous = sentHashes.get(k);
            if (previous == null ? hash == 0 : previous == hash) continue;
            sentHashes.put(k, hash);
            changed.add(section == null ? emptySection(sy) : section);
        }
        if (changed.isEmpty()) return;
        int x = snapshot.getX();
        int z = snapshot.getZ();
        link.sendTerrain(Wire.deflated(Wire.SECTIONS, out -> {
            out.writeInt(x);
            out.writeInt(z);
            out.writeShort(changed.size());
            for (byte[] s : changed) out.write(s);
        }));
    }

    private static byte[] emptySection(int sy) {
        return new byte[]{(byte) sy, 0, 0};
    }

    /**
     * Layout: {@code byte sy, short paletteSize, UTF[paletteSize], indices[4096]
     * (bytes, or shorts past 256 entries), byte biomeCount, UTF[biomeCount], byte[64]}.
     * Index order is {@code (y * 16 + z) * 16 + x}; biomes are the 4x4x4 cells in the
     * same order.
     */
    private static byte[] section(ChunkSnapshot snapshot, int sy) {
        Map<BlockData, Integer> palette = new HashMap<>();
        List<BlockData> order = new ArrayList<>();
        short[] indices = new short[4096];
        int baseY = sy << 4;
        int i = 0;
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                for (int x = 0; x < 16; x++) {
                    BlockData data = snapshot.getBlockData(x, baseY + y, z);
                    Integer id = palette.get(data);
                    if (id == null) {
                        id = order.size();
                        palette.put(data, id);
                        order.add(data);
                    }
                    indices[i++] = id.shortValue();
                }
            }
        }
        Map<Biome, Integer> biomes = new HashMap<>();
        List<Biome> biomeOrder = new ArrayList<>();
        byte[] cells = new byte[64];
        int c = 0;
        for (int y = 0; y < 4; y++) {
            for (int z = 0; z < 4; z++) {
                for (int x = 0; x < 4; x++) {
                    Biome biome = snapshot.getBiome(x * 4 + 2, baseY + y * 4 + 2, z * 4 + 2);
                    Integer id = biomes.get(biome);
                    if (id == null) {
                        id = biomeOrder.size();
                        biomes.put(biome, id);
                        biomeOrder.add(biome);
                    }
                    cells[c++] = id.byteValue();
                }
            }
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(4096 + 64 + order.size() * 32);
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            out.writeByte(sy);
            out.writeShort(order.size());
            for (BlockData data : order) out.writeUTF(data.getAsString());
            if (order.size() <= 256) {
                for (short s : indices) out.writeByte(s);
            } else {
                for (short s : indices) out.writeShort(s);
            }
            out.writeByte(biomeOrder.size());
            for (Biome biome : biomeOrder) out.writeUTF(biome.getKey().toString());
            out.write(cells);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bytes.toByteArray();
    }

    private static long fnv(byte[] bytes) {
        long h = 0xcbf29ce484222325L;
        for (byte b : bytes) {
            h ^= b & 0xff;
            h *= 0x100000001b3L;
        }
        return h == 0 ? 1 : h;
    }

    private static long key(int cx, int cz) {
        return ((long) cx << 32) | (cz & 0xffffffffL);
    }

    private static int cx(long k) {
        return (int) (k >> 32);
    }

    private static int cz(long k) {
        return (int) k;
    }

    private static long sectionKey(int cx, int sy, int cz) {
        return ((long) cx & 0x3ffffff) << 38 | ((long) cz & 0x3ffffff) << 12 | (sy & 0xfff);
    }

    private static int dist(long k, int ccx, int ccz) {
        int dx = cx(k) - ccx;
        int dz = cz(k) - ccz;
        return dx * dx + dz * dz;
    }
}
