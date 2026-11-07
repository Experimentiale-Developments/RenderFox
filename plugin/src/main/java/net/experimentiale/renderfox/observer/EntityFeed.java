package net.experimentiale.renderfox.observer;

import com.destroystokyo.paper.profile.ProfileProperty;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Mannequin;
import org.bukkit.entity.Player;
import org.bukkit.inventory.EntityEquipment;
import org.bukkit.inventory.ItemStack;
import org.bukkit.util.BoundingBox;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Sends every player and mannequin shown in the area, a full list per frame so the
 * engine can forget anything that is missing. Names, skins and equipment go separately
 * and only when they change, because they are large next to a position.
 *
 * Those are the only entities the engine draws. Players come from the world's player
 * list every frame, which costs nothing; mannequins need a search of the area, so that
 * is redone once a second and the ones found are followed in between.
 */
final class EntityFeed {
    static final int FLAG_SNEAKING = 1;
    static final int FLAG_SPRINTING = 2;
    static final int FLAG_SWING_MAIN = 4;
    static final int FLAG_SWING_OFF = 8;
    static final int FLAG_HURT = 16;
    static final int FLAG_BURNING = 32;
    static final int FLAG_ON_GROUND = 64;

    private final World world;
    private final Link link;
    private final BoundingBox box;
    private final Map<Integer, String> sentInfo = new HashMap<>();
    private final Map<Integer, Integer> events = new HashMap<>();
    private List<Mannequin> mannequins = List.of();
    private int untilScan;
    private long frame;

    EntityFeed(World world, Link link, int centerX, int centerZ, int radius) {
        this.world = world;
        this.link = link;
        this.box = new BoundingBox(centerX - radius, world.getMinHeight(), centerZ - radius,
                centerX + radius, world.getMaxHeight(), centerZ + radius);
    }

    void resync() {
        sentInfo.clear();
        events.clear();
        untilScan = 0;
    }

    void swing(Entity entity, boolean offHand) {
        events.merge(entity.getEntityId(), offHand ? FLAG_SWING_OFF : FLAG_SWING_MAIN, (a, b) -> a | b);
    }

    void hurt(Entity entity) {
        events.merge(entity.getEntityId(), FLAG_HURT, (a, b) -> a | b);
    }

    void tick() {
        if (!link.connected()) return;
        if (--untilScan <= 0) {
            untilScan = 20;
            List<Mannequin> found = new ArrayList<>();
            for (Entity e : world.getNearbyEntities(box, e -> e instanceof Mannequin)) found.add((Mannequin) e);
            mannequins = found;
        }
        List<Entity> visible = new ArrayList<>();
        for (Player p : world.getPlayers()) {
            if (inside(p) && shown(p)) visible.add(p);
        }
        for (Mannequin m : mannequins) {
            if (m.isValid() && m.getWorld() == world && inside(m) && shown(m)) visible.add(m);
        }
        Set<Integer> present = new HashSet<>();
        for (Entity e : visible) {
            present.add(e.getEntityId());
            String info = info(e);
            if (!info.equals(sentInfo.get(e.getEntityId()))) {
                sentInfo.put(e.getEntityId(), info);
                sendInfo(e);
            }
        }
        sentInfo.keySet().retainAll(present);
        long f = frame++;
        long worldTick = world.getGameTime();
        link.sendVolatile(Wire.message(Wire.ENTITIES, out -> {
            out.writeLong(f);
            out.writeLong(worldTick);
            out.writeShort(visible.size());
            for (Entity e : visible) {
                Location at = e.getLocation();
                out.writeInt(e.getEntityId());
                out.writeDouble(at.getX());
                out.writeDouble(at.getY());
                out.writeDouble(at.getZ());
                out.writeFloat(at.getYaw());
                out.writeFloat(at.getPitch());
                out.writeFloat(e instanceof LivingEntity living ? living.getBodyYaw() : at.getYaw());
                out.writeByte(e.getPose().ordinal());
                out.writeByte(flags(e));
            }
        }));
        events.clear();
    }

    private int flags(Entity e) {
        int flags = events.getOrDefault(e.getEntityId(), 0);
        if (e.isSneaking()) flags |= FLAG_SNEAKING;
        if (e instanceof Player p && p.isSprinting()) flags |= FLAG_SPRINTING;
        if (e.getFireTicks() > 0) flags |= FLAG_BURNING;
        if (e.isOnGround()) flags |= FLAG_ON_GROUND;
        return flags;
    }

    private boolean inside(Entity e) {
        Location at = e.getLocation();
        return box.contains(at.getX(), at.getY(), at.getZ());
    }

    /**
     * Spectators and vanished staff must never show up on a public stream. Vanish
     * plugins conventionally mark players with "vanished" metadata; anything hidden by
     * default or invisible is treated the same way.
     */
    private static boolean shown(Entity e) {
        if (!e.isVisibleByDefault() || e.isInvisible()) return false;
        if (e instanceof Player p) {
            if (p.getGameMode() == GameMode.SPECTATOR) return false;
            if (p.hasMetadata("vanished") && p.getMetadata("vanished").stream().anyMatch(m -> m.asBoolean()))
                return false;
        }
        return true;
    }

    private static String info(Entity e) {
        StringBuilder s = new StringBuilder(e.getType().getKey().toString());
        s.append('|').append(e instanceof Player p ? p.getName() : "");
        s.append('|').append(skin(e));
        for (String item : equipment(e)) s.append('|').append(item);
        return s.toString();
    }

    private void sendInfo(Entity e) {
        String type = e.getType().getKey().toString();
        String name = e instanceof Player p ? p.getName() : "";
        String skin = skin(e);
        List<String> gear = equipment(e);
        int id = e.getEntityId();
        link.sendInfo(Wire.message(Wire.ENTITY_INFO, out -> {
            out.writeInt(id);
            out.writeUTF(type);
            out.writeUTF(name);
            writeLongUTF(out, skin);
            out.writeByte(gear.size());
            for (String item : gear) out.writeUTF(item);
        }));
    }

    /** The base64 "textures" property, or empty for the default skin. */
    private static String skin(Entity e) {
        Iterable<ProfileProperty> properties;
        if (e instanceof Player p) {
            properties = p.getPlayerProfile().getProperties();
        } else if (e instanceof Mannequin m) {
            properties = m.getProfile().properties();
        } else {
            return "";
        }
        for (ProfileProperty property : properties) {
            if (property.getName().equals("textures")) return property.getValue();
        }
        if (e instanceof Player p) return "uuid:" + p.getUniqueId();
        if (e instanceof Mannequin m && m.getProfile().uuid() != null) return "uuid:" + m.getProfile().uuid();
        if (e instanceof Mannequin) return "uuid:" + e.getUniqueId();
        return "";
    }

    /** Main hand, off hand, head, chest, legs, feet; item keys, empty when bare. */
    private static List<String> equipment(Entity e) {
        List<String> gear = new ArrayList<>(6);
        if (!(e instanceof LivingEntity living)) return gear;
        EntityEquipment eq = living.getEquipment();
        if (eq == null) return gear;
        gear.add(key(eq.getItemInMainHand()));
        gear.add(key(eq.getItemInOffHand()));
        gear.add(key(eq.getHelmet()));
        gear.add(key(eq.getChestplate()));
        gear.add(key(eq.getLeggings()));
        gear.add(key(eq.getBoots()));
        return gear;
    }

    private static String key(ItemStack stack) {
        return stack == null || stack.getType().isAir() ? "" : stack.getType().getKey().toString();
    }

    /** Skin properties can exceed writeUTF's 64 KB limit only in theory; length-prefix them anyway. */
    private static void writeLongUTF(DataOutputStream out, String s) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }
}
