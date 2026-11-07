package net.experimentiale.renderfox.testactors;

import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.HeightMap;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Mannequin;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;

import java.util.ArrayList;
import java.util.List;

/**
 * Test servers only. Walkers circle the spawn so the stream shows moving players; a
 * builder raises and removes a small arch beside the platform so terrain updates can
 * be watched end to end. Everything it spawns is removed on stop.
 */
public class TestActors extends JavaPlugin {
    private final List<Mannequin> walkers = new ArrayList<>();
    private final List<Block> placed = new ArrayList<>();
    private BukkitTask task;
    private long tick;

    @Override
    public void onEnable() {
        Bukkit.getScheduler().runTaskLater(this, this::start, 100);
    }

    @Override
    public void onDisable() {
        stop();
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 1 && args[0].equalsIgnoreCase("stop")) stop();
        else start();
        sender.sendMessage(Component.text("Test actors " + (task == null ? "stopped" : "running")));
        return true;
    }

    private void start() {
        if (task != null) return;
        World world = Bukkit.getWorlds().getFirst();
        String[][] gear = {
                {"DIAMOND_HELMET", "DIAMOND_CHESTPLATE", "DIAMOND_LEGGINGS", "DIAMOND_BOOTS"},
                {"", "IRON_CHESTPLATE", "", "IRON_BOOTS"},
                {"GOLDEN_HELMET", "", "LEATHER_LEGGINGS", ""},
                {"", "", "", ""},
        };
        for (int i = 0; i < gear.length; i++) {
            Location at = surface(world, 6 + i * 3, 0);
            Mannequin m = world.spawn(at, Mannequin.class, spawned -> {
                spawned.setPersistent(false);
                spawned.setImmovable(true);
            });
            String[] g = gear[i];
            if (!g[0].isEmpty()) m.getEquipment().setHelmet(new ItemStack(Material.valueOf(g[0])));
            if (!g[1].isEmpty()) m.getEquipment().setChestplate(new ItemStack(Material.valueOf(g[1])));
            if (!g[2].isEmpty()) m.getEquipment().setLeggings(new ItemStack(Material.valueOf(g[2])));
            if (!g[3].isEmpty()) m.getEquipment().setBoots(new ItemStack(Material.valueOf(g[3])));
            walkers.add(m);
        }
        task = Bukkit.getScheduler().runTaskTimer(this, this::step, 1, 1);
        getLogger().info("Test actors running around 0, 0");
    }

    private void stop() {
        if (task != null) task.cancel();
        task = null;
        walkers.forEach(Mannequin::remove);
        walkers.clear();
        for (Block b : placed) b.setType(Material.AIR);
        placed.clear();
    }

    private Location surface(World world, double x, double z) {
        // Stand on the block's real top so slabs and stairs do not leave them floating.
        Block top = world.getHighestBlockAt((int) Math.floor(x), (int) Math.floor(z), HeightMap.MOTION_BLOCKING_NO_LEAVES);
        return new Location(world, x, top.getBoundingBox().getMaxY(), z);
    }

    private void step() {
        tick++;
        World world = walkers.isEmpty() ? null : walkers.getFirst().getWorld();
        for (int i = 0; i < walkers.size(); i++) {
            Mannequin m = walkers.get(i);
            // Walking pace, about 4.3 blocks a second like a player, around rings of
            // different sizes; every other one goes the other way.
            double radius = 6 + i * 3;
            double speed = 4.3 / 20 / radius * (i % 2 == 0 ? 1 : -1);
            double a = tick * speed + i;
            Location at = surface(world, Math.cos(a) * radius, Math.sin(a) * radius);
            double dx = -Math.sin(a) * Math.signum(speed), dz = Math.cos(a) * Math.signum(speed);
            float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
            at.setYaw(yaw);
            at.setPitch((float) (Math.sin(tick * 0.05 + i) * 15));
            m.teleport(at);
            m.setBodyYaw(yaw);
        }
        // Builder: every two seconds add one block of a small arch beside the platform,
        // and once it stands, take it down again.
        if (tick % 40 == 0 && world != null) {
            int[][] arch = {{9, 0}, {9, 1}, {9, 2}, {9, 3}, {10, 3}, {11, 3}, {12, 3}, {12, 2}, {12, 1}, {12, 0}};
            int phase = (int) (tick / 40 % (arch.length * 2));
            Location base = surface(world, 9.5, -8.5);
            if (phase < arch.length) {
                Block b = world.getBlockAt(base.getBlockX() + arch[phase][0] - 9, base.getBlockY() + arch[phase][1], base.getBlockZ());
                if (b.getType().isAir()) {
                    b.setType(phase % 2 == 0 ? Material.WHITE_WOOL : Material.CYAN_STAINED_GLASS);
                    placed.add(b);
                }
            } else if (!placed.isEmpty()) {
                placed.removeLast().setType(Material.AIR);
            }
        }
    }
}
