package net.experimentiale.renderfox.observer;

import com.destroystokyo.paper.event.server.ServerTickEndEvent;
import io.papermc.paper.event.player.PlayerArmSwingEvent;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.BlockState;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.event.Event;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockBurnEvent;
import org.bukkit.event.block.BlockExplodeEvent;
import org.bukkit.event.block.BlockFadeEvent;
import org.bukkit.event.block.BlockFertilizeEvent;
import org.bukkit.event.block.BlockFormEvent;
import org.bukkit.event.block.BlockFromToEvent;
import org.bukkit.event.block.BlockGrowEvent;
import org.bukkit.event.block.BlockMultiPlaceEvent;
import org.bukkit.event.block.BlockPistonEvent;
import org.bukkit.event.block.BlockPistonExtendEvent;
import org.bukkit.event.block.BlockPistonRetractEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.block.BlockRedstoneEvent;
import org.bukkit.event.block.BlockSpreadEvent;
import org.bukkit.event.block.FluidLevelChangeEvent;
import org.bukkit.event.block.LeavesDecayEvent;
import org.bukkit.event.entity.EntityChangeBlockEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityExplodeEvent;
import org.bukkit.event.player.PlayerBucketEmptyEvent;
import org.bukkit.event.player.PlayerBucketFillEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.world.StructureGrowEvent;
import org.bukkit.inventory.EquipmentSlot;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;

/**
 * Read-only observer for a Render Fox live stream. It watches an area of one world and
 * mirrors it to the engine; it never modifies the world, never spawns anything and adds
 * nothing to the player list.
 */
public final class RenderFoxPlugin extends JavaPlugin implements Listener {
    static final int DEFAULT_PORT = 25590;
    private static final String HELP = "/renderfox [status | pair <host[:port]> <code> | unpair | key | reload]";

    private Identity identity;
    private Link link;
    private ChunkFeed chunks;
    private EntityFeed entities;
    private World world;
    private BukkitTask watchdogTask;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        try {
            identity = Identity.loadOrCreate(getDataFolder().toPath(), getLogger());
        } catch (GeneralSecurityException | IOException | IllegalArgumentException e) {
            getLogger().severe("Cannot load this server's Render Fox key (" + e.getMessage() + "); the observer stays idle. "
                    + "Delete " + Identity.PRIVATE_FILE + " and " + Identity.PUBLIC_FILE + " to make a new pair, then pair again.");
        }
        getServer().getPluginManager().registerEvents(this, this);
        if (identity != null) start(getConfig().getString("engine", "").trim());
    }

    @Override
    public void onDisable() {
        stop();
    }

    private void start(String engine) {
        start(engine, null);
    }

    /**
     * Main thread. An empty address leaves the observer idle until it is paired. A pairing
     * is handed to the link before it starts, so its first connection already pairs.
     */
    private void start(String engine, Consumer<Link> beforeStart) {
        if (engine.isEmpty()) {
            getLogger().info("Not paired with a Render Fox engine yet: run /renderfox pair <host[:port]> <code> with the code"
                    + " `renderfox pair` shows on the render box. This server's key is " + identity.publicKeyText());
            return;
        }
        String worldName = getConfig().getString("world", "world");
        world = getServer().getWorld(worldName);
        if (world == null) {
            getLogger().severe("World '" + worldName + "' not found; the Render Fox observer stays idle. Set world in config.yml.");
            return;
        }
        Address address;
        try {
            address = Address.parse(engine);
        } catch (IllegalArgumentException e) {
            getLogger().severe("engine in config.yml: " + e.getMessage() + "; the Render Fox observer stays idle");
            return;
        }
        int cx = getConfig().getInt("center.x", 0);
        int cz = getConfig().getInt("center.z", 0);
        int radius = Math.clamp(getConfig().getInt("radius", 224), 16, 512);

        byte[] body = Wire.raw(out -> {
            out.writeInt(Wire.PROTOCOL);
            out.write(identity.publicKey());
            out.writeUTF(world.getKey().toString());
            out.writeInt(world.getMinHeight());
            out.writeInt(world.getMaxHeight());
            out.writeInt(cx);
            out.writeInt(cz);
            out.writeInt(radius);
            out.writeInt(20);
        });
        Link.Hello hello = nonce -> {
            ByteArrayOutputStream signed = new ByteArrayOutputStream();
            signed.writeBytes(Wire.HELLO_CONTEXT);
            signed.writeBytes(nonce);
            signed.writeBytes(body);
            ByteArrayOutputStream payload = new ByteArrayOutputStream();
            payload.writeBytes(body);
            payload.writeBytes(identity.sign(signed.toByteArray()));
            return Wire.frame(Wire.HELLO, payload.toByteArray());
        };

        link = new Link(address.host(), address.port(), hello, identity,
                () -> Bukkit.getScheduler().runTask(this, this::resync), getLogger());
        chunks = new ChunkFeed(world, link, cx, cz, radius, Math.max(1, getConfig().getInt("chunks-per-tick", 6)),
                Math.max(1, getConfig().getInt("sweep-chunks-per-second", 6)));
        entities = new EntityFeed(world, link, cx, cz, radius);
        watchdogTask = Bukkit.getScheduler().runTaskTimer(this, link::watchdog, 100, 100);
        if (beforeStart != null) beforeStart.accept(link);
        link.start();
        getLogger().info("Watching " + chunks.areaChunks() + " chunks around " + cx + ", " + cz + " in " + world.getName()
                + " for the engine at " + address + ". This server's key is " + identity.publicKeyText());
    }

    private void stop() {
        if (watchdogTask != null) {
            watchdogTask.cancel();
            watchdogTask = null;
        }
        if (link != null) {
            link.close();
            link = null;
        }
        if (chunks != null) {
            chunks.shutdown();
            chunks = null;
        }
        entities = null;
        world = null;
    }

    private void resync() {
        if (chunks != null) chunks.resync();
        if (entities != null) entities.resync();
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        String sub = args.length == 0 ? "status" : args[0].toLowerCase(Locale.ROOT);
        if (identity == null && !sub.equals("reload")) {
            sender.sendMessage(Component.text("Render Fox has no key, see the console.", NamedTextColor.RED));
            return true;
        }
        switch (sub) {
            case "status" -> status(sender);
            case "pair" -> {
                if (args.length != 3) {
                    sender.sendMessage(Component.text("Usage: /renderfox pair <host[:port]> <code>. Run `renderfox pair` on the"
                            + " render box for the code.", NamedTextColor.YELLOW));
                } else {
                    pair(sender, args[1], args[2]);
                }
            }
            case "unpair" -> {
                stop();
                getConfig().set("engine", "");
                saveConfig();
                sender.sendMessage(Component.text("This server no longer feeds a Render Fox engine. To pair the engine with"
                        + " another server, also run `renderfox unpair` on the render box.", NamedTextColor.GREEN));
            }
            case "key" -> sender.sendMessage(Component.text("This server's Render Fox key: " + identity.publicKeyText()));
            case "reload" -> {
                stop();
                reloadConfig();
                if (identity != null) start(getConfig().getString("engine", "").trim());
                sender.sendMessage(Component.text("Render Fox config reloaded.", NamedTextColor.GREEN));
            }
            default -> sender.sendMessage(Component.text(HELP, NamedTextColor.YELLOW));
        }
        return true;
    }

    private void status(CommandSender sender) {
        if (link == null || chunks == null) {
            sender.sendMessage(Component.text("Render Fox: not feeding an engine. Pair with /renderfox pair <host[:port]> <code>.",
                    NamedTextColor.YELLOW));
        } else {
            sender.sendMessage(Component.text("Render Fox: " + (link.pairing() ? "pairing with " : link.connected() ? "connected to "
                    : "not connected to ") + link.address() + ", " + chunks.areaChunks() + " chunks watched, " + chunks.pending()
                    + " pending" + (chunks.snapshotting() ? " (initial snapshot)" : "") + ", backlog " + link.backlog() / 1024
                    + " KiB, sent " + link.sentBytes() / (1024 * 1024) + " MiB"));
        }
        sender.sendMessage(Component.text("This server's key: " + identity.publicKeyText(), NamedTextColor.GRAY));
    }

    /**
     * Connects to the engine with a pairing request. The address is saved only once the
     * engine accepts; on failure the observer goes back to what it was doing before.
     */
    private void pair(CommandSender sender, String target, String code) {
        Address address;
        try {
            address = Address.parse(target);
        } catch (IllegalArgumentException e) {
            sender.sendMessage(Component.text(e.getMessage(), NamedTextColor.RED));
            return;
        }
        if (Pairing.normalize(code).isEmpty()) {
            sender.sendMessage(Component.text("That is not a pairing code.", NamedTextColor.RED));
            return;
        }
        String previous = getConfig().getString("engine", "").trim();
        stop();
        start(address.toString(), pairing -> pairing.pair(code, () -> Bukkit.getScheduler().runTask(this, () -> {
            getConfig().set("engine", address.toString());
            saveConfig();
            sender.sendMessage(Component.text("Paired with the Render Fox engine at " + address
                    + ". The stream shows this server from now on.", NamedTextColor.GREEN));
        }), reason -> Bukkit.getScheduler().runTask(this, () -> {
            sender.sendMessage(Component.text("Pairing failed. " + reason, NamedTextColor.RED));
            if (link == pairing) {
                stop();
                start(previous);
            }
        })));
        if (link == null) {
            sender.sendMessage(Component.text("Cannot start the observer, see the console.", NamedTextColor.RED));
            start(previous);
            return;
        }
        sender.sendMessage(Component.text("Pairing with the Render Fox engine at " + address + "...", NamedTextColor.GRAY));
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            String typed = args[0].toLowerCase(Locale.ROOT);
            return List.of("status", "pair", "unpair", "key", "reload").stream().filter(s -> s.startsWith(typed)).toList();
        }
        if (args.length == 2 && args[0].equalsIgnoreCase("pair")) return List.of("<host:" + DEFAULT_PORT + ">");
        if (args.length == 3 && args[0].equalsIgnoreCase("pair")) return List.of("<code>");
        return List.of();
    }

    /** Both feeds run as the tick ends, so what changed during it goes out straight away. */
    @EventHandler
    public void onTickEnd(ServerTickEndEvent e) {
        if (chunks != null) chunks.tick();
        if (entities != null) entities.tick();
    }

    private void dirty(Block block) {
        if (chunks != null && block.getWorld() == world) chunks.markDirty(block.getX(), block.getY(), block.getZ());
    }

    private void dirty(List<Block> blocks) {
        for (Block b : blocks) dirty(b);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockPlaceEvent e) {
        if (e instanceof BlockMultiPlaceEvent multi) {
            for (BlockState s : multi.getReplacedBlockStates()) dirty(s.getBlock());
        }
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockBreakEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockBurnEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockFadeEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockFormEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockSpreadEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockGrowEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockFromToEvent e) {
        dirty(e.getToBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(FluidLevelChangeEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(LeavesDecayEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockFertilizeEvent e) {
        for (BlockState s : e.getBlocks()) dirty(s.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockExplodeEvent e) {
        dirty(e.blockList());
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(EntityExplodeEvent e) {
        dirty(e.blockList());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockPistonExtendEvent e) {
        pistonMoved(e, e.getBlocks());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(BlockPistonRetractEvent e) {
        pistonMoved(e, e.getBlocks());
    }

    private void pistonMoved(BlockPistonEvent e, List<Block> moved) {
        dirty(e.getBlock());
        dirty(e.getBlock().getRelative(e.getDirection()));
        for (Block b : moved) {
            dirty(b);
            dirty(b.getRelative(e.getDirection()));
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(EntityChangeBlockEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(StructureGrowEvent e) {
        for (BlockState s : e.getBlocks()) dirty(s.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(PlayerBucketEmptyEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlock(PlayerBucketFillEvent e) {
        dirty(e.getBlock());
    }

    /**
     * Doors, trapdoors, gates, levers, buttons, repeaters and pressure plates change
     * state with no block event, only this one. A click that changed nothing costs one
     * section's encode, and the unchanged section is not sent.
     */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onBlock(PlayerInteractEvent e) {
        if (e.getClickedBlock() != null && e.useInteractedBlock() != Event.Result.DENY) dirty(e.getClickedBlock());
    }

    /** Lamps, powered doors and rails, wire. */
    @EventHandler(priority = EventPriority.MONITOR)
    public void onBlock(BlockRedstoneEvent e) {
        dirty(e.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onSwing(PlayerArmSwingEvent e) {
        if (entities != null && e.getPlayer().getWorld() == world) entities.swing(e.getPlayer(), e.getHand() == EquipmentSlot.OFF_HAND);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onHurt(EntityDamageEvent e) {
        if (entities != null && e.getEntity().getWorld() == world) entities.hurt(e.getEntity());
    }
}
