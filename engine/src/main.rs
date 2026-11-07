//! Render Fox engine: mirrors the spawn area from the observer plugin and renders a slow
//! orbit around it, entirely on the CPU, encoded for the stream.

mod assets;
mod base64;
mod decode;
mod entities;
mod flv;
// Reads recorded slices back to their macroblocks; for now only the splice tests do.
#[cfg_attr(not(test), allow(dead_code))]
mod h264syn;
mod jobs;
mod mesh;
mod nametags;
mod net;
mod overlay;
mod parts;
mod persist;
mod render;
mod replay;
mod sight;
// The SSE2 types the portable half of `wide` is built from; unused in AVX2 builds.
#[cfg_attr(target_feature = "avx2", allow(dead_code))]
mod simd;
mod stream;
mod sys;
mod vmath;
mod wide;
mod world;
mod x264;

use assets::{Assets, MAT_LEAVES, MAT_PLANT, jf32, val_bool, val_f32, val_i32, val_str};
use entities::Entities;
use jobs::Jobs;
use mesh::{MeshSettings, Mesher, QUAD_FLAT, Quad, SectionMesh};
use nametags::{NameTags, Tag};
use net::{FeedServer, Trust, engine_clock};
use overlay::{Overlay, Text};
use parts::Parts;
use render::{CameraPose, Changed, Footprint, Look, Renderer, ShadowMap, Shadows};
use sight::Changes;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::f32::consts::PI;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use vmath::*;
use wide::{F8, I8};
use world::{World, section_key};

struct Config {
    assets: String,
    cache_dir: String,
    port: i32,
    width: i32,
    height: i32,
    fps: i32,
    threads: i32,
    ahead: i32, // frames rendered ahead of the one going out
    center: Vec3,
    radius: f32,
    orbit_height: f32,
    look_height: f32,
    period: f32,
    fov: f32,
    fancy_leaves_radius: f32,
    static_every: i32, // full static renders every N frames; the rest reproject
    sway: bool,
    fonts: String, // no fonts, no overlay
    now_playing: String,
    text: Text,
    look: Look,
    bitrate: i32, // kbit/s, constant
    preset: String,
    encoder_threads: i32,
    // Replay the orbit when nothing in it changed, rather than render and encode it again.
    replay: bool,
    replay_store: String,
    settle: f32, // seconds a change takes to settle on screen, before a frame may be recorded
    // Seconds the renderer runs from a cold start before its frames are recorded, and ahead
    // of the end of a replay, so it hands over with its antialiasing settled.
    warm: f32,
    // Seconds natural changes, made with no player near, are held back to be shown
    // together (0: at once).
    nature: f32,
    // Macroblock rows per band of the encode (0: no bands): see stream::bands_for.
    band_rows: i32,
}

/// The config may carry // and /* */ comments.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let (mut in_string, mut escaped) = (false, false);
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                in_string = true;
                out.push(c);
            }
            ('/', Some('/')) => {
                while chars.next_if(|&n| n != '\n').is_some() {}
            }
            ('/', Some('*')) => {
                chars.next();
                let mut last = ' ';
                for n in chars.by_ref() {
                    if last == '*' && n == '/' {
                        break;
                    }
                    last = n;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// The config, and which server may feed the engine.
fn load_config(path: &str) -> Result<(Config, Trust), String> {
    let text = std::fs::read_to_string(path).map_err(|_| format!("cannot read config {path}"))?;
    let j: Value = serde_json::from_str(&strip_comments(&text)).map_err(|e| format!("bad config {path}: {e}"))?;
    let v3 = |v: &Value| Vec3::new(jf32(&v[0]), jf32(&v[1]), jf32(&v[2]));
    // Relative paths are the config's neighbours, not the working directory's: systemd
    // starts the stream in /, where the skin cache could never be made.
    let here = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
    let beside = |p: String| if p.is_empty() { p } else { here.join(p).to_string_lossy().into_owned() };
    let width = val_i32(&j, "width", 1280);
    let height = val_i32(&j, "height", 720);
    // observerKey pins the server by hand; without it the engine pairs (net.rs).
    let key_text = val_str(&j, "observerKey", "");
    let observer_key = if key_text.is_empty() { None } else { Some(net::parse_key(&key_text).map_err(|e| format!("observerKey {e}"))?) };
    let pairing = j.get("pairing").cloned().unwrap_or(Value::Null);
    let trust = Trust::load(
        observer_key,
        beside(val_str(&pairing, "keyFile", "observer.pub")),
        beside(val_str(&pairing, "codeFile", "run/pairing.code")),
    )?;
    let mut c = Config {
        assets: beside(val_str(&j, "assets", "assets/client")),
        cache_dir: beside(val_str(&j, "cache", "cache")),
        port: val_i32(&j, "port", 25590),
        width,
        height,
        fps: val_i32(&j, "fps", 30),
        threads: val_i32(&j, "threads", 3),
        ahead: val_i32(&j, "aheadFrames", 4).clamp(1, 16),
        center: Vec3::new(0.0, 72.0, 0.0),
        radius: 55.0,
        orbit_height: 22.0,
        look_height: 2.0,
        period: 360.0,
        fov: 70.0,
        fancy_leaves_radius: val_f32(&j, "fancyLeavesRadius", 96.0),
        static_every: val_i32(&j, "staticEvery", 2),
        sway: val_bool(&j, "plantSway", false),
        fonts: String::new(),
        now_playing: String::new(),
        text: Text::default(),
        look: Look::default(),
        bitrate: 10000,
        preset: "superfast".into(),
        encoder_threads: 2,
        replay: true,
        replay_store: beside("run/replay/region.bin".into()),
        settle: 4.0,
        warm: 1.5,
        nature: 1800.0,
        band_rows: 13,
    };
    if let Some(e) = j.get("encoder") {
        c.bitrate = val_i32(e, "bitrateKbps", c.bitrate);
        c.preset = val_str(e, "preset", &c.preset);
        c.encoder_threads = val_i32(e, "threads", c.encoder_threads);
    }
    if let Some(r) = j.get("replay") {
        c.replay = val_bool(r, "enabled", c.replay);
        c.replay_store = beside(val_str(r, "store", "run/replay/region.bin"));
        c.settle = val_f32(r, "settleSeconds", c.settle);
        c.warm = val_f32(r, "warmSeconds", c.warm);
        c.nature = val_f32(r, "natureSeconds", c.nature);
        c.band_rows = val_i32(r, "bandRows", c.band_rows);
    }
    if let Some(o) = j.get("orbit") {
        if let Some(center) = o.get("center") {
            c.center = v3(center);
        }
        c.radius = val_f32(o, "radius", c.radius);
        c.orbit_height = val_f32(o, "height", c.orbit_height);
        c.look_height = val_f32(o, "lookHeight", c.look_height);
        c.period = val_f32(o, "periodSeconds", c.period);
        c.fov = val_f32(o, "fov", c.fov);
    }
    if let Some(o) = j.get("overlay") {
        c.fonts = beside(val_str(o, "fonts", ""));
        c.now_playing = beside(val_str(o, "nowPlaying", ""));
        let text = &mut c.text;
        for (key, dst) in [
            ("station", &mut text.station),
            ("invite", &mut text.invite),
            ("joinLabel", &mut text.join_label),
            ("address", &mut text.address),
        ] {
            *dst = val_str(o, key, dst);
        }
    }
    if let Some(l) = j.get("look") {
        let look = &mut c.look;
        if let (Some(el), Some(az)) = (l.get("sunElevation"), l.get("sunAzimuth")) {
            let el = jf32(el) * PI / 180.0;
            let az = jf32(az) * PI / 180.0;
            look.sun_dir = normalize(Vec3::new(el.cos() * az.sin(), el.sin(), -el.cos() * az.cos()));
        }
        for (key, dst) in [
            ("sunColor", &mut look.sun_color),
            ("skyZenith", &mut look.sky_zenith),
            ("skyHorizon", &mut look.sky_horizon),
            ("ambient", &mut look.ambient),
        ] {
            if let Some(v) = l.get(key) {
                *dst = v3(v);
            }
        }
        look.sun_intensity = val_f32(l, "sunIntensity", look.sun_intensity);
        look.ambient_intensity = val_f32(l, "ambientIntensity", look.ambient_intensity);
        look.fog_start = val_f32(l, "fogStart", look.fog_start);
        look.fog_end = val_f32(l, "fogEnd", look.fog_end);
        look.exposure = val_f32(l, "exposure", look.exposure);
        look.saturation = val_f32(l, "saturation", look.saturation);
    }
    Ok((c, trust))
}

fn unpack_key(k: u64) -> (i32, i32, i32) {
    let cx = (((k >> 43) as u32) << 11) as i32 >> 11;
    let cz = ((((k >> 22) & 0x1fffff) as u32) << 11) as i32 >> 11;
    let sy = (((k & 0x3fffff) as u32) << 10) as i32 >> 10;
    (cx, sy, cz)
}

/// What one run keeps for the next: the world its recordings of the orbit show, so that
/// after a restart only the recordings a change could show are thrown away (stream.rs).
#[derive(Default)]
pub struct Keep {
    /// The last run's world, section key to hash; taken once this run's world is in.
    pub saved: Mutex<Option<HashMap<u64, u64>>>,
    /// When this run's world was compared with it: kept recordings count from then.
    pub ready_at: Mutex<Option<f64>>,
    /// The world as the worker last left it with every change it took shown: when, and
    /// each section's hash.
    pub snapshot: Mutex<Option<(f64, Vec<(u64, u64)>)>>,
}

/// Keeps lighting, meshes and shadows current as terrain arrives or changes. Someone
/// placing or breaking a block is handled the moment it arrives, at normal priority:
/// relit only as far as light can reach, remeshed, its shadow redrawn, and all of it
/// handed to the renderer at once. Bulk work (the first snapshot, a large WorldEdit) runs
/// at low priority instead, so the frame loop always wins the CPU.
struct WorldWorker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WorldWorker {
    #[allow(clippy::too_many_arguments)]
    fn start(world: Arc<World>, mesher: Arc<Mesher>, shadows: Shadows, feed: Arc<FeedServer>, center: Vec3,
             ready: Arc<AtomicBool>, changes: Arc<Changes>, nature: f64, keep: Arc<Keep>, rest: Arc<Rest>) -> WorldWorker {
        let stop = Arc::new(AtomicBool::new(false));
        let halt = stop.clone();
        let thread = std::thread::Builder::new()
            .name("world".into())
            .spawn(move || {
                world_loop(&world, &mesher, &shadows, &feed, center, &ready, &changes, Nature::new(nature), &keep, &rest, &halt)
            })
            .expect("spawn world worker");
        WorldWorker { stop, thread: Some(thread) }
    }
}

impl Drop for WorldWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Runs `f` on a thread of its own at low priority, and waits for it. Threads it starts
/// inherit the priority.
fn in_background<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| {
        s.spawn(|| {
            sys::lower_priority(10);
            f()
        })
        .join()
        .expect("world background work panicked")
    })
}

type MeshBatch = Vec<(u64, Option<Arc<SectionMesh>>)>;

/// Edits handled and what they cost, reported once a minute.
#[derive(Default)]
struct EditStats {
    count: u32,
    sections: usize,
    light: f64,
    meshes: f64,
    shadows: f64,
    longest: f64,
    since: f64,
}

#[allow(clippy::too_many_arguments)]
fn world_loop(world: &World, mesher: &Mesher, shadows: &Shadows, feed: &FeedServer, center: Vec3, ready: &AtomicBool,
              changes: &Changes, mut nature: Nature, keep: &Keep, rest: &Rest, stop: &AtomicBool) {
    let mut last_snapshot = engine_clock();
    let mut first_data = -1.0;
    let mut maps: Option<Arc<Vec<ShadowMap>>> = None;
    // Sections whose light an edit changed, still to be remeshed. Their faces brighten or
    // darken a moment after the edit shows, which reads as the light settling.
    let mut later: BTreeSet<u64> = BTreeSet::new();
    // A bulk change's shadows are redrawn at most once a second, over the union of it.
    let far = Vec3::splat(1e30);
    let (mut bulk_lo, mut bulk_hi) = (far, -far);
    let mut last_shadow = -100.0;
    let mut last_edit = -100.0;
    let mut stats = EditStats { since: engine_clock(), ..EditStats::default() };
    let mut asleep = false;
    let mut woke_at = -100.0;
    let players = |now: f64| -> Vec<Vec3> { feed.entities(now).iter().filter_map(|e| e.samples.back().map(|s| s.pos)).collect() };
    while !stop.load(Ordering::SeqCst) {
        if asleep {
            std::thread::sleep(Duration::from_millis(150));
            let now = engine_clock();
            // What arrives meanwhile: a big change, or edits to show now, wake it; natural
            // changes held back stay held as they would awake.
            let taken = if world.pending() { Some(world.take_changes()) } else { None };
            let (big, edits) = match taken {
                Some(t) if t.bulk => (Some(t), Vec::new()),
                t => (None, nature.sort(t.map(|t| t.edits).unwrap_or_default(), || players(now), now)),
            };
            if big.is_none() && edits.is_empty() && rest.stay() {
                continue;
            }
            let fresh = in_background(|| wake_up(world, mesher, shadows, center));
            shadows.install(fresh.clone());
            maps = Some(fresh);
            if let Some(taken) = big {
                let (lo, hi) = in_background(|| bulk(world, mesher, &taken, &mut later, changes, false));
                bulk_lo = vmin(bulk_lo, lo);
                bulk_hi = vmax(bulk_hi, hi);
            }
            if !edits.is_empty() {
                maps = Some(edit(world, mesher, shadows, maps.as_deref().unwrap(), &edits, &mut later, changes, &mut stats));
                last_edit = engine_clock();
            }
            asleep = false;
            woke_at = engine_clock();
            rest.woke();
            continue;
        }
        // Nothing to draw the world for a while and nothing owed: its meshes and shadow
        // maps go until something needs them.
        if maps.is_some()
            && engine_clock() - woke_at > 10.0
            && later.is_empty()
            && bulk_lo.x > bulk_hi.x
            && !world.pending()
            && rest.sleep()
        {
            maps = None;
            shadows.forget();
            mesher.clear();
            sys::trim_memory();
            asleep = true;
            eprintln!("world: resting, meshes and shadow maps let go until the world is drawn again");
            continue;
        }
        if maps.is_none() {
            std::thread::sleep(Duration::from_millis(150));
        } else {
            let owed = bulk_lo.x <= bulk_hi.x;
            world.wait(Duration::from_millis(if later.is_empty() && !owed { 150 } else { 20 }));
        }
        let now = engine_clock();
        let Some(current) = maps.clone() else {
            // Wait for the whole snapshot, then light and mesh it in one pass instead of
            // relighting tiles over and over as neighbours trickle in.
            if world.revision() == 0 {
                continue;
            }
            if first_data < 0.0 {
                first_data = now;
            }
            if !feed.snapshot_done() && now - first_data < 180.0 {
                continue;
            }
            let taken = world.take_changes();
            let fresh = in_background(|| {
                bulk(world, mesher, &taken, &mut later, changes, true);
                if std::env::var_os("RF_STATS").is_some() {
                    census(mesher, center);
                }
                let t0 = engine_clock();
                let fresh = shadows.build(&mesher.snapshot(), center, 240.0);
                eprintln!("world: shadows rebuilt in {:.3}s", engine_clock() - t0);
                fresh
            });
            shadows.install(fresh.clone());
            let everything = engine_clock();
            changes.everything(everything);
            if let Some(saved) = keep.saved.lock().unwrap().take() {
                since_last_run(world, shadows, changes, keep, &saved, everything);
            }
            maps = Some(fresh);
            last_shadow = now;
            ready.store(true, Ordering::SeqCst);
            continue;
        };
        // The plugin sends a tick's edits within a millisecond or two: take them together.
        let mut seen = world.revision();
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(2));
            let r = world.revision();
            if r == seen {
                break;
            }
            seen = r;
        }
        // Edits that keep coming (a redstone clock in view, someone filling an area by
        // hand) are taken together at most ten times a second, so they cannot take a core.
        let rest = last_edit + 0.1 - engine_clock();
        if rest > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(rest));
        }
        let taken = world.take_changes();
        if taken.bulk {
            let (lo, hi) = in_background(|| bulk(world, mesher, &taken, &mut later, changes, false));
            bulk_lo = vmin(bulk_lo, lo);
            bulk_hi = vmax(bulk_hi, hi);
        } else if let edits = nature.sort(taken.edits, || players(now), now)
            && !edits.is_empty()
        {
            maps = Some(edit(world, mesher, shadows, &current, &edits, &mut later, changes, &mut stats));
            last_edit = engine_clock();
        } else if !later.is_empty() {
            let keys: Vec<u64> = std::iter::from_fn(|| later.pop_first()).take(16).collect();
            let batch: MeshBatch = in_background(|| keys.iter().map(|&k| (k, build_mesh(world, mesher, k))).collect());
            mesher.publish_all(batch, || {});
            changes.boxes(engine_clock(), keys.into_iter().map(section_box));
        }
        if bulk_lo.x <= bulk_hi.x && now - last_shadow > 1.0 {
            let base = maps.clone().unwrap();
            let fresh = in_background(|| {
                let t0 = engine_clock();
                let meshes = mesher.snapshot();
                let full = bulk_hi.x - bulk_lo.x > 128.0 || bulk_hi.z - bulk_lo.z > 128.0;
                let fresh = if full { shadows.build(&meshes, center, 240.0) } else { shadows.update(&base, &meshes, &[(bulk_lo, bulk_hi)]) };
                eprintln!("world: shadows {} in {:.3}s", if full { "rebuilt" } else { "updated" }, engine_clock() - t0);
                (fresh, full)
            });
            let (fresh, full) = fresh;
            shadows.install(fresh.clone());
            if full {
                changes.everything(engine_clock());
            } else {
                changes.shadows(engine_clock(), &[(bulk_lo, bulk_hi)], shadows.to_sun(), |a, b| floor(world, a, b));
            }
            maps = Some(fresh);
            (bulk_lo, bulk_hi) = (far, -far);
            last_shadow = now;
        }
        // Once a minute, with nothing taken left unshown, the world as recordings now show it.
        if now - last_snapshot >= 60.0 && later.is_empty() && bulk_lo.x > bulk_hi.x {
            last_snapshot = now;
            let r0 = world.revision();
            if !world.pending() {
                let mut hashes = world.section_hashes();
                if world.revision() == r0 && !world.pending() {
                    // Natural changes held back are in the world and in no recording yet.
                    let held: BTreeSet<u64> = nature.held.iter().flat_map(edit_sections).collect();
                    for (k, h) in hashes.iter_mut() {
                        if held.contains(k) {
                            *h = 0;
                        }
                    }
                    *keep.snapshot.lock().unwrap() = Some((engine_clock(), hashes));
                }
            }
        }
        if now - stats.since >= 60.0 {
            if stats.count > 0 {
                let n = stats.count as f64;
                eprintln!(
                    "world: {} edits in the last minute, {} sections remeshed at once, {:.0} ms each on average (light {:.0}, meshes {:.0}, shadows {:.0}), longest {:.0} ms",
                    stats.count,
                    stats.sections,
                    (stats.light + stats.meshes + stats.shadows) / n,
                    stats.light / n,
                    stats.meshes / n,
                    stats.shadows / n,
                    stats.longest
                );
            }
            stats = EditStats { since: now, ..EditStats::default() };
        }
    }
}

/// The sections an edit touches.
fn edit_sections(e: &world::Edit) -> Vec<u64> {
    let s = |v: i32| v.div_euclid(16);
    let mut out = Vec::new();
    for sy in s(e.lo[1])..=s(e.hi[1]) {
        for cz in s(e.lo[2])..=s(e.hi[2]) {
            for cx in s(e.lo[0])..=s(e.hi[0]) {
                out.push(section_key(cx, sy, cz));
            }
        }
    }
    out
}

/// Compares the world as it arrived with the one the last run's recordings show, and
/// publishes what differs as changes, which throw away the recordings that could show
/// them. Kept recordings count from a moment after the snapshot's own changes, so those
/// (every section, as it arrived) leave them alone.
fn since_last_run(world: &World, shadows: &Shadows, changes: &Changes, keep: &Keep, saved: &HashMap<u64, u64>, arrived: f64) {
    let ready_at = arrived + 0.1;
    let now: HashMap<u64, u64> = world.section_hashes().into_iter().collect();
    let mut differ: Vec<u64> = now.iter().filter(|(k, h)| saved.get(k) != Some(h)).map(|(k, _)| *k).collect();
    differ.extend(saved.keys().filter(|k| !now.contains_key(k)));
    let wait = ready_at - engine_clock();
    if wait > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(wait));
    }
    let at = engine_clock();
    // Light reaches 15 blocks from a change; its shadow falls further, down the sun.
    let grow = Vec3::splat(16.0);
    changes.boxes(at, differ.iter().map(|&k| section_box(k)).map(|(lo, hi)| (lo - grow, hi + grow)));
    let boxes: Vec<(Vec3, Vec3)> = differ.iter().map(|&k| section_box(k)).collect();
    changes.shadows(at, &boxes, shadows.to_sun(), |a, b| floor(world, a, b));
    *keep.ready_at.lock().unwrap() = Some(ready_at);
    eprintln!("world: {} of {} sections differ from what the last run recorded", differ.len(), now.len());
}

fn section_box(key: u64) -> (Vec3, Vec3) {
    let (cx, sy, cz) = unpack_key(key);
    let lo = Vec3::new(cx as f32, sy as f32, cz as f32) * 16.0;
    (lo, lo + Vec3::splat(16.0))
}

/// The lowest ground under the box between a and b: sunlight going further down has
/// entered the ground everywhere under it.
fn floor(world: &World, a: Vec3, b: Vec3) -> f32 {
    let (lo, hi) = (vmin(a, b), vmax(a, b));
    let mut y = i32::MAX;
    for (x, z) in [(lo.x, lo.z), (hi.x, lo.z), (lo.x, hi.z), (hi.x, hi.z), ((lo.x + hi.x) * 0.5, (lo.z + hi.z) * 0.5)] {
        y = y.min(world.highest_solid(x.floor() as i32, z.floor() as i32) + 1);
    }
    y as f32
}

fn build_mesh(world: &World, mesher: &Mesher, key: u64) -> Option<Arc<SectionMesh>> {
    let (cx, sy, cz) = unpack_key(key);
    if sy < world.min_section() || sy > world.max_section() {
        return None;
    }
    mesher.build(cx, sy, cz)
}

/// Changes no player made: grass a sheep ate, a crop or bamboo that grew, leaves that
/// fell. Each one spoils every recording of the orbit that shows it, and one at spawn shows
/// in nearly all of them, so a few a minute, as spawn makes on its own, would keep the
/// stream rendering forever. They are held back and shown together every `wait` seconds;
/// anything within reach of a player, or next to what a player just changed, shows at once.
struct Nature {
    wait: f64,
    held: Vec<world::Edit>,
}

impl Nature {
    const REACH: f32 = 10.0;

    fn new(wait: f64) -> Nature {
        Nature { wait, held: Vec::new() }
    }

    /// The edits to show now, from those just arrived and those held back.
    fn sort(&mut self, edits: Vec<world::Edit>, players: impl FnOnce() -> Vec<Vec3>, now: f64) -> Vec<world::Edit> {
        if self.wait <= 0.0 || edits.is_empty() && self.held.is_empty() {
            return edits;
        }
        let players = players();
        let centre = |e: &world::Edit| Vec3::new((e.lo[0] + e.hi[0]) as f32, (e.lo[1] + e.hi[1]) as f32, (e.lo[2] + e.hi[2]) as f32) * 0.5;
        let (mut show, grew): (Vec<_>, Vec<_>) =
            edits.into_iter().partition(|e| players.iter().any(|&p| length(centre(e) - p) <= Nature::REACH));
        self.held.extend(grew);
        let oldest = self.held.iter().map(|e| e.arrived).fold(f64::MAX, f64::min);
        if now - oldest >= self.wait {
            eprintln!("world: showing {} natural changes held back over {:.0} minutes", self.held.len(), (now - oldest) / 60.0);
            show.extend(self.held.drain(..).map(|e| world::Edit { arrived: now, ..e }));
        } else if !show.is_empty() {
            // Relit and remeshed with a player's edit next to them, they would show anyway,
            // and with light that had not caught up.
            let section = |e: &world::Edit| [e.lo[0].div_euclid(16), e.lo[1].div_euclid(16), e.lo[2].div_euclid(16)];
            let touched: Vec<[i32; 3]> = show.iter().map(section).collect();
            let near = |e: &world::Edit| touched.iter().any(|t| (0..3).all(|i| (section(e)[i] - t[i]).abs() <= 1));
            let (with, rest): (Vec<_>, Vec<_>) = self.held.drain(..).partition(near);
            show.extend(with.into_iter().map(|e| world::Edit { arrived: now, ..e }));
            self.held = rest;
        }
        show
    }
}

/// A few edits, made by players: light around them, the sections they touch remeshed and
/// their shadows redrawn, then all of it published together with a note of what changed,
/// which brings the renderer's next full frame forward and lets the change show at once.
#[allow(clippy::too_many_arguments)]
fn edit(world: &World, mesher: &Mesher, shadows: &Shadows, maps: &[ShadowMap], edits: &[world::Edit], later: &mut BTreeSet<u64>,
        changes: &Changes, stats: &mut EditStats) -> Arc<Vec<ShadowMap>> {
    let t0 = engine_clock();
    let lit = world.relight_around(edits);
    let t1 = engine_clock();
    // Every section within a block of an edited block: its own faces, and the faces,
    // corner shading and light its neighbours read across the boundary.
    let mut now = BTreeSet::new();
    for e in edits {
        let s = |v: i32| v.div_euclid(16);
        for sy in s(e.lo[1] - 1)..=s(e.hi[1] + 1) {
            for cz in s(e.lo[2] - 1)..=s(e.hi[2] + 1) {
                for cx in s(e.lo[0] - 1)..=s(e.hi[0] + 1) {
                    now.insert(section_key(cx, sy, cz));
                }
            }
        }
    }
    for &k in &lit {
        let (cx, sy, cz) = unpack_key(k);
        later.insert(k);
        for d in 0..6 {
            later.insert(section_key(cx + DIR_DX[d], sy + DIR_DY[d], cz + DIR_DZ[d]));
        }
    }
    later.retain(|k| !now.contains(k));
    let batch: MeshBatch = now.into_iter().map(|k| (k, build_mesh(world, mesher, k))).collect();
    let t2 = engine_clock();
    stats.sections += batch.len();
    let block = |v: [i32; 3]| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
    let changed: Vec<Changed> =
        edits.iter().map(|e| Changed { lo: block(e.lo), hi: block(e.hi) + Vec3::splat(1.0), arrived: e.arrived }).collect();
    let boxes: Vec<(Vec3, Vec3)> = changed.iter().map(|c| (c.lo, c.hi)).collect();
    let fresh = shadows.update(maps, &mesher.snapshot_after(&batch), &boxes);
    let t3 = engine_clock();
    let keys: Vec<u64> = batch.iter().map(|b| b.0).collect();
    mesher.publish_all(batch, || {
        shadows.install(fresh.clone());
        shadows.changed(&changed);
    });
    let at = engine_clock();
    changes.boxes(at, keys.into_iter().map(section_box));
    changes.shadows(at, &boxes, shadows.to_sun(), |a, b| floor(world, a, b));
    stats.count += 1;
    stats.light += (t1 - t0) * 1000.0;
    stats.meshes += (t2 - t1) * 1000.0;
    stats.shadows += (t3 - t2) * 1000.0;
    stats.longest = stats.longest.max((t3 - t0) * 1000.0);
    fresh
}

/// A snapshot or a big edit: the tiles around every changed column relit, the changed
/// sections, their neighbours and everything whose light changed remeshed, published as
/// they are built. Returns the box the changed sections span, for the shadows.
fn bulk(world: &World, mesher: &Mesher, changes: &world::Changes, later: &mut BTreeSet<u64>, seen: &Changes, parallel: bool)
        -> (Vec3, Vec3) {
    let far = Vec3::splat(1e30);
    let (mut lo, mut hi) = (far, -far);
    for &k in &changes.sections {
        let (cx, sy, cz) = unpack_key(k);
        lo = vmin(lo, Vec3::new(cx as f32 * 16.0, sy as f32 * 16.0, cz as f32 * 16.0));
        hi = vmax(hi, Vec3::new(cx as f32 * 16.0 + 16.0, sy as f32 * 16.0 + 16.0, cz as f32 * 16.0 + 16.0));
    }
    let mut expanded = BTreeSet::new();
    for &(cx, cz) in &changes.columns {
        for dz in -1..=1 {
            for dx in -1..=1 {
                expanded.insert((cx + dx, cz + dz));
            }
        }
    }
    let t0 = engine_clock();
    let mut changed = world.relight(&expanded.iter().copied().collect::<Vec<_>>());
    // Sections with new blocks or new light, and their neighbours, whose faces and
    // corner lighting read across the shared boundary.
    changed.extend_from_slice(&changes.sections);
    let mut todo = std::mem::take(later);
    for &k in &changed {
        let (cx, sy, cz) = unpack_key(k);
        todo.insert(section_key(cx, sy, cz));
        for d in 0..6 {
            todo.insert(section_key(cx + DIR_DX[d], sy + DIR_DY[d], cz + DIR_DZ[d]));
        }
    }
    let list: Vec<u64> = todo.into_iter().collect();
    mesh_sections(world, mesher, &list, parallel);
    seen.boxes(engine_clock(), list.iter().copied().map(section_box));
    let (sections, bytes) = world.section_memory();
    eprintln!(
        "world: {} columns relit, {} sections meshed in {:.2}s, {} quads total; {} sections held in {:.1} MiB",
        expanded.len(),
        list.len(),
        engine_clock() - t0,
        mesher.quad_count(),
        sections,
        bytes as f64 / (1 << 20) as f64
    );
    (lo, hi)
}

/// Builds and publishes the meshes of the sections `list` names, on every core if
/// `parallel`.
fn mesh_sections(world: &World, mesher: &Mesher, list: &[u64], parallel: bool) {
    let next = AtomicUsize::new(0);
    let build = || {
        while let Some(&k) = list.get(next.fetch_add(1, Ordering::Relaxed)) {
            mesher.publish(k, build_mesh(world, mesher, k));
        }
    };
    let n = if parallel { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { 1 };
    std::thread::scope(|s| {
        for _ in 1..n {
            s.spawn(build);
        }
        build();
    });
}

/// Every section's mesh built again, as the first snapshot's were, after `Rest` let them
/// go: from the world as it is, with nothing published as changed. Returns fresh shadow
/// maps from them.
fn wake_up(world: &World, mesher: &Mesher, shadows: &Shadows, center: Vec3) -> Arc<Vec<ShadowMap>> {
    let t0 = engine_clock();
    let mut todo = BTreeSet::new();
    for k in world.section_keys() {
        let (cx, sy, cz) = unpack_key(k);
        todo.insert(k);
        for d in 0..6 {
            todo.insert(section_key(cx + DIR_DX[d], sy + DIR_DY[d], cz + DIR_DZ[d]));
        }
    }
    let list: Vec<u64> = todo.into_iter().collect();
    mesh_sections(world, mesher, &list, true);
    let t1 = engine_clock();
    let maps = shadows.build(&mesher.snapshot(), center, 240.0);
    eprintln!("world: awake: {} sections meshed in {:.2}s, {} quads, shadows in {:.2}s", list.len(), t1 - t0, mesher.quad_count(),
              engine_clock() - t1);
    maps
}

/// The world's meshes and shadow maps let go while nothing will draw the world (no one
/// online and every group of the orbit recorded, so every frame is replayed), and had back
/// before anything does: they are most of what the engine holds then, and take a second or
/// two of background work to build again. The stream allows it, or says it needs them;
/// the world worker lets them go and builds them again.
#[derive(Default)]
struct Rest(Mutex<RestState>);

#[derive(Default)]
struct RestState {
    allowed: bool,
    asleep: bool,
}

impl Rest {
    /// The stream: nothing needs the world drawn for now.
    fn allow(&self) {
        self.0.lock().unwrap().allowed = true;
    }

    /// The stream: something will draw the world. True when it can be drawn now; false
    /// while the meshes are gone or being built again.
    fn need(&self) -> bool {
        let mut s = self.0.lock().unwrap();
        s.allowed = false;
        !s.asleep
    }

    fn asleep(&self) -> bool {
        self.0.lock().unwrap().asleep
    }

    /// The world worker: whether it may let them go now, and if so, asleep from here on.
    /// Taken under the lock with `need`, so nothing draws once this says yes.
    fn sleep(&self) -> bool {
        let mut s = self.0.lock().unwrap();
        if s.allowed && !s.asleep {
            s.asleep = true;
        }
        s.asleep
    }

    /// The world worker: whether it should stay asleep.
    fn stay(&self) -> bool {
        self.0.lock().unwrap().allowed
    }

    /// The world worker: built again. Until the stream allows it afresh, it stays awake.
    fn woke(&self) {
        let mut s = self.0.lock().unwrap();
        s.asleep = false;
        s.allowed = false;
    }
}

/// Where the quads come from, by distance band and kind; guides the LOD budget.
fn census(mesher: &Mesher, center: Vec3) {
    const KINDS: [&str; 6] = ["solid", "cutout", "translucent", "plant", "leaves", "merged"];
    const BANDS: [&str; 4] = ["<64", "64-128", "128-192", ">192"];
    let mut counts = [[0i64; 6]; 4];
    for m in mesher.snapshot() {
        let d = ((m.cx * 16 + 8) as f32 - center.x).hypot((m.cz * 16 + 8) as f32 - center.z);
        let band = if d < 64.0 { 0 } else if d < 128.0 { 1 } else if d < 192.0 { 2 } else { 3 };
        let big = |e: Vec3| e.x.abs() + e.y.abs() + e.z.abs() > 1.5;
        for q in m.quads.iter().chain(&m.water) {
            let mut k = if q.mat & MAT_PLANT != 0 {
                3
            } else if q.mat & MAT_LEAVES != 0 {
                4
            } else {
                q.layer as usize
            };
            if q.flags & QUAD_FLAT != 0 && (big(q.e1) || big(q.e2)) {
                k = 5;
            }
            counts[band][k] += 1;
        }
    }
    for (b, name) in BANDS.iter().enumerate() {
        let line: Vec<String> = KINDS.iter().zip(counts[b]).map(|(k, n)| format!("{k}={n}")).collect();
        eprintln!("census {name:<8} {}", line.join(" "));
    }
}

// The orbit as a closed path: radius and eye height (above the centre) every half degree.
const ORBIT_N: usize = 720;

#[derive(Clone)]
struct OrbitPath {
    radius: [f32; ORBIT_N],
    height: [f32; ORBIT_N],
}

impl OrbitPath {
    fn zero() -> OrbitPath {
        OrbitPath { radius: [0.0; ORBIT_N], height: [0.0; ORBIT_N] }
    }
}

// Where something stands in the way, the camera pulls in towards spawn, and climbs only
// if even the tightest allowed circle is blocked. Lifting to clear each obstacle on the
// spot put the camera just above a large build's roof for 40 s a lap, the picture mostly
// white, with a hard cut in and out.
fn plan_orbit(c: &Config, world: &World) -> OrbitPath {
    const N: usize = ORBIT_N;
    let eye = c.center.y + c.orbit_height;
    let tightest = c.radius * 0.45;
    // Lowest clear eye height at this spot.
    let needed = |a: f32, r: f32| {
        let x = (c.center.x + a.cos() * r).floor() as i32;
        let z = (c.center.z + a.sin() * r).floor() as i32;
        let mut top = world.min_y();
        for dz in -2..=2 {
            for dx in -2..=2 {
                top = top.max(world.highest_solid(x + dx, z + dz));
            }
        }
        top as f32 + 4.0
    };
    let mut want = OrbitPath::zero();
    for i in 0..N {
        let a = ((i * 2) as f64 * std::f64::consts::PI / N as f64) as f32;
        let mut r = c.radius;
        while r > tightest && needed(a, r) > eye {
            r -= 1.0;
        }
        if needed(a, r) <= eye {
            want.radius[i] = r;
            want.height[i] = c.orbit_height;
        } else {
            want.radius[i] = c.radius;
            want.height[i] = needed(a, c.radius) - c.center.y;
        }
    }
    // Ease into every change: the radius falls, and the eye rises, by at most 0.35 m per
    // metre travelled, starting well before the obstacle.
    let arc = (2.0 * std::f64::consts::PI) as f32 * c.radius / N as f32;
    let slope = 0.35f32;
    let mut eased = want.clone();
    for i in 0..N {
        for j in 0..N {
            let d = i.abs_diff(j);
            let run = d.min(N - d) as f32 * arc * slope;
            eased.radius[i] = fmin(eased.radius[i], want.radius[j] + run);
            eased.height[i] = fmax(eased.height[i], want.height[j] - run);
        }
    }
    // Round the corners: a window minimum (maximum for height) and then a window average
    // over the same span, so the smoothed path never moves back into what it avoids.
    const S: i32 = 24;
    let mut tight = eased.clone();
    let mut smooth = OrbitPath::zero();
    let wrap = |i: usize, k: i32| (i as i32 + k + N as i32) as usize % N;
    for i in 0..N {
        for k in -S..=S {
            let j = wrap(i, k);
            tight.radius[i] = fmin(tight.radius[i], eased.radius[j]);
            tight.height[i] = fmax(tight.height[i], eased.height[j]);
        }
    }
    for i in 0..N {
        let (mut r, mut h) = (0.0f32, 0.0f32);
        for k in -S..=S {
            let j = wrap(i, k);
            r += tight.radius[j];
            h += tight.height[j];
        }
        smooth.radius[i] = r / (2 * S + 1) as f32;
        smooth.height[i] = h / (2 * S + 1) as f32;
    }
    smooth
}

fn orbit_pose(c: &Config, unix_seconds: f64, path: &OrbitPath) -> CameraPose {
    // The angle comes from the wall clock, so a restart resumes exactly where the camera
    // would have been.
    let turns = (unix_seconds / c.period as f64) % 1.0;
    let a = (turns * 2.0 * std::f64::consts::PI) as f32;
    let at = turns * ORBIT_N as f64;
    let i = at as i32 as usize % ORBIT_N;
    let j = (i + 1) % ORBIT_N;
    let f = (at - at.floor()) as f32;
    let r = path.radius[i] + (path.radius[j] - path.radius[i]) * f;
    let h = path.height[i] + (path.height[j] - path.height[i]) * f;
    CameraPose {
        eye: c.center + Vec3::new(a.cos() * r, h, a.sin() * r),
        target: c.center + Vec3::new(0.0, c.look_height, 0.0),
        fov_y: c.fov * PI / 180.0,
    }
}

fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// RGB24 to planar YUV 4:2:0, BT.709, limited range, chroma from each 2x2 block's
/// average: what the encoder wants, so it neither converts nor reads twice the bytes.
/// BT.709 limited range, 4:2:0, from 8-bit R'G'B' planes: Y for every pixel, Cb and Cr
/// from each 2 x 2 block's sums. Integer arithmetic over whole rows, which vectorises.
fn to_yuv420(rgb: &[Vec<u8>; 3], yuv: &mut [u8], w: i32, h: i32, jobs: &Jobs) {
    let (w, h) = (w as usize, h as usize);
    let (luma, chroma) = yuv.split_at_mut(w * h);
    let (u, v) = chroma.split_at_mut((w / 2) * (h / 2));
    let (ys, us, vs) = (Parts::chunks(luma, 2 * w), Parts::chunks(u, w / 2), Parts::chunks(v, w / 2));
    let [pr, pg, pb] = rgb;
    let s = F8::splat;
    let px = |p: &[u8], i: usize| I8::load_u8(p[i..i + 8].try_into().unwrap()).to_float();
    jobs.run((h / 2) as i32, |cy, _| {
        let cy = cy as usize;
        let mut y = ys.take(cy);
        for r in 0..2 {
            let at = (2 * cy + r) * w;
            for (m, dst) in y[r * w..(r + 1) * w].as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let x = at + 8 * m;
                // Integers below 2^24 throughout, so exactly the fixed-point formula.
                let sum = s(11966.0).mul_add(px(pr, x), s(40254.0).mul_add(px(pg, x), s(4064.0) * px(pb, x)));
                *dst = ((sum + s(32768.0 + 16.0 * 65536.0)) * s(1.0 / 65536.0)).to_int().to_bytes();
            }
        }
        // Each chroma sample from a 2 x 2 block: sixteen columns of both rows at a time.
        let (a, b) = (2 * cy * w, (2 * cy + 1) * w);
        let (mut u, mut v) = (us.take(cy), vs.take(cy));
        for (m, (du, dv)) in u.as_chunks_mut::<8>().0.iter_mut().zip(v.as_chunks_mut::<8>().0).enumerate() {
            let x = 16 * m;
            let block = |p: &[u8]| F8::pair_sums(px(p, a + x) + px(p, b + x), px(p, a + x + 8) + px(p, b + x + 8));
            let (r, g, bl) = (block(pr), block(pg), block(pb));
            let bias = s(131072.0 + 128.0 * 262144.0);
            *du = ((s(-6597.0).mul_add(r, s(-22187.0).mul_add(g, s(28784.0) * bl)) + bias) * s(1.0 / 262144.0)).to_int().to_bytes();
            *dv = ((s(28784.0).mul_add(r, s(-26147.0).mul_add(g, s(-2637.0) * bl)) + bias) * s(1.0 / 262144.0)).to_int().to_bytes();
        }
    });
}

fn write_png(path: &str, w: i32, h: i32, planes: &[Vec<u8>; 3]) -> Result<(), String> {
    let rgb: Vec<u8> = (0..(w * h) as usize).flat_map(|i| planes.each_ref().map(|p| p[i])).collect();
    let file = std::fs::File::create(path).map_err(|e| format!("cannot write {path}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(&rgb).map_err(|e| e.to_string())
}

type WorldSlot = Arc<Mutex<Option<(Arc<World>, Arc<Mesher>)>>>;

fn quitting() -> bool {
    sys::QUIT.load(Ordering::SeqCst)
}

const USAGE: &str = "usage: rfengine [--config file] [--stream [--epoch-file f] [--prerecord] | --snap out.png [--angle deg] | --bench frames [--angle deg] [--dump frames.gray] [--rgb frames.rgb] [--realtime]]";

fn main() {
    sys::return_freed_memory();
    std::process::exit(match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("rfengine: {e}");
            1
        }
    });
}

fn run() -> Result<i32, String> {
    let missing = wide::missing_cpu_features();
    if !missing.is_empty() {
        return Err(format!("built for a CPU with {}, which this one lacks: rebuild it here", missing.join(" and ")));
    }
    let mut config_path = "renderfox.json".to_string();
    let mut mode = "stream";
    let (mut snap_path, mut epoch_file, mut dump_path, mut rgb_path) = (String::new(), String::new(), String::new(), String::new());
    let mut realtime = false;
    let mut prerecord = false;
    let mut snap_angle = -1.0f64;
    let mut bench_frames = 60;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match (a.as_str(), it.as_slice().first()) {
            ("--config", Some(v)) => config_path = v.clone(),
            ("--snap", Some(v)) => {
                mode = "snap";
                snap_path = v.clone();
            }
            ("--angle", Some(v)) => snap_angle = v.parse().map_err(|_| format!("bad angle {v}"))?,
            ("--bench", Some(v)) => {
                mode = "bench";
                bench_frames = v.parse().map_err(|_| format!("bad frame count {v}"))?;
            }
            ("--stream", _) => {
                mode = "stream";
                continue;
            }
            ("--epoch-file", Some(v)) => epoch_file = v.clone(),
            ("--dump", Some(v)) => dump_path = v.clone(),
            ("--rgb", Some(v)) => rgb_path = v.clone(),
            ("--realtime", _) => {
                realtime = true;
                continue;
            }
            ("--prerecord", _) => {
                prerecord = true;
                continue;
            }
            _ => {
                eprintln!("{USAGE}");
                return Ok(2);
            }
        }
        it.next();
    }
    sys::catch_quit_signals();

    let (cfg, trust) = load_config(&config_path)?;
    let cfg = Arc::new(cfg);
    let assets = Arc::new(Assets::new(&cfg.assets));
    let jobs = Arc::new(Jobs::new((cfg.threads - 1).max(0) as u32));
    let mut renderer = Renderer::new(assets.clone(), jobs.clone(), cfg.width, cfg.height)?;
    renderer.set_look(cfg.look);
    renderer.set_static_every(cfg.static_every);
    renderer.set_sway(cfg.sway);
    renderer.set_loop(cfg.period);
    let mut entities = Entities::new(assets.clone(), &format!("{}/skins", cfg.cache_dir));
    let mut overlay = if cfg.fonts.is_empty() {
        None
    } else {
        Some(Overlay::new(jobs.clone(), &cfg.fonts, &cfg.now_playing, renderer.width(), renderer.height(), cfg.text.clone())?)
    };

    let slot: WorldSlot = Arc::new(Mutex::new(None));
    let feed = {
        let (slot, assets, cfg) = (slot.clone(), assets.clone(), cfg.clone());
        Arc::new(FeedServer::new(
            assets.clone(),
            cfg.port,
            trust,
            Box::new(move |h| {
                let mut s = slot.lock().unwrap();
                let (world, _) = s.get_or_insert_with(|| {
                    let world = Arc::new(World::new(assets.clone(), h.min_y, h.max_y - h.min_y));
                    let settings =
                        MeshSettings { focus: cfg.center, fancy_leaves_radius: cfg.fancy_leaves_radius, ..MeshSettings::default() };
                    let mesher = Arc::new(Mesher::new(world.clone(), settings));
                    (world, mesher)
                });
                world.clone()
            }),
        ))
    };

    // Everything a frame needs besides the renderer: the world once it exists, the world
    // worker, and the orbit, replanned as the world changes.
    let world_ready = Arc::new(AtomicBool::new(false));
    let mut scene = Scene {
        cfg: cfg.clone(),
        slot,
        feed: feed.clone(),
        shadows: renderer.shadows(),
        ready: world_ready.clone(),
        changes: Arc::new(Changes::default()),
        keep: Arc::new(Keep::default()),
        world: None,
        worker: None,
        path: OrbitPath::zero(),
        previous: OrbitPath::zero(),
        planned: 0.0,
        have_plan: false,
        dynamic: Vec::new(),
        bare: false,
        rest: Arc::new(Rest::default()),
        people_at: engine_clock(),
        // Players are posed this far behind the moment their frame is shown. Streaming, a
        // frame is rendered up to `ahead` slots before it goes out, and the newest position
        // must already be there: that lead, one tick of the feed and some margin.
        tags: NameTags::new(&assets.path("assets/minecraft/textures/font/ascii.png"))?,
        tag_fades: HashMap::new(),
        entity_delay: if mode == "stream" { (cfg.ahead + 1) as f64 / cfg.fps as f64 + 0.075 } else { 0.15 },
        latencies: Vec::new(),
    };

    if mode == "snap" || mode == "bench" {
        let start = engine_clock();
        while !scene.ready() {
            if quitting() || engine_clock() - start >= 600.0 {
                eprintln!("no world arrived in time");
                return Ok(1);
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let base = if snap_angle >= 0.0 { (snap_angle / 360.0) * cfg.period as f64 } else { unix_now() };
        let frames = if mode == "snap" { 24 } else { bench_frames };
        let (mut setup, mut raster, mut shade, mut post, mut full_ms, mut lite_ms) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut pass_full, mut pass_lite) = ([0.0; render::PASSES.len()], [0.0; render::PASSES.len()]);
        let mut fulls = 0;
        // Luma of every output frame, for measuring motion and stability offline.
        let mut dump = if dump_path.is_empty() {
            None
        } else {
            Some(std::io::BufWriter::new(std::fs::File::create(&dump_path).map_err(|e| format!("cannot write {dump_path}: {e}"))?))
        };
        // Every output frame as interleaved 8-bit RGB, for looking at players and motion.
        let mut rgb = if rgb_path.is_empty() {
            None
        } else {
            Some(std::io::BufWriter::new(std::fs::File::create(&rgb_path).map_err(|e| format!("cannot write {rgb_path}: {e}"))?))
        };
        let mut luma = vec![0u8; (renderer.width() * renderer.height()) as usize];
        let mut interleaved = Vec::new();
        let t0 = Instant::now();
        let clock0 = engine_clock();
        for f in 0..frames {
            // Players are posed on the engine clock, so a bench that should show them moving
            // as they do on the stream keeps each frame in its slot rather than running ahead.
            if realtime {
                let slot = clock0 + f as f64 / cfg.fps as f64;
                let wait = slot - engine_clock();
                if wait > 0.0 {
                    std::thread::sleep(Duration::from_secs_f64(wait));
                }
            }
            let f0 = Instant::now();
            let shown_at = if realtime { clock0 + f as f64 / cfg.fps as f64 } else { engine_clock() };
            scene.frame(&mut renderer, &mut entities, base + f as f64 / cfg.fps as f64, shown_at);
            let ms = f0.elapsed().as_secs_f64() * 1000.0;
            let sums = if renderer.stats.full {
                full_ms += ms;
                fulls += 1;
                &mut pass_full
            } else {
                lite_ms += ms;
                &mut pass_lite
            };
            for (sum, t) in sums.iter_mut().zip(renderer.stats.passes) {
                *sum += t;
            }
            setup += renderer.stats.setup;
            raster += renderer.stats.raster;
            shade += renderer.stats.shade;
            post += renderer.stats.post;
            if let Some(d) = &mut dump {
                let [r, g, b] = renderer.output();
                for (i, l) in luma.iter_mut().enumerate() {
                    *l = ((54 * r[i] as u32 + 183 * g[i] as u32 + 19 * b[i] as u32) >> 8) as u8;
                }
                d.write_all(&luma).map_err(|e| format!("cannot write {dump_path}: {e}"))?;
            }
            if let Some(d) = &mut rgb {
                let [r, g, b] = renderer.output();
                interleaved.clear();
                interleaved.extend(r.iter().zip(g).zip(b).flat_map(|((&r, &g), &b)| [r, g, b]));
                d.write_all(&interleaved).map_err(|e| format!("cannot write {rgb_path}: {e}"))?;
            }
        }
        if let Some(mut d) = rgb {
            d.flush().map_err(|e| format!("cannot write {rgb_path}: {e}"))?;
        }
        if let Some(mut d) = dump {
            d.flush().map_err(|e| format!("cannot write {dump_path}: {e}"))?;
        }
        let total = t0.elapsed().as_secs_f64() * 1000.0;
        let n = frames as f64;
        let s = &renderer.stats;
        eprintln!(
            "{frames} frames, {:.1} ms/frame (setup {:.1}, raster {:.1}, shade {:.1}, post {:.1}), {} prims, {} sections drawn, {} occluded",
            total / n,
            setup / n,
            raster / n,
            shade / n,
            post / n,
            s.prims,
            s.sections,
            s.occluded
        );
        eprintln!(
            "full frames {:.1} ms avg ({fulls}), in-between {:.1} ms avg ({})",
            if fulls > 0 { full_ms / fulls as f64 } else { 0.0 },
            if frames > fulls { lite_ms / (frames - fulls) as f64 } else { 0.0 },
            frames - fulls
        );
        let lites = (frames - fulls).max(1) as f64;
        for (i, name) in render::PASSES.iter().enumerate() {
            eprintln!("  {name:>15} {:6.2} ms full {:6.2} ms in-between", pass_full[i] / fulls.max(1) as f64, pass_lite[i] / lites);
        }
        if mode == "snap" {
            let mut image = renderer.output().clone();
            if let Some(o) = &mut overlay {
                // Two draws a few seconds apart, so the now playing card has finished fading in.
                o.draw(&mut image, base);
                image.clone_from(renderer.output());
                o.draw(&mut image, base + 5.0);
            }
            write_png(&snap_path, renderer.width(), renderer.height(), &image)?;
        }
        return Ok(0);
    }

    stream::run(&cfg, &mut scene, &mut renderer, &mut entities, overlay.as_mut(), &jobs, &feed, &epoch_file, prerecord);
    Ok(0)
}

struct Scene {
    cfg: Arc<Config>,
    slot: WorldSlot,
    feed: Arc<FeedServer>,
    shadows: Shadows,
    ready: Arc<AtomicBool>,
    // What the world worker changed, for the stream to tell which recordings it spoils.
    changes: Arc<Changes>,
    // What this run keeps of its world for the next one, and what the last one kept.
    keep: Arc<Keep>,
    world: Option<(Arc<World>, Arc<Mesher>)>,
    worker: Option<WorldWorker>,
    path: OrbitPath,
    previous: OrbitPath,
    planned: f64,
    have_plan: bool,
    dynamic: Vec<Quad>,
    tags: NameTags,
    tag_fades: HashMap<String, TagFade>,
    entity_delay: f64,
    // Seconds from an edit reaching the engine to the frame showing it going out.
    latencies: Vec<f64>,
    // The world alone, no players, shadows of them or name tags: what a recording holds.
    pub bare: bool,
    // Whether the world's meshes may go for now, and when anyone was last online.
    rest: Arc<Rest>,
    people_at: f64,
}

impl Scene {
    /// Starts the world worker once the observer has said hello; true once terrain is lit,
    /// meshed and shadowed.
    fn ready(&mut self) -> bool {
        if self.world.is_none() {
            self.world = self.slot.lock().unwrap().clone();
            if let Some((world, mesher)) = &self.world {
                self.worker = Some(WorldWorker::start(world.clone(), mesher.clone(), self.shadows.clone(), self.feed.clone(),
                                                      self.cfg.center, self.ready.clone(), self.changes.clone(),
                                                      if self.cfg.replay { self.cfg.nature as f64 } else { 0.0 }, self.keep.clone(),
                                                      self.rest.clone()));
            }
        }
        self.ready.load(Ordering::SeqCst)
    }

    // The orbit is replanned as the world changes; a new plan is blended in over a few
    // seconds, so a block placed on the path moves the camera rather than cutting it.
    fn path_now(&mut self) -> OrbitPath {
        let now = engine_clock();
        let cfg = &*self.cfg;
        // Not while the world rests: the camera stays on its path until it can be drawn.
        if let Some((world, _)) = &self.world
            && (!self.have_plan || now - self.planned > 20.0)
            && !self.rest.asleep()
        {
            if self.have_plan {
                self.previous = std::mem::replace(&mut self.path, plan_orbit(cfg, world));
            } else {
                self.path = plan_orbit(cfg, world);
                self.previous = self.path.clone();
                let closest = self.path.radius.iter().copied().fold(f32::INFINITY, fmin);
                let highest = self.path.height.iter().copied().fold(f32::NEG_INFINITY, fmax);
                let pulled = self.path.radius.iter().filter(|&&r| r < cfg.radius - 0.5).count();
                eprintln!(
                    "orbit: pulled in over {:.0} degrees (closest {:.1} of {:.0}), eye up to {:.1} (from {:.0})",
                    pulled as f64 * 360.0 / ORBIT_N as f64,
                    closest,
                    cfg.radius,
                    highest,
                    cfg.orbit_height
                );
            }
            self.planned = now;
            self.have_plan = true;
        }
        let k = ((now - self.planned) / 5.0).clamp(0.0, 1.0) as f32;
        let mut blend = OrbitPath::zero();
        for i in 0..ORBIT_N {
            blend.radius[i] = self.previous.radius[i] + (self.path.radius[i] - self.previous.radius[i]) * k;
            blend.height[i] = self.previous.height[i] + (self.path.height[i] - self.previous.height[i]) * k;
        }
        blend
    }

    /// Renders the frame shown at unix_seconds, which is `shown_at` on the engine clock.
    fn frame(&mut self, renderer: &mut Renderer, entities: &mut Entities, unix_seconds: f64, shown_at: f64) {
        let (pose, _) = self.prepare(renderer, entities, unix_seconds, shown_at);
        self.draw(renderer, entities, &pose, unix_seconds, shown_at);
    }

    /// Poses the camera and everything that moves for the frame shown at unix_seconds, and
    /// returns where each player is, as a box around its body and name tag.
    fn prepare(&mut self, renderer: &mut Renderer, entities: &mut Entities, unix_seconds: f64, shown_at: f64)
               -> (CameraPose, Vec<[Vec3; 8]>) {
        let path = self.path_now();
        let pose = orbit_pose(&self.cfg, unix_seconds, &path);
        self.dynamic.clear();
        if self.bare {
            renderer.set_casters(Vec::new());
            return (pose, Vec::new());
        }
        let posed = shown_at - self.entity_delay;
        let present = self.feed.entities(posed);
        if !present.is_empty() {
            self.people_at = shown_at;
            // Someone is here: the world's meshes come back if they went, and until they
            // have, nobody is drawn.
            if !self.rest.need() {
                renderer.set_casters(Vec::new());
                return (pose, Vec::new());
            }
        }
        let world = self.world.as_ref().map(|(w, _)| &**w);
        entities.build(&present, posed, world, &mut self.dynamic);
        renderer.set_casters(entities.footprints().iter().map(|f| Footprint { pos: f.pos }).collect());
        // Two blocks round the feet hold a player lying flat on an elytra, and 2.8 up the
        // name tag's anchor.
        let players = entities
            .footprints()
            .iter()
            .map(|f| sight::corners(f.pos - Vec3::new(2.0, 0.5, 2.0), f.pos + Vec3::new(2.0, 2.8, 2.0)))
            .collect();
        (pose, players)
    }

    /// The stream, once a frame: whether it will only replay for now (`idle`). After a
    /// minute of that with no one online, the world's meshes may go.
    fn rest(&mut self, idle: bool, shown_at: f64) {
        if idle && shown_at - self.people_at > 60.0 {
            self.rest.allow();
        } else {
            self.rest.need();
        }
    }

    /// Whether the world rests: its shadow maps can go too.
    fn resting(&self) -> bool {
        self.rest.asleep()
    }

    /// Waits for the world's meshes, if they went: drawing without them would show (and
    /// record) a world with nothing in it. The stream asks for them back before it draws,
    /// so this only waits after a change of plan it did not see coming.
    fn wake(&self) {
        if self.rest.need() {
            return;
        }
        let t0 = Instant::now();
        while !self.rest.need() && !quitting() {
            std::thread::sleep(Duration::from_millis(10));
        }
        eprintln!("world: waited {:.0} ms for the meshes to come back", t0.elapsed().as_secs_f64() * 1000.0);
    }

    /// Renders what `prepare` posed.
    fn draw(&mut self, renderer: &mut Renderer, entities: &Entities, pose: &CameraPose, unix_seconds: f64, shown_at: f64) {
        self.wake();
        let meshes = match &self.world {
            Some((_, m)) => m.snapshot_with(|| renderer.adopt()),
            None => Vec::new(),
        };
        renderer.render(&meshes, &self.dynamic, pose, unix_seconds.rem_euclid(self.cfg.period as f64));
        if !self.bare {
            self.draw_tags(renderer, entities, shown_at);
        }
        self.latencies.extend(renderer.take_shown().into_iter().map(|arrived| shown_at - arrived));
    }
}

impl Scene {
    /// A name tag over every player: where the game puts it, half a block above the head,
    /// dimmed while terrain hides the player or it sneaks, and the nearest drawn last.
    ///
    /// Hidden means most of the body has been behind solid blocks for a while, found by
    /// walking the world's blocks towards the camera. It used to be one point on the chest
    /// against the last full render's depth, from a camera that had since moved: bamboo,
    /// flowers and the edges of steps crossing that point turned a tag dark and bright
    /// again several times a second. And even measured exactly, spawn's terraces hide a
    /// walking player's body for a moment every few steps as the camera goes round, so a
    /// tag only dims once its player has stayed hidden (in a house, behind a hill), and
    /// fades rather than switching.
    fn draw_tags(&mut self, renderer: &mut Renderer, entities: &Entities, time: f64) {
        let tags = self.tags_now(renderer, entities, time);
        let (w, h) = (renderer.width(), renderer.height());
        self.tags.draw(renderer.output_mut(), w, h, &tags);
    }

    /// Draws the players and their name tags over `yuv`, a recorded frame of the orbit
    /// (planar 4:2:0), instead of rendering the frame: what `prepare` posed, at unix_seconds.
    /// Where overlay mode will draw for the frame at unix_seconds, as `prepare` posed it:
    /// rectangles of pixels, and the name tags.
    /// Two sets of rectangles: where players and their shadows are drawn, which needs the
    /// scene's depth, and where anything is drawn, name tags included, which is all that
    /// changes in the frame.
    fn overlay_plan(&mut self, renderer: &mut Renderer, entities: &Entities, pose: &CameraPose, unix_seconds: f64, shown_at: f64)
                    -> (Vec<[i32; 4]>, Vec<[i32; 4]>, Vec<Tag>) {
        let time = unix_seconds.rem_euclid(self.cfg.period as f64);
        let bodies = renderer.overlay_reach(&self.dynamic, pose, time);
        let tags = self.tags_now(renderer, entities, shown_at);
        let mut all = bodies.clone();
        all.extend(tags.iter().map(|t| self.tags.bounds(t)));
        (bodies, render::merge_rects(all, renderer.width(), renderer.height()), tags)
    }

    /// Draws what overlay_plan planned over `yuv`, the recorded frame (planar 4:2:0).
    #[allow(clippy::too_many_arguments)]
    fn overlay(&mut self, renderer: &mut Renderer, pose: &CameraPose, unix_seconds: f64, shown_at: f64, yuv: &mut [u8],
               bodies: &[[i32; 4]], rects: &[[i32; 4]], tags: &[Tag]) {
        let meshes = match &self.world {
            Some((_, m)) => m.snapshot_with(|| renderer.adopt()),
            None => Vec::new(),
        };
        let time = unix_seconds.rem_euclid(self.cfg.period as f64);
        let (w, h) = (renderer.width(), renderer.height());
        // Small rectangles, many passes over them: on this thread, without waking the pool.
        crate::jobs::here(|| {
            for &rect in rects {
                replay::yuv_rect_to_rgb(yuv, w as usize, h as usize, rect, renderer.output_mut());
            }
            for &rect in bodies {
                renderer.overlay_draw(&meshes, &self.dynamic, pose, time, rect);
            }
        });
        renderer.overlay_done();
        self.tags.draw(renderer.output_mut(), w, h, tags);
        for &rect in rects {
            replay::rgb_rect_to_yuv(renderer.output(), w as usize, h as usize, rect, yuv);
        }
        self.latencies.extend(renderer.take_shown().into_iter().map(|arrived| shown_at - arrived));
    }

    /// The name tags to draw now, nearest last.
    fn tags_now(&mut self, renderer: &Renderer, entities: &Entities, time: f64) -> Vec<Tag> {
        const FADE: f32 = 0.3;
        // Seconds a player must stay hidden before its tag dims, and in view before it clears.
        const TO_HIDE: f64 = 1.5;
        const TO_SHOW: f64 = 0.5;
        let eye = renderer.eye();
        let mut tags: Vec<(f32, Tag)> = Vec::new();
        for l in entities.labels() {
            // The game anchors the tag half a block above the player's height and hangs it
            // from there, facing the camera, a font pixel being 0.025 blocks. Enlarged, it
            // grows upwards from where its bottom edge would be.
            let anchor = l.pos + Vec3::new(0.0, l.height, 0.0);
            let Some((x, y, dist)) = renderer.project(anchor) else { continue };
            let font_pixel = 0.025 * renderer.pixels_per_block(dist);
            let behind = self.world.as_ref().map_or(0.0, |(world, _)| {
                BODY.iter().filter(|&&o| world.hidden(l.pos + o, eye)).count() as f32 / BODY.len() as f32
            });
            let fade = self.tag_fades.entry(l.name.clone()).or_insert(TagFade { hidden: behind >= 0.75, since: -1.0, shown: -1.0, at: time });
            // Hysteresis in space as well as time, so a player half behind something does not
            // chatter between the two.
            let flip = if fade.hidden { behind <= 0.45 } else { behind >= 0.75 };
            if !flip {
                fade.since = -1.0;
            } else if fade.since < 0.0 {
                fade.since = time;
            } else if time - fade.since >= if fade.hidden { TO_SHOW } else { TO_HIDE } {
                fade.hidden = !fade.hidden;
                fade.since = -1.0;
            }
            let target = if fade.hidden || l.sneaking { 0.0 } else { 1.0 };
            let step = ((time - fade.at) as f32 / FADE).max(0.0);
            fade.shown = if fade.shown < 0.0 { target } else { fade.shown + (target - fade.shown).clamp(-step, step) };
            fade.at = time;
            let tag = Tag { x, y: y + 9.0 * font_pixel, scale: (font_pixel.round() as i32).clamp(2, 6), text: l.name.clone(),
                            shown: fade.shown };
            tags.push((dist, tag));
        }
        // A player gone for a while starts afresh when it comes back.
        self.tag_fades.retain(|_, f| time - f.at < 5.0);
        tags.sort_by(|a, b| b.0.total_cmp(&a.0));
        tags.into_iter().map(|(_, t)| t).collect()
    }
}

/// Where a name tag stands: whether its player counts as hidden, since when it has looked
/// otherwise (-1: it has not), and how far its text has faded in, as of `at` on the engine
/// clock.
struct TagFade {
    hidden: bool,
    since: f64,
    shown: f32,
    at: f64,
}

/// Points over a standing player's body, feet at the origin, that decide whether terrain
/// hides it: four heights, the middle and the four sides. The lowest is well clear of the
/// feet, which the block in front hides from above wherever the terrain steps.
const BODY: [Vec3; 20] = {
    let mut p = [Vec3::new(0.0, 0.0, 0.0); 20];
    let heights = [0.6, 1.0, 1.4, 1.8];
    let sides = [(0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)];
    let mut i = 0;
    while i < 20 {
        p[i] = Vec3::new(sides[i % 5].0, heights[i / 5], sides[i % 5].1);
        i += 1;
    }
    p
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_changes_wait_and_players_do_not() {
        let at = |x: i32, y: i32, z: i32, arrived: f64| world::Edit { lo: [x, y, z], hi: [x, y, z], arrived };
        let player = || vec![Vec3::new(0.5, 70.0, 0.5)];
        let mut n = Nature::new(600.0);
        // A player's own edit shows; grass eaten 40 blocks off waits.
        let shown = n.sort(vec![at(2, 70, 1, 0.0), at(40, 70, -24, 0.0)], player, 0.0);
        assert_eq!(shown.len(), 1);
        assert_eq!(n.held.len(), 1);
        assert!(n.sort(Vec::new(), player, 300.0).is_empty());
        // A player building next to held changes takes them along.
        n.sort(vec![at(100, 70, 100, 310.0)], player, 310.0);
        let beside = n.sort(vec![at(41, 70, -24, 320.0)], || vec![Vec3::new(41.0, 70.0, -22.0)], 320.0);
        assert_eq!(beside.len(), 2);
        assert_eq!(n.held.len(), 1);
        // The rest shows once it has waited long enough.
        assert!(n.sort(Vec::new(), player, 900.0).is_empty());
        assert_eq!(n.sort(Vec::new(), player, 910.0).len(), 1);
        assert!(n.held.is_empty());
    }

    #[test]
    fn yuv_matches_the_fixed_point_formula() {
        let (w, h) = (64usize, 6usize);
        let mut seed = 7u32;
        let rgb: [Vec<u8>; 3] = std::array::from_fn(|_| {
            (0..w * h)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect()
        });
        let mut yuv = vec![0u8; w * h * 3 / 2];
        to_yuv420(&rgb, &mut yuv, w as i32, h as i32, &Jobs::new(1));
        let [pr, pg, pb] = &rgb;
        for i in 0..w * h {
            let (r, g, b) = (pr[i] as i32, pg[i] as i32, pb[i] as i32);
            assert_eq!(yuv[i] as i32, 16 + ((11966 * r + 40254 * g + 4064 * b + 32768) >> 16), "luma {i}");
        }
        for cy in 0..h / 2 {
            for cx in 0..w / 2 {
                let at = |p: &[u8], dy: usize, dx: usize| p[(2 * cy + dy) * w + 2 * cx + dx] as i32;
                let sum = |p: &[u8]| at(p, 0, 0) + at(p, 0, 1) + at(p, 1, 0) + at(p, 1, 1);
                let (r, g, b) = (sum(pr), sum(pg), sum(pb));
                let u = 128 + ((-6597 * r - 22187 * g + 28784 * b + (1 << 17)) >> 18);
                let v = 128 + ((28784 * r - 26147 * g - 2637 * b + (1 << 17)) >> 18);
                let k = cy * w / 2 + cx;
                assert!((yuv[w * h + k] as i32 - u).abs() <= 1, "u {cx},{cy}");
                assert!((yuv[w * h + w * h / 4 + k] as i32 - v).abs() <= 1, "v {cx},{cy}");
            }
        }
    }
}
