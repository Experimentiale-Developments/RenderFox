//! Turns the observer's entity feed into textured quads each frame: players and
//! mannequins with their skins, armour, elytra and animation. Other entity types are left
//! out rather than drawn wrong.

use crate::assets::{Assets, CUTOUT, MAT_ENTITY, SOLID, val_str};
use crate::mesh::{QUAD_FLAT, Quad};
use crate::net::{EF_SNEAKING, EntitySample, EntityState, POSE_FALL_FLYING};
use crate::vmath::*;
use crate::world::World;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f32::consts::PI;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

const PX: f32 = 0.9375 / 16.0; // players render at 15/16 scale, 16 texels to the block

/// Rotation matrix as three columns, plus translation.
#[derive(Clone, Copy)]
struct Xf {
    c0: Vec3,
    c1: Vec3,
    c2: Vec3,
    t: Vec3,
}

impl Xf {
    const IDENTITY: Xf = Xf {
        c0: Vec3::new(1.0, 0.0, 0.0),
        c1: Vec3::new(0.0, 1.0, 0.0),
        c2: Vec3::new(0.0, 0.0, 1.0),
        t: Vec3::ZERO,
    };

    fn apply(&self, p: Vec3) -> Vec3 {
        self.c0 * p.x + self.c1 * p.y + self.c2 * p.z + self.t
    }

    fn dir(&self, d: Vec3) -> Vec3 {
        self.c0 * d.x + self.c1 * d.y + self.c2 * d.z
    }
}

impl std::ops::Mul for Xf {
    type Output = Xf;
    fn mul(self, o: Xf) -> Xf {
        Xf { c0: self.dir(o.c0), c1: self.dir(o.c1), c2: self.dir(o.c2), t: self.apply(o.t) }
    }
}

/// Applied z, then y, then x, like the game's ModelPart (ZYX Euler).
fn rotation(rx: f32, ry: f32, rz: f32) -> Xf {
    let (cx, sx, cy, sy, cz, sz) = (rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin());
    let mut x = Xf::IDENTITY;
    let mut y = Xf::IDENTITY;
    let mut z = Xf::IDENTITY;
    x.c1 = Vec3::new(0.0, cx, sx);
    x.c2 = Vec3::new(0.0, -sx, cx);
    y.c0 = Vec3::new(cy, 0.0, -sy);
    y.c2 = Vec3::new(sy, 0.0, cy);
    z.c0 = Vec3::new(cz, sz, 0.0);
    z.c1 = Vec3::new(-sz, cz, 0.0);
    x * y * z
}

fn translation(t: Vec3) -> Xf {
    Xf { t, ..Xf::IDENTITY }
}

/// Rotate about a pivot in model pixels.
fn pivoted(pivot: Vec3, rx: f32, ry: f32, rz: f32) -> Xf {
    translation(pivot) * rotation(rx, ry, rz) * translation(pivot * -1.0)
}

fn wrap_degrees(a: f32) -> f32 {
    let mut a = a % 360.0;
    if a >= 180.0 {
        a -= 360.0;
    }
    if a < -180.0 {
        a += 360.0;
    }
    a
}

#[derive(Clone, Copy)]
struct BoxSpec {
    from: Vec3, // model pixels, un-inflated
    to: Vec3,
    u: i32, // texture offset
    v: i32,
    inflate: f32,
    overlay: bool, // alpha tested second layer
    mirror: bool,  // legacy left limbs reuse the right limb's texture, flipped
}

#[allow(clippy::too_many_arguments)]
fn spec(from: (f32, f32, f32), to: (f32, f32, f32), u: i32, v: i32, inflate: f32, overlay: bool, mirror: bool) -> BoxSpec {
    BoxSpec {
        from: Vec3::new(from.0, from.1, from.2),
        to: Vec3::new(to.0, to.1, to.2),
        u,
        v,
        inflate,
        overlay,
        mirror,
    }
}

#[derive(Clone, Copy, Default)]
struct Anim {
    last_time: f64,
    last_pos: Vec3,
    phase: f32,
    amount: f32,
    velocity: Vec3, // blocks a tick, smoothed
    flying: f32,    // ticks gliding on an elytra, 0 when not
    wings: Vec3,    // the elytra's rotation about x, y and z, easing to where the pose wants it
}

#[derive(Clone, Copy, Default)]
struct Skin {
    tex: u16,
    slim: bool,
    legacy: bool,
    pending: bool,
    cape: u16, // 0 for none
    cape_pending: bool,
}

/// Ground position of a player drawn this frame, for contact shadows.
#[derive(Clone, Copy)]
pub struct Footprint {
    pub pos: Vec3,
}

/// A player's name for its tag: where the player stands this frame, how far above that
/// the tag hangs, and whether sneaking.
pub struct Label {
    pub pos: Vec3,
    pub height: f32,
    pub name: String,
    pub sneaking: bool,
}

#[derive(Default)]
struct Downloads {
    queue: VecDeque<(String, String)>, // key, url
    requested: HashSet<String>,
    downloaded: HashMap<String, String>, // key -> file
    stop: bool,
}

struct Downloader {
    state: Mutex<Downloads>,
    wake: Condvar,
}

pub struct Entities {
    assets: Arc<Assets>,
    anims: HashMap<i32, Anim>,
    skins: HashMap<String, Skin>,
    armour: HashMap<String, u16>,
    footprints: Vec<Footprint>,
    labels: Vec<Label>,
    downloader: Arc<Downloader>,
    thread: Option<JoinHandle<()>>,
}

impl Entities {
    pub fn new(assets: Arc<Assets>, cache_dir: &str) -> Entities {
        if let Err(e) = std::fs::create_dir_all(cache_dir) {
            eprintln!("skins: cannot make {cache_dir} ({e}): every player will wear a default skin");
        }
        let downloader = Arc::new(Downloader { state: Mutex::new(Downloads::default()), wake: Condvar::new() });
        let d = downloader.clone();
        let dir = cache_dir.to_string();
        let thread = std::thread::Builder::new()
            .name("skins".into())
            .spawn(move || download_loop(&d, &dir))
            .expect("spawn skin downloader");
        Entities {
            assets,
            anims: HashMap::new(),
            skins: HashMap::new(),
            armour: HashMap::new(),
            footprints: Vec::new(),
            labels: Vec::new(),
            downloader,
            thread: Some(thread),
        }
    }

    pub fn footprints(&self) -> &[Footprint] {
        &self.footprints
    }

    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn skin_for(&mut self, e: &EntityState) -> Skin {
        let key = &e.skin;
        let cape_key = format!("{key}#cape");
        if let Some(s) = self.skins.get_mut(key) {
            if !s.pending && !s.cape_pending {
                return *s;
            }
            // Showing a stand-in while the real skin downloads: swap it in once it lands,
            // and put the cape on when that does.
            let (skin_file, cape_file) = {
                let d = self.downloader.state.lock().unwrap();
                (d.downloaded.get(key).filter(|_| s.pending).cloned(), d.downloaded.get(&cape_key).filter(|_| s.cape_pending).cloned())
            };
            if let Some(file) = skin_file {
                let tex = self.assets.texture_from_file(&format!("skin:{file}"), &file, true);
                if tex != 0 {
                    s.tex = tex;
                    s.legacy = self.assets.tex(tex).h == 32;
                }
                s.pending = false;
            }
            if let Some(file) = cape_file {
                s.cape = self.assets.texture_from_file(&format!("cape:{file}"), &file, true);
                s.cape_pending = false;
            }
            return *s;
        }
        // Vanilla DefaultPlayerSkin: nine characters, slim then wide, picked by UUID hash.
        let assets = &self.assets;
        let fallback = |idx: i32| -> Skin {
            const NAMES: [&str; 9] = ["alex", "ari", "efe", "kai", "makena", "noor", "steve", "sunny", "zuri"];
            let slim = idx < 9;
            let name = format!("entity/player/{}{}", if slim { "slim/" } else { "wide/" }, NAMES[(idx % 9) as usize]);
            let file = assets.path(&format!("assets/minecraft/textures/{name}.png"));
            Skin { tex: assets.texture_from_file(&format!("minecraft:{name}"), &file, true), slim, ..Skin::default() }
        };
        const STEVE: i32 = 9 + 6;
        let skin = if let Some(uuid) = key.strip_prefix("uuid:") {
            // java.util.UUID.hashCode(): the xor of its four 32-bit words.
            let hex: String = uuid.chars().filter(|&c| c != '-').collect();
            let x = if hex.len() == 32 {
                u64::from_str_radix(&hex[..16], 16).unwrap_or(0) ^ u64::from_str_radix(&hex[16..], 16).unwrap_or(0)
            } else {
                0
            };
            let h = ((x >> 32) as u32 ^ x as u32) as i32;
            fallback(((h % 18) + 18) % 18)
        } else if key.is_empty() {
            fallback(STEVE)
        } else {
            let (mut url, mut cape_url) = (String::new(), String::new());
            let mut slim = false;
            if let Ok(j) = serde_json::from_slice::<serde_json::Value>(&crate::base64::decode(key))
                && let Some(t) = j.get("textures")
            {
                if let Some(s) = t.get("SKIN").filter(|s| s.is_object()) {
                    url = val_str(s, "url", "");
                    slim = s.get("metadata").is_some_and(|m| val_str(m, "model", "") == "slim");
                }
                if let Some(c) = t.get("CAPE").filter(|c| c.is_object()) {
                    cape_url = val_str(c, "url", "");
                }
            }
            let mut skin = fallback(STEVE);
            let mut d = self.downloader.state.lock().unwrap();
            for (k, u) in [(key.clone(), url), (cape_key, cape_url)] {
                if !u.is_empty() && d.requested.insert(k.clone()) {
                    d.queue.push_back((k, u));
                    self.downloader.wake.notify_one();
                }
            }
            skin.slim = slim;
            skin.pending = d.requested.contains(key);
            skin.cape_pending = d.requested.contains(&format!("{key}#cape"));
            skin
        };
        self.skins.insert(key.clone(), skin);
        skin
    }

    fn elytra_texture(&mut self) -> u16 {
        const KEY: &str = "minecraft:entity/equipment/wings/elytra";
        if let Some(&id) = self.armour.get(KEY) {
            return id;
        }
        let path = self.assets.path("assets/minecraft/textures/entity/equipment/wings/elytra.png");
        let id = if std::path::Path::new(&path).exists() { self.assets.texture_from_file(KEY, &path, false) } else { 0 };
        self.armour.insert(KEY.into(), id);
        id
    }

    fn armour_texture(&mut self, item: &str, leggings: bool) -> u16 {
        let key = format!("{item}{}", if leggings { "|legs" } else { "" });
        if let Some(&id) = self.armour.get(&key) {
            return id;
        }
        let n = item.find(':').map_or(item, |c| &item[c + 1..]);
        let mut material = n.rfind('_').map_or(n, |u| &n[..u]).to_string();
        if material == "golden" {
            material = "gold".into();
        }
        if n == "turtle_helmet" {
            material = "turtle_scute".into();
        }
        let rel = format!("entity/equipment/{}{material}", if leggings { "humanoid_leggings/" } else { "humanoid/" });
        let path = self.assets.path(&format!("assets/minecraft/textures/{rel}.png"));
        let id = if std::path::Path::new(&path).exists() {
            self.assets.texture_from_file(&format!("minecraft:{rel}"), &path, false)
        } else {
            0
        };
        self.armour.insert(key, id);
        id
    }

    /// Poses everything as it was at `now` on the engine clock, which the caller keeps
    /// far enough behind the feed that positions on both sides of it have arrived.
    pub fn build(&mut self, states: &[EntityState], now: f64, world: Option<&World>, out: &mut Vec<Quad>) {
        self.footprints.clear();
        self.labels.clear();
        for e in states {
            if e.kind != "minecraft:player" && e.kind != "minecraft:mannequin" {
                continue;
            }
            let ss = &e.samples;
            // Interpolate between the two samples around now.
            let mut a: EntitySample = ss[0];
            let mut b: EntitySample = ss[0];
            for s in ss {
                if s.time <= now {
                    a = *s;
                }
                if s.time >= now {
                    b = *s;
                    break;
                }
                b = *s;
            }
            // Past the newest sample (the feed hiccupped), carry on at its speed for up to a
            // tenth of a second rather than stop dead and then jump.
            if b.time < now && ss.len() > 1 {
                a = ss[ss.len() - 2];
            }
            let f = if b.time > a.time {
                ((now - a.time) / (b.time - a.time)).clamp(0.0, 1.0 + 0.1 / (b.time - a.time)) as f32
            } else {
                1.0
            };
            let pos = mix3(a.pos, b.pos, f);
            let angle = |x: f32, y: f32| x + wrap_degrees(y - x) * f;
            let (yaw, pitch, body_yaw) = (angle(a.yaw, b.yaw), mix(a.pitch, b.pitch, f), angle(a.body_yaw, b.body_yaw));
            let flags = b.flags;

            let gliding = b.pose == POSE_FALL_FLYING;
            let sneaking = flags & EF_SNEAKING != 0;
            let an = self.anims.entry(e.id).or_insert(Anim { last_time: -1.0, wings: Vec3::new(0.2617994, 0.0, -0.2617994), ..Anim::default() });
            let dt = if an.last_time < 0.0 { 0.0 } else { (now - an.last_time) as f32 };
            if dt > 0.0 && dt < 0.5 {
                let moved = (pos.x - an.last_pos.x).hypot(pos.z - an.last_pos.z);
                let per_tick = moved / (dt * 20.0);
                let target = fmin(1.0, per_tick * 4.0);
                an.amount += (target - an.amount) * fmin(1.0, dt * 8.0);
                an.phase += an.amount * dt * 20.0;
                an.velocity = mix3(an.velocity, (pos - an.last_pos) * (1.0 / (dt * 20.0)), fmin(1.0, dt * 10.0));
                an.flying = if gliding { an.flying + dt * 20.0 } else { 0.0 };
                // The game's ElytraAnimationState: spread flat while gliding level, swept
                // back as the dive steepens, and folded in when sneaking; eased 0.3 a tick.
                let (mut want, spread) = (Vec3::new(0.2617994, 0.0, -0.2617994), 1.0f32);
                if gliding {
                    let v = an.velocity;
                    let spread = if v.y < 0.0 && length(v) > 1e-4 { 1.0 - (-v.y / length(v)).powf(1.5) } else { spread };
                    want = Vec3::new(mix(0.2617994, 0.34906584, spread), 0.0, mix(-0.2617994, -PI / 2.0, spread));
                } else if sneaking {
                    want = Vec3::new(0.6981317, 0.08726646, -PI / 4.0);
                }
                an.wings = mix3(an.wings, want, 1.0 - 0.7f32.powf(dt * 20.0));
            }
            an.last_time = now;
            an.last_pos = pos;
            let (phase, velocity, wings, flying) = (an.phase, an.velocity, an.wings, an.flying);
            // The game eases into the glide over its first ten ticks, and holds the limbs
            // still in proportion to the speed.
            let glide = fmin(1.0, flying * flying / 100.0);
            let limbs = if flying > 4.0 { fmax(1.0, (dot(velocity, velocity) / 0.2).powi(3)) } else { 1.0 };
            let amt = an.amount / limbs;

            let skin = self.skin_for(e);
            if skin.tex == 0 {
                continue;
            }
            let swing = ((now - e.last_swing) / 0.3) as f32;
            let swing_off = ((now - e.last_swing_off) / 0.3) as f32;
            let hurt = now - e.last_hurt < 0.4;

            let head_yaw = wrap_degrees(yaw - body_yaw).clamp(-75.0, 75.0) * PI / 180.0;
            // Gliding, the head looks along the flight rather than where the player does.
            let head_pitch = if flying > 4.0 { -PI / 4.0 } else { pitch * PI / 180.0 };
            let ph = phase * 0.6662;
            let r_leg = ph.cos() * 1.4 * amt;
            let l_leg = (ph + PI).cos() * 1.4 * amt;
            let mut r_arm = (ph + PI).cos() * amt;
            let mut l_arm = ph.cos() * amt;
            let idle = ((now * 1.8).cos() * 0.05 + 0.05) as f32;
            let mut body_twist = 0.0f32;
            if (0.0..1.0).contains(&swing) {
                let s = (swing * PI).sin();
                r_arm += s * 1.4;
                body_twist = (swing.sqrt() * 2.0 * PI).sin() * 0.2;
            }
            if (0.0..1.0).contains(&swing_off) {
                l_arm += (swing_off * PI).sin() * 1.4;
            }

            // Model space: pixels, feet at the origin, facing -z. World: rotate by body yaw.
            let phi = PI - body_yaw * PI / 180.0;
            let mut placed = translation(pos) * rotation(0.0, phi, 0.0);
            if gliding {
                // As the game's PlayerRenderer: tipped forward about the feet until lying
                // along the look, then rolled to where the flight is actually heading.
                placed = placed * rotation(glide * (-PI / 2.0 - pitch * PI / 180.0), 0.0, 0.0);
                let (sy, cy) = (yaw * PI / 180.0).sin_cos();
                let (lx, lz) = (-sy, cy);
                let (vx, vz) = (velocity.x, velocity.z);
                let (d0, d1) = (vx * vx + vz * vz, lx * lx + lz * lz);
                if d0 > 1e-6 && d1 > 1e-6 {
                    let d2 = ((vx * lx + vz * lz) / (d0 * d1).sqrt()).clamp(-1.0, 1.0);
                    let d3 = vx * lz - vz * lx;
                    placed = placed * rotation(0.0, d3.signum() * d2.acos(), 0.0);
                }
            }
            let scale = Xf {
                c0: Vec3::new(PX, 0.0, 0.0),
                c1: Vec3::new(0.0, PX, 0.0),
                c2: Vec3::new(0.0, 0.0, PX),
                t: Vec3::ZERO,
            };
            let mut base = placed * scale;
            let mut lean = Xf::IDENTITY; // upper body bend while sneaking, about the hips
            if sneaking {
                base = base * translation(Vec3::new(0.0, -2.5, 0.0));
                lean = pivoted(Vec3::new(0.0, 12.0, 0.0), -0.5, 0.0, 0.0);
            }
            let twist = pivoted(Vec3::new(0.0, 12.0, 0.0), 0.0, body_twist, 0.0);
            let upper = lean * twist;

            let light = world.map_or(0xf0, |w| {
                w.light_at(pos.x.floor() as i32, (pos.y + 1.0).floor() as i32, pos.z.floor() as i32)
            });
            let sky = ((light >> 4) as i32 * 17) as u8;
            let blk = ((light & 15) as i32 * 17) as u8;

            let emit = |out: &mut Vec<Quad>, xf: &Xf, b: BoxSpec, tex: u16, tex_w: i32, tex_h: i32, tint: u32| {
                let inf = Vec3::splat(b.inflate);
                let (lo, hi) = (b.from - inf, b.to + inf);
                let (dx, dy, dz) = (b.to.x - b.from.x, b.to.y - b.from.y, b.to.z - b.from.z);
                let (u, v) = (b.u as f32, b.v as f32);
                let (w, h, d) = (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
                // East and west swap sides when a limb's texture is mirrored.
                let east_u = if b.mirror { u + dz + dx } else { u };
                let west_u = if b.mirror { u } else { u + dz + dx };
                let v3 = Vec3::new;
                let faces: [(Vec3, Vec3, Vec3, f32, f32, f32, f32); 6] = [
                    (v3(hi.x, lo.y, lo.z), v3(-w, 0.0, 0.0), v3(0.0, h, 0.0), u + dz, v + dz, u + dz + dx, v + dz + dy), // north
                    (
                        v3(lo.x, lo.y, hi.z),
                        v3(w, 0.0, 0.0),
                        v3(0.0, h, 0.0),
                        u + 2.0 * dz + dx,
                        v + dz,
                        u + 2.0 * dz + 2.0 * dx,
                        v + dz + dy,
                    ), // south
                    (v3(hi.x, lo.y, hi.z), v3(0.0, 0.0, -d), v3(0.0, h, 0.0), east_u, v + dz, east_u + dz, v + dz + dy), // east
                    (v3(lo.x, lo.y, lo.z), v3(0.0, 0.0, d), v3(0.0, h, 0.0), west_u, v + dz, west_u + dz, v + dz + dy), // west
                    (v3(hi.x, hi.y, lo.z), v3(-w, 0.0, 0.0), v3(0.0, 0.0, d), u + dz, v, u + dz + dx, v + dz), // up
                    (v3(hi.x, lo.y, hi.z), v3(-w, 0.0, 0.0), v3(0.0, 0.0, -d), u + dz + dx, v, u + dz + 2.0 * dx, v + dz), // down
                ];
                let (tw, th) = (tex_w as f32, tex_h as f32);
                for (p, e1, e2, fu0, v0, fu1, v1) in faces {
                    let (mut u0, mut u1) = (fu0, fu1);
                    if b.mirror {
                        std::mem::swap(&mut u0, &mut u1);
                    }
                    let (qe1, qe2) = (xf.dir(e1), xf.dir(e2));
                    // p is the texture's bottom-left: (u0, v1); e1 runs along u, e2 up to v0.
                    out.push(Quad {
                        p: xf.apply(p),
                        e1: qe1,
                        e2: qe2,
                        n: normalize(cross(qe1, qe2)),
                        uv: Vec2::new(u0 / tw, v1 / th),
                        uv_s: Vec2::new((u1 - u0) / tw, 0.0),
                        uv_t: Vec2::new(0.0, (v0 - v1) / th),
                        tex,
                        tint,
                        mat: MAT_ENTITY,
                        layer: if b.overlay { CUTOUT } else { SOLID },
                        flags: QUAD_FLAT,
                        sky: [sky; 4],
                        blk: [blk; 4],
                        ao: [255; 4],
                        wave: 0,
                    });
                }
            };

            let st = self.assets.tex(skin.tex);
            let (tw, th) = (st.w, st.h);
            let tint: u32 = if hurt { 0xff8080 } else { 0xffffff };
            let arm_w = if skin.slim { 3.0 } else { 4.0 };
            let lift = if sneaking { 0.4 } else { 0.0 };
            let head = base * upper * pivoted(Vec3::new(0.0, 24.0, 0.0), -head_pitch, -head_yaw, 0.0);
            let body = base * upper;
            let right_arm = base * upper * pivoted(Vec3::new(5.0, 22.0, 0.0), r_arm + lift, 0.0, idle);
            let left_arm = base * upper * pivoted(Vec3::new(-5.0, 22.0, 0.0), l_arm + lift, 0.0, -idle);
            let leg_shift = if sneaking { translation(Vec3::new(0.0, 0.0, 4.0)) } else { Xf::IDENTITY };
            let right_leg = base * leg_shift * pivoted(Vec3::new(2.0, 12.0, 0.0), r_leg, 0.0, 0.0);
            let left_leg = base * leg_shift * pivoted(Vec3::new(-2.0, 12.0, 0.0), l_leg, 0.0, 0.0);

            let t = skin.tex;
            emit(out, &head, spec((-4.0, 24.0, -4.0), (4.0, 32.0, 4.0), 0, 0, 0.0, false, false), t, tw, th, tint);
            emit(out, &head, spec((-4.0, 24.0, -4.0), (4.0, 32.0, 4.0), 32, 0, 0.5, true, false), t, tw, th, tint);
            emit(out, &body, spec((-4.0, 12.0, -2.0), (4.0, 24.0, 2.0), 16, 16, 0.0, false, false), t, tw, th, tint);
            emit(out, &right_arm, spec((4.0, 12.0, -2.0), (4.0 + arm_w, 24.0, 2.0), 40, 16, 0.0, false, false), t, tw, th, tint);
            emit(out, &right_leg, spec((0.0, 0.0, -2.0), (4.0, 12.0, 2.0), 0, 16, 0.0, false, false), t, tw, th, tint);
            if skin.legacy {
                emit(out, &left_arm, spec((-4.0 - arm_w, 12.0, -2.0), (-4.0, 24.0, 2.0), 40, 16, 0.0, false, true), t, tw, th, tint);
                emit(out, &left_leg, spec((-4.0, 0.0, -2.0), (0.0, 12.0, 2.0), 0, 16, 0.0, false, true), t, tw, th, tint);
            } else {
                emit(out, &left_arm, spec((-4.0 - arm_w, 12.0, -2.0), (-4.0, 24.0, 2.0), 32, 48, 0.0, false, false), t, tw, th, tint);
                emit(out, &left_leg, spec((-4.0, 0.0, -2.0), (0.0, 12.0, 2.0), 16, 48, 0.0, false, false), t, tw, th, tint);
                emit(out, &body, spec((-4.0, 12.0, -2.0), (4.0, 24.0, 2.0), 16, 32, 0.25, true, false), t, tw, th, tint);
                emit(out, &right_arm, spec((4.0, 12.0, -2.0), (4.0 + arm_w, 24.0, 2.0), 40, 32, 0.25, true, false), t, tw, th, tint);
                emit(out, &left_arm, spec((-4.0 - arm_w, 12.0, -2.0), (-4.0, 24.0, 2.0), 48, 48, 0.25, true, false), t, tw, th, tint);
                emit(out, &right_leg, spec((0.0, 0.0, -2.0), (4.0, 12.0, 2.0), 0, 32, 0.25, true, false), t, tw, th, tint);
                emit(out, &left_leg, spec((-4.0, 0.0, -2.0), (0.0, 12.0, 2.0), 0, 48, 0.25, true, false), t, tw, th, tint);
            }

            // Armour: gear is main, off, head, chest, legs, feet.
            let gear = |i: usize| e.gear.get(i).cloned().unwrap_or_default();
            let armour_tint = |item: &str| {
                if item.contains("leather") {
                    if hurt { 0xc05030 } else { 0xa06540 }
                } else {
                    tint
                }
            };
            let it = gear(2);
            if !it.is_empty() {
                let at = self.armour_texture(&it, false);
                if at != 0 {
                    emit(out, &head, spec((-4.0, 24.0, -4.0), (4.0, 32.0, 4.0), 0, 0, 1.0, true, false), at, 64, 32, armour_tint(&it));
                }
            }
            let it = gear(3);
            if !it.is_empty() {
                let at = self.armour_texture(&it, false);
                if at != 0 {
                    let c = armour_tint(&it);
                    emit(out, &body, spec((-4.0, 12.0, -2.0), (4.0, 24.0, 2.0), 16, 16, 1.0, true, false), at, 64, 32, c);
                    emit(out, &right_arm, spec((4.0, 12.0, -2.0), (8.0, 24.0, 2.0), 40, 16, 1.0, true, false), at, 64, 32, c);
                    emit(out, &left_arm, spec((-8.0, 12.0, -2.0), (-4.0, 24.0, 2.0), 40, 16, 1.0, true, true), at, 64, 32, c);
                }
            }
            let it = gear(4);
            if !it.is_empty() {
                let at = self.armour_texture(&it, true);
                if at != 0 {
                    let c = armour_tint(&it);
                    emit(out, &body, spec((-4.0, 12.0, -2.0), (4.0, 24.0, 2.0), 16, 16, 0.5, true, false), at, 64, 32, c);
                    emit(out, &right_leg, spec((0.0, 0.0, -2.0), (4.0, 12.0, 2.0), 0, 16, 0.5, true, false), at, 64, 32, c);
                    emit(out, &left_leg, spec((-4.0, 0.0, -2.0), (0.0, 12.0, 2.0), 0, 16, 0.5, true, true), at, 64, 32, c);
                }
            }
            let it = gear(5);
            if !it.is_empty() {
                let at = self.armour_texture(&it, false);
                if at != 0 {
                    let c = armour_tint(&it);
                    emit(out, &right_leg, spec((0.0, 0.0, -2.0), (4.0, 12.0, 2.0), 0, 16, 1.0, true, false), at, 64, 32, c);
                    emit(out, &left_leg, spec((-4.0, 0.0, -2.0), (0.0, 12.0, 2.0), 0, 16, 1.0, true, true), at, 64, 32, c);
                }
            }
            // The cape hangs from the shoulders and swings back as the player walks. An
            // elytra takes its place, as in the game.
            let elytra = gear(3) == "minecraft:elytra";
            if skin.cape != 0 && !elytra {
                let tilt = (6.0 + 60.0 * amt + if sneaking { 25.0 } else { 0.0 }) * PI / 180.0;
                let cape = body * translation(Vec3::new(0.0, 24.0, 2.0)) * rotation(-tilt, PI, 0.0);
                emit(out, &cape, spec((-5.0, -16.0, -1.0), (5.0, 0.0, 0.0), 0, 0, 0.0, false, false), skin.cape, 64, 32, tint);
            }
            // The game's ElytraModel, two wings of 10 x 20 x 2 pixels hung from the shoulder
            // blades, two pixels behind the back. A cape's texture holds its own elytra.
            if elytra {
                let tex = if skin.cape != 0 { skin.cape } else { self.elytra_texture() };
                if tex != 0 {
                    let drop = if sneaking { 3.0 } else { 0.0 };
                    // The game's model space has x and y the other way round, which turns
                    // rotations about x and y round too.
                    let wing = |side: f32| {
                        base * translation(Vec3::new(5.0 * side, 24.0 - drop, 2.0))
                            * rotation(0.0, 0.0, -side * wings.z)
                            * rotation(0.0, side * wings.y, 0.0)
                            * rotation(-wings.x, 0.0, 0.0)
                    };
                    emit(out, &wing(-1.0), spec((0.0, -20.0, 0.0), (10.0, 0.0, 2.0), 22, 0, 1.0, true, false), tex, 64, 32, tint);
                    emit(out, &wing(1.0), spec((-10.0, -20.0, 0.0), (0.0, 0.0, 2.0), 22, 0, 1.0, true, true), tex, 64, 32, tint);
                }
            }
            self.footprints.push(Footprint { pos });
            if e.kind == "minecraft:player" && !e.name.is_empty() {
                // The game hangs the tag half a block over the player's box, which is 0.6
                // tall while gliding.
                let height = mix(if sneaking { 2.0 } else { 2.3 }, 1.1, glide);
                self.labels.push(Label { pos, height, name: e.name.clone(), sneaking });
            }
        }
        self.anims.retain(|_, a| now - a.last_time <= 5.0);
    }
}

impl Drop for Entities {
    fn drop(&mut self) {
        self.downloader.state.lock().unwrap().stop = true;
        self.downloader.wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn download_loop(d: &Downloader, cache_dir: &str) {
    loop {
        let (key, url) = {
            let mut st = d.state.lock().unwrap();
            while !st.stop && st.queue.is_empty() {
                st = d.wake.wait(st).unwrap();
            }
            if st.stop {
                return;
            }
            st.queue.pop_front().unwrap()
        };
        // Only Mojang's texture host, and only plain URL characters.
        let safe = (url.starts_with("http://textures.minecraft.net/texture/")
            || url.starts_with("https://textures.minecraft.net/texture/"))
            && url.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '/' | '.' | '_' | '-'));
        let name = url.rsplit('/').next().unwrap_or("");
        let file = format!("{cache_dir}/{name}.png");
        let mut ok = std::fs::metadata(&file).is_ok_and(|m| m.len() > 0);
        if !ok && safe {
            let part = format!("{file}.part");
            // curl is run directly, not through a shell, so nothing in the URL is interpreted.
            ok = std::process::Command::new("curl")
                .args(["-sf", "--max-time", "15", "-o", &part, &url])
                .status()
                .is_ok_and(|s| s.success())
                && std::fs::rename(&part, &file).is_ok();
        }
        if ok {
            d.state.lock().unwrap().downloaded.insert(key, file);
        } else {
            eprintln!("skins: could not fetch {url}; that player keeps a default skin until the engine restarts");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::Jobs;
    use crate::net::EntitySample;
    use crate::render::{CameraPose, Renderer};

    /// Four players with an elytra, seen from the side: standing, gliding level, diving and
    /// sneaking, to target/tmp/elytra.png for a look.
    #[test]
    #[ignore = "needs the client assets: RF_ASSETS=../assets/client cargo test --release -- --ignored elytra"]
    fn elytra_poses() {
        let assets = Arc::new(Assets::new(&std::env::var("RF_ASSETS").expect("RF_ASSETS")));
        let mut entities = Entities::new(assets.clone(), "target/tmp/skins");
        let mut renderer = Renderer::new(assets, Arc::new(Jobs::new(3)), 960, 544).unwrap();
        // Where each ends up, its speed in blocks a tick, pitch, pose and flags.
        let cast = [
            (Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 0.0, 0, 0),
            (Vec3::new(3.0, 0.5, 0.0), Vec3::new(1.5, 0.0, 0.0), 0.0, POSE_FALL_FLYING, 0),
            (Vec3::new(6.0, 0.5, 0.0), Vec3::new(1.0, -1.4, 0.0), 50.0, POSE_FALL_FLYING, 0),
            (Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO, 0.0, 0, EF_SNEAKING),
        ];
        let end = 1.5;
        let states: Vec<EntityState> = cast
            .iter()
            .enumerate()
            .map(|(i, &(at, v, pitch, pose, flags))| EntityState {
                id: i as i32,
                kind: "minecraft:player".into(),
                name: String::new(),
                skin: String::new(),
                gear: ["", "", "", "minecraft:elytra", "", ""].map(String::from).to_vec(),
                samples: (0..=30)
                    .map(|tick| {
                        let time = tick as f64 / 20.0;
                        EntitySample { time, pos: at - v * ((end - time) * 20.0) as f32, yaw: -90.0, pitch, body_yaw: -90.0, pose, flags }
                    })
                    .collect(),
                last_swing: -10.0,
                last_swing_off: -10.0,
                last_hurt: -10.0,
                last_seen: 0.0,
            })
            .collect();
        std::fs::create_dir_all("target/tmp").unwrap();
        for (name, eye) in [("side", Vec3::new(4.5, 2.5, 7.0)), ("behind", Vec3::new(-3.0, 5.0, 2.5))] {
            let pose = CameraPose { eye, target: Vec3::new(4.5, 0.8, 0.0), fov_y: 0.9 };
            let mut quads = Vec::new();
            renderer.restart();
            for f in 0..=36 {
                let t = f as f64 / 24.0;
                quads.clear();
                entities.build(&states, t, None, &mut quads);
                renderer.render(&[], &quads, &pose, t);
            }
            crate::write_png(&format!("target/tmp/elytra-{name}.png"), 960, 544, renderer.output()).unwrap();
        }
    }
}
