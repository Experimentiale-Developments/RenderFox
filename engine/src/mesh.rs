//! Turns world sections into quads with baked light, per section.

use crate::assets::*;
use crate::vmath::*;
use crate::world::{Window, World};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub const QUAD_TRI: u8 = 1; // only the s + t <= 1 half exists
pub const QUAD_FLAT: u8 = 2; // uniform lighting, no corner interpolation needed

const UNKNOWN: u16 = 0xffff;

/// A planar parallelogram (or triangle) in world space: p + s * e1 + t * e2 with
/// s, t in 0..1. Minecraft geometry is almost entirely such quads, and treating the
/// quad rather than two triangles as the primitive halves the setup work.
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub p: Vec3,
    pub e1: Vec3,
    pub e2: Vec3,
    pub n: Vec3,
    pub uv: Vec2, // uv(s, t) = uv + s * uv_s + t * uv_t, texture space 0..1
    pub uv_s: Vec2,
    pub uv_t: Vec2,
    pub sky: [u8; 4], // corners (0,0) (1,0) (0,1) (1,1); 0..255
    pub blk: [u8; 4],
    pub ao: [u8; 4],
    pub tint: u32, // sRGB
    pub tex: u16,
    pub mat: u16,
    pub layer: u8,
    pub flags: u8,
    pub wave: u8, // 0 none, 1 plant base (anchored), 2 plant top, 3 leaves
}

impl Default for Quad {
    fn default() -> Self {
        Quad {
            p: Vec3::ZERO,
            e1: Vec3::ZERO,
            e2: Vec3::ZERO,
            n: Vec3::ZERO,
            uv: Vec2::default(),
            uv_s: Vec2::default(),
            uv_t: Vec2::default(),
            sky: [0; 4],
            blk: [0; 4],
            ao: [0; 4],
            tint: 0xffffff,
            tex: 0,
            mat: 0,
            layer: SOLID,
            flags: 0,
            wave: 0,
        }
    }
}

pub struct SectionMesh {
    pub cx: i32,
    #[allow(dead_code)]
    pub sy: i32,
    pub cz: i32,
    pub lo: Vec3, // bounds of the geometry
    pub hi: Vec3,
    /// Solid and cutout, grouped by facing: quads[group_start[g] .. group_start[g + 1])
    /// face axis direction g (0..5), group 6 holds everything else. group_plane[g] is
    /// the plane coordinate furthest along the facing, so the camera sees none of group
    /// g when it is behind that plane.
    pub quads: Vec<Quad>,
    pub group_start: [u32; 8],
    pub group_plane: [f32; 6],
    pub water: Vec<Quad>, // translucent
    #[allow(dead_code)] // the world version this mesh was built from, for debugging
    pub version: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshSettings {
    pub focus: Vec3, // orbit centre; leaves far from it render culled
    pub fancy_leaves_radius: f32,
    pub merge_radius: f32,        // beyond this, full faces merge into larger quads
    pub plant_radius: f32,        // beyond this, grass and flowers are not drawn
    pub solid_leaves_radius: f32, // beyond this, leaves are opaque like fast graphics
}

impl Default for MeshSettings {
    fn default() -> Self {
        MeshSettings {
            focus: Vec3::ZERO,
            fancy_leaves_radius: 96.0,
            merge_radius: 64.0,
            plant_radius: 104.0,
            solid_leaves_radius: 128.0,
        }
    }
}

// Vanilla FaceInfo corner order (0 = from, 1 = to per axis), as in assets.rs.
const FACE_CORNERS: [[[i32; 3]; 4]; 6] = [
    [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
    [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
];

/// java.util.Random, for vanilla's weighted variant pick.
struct JavaRandom {
    seed: i64,
}

impl JavaRandom {
    const MASK: i64 = (1 << 48) - 1;

    fn new(s: i64) -> Self {
        JavaRandom { seed: (s ^ 0x5DEECE66D) & Self::MASK }
    }

    fn next(&mut self, bits: i32) -> i32 {
        self.seed = (self.seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & Self::MASK;
        ((self.seed as u64) >> (48 - bits)) as i32
    }

    fn next_int(&mut self, bound: i32) -> i32 {
        if (bound & bound.wrapping_neg()) == bound {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
                return val;
            }
        }
    }
}

#[derive(PartialEq, Eq)]
enum Offset {
    None,
    Xz,
    Xyz,
}

fn offset_type(name: &str) -> Offset {
    let Some(n) = name.strip_prefix("minecraft:") else { return Offset::None };
    match n {
        "short_grass" | "fern" | "short_dry_grass" | "tall_dry_grass" | "bush" => Offset::Xyz,
        "tall_grass" | "large_fern" | "dandelion" | "poppy" | "blue_orchid" | "allium" | "azure_bluet" | "red_tulip"
        | "orange_tulip" | "white_tulip" | "pink_tulip" | "oxeye_daisy" | "cornflower" | "lily_of_the_valley"
        | "wither_rose" | "torchflower" | "sunflower" | "lilac" | "rose_bush" | "peony" | "pitcher_plant"
        | "open_eyeblossom" | "closed_eyeblossom" | "bamboo" | "mangrove_propagule" | "pointed_dripstone"
        | "small_dripleaf" | "hanging_roots" | "warped_roots" | "crimson_roots" | "nether_sprouts" | "firefly_bush"
        | "cactus_flower" => Offset::Xz,
        _ => Offset::None,
    }
}

fn occludes(assets: &Assets, neighbour: u16, this: &BlockInfo, fast_leaves: bool, solid_leaves: bool) -> bool {
    if neighbour == UNKNOWN {
        return true;
    }
    if neighbour == 0 {
        return false;
    }
    let n = assets.block(neighbour);
    if n.full_opaque {
        return true;
    }
    if solid_leaves && n.leaves && (this.full_opaque || this.leaves) {
        return true;
    }
    if this.full_cube && n.full_cube && !this.leaves && n.name == this.name {
        return true; // glass to glass
    }
    fast_leaves && this.leaves && n.leaves
}

fn opaque_for_ao(assets: &Assets, id: u16) -> bool {
    id == UNKNOWN || (id != 0 && assets.block(id).full_opaque)
}

fn tint_for(b: &BlockInfo, biome: &Biome) -> u32 {
    match b.tint {
        Tint::Grass => biome.grass,
        Tint::Foliage => biome.foliage,
        Tint::DryFoliage => biome.dry_foliage,
        Tint::Water => biome.water,
        Tint::Fixed => b.fixed_tint,
        Tint::Redstone => {
            let power: i32 = b.props.get("power").map_or(0, |p| p.parse().unwrap_or(0));
            let f = power as f32 / 15.0;
            let r = f * 0.6 + if f > 0.0 { 0.4 } else { 0.3 };
            let g = clamp01(f * f * 0.7 - 0.5);
            let bl = clamp01(f * f * 0.6 - 0.7);
            ((r * 255.0) as u32) << 16 | ((g * 255.0) as u32) << 8 | (bl * 255.0) as u32
        }
        Tint::Stem => {
            let age: i32 = b.props.get("age").map_or(7, |a| a.parse().unwrap_or(0));
            ((age * 32) as u32) << 16 | ((255 - age * 8) as u32) << 8 | (age * 4) as u32
        }
        Tint::None => 0xffffff,
    }
}

fn average_color(colors: &[u32]) -> u32 {
    let (mut r, mut g, mut b) = (0i32, 0i32, 0i32);
    for &c in colors {
        r += (c >> 16 & 0xff) as i32;
        g += (c >> 8 & 0xff) as i32;
        b += (c & 0xff) as i32;
    }
    let n = colors.len() as i32;
    ((r / n) as u32) << 16 | ((g / n) as u32) << 8 | (b / n) as u32
}

/// Vanilla's per-position seed, used for random model variants and plant offsets so
/// grass tops and flowers land the same way the game draws them.
pub fn mc_seed(x: i32, y: i32, z: i32) -> i64 {
    let mut l = ((x as u32).wrapping_mul(3129871) as i32 as i64) ^ (z as i64).wrapping_mul(116129781) ^ y as i64;
    l = l.wrapping_mul(l).wrapping_mul(42317861).wrapping_add(l.wrapping_mul(11));
    l >> 16
}

// Far away, full faces that share texture, tint and light fuse into larger quads
// with world-aligned tiling. Keyed per direction and plane inside the section.
#[derive(Clone, Copy, Default)]
struct Cell {
    tint: u32,
    tex: u16,
    mat: u16,
    sky: u8,
    blk: u8,
    ao: u8,
    layer: u8,
    set: bool,
    used: bool,
}

impl Cell {
    fn same(&self, o: &Cell) -> bool {
        o.set
            && !o.used
            && self.tex == o.tex
            && self.tint == o.tint
            && self.mat == o.mat
            && self.sky == o.sky
            && self.blk == o.blk
            && self.ao == o.ao
            && self.layer == o.layer
    }
}

fn cell_index(dir: usize, layer: i32, b2: i32, a: i32) -> usize {
    ((dir as i32 * 16 + layer) * 16 + b2) as usize * 16 + a as usize
}

fn cell_for(dir: u8, x: i32, y: i32, z: i32) -> usize {
    let layer = if dir <= UP { y } else if dir <= SOUTH { z } else { x };
    let a = if dir <= SOUTH { x } else { z };
    let b2 = if dir <= UP { z } else { y };
    cell_index(dir as usize, layer, b2, a)
}

fn emit(mesh: &mut SectionMesh, q: Quad, translucent: bool) {
    for v in [q.p, q.p + q.e1, q.p + q.e2, q.p + q.e1 + q.e2] {
        mesh.lo = vmin(mesh.lo, v);
        mesh.hi = vmax(mesh.hi, v);
    }
    if translucent { mesh.water.push(q) } else { mesh.quads.push(q) }
}

// Corner order of a Quad: (0,0)=v1, (1,0)=v2, (0,1)=v0, (1,1)=v3 of the vanilla face.
const CORNER_OF: [usize; 4] = [1, 2, 0, 3];

pub struct Mesher {
    world: Arc<World>,
    settings: MeshSettings,
    meshes: Mutex<BTreeMap<u64, Arc<SectionMesh>>>,
}

impl Mesher {
    pub fn new(world: Arc<World>, settings: MeshSettings) -> Mesher {
        Mesher { world, settings, meshes: Mutex::new(BTreeMap::new()) }
    }

    pub fn build(&self, cx: i32, sy: i32, cz: i32) -> Option<Arc<SectionMesh>> {
        let mut win = Window::new();
        let version = self.world.window(cx, sy, cz, &mut win)?;
        let assets = self.world.assets();
        let s = &self.settings;
        let mut mesh = SectionMesh {
            cx,
            sy,
            cz,
            lo: Vec3::splat(1e9),
            hi: Vec3::splat(-1e9),
            quads: Vec::new(),
            group_start: [0; 8],
            group_plane: [0.0; 6],
            water: Vec::new(),
            version,
        };
        let (bx, by, bz) = (cx * 16, sy * 16, cz * 16);
        let centre = Vec3::new(bx as f32 + 8.0, by as f32 + 8.0, bz as f32 + 8.0);
        let horiz = (centre.x - s.focus.x).hypot(centre.z - s.focus.z);
        let fast_leaves = horiz > s.fancy_leaves_radius;
        let merge = horiz > s.merge_radius;
        let drop_plants = horiz > s.plant_radius;
        let solid_leaves = horiz > s.solid_leaves_radius;

        let mut cells: Vec<Cell> = if merge { vec![Cell::default(); 6 * 16 * 256] } else { Vec::new() };
        let win = &*win;
        let at = |x: i32, y: i32, z: i32| win.blocks[Window::idx(x, y, z)];
        let light_at = |x: i32, y: i32, z: i32| win.light[Window::idx(x, y, z)];

        // Biome tint blended over a 5x5 block area (vanilla's default blend), within the
        // section; computed lazily per column since most blocks are untinted.
        let blended_tint = |b: &BlockInfo, x: i32, y: i32, z: i32| -> u32 {
            if matches!(b.tint, Tint::Fixed | Tint::Redstone | Tint::Stem) {
                return tint_for(b, assets.biome(0));
            }
            let mut samples = [0u32; 25];
            let mut n = 0;
            for dz in -2..=2 {
                for dx in -2..=2 {
                    let (sx, sz) = ((x + dx).clamp(0, 15), (z + dz).clamp(0, 15));
                    let biome = win.biomes[(((y >> 2) * 4 + (sz >> 2)) * 4 + (sx >> 2)) as usize];
                    samples[n] = tint_for(b, assets.biome(biome));
                    n += 1;
                }
            }
            average_color(&samples[..n])
        };

        let smooth_corner = |c: (i32, i32, i32), a: (i32, i32, i32), bv: (i32, i32, i32)| -> (u8, u8, u8) {
            let lc = light_at(c.0, c.1, c.2);
            let e1 = at(c.0 + a.0, c.1 + a.1, c.2 + a.2);
            let e2 = at(c.0 + bv.0, c.1 + bv.1, c.2 + bv.2);
            let k = at(c.0 + a.0 + bv.0, c.1 + a.1 + bv.1, c.2 + a.2 + bv.2);
            let (o1, o2) = (opaque_for_ao(assets, e1), opaque_for_ao(assets, e2));
            let ok = opaque_for_ao(assets, k) || (o1 && o2);
            let l1 = if o1 { lc } else { light_at(c.0 + a.0, c.1 + a.1, c.2 + a.2) };
            let l2 = if o2 { lc } else { light_at(c.0 + bv.0, c.1 + bv.1, c.2 + bv.2) };
            let lk = if ok { lc } else { light_at(c.0 + a.0 + bv.0, c.1 + a.1 + bv.1, c.2 + a.2 + bv.2) };
            let s = (lc >> 4) as i32 + (l1 >> 4) as i32 + (l2 >> 4) as i32 + (lk >> 4) as i32;
            let b = (lc & 15) as i32 + (l1 & 15) as i32 + (l2 & 15) as i32 + (lk & 15) as i32;
            let w = |o: bool| if o { 0.2f32 } else { 1.0 };
            let shade = (((1.0f32 + w(o1)) + w(o2)) + w(ok)) / 4.0;
            ((s * 17 / 4) as u8, (b * 17 / 4) as u8, (shade * 255.0) as u8)
        };

        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let id = at(x, y, z);
                    if id == 0 {
                        continue;
                    }
                    let b = assets.block(id);
                    if b.air {
                        continue;
                    }
                    let (wx, wy, wz) = (bx + x, by + y, bz + z);

                    if b.fluid != 0 || b.waterlogged {
                        let lava = b.fluid == 2;
                        let same_fluid = |n: u16| {
                            if n == 0 || n == UNKNOWN {
                                return false;
                            }
                            let o = assets.block(n);
                            if lava { o.fluid == 2 } else { o.fluid == 1 || o.waterlogged }
                        };
                        let own_height = |n: u16| {
                            let o = assets.block(n);
                            let amount = if o.fluid != 0 { if o.fluid_level >= 8 { 8 } else { 8 - o.fluid_level as i32 } } else { 8 };
                            amount as f32 / 9.0
                        };
                        let height_at = |hx: i32, hy: i32, hz: i32| -> f32 {
                            let n = at(hx, hy, hz);
                            if same_fluid(n) {
                                return if same_fluid(at(hx, hy + 1, hz)) { 1.0 } else { own_height(n) };
                            }
                            if n == UNKNOWN {
                                return -1.0;
                            }
                            if n == 0 || !assets.block(n).full_opaque { 0.0 } else { -1.0 }
                        };
                        let h = height_at(x, y, z);
                        let weighted = |acc: &mut [f32; 2], v: f32| {
                            if v >= 0.8 {
                                acc[0] += v * 10.0;
                                acc[1] += 10.0;
                            } else if v >= 0.0 {
                                acc[0] += v;
                                acc[1] += 1.0;
                            }
                        };
                        let corner = |dx: i32, dz: i32| -> f32 {
                            let (h1, h2) = (height_at(x + dx, y, z), height_at(x, y, z + dz));
                            if h1 >= 1.0 || h2 >= 1.0 {
                                return 1.0;
                            }
                            let mut acc = [0.0f32; 2];
                            if h1 > 0.0 || h2 > 0.0 {
                                let d = height_at(x + dx, y, z + dz);
                                if d >= 1.0 {
                                    return 1.0;
                                }
                                weighted(&mut acc, d);
                            }
                            weighted(&mut acc, h);
                            weighted(&mut acc, h1);
                            weighted(&mut acc, h2);
                            if acc[1] > 0.0 { acc[0] / acc[1] } else { h }
                        };
                        let (h_nw, h_ne, h_sw, h_se) = (corner(-1, -1), corner(1, -1), corner(-1, 1), corner(1, 1));
                        let lc = light_at(x, y, z).max(light_at(x, y + 1, z));
                        let dark = lc == 0 && !lava;
                        if dark && b.fluid != 0 {
                            continue;
                        }
                        if !dark {
                            let biome = assets.biome(win.biomes[(((y >> 2) * 4 + (z >> 2)) * 4 + (x >> 2)) as usize]);
                            let still = assets.texture(if lava { "block/lava_still" } else { "block/water_still" });
                            let flow = assets.texture(if lava { "block/lava_flow" } else { "block/water_flow" });
                            let fill = |q: &mut Quad| {
                                let s = ((lc >> 4) as i32 * 17) as u8;
                                let bl = if lava { 255 } else { ((lc & 15) as i32 * 17) as u8 };
                                q.sky = [s; 4];
                                q.blk = [bl; 4];
                                q.ao = [255; 4];
                                q.tint = if lava { 0xffffff } else { biome.water };
                                q.mat = if lava { MAT_LAVA | MAT_EMISSIVE } else { MAT_WATER | MAT_NO_SHADOW };
                                q.layer = if lava { SOLID } else { TRANSLUCENT };
                                q.flags = QUAD_FLAT;
                            };
                            let o = Vec3::new(wx as f32, wy as f32, wz as f32);
                            if !same_fluid(at(x, y + 1, z)) {
                                // Top: split along the diagonal when the corners are not coplanar.
                                let nw = o + Vec3::new(0.0, h_nw, 0.0);
                                let ne = o + Vec3::new(1.0, h_ne, 0.0);
                                let sw = o + Vec3::new(0.0, h_sw, 1.0);
                                let se = o + Vec3::new(1.0, h_se, 1.0);
                                let planar = ((h_nw + h_se) - (h_ne + h_sw)).abs() < 1e-4;
                                let mut q = Quad::default();
                                fill(&mut q);
                                q.tex = still;
                                if planar {
                                    q.p = sw;
                                    q.e1 = se - sw;
                                    q.e2 = nw - sw;
                                    q.uv = Vec2::new(0.0, 1.0);
                                    q.uv_s = Vec2::new(1.0, 0.0);
                                    q.uv_t = Vec2::new(0.0, -1.0);
                                    q.n = normalize(cross(q.e1, q.e2));
                                    emit(&mut mesh, q, !lava);
                                } else {
                                    let mut a = q;
                                    a.flags |= QUAD_TRI;
                                    a.p = sw;
                                    a.e1 = se - sw;
                                    a.e2 = nw - sw;
                                    a.uv = Vec2::new(0.0, 1.0);
                                    a.uv_s = Vec2::new(1.0, 0.0);
                                    a.uv_t = Vec2::new(0.0, -1.0);
                                    a.n = normalize(cross(a.e1, a.e2));
                                    emit(&mut mesh, a, !lava);
                                    let mut c = q;
                                    c.flags |= QUAD_TRI;
                                    c.p = ne;
                                    c.e1 = nw - ne;
                                    c.e2 = se - ne;
                                    c.uv = Vec2::new(1.0, 0.0);
                                    c.uv_s = Vec2::new(-1.0, 0.0);
                                    c.uv_t = Vec2::new(0.0, 1.0);
                                    c.n = normalize(cross(c.e1, c.e2));
                                    emit(&mut mesh, c, !lava);
                                }
                            }
                            // Each side runs from corner a to corner b with the fluid on its left
                            // when seen from outside, so the normal points out.
                            let sides = [
                                (0, -1, o + Vec3::new(1.0, 0.0, 0.0), o + Vec3::new(0.0, 0.0, 0.0), h_ne, h_nw),
                                (0, 1, o + Vec3::new(0.0, 0.0, 1.0), o + Vec3::new(1.0, 0.0, 1.0), h_sw, h_se),
                                (-1, 0, o + Vec3::new(0.0, 0.0, 0.0), o + Vec3::new(0.0, 0.0, 1.0), h_nw, h_sw),
                                (1, 0, o + Vec3::new(1.0, 0.0, 1.0), o + Vec3::new(1.0, 0.0, 0.0), h_se, h_ne),
                            ];
                            for (dx, dz, sa, sb, ha, hb) in sides {
                                let n = at(x + dx, y, z + dz);
                                if same_fluid(n) || n == UNKNOWN || (n != 0 && assets.block(n).full_opaque) {
                                    continue;
                                }
                                let mut q = Quad::default();
                                fill(&mut q);
                                q.tex = flow;
                                q.p = sa;
                                q.e1 = sb - sa;
                                if (ha - hb).abs() < 1e-4 {
                                    q.e2 = Vec3::new(0.0, ha, 0.0);
                                    q.uv = Vec2::new(0.0, 0.5);
                                    q.uv_s = Vec2::new(0.5, 0.0);
                                    q.uv_t = Vec2::new(0.0, -0.5 * ha);
                                    q.n = normalize(cross(q.e1, q.e2));
                                    emit(&mut mesh, q, !lava);
                                } else {
                                    let mut lower = q;
                                    lower.flags |= QUAD_TRI;
                                    lower.e2 = Vec3::new(0.0, ha, 0.0);
                                    lower.uv = Vec2::new(0.0, 0.5);
                                    lower.uv_s = Vec2::new(0.5, 0.0);
                                    lower.uv_t = Vec2::new(0.0, -0.5 * ha);
                                    lower.n = normalize(cross(lower.e1, lower.e2));
                                    emit(&mut mesh, lower, !lava);
                                    let mut upper = q;
                                    upper.flags |= QUAD_TRI;
                                    upper.p = sb + Vec3::new(0.0, hb, 0.0);
                                    upper.e1 = (sa + Vec3::new(0.0, ha, 0.0)) - upper.p;
                                    upper.e2 = Vec3::new(0.0, -hb, 0.0);
                                    upper.uv = Vec2::new(0.5, 0.5 - 0.5 * hb);
                                    upper.uv_s = Vec2::new(-0.5, 0.5 * (hb - ha));
                                    upper.uv_t = Vec2::new(0.0, 0.5 * hb);
                                    upper.n = normalize(cross(upper.e1, upper.e2));
                                    emit(&mut mesh, upper, !lava);
                                }
                            }
                            if b.fluid != 0 {
                                continue;
                            }
                        }
                    }

                    // The block's model.
                    if b.parts.is_empty() {
                        continue;
                    }
                    if drop_plants && (b.mat & MAT_PLANT) != 0 && !b.full_cube {
                        continue;
                    }
                    let seed = mc_seed(wx, wy, wz);
                    let mut coplanar = [0i32; 6];
                    let mut offset = Vec3::ZERO;
                    let ot = offset_type(&b.name);
                    if ot != Offset::None {
                        let l = mc_seed(wx, 0, wz);
                        offset.x = (((l & 15) as f32 / 15.0 - 0.5) * 0.5).clamp(-0.25, 0.25);
                        offset.z = (((l >> 8 & 15) as f32 / 15.0 - 0.5) * 0.5).clamp(-0.25, 0.25);
                        if ot == Offset::Xyz {
                            offset.y = ((l >> 4 & 15) as f32 / 15.0 - 1.0) * 0.2;
                        }
                    }
                    let upper_half = b.props.get("half").is_some_and(|h| h == "upper") && (b.mat & MAT_PLANT) != 0;
                    let tint = if b.tint == Tint::None { 0xffffff } else { blended_tint(b, x, y, z) };

                    for part in &b.parts {
                        let Some((first, _)) = part.first() else { continue };
                        let mut model = first;
                        if part.len() > 1 {
                            let total: i32 = part.iter().map(|(_, w)| *w).sum();
                            let mut rnd = JavaRandom::new(seed);
                            let mut pick = rnd.next_int(total.max(1));
                            for (m, w) in part {
                                pick -= w;
                                if pick < 0 {
                                    model = m;
                                    break;
                                }
                            }
                        }
                        for bq in &model.quads {
                            if bq.cull != NONE {
                                let c = bq.cull as usize;
                                if occludes(assets, at(x + DIR_DX[c], y + DIR_DY[c], z + DIR_DZ[c]), b, fast_leaves, solid_leaves) {
                                    continue;
                                }
                            }
                            let t = assets.tex(bq.tex);
                            let mut q = Quad::default();
                            let base = Vec3::new(wx as f32, wy as f32, wz as f32);
                            let v: [Vec3; 4] = std::array::from_fn(|i| base + bq.v[i] + offset);
                            q.p = v[1];
                            q.e1 = v[2] - v[1];
                            q.e2 = v[0] - v[1];
                            q.uv = bq.uv[1];
                            q.uv_s = Vec2::new(bq.uv[2].x - bq.uv[1].x, bq.uv[2].y - bq.uv[1].y);
                            q.uv_t = Vec2::new(bq.uv[0].x - bq.uv[1].x, bq.uv[0].y - bq.uv[1].y);
                            q.n = bq.normal;
                            q.tex = bq.tex;
                            q.tint = if bq.tint_index >= 0 { tint } else { 0xffffff };
                            q.mat = b.mat;
                            if bq.emission > 0 {
                                q.mat |= MAT_EMISSIVE;
                            }
                            q.layer = if t.has_translucent || bq.translucent {
                                TRANSLUCENT
                            } else if t.has_cutout {
                                CUTOUT
                            } else {
                                SOLID
                            };
                            if solid_leaves && b.leaves && q.layer == CUTOUT {
                                q.tex = assets.opaque_variant(bq.tex);
                                q.layer = SOLID;
                            }
                            if (b.mat & MAT_PLANT) != 0 {
                                q.wave = if upper_half { 2 } else { 1 };
                            }

                            // Is the face flush with the block boundary? Then the cell in front
                            // supplies the light, as in vanilla.
                            let mut flush = false;
                            if bq.dir != NONE {
                                let d = bq.dir;
                                let plane = if d == UP || d == SOUTH || d == EAST { 1.0 } else { 0.0 };
                                let coord = if d <= UP { bq.v[0].y } else if d <= SOUTH { bq.v[0].z } else { bq.v[0].x };
                                flush = (coord - plane).abs() < 1e-4;
                            }
                            let smooth = model.ao && bq.dir != NONE && (b.mat & MAT_PLANT) == 0;
                            if smooth {
                                let d = bq.dir as usize;
                                let (ddx, ddy, ddz) = (DIR_DX[d], DIR_DY[d], DIR_DZ[d]);
                                let o = if flush { (x + ddx, y + ddy, z + ddz) } else { (x, y, z) };
                                for ci in 0..4 {
                                    let lv = bq.v[CORNER_OF[ci]];
                                    let sgn = |v: f32| if v > 0.5 { 1 } else { -1 };
                                    let (a, bv) = if bq.dir <= UP {
                                        ((sgn(lv.x), 0, 0), (0, 0, sgn(lv.z)))
                                    } else if bq.dir <= SOUTH {
                                        ((sgn(lv.x), 0, 0), (0, sgn(lv.y), 0))
                                    } else {
                                        ((0, 0, sgn(lv.z)), (0, sgn(lv.y), 0))
                                    };
                                    let (s, bl, ao) = smooth_corner(o, a, bv);
                                    q.sky[ci] = s;
                                    q.blk[ci] = bl;
                                    q.ao[ci] = ao;
                                }
                            } else {
                                let mut l = light_at(x, y, z).max(light_at(x, y + 1, z));
                                if bq.dir != NONE && (b.mat & MAT_PLANT) == 0 {
                                    let d = bq.dir as usize;
                                    l = l.max(light_at(x + DIR_DX[d], y + DIR_DY[d], z + DIR_DZ[d]));
                                }
                                q.sky = [((l >> 4) as i32 * 17) as u8; 4];
                                q.blk = [((l & 15) as i32 * 17) as u8; 4];
                                q.ao = [255; 4];
                                q.flags |= QUAD_FLAT;
                            }
                            // Sealed caves: no sky or block light reaches the face, so the orbit
                            // can never see it. This is most of the terrain's faces.
                            let lit = q.sky.iter().chain(q.blk.iter()).fold(0u8, |acc, &v| acc | v);
                            if lit == 0 && bq.emission == 0 && b.emission == 0 {
                                continue;
                            }
                            if bq.emission > 0 {
                                for bl in q.blk.iter_mut() {
                                    *bl = (*bl).max(bq.emission.wrapping_mul(17));
                                }
                            }
                            if b.emission > 0 {
                                for bl in q.blk.iter_mut() {
                                    *bl = (*bl).max(b.emission.wrapping_mul(17));
                                }
                            }
                            if flush {
                                // A second face in the same plane (grass side overlay) is an
                                // overlay: lift it off the base so the two never z-fight.
                                let layer_index = coplanar[bq.dir as usize];
                                coplanar[bq.dir as usize] += 1;
                                if layer_index > 0 {
                                    q.p += q.n * (0.0015 * layer_index as f32);
                                }
                                let uniform = q.sky.iter().all(|&v| v == q.sky[0])
                                    && q.blk.iter().all(|&v| v == q.blk[0])
                                    && q.ao.iter().all(|&v| v == q.ao[0]);
                                if merge
                                    && layer_index == 0
                                    && b.full_cube
                                    && bq.cull == bq.dir
                                    && uniform
                                    && q.layer != TRANSLUCENT
                                    && (q.mat & MAT_EMISSIVE) == 0
                                    && t.frames.len() == 1
                                {
                                    cells[cell_for(bq.dir, x, y, z)] = Cell {
                                        tint: q.tint,
                                        tex: q.tex,
                                        mat: q.mat,
                                        sky: q.sky[0],
                                        blk: q.blk[0],
                                        ao: q.ao[0],
                                        layer: q.layer,
                                        set: true,
                                        used: false,
                                    };
                                    continue;
                                }
                            }
                            emit(&mut mesh, q, q.layer == TRANSLUCENT);
                        }
                    }
                }
            }
        }

        if merge {
            for dir in 0..6usize {
                for layer in 0..16 {
                    for b2 in 0..16 {
                        for a in 0..16 {
                            let start = cells[cell_index(dir, layer, b2, a)];
                            if !start.set || start.used {
                                continue;
                            }
                            let mut w = 1;
                            while a + w < 16 && start.same(&cells[cell_index(dir, layer, b2, a + w)]) {
                                w += 1;
                            }
                            let mut h = 1;
                            while b2 + h < 16 {
                                let row = (0..w).all(|k| start.same(&cells[cell_index(dir, layer, b2 + h, a + k)]));
                                if !row {
                                    break;
                                }
                                h += 1;
                            }
                            for j in 0..h {
                                for k in 0..w {
                                    cells[cell_index(dir, layer, b2 + j, a + k)].used = true;
                                }
                            }
                            // Box in world units, then vanilla's corner order and default uvs,
                            // counted in blocks so the texture repeats once per block.
                            let (lo, hi) = (layer as f32, (layer + 1) as f32);
                            let (af, bf, aw, bh) = (a as f32, b2 as f32, (a + w) as f32, (b2 + h) as f32);
                            let (mut from, mut to) = match dir as u8 {
                                d if d <= UP => (Vec3::new(af, lo, bf), Vec3::new(aw, hi, bh)),
                                d if d <= SOUTH => (Vec3::new(af, bf, lo), Vec3::new(aw, bh, hi)),
                                _ => (Vec3::new(lo, bf, af), Vec3::new(hi, bh, aw)),
                            };
                            let origin = Vec3::new(bx as f32, by as f32, bz as f32);
                            from += origin;
                            to += origin;
                            let uv = match dir as u8 {
                                DOWN => [from.x, -to.z, to.x, -from.z],
                                UP => [from.x, from.z, to.x, to.z],
                                NORTH => [-to.x, -to.y, -from.x, -from.y],
                                SOUTH => [from.x, -to.y, to.x, -from.y],
                                WEST => [from.z, -to.y, to.z, -from.y],
                                _ => [-to.z, -to.y, -from.z, -from.y],
                            };
                            let mut v = [Vec3::ZERO; 4];
                            let mut tuv = [Vec2::default(); 4];
                            for i in 0..4 {
                                let c = FACE_CORNERS[dir][i];
                                v[i] = Vec3::new(
                                    if c[0] != 0 { to.x } else { from.x },
                                    if c[1] != 0 { to.y } else { from.y },
                                    if c[2] != 0 { to.z } else { from.z },
                                );
                                tuv[i] = Vec2::new(
                                    if i == 0 || i == 1 { uv[0] } else { uv[2] },
                                    if i == 0 || i == 3 { uv[1] } else { uv[3] },
                                );
                            }
                            let q = Quad {
                                p: v[1],
                                e1: v[2] - v[1],
                                e2: v[0] - v[1],
                                uv: tuv[1],
                                uv_s: Vec2::new(tuv[2].x - tuv[1].x, tuv[2].y - tuv[1].y),
                                uv_t: Vec2::new(tuv[0].x - tuv[1].x, tuv[0].y - tuv[1].y),
                                n: dir_normal(dir),
                                tex: start.tex,
                                tint: start.tint,
                                mat: start.mat,
                                layer: start.layer,
                                flags: QUAD_FLAT,
                                sky: [start.sky; 4],
                                blk: [start.blk; 4],
                                ao: [start.ao; 4],
                                wave: 0,
                            };
                            emit(&mut mesh, q, false);
                        }
                    }
                }
            }
        }

        let group_of = |q: &Quad| -> usize {
            if q.wave != 0 {
                return 6;
            }
            (0..6).find(|&d| dot(q.n, dir_normal(d)) > 0.9999).unwrap_or(6)
        };
        mesh.quads.sort_by_key(group_of); // stable, like std::stable_sort
        for g in 0..6 {
            mesh.group_plane[g] = if g & 1 != 0 { 1e30 } else { -1e30 };
        }
        let mut counts = [0u32; 7];
        for q in &mesh.quads {
            let g = group_of(q);
            counts[g] += 1;
            if g < 6 {
                let axis = match g / 2 {
                    0 => 1,
                    1 => 2,
                    _ => 0,
                };
                let c = q.p[axis];
                // Positive facings are seen from above their plane: keep the lowest plane.
                // Negative facings are seen from below: keep the highest.
                mesh.group_plane[g] =
                    if g & 1 != 0 { fmin(mesh.group_plane[g], c) } else { fmax(mesh.group_plane[g], c) };
            }
        }
        mesh.group_start[0] = 0;
        for g in 0..7 {
            mesh.group_start[g + 1] = mesh.group_start[g] + counts[g];
        }
        // Kept for as long as the section stands: none of the room grown into while building.
        mesh.quads.shrink_to_fit();
        mesh.water.shrink_to_fit();
        Some(Arc::new(mesh))
    }

    /// Published meshes, swapped atomically per section.
    pub fn publish(&self, key: u64, mesh: Option<Arc<SectionMesh>>) {
        Self::put(&mut self.meshes.lock().unwrap(), key, mesh);
    }

    /// Publishes a batch at once, running `with` before anyone sees it, so whatever else
    /// belongs to the same change (its shadows) cannot show up a frame apart from it.
    pub fn publish_all(&self, batch: Vec<(u64, Option<Arc<SectionMesh>>)>, with: impl FnOnce()) {
        let mut m = self.meshes.lock().unwrap();
        for (key, mesh) in batch {
            Self::put(&mut m, key, mesh);
        }
        with();
    }

    fn put(m: &mut BTreeMap<u64, Arc<SectionMesh>>, key: u64, mesh: Option<Arc<SectionMesh>>) {
        match mesh {
            Some(mesh) if !(mesh.quads.is_empty() && mesh.water.is_empty()) => {
                m.insert(key, mesh);
            }
            _ => {
                m.remove(&key);
            }
        }
    }

    /// Every published mesh, in section key order.
    pub fn snapshot(&self) -> Vec<Arc<SectionMesh>> {
        self.meshes.lock().unwrap().values().cloned().collect()
    }

    /// The same, taken together with `with`: what `publish_all` published alongside.
    pub fn snapshot_with(&self, with: impl FnOnce()) -> Vec<Arc<SectionMesh>> {
        let m = self.meshes.lock().unwrap();
        with();
        m.values().cloned().collect()
    }

    /// Every published mesh as it will be once `batch` is.
    pub fn snapshot_after(&self, batch: &[(u64, Option<Arc<SectionMesh>>)]) -> Vec<Arc<SectionMesh>> {
        let mut m = self.meshes.lock().unwrap().clone();
        for (key, mesh) in batch {
            Self::put(&mut m, *key, mesh.clone());
        }
        m.into_values().collect()
    }

    /// Lets every mesh go.
    pub fn clear(&self) {
        self.meshes.lock().unwrap().clear();
    }

    pub fn quad_count(&self) -> usize {
        self.meshes.lock().unwrap().values().map(|m| m.quads.len() + m.water.len()).sum()
    }
}
