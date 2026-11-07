//! Block states, models, textures and biomes from the extracted client jar.

use crate::vmath::*;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

// How a surface is drawn. Solid writes depth and colour, cutout is alpha tested,
// translucent is blended over the opaque image (one layer, nearest wins).
pub const SOLID: u8 = 0;
pub const CUTOUT: u8 = 1;
pub const TRANSLUCENT: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tint {
    #[default]
    None,
    Grass,
    Foliage,
    DryFoliage,
    Water,
    Fixed,
    Redstone,
    Stem,
}

// Material hints the shader acts on, beyond plain albedo.
pub const MAT_PLANT: u16 = 1; // lit as foliage: normal treated as up, soft translucency, sways
pub const MAT_LEAVES: u16 = 2; // sways slightly
pub const MAT_WATER: u16 = 4;
pub const MAT_LAVA: u16 = 8;
pub const MAT_EMISSIVE: u16 = 16; // self-lit texels (lava, glowstone, lanterns' flame)
pub const MAT_METAL: u16 = 32;
pub const MAT_GLASS: u16 = 64;
pub const MAT_NO_SHADOW: u16 = 128; // does not cast sun shadow (tiny details, water)
pub const MAT_ENTITY: u16 = 256;

#[derive(Clone, Debug, Default)]
pub struct Texture {
    pub w: i32, // one frame
    pub h: i32,
    pub levels: i32, // mip levels
    /// frames[f] holds all mip levels back to back, RGBA8 (R in the low byte) in sRGB
    /// storage; sampling converts through a LUT.
    pub frames: Vec<Vec<u32>>,
    pub level_offset: Vec<i32>, // offset of each mip level inside a frame
    pub frame_order: Vec<i32>,  // animation: frame index per step
    pub frame_time: i32,        // game ticks per animation step
    pub has_cutout: bool,       // any alpha == 0 texel
    pub has_translucent: bool,  // any 0 < alpha < 255 texel
    pub average: Vec3,          // linear average colour of opaque texels
    pub name: String,
}

#[derive(Clone, Copy, Debug)]
pub struct BakedQuad {
    pub v: [Vec3; 4],  // block-local, 0..1
    pub uv: [Vec2; 4], // 0..1 inside the texture
    pub normal: Vec3,
    pub tex: u16,
    pub tint_index: i8,
    pub cull: u8, // face is hidden when the neighbour on this side occludes
    pub dir: u8,  // axis-aligned facing, NONE for rotated faces
    #[allow(dead_code)] // decoded from the model, not used by this renderer's lighting
    pub shade: bool,
    pub emission: u8,      // 0..15 from the model element's light_emission
    pub translucent: bool, // force_translucent in the model
}

impl Default for BakedQuad {
    fn default() -> Self {
        BakedQuad {
            v: [Vec3::ZERO; 4],
            uv: [Vec2::default(); 4],
            normal: Vec3::ZERO,
            tex: 0,
            tint_index: -1,
            cull: NONE,
            dir: NONE,
            shade: true,
            emission: 0,
            translucent: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct BlockModel {
    pub quads: Vec<BakedQuad>,
    pub ao: bool,    // model's ambientocclusion flag
    pub plant: bool, // derived from cross/crop style parents
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Special {
    #[default]
    None,
    Chest,
    Bed,
    Sign,
    WallSign,
    HangingSign,
    Skull,
    Banner,
    Shulker,
}

/// Everything the mesher and lighter need to know about one block state.
#[derive(Debug)]
pub struct BlockInfo {
    pub name: String,  // minecraft:oak_stairs
    #[allow(dead_code)] // full state string, kept for debugging
    pub state: String,
    /// One entry per multipart part (a single one for variants), each a weighted list
    /// of models; the mesher picks per block position like vanilla's random variants.
    pub parts: Vec<Vec<(Arc<BlockModel>, i32)>>,
    pub air: bool,
    pub full_opaque: bool, // occludes neighbour faces and blocks light fully
    pub full_cube: bool,   // full cube geometry, maybe transparent (glass, leaves)
    pub leaves: bool,
    pub light_block: u8, // 0..15 light attenuation
    pub emission: u8,    // 0..15 block light emitted
    pub fluid: u8,       // 0 none, 1 water, 2 lava
    pub fluid_level: u8, // 0 source .. 7, 8+ falling
    pub waterlogged: bool,
    pub tint: Tint,
    pub fixed_tint: u32,
    pub mat: u16,
    pub special: Special,
    pub props: HashMap<String, String>,
    pub map_color: u32, // average colour for far LOD
}

impl BlockInfo {
    fn new(state: &str) -> BlockInfo {
        BlockInfo {
            name: String::new(),
            state: state.to_string(),
            parts: Vec::new(),
            air: false,
            full_opaque: false,
            full_cube: false,
            leaves: false,
            light_block: 0,
            emission: 0,
            fluid: 0,
            fluid_level: 0,
            waterlogged: false,
            tint: Tint::None,
            fixed_tint: 0xffffff,
            mat: 0,
            special: Special::None,
            props: HashMap::new(),
            map_color: 0,
        }
    }

    fn prop(&self, k: &str) -> &str {
        self.props.get(k).map(String::as_str).unwrap_or("")
    }
}

#[derive(Clone, Debug)]
pub struct Biome {
    #[allow(dead_code)]
    pub name: String,
    pub temperature: f32,
    pub downfall: f32,
    pub water: u32,
    pub grass_override: i32,
    pub foliage_override: i32,
    pub dry_foliage_override: i32,
    pub grass_modifier: i32, // 0 none, 1 dark_forest, 2 swamp
    pub sky: u32,
    pub grass: u32, // resolved
    pub foliage: u32,
    pub dry_foliage: u32,
}

const MAX_BLOCKS: usize = 65536;
const MAX_TEXTURES: usize = 65536;
const MAX_BIOMES: usize = 4096;

static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut t = [0.0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = srgb_to_linear(i as f32 / 255.0);
    }
    t
});

/// Linear 0..1 lookup for sRGB bytes.
pub fn lut() -> &'static [f32; 256] {
    &LUT
}

#[inline]
pub fn lin(c: u8) -> f32 {
    LUT[c as usize]
}

pub fn pack_rgb(c: Vec3) -> u32 {
    let q = |v: f32| (clamp01(linear_to_srgb(clamp01(v))) * 255.0).round() as u32;
    q(c.x) << 16 | q(c.y) << 8 | q(c.z)
}

struct Inner {
    state_ids: HashMap<String, u16>,
    block_count: usize,
    texture_ids: HashMap<String, u16>,
    texture_count: usize,
    biome_ids: HashMap<String, u16>,
    biome_count: usize,
    baked_models: HashMap<String, Arc<BlockModel>>,
    opaque_ids: HashMap<u16, u16>,
    grass_map: Vec<u32>,
    foliage_map: Vec<u32>,
    dry_foliage_map: Vec<u32>,
}

/// Readers index the tables without the lock while the network thread may be interning
/// new entries: every slot is written once, before its id is handed out.
pub struct Assets {
    root: String,
    blocks: Box<[OnceLock<Box<BlockInfo>>]>,
    textures: Box<[OnceLock<Box<Texture>>]>,
    biomes: Box<[OnceLock<Biome>]>,
    inner: Mutex<Inner>,
}

fn strip(id: &str) -> &str {
    match id.find(':') {
        Some(c) => &id[c + 1..],
        None => id,
    }
}

fn qualify(id: &str) -> String {
    if id.contains(':') { id.to_string() } else { format!("minecraft:{id}") }
}

pub fn read_json(path: &str) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    match serde_json::from_str(text) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("bad json {path}: {e}");
            None
        }
    }
}

// Accessors in the spirit of nlohmann's value(key, default).
pub fn jf32(v: &Value) -> f32 {
    v.as_f64().unwrap_or(0.0) as f32
}

pub fn ji32(v: &Value) -> i32 {
    match v.as_i64() {
        Some(i) => i as i32,
        None => v.as_f64().unwrap_or(0.0) as i32,
    }
}

pub fn val_i32(j: &Value, k: &str, def: i32) -> i32 {
    j.get(k).map(ji32).unwrap_or(def)
}

pub fn val_f32(j: &Value, k: &str, def: f32) -> f32 {
    j.get(k).map(jf32).unwrap_or(def)
}

pub fn val_bool(j: &Value, k: &str, def: bool) -> bool {
    j.get(k).and_then(Value::as_bool).unwrap_or(def)
}

pub fn val_str(j: &Value, k: &str, def: &str) -> String {
    j.get(k).and_then(Value::as_str).unwrap_or(def).to_string()
}

fn parse_color(j: &Value) -> u32 {
    if let Some(i) = j.as_i64() {
        return (i as u32) & 0xffffff;
    }
    if let Some(s) = j.as_str()
        && let Some(hex) = s.strip_prefix('#')
    {
        return u32::from_str_radix(hex, 16).unwrap_or(0) & 0xffffff;
    }
    0xffffff
}

fn dir_from_name(s: &str) -> u8 {
    match s {
        "down" | "bottom" => DOWN,
        "up" => UP,
        "north" => NORTH,
        "south" => SOUTH,
        "west" => WEST,
        "east" => EAST,
        _ => NONE,
    }
}

/// Decodes a PNG to RGBA8 the way stb_image does with 4 requested channels.
pub fn load_png_rgba(path: &str) -> Option<(i32, i32, Vec<u8>)> {
    let file = std::fs::File::open(path).ok()?;
    load_png_rgba_from(std::io::BufReader::new(file))
}

pub fn load_png_rgba_from<R: std::io::BufRead + std::io::Seek>(r: R) -> Option<(i32, i32, Vec<u8>)> {
    let mut dec = png::Decoder::new(r);
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let src = &buf[..info.buffer_size()];
    let mut out = vec![0u8; w * h * 4];
    match info.color_type {
        png::ColorType::Rgba => out.copy_from_slice(&src[..w * h * 4]),
        png::ColorType::Rgb => {
            for i in 0..w * h {
                out[i * 4..i * 4 + 3].copy_from_slice(&src[i * 3..i * 3 + 3]);
                out[i * 4 + 3] = 255;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..w * h {
                let g = src[i * 2];
                out[i * 4..i * 4 + 4].copy_from_slice(&[g, g, g, src[i * 2 + 1]]);
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..w * h {
                let g = src[i];
                out[i * 4..i * 4 + 4].copy_from_slice(&[g, g, g, 255]);
            }
        }
        png::ColorType::Indexed => return None, // EXPAND turns palettes into RGB(A)
    }
    Some((w as i32, h as i32, out))
}

fn rgba_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// Vanilla FaceInfo corner order: which of min/max each vertex takes, per direction.
// Encoded as {x, y, z} with 0 = from, 1 = to.
const CORNERS: [[[i32; 3]; 4]; 6] = [
    [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]], // down
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], // up
    [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]], // north
    [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]], // south
    [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], // west
    [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]], // east
];

fn default_uv(dir: u8, from: Vec3, to: Vec3) -> [f32; 4] {
    match dir {
        DOWN => [from.x, 16.0 - to.z, to.x, 16.0 - from.z],
        UP => [from.x, from.z, to.x, to.z],
        NORTH => [16.0 - to.x, 16.0 - to.y, 16.0 - from.x, 16.0 - from.y],
        SOUTH => [from.x, 16.0 - to.y, to.x, 16.0 - from.y],
        WEST => [from.z, 16.0 - to.y, to.z, 16.0 - from.y],
        _ => [16.0 - to.z, 16.0 - to.y, 16.0 - from.z, 16.0 - from.y],
    }
}

// Variant rotations, in 90 degree steps, about the block centre.
fn rot_x(mut p: Vec3, steps: i32) -> Vec3 {
    for _ in 0..steps {
        p = Vec3::new(p.x, p.z, 16.0 - p.y); // north -> down per step
    }
    p
}

fn rot_y(mut p: Vec3, steps: i32) -> Vec3 {
    for _ in 0..steps {
        p = Vec3::new(16.0 - p.z, p.y, p.x); // north -> east per step
    }
    p
}

fn rot_dir_x(mut d: u8, steps: i32) -> u8 {
    const NEXT: [u8; 6] = [SOUTH, NORTH, DOWN, UP, WEST, EAST];
    for _ in 0..steps {
        d = NEXT[d as usize];
    }
    d
}

fn rot_dir_y(mut d: u8, steps: i32) -> u8 {
    const NEXT: [u8; 6] = [DOWN, UP, EAST, WEST, NORTH, SOUTH];
    for _ in 0..steps {
        d = NEXT[d as usize];
    }
    d
}

fn vec3_of(j: &Value) -> Vec3 {
    Vec3::new(jf32(&j[0]), jf32(&j[1]), jf32(&j[2]))
}

/// Emitted block light, from vanilla's Blocks table for the states that matter here.
fn emission_for(b: &BlockInfo) -> u8 {
    let n = strip(&b.name);
    let lit = || b.props.get("lit").is_none_or(|v| v == "true");
    let has_lit_true = b.props.get("lit").is_some_and(|v| v == "true");
    let fixed = match n {
        "glowstone" | "sea_lantern" | "shroomlight" | "lantern" | "jack_o_lantern" | "beacon" | "lava" | "fire"
        | "conduit" | "end_gateway" | "ochre_froglight" | "verdant_froglight" | "pearlescent_froglight"
        | "copper_lantern" | "exposed_copper_lantern" | "weathered_copper_lantern" | "oxidized_copper_lantern"
        | "waxed_copper_lantern" => Some(15),
        "torch" | "wall_torch" | "end_rod" | "copper_torch" | "copper_wall_torch" => Some(14),
        "soul_lantern" | "soul_torch" | "soul_wall_torch" | "soul_fire" | "crying_obsidian" => Some(10),
        "enchanting_table" | "ender_chest" | "glow_lichen" | "redstone_torch" | "redstone_wall_torch" => Some(7),
        "amethyst_cluster" => Some(5),
        "large_amethyst_bud" => Some(4),
        "magma_block" => Some(3),
        "medium_amethyst_bud" => Some(2),
        "small_amethyst_bud" | "brewing_stand" | "brown_mushroom" | "dragon_egg" | "end_portal_frame"
        | "sculk_sensor" => Some(1),
        "nether_portal" => Some(11),
        "respawn_anchor" => Some(0),
        _ => None,
    };
    if let Some(v) = fixed {
        if n == "redstone_torch" || n == "redstone_wall_torch" {
            return if lit() { 7 } else { 0 };
        }
        return v;
    }
    if matches!(n, "redstone_lamp" | "furnace" | "blast_furnace" | "smoker" | "redstone_ore" | "deepslate_redstone_ore") {
        return if has_lit_true {
            if n == "redstone_lamp" {
                15
            } else if matches!(n, "furnace" | "blast_furnace" | "smoker") {
                13
            } else {
                9
            }
        } else {
            0
        };
    }
    if n == "campfire" {
        return if lit() { 15 } else { 0 };
    }
    if n == "soul_campfire" {
        return if lit() { 10 } else { 0 };
    }
    if n.ends_with("candle") || n.ends_with("candle_cake") {
        if !lit() || !b.props.contains_key("lit") {
            return 0;
        }
        let count: i32 = b.props.get("candles").map(|c| c.parse().unwrap_or(0)).unwrap_or(1);
        return (3 * count) as u8;
    }
    if n.ends_with("copper_bulb") {
        return if has_lit_true { 15 } else { 0 };
    }
    if n == "sea_pickle" {
        if b.props.get("waterlogged").is_some_and(|v| v == "false") {
            return 0;
        }
        let pickles: i32 = b.props.get("pickles").map_or("1", String::as_str).parse().unwrap_or(0);
        return (3 + 3 * (pickles - 1)) as u8;
    }
    if n == "light" {
        return b.props.get("level").map_or(15, |l| l.parse::<i32>().unwrap_or(0)) as u8;
    }
    0
}

impl Assets {
    /// root = extracted client jar directory.
    pub fn new(root: &str) -> Assets {
        let load_map = |rel: &str| -> Vec<u32> {
            match load_png_rgba(&format!("{root}/{rel}")) {
                Some((256, 256, px)) => {
                    px.chunks_exact(4).map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32).collect()
                }
                _ => panic!("colormap missing: {rel}"),
            }
        };
        let inner = Inner {
            state_ids: HashMap::new(),
            block_count: 0,
            texture_ids: HashMap::new(),
            texture_count: 0,
            biome_ids: HashMap::new(),
            biome_count: 0,
            baked_models: HashMap::new(),
            opaque_ids: HashMap::new(),
            grass_map: load_map("assets/minecraft/textures/colormap/grass.png"),
            foliage_map: load_map("assets/minecraft/textures/colormap/foliage.png"),
            dry_foliage_map: load_map("assets/minecraft/textures/colormap/dry_foliage.png"),
        };
        let a = Assets {
            root: root.to_string(),
            blocks: (0..MAX_BLOCKS).map(|_| OnceLock::new()).collect(),
            textures: (0..MAX_TEXTURES).map(|_| OnceLock::new()).collect(),
            biomes: (0..MAX_BIOMES).map(|_| OnceLock::new()).collect(),
            inner: Mutex::new(inner),
        };
        a.state_id("minecraft:air"); // id 0 is always air
        a.texture("minecraft:missingno");
        a.biome_id("minecraft:plains");
        a
    }

    pub fn path(&self, rel: &str) -> String {
        format!("{}/{}", self.root, rel)
    }

    #[inline]
    pub fn block(&self, id: u16) -> &BlockInfo {
        self.blocks[id as usize].get().expect("block id handed out before its slot was filled")
    }

    #[inline]
    pub fn biome(&self, id: u16) -> &Biome {
        self.biomes[id as usize].get().expect("biome id handed out before its slot was filled")
    }

    #[inline]
    pub fn tex(&self, id: u16) -> &Texture {
        self.textures[id as usize].get().expect("texture id handed out before its slot was filled")
    }

    pub fn texture_count(&self) -> usize {
        self.inner.lock().unwrap().texture_count
    }

    fn register_texture(&self, g: &mut Inner, t: Texture) -> u16 {
        let id = g.texture_count;
        assert!(id < MAX_TEXTURES, "texture table full");
        let name = t.name.clone();
        if self.textures[id].set(Box::new(t)).is_err() {
            unreachable!("texture slot {id} written twice");
        }
        g.texture_count += 1;
        g.texture_ids.insert(name, id as u16);
        id as u16
    }

    /// An arbitrary image (skins, entity textures) registered under a name.
    pub fn texture_from_pixels(&self, name: &str, w: i32, h: i32, rgba: &[u32], mips: bool) -> u16 {
        let mut g = self.inner.lock().unwrap();
        self.register_texture(&mut g, build_texture(name, w, h, rgba, mips))
    }

    pub fn texture_from_file(&self, name: &str, file: &str, mips: bool) -> u16 {
        match load_png_rgba(file) {
            Some((w, h, px)) => self.texture_from_pixels(name, w, h, &rgba_words(&px), mips),
            None => 0,
        }
    }

    /// "minecraft:block/stone"
    pub fn texture(&self, raw: &str) -> u16 {
        let mut g = self.inner.lock().unwrap();
        self.texture_locked(&mut g, raw)
    }

    fn texture_locked(&self, g: &mut Inner, raw: &str) -> u16 {
        let name = qualify(raw);
        if let Some(&id) = g.texture_ids.get(&name) {
            return id;
        }
        let file = self.path(&format!("assets/minecraft/textures/{}.png", strip(&name)));
        let Some((w, h, px)) = load_png_rgba(&file) else {
            // Magenta-black checker like vanilla so a gap is obvious in review renders.
            let checker: Vec<u32> =
                (0..256).map(|i| if ((i / 8 + i / 128) & 1) != 0 { 0xff000000 } else { 0xfff800f8 }).collect();
            if name != "minecraft:missingno" {
                eprintln!("missing texture {name}");
            }
            return self.register_texture(g, build_texture(&name, 16, 16, &checker, true));
        };
        let px = rgba_words(&px);
        let (mut frame_w, mut frame_h) = (w, w);
        let mut order: Vec<i32> = Vec::new();
        let mut frame_time = 1;
        let mut animated = false;
        if let Some(meta) = read_json(&format!("{file}.mcmeta"))
            && let Some(a) = meta.get("animation")
        {
            animated = true;
            frame_time = val_i32(a, "frametime", 1);
            if let Some(v) = a.get("width") {
                frame_w = ji32(v);
            }
            if let Some(v) = a.get("height") {
                frame_h = ji32(v);
            }
            if a.get("height").is_none() && a.get("width").is_none() {
                frame_w = w.min(h);
                frame_h = frame_w;
            }
            if let Some(frames) = a.get("frames").and_then(Value::as_array) {
                for f in frames {
                    if f.is_number() {
                        order.push(ji32(f));
                    } else {
                        let idx = val_i32(f, "index", 0);
                        let time = val_i32(f, "time", frame_time);
                        for _ in 0..(time / frame_time.max(1)).max(1) {
                            order.push(idx);
                        }
                    }
                }
            }
        }
        if !animated {
            frame_w = w;
            frame_h = h;
        }
        let count = ((w / frame_w) * (h / frame_h)).max(1);
        if order.is_empty() {
            order = (0..count).collect();
        }
        let per_row = w / frame_w;
        let copy_frame = |f: i32| -> Vec<u32> {
            let (fx, fy) = ((f % per_row) * frame_w, (f / per_row) * frame_h);
            let mut dst = vec![0u32; (frame_w * frame_h) as usize];
            for y in 0..frame_h {
                let src = ((fy + y) * w + fx) as usize;
                dst[(y * frame_w) as usize..((y + 1) * frame_w) as usize]
                    .copy_from_slice(&px[src..src + frame_w as usize]);
            }
            dst
        };
        let mut t = build_texture(&name, frame_w, frame_h, &copy_frame(0), true);
        t.frame_time = frame_time;
        if count > 1 {
            for f in 1..count {
                let extra = build_texture(&format!("{name}#frame"), frame_w, frame_h, &copy_frame(f), true);
                t.frames.push(extra.frames.into_iter().next().unwrap());
            }
            t.frame_order = order;
        }
        self.register_texture(g, t)
    }

    /// The texture with transparent texels filled by its own darkened average, like the
    /// game's fast leaves: far foliage drawn solid.
    pub fn opaque_variant(&self, id: u16) -> u16 {
        let mut g = self.inner.lock().unwrap();
        if let Some(&out) = g.opaque_ids.get(&id) {
            return out;
        }
        let t = self.tex(id);
        let mut px: Vec<u32> = t.frames[0][..(t.w * t.h) as usize].to_vec();
        let fill = pack_rgb(t.average * 0.55);
        let abgr = 0xff000000u32 | (fill & 0xff) << 16 | (fill & 0xff00) | (fill >> 16 & 0xff);
        for p in px.iter_mut() {
            if (*p >> 24) < 128 {
                *p = abgr;
            } else {
                *p |= 0xff000000;
            }
        }
        let out = self.register_texture(&mut g, build_texture(&format!("{}#opaque", t.name), t.w, t.h, &px, true));
        g.opaque_ids.insert(id, out);
        out
    }

    fn model(&self, g: &mut Inner, raw: &str, rx: i32, ry: i32, uvlock: bool) -> Arc<BlockModel> {
        let name = qualify(raw);
        let key = format!("{name}|{rx}|{ry}{}", if uvlock { "|u" } else { "" });
        if let Some(m) = g.baked_models.get(&key) {
            return m.clone();
        }

        let mut textures: HashMap<String, String> = HashMap::new();
        let mut translucent: BTreeSet<String> = BTreeSet::new(); // texture variables marked force_translucent
        let mut elements = Value::Null;
        let mut ao = true;
        let mut plant = false;
        let mut chain: Vec<String> = Vec::new();
        let mut jsons: Vec<Value> = Vec::new();
        let mut current = name.clone();
        while !current.is_empty() {
            chain.push(current.clone());
            let Some(j) = read_json(&self.path(&format!("assets/minecraft/models/{}.json", strip(&current)))) else {
                break;
            };
            current = j.get("parent").and_then(Value::as_str).map(qualify).unwrap_or_default();
            jsons.push(j);
            if chain.len() > 16 {
                break;
            }
        }
        for j in jsons.iter().rev() {
            if let Some(tex) = j.get("textures").and_then(Value::as_object) {
                for (k, v) in tex {
                    if let Some(s) = v.as_str() {
                        textures.insert(k.clone(), s.to_string());
                    } else if v.is_object() && v.get("sprite").is_some() {
                        // 26.x form: {"sprite": ..., "force_translucent": true}
                        textures.insert(k.clone(), v["sprite"].as_str().unwrap_or("").to_string());
                        if val_bool(v, "force_translucent", false) {
                            translucent.insert(k.clone());
                        } else {
                            translucent.remove(k);
                        }
                    }
                }
            }
            if let Some(e) = j.get("elements") {
                elements = e.clone();
            }
            if let Some(a) = j.get("ambientocclusion").and_then(Value::as_bool) {
                ao = a;
            }
        }
        for c in &chain {
            let s = strip(c);
            if matches!(
                s,
                "block/cross"
                    | "block/tinted_cross"
                    | "block/crop"
                    | "block/flower_pot_cross"
                    | "block/stem_growth0"
                    | "block/coral_fan"
                    | "block/cross_emissive"
            ) || s.starts_with("block/template_seagrass")
                || s.starts_with("block/flowerbed")
            {
                plant = true;
            }
        }
        let forced_translucent = |mut r: String| -> bool {
            for _ in 0..8 {
                let Some(var) = r.strip_prefix('#') else { break };
                if translucent.contains(var) {
                    return true;
                }
                match textures.get(var) {
                    Some(next) => r = next.clone(),
                    None => return false,
                }
            }
            false
        };
        let resolve = |mut r: String| -> String {
            for _ in 0..8 {
                let Some(var) = r.strip_prefix('#') else { break };
                match textures.get(var) {
                    Some(next) => r = next.clone(),
                    None => {
                        eprintln!("model {name}: unresolved texture {r}");
                        return "minecraft:missingno".to_string();
                    }
                }
            }
            r
        };

        let mut out = BlockModel { quads: Vec::new(), ao, plant };
        for el in elements.as_array().map(Vec::as_slice).unwrap_or(&[]) {
            let from = vec3_of(&el["from"]);
            let to = vec3_of(&el["to"]);
            let shade = val_bool(el, "shade", true);
            let emission = val_i32(el, "light_emission", 0);
            // Element rotation: one axis, any angle, optional rescale to keep the span.
            let mut rotated = false;
            let mut origin = Vec3::ZERO;
            let mut axis = 1;
            let mut angle = 0.0f32;
            let mut rescale = false;
            if let Some(r) = el.get("rotation")
                && r.get("angle").is_some()
                && r.get("axis").is_some()
            {
                origin = vec3_of(&r["origin"]);
                axis = match r["axis"].as_str() {
                    Some("x") => 0,
                    Some("y") => 1,
                    _ => 2,
                };
                angle = jf32(&r["angle"]) * std::f32::consts::PI / 180.0;
                rescale = val_bool(r, "rescale", false);
                rotated = angle.abs() > 1e-6;
            }
            let Some(faces) = el.get("faces").and_then(Value::as_object) else { continue };
            for (face_name, face) in faces {
                let dir = dir_from_name(face_name);
                if dir == NONE {
                    continue;
                }
                let uv = match face.get("uv") {
                    Some(u) => [jf32(&u[0]), jf32(&u[1]), jf32(&u[2]), jf32(&u[3])],
                    None => default_uv(dir, from, to),
                };
                let face_rot = val_i32(face, "rotation", 0) / 90;
                let tex_ref = val_str(face, "texture", "#missing");
                let mut q = BakedQuad {
                    tex: self.texture_locked(g, &resolve(tex_ref.clone())),
                    translucent: forced_translucent(tex_ref),
                    tint_index: val_i32(face, "tintindex", -1) as i8,
                    cull: face.get("cullface").and_then(Value::as_str).map_or(NONE, dir_from_name),
                    shade,
                    emission: emission as u8,
                    ..BakedQuad::default()
                };
                let mut pts = [Vec3::ZERO; 4];
                for i in 0..4 {
                    let c = CORNERS[dir as usize][i];
                    pts[i] = Vec3::new(
                        if c[0] != 0 { to.x } else { from.x },
                        if c[1] != 0 { to.y } else { from.y },
                        if c[2] != 0 { to.z } else { from.z },
                    );
                    let s = (i as i32 + face_rot) % 4;
                    q.uv[i] = Vec2::new(
                        (if s == 0 || s == 1 { uv[0] } else { uv[2] }) / 16.0,
                        (if s == 0 || s == 3 { uv[1] } else { uv[3] }) / 16.0,
                    );
                }
                if rotated {
                    let (c, s) = (angle.cos(), angle.sin());
                    let a = angle.abs();
                    let quarter = std::f32::consts::PI / 4.0;
                    let scale = 1.0 / (if a > quarter { std::f32::consts::PI / 2.0 - a } else { a }).cos();
                    for p in pts.iter_mut() {
                        let d = *p - origin;
                        let mut r = d;
                        match axis {
                            0 => {
                                r.y = d.y * c - d.z * s;
                                r.z = d.y * s + d.z * c;
                            }
                            1 => {
                                r.x = d.x * c + d.z * s;
                                r.z = -d.x * s + d.z * c;
                            }
                            _ => {
                                r.x = d.x * c - d.y * s;
                                r.y = d.x * s + d.y * c;
                            }
                        }
                        if rescale {
                            if axis != 0 {
                                r.x *= scale;
                            }
                            if axis != 1 {
                                r.y *= scale;
                            }
                            if axis != 2 {
                                r.z *= scale;
                            }
                        }
                        *p = origin + r;
                    }
                }
                let mut new_dir = dir;
                let mut new_cull = q.cull;
                if rx != 0 {
                    for p in pts.iter_mut() {
                        *p = rot_x(*p, rx);
                    }
                    new_dir = rot_dir_x(new_dir, rx);
                    if new_cull != NONE {
                        new_cull = rot_dir_x(new_cull, rx);
                    }
                }
                if ry != 0 {
                    for p in pts.iter_mut() {
                        *p = rot_y(*p, ry);
                    }
                    new_dir = rot_dir_y(new_dir, ry);
                    if new_cull != NONE {
                        new_cull = rot_dir_y(new_cull, ry);
                    }
                }
                if uvlock && (rx != 0 || ry != 0) && !rotated {
                    let lo = vmin(vmin(pts[0], pts[1]), vmin(pts[2], pts[3]));
                    let hi = vmax(vmax(pts[0], pts[1]), vmax(pts[2], pts[3]));
                    let luv = default_uv(new_dir, lo, hi);
                    // Re-derive corners in vanilla order for the new direction so the
                    // world-aligned uvs land on the right vertices.
                    for i in 0..4 {
                        let c = CORNERS[new_dir as usize][i];
                        pts[i] = Vec3::new(
                            if c[0] != 0 { hi.x } else { lo.x },
                            if c[1] != 0 { hi.y } else { lo.y },
                            if c[2] != 0 { hi.z } else { lo.z },
                        );
                        q.uv[i] = Vec2::new(
                            (if i == 0 || i == 1 { luv[0] } else { luv[2] }) / 16.0,
                            (if i == 0 || i == 3 { luv[1] } else { luv[3] }) / 16.0,
                        );
                    }
                }
                for i in 0..4 {
                    q.v[i] = pts[i] / 16.0;
                }
                q.cull = new_cull;
                let edge = cross(q.v[1] - q.v[0], q.v[3] - q.v[0]);
                let mut n = normalize(edge);
                if length(edge) < 1e-9 {
                    n = dir_normal(new_dir as usize);
                }
                q.normal = n;
                q.dir = if rotated { NONE } else { new_dir };
                if rotated {
                    // A rotated face can still be axis aligned (e.g. 90 degrees); keep it
                    // classified when its normal is.
                    for d in 0..6 {
                        if dot(n, dir_normal(d)) > 0.999 {
                            q.dir = d as u8;
                        }
                    }
                }
                out.quads.push(q);
            }
        }
        let out = Arc::new(out);
        g.baked_models.insert(key, out.clone());
        out
    }

    /// Thread safe, interns.
    pub fn state_id(&self, state: &str) -> u16 {
        let mut g = self.inner.lock().unwrap();
        if let Some(&id) = g.state_ids.get(state) {
            return id;
        }
        let mut info = BlockInfo::new(state);
        let bracket = state.find('[');
        info.name = qualify(match bracket {
            Some(b) => &state[..b],
            None => state,
        });
        if let Some(b) = bracket {
            let body = &state[b + 1..state.len().saturating_sub(1).max(b + 1)];
            for kv in body.split(',') {
                if let Some(eq) = kv.find('=') {
                    info.props.insert(kv[..eq].to_string(), kv[eq + 1..].to_string());
                }
            }
        }
        self.resolve_block(&mut g, &mut info);
        self.classify(&mut info);
        if g.block_count >= 65535 {
            panic!("block state table full");
        }
        let id = g.block_count;
        if self.blocks[id].set(Box::new(info)).is_err() {
            unreachable!("block slot {id} written twice");
        }
        g.block_count += 1;
        g.state_ids.insert(state.to_string(), id as u16);
        id as u16
    }

    fn resolve_block(&self, g: &mut Inner, info: &mut BlockInfo) {
        let n = strip(&info.name).to_string();
        info.air = n == "air" || n == "cave_air" || n == "void_air";
        if info.air {
            return;
        }
        let Some(bs) = read_json(&self.path(&format!("assets/minecraft/blockstates/{n}.json"))) else {
            eprintln!("no blockstate for {}", info.name);
            return;
        };
        let props = &info.props;
        let matches_cond = |condition: &str| -> bool {
            if condition.is_empty() {
                return true;
            }
            condition.split(',').all(|kv| {
                let (k, want) = match kv.find('=') {
                    Some(eq) => (&kv[..eq], &kv[eq + 1..]),
                    None => (kv, kv),
                };
                props.get(k).is_some_and(|have| have == want)
            })
        };
        let options = |g: &mut Inner, v: &Value| -> Vec<(Arc<BlockModel>, i32)> {
            let mut list = Vec::new();
            let mut add = |m: &Value| {
                let model = self.model(
                    g,
                    m["model"].as_str().unwrap_or(""),
                    val_i32(m, "x", 0) / 90 & 3,
                    val_i32(m, "y", 0) / 90 & 3,
                    val_bool(m, "uvlock", false),
                );
                list.push((model, val_i32(m, "weight", 1)));
            };
            match v.as_array() {
                Some(arr) => arr.iter().for_each(&mut add),
                None => add(v),
            }
            list
        };
        let mut parts = Vec::new();
        if let Some(variants) = bs.get("variants").and_then(Value::as_object) {
            for (cond, v) in variants {
                if matches_cond(cond) {
                    parts.push(options(g, v));
                    break;
                }
            }
        } else if let Some(multipart) = bs.get("multipart").and_then(Value::as_array) {
            fn when(w: &Value, props: &HashMap<String, String>) -> bool {
                if let Some(or) = w.get("OR") {
                    return or.as_array().is_some_and(|a| a.iter().any(|c| when(c, props)));
                }
                if let Some(and) = w.get("AND") {
                    return and.as_array().is_none_or(|a| a.iter().all(|c| when(c, props)));
                }
                let Some(obj) = w.as_object() else { return true };
                for (k, v) in obj {
                    let mut want = match v.as_str() {
                        Some(s) => s.to_string(),
                        None => v.to_string(),
                    };
                    let have = props.get(k).map(String::as_str).unwrap_or("");
                    let negate = want.starts_with('!');
                    if negate {
                        want.remove(0);
                    }
                    let any = want.split('|').any(|alt| alt == have);
                    if any == negate {
                        return false;
                    }
                }
                true
            }
            for part in multipart {
                if part.get("when").is_none_or(|w| when(w, props)) {
                    parts.push(options(g, &part["apply"]));
                }
            }
        }
        info.parts = parts;
    }

    fn classify(&self, b: &mut BlockInfo) {
        if b.air {
            return;
        }
        let n = strip(&b.name).to_string();
        let n = n.as_str();

        if n == "water" || n == "bubble_column" {
            b.fluid = 1;
            b.fluid_level = if n == "water" { b.prop("level").parse::<i32>().unwrap_or(0) as u8 } else { 0 };
        } else if n == "lava" {
            b.fluid = 2;
            b.fluid_level = b.prop("level").parse::<i32>().unwrap_or(0) as u8;
        }
        b.waterlogged = b.prop("waterlogged") == "true"
            || matches!(n, "kelp" | "kelp_plant" | "seagrass" | "tall_seagrass" | "bubble_column");

        // Occlusion: a face on each side that covers the whole block with an opaque texture.
        let mut covered = [false; 6];
        let mut full = [false; 6];
        for part in &b.parts {
            let Some((m, _)) = part.first() else { continue };
            for q in &m.quads {
                if q.cull == NONE || q.dir != q.cull {
                    continue;
                }
                let lo = vmin(vmin(q.v[0], q.v[1]), vmin(q.v[2], q.v[3]));
                let hi = vmax(vmax(q.v[0], q.v[1]), vmax(q.v[2], q.v[3]));
                let spans = match q.dir / 2 {
                    // 0 = y, 1 = z, 2 = x
                    0 => lo.x <= 0.001 && hi.x >= 0.999 && lo.z <= 0.001 && hi.z >= 0.999,
                    1 => lo.x <= 0.001 && hi.x >= 0.999 && lo.y <= 0.001 && hi.y >= 0.999,
                    _ => lo.z <= 0.001 && hi.z >= 0.999 && lo.y <= 0.001 && hi.y >= 0.999,
                };
                if !spans {
                    continue;
                }
                let t = self.tex(q.tex);
                full[q.dir as usize] = true;
                if !(t.has_cutout || t.has_translucent) {
                    covered[q.dir as usize] = true;
                }
            }
        }
        b.full_cube = full.iter().all(|&f| f);
        b.full_opaque = covered.iter().all(|&c| c);
        b.leaves = n.ends_with("_leaves");
        if b.leaves {
            b.full_opaque = false;
        }
        // Blocks that vanilla renders translucent even though every face is "full".
        if matches!(
            n,
            "ice" | "slime_block" | "honey_block" | "tinted_glass" | "glass" | "barrier" | "spawner" | "trial_spawner"
                | "vault"
        ) || n.ends_with("stained_glass")
        {
            b.full_opaque = false;
        }

        if b.full_opaque {
            b.light_block = 15;
        } else if b.leaves
            || b.fluid == 1
            || b.waterlogged
            || matches!(n, "ice" | "frosted_ice" | "cobweb" | "slime_block" | "honey_block")
        {
            b.light_block = 1;
        } else if b.fluid == 2 {
            b.light_block = 15;
        }
        b.emission = emission_for(b);

        // Vanilla BlockColors: which blocks tint their tintindex faces, and with what.
        b.tint = match n {
            "grass_block" | "short_grass" | "tall_grass" | "fern" | "large_fern" | "potted_fern" | "sugar_cane"
            | "bush" | "pink_petals" | "wildflowers" => Tint::Grass,
            "oak_leaves" | "jungle_leaves" | "acacia_leaves" | "dark_oak_leaves" | "mangrove_leaves" | "vine" => {
                Tint::Foliage
            }
            "leaf_litter" => Tint::DryFoliage,
            "water" | "bubble_column" | "water_cauldron" => Tint::Water,
            "redstone_wire" => Tint::Redstone,
            "melon_stem" | "pumpkin_stem" => Tint::Stem,
            _ => b.tint,
        };
        let fixed = match n {
            "birch_leaves" => Some(0x80a755),
            "spruce_leaves" => Some(0x619961),
            "lily_pad" => Some(0x208030),
            "attached_melon_stem" | "attached_pumpkin_stem" => Some(0xe0c71c),
            _ => None,
        };
        if let Some(c) = fixed {
            b.tint = Tint::Fixed;
            b.fixed_tint = c;
        }

        let plant = b.parts.iter().flatten().any(|(m, _)| m.plant);
        if plant
            || matches!(
                n,
                "short_grass"
                    | "tall_grass"
                    | "fern"
                    | "large_fern"
                    | "sugar_cane"
                    | "sweet_berry_bush"
                    | "bush"
                    | "dead_bush"
                    | "vine"
                    | "wheat"
                    | "carrots"
                    | "potatoes"
                    | "beetroots"
            )
            || n.ends_with("_sapling")
        {
            b.mat |= MAT_PLANT;
        }
        if b.leaves {
            b.mat |= MAT_LEAVES;
        }
        if b.fluid == 1 {
            b.mat |= MAT_WATER | MAT_NO_SHADOW;
        }
        if b.fluid == 2 {
            b.mat |= MAT_LAVA | MAT_EMISSIVE;
        }
        if b.emission >= 7 {
            b.mat |= MAT_EMISSIVE;
        }
        if n.contains("glass") {
            b.mat |= MAT_GLASS;
        }
        if matches!(n, "iron_block" | "gold_block" | "iron_bars" | "chain" | "lantern" | "anvil") || n.contains("copper") {
            b.mat |= MAT_METAL;
        }

        b.special = if matches!(n, "chest" | "trapped_chest" | "ender_chest") || n.ends_with("copper_chest") {
            Special::Chest
        } else if n.ends_with("_bed") {
            Special::Bed
        } else if n.ends_with("_wall_hanging_sign") || n.ends_with("_hanging_sign") {
            Special::HangingSign
        } else if n.ends_with("_wall_sign") {
            Special::WallSign
        } else if n.ends_with("_sign") {
            Special::Sign
        } else if n.ends_with("_skull") || n.ends_with("_head") {
            Special::Skull
        } else if n.ends_with("_banner") {
            Special::Banner
        } else if n.ends_with("shulker_box") {
            Special::Shulker
        } else {
            Special::None
        };

        let mut avg = Vec3::ZERO;
        let mut count = 0;
        for part in &b.parts {
            if let Some((m, _)) = part.first() {
                for q in &m.quads {
                    if q.dir == UP || q.dir == NONE {
                        avg += self.tex(q.tex).average;
                        count += 1;
                    }
                }
            }
        }
        b.map_color = if count > 0 { pack_rgb(avg / count as f32) } else { 0x808080 };
    }

    pub fn biome_id(&self, raw: &str) -> u16 {
        let mut g = self.inner.lock().unwrap();
        let name = qualify(raw);
        if let Some(&id) = g.biome_ids.get(&name) {
            return id;
        }
        let mut b = Biome {
            name: name.clone(),
            temperature: 0.8,
            downfall: 0.4,
            water: 0x3f76e4,
            grass_override: -1,
            foliage_override: -1,
            dry_foliage_override: -1,
            grass_modifier: 0,
            sky: 0x78a7ff,
            grass: 0,
            foliage: 0,
            dry_foliage: 0,
        };
        if let Some(j) = read_json(&self.path(&format!("data/minecraft/worldgen/biome/{}.json", strip(&name)))) {
            b.temperature = val_f32(&j, "temperature", 0.8);
            b.downfall = val_f32(&j, "downfall", 0.4);
            if let Some(e) = j.get("effects") {
                if let Some(c) = e.get("water_color") {
                    b.water = parse_color(c);
                }
                if let Some(c) = e.get("grass_color") {
                    b.grass_override = parse_color(c) as i32;
                }
                if let Some(c) = e.get("foliage_color") {
                    b.foliage_override = parse_color(c) as i32;
                }
                if let Some(c) = e.get("dry_foliage_color") {
                    b.dry_foliage_override = parse_color(c) as i32;
                }
                b.grass_modifier = match val_str(e, "grass_color_modifier", "none").as_str() {
                    "dark_forest" => 1,
                    "swamp" => 2,
                    _ => 0,
                };
            }
            if let Some(c) = j.get("attributes").and_then(|a| a.get("minecraft:visual/sky_color")) {
                b.sky = parse_color(c);
            }
        } else {
            eprintln!("unknown biome {name}, using plains colours");
        }
        let t = b.temperature.clamp(0.0, 1.0);
        let d = b.downfall.clamp(0.0, 1.0) * t;
        let idx = (((1.0 - d) * 255.0) as i32 * 256 + ((1.0 - t) * 255.0) as i32) as usize;
        b.grass = if b.grass_override >= 0 { b.grass_override as u32 } else { g.grass_map[idx] };
        b.foliage = if b.foliage_override >= 0 { b.foliage_override as u32 } else { g.foliage_map[idx] };
        b.dry_foliage = if b.dry_foliage_override >= 0 { b.dry_foliage_override as u32 } else { g.dry_foliage_map[idx] };
        if b.grass_modifier == 1 {
            b.grass = ((b.grass & 0xfefefe) + 0x28340a) >> 1;
        }
        if b.grass_modifier == 2 {
            b.grass = 0x6a7039;
        }
        let id = g.biome_count;
        assert!(id < MAX_BIOMES, "biome table full");
        if self.biomes[id].set(b).is_err() {
            unreachable!("biome slot {id} written twice");
        }
        g.biome_count += 1;
        g.biome_ids.insert(name, id as u16);
        id as u16
    }
}

fn build_texture(name: &str, w: i32, h: i32, rgba: &[u32], mips: bool) -> Texture {
    let mut t = Texture {
        name: name.to_string(),
        w,
        h,
        levels: 1,
        frames: vec![rgba[..(w * h) as usize].to_vec()],
        level_offset: vec![0],
        frame_order: vec![0],
        frame_time: 1,
        ..Texture::default()
    };
    let mut sum = Vec3::ZERO;
    let mut opaque = 0;
    for &p in &rgba[..(w * h) as usize] {
        let a = p >> 24;
        if a == 0 {
            t.has_cutout = true;
        } else if a < 255 {
            t.has_translucent = true;
        }
        if a > 0 {
            sum += Vec3::new(lin(p as u8), lin((p >> 8) as u8), lin((p >> 16) as u8));
            opaque += 1;
        }
    }
    t.average = if opaque > 0 { sum / opaque as f32 } else { Vec3::new(0.5, 0.5, 0.5) };
    if mips && w == h {
        let frame = &mut t.frames[0];
        let mut size = w;
        let mut prev_offset = 0usize;
        while size > 1 {
            let next = size / 2;
            let offset = frame.len();
            frame.resize(frame.len() + (next * next) as usize, 0);
            for y in 0..next {
                for x in 0..next {
                    let mut c = Vec3::ZERO;
                    let mut alpha = 0.0f32;
                    let mut weight = 0.0f32;
                    for k in 0..4 {
                        let p = frame[prev_offset + ((y * 2 + k / 2) * size + x * 2 + k % 2) as usize];
                        let a = (p >> 24) as f32 / 255.0;
                        c += Vec3::new(lin(p as u8), lin((p >> 8) as u8), lin((p >> 16) as u8)) * a;
                        weight += a;
                        alpha += a;
                    }
                    c = if weight > 0.0 { c / weight } else { Vec3::ZERO };
                    // Cutouts keep their coverage at a distance instead of thinning out: the
                    // renderer tests alpha against 0.1 like vanilla's mipped cutout layer.
                    let packed = pack_rgb(c);
                    let a = (clamp01(alpha / 4.0) * 255.0).round() as u32;
                    frame[offset + (y * next + x) as usize] =
                        (a << 24) | (packed & 0xff) << 16 | (packed & 0xff00) | (packed >> 16 & 0xff);
                }
            }
            t.level_offset.push(offset as i32);
            t.levels += 1;
            prev_offset = offset;
            size = next;
        }
    }
    t
}
