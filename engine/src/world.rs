//! The terrain the engine knows about. Written by the network thread, read by the
//! mesher and lighter; a reader-writer lock keeps both sides simple since writes are
//! rare next to reads.

use crate::assets::{Assets, MAT_PLANT};
use crate::vmath::Vec3;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

pub struct Section {
    pub blocks: Packed,    // (y * 16 + z) * 16 + x
    pub biomes: [u16; 64], // 4x4x4 cells, same order
    pub light: Packed,     // sky << 4 | block
    pub empty: bool,
    pub version: u32, // bumped on block or light change
    stamp: u64,       // world-unique, renewed when the blocks or biomes change
}

impl Section {
    /// Air in open sky, which is also how a missing section reads.
    fn new() -> Box<Section> {
        Box::new(Section { blocks: Packed::uniform(0), biomes: [0; 64], light: Packed::uniform(0xf0), empty: true, version: 0, stamp: 0 })
    }
}

/// A section's 4096 values (block ids, or light), in as few bits each as the number of
/// different values among them needs: a section of a dozen kinds of block takes 2 KiB
/// rather than 8, one of a single light level nothing past its header.
#[derive(Clone)]
pub struct Packed {
    palette: Vec<u16>, // the values the indices stand for; unused at 16 bits, which hold values
    bits: u32,         // 0, 1, 2, 4, 8 or 16
    words: Box<[u64]>,
}

impl Packed {
    pub fn uniform(v: u16) -> Packed {
        Packed { palette: vec![v], bits: 0, words: Box::new([]) }
    }

    pub fn new(values: &[u16; 4096]) -> Packed {
        // Values to palette indices through a small open-addressed table.
        const SLOTS: usize = 1024;
        let mut keys = [0u16; SLOTS];
        let mut index = [u16::MAX; SLOTS];
        let mut palette: Vec<u16> = Vec::new();
        let mut idx = [0u16; 4096];
        let mut last = (u16::MAX, u16::MAX); // value, its index: runs are common
        for (i, &v) in values.iter().enumerate() {
            if v == last.0 && last.1 != u16::MAX {
                idx[i] = last.1;
                continue;
            }
            let mut h = (v as usize).wrapping_mul(0x9e37) >> 6 & (SLOTS - 1);
            let at = loop {
                if index[h] == u16::MAX {
                    if palette.len() == 256 {
                        // Too many to be worth a palette: the values themselves.
                        return Packed::raw(values);
                    }
                    keys[h] = v;
                    index[h] = palette.len() as u16;
                    palette.push(v);
                    break index[h];
                }
                if keys[h] == v {
                    break index[h];
                }
                h = (h + 1) & (SLOTS - 1);
            };
            idx[i] = at;
            last = (v, at);
        }
        let bits = match palette.len() {
            1 => return Packed::uniform(palette[0]),
            2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        let mut words = vec![0u64; 4096 * bits as usize / 64].into_boxed_slice();
        for (i, &x) in idx.iter().enumerate() {
            let at = i * bits as usize;
            words[at / 64] |= (x as u64) << (at % 64);
        }
        palette.shrink_to_fit();
        Packed { palette, bits, words }
    }

    fn raw(values: &[u16; 4096]) -> Packed {
        let mut words = vec![0u64; 1024].into_boxed_slice();
        for (i, &v) in values.iter().enumerate() {
            words[i / 4] |= (v as u64) << (16 * (i % 4));
        }
        Packed { palette: Vec::new(), bits: 16, words }
    }

    pub fn from_u8(values: &[u8; 4096]) -> Packed {
        Packed::new(&values.map(|v| v as u16))
    }

    #[inline]
    pub fn get(&self, i: usize) -> u16 {
        match self.bits {
            0 => self.palette[0],
            16 => (self.words[i / 4] >> (16 * (i % 4))) as u16,
            bits => {
                let at = i * bits as usize;
                let x = (self.words[at / 64] >> (at % 64)) & ((1 << bits) - 1);
                self.palette[x as usize]
            }
        }
    }

    pub fn unpack(&self) -> [u16; 4096] {
        let mut out = [0u16; 4096];
        match self.bits {
            0 => out.fill(self.palette[0]),
            16 => {
                for (i, w) in self.words.iter().enumerate() {
                    for k in 0..4 {
                        out[i * 4 + k] = (w >> (16 * k)) as u16;
                    }
                }
            }
            bits => {
                let per = 64 / bits as usize;
                let mask = (1u64 << bits) - 1;
                for (i, w) in self.words.iter().enumerate() {
                    for k in 0..per {
                        out[i * per + k] = self.palette[((w >> (k * bits as usize)) & mask) as usize];
                    }
                }
            }
        }
        out
    }

    pub fn unpack_u8(&self) -> [u8; 4096] {
        std::array::from_fn(|i| self.get(i) as u8)
    }

    pub fn all(&self, v: u16) -> bool {
        self.bits == 0 && self.palette[0] == v
    }

    /// Heap bytes held.
    pub fn bytes(&self) -> usize {
        self.palette.capacity() * 2 + self.words.len() * 8
    }
}

pub fn section_key(cx: i32, sy: i32, cz: i32) -> u64 {
    ((cx as u32 as u64 & 0x1fffff) << 43) | ((cz as u32 as u64 & 0x1fffff) << 22) | (sy as u32 as u64 & 0x3fffff)
}

pub fn column_key(cx: i32, cz: i32) -> u64 {
    ((cx as u32 as u64) << 32) | cz as u32 as u64
}

const UNKNOWN: u16 = 0xffff;
const TILE: i32 = 4; // chunks per relight tile edge
const MARGIN: i32 = 16; // light never travels further than this
const REACH: i32 = 15; // nor does a change of it, sideways from the block that caused it
// Past this many edits between two passes of the world worker, it stops listing them and
// treats the lot as a bulk change: a snapshot or a big WorldEdit, not someone building.
const MAX_EDITS: usize = 256;

/// Blocks that changed in one section: an inclusive box in world block coordinates, and
/// when the feed delivered it (engine clock).
#[derive(Clone, Copy, Debug)]
pub struct Edit {
    pub lo: [i32; 3],
    pub hi: [i32; 3],
    pub arrived: f64,
}

/// Everything that changed since the world worker last looked.
#[derive(Default)]
pub struct Changes {
    pub columns: Vec<(i32, i32)>,
    pub sections: Vec<u64>, // whose blocks changed
    pub edits: Vec<Edit>,   // the same changes block by block, unless `bulk`
    pub bulk: bool,
}

/// A rectangle of block columns, end exclusive, lit over the world's full height.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Rect {
    x0: i32,
    z0: i32,
    x1: i32,
    z1: i32,
}

impl Rect {
    fn overlaps(&self, o: &Rect) -> bool {
        self.x0 < o.x1 && o.x0 < self.x1 && self.z0 < o.z1 && o.z0 < self.z1
    }
}

fn floor_div(a: i32, b: i32) -> i32 {
    (if a >= 0 { a } else { a - b + 1 }) / b
}

/// An 18^3 window around a section (one block of border), for the mesher. Missing
/// neighbours read as air with full sky light.
pub struct Window {
    pub blocks: [u16; 18 * 18 * 18],
    pub light: [u8; 18 * 18 * 18],
    pub biomes: [u16; 64],
}

impl Window {
    pub fn new() -> Box<Window> {
        Box::new(Window { blocks: [0; 5832], light: [0; 5832], biomes: [0; 64] })
    }

    #[inline]
    pub fn idx(x: i32, y: i32, z: i32) -> usize {
        (((y + 1) * 18 + (z + 1)) * 18 + (x + 1)) as usize
    }
}

#[derive(Default)]
struct Inner {
    sections: HashMap<u64, Box<Section>>,
    columns: HashMap<u64, i32>, // column -> section count, for bounds
}

impl Inner {
    fn section(&self, cx: i32, sy: i32, cz: i32) -> Option<&Section> {
        self.sections.get(&section_key(cx, sy, cz)).map(|b| &**b)
    }

    fn has_column(&self, cx: i32, cz: i32) -> bool {
        self.columns.contains_key(&column_key(cx, cz))
    }
}

#[derive(Default)]
struct Dirty {
    columns: BTreeSet<(i32, i32)>,
    changed: BTreeSet<u64>,
    edits: Vec<Edit>,
    bulk: bool,
}

pub struct World {
    assets: Arc<Assets>,
    min_y: i32,
    height: i32,
    inner: RwLock<Inner>,
    dirty: Mutex<Dirty>,
    wake: Condvar,
    revision: AtomicU32,
    stamps: AtomicU64,
    // section_hashes' last results, by section and stamp: most sections never change.
    hashed: Mutex<HashMap<u64, (u64, u64)>>,
}

impl World {
    pub fn new(assets: Arc<Assets>, min_y: i32, height: i32) -> World {
        World {
            assets,
            min_y,
            height,
            inner: RwLock::new(Inner::default()),
            dirty: Mutex::new(Dirty::default()),
            wake: Condvar::new(),
            revision: AtomicU32::new(0),
            stamps: AtomicU64::new(1),
            hashed: Mutex::new(HashMap::new()),
        }
    }

    pub fn assets(&self) -> &Assets {
        &self.assets
    }

    pub fn min_y(&self) -> i32 {
        self.min_y
    }

    /// The key of every section held.
    pub fn section_keys(&self) -> Vec<u64> {
        self.inner.read().unwrap().sections.keys().copied().collect()
    }

    /// How many sections are held, and the bytes they take.
    pub fn section_memory(&self) -> (usize, usize) {
        let g = self.inner.read().unwrap();
        let bytes = g.sections.values().map(|s| std::mem::size_of::<Section>() + s.blocks.bytes() + s.light.bytes()).sum();
        (g.sections.len(), bytes)
    }

    pub fn min_section(&self) -> i32 {
        self.min_y >> 4
    }

    pub fn max_section(&self) -> i32 {
        (self.min_y + self.height - 1) >> 4
    }

    pub fn revision(&self) -> u32 {
        self.revision.load(Ordering::SeqCst)
    }

    fn mark(&self, cx: i32, sy: i32, cz: i32, lo: [i32; 3], hi: [i32; 3]) {
        let (x0, y0, z0) = (cx * 16, sy * 16, cz * 16);
        let edit = Edit { lo: [x0 + lo[0], y0 + lo[1], z0 + lo[2]], hi: [x0 + hi[0], y0 + hi[1], z0 + hi[2]], arrived: crate::net::engine_clock() };
        let mut d = self.dirty.lock().unwrap();
        d.columns.insert((cx, cz));
        d.changed.insert(section_key(cx, sy, cz));
        if !d.bulk {
            d.edits.push(edit);
            if d.edits.len() > MAX_EDITS {
                d.bulk = true;
                d.edits = Vec::new();
            }
        }
        self.revision.fetch_add(1, Ordering::SeqCst);
        self.wake.notify_all();
    }

    /// Network side.
    pub fn set_section(&self, cx: i32, sy: i32, cz: i32, blocks: &[u16; 4096], biomes: &[u16; 64]) {
        let (mut lo, mut hi) = ([0, 0, 0], [15, 15, 15]);
        {
            let mut g = self.inner.write().unwrap();
            let key = section_key(cx, sy, cz);
            match g.sections.get(&key) {
                // A reconnecting observer resends the whole area; what did not change
                // costs nothing to relight or remesh.
                Some(s) if s.biomes == *biomes && s.blocks.unpack() == *blocks => return,
                // Only the blocks that differ: one placed block is a one-block edit.
                Some(s) if s.biomes == *biomes => {
                    (lo, hi) = ([15, 15, 15], [0, 0, 0]);
                    for (i, _) in s.blocks.unpack().iter().zip(blocks).enumerate().filter(|(_, (a, b))| a != b) {
                        let at = [i as i32 & 15, i as i32 >> 8, (i as i32 >> 4) & 15];
                        for k in 0..3 {
                            lo[k] = lo[k].min(at[k]);
                            hi[k] = hi[k].max(at[k]);
                        }
                    }
                }
                Some(_) => {}
                None => {
                    g.sections.insert(key, Section::new());
                    *g.columns.entry(column_key(cx, cz)).or_insert(0) += 1;
                }
            }
            let slot = g.sections.get_mut(&key).unwrap();
            slot.blocks = Packed::new(blocks);
            slot.biomes = *biomes;
            slot.stamp = self.stamps.fetch_add(1, Ordering::Relaxed);
            slot.empty = slot.blocks.all(0);
            slot.version = slot.version.wrapping_add(1);
        }
        self.mark(cx, sy, cz, lo, hi);
    }

    pub fn clear_section(&self, cx: i32, sy: i32, cz: i32) {
        {
            let mut g = self.inner.write().unwrap();
            let Some(s) = g.sections.get_mut(&section_key(cx, sy, cz)) else { return };
            if s.empty {
                return;
            }
            s.blocks = Packed::uniform(0);
            s.stamp = self.stamps.fetch_add(1, Ordering::Relaxed);
            s.empty = true;
            s.version = s.version.wrapping_add(1);
        }
        self.mark(cx, sy, cz, [0, 0, 0], [15, 15, 15]);
    }

    /// Waits up to `timeout` for something to change, returning at once if something
    /// already has.
    pub fn wait(&self, timeout: Duration) {
        let d = self.dirty.lock().unwrap();
        if d.changed.is_empty() {
            drop(self.wake.wait_timeout(d, timeout).unwrap());
        }
    }

    pub fn take_changes(&self) -> Changes {
        let d = std::mem::take(&mut *self.dirty.lock().unwrap());
        Changes { columns: d.columns.into_iter().collect(), sections: d.changed.into_iter().collect(), edits: d.edits, bulk: d.bulk }
    }

    /// Every section's blocks and biomes as a hash: what a run keeps of the world its
    /// recordings show, to tell what changed by the next run.
    pub fn section_hashes(&self) -> Vec<(u64, u64)> {
        let g = self.inner.read().unwrap();
        // Block and biome ids are handed out per run, in the order the feed names them:
        // what is hashed is each id's own hash of its state string, which is the same in
        // every run.
        let mut names = vec![0u64; 1 << 16];
        let mut name = |id: u16| {
            let n = &mut names[id as usize];
            if *n == 0 {
                *n = crate::persist::hash(self.assets.block(id).state.as_bytes(), 1) | 1;
            }
            *n
        };
        let mut biomes: HashMap<u16, u64> = HashMap::new();
        let mut biome = |id: u16| *biomes.entry(id).or_insert_with(|| crate::persist::hash(self.assets.biome(id).name.as_bytes(), 2));
        let mut words = vec![0u8; 8 * (4096 + 64)];
        let old = std::mem::take(&mut *self.hashed.lock().unwrap());
        let mut kept = HashMap::with_capacity(old.len());
        let mut out: Vec<(u64, u64)> = g
            .sections
            .iter()
            .filter(|(_, s)| !s.empty)
            .map(|(&k, s)| {
                let h = match old.get(&k) {
                    Some(&(stamp, h)) if stamp == s.stamp => h,
                    _ => {
                        let ids = s.blocks.unpack().into_iter().map(&mut name).chain(s.biomes.iter().map(|&b| biome(b)));
                        for (w, v) in words.chunks_exact_mut(8).zip(ids) {
                            w.copy_from_slice(&v.to_le_bytes());
                        }
                        crate::persist::hash(&words, 0x5ec7)
                    }
                };
                kept.insert(k, (s.stamp, h));
                (k, h)
            })
            .collect();
        *self.hashed.lock().unwrap() = kept;
        out.sort_unstable();
        out
    }

    /// Whether the feed has changed anything the world worker has not taken yet.
    pub fn pending(&self) -> bool {
        !self.dirty.lock().unwrap().changed.is_empty()
    }

    /// Recompute light for the given chunk columns (and nothing else), a tile of them at
    /// a time. Returns keys of sections whose light changed, which the mesher must
    /// rebuild along with their neighbours.
    pub fn relight(&self, columns: &[(i32, i32)]) -> Vec<u64> {
        let tiles: BTreeSet<(i32, i32)> = columns.iter().map(|&(cx, cz)| (floor_div(cx, TILE), floor_div(cz, TILE))).collect();
        let size = TILE * 16;
        self.relight_rects(&tiles.into_iter().map(|(tx, tz)| Rect { x0: tx * size, z0: tz * size, x1: tx * size + size, z1: tz * size + size }).collect::<Vec<_>>())
    }

    /// Recompute light only as far as these edits can have changed it: REACH blocks to
    /// every side of each, over the full height (a block high up shades its whole
    /// column). For one placed block that is under half a tile, where every tile the
    /// edited columns touched, up to four, used to be relit.
    pub fn relight_around(&self, edits: &[Edit]) -> Vec<u64> {
        let mut rects: Vec<Rect> = Vec::new();
        for e in edits {
            let mut r = Rect { x0: e.lo[0] - REACH, z0: e.lo[2] - REACH, x1: e.hi[0] + REACH + 1, z1: e.hi[2] + REACH + 1 };
            // Overlapping areas are lit once, as their bounding rectangle.
            while let Some(i) = rects.iter().position(|o| o.overlaps(&r)) {
                let o = rects.swap_remove(i);
                r = Rect { x0: r.x0.min(o.x0), z0: r.z0.min(o.z0), x1: r.x1.max(o.x1), z1: r.z1.max(o.z1) };
            }
            rects.push(r);
        }
        self.relight_rects(&rects)
    }

    fn relight_rects(&self, list: &[Rect]) -> Vec<u64> {
        let changed: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        let (min_section, max_section, min_y) = (self.min_section(), self.max_section(), self.min_y);
        let work = || {
            let y_size = self.height;
            let (mut opacity, mut sky, mut blk, mut emission) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut queue: Vec<u32> = Vec::new();
            loop {
                let t = next.fetch_add(1, Ordering::SeqCst);
                if t >= list.len() {
                    break;
                }
                let rect = list[t];
                let (bx0, bz0) = (rect.x0 - MARGIN, rect.z0 - MARGIN);
                let (x_size, z_size) = (rect.x1 - rect.x0 + 2 * MARGIN, rect.z1 - rect.z0 + 2 * MARGIN);
                let n = (x_size * z_size * y_size) as usize;
                for v in [&mut opacity, &mut sky, &mut blk, &mut emission] {
                    v.clear();
                    v.resize(n, 0);
                }
                let (xs, zs) = (x_size as usize, z_size as usize);
                let at = |x: i32, y: i32, z: i32| (y as usize * zs + z as usize) * xs + x as usize;
                {
                    let g = self.inner.read().unwrap();
                    for ccz in floor_div(bz0, 16)..=floor_div(bz0 + z_size - 1, 16) {
                        for ccx in floor_div(bx0, 16)..=floor_div(bx0 + x_size - 1, 16) {
                            for sy in min_section..=max_section {
                                let Some(s) = g.section(ccx, sy, ccz) else { continue };
                                if s.empty {
                                    continue;
                                }
                                let blocks = s.blocks.unpack();
                                for ly in 0..16 {
                                    let y = sy * 16 + ly - min_y;
                                    for lz in 0..16 {
                                        let z = ccz * 16 + lz - bz0;
                                        if !(0..z_size).contains(&z) {
                                            continue;
                                        }
                                        for lx in 0..16 {
                                            let x = ccx * 16 + lx - bx0;
                                            if !(0..x_size).contains(&x) {
                                                continue;
                                            }
                                            let b = self.assets.block(blocks[((ly * 16 + lz) * 16 + lx) as usize]);
                                            let i = at(x, y, z);
                                            opacity[i] = b.light_block;
                                            emission[i] = b.emission;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Sky: straight down from the top, then spread.
                queue.clear();
                for z in 0..z_size {
                    for x in 0..x_size {
                        let mut level: i32 = 15;
                        for y in (0..y_size).rev() {
                            let i = at(x, y, z);
                            let op = opacity[i] as i32;
                            if op >= 15 {
                                level = 0;
                            } else if op > 0 {
                                level = (level - op).max(0);
                            }
                            sky[i] = level as u8;
                        }
                    }
                }
                let plane = (x_size * z_size) as usize;
                for y in 0..y_size {
                    for z in 0..z_size {
                        for x in 0..x_size {
                            let i = at(x, y, z);
                            if sky[i] < 2 {
                                continue;
                            }
                            let s1 = sky[i] as i32 - 1;
                            let edge = (x > 0 && (sky[i - 1] as i32) < s1)
                                || (x < x_size - 1 && (sky[i + 1] as i32) < s1)
                                || (z > 0 && (sky[i - xs] as i32) < s1)
                                || (z < z_size - 1 && (sky[i + xs] as i32) < s1)
                                || (y > 0 && (sky[i - plane] as i32) < s1);
                            if edge {
                                queue.push(i as u32);
                            }
                        }
                    }
                }
                let spread = |level: &mut [u8], queue: &mut Vec<u32>| {
                    let mut h = 0;
                    while h < queue.len() {
                        let i = queue[h] as usize;
                        h += 1;
                        let (x, z, y) = ((i % xs) as i32, (i / xs % zs) as i32, (i / plane) as i32);
                        let l = level[i] as i32;
                        let mut visit = |n: usize| {
                            let op = opacity[n] as i32;
                            if op >= 15 {
                                return;
                            }
                            let nl = l - op.max(1);
                            if nl > level[n] as i32 {
                                level[n] = nl as u8;
                                if nl > 1 {
                                    queue.push(n as u32);
                                }
                            }
                        };
                        if x > 0 {
                            visit(i - 1);
                        }
                        if x < x_size - 1 {
                            visit(i + 1);
                        }
                        if z > 0 {
                            visit(i - xs);
                        }
                        if z < z_size - 1 {
                            visit(i + xs);
                        }
                        if y > 0 {
                            visit(i - plane);
                        }
                        if y < y_size - 1 {
                            visit(i + plane);
                        }
                    }
                };
                spread(&mut sky, &mut queue);
                queue.clear();
                for i in 0..n {
                    blk[i] = emission[i];
                    if emission[i] > 1 {
                        queue.push(i as u32);
                    }
                }
                spread(&mut blk, &mut queue);

                // Write back the interior, creating air sections up to one above the
                // highest real one so faces under overhangs read correct light. Outside
                // the interior, light is not known exactly and stays as it was.
                let mut g = self.inner.write().unwrap();
                for ccz in floor_div(rect.z0, 16)..=floor_div(rect.z1 - 1, 16) {
                    for ccx in floor_div(rect.x0, 16)..=floor_div(rect.x1 - 1, 16) {
                        if !g.has_column(ccx, ccz) {
                            continue;
                        }
                        let (lx0, lx1) = ((rect.x0 - ccx * 16).max(0), (rect.x1 - ccx * 16).min(16));
                        let (lz0, lz1) = ((rect.z0 - ccz * 16).max(0), (rect.z1 - ccz * 16).min(16));
                        let mut top = min_section - 1;
                        for sy in min_section..=max_section {
                            if g.section(ccx, sy, ccz).is_some_and(|s| !s.empty) {
                                top = sy;
                            }
                        }
                        for sy in min_section..=(top + 1).min(max_section) {
                            let key = section_key(ccx, sy, ccz);
                            if !g.sections.contains_key(&key) {
                                g.sections.insert(key, Section::new());
                                *g.columns.entry(column_key(ccx, ccz)).or_insert(0) += 1;
                            }
                            let slot = g.sections.get_mut(&key).unwrap();
                            let old = slot.light.unpack_u8();
                            let mut fresh = old;
                            for ly in 0..16 {
                                let y = sy * 16 + ly - min_y;
                                for lz in lz0..lz1 {
                                    for lx in lx0..lx1 {
                                        let i = at(ccx * 16 + lx - bx0, y, ccz * 16 + lz - bz0);
                                        fresh[((ly * 16 + lz) * 16 + lx) as usize] = sky[i] << 4 | blk[i];
                                    }
                                }
                            }
                            if fresh != old {
                                slot.light = Packed::from_u8(&fresh);
                                slot.version = slot.version.wrapping_add(1);
                                changed.lock().unwrap().push(key);
                            }
                        }
                    }
                }
            }
        };
        let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = list.len().min(hw).max(1);
        std::thread::scope(|s| {
            for _ in 1..threads {
                s.spawn(&work);
            }
            work();
        });
        let mut changed = changed.into_inner().unwrap();
        changed.sort_unstable();
        changed.dedup();
        self.revision.fetch_add(1, Ordering::SeqCst);
        changed
    }

    /// Fills `out` with the section and a one-block border; false when the section is
    /// missing or empty. Returns the section's version alongside.
    pub fn window(&self, cx: i32, sy: i32, cz: i32, out: &mut Window) -> Option<u32> {
        let g = self.inner.read().unwrap();
        let this = g.section(cx, sy, cz)?;
        if this.empty {
            return None;
        }
        let version = this.version;
        out.biomes = this.biomes;
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let s = g.section(cx + dx, sy + dy, cz + dz);
                    let column_known = g.has_column(cx + dx, cz + dz);
                    let (xs, xe) = (if dx < 0 { 15 } else { 0 }, if dx > 0 { 0 } else { 15 });
                    let (zs, ze) = (if dz < 0 { 15 } else { 0 }, if dz > 0 { 0 } else { 15 });
                    let (ys, ye) = (if dy < 0 { 15 } else { 0 }, if dy > 0 { 0 } else { 15 });
                    for y in ys..=ye {
                        for z in zs..=ze {
                            for x in xs..=xe {
                                let wi = Window::idx(x + dx * 16, y + dy * 16, z + dz * 16);
                                match s {
                                    Some(s) => {
                                        let li = ((y * 16 + z) * 16 + x) as usize;
                                        out.blocks[wi] = s.blocks.get(li);
                                        out.light[wi] = s.light.get(li) as u8;
                                    }
                                    None => {
                                        // Outside the known world: occlude sideways so the area's
                                        // edge is not a wall of exposed cliff faces.
                                        let below = sy + dy < self.min_section();
                                        out.blocks[wi] = if !column_known || below { UNKNOWN } else { 0 };
                                        out.light[wi] = 0xf0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Some(version)
    }

    /// sky << 4 | block; open sky where unknown.
    pub fn light_at(&self, x: i32, y: i32, z: i32) -> u8 {
        let g = self.inner.read().unwrap();
        g.section(x >> 4, y >> 4, z >> 4).map_or(0xf0, |s| s.light.get((((y & 15) * 16 + (z & 15)) * 16 + (x & 15)) as usize) as u8)
    }

    /// Top non-air, non-plant block, or min_y - 1.
    pub fn highest_solid(&self, x: i32, z: i32) -> i32 {
        let g = self.inner.read().unwrap();
        for sy in (self.min_section()..=self.max_section()).rev() {
            let Some(s) = g.section(x >> 4, sy, z >> 4) else { continue };
            if s.empty {
                continue;
            }
            for ly in (0..16).rev() {
                let b = s.blocks.get(((ly * 16 + (z & 15)) * 16 + (x & 15)) as usize);
                if b == 0 {
                    continue;
                }
                let info = self.assets.block(b);
                if info.air || (info.mat & MAT_PLANT) != 0 {
                    continue;
                }
                return sy * 16 + ly;
            }
        }
        self.min_y - 1
    }

    /// Whether a block that hides what is behind it, an opaque full block or leaves, stands
    /// on the straight line from `from` to `to`, the blocks holding the two ends left out.
    /// Plants, glass, bamboo, fences and slabs let a player be seen, so they do not count.
    pub fn hidden(&self, from: Vec3, to: Vec3) -> bool {
        let g = self.inner.read().unwrap();
        let d = to - from;
        let cell = |v: Vec3| [v.x.floor() as i32, v.y.floor() as i32, v.z.floor() as i32];
        let (mut c, end) = (cell(from), cell(to));
        let (p, v) = ([from.x, from.y, from.z], [d.x, d.y, d.z]);
        let step: [i32; 3] = std::array::from_fn(|k| if v[k] > 0.0 { 1 } else if v[k] < 0.0 { -1 } else { 0 });
        // How far along the line, 0 at `from` and 1 at `to`, the next cell boundary lies on
        // each axis, and how far apart those boundaries are.
        let mut next: [f32; 3] = std::array::from_fn(|k| match step[k] {
            1 => ((c[k] + 1) as f32 - p[k]) / v[k],
            -1 => (c[k] as f32 - p[k]) / v[k],
            _ => f32::INFINITY,
        });
        let delta: [f32; 3] = std::array::from_fn(|k| if step[k] != 0 { 1.0 / v[k].abs() } else { f32::INFINITY });
        let (mut key, mut section) = (u64::MAX, None);
        loop {
            let k = if next[0] <= next[1] && next[0] <= next[2] { 0 } else if next[1] <= next[2] { 1 } else { 2 };
            if next[k] > 1.0 {
                return false;
            }
            c[k] += step[k];
            next[k] += delta[k];
            if c == end {
                return false;
            }
            let here = section_key(c[0] >> 4, c[1] >> 4, c[2] >> 4);
            if here != key {
                key = here;
                section = g.section(c[0] >> 4, c[1] >> 4, c[2] >> 4).filter(|s| !s.empty);
            }
            let Some(s) = section else { continue };
            let b = s.blocks.get((((c[1] & 15) * 16 + (c[2] & 15)) * 16 + (c[0] & 15)) as usize);
            let info = self.assets.block(b);
            if info.full_opaque || info.leaves {
                return true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_values_come_back() {
        let mut seed = 7u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        for kinds in [1u32, 2, 3, 4, 9, 16, 17, 200, 256, 257, 4000] {
            let palette: Vec<u16> = (0..kinds).map(|k| (k * 7919 % 65521) as u16).collect();
            let values: [u16; 4096] = std::array::from_fn(|i| palette[if i < kinds as usize { i } else { rnd() as usize % kinds as usize }]);
            let p = Packed::new(&values);
            assert_eq!(p.unpack(), values, "{kinds} kinds");
            let bits = match kinds { 1 => 0, 2 => 1, 3..=4 => 2, 5..=16 => 4, 17..=256 => 8, _ => 16 };
            assert_eq!(p.bits, bits, "{kinds} kinds");
        }
        let light: [u8; 4096] = std::array::from_fn(|i| (i % 3) as u8 * 0x11);
        assert_eq!(Packed::from_u8(&light).unpack_u8(), light);
        assert!(Packed::uniform(0).all(0) && !Packed::new(&std::array::from_fn(|i| (i == 9) as u16)).all(0));
    }

    /// Terrain over five by five chunks: stone up to y 63 with dirt on top, pillars,
    /// a glass wall, an overhang and a lamp under it, so both kinds of light spread.
    fn terrain(assets: &Assets, x: i32, y: i32, z: i32) -> u16 {
        let h = 64 + ((x * 7 + z * 13).rem_euclid(5) == 0) as i32 * ((x ^ z).rem_euclid(9));
        let id = |s: &str| assets.state_id(s);
        match () {
            _ if y < 63 && y < h => id("minecraft:stone"),
            _ if y < h => id("minecraft:dirt"),
            _ if (x - 5).abs() < 4 && (z + 3).abs() < 6 && y == 75 => id("minecraft:stone"),
            _ if x == 3 && z == -3 && y == 67 => id("minecraft:glowstone"),
            _ if z == 10 && (64..72).contains(&y) => id("minecraft:glass"),
            _ => 0,
        }
    }

    fn world_with(assets: &Arc<Assets>, edit: &dyn Fn(i32, i32, i32) -> Option<u16>) -> World {
        let w = World::new(assets.clone(), -64, 384);
        for cz in -2..=2 {
            for cx in -2..=2 {
                for sy in -4..=6 {
                    let mut blocks = [0u16; 4096];
                    for (i, b) in blocks.iter_mut().enumerate() {
                        let (x, y, z) = (cx * 16 + (i as i32 & 15), sy * 16 + (i as i32 >> 8), cz * 16 + ((i as i32 >> 4) & 15));
                        *b = edit(x, y, z).unwrap_or_else(|| terrain(assets, x, y, z));
                    }
                    w.set_section(cx, sy, cz, &blocks, &[0; 64]);
                }
            }
        }
        w
    }

    fn all_columns() -> Vec<(i32, i32)> {
        (-2..=2).flat_map(|cz| (-2..=2).map(move |cx| (cx, cz))).collect()
    }

    #[test]
    #[ignore = "needs the client assets: RF_ASSETS=../assets/client cargo test --release -- --ignored"]
    fn light_around_edits_matches_a_full_relight() {
        let assets = Arc::new(Assets::new(&std::env::var("RF_ASSETS").expect("RF_ASSETS")));
        let (stone, glow) = (assets.state_id("minecraft:stone"), assets.state_id("minecraft:glowstone"));
        // A roof high above the ground, which shades whole columns down to it; a lamp
        // placed in the open; the lamp under the overhang taken away; a hole in the glass
        // wall filled; a stone pillar on the chunk corner at the origin.
        let edit = |x: i32, y: i32, z: i32| match () {
            _ if (-6..=-2).contains(&x) && (6..=9).contains(&z) && y == 100 => Some(stone),
            _ if x == 12 && z == 12 && y == 64 => Some(glow),
            _ if x == 3 && z == -3 && y == 67 => Some(0),
            _ if x == 0 && z == 10 && y == 66 => Some(stone),
            _ if x == 0 && z == 0 && (64..70).contains(&y) => Some(stone),
            _ => None,
        };
        let edited = world_with(&assets, &|_, _, _| None);
        edited.take_changes();
        edited.relight(&all_columns());
        // Apply the edit the way the feed does, section by section.
        for cz in -2..=2 {
            for cx in -2..=2 {
                for sy in -4..=6 {
                    let mut blocks = [0u16; 4096];
                    let mut touched = false;
                    for (i, b) in blocks.iter_mut().enumerate() {
                        let (x, y, z) = (cx * 16 + (i as i32 & 15), sy * 16 + (i as i32 >> 8), cz * 16 + ((i as i32 >> 4) & 15));
                        touched |= edit(x, y, z).is_some();
                        *b = edit(x, y, z).unwrap_or_else(|| terrain(&assets, x, y, z));
                    }
                    if touched {
                        edited.set_section(cx, sy, cz, &blocks, &[0; 64]);
                    }
                }
            }
        }
        let changes = edited.take_changes();
        // One edit per section: the roof, the lamp taken away, and the rest, which share one.
        assert!(!changes.bulk && changes.edits.len() == 3, "{} edits", changes.edits.len());
        let boxes: Vec<_> = changes.edits.iter().map(|e| (e.lo, e.hi)).collect();
        for expected in [([-6, 100, 6], [-2, 100, 9]), ([3, 67, -3], [3, 67, -3]), ([0, 64, 0], [12, 69, 12])] {
            assert!(boxes.contains(&expected), "edit boxes are exact: {boxes:?}");
        }
        let fresh = world_with(&assets, &edit);
        fresh.relight(&all_columns());
        let differ = |a: &World, b: &World| {
            let mut n = 0;
            for x in -32..48 {
                for z in -32..48 {
                    for y in -64..320 {
                        n += (a.light_at(x, y, z) != b.light_at(x, y, z)) as usize;
                    }
                }
            }
            n
        };
        assert!(differ(&edited, &fresh) > 1000, "the edits change the light");
        edited.relight_around(&changes.edits);
        assert_eq!(differ(&edited, &fresh), 0, "cells lit differently around the edits than by a full relight");
    }
}
