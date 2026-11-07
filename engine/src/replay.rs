//! Replaying the orbit instead of rendering it again. With nothing in the world changing,
//! each orbit is the last one over again, so the stream records the encoded pictures of
//! one and sends them again the next time round, the renderer and encoder idle. Only
//! the now playing card changes under the music; it sits in a region x264 codes apart
//! (x264.rs), and each replayed picture has that region recoded over the recorded scene
//! beneath it and spliced in.

use crate::persist::{self, Reader, Writer};
use crate::sight::{Change, PLAYER_MARGIN, View};
use crate::vmath::Vec3;
use crate::wide::{F8, I8};
use crate::x264::{Nal, Picture};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A rectangle of macroblocks, [x0, y0, x1, y1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub mb_width: i32,
}

impl Region {
    /// The macroblocks covering pixels [x0, y0, x1, y1) of a width x height frame.
    pub fn covering(px: [i32; 4], width: i32, height: i32) -> Region {
        let mb = |v: i32| (v + 15) / 16;
        Region { x0: px[0] / 16, y0: px[1] / 16, x1: mb(px[2]), y1: mb(px[3]).min(mb(height)), mb_width: mb(width) }
    }

    pub fn holds(&self, first_mb: i32) -> bool {
        let (x, y) = (first_mb % self.mb_width, first_mb / self.mb_width);
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    /// The zone a macroblock is in, as x264 numbers them: 0 in the region, 1 + its band
    /// (bands cut at the rows in `bands`) elsewhere.
    pub fn zone_of(&self, bands: &[i32], mb: i32) -> i32 {
        if self.x1 > self.x0 && self.holds(mb) {
            return 0;
        }
        let y = mb / self.mb_width;
        1 + bands.iter().filter(|&&cut| y >= cut).count() as i32
    }

    /// Its pixels within a width x height frame: x, y, w, h.
    pub fn pixels(&self, width: i32, height: i32) -> (i32, i32, i32, i32) {
        let (x, y) = (16 * self.x0, 16 * self.y0);
        (x, y, (16 * self.x1).min(width) - x, (16 * self.y1).min(height) - y)
    }
}

/// One picture made of two encodes of the same frame: `outer`'s slices outside the
/// region, `inner`'s inside it, with `outer`'s SEI and filler around them. Both come from
/// encoders with the same settings and region and the same frame type, quantiser and IDR
/// id for this picture; the region keeps their predictions apart, so the decoder rebuilds
/// each part exactly as its own encoder did.
/// `take_inner` says, by a slice's first macroblock, which slices come from `inner`.
pub fn splice(outer: &Picture, inner: &Picture, take_inner: impl Fn(i32) -> bool, out: &mut Picture) {
    out.data.clear();
    out.nals.clear();
    (out.pts, out.dts, out.kind, out.qp) = (outer.pts, outer.dts, outer.kind, outer.qp);
    let mut push = |p: &Picture, n: &Nal| {
        out.nals.push(Nal { at: out.data.len(), ..*n });
        out.data.extend_from_slice(p.nal_bytes(n));
    };
    let first_slice = outer.nals.iter().position(Nal::is_slice).unwrap_or(outer.nals.len());
    for n in &outer.nals[..first_slice] {
        push(outer, n);
    }
    // Slices go in raster order: the outer's with the inner's merged in where they start.
    let mut inner_slices = inner.nals.iter().filter(|n| n.is_slice() && take_inner(n.first_mb)).peekable();
    for n in outer.nals[first_slice..].iter().filter(|n| n.is_slice() && !take_inner(n.first_mb)) {
        while let Some(i) = inner_slices.next_if(|i| i.first_mb < n.first_mb) {
            push(inner, i);
        }
        push(outer, n);
    }
    for i in inner_slices {
        push(inner, i);
    }
    for n in outer.nals[first_slice..].iter().filter(|n| !n.is_slice()) {
        push(outer, n);
    }
}

/// One recorded group of pictures, from its IDR up to the next: the live encoder's
/// pictures in decoding order, their pts made frames into the group.
pub struct Gop {
    /// Without their bytes once the group is in a picture store.
    pictures: Vec<Picture>,
    stored: Option<Stored>,
    /// Type and quantiser of each frame, in display order, for the region's encoder.
    pub plan: Vec<(i32, i32)>,
    /// After which frames, in display order, replaying may stop: those where everything
    /// sent so far is exactly the frames up to there. B-frames go out after the frame
    /// following them, so not after every frame.
    pub cuts: Vec<bool>,
    /// What its frames show: the full frames' views from while it was rendered, and where
    /// the camera stood for each frame, in display order.
    pub views: Vec<Arc<View>>,
    pub eyes: Vec<Vec3>,
    /// When its first frame was drawn (engine clock): a change published later may be
    /// missing from it.
    pub from: f64,
    // The newest change it has been checked against, as f64 bits.
    checked: AtomicU64,
}

impl Gop {
    pub fn new(pictures: Vec<Picture>, views: Vec<Arc<View>>, eyes: Vec<Vec3>, from: f64) -> Gop {
        let n = pictures.len();
        let mut plan = vec![(0, 0); n];
        let mut cuts = vec![false; n];
        let mut last = -1;
        for (i, p) in pictures.iter().enumerate() {
            plan[p.pts as usize] = (p.kind, p.qp);
            last = last.max(p.pts);
            if last == i as i64 {
                cuts[i] = true;
            }
        }
        Gop { pictures, stored: None, plan, cuts, views, eyes, from, checked: AtomicU64::new(f64::MIN.to_bits()) }
    }

    /// Pictures in the group.
    pub fn len(&self) -> usize {
        self.pictures.len()
    }

    /// Picture i (decoding order) into `out`, from memory or from the picture store.
    pub fn picture(&self, i: usize, out: &mut Picture) -> Result<(), String> {
        let p = &self.pictures[i];
        (out.pts, out.dts, out.kind, out.qp) = (p.pts, p.dts, p.kind, p.qp);
        out.nals.clone_from(&p.nals);
        match &self.stored {
            None => {
                out.data.clone_from(&p.data);
                Ok(())
            }
            Some(s) => s.store.read(s, i, &mut out.data),
        }
    }

    /// Its record for the index of a kept orbit; nothing for a group not in the store.
    pub fn save(&self, w: &mut Writer) -> bool {
        let Some(st) = &self.stored else { return false };
        w.len(self.pictures.len());
        for (p, &(at, len)) in self.pictures.iter().zip(&st.spans) {
            w.i64(p.pts);
            w.i64(p.dts);
            w.i32(p.kind);
            w.i32(p.qp);
            w.u64(at);
            w.len(len);
            w.len(p.nals.len());
            for n in &p.nals {
                w.u8(n.kind);
                w.i32(n.first_mb);
                w.len(n.at);
                w.len(n.len);
            }
        }
        w.u64(st.slot);
        w.u64(st.generation);
        w.len(self.views.len());
        for v in &self.views {
            v.save(w);
        }
        w.len(self.eyes.len());
        for e in &self.eyes {
            for c in [e.x, e.y, e.z] {
                w.f32(c);
            }
        }
        true
    }

    /// A group from the index of a kept orbit, its bytes in `store`, as if recorded at
    /// `from` on this run's clock.
    pub fn load(r: &mut Reader, store: &Arc<PictureStore>, from: f64) -> persist::Result<Gop> {
        let n = r.len(40)?;
        let mut pictures = Vec::with_capacity(n);
        let mut spans = Vec::with_capacity(n);
        for _ in 0..n {
            let (pts, dts, kind, qp) = (r.i64()?, r.i64()?, r.i32()?, r.i32()?);
            spans.push((r.u64()?, r.u32()? as usize));
            let count = r.len(13)?;
            let mut nals = Vec::with_capacity(count);
            for _ in 0..count {
                nals.push(Nal { kind: r.u8()?, first_mb: r.i32()?, at: r.u32()? as usize, len: r.u32()? as usize });
            }
            if pts < 0 || pts as usize >= n {
                return Err("picture out of its group".into());
            }
            pictures.push(Picture { pts, dts, kind, qp, data: Vec::new(), nals });
        }
        let (slot, generation) = (r.u64()?, r.u64()?);
        let views = (0..r.len(16)?).map(|_| View::load(r).map(Arc::new)).collect::<persist::Result<Vec<_>>>()?;
        let eyes = (0..r.len(12)?).map(|_| Ok(Vec3::new(r.f32()?, r.f32()?, r.f32()?))).collect::<persist::Result<Vec<_>>>()?;
        let mut gop = Gop::new(pictures, views, eyes, from);
        gop.stored = Some(Stored { store: store.clone(), slot, generation, spans });
        Ok(gop)
    }

    /// Moves the pictures' bytes into the store; they stay in memory if it has no room.
    fn store_in(&mut self, store: &Arc<PictureStore>, g: usize) {
        if let Some(stored) = store.write(g, &self.pictures) {
            for p in &mut self.pictures {
                p.data = Vec::new();
                // Held as long as the group is: none of the room the encoder grew it with.
                p.nals.shrink_to_fit();
            }
            self.stored = Some(stored);
        }
    }

    /// Whether `c`, published after this group's first frame was drawn, shows in it.
    pub fn spoilt_by(&self, c: &Change) -> bool {
        c.at > self.from - 0.05 && self.views.iter().any(|v| c.seen_by(v))
    }

    pub fn checked(&self) -> f64 {
        f64::from_bits(self.checked.load(Ordering::Relaxed))
    }

    pub fn set_checked(&self, at: f64) {
        self.checked.store(at.to_bits(), Ordering::Relaxed);
    }

    /// Whether any of these players, or its name tag, could be in the group's frames.
    pub fn shows(&self, players: &[[Vec3; 8]]) -> bool {
        players.iter().any(|p| self.views.iter().any(|v| v.sees(p, PLAYER_MARGIN, false)))
    }
}

/// The recorded orbit, a group of pictures per slot; shared by the thread that decides
/// what to replay and the one that records and replays.
pub struct Cache {
    gops: Mutex<Vec<Option<Arc<Gop>>>>,
    store: Option<Arc<PictureStore>>,
}

impl Cache {
    /// With a store, the pictures' bytes live on disk (the page cache keeps what it can)
    /// rather than in the engine's memory: an orbit is about 450 MB at 10 Mbit/s.
    pub fn new(groups: usize, store: Option<PictureStore>) -> Cache {
        Cache { gops: Mutex::new(vec![None; groups]), store: store.map(Arc::new) }
    }

    pub fn get(&self, g: usize) -> Option<Arc<Gop>> {
        self.gops.lock().unwrap()[g].clone()
    }

    pub fn put(&self, g: usize, mut gop: Gop) {
        if let Some(store) = &self.store {
            gop.store_in(store, g);
        }
        self.gops.lock().unwrap()[g] = Some(Arc::new(gop));
    }

    /// Forgets every group `keep` says no to.
    pub fn retain(&self, mut keep: impl FnMut(&Gop) -> bool) {
        for g in self.gops.lock().unwrap().iter_mut() {
            if g.as_ref().is_some_and(|gop| !keep(gop)) {
                *g = None;
            }
        }
    }

    /// Forgets group g if `keep` says no to it.
    pub fn check(&self, g: usize, keep: impl FnOnce(&Gop) -> bool) {
        let mut gops = self.gops.lock().unwrap();
        if gops[g].as_ref().is_some_and(|gop| !keep(gop)) {
            gops[g] = None;
        }
    }

    pub fn recorded(&self) -> usize {
        self.gops.lock().unwrap().iter().filter(|g| g.is_some()).count()
    }

    /// The index of every stored group, for a later run: `fingerprint` names everything the
    /// recordings depend on besides the world.
    pub fn index(&self, fingerprint: u64) -> Vec<u8> {
        let mut w = Writer::default();
        w.u64(INDEX_MAGIC);
        w.u64(fingerprint);
        let gops = self.gops.lock().unwrap().clone();
        for (g, gop) in gops.iter().enumerate() {
            if let Some(gop) = gop {
                let mark = w.bytes.len();
                w.len(g);
                if !gop.save(&mut w) {
                    w.bytes.truncate(mark);
                }
            }
        }
        w.bytes
    }

    /// The groups an index lists, if it was written with this fingerprint, as recorded at
    /// `from`. They are not put in the cache: the caller does once the world is known.
    pub fn read_index(&self, bytes: &[u8], fingerprint: u64, from: f64) -> persist::Result<Vec<(usize, Gop)>> {
        let store = self.store.as_ref().ok_or("no picture store")?;
        let mut r = Reader::new(bytes);
        if r.u64()? != INDEX_MAGIC {
            return Err("not an orbit index".into());
        }
        if r.u64()? != fingerprint {
            return Err("recorded with other settings or another build".into());
        }
        let groups = self.gops.lock().unwrap().len();
        let mut out = Vec::new();
        while !r.done() {
            let g = r.u32()? as usize;
            if g >= groups {
                return Err("group out of the orbit".into());
            }
            let gop = Gop::load(&mut r, store, from)?;
            store.adopt(g, gop.stored.as_ref().unwrap());
            out.push((g, gop));
        }
        Ok(out)
    }

    pub fn put_loaded(&self, g: usize, gop: Gop) {
        self.gops.lock().unwrap()[g] = Some(Arc::new(gop));
    }
}

const INDEX_MAGIC: u64 = u64::from_le_bytes(*b"RFORBIT1");

/// Where a stored group's bytes are: its slot, the generation written there with it, and
/// each picture's bytes as (offset, length) in the file.
struct Stored {
    store: Arc<PictureStore>,
    slot: u64,
    generation: u64,
    spans: Vec<(u64, usize)>,
}

/// Recorded pictures on disk: two slots per group, written in turn, so that a group can be
/// recorded again while the encoder is still sending the last recording of it. Each slot
/// starts with the generation of what it holds, checked after every read.
pub struct PictureStore {
    file: File,
    slot: u64,
    generation: AtomicU64,
    turn: Mutex<Vec<bool>>,
}

impl PictureStore {
    /// Room for a group of `slot` bytes, far more than a constant bitrate group needs; the
    /// file is sparse, so only what is written takes disk.
    /// `keep`: open what an earlier run left, if it is of this size, rather than start empty.
    pub fn create(path: &str, groups: usize, slot: u64, keep: bool) -> Result<PictureStore, String> {
        let fail = |e: std::io::Error| format!("picture store {path}: {e}");
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(fail)?;
        }
        let size = groups as u64 * 2 * slot;
        let kept = keep && std::fs::metadata(path).is_ok_and(|m| m.len() == size);
        let file = File::options().read(true).write(true).create(true).truncate(!kept).open(path).map_err(fail)?;
        file.set_len(size).map_err(fail)?;
        // Generations count on from the time, so no two runs ever write the same one: a slot
        // this run writes can never pass for the last run's recording that was there.
        let micros = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_micros() as u64);
        Ok(PictureStore { file, slot, generation: AtomicU64::new(micros), turn: Mutex::new(vec![false; groups]) })
    }

    /// A group loaded from an earlier run holds this slot and generation: write the next
    /// recording of it to its other slot, and never reuse the generation.
    fn adopt(&self, g: usize, s: &Stored) {
        self.turn.lock().unwrap()[g] = (s.slot / self.slot) % 2 == 1;
        self.generation.fetch_max(s.generation + 1, Ordering::Relaxed);
    }

    fn write(self: &Arc<Self>, g: usize, pictures: &[Picture]) -> Option<Stored> {
        let total: usize = pictures.iter().map(|p| p.data.len()).sum();
        if total as u64 + 8 > self.slot {
            return None;
        }
        let slot = {
            let mut turn = self.turn.lock().unwrap();
            turn[g] = !turn[g];
            (2 * g as u64 + turn[g] as u64) * self.slot
        };
        let generation = self.generation.fetch_add(1, Ordering::Relaxed);
        let mut bytes = Vec::with_capacity(total + 8);
        bytes.extend_from_slice(&generation.to_le_bytes());
        let mut spans = Vec::with_capacity(pictures.len());
        for p in pictures {
            spans.push((slot + bytes.len() as u64, p.data.len()));
            bytes.extend_from_slice(&p.data);
        }
        if let Err(e) = self.file.write_all_at(&bytes, slot) {
            eprintln!("replay: cannot write the picture store: {e}");
            return None;
        }
        Some(Stored { store: self.clone(), slot, generation, spans })
    }

    fn read(&self, s: &Stored, i: usize, out: &mut Vec<u8>) -> Result<(), String> {
        let (at, len) = s.spans[i];
        out.resize(len, 0);
        let mut head = [0u8; 8];
        self.file
            .read_exact_at(out, at)
            .and_then(|_| self.file.read_exact_at(&mut head, s.slot))
            .map_err(|e| format!("cannot read the picture store: {e}"))?;
        if u64::from_le_bytes(head) != s.generation {
            return Err("a recorded group was overwritten while in use".into());
        }
        Ok(())
    }
}

/// The scene under the region for every frame of the orbit, as 4:2:0: what the now
/// playing card is drawn over again when a frame is replayed. On disk, a slot per frame;
/// the page cache keeps what fits.
pub struct RegionStore {
    file: File,
    slot: usize,
}

impl RegionStore {
    pub fn create(path: &str, slots: usize, slot: usize, keep: bool) -> Result<RegionStore, String> {
        let fail = |e: std::io::Error| format!("replay store {path}: {e}");
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(fail)?;
        }
        let size = (slots * slot) as u64;
        let kept = keep && std::fs::metadata(path).is_ok_and(|m| m.len() == size);
        let file = File::options().read(true).write(true).create(true).truncate(!kept).open(path).map_err(fail)?;
        file.set_len(size).map_err(fail)?;
        Ok(RegionStore { file, slot })
    }

    pub fn write(&self, index: usize, yuv: &[u8]) -> std::io::Result<()> {
        self.file.write_all_at(yuv, (index * self.slot) as u64)
    }

    pub fn read(&self, index: usize, yuv: &mut [u8]) -> std::io::Result<()> {
        self.file.read_exact_at(yuv, (index * self.slot) as u64)
    }
}

/// BT.709 limited range 4:2:0 back to 8-bit R'G'B' planes: the inverse of `to_yuv420`
/// to within a step, each chroma sample spread over its 2 x 2 block.
/// Pixels [x0, x1) x [y0, y1) (even bounds) of a w x h planar 4:2:0 frame into R'G'B'
/// planes of the frame's size, with from_yuv420's arithmetic, so the same bytes.
pub fn yuv_rect_to_rgb(yuv: &[u8], w: usize, h: usize, rect: [i32; 4], rgb: &mut [Vec<u8>; 3]) {
    let [x0, y0, x1, y1] = rect.map(|v| v as usize);
    let (luma, chroma) = yuv.split_at(w * h);
    let (u, v) = chroma.split_at((w / 2) * (h / 2));
    let [r, g, b] = rgb;
    let s = F8::splat;
    let (lo, hi) = (s(0.0), s(255.0));
    for y in y0..y1 {
        let row = y * w;
        let crow = (y / 2) * (w / 2);
        let mut x = x0;
        while x + 8 <= x1 {
            let l = I8::load_u8(luma[row + x..][..8].try_into().unwrap()).to_float();
            let spread = |c: &[u8]| {
                let q = &c[crow + x / 2..][..4];
                I8::load_u8(&[q[0], q[0], q[1], q[1], q[2], q[2], q[3], q[3]]).to_float()
            };
            let (cb, cr) = (spread(u) - s(128.0), spread(v) - s(128.0));
            let l = (l - s(16.0)) * s(1.164_384);
            let rr = l + s(1.792_741) * cr + s(0.5);
            let gg = l - s(0.213_249) * cb - s(0.532_909) * cr + s(0.5);
            let bb = l + s(2.112_402) * cb + s(0.5);
            for (plane, c) in [(&mut *r, rr), (&mut *g, gg), (&mut *b, bb)] {
                plane[row + x..][..8].copy_from_slice(&c.max(lo).min(hi).to_int().to_bytes());
            }
            x += 8;
        }
        for x in x..x1 {
            let c = crow + x / 2;
            let l = 1.164_384 * (luma[row + x] as f32 - 16.0);
            let (cb, cr) = (u[c] as f32 - 128.0, v[c] as f32 - 128.0);
            r[row + x] = (l + 1.792_741 * cr + 0.5).clamp(0.0, 255.0) as u8;
            g[row + x] = (l - 0.213_249 * cb - 0.532_909 * cr + 0.5).clamp(0.0, 255.0) as u8;
            b[row + x] = (l + 2.112_402 * cb + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// The same rectangle back from R'G'B' planes of the frame's size, with the engine's
/// fixed-point BT.709 formula (studio range; chroma from each 2x2 block's sum).
pub fn rgb_rect_to_yuv(rgb: &[Vec<u8>; 3], w: usize, h: usize, rect: [i32; 4], yuv: &mut [u8]) {
    let [x0, y0, x1, y1] = rect.map(|v| v as usize);
    let (luma, chroma) = yuv.split_at_mut(w * h);
    let (u, v) = chroma.split_at_mut((w / 2) * (h / 2));
    let [pr, pg, pb] = rgb;
    let k = I8::splat;
    for y in y0..y1 {
        let row = y * w;
        let mut x = x0;
        while x + 8 <= x1 {
            let at = |p: &[u8]| I8::load_u8(p[row + x..][..8].try_into().unwrap());
            let sum = at(pr) * k(11966) + at(pg) * k(40254) + at(pb) * k(4064) + k(32768);
            luma[row + x..][..8].copy_from_slice(&(sum.shr(k(16)) + k(16)).to_bytes());
            x += 8;
        }
        for x in x..x1 {
            let i = row + x;
            let (r, g, b) = (pr[i] as i32, pg[i] as i32, pb[i] as i32);
            luma[i] = (16 + ((11966 * r + 40254 * g + 4064 * b + 32768) >> 16)) as u8;
        }
    }
    for cy in y0 / 2..y1 / 2 {
        for cx in x0 / 2..x1 / 2 {
            let at = |p: &[u8], dy: usize, dx: usize| p[(2 * cy + dy) * w + 2 * cx + dx] as i32;
            let sum = |p: &[u8]| at(p, 0, 0) + at(p, 0, 1) + at(p, 1, 0) + at(p, 1, 1);
            let (r, g, b) = (sum(pr), sum(pg), sum(pb));
            let k = cy * (w / 2) + cx;
            u[k] = (128 + ((-6597 * r - 22187 * g + 28784 * b + (1 << 17)) >> 18)).clamp(0, 255) as u8;
            v[k] = (128 + ((28784 * r - 26147 * g - 2637 * b + (1 << 17)) >> 18)).clamp(0, 255) as u8;
        }
    }
}

pub fn from_yuv420(yuv: &[u8], w: usize, h: usize, rgb: &mut [Vec<u8>; 3]) {
    let (luma, chroma) = yuv.split_at(w * h);
    let (u, v) = chroma.split_at((w / 2) * (h / 2));
    for p in rgb.iter_mut() {
        p.resize(w * h, 0);
    }
    let [r, g, b] = rgb;
    let s = F8::splat;
    let (lo, hi) = (s(0.0), s(255.0));
    // Eight pixels at a time, each chroma sample spread over the two it covers; the same
    // arithmetic as one pixel at a time, so the same bytes.
    let pixel = |l: f32, cb: f32, cr: f32| {
        let l = 1.164_384 * (l - 16.0);
        let (cb, cr) = (cb - 128.0, cr - 128.0);
        [l + 1.792_741 * cr + 0.5, l - 0.213_249 * cb - 0.532_909 * cr + 0.5, l + 2.112_402 * cb + 0.5]
    };
    for y in 0..h {
        let row = y * w;
        let crow = (y / 2) * (w / 2);
        let n = w / 8;
        for m in 0..n {
            let x = 8 * m;
            let l = I8::load_u8(luma[row + x..][..8].try_into().unwrap()).to_float();
            let spread = |c: &[u8]| {
                let q = &c[crow + x / 2..][..4];
                I8::load_u8(&[q[0], q[0], q[1], q[1], q[2], q[2], q[3], q[3]]).to_float()
            };
            let (cb, cr) = (spread(u) - s(128.0), spread(v) - s(128.0));
            let l = (l - s(16.0)) * s(1.164_384);
            let rr = l + s(1.792_741) * cr + s(0.5);
            let gg = l - s(0.213_249) * cb - s(0.532_909) * cr + s(0.5);
            let bb = l + s(2.112_402) * cb + s(0.5);
            for (plane, c) in [(&mut *r, rr), (&mut *g, gg), (&mut *b, bb)] {
                plane[row + x..][..8].copy_from_slice(&c.max(lo).min(hi).to_int().to_bytes());
            }
        }
        for x in 8 * n..w {
            let c = (y / 2) * (w / 2) + x / 2;
            let px = pixel(luma[row + x] as f32, u[c] as f32, v[c] as f32);
            r[row + x] = px[0].clamp(0.0, 255.0) as u8;
            g[row + x] = px[1].clamp(0.0, 255.0) as u8;
            b[row + x] = px[2].clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x264::{Encoder, Settings, TYPE_AUTO, TYPE_IDR};
    use std::path::Path;
    use std::process::Command;

    fn settings(width: i32, height: i32, region: [i32; 4]) -> Settings {
        let options = [
            "level=4.1", "threads=2", "bitrate=10000", "vbv-maxrate=10000", "vbv-bufsize=10000", "keyint=48",
            "min-keyint=48", "scenecut=0", "bframes=2", "nal-hrd=cbr", "force-cfr=1", "mbtree=1", "rc-lookahead=10",
            "direct=spatial", "stitchable=1", "colorprim=bt709", "transfer=bt709", "colormatrix=bt709", "fullrange=off",
        ];
        Settings {
            preset: "superfast".into(),
            tune: String::new(),
            profile: "high".into(),
            width,
            height,
            fps: 24,
            options: options.iter().map(|s| s.to_string()).collect(),
            region,
            region_only: false,
            bands: Vec::new(),
            window: [0, 0],
        }
    }

    fn noise(x: i32, y: i32, seed: u32) -> u8 {
        let mut h = (x as u32).wrapping_mul(0x9e37_79b1) ^ (y as u32).wrapping_mul(0x85eb_ca77) ^ seed;
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        (h >> 24) as u8
    }

    /// A textured scene sliding down and right (so motion search is drawn across the
    /// region's edges), and in the region something else entirely for the inner encode.
    fn frame(w: i32, h: i32, t: i32, inside: Option<(&Region, u32)>) -> Vec<u8> {
        let mut yuv = vec![128u8; (w * h * 3 / 2) as usize];
        let texel = |x: i32, y: i32, seed: u32| {
            let (bx, by) = (x.div_euclid(24), y.div_euclid(24));
            let base = ((bx * 37 + by * 91) & 127) as i32 + 64;
            (base + (noise(x / 3, y / 3, seed) as i32 - 128) / 4).clamp(16, 235) as u8
        };
        for y in 0..h {
            for x in 0..w {
                let (mx, my) = (x / 16, y / 16);
                let v = match inside {
                    Some((r, seed)) if mx >= r.x0 && mx < r.x1 && my >= r.y0 && my < r.y1 => texel(x - 2 * t, y + t, seed),
                    Some(_) => 128,
                    None => texel(x - 3 * t, y - 2 * t, 7),
                };
                yuv[(y * w + x) as usize] = v;
            }
        }
        let (cw, ch) = (w / 2, h / 2);
        for (plane, seed) in [(0, 11u32), (1, 23u32)] {
            let base = (w * h + plane * cw * ch) as usize;
            for y in 0..ch {
                for x in 0..cw {
                    let (mx, my) = (x / 8, y / 8);
                    let v = match inside {
                        Some((r, s)) if mx >= r.x0 && mx < r.x1 && my >= r.y0 && my < r.y1 => 128 + (noise(x + t, y, s ^ seed) as i32 - 128) / 8,
                        Some(_) => 128,
                        None => 128 + (noise((x - t) / 4, (y - t) / 4, seed) as i32 - 128) / 6,
                    };
                    yuv[base + (y * cw + x) as usize] = v as u8;
                }
            }
        }
        yuv
    }

    fn encode(enc: &mut Encoder, w: i32, h: i32, frames: &[Vec<u8>], plan: &[(i32, i32)]) -> Vec<Picture> {
        encode_zones(enc, w, h, frames, plan, 0)
    }

    fn encode_zones(enc: &mut Encoder, w: i32, h: i32, frames: &[Vec<u8>], plan: &[(i32, i32)], zones: i32) -> Vec<Picture> {
        encode_zones_with(enc, w, h, frames, plan, zones, None)
    }

    fn encode_zones_with(enc: &mut Encoder, w: i32, h: i32, frames: &[Vec<u8>], plan: &[(i32, i32)], zones: i32,
                         offsets: Option<&[Vec<f32>]>) -> Vec<Picture> {
        let (cw, ch) = (w / 2, h / 2);
        let mut out = Vec::new();
        let mut pic = Picture::default();
        for (i, f) in frames.iter().enumerate() {
            let planes = [&f[..(w * h) as usize], &f[(w * h) as usize..(w * h + cw * ch) as usize], &f[(w * h + cw * ch) as usize..]];
            let (kind, qp) = plan[i];
            let q = offsets.map(|o| &o[i][..]);
            if enc.encode_with(Some((planes, [w, cw, cw])), i as i64, kind, qp, i as i32, zones, q, &mut pic).unwrap() {
                out.push(pic.clone());
            }
        }
        while enc.delayed() > 0 {
            if enc.encode(None, 0, 0, -1, -1, zones, &mut pic).unwrap() {
                out.push(pic.clone());
            }
        }
        out
    }

    fn annexb(path: &Path, sps: &[u8], pps: &[u8], pictures: &[Picture]) {
        let mut bytes = Vec::new();
        for ps in [sps, pps] {
            bytes.extend_from_slice(&[0, 0, 0, 1]);
            bytes.extend_from_slice(ps);
        }
        for p in pictures {
            for n in &p.nals {
                bytes.extend_from_slice(&[0, 0, 0, 1]);
                bytes.extend_from_slice(&p.nal_bytes(n)[4..]);
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn decode(path: &Path) -> Vec<u8> {
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-xerror", "-i"])
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            .output()
            .expect("ffmpeg");
        assert!(out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
        out.stdout
    }

    /// Encodes `outer` whole, then `inner` with the outer's frame types and quantisers, and
    /// splices them. Decoded, the result must be the outer encode outside the region and the
    /// inner one inside it, to the bit: any prediction across the region's edge would drift.
    fn check(name: &str, w: i32, h: i32, r: Region, outer_frames: &[Vec<u8>], inner_frames: &[Vec<u8>]) {
        let n = outer_frames.len();
        let s = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
        let mut outer_enc = Encoder::open(&s).unwrap();
        let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
        let outer = encode(&mut outer_enc, w, h, outer_frames, &plan);
        assert_eq!(outer.len(), n);

        let mut inner_settings = s.clone();
        inner_settings.options.extend(["threads=1", "mbtree=0"].map(String::from));
        let mut inner_enc = Encoder::open(&inner_settings).unwrap();
        assert_eq!(outer_enc.parameter_sets().unwrap(), inner_enc.parameter_sets().unwrap(), "parameter sets differ");
        let mut by_pts = vec![(0, 0); n];
        for p in &outer {
            by_pts[p.pts as usize] = (p.kind, p.qp);
        }
        let inner = encode(&mut inner_enc, w, h, inner_frames, &by_pts);
        let order = |ps: &[Picture]| ps.iter().map(|p| (p.pts, p.kind)).collect::<Vec<_>>();
        assert_eq!(order(&outer), order(&inner), "the inner encoder did not follow the outer's frame types");

        let spliced: Vec<Picture> = outer
            .iter()
            .zip(&inner)
            .map(|(o, i)| {
                let mut p = Picture::default();
                splice(o, i, |mb| r.holds(mb), &mut p);
                p
            })
            .collect();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/splice");
        std::fs::create_dir_all(&dir).unwrap();
        let (sps, pps) = outer_enc.parameter_sets().unwrap();
        let decoded: Vec<Vec<u8>> = [("outer", &outer), ("inner", &inner), ("spliced", &spliced)]
            .iter()
            .map(|(part, pictures)| {
                let path = dir.join(format!("{name}-{part}.h264"));
                annexb(&path, &sps, &pps, pictures);
                decode(&path)
            })
            .collect();
        let (o, i, sp) = (&decoded[0], &decoded[1], &decoded[2]);
        assert_eq!(o.len(), sp.len());
        let (cw, ch) = (w / 2, h / 2);
        let frame_len = (w * h * 3 / 2) as usize;
        let (mut outside, mut inside, mut inside_differs) = (0usize, 0usize, false);
        for f in 0..n {
            for (plane_at, pw, ph, mb) in [(0, w, h, 16), ((w * h) as usize, cw, ch, 8), ((w * h + cw * ch) as usize, cw, ch, 8)] {
                for y in 0..ph {
                    for x in 0..pw {
                        let k = f * frame_len + plane_at + (y * pw + x) as usize;
                        if x / mb >= r.x0 && x / mb < r.x1 && y / mb >= r.y0 && y / mb < r.y1 {
                            inside += (sp[k] != i[k]) as usize;
                            inside_differs |= o[k] != i[k];
                        } else {
                            outside += (sp[k] != o[k]) as usize;
                        }
                    }
                }
            }
        }
        assert!(inside_differs, "{name}: the two encodes do not differ in the region, so this proves nothing");
        assert_eq!((outside, inside), (0, 0), "{name}: samples that differ from their own encode (outside, inside)");

        // What the stream uses: an encoder that encodes only the region, its slices outside
        // left empty. Its own stream does not decode outside the region, but the decoder
        // conceals that and decodes the region from the region's slices alone, so spliced
        // with the outer encode it must give the outer decode outside and that inside.
        inner_settings.region_only = true;
        inner_settings.options.extend(["rc-lookahead=0", "b-adapt=0"].map(String::from));
        let mut only_enc = Encoder::open(&inner_settings).unwrap();
        assert_eq!(outer_enc.parameter_sets().unwrap(), only_enc.parameter_sets().unwrap(), "parameter sets differ (region only)");
        let only = encode(&mut only_enc, w, h, inner_frames, &by_pts);
        assert_eq!(order(&outer), order(&only), "the region-only encoder did not follow the outer's frame types");
        let spliced: Vec<Picture> = outer
            .iter()
            .zip(&only)
            .map(|(o, i)| {
                let mut p = Picture::default();
                splice(o, i, |mb| r.holds(mb), &mut p);
                p
            })
            .collect();
        let decoded: Vec<Vec<u8>> = [("only", &only, false), ("only-spliced", &spliced, true)]
            .iter()
            .map(|(part, pictures, strict)| {
                let path = dir.join(format!("{name}-{part}.h264"));
                annexb(&path, &sps, &pps, pictures);
                decode_with(&path, *strict)
            })
            .collect();
        let (i, sp) = (&decoded[0], &decoded[1]);
        assert_eq!(i.len(), sp.len());
        let (mut outside, mut inside) = (0usize, 0usize);
        let mut err = 0.0f64;
        for f in 0..n {
            for (plane_at, pw, ph, mb) in [(0, w, h, 16), ((w * h) as usize, cw, ch, 8), ((w * h + cw * ch) as usize, cw, ch, 8)] {
                for y in 0..ph {
                    for x in 0..pw {
                        let k = f * frame_len + plane_at + (y * pw + x) as usize;
                        if x / mb >= r.x0 && x / mb < r.x1 && y / mb >= r.y0 && y / mb < r.y1 {
                            inside += (sp[k] != i[k]) as usize;
                            let d = sp[k] as f64 - inner_frames[f][k - f * frame_len] as f64;
                            err += d * d;
                        } else {
                            outside += (sp[k] != o[k]) as usize;
                        }
                    }
                }
            }
        }
        assert_eq!((outside, inside), (0, 0), "{name}, region only: samples that differ from their own encode (outside, inside)");
        let area = (n as i32 * (r.x1 - r.x0) * (r.y1 - r.y0) * 384) as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / (err / area)).log10();
        assert!(psnr > 30.0, "{name}, region only: the region is a poor copy of its source, {psnr:.1} dB");
    }

    fn decode_with(path: &Path, strict: bool) -> Vec<u8> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error"]);
        if strict {
            cmd.arg("-xerror");
        }
        let out = cmd.arg("-i").arg(path).args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]).output().expect("ffmpeg");
        assert!(!strict || out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
        out.stdout
    }

    #[test]
    #[ignore = "needs ffmpeg; run with --ignored"]
    fn spliced_pictures_decode_to_both_encodes_exactly() {
        let (w, h) = (640, 368);
        let mb_width = w / 16;
        let regions = [
            ("left", Region { x0: 0, y0: 16, x1: 17, y1: 21, mb_width }),
            ("middle", Region { x0: 10, y0: 5, x1: 25, y1: 12, mb_width }),
            ("corner", Region { x0: 30, y0: 15, x1: 40, y1: 23, mb_width }),
        ];
        let outer: Vec<Vec<u8>> = (0..100).map(|t| frame(w, h, t, None)).collect();
        for (name, r) in regions {
            let inner: Vec<Vec<u8>> = (0..100).map(|t| frame(w, h, t, Some((&r, 99)))).collect();
            check(name, w, h, r, &outer, &inner);
        }
    }

    /// With bands as well as the region: any set of zones recoded and spliced in decodes
    /// exactly, outside them as the first encode and inside as the second, whether the
    /// second encodes everything or only those zones.
    #[test]
    #[ignore = "needs ffmpeg; run with --ignored"]
    fn spliced_bands_decode_exactly() {
        let (w, h) = (640, 368);
        let mb_width = w / 16;
        // 23 rows of macroblocks: bands of rows 0-3, 4-7, 8-11 (with the region on its left),
        // 12-15, 16-19 and 20-22.
        let bands = vec![4, 8, 12, 16, 20];
        let r = Region { x0: 0, y0: 8, x1: 17, y1: 12, mb_width };
        let zone = |x: i32, y: i32| r.zone_of(&bands, y * mb_width + x);
        let outer_frames: Vec<Vec<u8>> = (0..100).map(|t| frame(w, h, t, None)).collect();
        for (name, zones) in [("region", 1), ("one band", 1 << 4), ("region and two bands", 1 | 1 << 2 | 1 << 5), ("region's band", 1 << 3)] {
            // The second encode's content: the first's moved the other way where it is recoded.
            let inner_frames: Vec<Vec<u8>> = (0..100)
                .map(|t| {
                    let (a, b) = (frame(w, h, t, None), frame(w, h, 100 - t, None));
                    let mut f = a.clone();
                    for (plane, pw, ph, mb) in [(0usize, w, h, 16), ((w * h) as usize, w / 2, h / 2, 8), ((w * h * 5 / 4) as usize, w / 2, h / 2, 8)] {
                        for y in 0..ph {
                            for x in 0..pw {
                                if (zones >> zone(x / mb, y / mb)) & 1 != 0 {
                                    f[plane + (y * pw + x) as usize] = b[plane + (y * pw + x) as usize];
                                }
                            }
                        }
                    }
                    f
                })
                .collect();
            let mut s = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
            s.bands = bands.clone();
            let n = outer_frames.len();
            let mut outer_enc = Encoder::open(&s).unwrap();
            let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
            let outer = encode(&mut outer_enc, w, h, &outer_frames, &plan);
            let mut by_pts = vec![(0, 0); n];
            for p in &outer {
                by_pts[p.pts as usize] = (p.kind, p.qp);
            }
            // And again from an encoder given only the rows of the zones it encodes.
            let rows: Vec<i32> = (0..h / 16).filter(|&y| (0..mb_width).any(|x| (zones >> zone(x, y)) & 1 != 0)).collect();
            for window in [[0, 0], [rows[0], rows[rows.len() - 1] + 1]] {
                let name = &format!("{name}, window {window:?}");
                let mut inner_settings = s.clone();
                inner_settings.options.extend(["threads=1", "mbtree=0", "rc-lookahead=0", "b-adapt=0"].map(String::from));
                inner_settings.region_only = true;
                inner_settings.window = window;
                if window != [0, 0] {
                    // As the stream's windows: no HRD, so no filler.
                    inner_settings.options.retain(|o| o != "nal-hrd=cbr");
                    inner_settings.options.push("nal-hrd=none".into());
                }
                let mut only_enc = Encoder::open(&inner_settings).unwrap();
                if window == [0, 0] {
                    assert_eq!(outer_enc.parameter_sets().unwrap(), only_enc.parameter_sets().unwrap(), "{name}: parameter sets differ");
                }
                let only = encode_zones(&mut only_enc, w, h, &inner_frames, &by_pts, zones);
                if std::env::var_os("RF_DEBUG_NALS").is_some() {
                    eprintln!("{name}: outer {:?}", outer[1].nals.iter().filter(|n| n.is_slice()).map(|n| (n.first_mb, n.len)).collect::<Vec<_>>());
                    eprintln!("{name}: only  {:?}", only[1].nals.iter().filter(|n| n.is_slice()).map(|n| (n.first_mb, n.len)).collect::<Vec<_>>());
                }
                let order = |ps: &[Picture]| ps.iter().map(|p| (p.pts, p.kind)).collect::<Vec<_>>();
                assert_eq!(order(&outer), order(&only), "{name}: the region-only encoder did not follow the outer's frame types");
                let take = |mb: i32| (zones >> r.zone_of(&bands, mb)) & 1 != 0;
                let spliced: Vec<Picture> = outer
                    .iter()
                    .zip(&only)
                    .map(|(o, i)| {
                        let mut p = Picture::default();
                        splice(o, i, take, &mut p);
                        p
                    })
                    .collect();
                let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/splice");
                std::fs::create_dir_all(&dir).unwrap();
                let (sps, pps) = outer_enc.parameter_sets().unwrap();
                let file = |part: &str| dir.join(format!("bands-{}-{part}.h264", name.replace([' ', ',', '[', ']'], "-")));
                annexb(&file("outer"), &sps, &pps, &outer);
                annexb(&file("only"), &sps, &pps, &only);
                annexb(&file("spliced"), &sps, &pps, &spliced);
                let (o, i, sp) = (decode_with(&file("outer"), true), decode_with(&file("only"), false), decode_with(&file("spliced"), true));
                assert_eq!(o.len(), sp.len());
                let frame_len = (w * h * 3 / 2) as usize;
                let (mut outside, mut inside, mut differs) = (0usize, 0usize, false);
                for f in 0..n {
                    for (plane, pw, ph, mb) in [(0usize, w, h, 16), ((w * h) as usize, w / 2, h / 2, 8), ((w * h * 5 / 4) as usize, w / 2, h / 2, 8)] {
                        for y in 0..ph {
                            for x in 0..pw {
                                let k = f * frame_len + plane + (y * pw + x) as usize;
                                if (zones >> zone(x / mb, y / mb)) & 1 != 0 {
                                    inside += (sp[k] != i[k]) as usize;
                                    differs |= o[k] != i[k];
                                } else {
                                    outside += (sp[k] != o[k]) as usize;
                                }
                            }
                        }
                    }
                }
                assert!(differs, "{name}: the encodes do not differ where recoded, so this proves nothing");
                assert_eq!((outside, inside), (0, 0), "{name}: samples that differ from their own encode (outside, inside)");
            }
        }
    }

    /// The same on real footage: RF_SPLICE_CLIP names raw 1920x1080 yuv420p frames (say ten
    /// seconds of the stream), and the region is recoded with the clip's own pixels from
    /// four seconds later.
    #[test]
    #[ignore = "needs ffmpeg and RF_SPLICE_CLIP; run with --ignored"]
    fn spliced_footage_decodes_to_both_encodes_exactly() {
        let (w, h) = (1920, 1080);
        let clip = std::fs::read(std::env::var("RF_SPLICE_CLIP").expect("RF_SPLICE_CLIP")).unwrap();
        let frame_len = (w * h * 3 / 2) as usize;
        let frames: Vec<Vec<u8>> = clip.chunks_exact(frame_len).map(<[u8]>::to_vec).collect();
        let n = frames.len() - 96;
        let r = Region { x0: 0, y0: 58, x1: 47, y1: 67, mb_width: 120 };
        let inner: Vec<Vec<u8>> = (0..n)
            .map(|f| {
                let mut v = frames[f].clone();
                let later = &frames[f + 96];
                for (at, pw, ph, mb) in [(0, w, h, 16), ((w * h) as usize, w / 2, h / 2, 8), ((w * h * 5 / 4) as usize, w / 2, h / 2, 8)] {
                    for y in r.y0 * mb..(r.y1 * mb).min(ph) {
                        let row = at + (y * pw) as usize;
                        v[row + (r.x0 * mb) as usize..row + (r.x1 * mb) as usize]
                            .copy_from_slice(&later[row + (r.x0 * mb) as usize..row + (r.x1 * mb) as usize]);
                    }
                }
                v
            })
            .collect();
        check("footage", w, h, r, &frames[..n], &inner);
    }

    /// What bands cost the stream at its constant 10 Mbit/s: luma PSNR of real footage
    /// (RF_SPLICE_CLIP, raw 1920x1080 yuv420p) encoded as the live encoder does, with the
    /// card's region alone and with bands of several heights.
    #[test]
    #[ignore = "a benchmark; needs ffmpeg and RF_SPLICE_CLIP; run with --ignored --nocapture"]
    fn band_cost() {
        let (w, h) = (1920, 1080);
        let clip = std::fs::read(std::env::var("RF_SPLICE_CLIP").expect("RF_SPLICE_CLIP")).unwrap();
        let frame_len = (w * h * 3 / 2) as usize;
        let frames: Vec<Vec<u8>> = clip.chunks_exact(frame_len).map(<[u8]>::to_vec).collect();
        let n = frames.len();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/splice");
        std::fs::create_dir_all(&dir).unwrap();
        let cards = [0, 58, 47, 68];
        let mut sets = vec![
            ("no region".to_string(), [0; 4], vec![]),
            ("card".to_string(), cards, vec![]),
            ("card, sky + 13-row bands".to_string(), cards, vec![19, 32, 45, 58]),
        ];
        // RF_BAND_ROWS: more band heights to try (","), cut upwards from the card as the
        // stream cuts them.
        for rows in std::env::var("RF_BAND_ROWS").unwrap_or_default().split(',').filter(|r| !r.is_empty()) {
            let rows: i32 = rows.trim().parse().unwrap();
            let mut cuts: Vec<i32> = (0..).map(|k| 58 - k * rows).take_while(|&c| c >= 3).take(16).collect();
            cuts.sort_unstable();
            sets.push((format!("card, {rows}-row bands {cuts:?}"), cards, cuts));
        }
        for (name, region, bands) in sets {
            let mut s = settings(w, h, region);
            s.bands = bands;
            let mut enc = Encoder::open(&s).unwrap();
            let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
            let pics = encode(&mut enc, w, h, &frames, &plan);
            let (sps, pps) = enc.parameter_sets().unwrap();
            let path = dir.join("band-cost.h264");
            annexb(&path, &sps, &pps, &pics);
            let out = decode(&path);
            let bytes: usize = pics.iter().map(|p| p.data.len()).sum();
            let mut se = 0.0f64;
            for f in 0..n {
                for k in 0..(w * h) as usize {
                    let d = out[f * frame_len + k] as f64 - frames[f][k] as f64;
                    se += d * d;
                }
            }
            let psnr = 10.0 * (255.0f64 * 255.0 / (se / (n as f64 * (w * h) as f64))).log10();
            eprintln!("{name}: {:.0} kbit/s, luma PSNR {psnr:.3} dB", bytes as f64 * 8.0 * 24.0 / n as f64 / 1000.0);
        }
    }

    /// How a band recoded at the recorded frame's quantiser compares with the recording
    /// itself, in bits and luma PSNR, for the same pictures (RF_SPLICE_CLIP, 1080p yuv420p),
    /// for each set of extra options in RF_INNER_VARIANTS.
    #[test]
    #[ignore = "a benchmark; needs ffmpeg and RF_SPLICE_CLIP; run with --ignored --nocapture"]
    fn recode_quality() {
        let (w, h) = (1920, 1080);
        let clip = std::fs::read(std::env::var("RF_SPLICE_CLIP").expect("RF_SPLICE_CLIP")).unwrap();
        let frame_len = (w * h * 3 / 2) as usize;
        let frames: Vec<Vec<u8>> = clip.chunks_exact(frame_len).take(144).map(<[u8]>::to_vec).collect();
        let n = frames.len();
        let r = Region { x0: 0, y0: 58, x1: 47, y1: 68, mb_width: 120 };
        let bands = vec![19, 32, 45, 58];
        let mut s = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
        s.bands = bands.clone();
        let mut outer_enc = Encoder::open(&s).unwrap();
        let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
        let outer = encode(&mut outer_enc, w, h, &frames, &plan);
        let mut by_pts = vec![(0, 0); n];
        for p in &outer {
            by_pts[p.pts as usize] = (p.kind, p.qp);
        }
        let zones = 1 << 2 | 1 << 3; // two bands of the busy ground
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/splice");
        std::fs::create_dir_all(&dir).unwrap();
        let (sps, pps) = outer_enc.parameter_sets().unwrap();
        let measure = |name: &str, pictures: &[Picture], from: &[Picture]| {
            let path = dir.join("recode-quality.h264");
            annexb(&path, &sps, &pps, pictures);
            let out = decode(&path);
            let (mut se, mut count) = (0.0f64, 0usize);
            for f in 0..n {
                for y in 0..h {
                    if (zones >> r.zone_of(&bands, (y / 16) * 120)) & 1 == 0 {
                        continue;
                    }
                    for x in 0..w as usize {
                        let k = (y * w) as usize + x;
                        let d = out[f * frame_len + k] as f64 - frames[f][k] as f64;
                        se += d * d;
                        count += 1;
                    }
                }
            }
            let bits: usize = from.iter().flat_map(|p| p.nals.iter().filter(|x| x.is_slice() && (zones >> r.zone_of(&bands, x.first_mb)) & 1 != 0).map(|x| x.len)).sum();
            eprintln!("{name}: the bands {:.0} kbit/s, luma PSNR {:.3} dB", bits as f64 * 8.0 * 24.0 / n as f64 / 1000.0,
                      10.0 * (255.0f64 * 255.0 / (se / count as f64)).log10());
        };
        measure("recorded", &outer, &outer);
        // What the engine recodes from: the recording, decoded, with the quantiser each
        // macroblock was coded at.
        let mut dec = crate::decode::Decoder::open(&sps, &pps, w, h, false, false).unwrap();
        let mut decoded = vec![Vec::new(); n];
        let mut mb_qp = vec![Vec::new(); n];
        let (mut buf, mut side) = (Vec::new(), crate::decode::Side::default());
        for p in outer.iter().map(Some).chain(std::iter::once(None)) {
            dec.send(p).unwrap();
            while let Some(pts) = dec.receive(&mut buf, &mut side, None).unwrap() {
                decoded[pts as usize] = buf.clone();
                mb_qp[pts as usize] = side.qp.clone();
            }
        }
        assert!(mb_qp.iter().all(|q| q.iter().all(|&q| q <= 51)), "a quantiser the decoder did not give");
        let offsets: Vec<Vec<f32>> = (0..n).map(|i| mb_qp[i].iter().map(|&q| q as f32 - by_pts[i].1 as f32).collect()).collect();
        // RF_INNER_VARIANTS: sets of extra options (";" between sets, "," within one);
        // RF_QP_DELTAS: steps added to the recorded quantisers (default 0).
        let variants = std::env::var("RF_INNER_VARIANTS").unwrap_or_default();
        let deltas: Vec<i32> = std::env::var("RF_QP_DELTAS").unwrap_or("0".into()).split(',').map(|d| d.trim().parse().unwrap()).collect();
        let mut runs = vec![("originals", &frames, None, String::new()), ("decoded", &decoded, None, String::new())];
        for v in std::iter::once("").chain(variants.split(';').filter(|v| !v.is_empty())) {
            for &d in &deltas {
                runs.push(("decoded", &decoded, Some(d), v.to_string()));
            }
        }
        for (name, sources, delta, v) in runs {
            let mut inner_settings = s.clone();
            inner_settings.options.extend(["threads=1", "mbtree=0", "rc-lookahead=0", "b-adapt=0"].map(String::from));
            inner_settings.options.extend(v.split(',').filter(|o| !o.is_empty()).map(String::from));
            inner_settings.region_only = true;
            let mut enc = Encoder::open(&inner_settings).unwrap();
            let shifted: Option<Vec<Vec<f32>>> = delta.map(|d| offsets.iter().map(|o| o.iter().map(|q| q + d as f32).collect()).collect());
            let t0 = std::time::Instant::now();
            let only = encode_zones_with(&mut enc, w, h, sources, &by_pts, zones, shifted.as_deref());
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let spliced: Vec<Picture> = outer
                .iter()
                .zip(&only)
                .map(|(o, i)| {
                    let mut p = Picture::default();
                    splice(o, i, |mb| (zones >> r.zone_of(&bands, mb)) & 1 != 0, &mut p);
                    p
                })
                .collect();
            let how = delta.map_or("own aq".to_string(), |d| format!("recorded qp {d:+}"));
            measure(&format!("recoded from {name}, {how} [{v}] {ms:.1} ms"), &spliced, &only);
        }
    }

    /// What recoding the now playing card's region costs per frame at 1080p, for each set
    /// of extra options in RF_INNER_VARIANTS (";" between sets, "," within one).
    #[test]
    #[ignore = "a benchmark; run with --ignored --nocapture"]
    fn region_recode_cost() {
        let (w, h) = (1920, 1080);
        let r = Region { x0: 0, y0: 58, x1: 47, y1: 68, mb_width: 120 };
        let n = 96;
        let outer_frames: Vec<Vec<u8>> = (0..n).map(|t| frame(w, h, t as i32, None)).collect();
        let s = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
        let mut outer_enc = Encoder::open(&s).unwrap();
        let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
        let outer = encode(&mut outer_enc, w, h, &outer_frames, &plan);
        let mut by_pts = vec![(0, 0); n];
        for p in &outer {
            by_pts[p.pts as usize] = (p.kind, p.qp);
        }
        let inner_frames: Vec<Vec<u8>> = (0..n).map(|t| frame(w, h, t as i32, Some((&r, 5)))).collect();
        let variants = std::env::var("RF_INNER_VARIANTS").unwrap_or_default();
        for v in std::iter::once("").chain(variants.split(';').filter(|v| !v.is_empty())) {
            let mut inner_settings = s.clone();
            inner_settings.options.extend(["threads=1", "mbtree=0"].map(String::from));
            inner_settings.options.extend(v.split(',').filter(|o| !o.is_empty()).map(String::from));
            inner_settings.region_only = true;
            let mut enc = Encoder::open(&inner_settings).unwrap();
            let same = outer_enc.parameter_sets().unwrap() == enc.parameter_sets().unwrap();
            let t0 = std::time::Instant::now();
            let out = encode(&mut enc, w, h, &inner_frames, &by_pts);
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let bytes: usize = out.iter().flat_map(|p| p.nals.iter().filter(|x| x.is_slice() && r.holds(x.first_mb)).map(|x| x.len)).sum();
            eprintln!("inner [{v}]: {ms:.2} ms/frame, region {} kbit/s, same headers {same}", bytes * 8 * 24 / n / 1000);
        }
    }

    /// The syntax reader reads every slice x264 writes for the stream (bands, the card's
    /// region, B-frames, 8x8 transforms) to its end, every macroblock once, and finds the
    /// quantiser the decoder finds in each. RF_SPLICE_CLIP: real footage to read (raw
    /// 1080p yuv420p); without it, the synthetic frames.
    #[test]
    #[ignore = "needs libavcodec; run with --ignored"]
    fn syntax_reader_reads_every_slice() {
        use crate::h264syn::{Params, Reader};
        let (w, h, frames): (i32, i32, Vec<Vec<u8>>) = match std::env::var("RF_SPLICE_CLIP") {
            Ok(path) => {
                let clip = std::fs::read(path).unwrap();
                (1920, 1080, clip.chunks_exact(1920 * 1080 * 3 / 2).take(96).map(<[u8]>::to_vec).collect())
            }
            Err(_) => (640, 368, (0..96).map(|t| frame(640, 368, t, None)).collect()),
        };
        let mb_width = (w + 15) / 16;
        let mbs = (mb_width * ((h + 15) / 16)) as usize;
        let region = if w == 1920 { [0, 58, 47, 68] } else { [0, 8, 17, 12] };
        let mut s = settings(w, h, region);
        s.bands = if w == 1920 { vec![19, 32, 45, 58] } else { vec![4, 8, 12, 16, 20] };
        let mut enc = Encoder::open(&s).unwrap();
        let plan: Vec<(i32, i32)> = (0..frames.len()).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
        let pics = encode(&mut enc, w, h, &frames, &plan);
        let (sps, pps) = enc.parameter_sets().unwrap();
        let mut reader = Reader::new(Params::new(&sps, &pps).unwrap());
        let mut dec = crate::decode::Decoder::open(&sps, &pps, w, h, false, true).unwrap();
        let mut sides = vec![crate::decode::Side::default(); frames.len()];
        let (mut buf, mut side) = (Vec::new(), crate::decode::Side::default());
        for p in pics.iter().map(Some).chain(std::iter::once(None)) {
            dec.send(p).unwrap();
            while let Some(pts) = dec.receive(&mut buf, &mut side, None).unwrap() {
                sides[pts as usize] = side.clone();
            }
        }
        let qps: Vec<Vec<u8>> = sides.iter().map(|s| s.qp.clone()).collect();
        let mut mbs_read = Vec::new();
        let (mut kinds, mut t8, mut slices) = (std::collections::BTreeMap::new(), 0, 0);
        for p in &pics {
            let mut seen = vec![false; mbs];
            for n in p.nals.iter().filter(|n| n.is_slice()) {
                let header = reader.slice(&p.nal_bytes(n)[4..], &mut mbs_read).unwrap_or_else(|e| panic!("frame {}: slice at {}: {e}", p.pts, n.first_mb));
                assert_eq!(header.first_mb, n.first_mb);
                slices += 1;
                for m in &mbs_read {
                    assert!(!seen[m.addr as usize], "frame {}: macroblock {} twice", p.pts, m.addr);
                    seen[m.addr as usize] = true;
                    assert_eq!(m.qp, qps[p.pts as usize][m.addr as usize], "frame {}: macroblock {} ({:?}) quantiser", p.pts, m.addr, m.kind);
                    // The decoder's vectors agree with the syntax on which lists predict.
                    use crate::h264syn::Kind;
                    let uses: Vec<[i8; 2]> = (0..4).map(|b| {
                        let k = (m.addr as usize * 4 + b) * 2;
                        [sides[p.pts as usize].mv_use[k], sides[p.pts as usize].mv_use[k + 1]]
                    }).collect();
                    let want: Option<[i8; 2]> = match m.kind {
                        Kind::I4x4 | Kind::I8x8 | Kind::I16x16 => Some([0, 0]),
                        Kind::P16x16 | Kind::PSkip | Kind::B(1) => Some([1, 0]),
                        Kind::B(2) => Some([0, 1]),
                        Kind::B(3) => Some([1, 1]),
                        _ => None,
                    };
                    match want {
                        Some(w) => assert!(uses.iter().all(|&u| u == w), "frame {}: macroblock {} ({:?}) lists {uses:?}", p.pts, m.addr, m.kind),
                        None => assert!(uses.iter().all(|&u| u != [0, 0] && !u.contains(&2)), "frame {}: macroblock {} ({:?}) lists {uses:?}", p.pts, m.addr, m.kind),
                    }
                    *kinds.entry(format!("{:?}", m.kind)).or_insert(0) += 1;
                    t8 += m.t8x8 as usize;
                }
            }
            assert!(seen.iter().all(|&s| s), "frame {}: macroblocks no slice holds", p.pts);
        }
        eprintln!("{slices} slices read; kinds {kinds:?}; {t8} with 8x8 transforms");
    }

    /// Recoding bands with every macroblock given as it was coded writes the recording's
    /// own slices again, byte for byte, from the decoded recording and its quantisers.
    /// RF_SPLICE_CLIP: real footage (raw 1080p yuv420p); without it, synthetic frames.
    #[test]
    #[ignore = "needs libavcodec; run with --ignored"]
    fn copied_macroblocks_come_out_as_recorded() {
        use crate::h264syn::{copy_of, Params, Reader};
        use crate::x264::Copies;
        let (w, h, frames): (i32, i32, Vec<Vec<u8>>) = match std::env::var("RF_SPLICE_CLIP") {
            Ok(path) => {
                let clip = std::fs::read(path).unwrap();
                (1920, 1080, clip.chunks_exact(1920 * 1080 * 3 / 2).take(96).map(<[u8]>::to_vec).collect())
            }
            Err(_) => (640, 368, (0..96).map(|t| frame(640, 368, t, None)).collect()),
        };
        let n = frames.len();
        let mb_width = (w + 15) / 16;
        let mbs = (mb_width * ((h + 15) / 16)) as usize;
        let (r, bands) = if w == 1920 {
            (Region { x0: 0, y0: 58, x1: 47, y1: 68, mb_width }, vec![19, 32, 45, 58])
        } else {
            (Region { x0: 0, y0: 8, x1: 17, y1: 12, mb_width }, vec![4, 8, 12, 16, 20])
        };
        let mut s = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
        s.bands = bands.clone();
        let mut outer_enc = Encoder::open(&s).unwrap();
        let plan: Vec<(i32, i32)> = (0..n).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
        let outer = encode(&mut outer_enc, w, h, &frames, &plan);
        let (sps, pps) = outer_enc.parameter_sets().unwrap();
        let mut by_pts = vec![(0, 0); n];
        for p in &outer {
            by_pts[p.pts as usize] = (p.kind, p.qp);
        }
        let mut dec = crate::decode::Decoder::open(&sps, &pps, w, h, false, true).unwrap();
        let mut decoded = vec![Vec::new(); n];
        let mut sides = vec![crate::decode::Side::default(); n];
        let (mut buf, mut side) = (Vec::new(), crate::decode::Side::default());
        for p in outer.iter().map(Some).chain(std::iter::once(None)) {
            dec.send(p).unwrap();
            while let Some(pts) = dec.receive(&mut buf, &mut side, None).unwrap() {
                decoded[pts as usize] = buf.clone();
                sides[pts as usize] = side.clone();
            }
        }
        // Every recorded macroblock, as the syntax reader reads it.
        let mut reader = Reader::new(Params::new(&sps, &pps).unwrap());
        let mut mbs_of = vec![Vec::new(); n];
        let mut read = Vec::new();
        for p in &outer {
            for nal in p.nals.iter().filter(|x| x.is_slice()) {
                reader.slice(&p.nal_bytes(nal)[4..], &mut read).unwrap();
                mbs_of[p.pts as usize].extend(read.drain(..));
            }
        }
        let zones = 1 << 2 | 1 << 3;
        let rows: Vec<i32> = (0..(h + 15) / 16).filter(|&y| (zones >> r.zone_of(&bands, y * mb_width + mb_width - 1)) & 1 != 0).collect();
        for window in [[0, 0], [rows[0], rows[rows.len() - 1] + 1]] {
            let mut inner_settings = s.clone();
            inner_settings.options.extend(["threads=1", "mbtree=0", "rc-lookahead=0", "b-adapt=0", "deadzone-inter=6", "deadzone-intra=3"].map(String::from));
            inner_settings.region_only = true;
            inner_settings.window = window;
            let mut enc = Encoder::open(&inner_settings).unwrap();
            let mut out = Vec::new();
            let mut pic = Picture::default();
            let (cw, ch) = (w / 2, h / 2);
            let mut refused = 0;
            let mut encode_s = 0.0;
            let mut parse_ms = 0.0;
            for i in 0..n {
                let f = &decoded[i];
                let planes = [&f[..(w * h) as usize], &f[(w * h) as usize..(w * h + cw * ch) as usize], &f[(w * h + cw * ch) as usize..]];
                let (kind, qp) = by_pts[i];
                let offsets: Vec<f32> = sides[i].qp.iter().map(|&q| q as f32 - qp as f32).collect();
                let mut copies = Copies { index: vec![-1; mbs], pool: Vec::new() };
                for m in &mbs_of[i] {
                    if (zones >> r.zone_of(&bands, m.addr)) & 1 == 0 {
                        continue;
                    }
                    match copy_of(m, &sides[i]) {
                        Some(c) => {
                            copies.index[m.addr as usize] = copies.pool.len() as i32;
                            copies.pool.push(c);
                        }
                        None => refused += 1,
                    }
                }
                let t = std::time::Instant::now();
                let got = enc.encode_copying(Some((planes, [w, cw, cw])), i as i64, kind, qp, i as i32, zones, Some(&offsets), Some(copies), &mut pic).unwrap();
                encode_s += t.elapsed().as_secs_f64();
                if got {
                    out.push(pic.clone());
                }
            }
            while enc.delayed() > 0 {
                if enc.encode(None, 0, 0, -1, -1, zones, &mut pic).unwrap() {
                    out.push(pic.clone());
                }
            }
            let copy_ms = encode_s * 1000.0 / n as f64;
            // The same recode encoded afresh, for what copying saves.
            let mut fresh = Encoder::open(&inner_settings).unwrap();
            let mut fresh_s = 0.0;
            let fresh_n = if std::env::var_os("RF_COPY_ONLY").is_some() { 0 } else { n };
            for i in 0..fresh_n {
                let f = &decoded[i];
                let planes = [&f[..(w * h) as usize], &f[(w * h) as usize..(w * h + cw * ch) as usize], &f[(w * h + cw * ch) as usize..]];
                let (kind, qp) = by_pts[i];
                let offsets: Vec<f32> = sides[i].qp.iter().map(|&q| q as f32 - qp as f32).collect();
                let t = std::time::Instant::now();
                fresh.encode_with(Some((planes, [w, cw, cw])), i as i64, kind, qp, i as i32, zones, Some(&offsets), &mut pic).unwrap();
                fresh_s += t.elapsed().as_secs_f64();
            }
            let fresh_ms = fresh_s * 1000.0 / n as f64;
            // And the reading alone.
            let t2 = std::time::Instant::now();
            let mut rd = Reader::new(Params::new(&sps, &pps).unwrap());
            for p in &outer {
                for nal in p.nals.iter().filter(|x| x.is_slice() && (zones >> r.zone_of(&bands, x.first_mb)) & 1 != 0) {
                    rd.slice(&p.nal_bytes(nal)[4..], &mut read).unwrap();
                }
            }
            parse_ms += t2.elapsed().as_secs_f64() * 1000.0 / n as f64;
            eprintln!("window {window:?}: {copy_ms:.2} ms a frame copying, {fresh_ms:.2} encoding afresh; reading the slices {parse_ms:.2} ms");
            assert_eq!(out.len(), n);
            let (mut same, mut differ, mut first) = (0, 0, None);
            for (o, i) in outer.iter().zip(&out) {
                assert_eq!((o.pts, o.kind), (i.pts, i.kind));
                for nal in i.nals.iter().filter(|x| x.is_slice() && (zones >> r.zone_of(&bands, x.first_mb)) & 1 != 0) {
                    let rec = o.nals.iter().find(|x| x.is_slice() && x.first_mb == nal.first_mb).expect("the slice in the recording");
                    if o.nal_bytes(rec) == i.nal_bytes(nal) {
                        same += 1;
                    } else {
                        differ += 1;
                        if first.is_none() {
                            // Where they part: the first macroblock read differently.
                            let mut rd = Reader::new(Params::new(&sps, &pps).unwrap());
                            let (mut a, mut b) = (Vec::new(), Vec::new());
                            let ha = rd.slice(&o.nal_bytes(rec)[4..], &mut a);
                            let hb = rd.slice(&i.nal_bytes(nal)[4..], &mut b);
                            eprintln!("headers: {ha:?}\n         {hb:?}");
                            for (x, y) in a.iter().zip(&b) {
                                let fx = format!("{:?} t8 {} cbp {} {} qp {} modes {:?} cm {} i16 {}", x.kind, x.t8x8, x.cbp_luma, x.cbp_chroma, x.qp, x.intra_modes, x.chroma_mode, x.i16_mode);
                                let fy = format!("{:?} t8 {} cbp {} {} qp {} modes {:?} cm {} i16 {}", y.kind, y.t8x8, y.cbp_luma, y.cbp_chroma, y.qp, y.intra_modes, y.chroma_mode, y.i16_mode);
                                let levels = x.luma4x4 == y.luma4x4 && x.luma8x8 == y.luma8x8 && x.luma_dc == y.luma_dc && x.chroma_dc == y.chroma_dc && x.chroma_ac == y.chroma_ac;
                                if fx != fy || !levels {
                                    eprintln!("macroblock {}: recorded {fx}\n                copied   {fy}; levels same {levels}", x.addr);
                                    let k = x.addr as usize * 8;
                                    eprintln!("  vectors {:?} use {:?}", &sides[o.pts as usize].mv[k * 2..k * 2 + 16], &sides[o.pts as usize].mv_use[k..k + 8]);
                                    break;
                                }
                            }
                        }
                        first.get_or_insert((o.pts, o.kind, nal.first_mb, o.nal_bytes(rec).len(), i.nal_bytes(nal).len()));
                    }
                }
            }
            eprintln!("window {window:?}: {same} slices the same, {differ} not (first: {first:?}); {refused} macroblocks not copyable");
            assert_eq!(differ, 0);
        }
    }

    #[test]
    fn rect_conversions_match_the_whole_frames() {
        let (w, h) = (64usize, 32usize);
        let mut seed = 3u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };
        let yuv: Vec<u8> = (0..w * h * 3 / 2).map(|_| rnd()).collect();
        let mut whole = [Vec::new(), Vec::new(), Vec::new()];
        from_yuv420(&yuv, w, h, &mut whole);
        let mut part = [vec![0; w * h], vec![0; w * h], vec![0; w * h]];
        let rect = [6, 4, 50, 20];
        yuv_rect_to_rgb(&yuv, w, h, rect, &mut part);
        for y in 4..20 {
            for x in 6..50 {
                for c in 0..3 {
                    assert_eq!(part[c][y * w + x], whole[c][y * w + x], "({x}, {y})");
                }
            }
        }
        // And back: the vector and scalar luma agree.
        let mut a = yuv.clone();
        rgb_rect_to_yuv(&whole, w, h, rect, &mut a);
        for y in 4..20 {
            for x in 6..50 {
                let i = y * w + x;
                let (r, g, b) = (whole[0][i] as i32, whole[1][i] as i32, whole[2][i] as i32);
                assert_eq!(a[i], (16 + ((11966 * r + 40254 * g + 4064 * b + 32768) >> 16)) as u8);
            }
        }
    }

    #[test]
    fn a_windows_sps_agrees_where_slices_look() {
        let s = settings(1920, 1080, [0, 58, 47, 68]);
        let mut inner = s.clone();
        inner.options.extend(["threads=1", "mbtree=0", "rc-lookahead=0", "b-adapt=0"].map(String::from));
        inner.region_only = true;
        inner.window = [58, 68];
        inner.options.retain(|o| o != "nal-hrd=cbr");
        inner.options.push("nal-hrd=none".into());
        let (full, _) = Encoder::open(&s).unwrap().parameter_sets().unwrap();
        let (window, _) = Encoder::open(&inner).unwrap().parameter_sets().unwrap();
        assert_ne!(full, window, "the window's frame is smaller");
        let fields = crate::x264::sps_slice_fields(&full);
        assert!(fields.is_some());
        assert_eq!(fields, crate::x264::sps_slice_fields(&window));
        let mut other = s.clone();
        other.options.push("bframes=0".into());
        let (no_b, _) = Encoder::open(&other).unwrap().parameter_sets().unwrap();
        assert_ne!(fields, crate::x264::sps_slice_fields(&no_b), "without B-frames the picture order count differs");
    }

    #[test]
    fn replay_stops_only_where_all_sent_is_a_run_of_frames() {
        let kinds = [(0, 1), (3, 3), (1, 4), (2, 5), (4, 3)];
        let pictures = kinds.iter().map(|&(pts, kind)| Picture { pts, kind, ..Picture::default() }).collect();
        let gop = Gop::new(pictures, Vec::new(), Vec::new(), 0.0);
        assert_eq!(gop.len(), 5);
        assert_eq!(gop.cuts, [true, false, false, true, true]);
        assert_eq!(gop.plan.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 4, 5, 3, 3]);
    }

    #[test]
    fn the_region_covers_the_card_to_the_frame_edges() {
        let r = Region::covering([12, 931, 740, 1080], 1920, 1080);
        assert_eq!((r.x0, r.y0, r.x1, r.y1, r.mb_width), (0, 58, 47, 68, 120));
        assert_eq!(r.pixels(1920, 1080), (0, 928, 752, 152));
        assert!(r.holds(58 * 120) && r.holds(67 * 120 + 46) && !r.holds(58 * 120 + 47) && !r.holds(57 * 120 + 5));
    }

    #[test]
    fn yuv_round_trip_is_within_a_step() {
        let (w, h) = (32, 16);
        // Smooth, as scenes mostly are; sharp colour edges clip on the way back, which is
        // why a replay only converts where the card is drawn.
        let rgb: [Vec<u8>; 3] = std::array::from_fn(|c| (0..w * h).map(|i| (40 + (i % w) * 3 + (i / w) * (2 + c)) as u8).collect());
        let jobs = crate::jobs::Jobs::new(0);
        let mut yuv = vec![0u8; w * h * 3 / 2];
        crate::to_yuv420(&rgb, &mut yuv, w as i32, h as i32, &jobs);
        let mut back: [Vec<u8>; 3] = Default::default();
        from_yuv420(&yuv, w, h, &mut back);
        let mut again = vec![0u8; yuv.len()];
        crate::to_yuv420(&back, &mut again, w as i32, h as i32, &jobs);
        let worst = yuv.iter().zip(&again).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(worst <= 2, "4:2:0 to RGB and back moved a sample by {worst}");
    }

    /// The resident memory each kind of encoder holds at 1080p once it is going, for each
    /// set of extra options in RF_INNER_VARIANTS.
    #[test]
    #[ignore = "a measurement; run with --ignored --nocapture --test-threads=1"]
    fn encoder_memory() {
        fn rss() -> f64 {
            let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
            statm.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() * 4096.0 / 1e6
        }
        let (w, h) = (1920, 1080);
        let r = Region { x0: 0, y0: 58, x1: 47, y1: 68, mb_width: 120 };
        let frames: Vec<Vec<u8>> = (0..30).map(|t| frame(w, h, t, None)).collect();
        let base = settings(w, h, [r.x0, r.y0, r.x1, r.y1]);
        let variants = std::env::var("RF_INNER_VARIANTS").unwrap_or_default();
        let mut runs = vec![("live".to_string(), base.clone())];
        for v in std::iter::once("").chain(variants.split(';').filter(|v| !v.is_empty())) {
            let mut s = base.clone();
            s.options.extend(["threads=1", "mbtree=0", "rc-lookahead=0", "b-adapt=0"].map(String::from));
            s.options.extend(v.split(',').filter(|o| !o.is_empty()).map(String::from));
            s.region_only = true;
            runs.push((format!("inner [{v}]"), s));
        }
        let mut by_pts = Vec::new();
        for (name, s) in runs {
            let before = rss();
            let mut enc = Encoder::open(&s).unwrap();
            if !s.region_only {
                let plan: Vec<(i32, i32)> = (0..frames.len()).map(|i| (if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO }, -1)).collect();
                let out = encode(&mut enc, w, h, &frames, &plan);
                by_pts = vec![(0, 0); frames.len()];
                for p in &out {
                    by_pts[p.pts as usize] = (p.kind, p.qp);
                }
            } else {
                let _ = encode_zones(&mut enc, w, h, &frames, &by_pts, 1);
            }
            eprintln!("{name}: {:.0} MB", rss() - before);
            drop(enc);
        }
    }
}
