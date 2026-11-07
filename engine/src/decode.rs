//! The stream's own pictures decoded again (libavcodec, through src/avglue.c), and the
//! recorded orbit as frames: what overlay mode draws players over instead of rendering
//! the world again.

use crate::replay::{Gop, Region};
use crate::x264::{Nal, Picture};
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

unsafe extern "C" {
    fn rf_dec_open(sps: *const u8, sps_len: c_int, pps: *const u8, pps_len: c_int, partial: c_int, vectors: c_int) -> *mut c_void;
    fn rf_dec_send(d: *mut c_void, data: *const u8, len: c_int, pts: i64) -> c_int;
    fn rf_dec_receive(d: *mut c_void, yuv: *mut u8, rows: *const u8, qp: *mut u8, mv: *mut i16, mv_use: *mut i8, width: c_int,
                      height: c_int, pts: *mut i64) -> c_int;
    fn rf_dec_reset(d: *mut c_void);
    fn rf_dec_close(d: *mut c_void);
}

/// What the decoder found of each macroblock of a frame (only right in the zones decoded).
#[derive(Clone, Default)]
pub struct Side {
    /// The quantiser it was coded at, 255 where the decoder does not say.
    pub qp: Vec<u8>,
    /// The motion vectors of its 8x8 blocks, in quarter pixels (only from a decoder opened
    /// with `vectors`; empty otherwise):
    /// [((mb * 4 + b8) * 2 + list) * 2 + xy], b8 raster order within the macroblock.
    pub mv: Vec<i16>,
    /// [(mb * 4 + b8) * 2 + list]: 1 where the list predicts the block, 0 where not, 2
    /// where the block's parts have different vectors.
    pub mv_use: Vec<i8>,
}

pub struct Decoder {
    d: NonNull<c_void>,
    w: i32,
    h: i32,
    vectors: bool,
}

// libavcodec contexts have no thread affinity; one is only driven from one thread at a time.
unsafe impl Send for Decoder {}

impl Decoder {
    /// `partial`: pictures may lack slices, whose macroblocks are then left undecoded.
    /// `vectors`: Side::mv and mv_use are filled in, which costs the decoder a list of
    /// every motion vector of every frame.
    pub fn open(sps: &[u8], pps: &[u8], w: i32, h: i32, partial: bool, vectors: bool) -> Result<Decoder, String> {
        // SAFETY: both buffers outlive the call; the glue copies them.
        let d = unsafe {
            rf_dec_open(sps.as_ptr(), sps.len() as c_int, pps.as_ptr(), pps.len() as c_int, partial as c_int, vectors as c_int)
        };
        NonNull::new(d).map(|d| Decoder { d, w, h, vectors }).ok_or_else(|| "libavcodec has no usable H.264 decoder".to_string())
    }

    /// One picture in decoding order, or None to drain what is held.
    pub fn send(&mut self, pic: Option<&Picture>) -> Result<(), String> {
        // SAFETY: the data outlives the call; the glue copies it.
        let r = unsafe {
            match pic {
                Some(p) => rf_dec_send(self.d.as_ptr(), p.data.as_ptr(), p.data.len() as c_int, p.pts),
                None => rf_dec_send(self.d.as_ptr(), std::ptr::null(), 0, 0),
            }
        };
        if r < 0 { Err("the decoder refused a picture".into()) } else { Ok(()) }
    }

    /// The next frame in display order, into `yuv`, what the decoder found of each
    /// macroblock into `side`, and its pts. With `rows` (a byte for each macroblock row),
    /// only the rows whose byte is not 0 are written into `yuv`.
    pub fn receive(&mut self, yuv: &mut Vec<u8>, side: &mut Side, rows: Option<&[u8]>) -> Result<Option<i64>, String> {
        yuv.resize((self.w * self.h * 3 / 2) as usize, 0);
        let mb_rows = ((self.h + 15) / 16) as usize;
        let mbs = ((self.w + 15) / 16) as usize * mb_rows;
        assert!(rows.is_none_or(|r| r.len() >= mb_rows));
        side.qp.resize(mbs, 255);
        let (mv, mv_use) = if self.vectors {
            side.mv.resize(mbs * 16, 0);
            side.mv_use.resize(mbs * 8, 0);
            (side.mv.as_mut_ptr(), side.mv_use.as_mut_ptr())
        } else {
            side.mv.clear();
            side.mv_use.clear();
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        let mut pts = 0i64;
        // SAFETY: yuv holds a whole frame of the decoder's size, rows a byte for each of
        // its macroblock rows, and side a macroblock's worth of each for every macroblock
        // (the vectors only when asked for, and null otherwise).
        match unsafe {
            rf_dec_receive(self.d.as_ptr(), yuv.as_mut_ptr(), rows.map_or(std::ptr::null(), |r| r.as_ptr()), side.qp.as_mut_ptr(),
                           mv, mv_use, self.w, self.h, &mut pts)
        } {
            1 => Ok(Some(pts)),
            0 => Ok(None),
            _ => Err("the decoder failed on a picture".into()),
        }
    }

    pub fn reset(&mut self) {
        // SAFETY: the handle is open.
        unsafe { rf_dec_reset(self.d.as_ptr()) }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: opened by rf_dec_open, closed once.
        unsafe { rf_dec_close(self.d.as_ptr()) }
    }
}

/// Frames of recorded groups, decoded in order as they are asked for. Asking for the next
/// frame of the same group costs about one picture's decoding; starting a group part way
/// decodes it from its IDR.
pub struct Orbit {
    dec: Decoder,
    region: Region,
    bands: Vec<i32>,
    zones: i32,    // the zones of `gop` decoded, -1 for all
    rows: Vec<u8>, // and the macroblock rows they are in, a byte each (nothing for all)
    gop: Option<Arc<Gop>>,
    fed: usize,    // pictures of `gop` sent
    drained: bool, // and the end of it
    from: usize,   // frames before this one were let go
    ready: Vec<(i64, Vec<u8>, Side)>,
    spare: Vec<(Vec<u8>, Side)>,
    side: Side, // of the frame last given out
    picture: Picture,
    kept: Picture, // `picture`'s slices of the zones decoded
}

impl Orbit {
    /// Zones are numbered as Region::zone_of numbers them, for that region and bands.
    pub fn new(sps: &[u8], pps: &[u8], w: i32, h: i32, region: Region, bands: Vec<i32>) -> Result<Orbit, String> {
        Ok(Orbit { dec: Decoder::open(sps, pps, w, h, true, false)?, region, bands, zones: -1, rows: Vec::new(), gop: None, fed: 0, drained: false, from: 0, ready: Vec::new(), spare: Vec::new(), side: Side::default(), picture: Picture::default(), kept: Picture::default() })
    }

    /// Frame j (display order) of `gop`, as planar 4:2:0, swapped into `out`, right in the
    /// zones `zones` (bit z for zone z, -1 for all) and anything elsewhere. Zones only ever
    /// predict from themselves, so the others' slices are not decoded at all; asking for
    /// more zones part way through a group decodes it again from its IDR.
    pub fn frame(&mut self, gop: &Arc<Gop>, j: usize, out: &mut Vec<u8>, zones: i32) -> Result<(), String> {
        match self.try_frame(gop, j, out, zones) {
            Ok(()) => Ok(()),
            Err(e) if zones != -1 => {
                // Once more, the whole group and every zone, before giving up.
                eprintln!("overlay: {e} decoding zones {zones:#x}, decoding the group whole");
                self.gop = None;
                self.try_frame(gop, j, out, -1)
            }
            Err(e) => Err(e),
        }
    }

    /// What the decoder found of each macroblock of the frame `frame` last gave; right
    /// only in the zones decoded.
    pub fn side(&self) -> &Side {
        &self.side
    }

    fn try_frame(&mut self, gop: &Arc<Gop>, j: usize, out: &mut Vec<u8>, zones: i32) -> Result<(), String> {
        let same = self.gop.as_ref().is_some_and(|g| Arc::ptr_eq(g, gop));
        if !same || j < self.from || zones & !self.zones != 0 {
            self.start(gop);
            self.zones = zones;
            // Only these rows of each frame are copied out: nothing reads the others.
            self.rows.clear();
            if zones != -1 {
                let (w, h) = (self.dec.w, self.dec.h);
                let mb_width = (w + 15) / 16;
                let (region, bands) = (self.region, &self.bands);
                self.rows = (0..(h + 15) / 16)
                    .map(|y| (0..mb_width).any(|x| (zones >> region.zone_of(bands, y * mb_width + x)) & 1 != 0) as u8)
                    .collect();
            }
        }
        self.from = j;
        loop {
            // Frames before j will not be asked for again.
            let mut k = 0;
            while k < self.ready.len() {
                if self.ready[k].0 < j as i64 {
                    let (_, buf, qp) = self.ready.swap_remove(k);
                    self.spare.push((buf, qp));
                } else {
                    k += 1;
                }
            }
            if let Some(i) = self.ready.iter().position(|f| f.0 == j as i64) {
                let (_, buf, qp) = self.ready.swap_remove(i);
                let qp = std::mem::replace(&mut self.side, qp);
                self.spare.push((std::mem::replace(out, buf), qp));
                // A frame or two to decode into is all that is ever needed again.
                self.spare.truncate(2);
                return Ok(());
            }
            let gop = self.gop.clone().unwrap();
            if self.fed < gop.len() {
                gop.picture(self.fed, &mut self.picture)?;
                if self.zones != -1 {
                    let (region, bands, zones) = (self.region, &self.bands, self.zones);
                    let p = &self.picture;
                    let kept = &mut self.kept;
                    (kept.pts, kept.dts, kept.kind, kept.qp) = (p.pts, p.dts, p.kind, p.qp);
                    kept.data.clear();
                    kept.nals.clear();
                    for n in &p.nals {
                        if !n.is_slice() || (zones >> region.zone_of(bands, n.first_mb)) & 1 != 0 {
                            kept.nals.push(Nal { at: kept.data.len(), ..*n });
                            kept.data.extend_from_slice(p.nal_bytes(n));
                        }
                    }
                    self.dec.send(Some(&self.kept))?;
                } else {
                    self.dec.send(Some(&self.picture))?;
                }
                self.fed += 1;
            } else if !self.drained {
                self.dec.send(None)?;
                self.drained = true;
            } else {
                return Err(format!("frame {j} never came out of its group"));
            }
            loop {
                let (mut buf, mut qp) = self.spare.pop().unwrap_or_default();
                let rows = Some(&self.rows[..]).filter(|r| !r.is_empty());
                match self.dec.receive(&mut buf, &mut qp, rows)? {
                    Some(pts) => self.ready.push((pts, buf, qp)),
                    None => {
                        self.spare.push((buf, qp));
                        break;
                    }
                }
            }
        }
    }

    fn start(&mut self, gop: &Arc<Gop>) {
        self.dec.reset();
        self.gop = Some(gop.clone());
        self.fed = 0;
        self.from = 0;
        self.drained = false;
        self.spare.extend(self.ready.drain(..).map(|f| (f.1, f.2)));
        self.spare.truncate(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x264::{Encoder, Settings, TYPE_AUTO, TYPE_IDR};

    /// Decoding only some zones' slices gives those zones exactly as decoding everything,
    /// for a fraction of the time. RF_SPLICE_CLIP: raw 1920x1080 yuv420p frames.
    #[test]
    #[ignore = "a benchmark; needs RF_SPLICE_CLIP; run with --ignored --nocapture"]
    fn partial_decode() {
        let (w, h) = (1920, 1080);
        let clip = std::fs::read(std::env::var("RF_SPLICE_CLIP").expect("RF_SPLICE_CLIP")).unwrap();
        let frame_len = (w * h * 3 / 2) as usize;
        let frames: Vec<&[u8]> = clip.chunks_exact(frame_len).take(96).collect();
        let r = Region { x0: 0, y0: 58, x1: 47, y1: 68, mb_width: 120 };
        let bands = vec![19, 32, 45, 58];
        let options = ["level=4.1", "threads=2", "bitrate=10000", "vbv-maxrate=10000", "vbv-bufsize=10000", "keyint=48",
                       "min-keyint=48", "scenecut=0", "bframes=2", "nal-hrd=cbr", "force-cfr=1", "mbtree=1", "rc-lookahead=10",
                       "direct=spatial", "stitchable=1"];
        let s = Settings { preset: "superfast".into(), tune: String::new(), profile: "high".into(), width: w, height: h, fps: 24,
                           options: options.iter().map(|o| o.to_string()).collect(), region: [r.x0, r.y0, r.x1, r.y1],
                           region_only: false, bands: bands.clone(), window: [0, 0] };
        let mut enc = Encoder::open(&s).unwrap();
        let (sps, pps) = enc.parameter_sets().unwrap();
        let mut pics = Vec::new();
        let mut pic = Picture::default();
        for (i, f) in frames.iter().enumerate() {
            let planes = [&f[..(w * h) as usize], &f[(w * h) as usize..(w * h * 5 / 4) as usize], &f[(w * h * 5 / 4) as usize..]];
            let kind = if i % 48 == 0 { TYPE_IDR } else { TYPE_AUTO };
            if enc.encode(Some((planes, [w, w / 2, w / 2])), i as i64, kind, -1, i as i32, 0, &mut pic).unwrap() {
                pics.push(pic.clone());
            }
        }
        while enc.delayed() > 0 {
            if enc.encode(None, 0, 0, -1, -1, 0, &mut pic).unwrap() {
                pics.push(pic.clone());
            }
        }
        let decode_all = |zones: Option<i32>| -> (Vec<Vec<u8>>, f64) {
            let mut d = Decoder::open(&sps, &pps, w, h, zones.is_some(), false).unwrap();
            let mut out = vec![Vec::new(); pics.len()];
            let t0 = std::time::Instant::now();
            let (mut buf, mut side) = (Vec::new(), crate::decode::Side::default());
            let mut feed = |p: Option<&Picture>, d: &mut Decoder, out: &mut Vec<Vec<u8>>| {
                d.send(p).unwrap();
                while let Some(pts) = d.receive(&mut buf, &mut side, None).unwrap() {
                    out[pts as usize] = buf.clone();
                }
            };
            for p in &pics {
                let kept = match zones {
                    None => p.clone(),
                    Some(z) => {
                        let mut q = Picture { pts: p.pts, ..Picture::default() };
                        for n in &p.nals {
                            if !n.is_slice() || (z >> r.zone_of(&bands, n.first_mb)) & 1 != 0 {
                                q.nals.push(Nal { at: q.data.len(), ..*n });
                                q.data.extend_from_slice(p.nal_bytes(n));
                            }
                        }
                        q
                    }
                };
                feed(Some(&kept), &mut d, &mut out);
            }
            feed(None, &mut d, &mut out);
            (out, t0.elapsed().as_secs_f64() * 1000.0 / pics.len() as f64)
        };
        let (full, full_ms) = decode_all(None);
        eprintln!("full decode: {full_ms:.2} ms a frame");
        for zones in [1, 1 | 1 << 3, 1 | 1 << 2 | 1 << 3] {
            let (part, ms) = decode_all(Some(zones));
            let mut wrong = 0usize;
            for f in 0..full.len() {
                for (plane, pw, ph, mb) in [(0usize, w, h, 16), ((w * h) as usize, w / 2, h / 2, 8), ((w * h * 5 / 4) as usize, w / 2, h / 2, 8)] {
                    for y in 0..ph {
                        for x in 0..pw {
                            let z = r.zone_of(&bands, (y / mb) * 120 + x / mb);
                            if (zones >> z) & 1 != 0 {
                                let k = plane + (y * pw + x) as usize;
                                wrong += (full[f][k] != part[f][k]) as usize;
                            }
                        }
                    }
                }
            }
            eprintln!("zones {zones:#b}: {ms:.2} ms a frame, {wrong} samples differ from the full decode");
            assert_eq!(wrong, 0);
        }
        // What exporting motion vectors and copying out every row cost: the best of five
        // decodes of all the pictures.
        let rows: Vec<u8> = (0..(h + 15) / 16).map(|y| (19..32).contains(&y) as u8).collect();
        for (vectors, rows) in [(true, None), (false, None), (false, Some(&rows[..]))] {
            let mut best = f64::MAX;
            let (mut buf, mut side) = (Vec::new(), crate::decode::Side::default());
            for _ in 0..5 {
                let mut d = Decoder::open(&sps, &pps, w, h, false, vectors).unwrap();
                let t0 = std::time::Instant::now();
                for p in pics.iter().map(Some).chain(std::iter::once(None)) {
                    d.send(p).unwrap();
                    while d.receive(&mut buf, &mut side, rows).unwrap().is_some() {}
                }
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / pics.len() as f64);
            }
            eprintln!("vectors {vectors}, rows {}: {best:.2} ms a frame", if rows.is_some() { "13" } else { "all" });
        }
    }
}
