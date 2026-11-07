//! The patched x264 (`x264/` at the top of the repository), linked statically. Its one addition is a region: a
//! rectangle of macroblocks coded in slices of its own and predicted only from itself,
//! so another encoder with the same settings can recode just that rectangle of a picture
//! and its slices take the place of the first encoder's (see replay.rs).

use std::ffi::{CString, c_char, c_int, c_void};
use std::ptr::NonNull;

pub const TYPE_AUTO: i32 = 0;
pub const TYPE_IDR: i32 = 1;

pub const NAL_SLICE: u8 = 1;
pub const NAL_SLICE_IDR: u8 = 5;
pub const NAL_SPS: u8 = 7;
pub const NAL_PPS: u8 = 8;

#[repr(C)]
struct RawNal {
    ref_idc: c_int,
    kind: c_int,
    long_startcode: c_int,
    first_mb: c_int,
    last_mb: c_int,
    payload_len: c_int,
    payload: *mut u8,
    padding: c_int,
}

#[repr(C)]
#[derive(Default)]
struct RawPicture {
    pts: i64,
    dts: i64,
    kind: c_int,
    qp: c_int,
    keyframe: c_int,
}

unsafe extern "C" {
    fn rf_x264_open(preset: *const c_char, tune: *const c_char, profile: *const c_char, width: c_int, height: c_int,
                    fps: c_int, options: *const *const c_char, n_options: c_int, region: *const c_int, bands: *const c_int,
                    n_bands: c_int, err: *mut c_char, err_len: usize) -> *mut c_void;
    fn rf_x264_headers(h: *mut c_void, nals: *mut *mut RawNal) -> c_int;
    fn rf_x264_encode(h: *mut c_void, planes: *const *mut u8, strides: *const c_int, pts: i64, kind: c_int, qp: c_int,
                      idr_pic_id: c_int, zones: c_int, quant_offsets: *mut f32, copy_index: *mut i32, copy_pool: *mut CopyMb, nals: *mut *mut RawNal, n_nals: *mut c_int, out: *mut RawPicture) -> c_int;
    fn rf_x264_delayed(h: *mut c_void) -> c_int;
    fn rf_x264_copy_mb_size() -> c_int;
    fn rf_x264_max_delayed(h: *mut c_void) -> c_int;
    fn rf_x264_close(h: *mut c_void);
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub preset: String,
    pub tune: String,
    pub profile: String,
    pub width: i32,
    pub height: i32,
    pub fps: i32,
    /// x264 option strings, key=value, as x264_param_parse takes them.
    pub options: Vec<String>,
    /// Macroblocks [x0, y0, x1, y1); all zero for none.
    pub region: [i32; 4],
    /// Only some zones' slices are wanted (each picture says which): nothing else is encoded.
    pub region_only: bool,
    /// Macroblock rows where one band ends and the next begins: bands are coded apart
    /// from one another, as the region is from the rest.
    pub bands: Vec<i32>,
    /// For an encoder with region_only: the macroblock rows [top, bottom) it is given of
    /// each frame, the rest of which it would only hold and never encode ([0, 0] for the
    /// whole frame). Everything else stays in the whole frame's terms: the region, the
    /// bands, the frames passed in, the zones asked for and the slices that come out.
    pub window: [i32; 2],
}

/// One NAL unit of a picture: its bytes in `Picture::data`, 4-byte length first.
#[derive(Clone, Copy, Debug)]
pub struct Nal {
    pub kind: u8,
    pub first_mb: i32,
    pub at: usize,
    pub len: usize,
}

impl Nal {
    pub fn is_slice(&self) -> bool {
        self.kind == NAL_SLICE || self.kind == NAL_SLICE_IDR
    }
}

/// An encoded picture, in decoding order.
#[derive(Clone, Debug, Default)]
pub struct Picture {
    pub pts: i64,
    pub dts: i64,
    pub kind: i32,
    pub qp: i32,
    pub data: Vec<u8>,
    pub nals: Vec<Nal>,
}

impl Picture {
    pub fn nal_bytes(&self, n: &Nal) -> &[u8] {
        &self.data[n.at..n.at + n.len]
    }
}

/// What of an SPS (without its NAL header byte or start code) a slice's syntax depends
/// on: the profile, chroma format and bit depths, frame_num's and the picture order count's
/// sizes and kind, frame_mbs_only and direct_8x8_inference. Two encoders whose SPS agree
/// in these write slices that decode under either's SPS, whatever their frame sizes.
pub fn sps_slice_fields(sps: &[u8]) -> Option<Vec<u32>> {
    // Emulation prevention bytes out first.
    let mut rbsp = Vec::with_capacity(sps.len());
    let mut zeros = 0;
    for &b in sps.get(1..)? {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        rbsp.push(b);
    }
    let mut at = 0usize;
    let mut bit = || -> Option<u32> {
        let b = (rbsp.get(at / 8)? >> (7 - at % 8)) & 1;
        at += 1;
        Some(b as u32)
    };
    let bits = |n: u32, bit: &mut dyn FnMut() -> Option<u32>| -> Option<u32> { (0..n).try_fold(0, |v, _| Some(v << 1 | bit()?)) };
    let ue = |bit: &mut dyn FnMut() -> Option<u32>| -> Option<u32> {
        let mut zeros = 0;
        while bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let mut v = 0u32;
        for _ in 0..zeros {
            v = v << 1 | bit()?;
        }
        Some((1u32 << zeros) - 1 + v)
    };
    let profile = bits(8, &mut bit)?;
    let _constraints_and_level = bits(16, &mut bit)?;
    let mut f = vec![profile, ue(&mut bit)?];
    if [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].contains(&profile) {
        let chroma = ue(&mut bit)?;
        f.push(chroma);
        if chroma == 3 {
            f.push(bit()?);
        }
        f.extend([ue(&mut bit)?, ue(&mut bit)?, bit()?]);
        if bit()? == 1 {
            return None; // scaling matrices: x264 writes none without --cqm
        }
    }
    let log2_max_frame_num = ue(&mut bit)?;
    let poc_type = ue(&mut bit)?;
    f.extend([log2_max_frame_num, poc_type]);
    match poc_type {
        0 => f.push(ue(&mut bit)?),
        2 => {}
        _ => return None,
    }
    let _refs = ue(&mut bit)?;
    let _gaps = bit()?;
    let _width = ue(&mut bit)?;
    let _height = ue(&mut bit)?;
    let frame_mbs_only = bit()?;
    f.push(frame_mbs_only);
    if frame_mbs_only == 0 {
        f.push(bit()?);
    }
    f.push(bit()?); // direct_8x8_inference
    Some(f)
}

/// A macroblock as it was coded, for a region-only encoder to write as it was
/// (x264_copy_mb_t in x264.h, patch 0006).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CopyMb {
    pub kind: u8,
    pub t8x8: u8,
    pub qp: u8,
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub i16_mode: u8,
    pub chroma_mode: u8,
    pub intra_modes: [u8; 16],
    pub ref_idx: [[i8; 4]; 2],
    pub mv: [[[i16; 2]; 4]; 2],
    pub luma: [i16; 256],
    pub luma_dc: [i16; 16],
    pub chroma_dc: [[i16; 4]; 2],
    pub chroma_ac: [[i16; 16]; 8],
}

// x264_copy_mb_t's types. Copying macroblocks is, for now, used by the splice tests only.
#[cfg_attr(not(test), allow(dead_code))]
impl CopyMb {
    pub const I4X4: u8 = 0;
    pub const I8X8: u8 = 1;
    pub const I16X16: u8 = 2;
    pub const P16X16: u8 = 3;
    pub const P_SKIP: u8 = 4;
    pub const B_DIRECT: u8 = 5;
    pub const B_SKIP: u8 = 6;
    pub const B16X16: u8 = 7;
}

/// Macroblocks of a frame to write as they were coded: an entry per macroblock of the
/// whole frame, -1 or an index into `pool`.
#[derive(Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct Copies {
    pub index: Vec<i32>,
    pub pool: Vec<CopyMb>,
}

/// A qp offset for a macroblock that leaves it to adaptive quantization (x264.h).
pub const QUANT_OFFSET_AQ: f32 = 1e9;

pub struct Encoder {
    h: NonNull<c_void>,
    /// Macroblocks in a frame.
    mbs: usize,
    /// The window's first macroblock row, and how many bands lie wholly above it.
    top: i32,
    bands_above: i32,
    mb_width: usize,
    /// Pixel rows given to x264.
    height: i32,
    /// Copies given with frames not yet out, by pts: x264 reads them when it encodes.
    copies: Vec<(i64, Copies)>,
}

// x264 has no thread affinity; one encoder is only ever driven from one thread at a time.
unsafe impl Send for Encoder {}

fn collect(nals: *mut RawNal, n: c_int, out: &mut Picture) {
    out.data.clear();
    out.nals.clear();
    for i in 0..n.max(0) as usize {
        // SAFETY: x264 returned n NALs whose payloads stay valid until its next call.
        let nal = unsafe { &*nals.add(i) };
        let bytes = unsafe { std::slice::from_raw_parts(nal.payload, nal.payload_len as usize) };
        out.nals.push(Nal { kind: nal.kind as u8, first_mb: nal.first_mb, at: out.data.len(), len: bytes.len() });
        out.data.extend_from_slice(bytes);
    }
}

impl Encoder {
    pub fn open(s: &Settings) -> Result<Encoder, String> {
        let text = |v: &str| CString::new(v).map_err(|_| format!("x264 setting {v:?} has a NUL"));
        let options = s.options.iter().map(|o| text(o)).collect::<Result<Vec<_>, _>>()?;
        let pointers: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
        let (preset, tune, profile) = (text(&s.preset)?, text(&s.tune)?, text(&s.profile)?);
        // SAFETY: a constant.
        if unsafe { rf_x264_copy_mb_size() } as usize != std::mem::size_of::<CopyMb>() {
            return Err("x264_copy_mb_t and CopyMb differ".into());
        }
        let mb_width = (s.width + 15) / 16;
        let [top, bottom] = if s.window == [0, 0] { [0, (s.height + 15) / 16] } else { s.window };
        if !(0 <= top && top < bottom && bottom <= (s.height + 15) / 16) || (top > 0 && !s.region_only) {
            return Err(format!("no window {:?} in {} rows of macroblocks, or not for a region_only encoder", s.window, (s.height + 15) / 16));
        }
        let height = (16 * bottom).min(s.height) - 16 * top;
        // Bands at or above the top are gone, as are those at or below the bottom.
        let bands: Vec<i32> = s.bands.iter().filter(|&&c| c > top && c < bottom).map(|&c| c - top).collect();
        let bands_above = s.bands.iter().filter(|&&c| c <= top).count() as i32;
        // A region outside the window is none: its rows are a band's, so all or none of
        // them are in it.
        let r = if s.region[1] >= top && s.region[3] <= bottom { s.region } else { [0; 4] };
        let region = [r[0], (r[1] - top).max(0), r[2], (r[3] - top).max(0), s.region_only as i32, top * mb_width, (top > 0) as i32,
                      (bottom < (s.height + 15) / 16) as i32];
        let mut err = [0 as c_char; 256];
        // SAFETY: every pointer outlives the call; x264 copies what it keeps.
        let h = unsafe {
            rf_x264_open(preset.as_ptr(), tune.as_ptr(), profile.as_ptr(), s.width, height, s.fps, pointers.as_ptr(),
                         pointers.len() as c_int, region.as_ptr(), bands.as_ptr(), bands.len() as c_int, err.as_mut_ptr(), err.len())
        };
        let mbs = mb_width as usize * ((height + 15) / 16) as usize;
        NonNull::new(h).map(|h| Encoder { h, mbs, top, bands_above, mb_width: mb_width as usize, height, copies: Vec::new() }).ok_or_else(|| {
            let bytes: Vec<u8> = err.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// SPS and PPS, without length prefixes.
    pub fn parameter_sets(&mut self) -> Result<(Vec<u8>, Vec<u8>), String> {
        let mut nals = std::ptr::null_mut();
        // SAFETY: the handle is open; the NALs are read before the next call.
        let n = unsafe { rf_x264_headers(self.h.as_ptr(), &mut nals) };
        let mut headers = Picture::default();
        collect(nals, n, &mut headers);
        let find = |kind| {
            headers.nals.iter().find(|n| n.kind == kind).map(|n| headers.nal_bytes(n)[4..].to_vec())
        };
        match (find(NAL_SPS), find(NAL_PPS)) {
            (Some(sps), Some(pps)) => Ok((sps, pps)),
            _ => Err("x264 gave no SPS and PPS".into()),
        }
    }

    /// Encodes one 4:2:0 frame (or drains a delayed picture with `frame` None) and returns
    /// the picture that came out, if any. `kind` forces the frame type (TYPE_AUTO leaves it
    /// to x264), `qp` the quantiser (-1 leaves it to rate control).
    /// `zones`, for an encoder with region_only: which zones to encode, bit z for zone z (0
    /// the region, 1 + k band k); 0 for the region alone.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&mut self, frame: Option<([&[u8]; 3], [i32; 3])>, pts: i64, kind: i32, qp: i32, idr_pic_id: i32, zones: i32,
                  out: &mut Picture) -> Result<bool, String> {
        self.encode_with(frame, pts, kind, qp, idr_pic_id, zones, None, out)
    }

    /// As encode_with, with macroblocks to write as they were coded (in the whole frame's
    /// terms, as everything is), for an encoder with region_only.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn encode_copying(&mut self, frame: Option<([&[u8]; 3], [i32; 3])>, pts: i64, kind: i32, qp: i32, idr_pic_id: i32,
                          zones: i32, quant_offsets: Option<&[f32]>, copies: Option<Copies>, out: &mut Picture) -> Result<bool, String> {
        let mut index_ptr = std::ptr::null_mut();
        let mut pool_ptr = std::ptr::null_mut();
        if let (Some(mut c), Some(_)) = (copies, frame) {
            let skip = self.top as usize * self.mb_width;
            if c.index.len() < skip + self.mbs || c.index.iter().any(|&i| i >= c.pool.len() as i32) {
                return Err("copies for fewer macroblocks than the frame has, or past their pool".into());
            }
            // The window's part.
            c.index.drain(..skip);
            c.index.truncate(self.mbs);
            index_ptr = c.index.as_mut_ptr();
            pool_ptr = c.pool.as_mut_ptr();
            // Moving the Copies into the list moves the Vecs, not their heap buffers.
            self.copies.push((pts, c));
        }
        let r = self.encode_inner(frame, pts, kind, qp, idr_pic_id, zones, quant_offsets, index_ptr, pool_ptr, out);
        if let Ok(true) = r {
            // x264 is done with a frame once its picture is out.
            self.copies.retain(|(p, _)| *p != out.pts);
        }
        r
    }

    /// As encode, with a qp offset for every macroblock (raster order) added to `qp`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with(&mut self, frame: Option<([&[u8]; 3], [i32; 3])>, pts: i64, kind: i32, qp: i32, idr_pic_id: i32,
                       zones: i32, quant_offsets: Option<&[f32]>, out: &mut Picture) -> Result<bool, String> {
        let r = self.encode_inner(frame, pts, kind, qp, idr_pic_id, zones, quant_offsets, std::ptr::null_mut(), std::ptr::null_mut(), out);
        if let Ok(true) = r {
            self.copies.retain(|(p, _)| *p != out.pts);
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_inner(&mut self, frame: Option<([&[u8]; 3], [i32; 3])>, pts: i64, kind: i32, qp: i32, idr_pic_id: i32, zones: i32,
                    quant_offsets: Option<&[f32]>, copy_index: *mut i32, copy_pool: *mut CopyMb, out: &mut Picture)
                    -> Result<bool, String> {
        let mut nals = std::ptr::null_mut();
        let mut n = 0;
        let mut raw = RawPicture::default();
        // The window's part of the frame, and of the offsets.
        let skip = self.top as usize * self.mb_width;
        let quant_offsets = quant_offsets.map(|q| q.get(skip..).unwrap_or(&[]));
        if quant_offsets.is_some_and(|q| q.len() < self.mbs) {
            return Err("qp offsets for fewer macroblocks than the frame has".into());
        }
        let rows = [16 * self.top as usize, 8 * self.top as usize, 8 * self.top as usize];
        let heights = [self.height as usize, self.height as usize / 2, self.height as usize / 2];
        let frame = match frame {
            Some((p, st)) => {
                let mut planes = [&[][..]; 3];
                for i in 0..3 {
                    let at = rows[i] * st[i] as usize;
                    planes[i] = match p[i].get(at..) {
                        Some(rest) if rest.len() >= heights[i] * st[i] as usize => rest,
                        _ => return Err("a frame smaller than the encoder's".into()),
                    };
                }
                Some((planes, st))
            }
            None => None,
        };
        // Zones in the window's numbering: the region's stays 0, band k is band k less
        // those above.
        let zones = if self.bands_above == 0 { zones } else { (zones & 1) | ((zones >> 1 >> self.bands_above) << 1) };
        // x264 reads the planes, never writes them, despite its non-const pointers.
        let planes = frame.map(|(p, _)| p.map(|p| p.as_ptr() as *mut u8));
        let strides = frame.map_or([0; 3], |(_, s)| s);
        // x264 reads the offsets as the frame is taken in and never writes them.
        let quant = quant_offsets.filter(|_| frame.is_some()).map_or(std::ptr::null_mut(), |q| q.as_ptr() as *mut f32);
        // SAFETY: the planes cover the frame at the given strides (checked above, for
        // rows of a stride's width), the offsets every macroblock; the copies (an index
        // per macroblock into a pool, checked by encode_copying) stay in `self.copies`
        // until the frame's picture comes out or the encoder is closed; the NALs are read
        // before the next call.
        let bytes = unsafe {
            rf_x264_encode(self.h.as_ptr(), planes.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()), strides.as_ptr(),
                           pts, kind, qp, idr_pic_id, zones, quant, copy_index, copy_pool, &mut nals, &mut n, &mut raw)
        };
        if bytes < 0 {
            return Err("x264 failed to encode a frame".into());
        }
        if bytes == 0 {
            return Ok(false);
        }
        collect(nals, n, out);
        (out.pts, out.dts, out.kind, out.qp) = (raw.pts, raw.dts, raw.kind, raw.qp);
        Ok(true)
    }

    pub fn delayed(&self) -> i32 {
        // SAFETY: the handle is open.
        unsafe { rf_x264_delayed(self.h.as_ptr()) }
    }

    /// The most frames it ever holds before a picture comes out.
    pub fn max_delayed(&self) -> i32 {
        // SAFETY: the handle is open.
        unsafe { rf_x264_max_delayed(self.h.as_ptr()) }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: opened by rf_x264_open, closed once.
        unsafe { rf_x264_close(self.h.as_ptr()) }
    }
}
