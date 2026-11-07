//! Reads the macroblock syntax of H.264 slices as the engine's x264 writes them (High
//! profile, 4:2:0, 8-bit, progressive frames, CABAC), back to what each macroblock was
//! coded as: its type and partitions, intra prediction modes, coded block pattern,
//! quantiser and the levels of every block, in the order they are coded. Motion vectors
//! are left to the decoder, which works them out anyway; this reads the differences only
//! to stay in step.
//!
//! It is what lets a recorded macroblock be written again as it was rather than encoded
//! afresh. Anything outside what x264 is set up to write here (interlacing, PCM,
//! scaling matrices, more than one slice group) is refused with an error, never guessed.

// x264's CABAC tables: context initialisation (m, n) for I slices and for the three
// cabac_init_idc of P and B slices, the LPS ranges and the state transitions, with the
// state kept as x264 keeps it, (pStateIdx << 1) | valMPS.
#[allow(non_upper_case_globals)]
unsafe extern "C" {
    static x264_cabac_context_init_I: [[i8; 2]; 1024];
    static x264_cabac_context_init_PB: [[[i8; 2]; 1024]; 3];
    static x264_cabac_range_lps: [[u8; 4]; 64];
    static x264_cabac_transition: [[u8; 2]; 128];
}

pub type Result<T> = std::result::Result<T, String>;

const CONTEXTS: usize = 460;

/// What of the SPS and PPS the slice syntax depends on.
#[derive(Clone, Debug, Default)]
pub struct Params {
    pub mb_width: i32,
    pub mb_height: i32,
    log2_max_frame_num: u32,
    poc_type: u32,
    log2_max_poc_lsb: u32,
    pub direct_8x8_inference: bool,
    num_ref_idx_default: [u32; 2],
    weighted_pred: bool,
    weighted_bipred_idc: u32,
    pic_init_qp: i32,
    deblocking_control: bool,
    redundant_pic_cnt_present: bool,
    bottom_field_poc_present: bool,
    pub transform_8x8_mode: bool,
}

/// Emulation prevention bytes out of a NAL unit's payload (header byte included).
pub fn unescape(nal: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
}

struct Bits<'a> {
    data: &'a [u8],
    at: usize, // in bits
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], at: usize) -> Bits<'a> {
        Bits { data, at }
    }
    fn bit(&mut self) -> Result<u32> {
        let b = *self.data.get(self.at / 8).ok_or("the slice ends early")?;
        self.at += 1;
        Ok(((b >> (7 - (self.at - 1) % 8)) & 1) as u32)
    }
    fn bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0;
        for _ in 0..n {
            v = v << 1 | self.bit()?;
        }
        Ok(v)
    }
    fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err("an exp-Golomb code too long".into());
            }
        }
        Ok((1u64 << zeros) as u32 - 1 + self.bits(zeros)?)
    }
    fn se(&mut self) -> Result<i32> {
        let k = self.ue()? as i64;
        Ok(if k & 1 == 1 { ((k + 1) / 2) as i32 } else { -(k / 2) as i32 })
    }
}

impl Params {
    /// From the SPS and PPS, each without its start code or length but with its NAL
    /// header byte, escaped as in the stream.
    pub fn new(sps: &[u8], pps: &[u8]) -> Result<Params> {
        let mut raw = Vec::new();
        let mut p = Params::default();
        unescape(sps, &mut raw);
        let mut b = Bits::new(&raw, 8);
        let profile = b.bits(8)?;
        b.bits(16)?;
        b.ue()?; // sps id
        if [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].contains(&profile) {
            if b.ue()? != 1 {
                return Err("not 4:2:0".into());
            }
            if b.ue()? != 0 || b.ue()? != 0 {
                return Err("not 8-bit".into());
            }
            b.bit()?; // qpprime_y_zero_transform_bypass
            if b.bit()? != 0 {
                return Err("scaling matrices".into());
            }
        }
        p.log2_max_frame_num = b.ue()? + 4;
        p.poc_type = b.ue()?;
        match p.poc_type {
            0 => p.log2_max_poc_lsb = b.ue()? + 4,
            2 => {}
            _ => return Err("picture order count type 1".into()),
        }
        b.ue()?; // max_num_ref_frames
        b.bit()?;
        p.mb_width = b.ue()? as i32 + 1;
        p.mb_height = b.ue()? as i32 + 1;
        if b.bit()? != 1 {
            return Err("interlaced coding".into());
        }
        p.direct_8x8_inference = b.bit()? == 1;

        unescape(pps, &mut raw);
        let mut b = Bits::new(&raw, 8);
        b.ue()?; // pps id
        b.ue()?; // sps id
        if b.bit()? != 1 {
            return Err("CAVLC".into());
        }
        p.bottom_field_poc_present = b.bit()? == 1;
        if b.ue()? != 0 {
            return Err("slice groups".into());
        }
        p.num_ref_idx_default = [b.ue()? + 1, b.ue()? + 1];
        p.weighted_pred = b.bit()? == 1;
        p.weighted_bipred_idc = b.bits(2)?;
        p.pic_init_qp = 26 + b.se()?;
        b.se()?; // pic_init_qs
        b.se()?; // chroma_qp_index_offset
        p.deblocking_control = b.bit()? == 1;
        if b.bit()? != 0 {
            return Err("constrained intra prediction".into());
        }
        p.redundant_pic_cnt_present = b.bit()? == 1;
        // More of the PPS (High profile): the 8x8 transform flag.
        let left = raw.len() * 8 - b.at;
        if left > 8 || (left > 0 && raw[raw.len() - 1] & ((1 << left) - 1) as u8 != 1 << (left - 1)) {
            p.transform_8x8_mode = b.bit()? == 1;
            if b.bit()? != 0 {
                return Err("PPS scaling matrices".into());
            }
        }
        Ok(p)
    }
}

pub const SLICE_P: u8 = 0;
pub const SLICE_B: u8 = 1;
pub const SLICE_I: u8 = 2;

#[derive(Clone, Debug, Default)]
pub struct SliceHeader {
    pub first_mb: i32,
    pub kind: u8,
    pub direct_spatial: bool,
    pub num_ref_idx: [u32; 2],
    /// Whether the slice gives weights for its prediction (the encoder recoding over it
    /// must then predict with the same ones).
    pub weighted: bool,
    pub list_modified: bool,
    pub cabac_init_idc: u32,
    pub qp: i32,
    pub deblock: (u32, i32, i32),
    data_bit: usize,
}

fn read_header(p: &Params, nal: &[u8]) -> Result<SliceHeader> {
    let nal_ref_idc = (nal[0] >> 5) & 3;
    let idr = nal[0] & 31 == 5;
    let mut b = Bits::new(nal, 8);
    let mut h = SliceHeader { first_mb: b.ue()? as i32, ..Default::default() };
    h.kind = match b.ue()? % 5 {
        0 => SLICE_P,
        1 => SLICE_B,
        2 => SLICE_I,
        _ => return Err("SP or SI slice".into()),
    };
    b.ue()?; // pps id
    b.bits(p.log2_max_frame_num)?;
    if idr {
        b.ue()?;
    }
    if p.poc_type == 0 {
        b.bits(p.log2_max_poc_lsb)?;
        if p.bottom_field_poc_present {
            b.se()?;
        }
    }
    if p.redundant_pic_cnt_present {
        b.ue()?;
    }
    if h.kind == SLICE_B {
        h.direct_spatial = b.bit()? == 1;
    }
    h.num_ref_idx = p.num_ref_idx_default;
    if h.kind != SLICE_I && b.bit()? == 1 {
        h.num_ref_idx[0] = b.ue()? + 1;
        if h.kind == SLICE_B {
            h.num_ref_idx[1] = b.ue()? + 1;
        }
    }
    let lists = match h.kind {
        SLICE_P => 1,
        SLICE_B => 2,
        _ => 0,
    };
    for _ in 0..lists {
        if b.bit()? == 1 {
            h.list_modified = true;
            loop {
                match b.ue()? {
                    3 => break,
                    0..=2 => {
                        b.ue()?;
                    }
                    _ => return Err("a bad reference list modification".into()),
                }
            }
        }
    }
    if (p.weighted_pred && h.kind == SLICE_P) || (p.weighted_bipred_idc == 1 && h.kind == SLICE_B) {
        b.ue()?;
        b.ue()?;
        for l in 0..lists {
            for _ in 0..h.num_ref_idx[l] {
                if b.bit()? == 1 {
                    h.weighted = true;
                    b.se()?;
                    b.se()?;
                }
                if b.bit()? == 1 {
                    h.weighted = true;
                    for _ in 0..4 {
                        b.se()?;
                    }
                }
            }
        }
    }
    if nal_ref_idc != 0 {
        if idr {
            b.bits(2)?;
        } else if b.bit()? == 1 {
            loop {
                match b.ue()? {
                    0 => break,
                    1 | 2 => {
                        b.ue()?;
                    }
                    3 => {
                        b.ue()?;
                        b.ue()?;
                    }
                    4 | 6 => {
                        b.ue()?;
                    }
                    5 => {}
                    _ => return Err("a bad memory management operation".into()),
                }
            }
        }
    }
    if h.kind != SLICE_I {
        h.cabac_init_idc = b.ue()?;
        if h.cabac_init_idc > 2 {
            return Err("cabac_init_idc past 2".into());
        }
    }
    h.qp = p.pic_init_qp + b.se()?;
    if !(0..=51).contains(&h.qp) {
        return Err("slice qp out of range".into());
    }
    if p.deblocking_control {
        let idc = b.ue()?;
        h.deblock = (idc, 0, 0);
        if idc != 1 {
            h.deblock.1 = b.se()?;
            h.deblock.2 = b.se()?;
        }
    }
    // cabac_alignment_one_bit
    while b.at % 8 != 0 {
        if b.bit()? != 1 {
            return Err("cabac alignment bits not ones".into());
        }
    }
    h.data_bit = b.at;
    Ok(h)
}

struct Cabac<'a> {
    data: &'a [u8],
    at: usize, // next byte
    range: u32,
    offset: u32,
    bits_left: i32, // bits of `value` not yet shifted into offset
    value: u32,
    state: [u8; CONTEXTS],
}

impl<'a> Cabac<'a> {
    fn new(data: &'a [u8], kind: u8, cabac_init_idc: u32, qp: i32) -> Cabac<'a> {
        let mut state = [0u8; CONTEXTS];
        // SAFETY: read-only tables of x264, linked in.
        let init = unsafe { if kind == SLICE_I { &x264_cabac_context_init_I } else { &x264_cabac_context_init_PB[cabac_init_idc as usize] } };
        for (i, s) in state.iter_mut().enumerate() {
            let [m, n] = init[i];
            let v = (((m as i32 * qp) >> 4) + n as i32).clamp(1, 126);
            *s = ((v.min(127 - v) << 1) | (v >> 6)) as u8;
        }
        let mut c = Cabac { data, at: 0, range: 510, offset: 0, bits_left: 0, value: 0, state };
        for _ in 0..9 {
            c.offset = c.offset << 1 | c.read_bit();
        }
        c
    }

    #[inline]
    fn read_bit(&mut self) -> u32 {
        if self.bits_left == 0 {
            // Past the end reads zeros; a slice that needs them is caught by its length.
            self.value = *self.data.get(self.at).unwrap_or(&0) as u32;
            self.at += 1;
            self.bits_left = 8;
        }
        self.bits_left -= 1;
        (self.value >> self.bits_left) & 1
    }

    #[inline]
    fn decision(&mut self, ctx: usize) -> u32 {
        let s = self.state[ctx];
        // SAFETY: read-only tables of x264, linked in; indices in range by construction.
        let lps = unsafe { x264_cabac_range_lps[(s >> 1) as usize][((self.range >> 6) & 3) as usize] } as u32;
        self.range -= lps;
        let bin = if self.offset >= self.range {
            self.offset -= self.range;
            self.range = lps;
            (s & 1) as u32 ^ 1
        } else {
            (s & 1) as u32
        };
        self.state[ctx] = unsafe { x264_cabac_transition[s as usize][bin as usize] };
        while self.range < 256 {
            self.range <<= 1;
            self.offset = self.offset << 1 | self.read_bit();
        }
        bin
    }

    #[inline]
    fn bypass(&mut self) -> u32 {
        self.offset = self.offset << 1 | self.read_bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    fn terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = self.offset << 1 | self.read_bit();
            }
            0
        }
    }

    /// Bytes read so far, at the end of a slice: all of them, less the stop bit's byte.
    fn bytes_read(&self) -> usize {
        self.at
    }
}

/// Macroblock kinds as far as reading the syntax (and copying it) goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    I4x4,
    I8x8,
    /// mb_type 1..=24 of an I slice's table: prediction mode, coded block pattern.
    I16x16,
    /// P_L0 partitions: 16x16, 16x8, 8x16, or 8x8 with sub types.
    P16x16,
    P16x8,
    P8x16,
    P8x8,
    PSkip,
    /// B_Direct_16x16.
    BDirect,
    BSkip,
    /// The B types 1..=21 by Table 7-14 number, and 22 (B_8x8) with sub types.
    B(u8),
}

impl Kind {
    pub fn intra(self) -> bool {
        matches!(self, Kind::I4x4 | Kind::I8x8 | Kind::I16x16)
    }
    fn skip(self) -> bool {
        matches!(self, Kind::PSkip | Kind::BSkip)
    }
}

/// One macroblock as coded.
#[derive(Clone, Debug)]
pub struct Mb {
    pub addr: i32,
    pub kind: Kind,
    /// I16x16: its prediction mode; I4x4 and I8x8: modes per 4x4 block in
    /// luma4x4BlkIdx order (an 8x8 block's mode in each of its four).
    pub i16_mode: u8,
    pub intra_modes: [u8; 16],
    pub chroma_mode: u8,
    /// P8x8 and B8x8 (B(22)): sub_mb_type of each 8x8 block.
    pub sub: [u8; 4],
    pub ref_idx: [[i8; 4]; 2],
    pub t8x8: bool,
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub qp: u8,
    /// Levels in scan order (as coded): 4x4 luma blocks in luma4x4BlkIdx order (for
    /// I16x16 the AC, position 0 unused), 8x8 luma blocks, the I16x16 DC, chroma DC (Cb,
    /// Cr) and chroma AC (Cb 0..3, Cr 0..3; position 0 unused).
    pub luma4x4: [[i16; 16]; 16],
    pub luma8x8: [[i16; 64]; 4],
    pub luma_dc: [i16; 16],
    pub chroma_dc: [[i16; 4]; 2],
    pub chroma_ac: [[i16; 16]; 8],
}

impl Default for Mb {
    fn default() -> Mb {
        Mb {
            addr: 0,
            kind: Kind::default(),
            i16_mode: 0,
            intra_modes: [0; 16],
            chroma_mode: 0,
            sub: [0; 4],
            ref_idx: [[-1; 4]; 2],
            t8x8: false,
            cbp_luma: 0,
            cbp_chroma: 0,
            qp: 0,
            luma4x4: [[0; 16]; 16],
            luma8x8: [[0; 64]; 4],
            luma_dc: [0; 16],
            chroma_dc: [[0; 4]; 2],
            chroma_ac: [[0; 16]; 8],
        }
    }
}

/// What the neighbours' syntax holds for the context of the next.
#[derive(Clone, Copy, Default)]
struct Near {
    slice: u32, // 0: none yet
    kind: Kind,
    intra_modes: [u8; 16], // raster 4x4
    chroma_mode: u8,
    t8x8: bool,
    cbp_luma: u8,
    cbp_chroma: u8,
    mvd: [[[u8; 2]; 16]; 2], // list, raster 4x4, component (absolute, capped)
    ref_idx: [[i8; 4]; 2],   // raster 8x8, -1 unused
    direct8: [bool; 4],      // raster 8x8: coded in direct mode
    cbf: [bool; 16],         // raster 4x4 luma
    cbf_dc: bool,
    cbf_cdc: [bool; 2],
    cbf_cac: [[bool; 4]; 2], // component, raster 2x2
}

/// Reads slices, keeping each macroblock's syntax for the context of later ones.
pub struct Reader {
    pub p: Params,
    near: Vec<Near>,
    slices: u32,
    raw: Vec<u8>,
}

// luma4x4BlkIdx to raster 4x4 index and back.
const BLK_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

// Significance and last contexts of 8x8 blocks in frame coding (Table 9-43).
const SIG_8X8: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7, 7, 6, 11, 12, 13, 11, 6,
    7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13, 11, 14, 10, 12,
];
const LAST_8X8: [u8; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

/// Partitions of the B types 1..21 (Table 7-14): shape (0 16x16, 1 16x8, 2 8x16) and
/// each partition's lists (bit 0 L0, bit 1 L1).
fn b_type(t: u8) -> (u8, [u8; 2]) {
    const L0: u8 = 1;
    const L1: u8 = 2;
    const BI: u8 = 3;
    match t {
        1 => (0, [L0, 0]),
        2 => (0, [L1, 0]),
        3 => (0, [BI, 0]),
        _ => {
            let pairs = [[L0, L0], [L1, L1], [L0, L1], [L1, L0], [L0, BI], [L1, BI], [BI, L0], [BI, L1], [BI, BI]];
            (1 + (t - 4) % 2, pairs[((t - 4) / 2) as usize])
        }
    }
}

/// B sub_mb_type: lists used and whether its partitions are 8x8.
fn b_sub(t: u8) -> (u8, bool) {
    match t {
        0 => (0, true), // direct
        1 => (1, true),
        2 => (2, true),
        3 => (3, true),
        4 | 5 | 10 => (1, false),
        6 | 7 | 11 => (2, false),
        _ => (3, false),
    }
}

impl Reader {
    pub fn new(p: Params) -> Reader {
        let n = (p.mb_width * p.mb_height) as usize;
        Reader { p, near: vec![Near::default(); n], slices: 0, raw: Vec::new() }
    }

    /// Reads one slice NAL (header byte first, escaped as in the stream) into `out`, one
    /// entry per macroblock in decoding order, and returns its header.
    pub fn slice(&mut self, nal: &[u8], out: &mut Vec<Mb>) -> Result<SliceHeader> {
        let mut raw = std::mem::take(&mut self.raw);
        unescape(nal, &mut raw);
        let r = self.slice_raw(&raw, out);
        self.raw = raw;
        r
    }

    fn slice_raw(&mut self, raw: &[u8], out: &mut Vec<Mb>) -> Result<SliceHeader> {
        out.clear();
        let h = read_header(&self.p, raw)?;
        self.slices = self.slices.wrapping_add(1).max(1);
        let slice = self.slices;
        let data = &raw[h.data_bit / 8..];
        let mut c = Cabac::new(data, h.kind, h.cabac_init_idc, h.qp);
        let mbs = self.p.mb_width * self.p.mb_height;
        let mut addr = h.first_mb;
        let mut qp = h.qp;
        let mut last_dqp_ctx = false; // the previous macroblock's mb_qp_delta was non-zero (and present)
        loop {
            if addr >= mbs {
                return Err("a slice past the frame's last macroblock".into());
            }
            let mut mb = Mb { addr, ..Default::default() };
            let skipped = h.kind != SLICE_I && {
                let ctx = (if h.kind == SLICE_P { 11 } else { 24 })
                    + [self.left(addr, slice), self.top(addr, slice)].iter().filter(|n| n.is_some_and(|n| !n.kind.skip())).count();
                c.decision(ctx) == 1
            };
            let mut near = Near { slice, ..Default::default() };
            if skipped {
                mb.kind = if h.kind == SLICE_P { Kind::PSkip } else { Kind::BSkip };
                near.kind = mb.kind;
                if h.kind == SLICE_P {
                    near.ref_idx = [[0; 4], [-1; 4]];
                } else {
                    near.direct8 = [true; 4];
                }
                last_dqp_ctx = false;
            } else {
                let dqp_ctx = last_dqp_ctx;
                last_dqp_ctx = self.macroblock(&mut c, &h, slice, addr, &mut mb, &mut near, &mut qp, dqp_ctx)?;
            }
            mb.qp = qp as u8;
            near.kind = mb.kind;
            self.near[addr as usize] = near;
            out.push(mb);
            if c.terminate() == 1 {
                break;
            }
            addr += 1;
        }
        // What is left is the encoder's flush and rbsp trailing bits, a byte or two.
        if c.bytes_read() > data.len() || c.bytes_read() + 3 < data.len() {
            return Err(format!("a slice read to byte {} of {}", c.bytes_read(), data.len()));
        }
        Ok(h)
    }

    fn left(&self, addr: i32, slice: u32) -> Option<&Near> {
        (addr % self.p.mb_width != 0).then(|| &self.near[addr as usize - 1]).filter(|n| n.slice == slice)
    }

    fn top(&self, addr: i32, slice: u32) -> Option<&Near> {
        (addr >= self.p.mb_width).then(|| &self.near[(addr - self.p.mb_width) as usize]).filter(|n| n.slice == slice)
    }

    /// The left and top neighbours of raster 4x4 block (x, y): the macroblock state they
    /// are in (None if unavailable) and their raster 4x4 index there.
    fn near4<'a>(&'a self, cur: &'a Near, addr: i32, slice: u32, x: usize, y: usize) -> [(Option<&'a Near>, usize); 2] {
        let a = if x > 0 { (Some(cur), y * 4 + x - 1) } else { (self.left(addr, slice), y * 4 + 3) };
        let b = if y > 0 { (Some(cur), (y - 1) * 4 + x) } else { (self.top(addr, slice), 12 + x) };
        [a, b]
    }

    #[allow(clippy::too_many_arguments)]
    fn macroblock(&mut self, c: &mut Cabac, h: &SliceHeader, slice: u32, addr: i32, mb: &mut Mb, near: &mut Near, qp: &mut i32,
                  dqp_ctx: bool) -> Result<bool> {
        let (left, top) = (self.left(addr, slice).copied(), self.top(addr, slice).copied());
        // mb_type
        let mut intra_type: Option<u8> = None; // 0 I_NxN, 1..24 I16x16
        match h.kind {
            SLICE_I => {
                let ctx = 3 + [left, top].iter().filter(|n| n.is_some_and(|n| n.kind == Kind::I16x16)).count();
                intra_type = Some(self.intra_type(c, ctx, true)?);
            }
            SLICE_P => {
                if c.decision(14) == 0 {
                    mb.kind = if c.decision(15) == 0 {
                        if c.decision(16) == 0 { Kind::P16x16 } else { Kind::P8x8 }
                    } else if c.decision(17) == 0 {
                        Kind::P8x16
                    } else {
                        Kind::P16x8
                    };
                } else {
                    intra_type = Some(self.intra_type(c, 17, false)?);
                }
            }
            _ => {
                let ctx = 27 + [left, top].iter().filter(|n| n.is_some_and(|n| !matches!(n.kind, Kind::BSkip | Kind::BDirect))).count();
                if c.decision(ctx) == 0 {
                    mb.kind = Kind::BDirect;
                } else if c.decision(27 + 3) == 0 {
                    mb.kind = Kind::B(1 + c.decision(27 + 5) as u8);
                } else {
                    let mut bits = c.decision(27 + 4) << 3;
                    bits |= c.decision(27 + 5) << 2;
                    bits |= c.decision(27 + 5) << 1;
                    bits |= c.decision(27 + 5);
                    mb.kind = match bits {
                        0..=7 => Kind::B(bits as u8 + 3),
                        13 => {
                            intra_type = Some(self.intra_type(c, 32, false)?);
                            Kind::I4x4
                        }
                        14 => Kind::B(11),
                        15 => Kind::B(22),
                        _ => Kind::B((((bits << 1) | c.decision(27 + 5)) - 4) as u8),
                    };
                }
            }
        }
        let mut cbp_luma;
        let mut cbp_chroma;
        let mut dqp_needed;
        if let Some(t) = intra_type {
            if t == 0 {
                mb.t8x8 = self.p.transform_8x8_mode && {
                    let ctx = 399 + [left, top].iter().filter(|n| n.is_some_and(|n| n.t8x8)).count();
                    c.decision(ctx) == 1
                };
                mb.kind = if mb.t8x8 { Kind::I8x8 } else { Kind::I4x4 };
                near.kind = mb.kind;
                near.t8x8 = mb.t8x8;
                // Prediction modes, block by block, each from its neighbours'.
                let step = if mb.t8x8 { 4 } else { 1 };
                for blk in (0..16).step_by(step) {
                    let r = BLK_RASTER[blk];
                    let (x, y) = (r % 4, r / 4);
                    let pred = {
                        let ns = self.near4(near, addr, slice, x, y);
                        let mode = |n: (Option<&Near>, usize)| -> Option<u8> {
                            let s = n.0?;
                            Some(if matches!(s.kind, Kind::I4x4 | Kind::I8x8) { s.intra_modes[n.1] } else { 2 })
                        };
                        match (mode(ns[0]), mode(ns[1])) {
                            (Some(a), Some(b)) => a.min(b),
                            _ => 2,
                        }
                    };
                    let m = if c.decision(68) == 1 {
                        pred
                    } else {
                        let rem = (c.decision(69) | c.decision(69) << 1 | c.decision(69) << 2) as u8;
                        if rem < pred { rem } else { rem + 1 }
                    };
                    for k in 0..step {
                        let rr = BLK_RASTER[blk + k];
                        near.intra_modes[rr] = m;
                        mb.intra_modes[blk + k] = m;
                    }
                }
                cbp_luma = 0;
                cbp_chroma = 0;
                dqp_needed = false;
            } else {
                mb.kind = Kind::I16x16;
                near.kind = mb.kind;
                mb.i16_mode = (t - 1) % 4;
                cbp_chroma = ((t - 1) / 4) % 3;
                cbp_luma = if t >= 13 { 15 } else { 0 };
                dqp_needed = true;
            }
            // intra_chroma_pred_mode
            let cond = |n: Option<Near>| n.is_some_and(|n| n.kind.intra() && n.chroma_mode != 0);
            let ctx = 64 + cond(left) as usize + cond(top) as usize;
            let mut m = 0;
            if c.decision(ctx) == 1 {
                m = 1;
                if c.decision(67) == 1 {
                    m = 2 + c.decision(67) as u8;
                }
            }
            mb.chroma_mode = m;
            near.chroma_mode = m;
            near.ref_idx = [[-1; 4]; 2];
        } else {
            near.kind = mb.kind;
            dqp_needed = false;
            cbp_luma = 0;
            cbp_chroma = 0;
            self.inter_pred(c, h, slice, addr, mb, near)?;
        }
        let mut sub_small = false;
        if matches!(mb.kind, Kind::P8x8) {
            sub_small = mb.sub.iter().any(|&s| s != 0);
        }
        if mb.kind == Kind::B(22) {
            sub_small = mb.sub.iter().any(|&s| !b_sub(s).1 || (s == 0 && !self.p.direct_8x8_inference));
        }
        if mb.kind != Kind::I16x16 {
            // coded_block_pattern: luma, 8x8 block by block, then chroma.
            let bit = |n: Option<Near>, b8: usize| -> bool {
                // condTermFlagN: 0 if unavailable or the bit set (and not skip).
                match n {
                    None => false,
                    Some(n) => n.kind.skip() || (n.cbp_luma >> b8) & 1 == 0,
                }
            };
            let mut cbp = 0u8;
            for b8 in 0..4usize {
                let (bx, by) = (b8 & 1, b8 >> 1);
                let a = if bx > 0 { (cbp >> (b8 - 1)) & 1 == 0 } else { bit(left, b8 + 1) };
                let b = if by > 0 { (cbp >> (b8 - 2)) & 1 == 0 } else { bit(top, b8 + 2) };
                cbp |= (c.decision(73 + a as usize + 2 * b as usize) as u8) << b8;
            }
            cbp_luma = cbp;
            let ca = |n: Option<Near>, two: bool| n.is_some_and(|n| !n.kind.skip() && if two { n.cbp_chroma == 2 } else { n.cbp_chroma != 0 });
            cbp_chroma = 0;
            if c.decision(77 + ca(left, false) as usize + 2 * ca(top, false) as usize) == 1 {
                cbp_chroma = 1 + c.decision(77 + 4 + ca(left, true) as usize + 2 * ca(top, true) as usize) as u8;
            }
            if cbp_luma != 0 && self.p.transform_8x8_mode && !mb.kind.intra() && !sub_small
                && (mb.kind != Kind::BDirect || self.p.direct_8x8_inference)
            {
                let ctx = 399 + [left, top].iter().filter(|n| n.is_some_and(|n| n.t8x8)).count();
                mb.t8x8 = c.decision(ctx) == 1;
                near.t8x8 = mb.t8x8;
            }
            dqp_needed = cbp_luma != 0 || cbp_chroma != 0;
        }
        mb.cbp_luma = cbp_luma;
        mb.cbp_chroma = cbp_chroma;
        near.cbp_luma = cbp_luma;
        near.cbp_chroma = cbp_chroma;
        let mut dqp_nonzero = false;
        if dqp_needed {
            let mut ctx = 60 + dqp_ctx as usize;
            let mut k = 0u32;
            while c.decision(ctx) == 1 {
                k += 1;
                ctx = if k == 1 { 62 } else { 63 };
                if k > 102 {
                    return Err("mb_qp_delta out of range".into());
                }
            }
            let delta = if k & 1 == 1 { (k as i32 + 1) / 2 } else { -(k as i32 / 2) };
            dqp_nonzero = delta != 0;
            *qp = (*qp + delta + 52) % 52;
            self.residual(c, slice, addr, mb, near)?;
        }
        Ok(dqp_nonzero)
    }

    /// The intra mb_type: 0 I_NxN, 1..24 I_16x16; PCM refused. `ctx` is the first bin's
    /// context: 3..5 in an I slice, 17 or 32 as the suffix of a P or B mb_type.
    fn intra_type(&self, c: &mut Cabac, ctx: usize, intra_slice: bool) -> Result<u8> {
        if c.decision(ctx) == 0 {
            return Ok(0);
        }
        if c.terminate() == 1 {
            return Err("I_PCM".into());
        }
        // The rest of the bins' contexts (Table 9-39): luma, chroma, chroma == 2, then the
        // two bits of the prediction mode.
        let [luma, chroma, chroma2, pred1, pred0] = if intra_slice { [6, 7, 8, 9, 10] } else { [ctx + 1, ctx + 2, ctx + 2, ctx + 3, ctx + 3] };
        let mut t = 1u8;
        t += 12 * c.decision(luma) as u8;
        if c.decision(chroma) == 1 {
            t += 4 + 4 * c.decision(chroma2) as u8;
        }
        t += 2 * c.decision(pred1) as u8;
        t += c.decision(pred0) as u8;
        Ok(t)
    }

    fn ref_idx(&self, c: &mut Cabac, slice: u32, addr: i32, near: &Near, list: usize, b8: usize) -> Result<i8> {
        let (bx, by) = (b8 & 1, b8 >> 1);
        let cond = |n: Option<&Near>, i: usize| -> bool {
            n.is_some_and(|n| !n.kind.skip() && !n.kind.intra() && !n.direct8[i] && n.ref_idx[list][i] > 0)
        };
        let a = if bx > 0 { cond(Some(near), b8 - 1) } else { cond(self.left(addr, slice), b8 + 1) };
        let b = if by > 0 { cond(Some(near), b8 - 2) } else { cond(self.top(addr, slice), b8 + 2) };
        let mut ctx = 54 + a as usize + 2 * b as usize;
        let mut r = 0i8;
        while c.decision(ctx) == 1 {
            r += 1;
            ctx = if r == 1 { 54 + 4 } else { 54 + 5 };
            if r > 31 {
                return Err("ref_idx out of range".into());
            }
        }
        Ok(r)
    }

    /// mvd of the partition whose top-left raster 4x4 block is (x, y), w x h blocks; its
    /// absolute value is kept in `near` for the context of later ones.
    #[allow(clippy::too_many_arguments)]
    fn mvd(&self, c: &mut Cabac, slice: u32, addr: i32, near: &mut Near, list: usize, x: usize, y: usize, w: usize, h: usize) -> Result<()> {
        let mut abs = [0u8; 2];
        for comp in 0..2 {
            let sum = {
                let ns = self.near4(near, addr, slice, x, y);
                ns.iter().map(|&(n, i)| n.map_or(0, |n| if n.kind.intra() { 0 } else { n.mvd[list][i][comp] as u32 })).sum::<u32>()
            };
            let base = if comp == 0 { 40 } else { 47 };
            let inc = if sum < 3 { 0 } else if sum <= 32 { 1 } else { 2 };
            let mut v: u32 = 0;
            if c.decision(base + inc) == 1 {
                v = 1;
                const CTX: [usize; 8] = [3, 4, 5, 6, 6, 6, 6, 6];
                while v < 9 && c.decision(base + CTX[v as usize - 1]) == 1 {
                    v += 1;
                }
                if v >= 9 {
                    // Exp-Golomb, k = 3, bypass.
                    let mut k = 3;
                    while c.bypass() == 1 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err("an mvd too large".into());
                        }
                    }
                    for i in (0..k).rev() {
                        v += c.bypass() << i;
                    }
                }
                c.bypass(); // sign
            }
            abs[comp] = v.min(127) as u8;
        }
        for yy in y..y + h {
            for xx in x..x + w {
                near.mvd[list][yy * 4 + xx] = abs;
            }
        }
        Ok(())
    }

    fn inter_pred(&mut self, c: &mut Cabac, h: &SliceHeader, slice: u32, addr: i32, mb: &mut Mb, near: &mut Near) -> Result<()> {
        near.ref_idx = [[-1; 4]; 2];
        let refs = h.num_ref_idx;
        // Each partition: top-left raster 4x4 (x, y), size in 4x4 blocks, lists used.
        let mut parts: Vec<(usize, usize, usize, usize, u8)> = Vec::new();
        match mb.kind {
            Kind::BDirect => {
                near.direct8 = [true; 4];
                mb.ref_idx = [[-1; 4]; 2];
                return Ok(());
            }
            Kind::P16x16 => parts.push((0, 0, 4, 4, 1)),
            Kind::P16x8 => parts.extend([(0, 0, 4, 2, 1), (0, 2, 4, 2, 1)]),
            Kind::P8x16 => parts.extend([(0, 0, 2, 4, 1), (2, 0, 2, 4, 1)]),
            Kind::B(t) if t < 22 => {
                let (shape, lists) = b_type(t);
                match shape {
                    0 => parts.push((0, 0, 4, 4, lists[0])),
                    1 => parts.extend([(0, 0, 4, 2, lists[0]), (0, 2, 4, 2, lists[1])]),
                    _ => parts.extend([(0, 0, 2, 4, lists[0]), (2, 0, 2, 4, lists[1])]),
                }
            }
            Kind::P8x8 | Kind::B(22) => return self.sub_pred(c, h, slice, addr, mb, near),
            _ => return Err("an unexpected inter type".into()),
        }
        for list in 0..2 {
            for &(x, y, w, hh, lists) in &parts {
                if lists >> list & 1 == 0 {
                    continue;
                }
                let r = if refs[list] > 1 { self.ref_idx(c, slice, addr, near, list, (y / 2) * 2 + x / 2)? } else { 0 };
                for by in y / 2..(y + hh) / 2 {
                    for bx in x / 2..(x + w) / 2 {
                        near.ref_idx[list][by * 2 + bx] = r;
                    }
                }
            }
        }
        for list in 0..2 {
            for &(x, y, w, hh, lists) in &parts {
                if lists >> list & 1 != 0 {
                    self.mvd(c, slice, addr, near, list, x, y, w, hh)?;
                }
            }
        }
        mb.ref_idx = near.ref_idx;
        Ok(())
    }

    fn sub_pred(&mut self, c: &mut Cabac, h: &SliceHeader, slice: u32, addr: i32, mb: &mut Mb, near: &mut Near) -> Result<()> {
        let b = mb.kind == Kind::B(22);
        for i in 0..4 {
            mb.sub[i] = if !b {
                if c.decision(21) == 1 {
                    0
                } else if c.decision(22) == 0 {
                    1
                } else if c.decision(23) == 1 {
                    2
                } else {
                    3
                }
            } else if c.decision(36) == 0 {
                0
            } else if c.decision(37) == 0 {
                1 + c.decision(39) as u8
            } else {
                let mut t = 3u8;
                if c.decision(38) == 1 {
                    if c.decision(39) == 1 {
                        t = 11 + c.decision(39) as u8;
                        mb.sub[i] = t;
                        continue;
                    }
                    t += 4;
                }
                t += 2 * c.decision(39) as u8;
                t += c.decision(39) as u8;
                t
            };
        }
        let lists_of = |s: u8| if b { b_sub(s).0 } else { 1 };
        if b {
            for i in 0..4 {
                near.direct8[i] = mb.sub[i] == 0;
            }
        }
        for list in 0..2 {
            for i in 0..4 {
                if lists_of(mb.sub[i]) >> list & 1 == 0 {
                    continue;
                }
                let r = if h.num_ref_idx[list] > 1 { self.ref_idx(c, slice, addr, near, list, i)? } else { 0 };
                near.ref_idx[list][i] = r;
            }
        }
        for list in 0..2 {
            for i in 0..4 {
                let s = mb.sub[i];
                if lists_of(s) >> list & 1 == 0 {
                    continue;
                }
                let (x0, y0) = ((i & 1) * 2, (i >> 1) * 2);
                // Sub-partition shape: 8x8, 8x4, 4x8 or 4x4.
                let shape = if !b {
                    s
                } else {
                    match s {
                        1..=3 => 0,
                        4 | 6 | 8 => 1,
                        5 | 7 | 9 => 2,
                        _ => 3,
                    }
                };
                match shape {
                    0 => self.mvd(c, slice, addr, near, list, x0, y0, 2, 2)?,
                    1 => {
                        self.mvd(c, slice, addr, near, list, x0, y0, 2, 1)?;
                        self.mvd(c, slice, addr, near, list, x0, y0 + 1, 2, 1)?;
                    }
                    2 => {
                        self.mvd(c, slice, addr, near, list, x0, y0, 1, 2)?;
                        self.mvd(c, slice, addr, near, list, x0 + 1, y0, 1, 2)?;
                    }
                    _ => {
                        for k in 0..4 {
                            self.mvd(c, slice, addr, near, list, x0 + (k & 1), y0 + (k >> 1), 1, 1)?;
                        }
                    }
                }
            }
        }
        mb.ref_idx = near.ref_idx;
        Ok(())
    }

    /// coded_block_flag's context increment from the two neighbouring blocks' flags:
    /// `flag(n, i)` reads block i of macroblock n, None where that block is not coded.
    fn cbf_inc(n: [(Option<&Near>, usize); 2], intra: bool, flag: impl Fn(&Near, usize) -> Option<bool>) -> usize {
        let mut inc = 0;
        for (k, &(m, i)) in n.iter().enumerate() {
            let cond = match m {
                None => intra,
                Some(m) => flag(m, i).unwrap_or(false),
            };
            inc += (cond as usize) << k;
        }
        inc
    }

    /// One block's levels, in scan order from `start`: cat as in Table 9-42 (0 luma DC,
    /// 1 luma AC, 2 luma 4x4, 3 chroma DC, 4 chroma AC, 5 luma 8x8).
    fn block(c: &mut Cabac, cat: usize, start: usize, out: &mut [i16]) -> Result<()> {
        let n = out.len() - start;
        let (sig_base, last_base, abs_base) = match cat {
            5 => (402, 417, 426),
            _ => {
                let o = [0, 15, 29, 44, 47][cat];
                (105 + o, 166 + o, 227 + [0, 10, 20, 30, 39][cat])
            }
        };
        let mut sig = [0usize; 64];
        let mut count = 0;
        let mut i = 0;
        while i < n - 1 {
            let (s, l) = match cat {
                5 => (sig_base + SIG_8X8[i] as usize, last_base + LAST_8X8[i] as usize),
                3 => (sig_base + i.min(2), last_base + i.min(2)),
                _ => (sig_base + i, last_base + i),
            };
            if c.decision(s) == 1 {
                sig[count] = i;
                count += 1;
                if c.decision(l) == 1 {
                    break;
                }
            }
            i += 1;
        }
        if i == n - 1 {
            sig[count] = n - 1;
            count += 1;
        }
        let (mut eq1, mut gt1) = (0usize, 0usize);
        for k in (0..count).rev() {
            let ctx = abs_base + if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
            let mut v: u32 = 1;
            if c.decision(ctx) == 1 {
                let ctx = abs_base + 5 + gt1.min(4 - (cat == 3) as usize);
                v = 2;
                while v < 15 && c.decision(ctx) == 1 {
                    v += 1;
                }
                if v == 15 {
                    let mut k = 0;
                    while c.bypass() == 1 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err("a level too large".into());
                        }
                    }
                    for b in (0..k).rev() {
                        v += c.bypass() << b;
                    }
                }
            }
            if v == 1 {
                eq1 += 1;
            } else {
                gt1 += 1;
            }
            let level = if c.bypass() == 1 { -(v as i32) } else { v as i32 };
            out[start + sig[k]] = level.clamp(-32768, 32767) as i16;
        }
        Ok(())
    }

    fn residual(&mut self, c: &mut Cabac, slice: u32, addr: i32, mb: &mut Mb, near: &mut Near) -> Result<()> {
        let intra = mb.kind.intra();
        let (left, top) = (self.left(addr, slice).copied(), self.top(addr, slice).copied());
        if mb.kind == Kind::I16x16 {
            let ctx = 85 + Reader::cbf_inc([(left.as_ref(), 0), (top.as_ref(), 0)], true, |n, _| Some(n.kind == Kind::I16x16 && n.cbf_dc));
            if c.decision(ctx) == 1 {
                near.cbf_dc = true;
                Reader::block(c, 0, 0, &mut mb.luma_dc)?;
            }
        }
        // Luma: 4x4 blocks' flags from their neighbours'; an 8x8 transform's block is
        // always coded when its bit of the pattern is set.
        let luma_flag = |n: &Near, i: usize| -> Option<bool> {
            if n.kind.skip() || (n.cbp_luma >> ((i / 8) * 2 + (i % 4) / 2)) & 1 == 0 {
                return None;
            }
            Some(n.cbf[i])
        };
        for b8 in 0..4 {
            if (mb.cbp_luma >> b8) & 1 == 0 {
                continue;
            }
            if mb.t8x8 {
                Reader::block(c, 5, 0, &mut mb.luma8x8[b8])?;
                for k in 0..4 {
                    near.cbf[BLK_RASTER[b8 * 4 + k]] = true;
                }
                continue;
            }
            for b4 in 0..4 {
                let blk = b8 * 4 + b4;
                let r = BLK_RASTER[blk];
                let ns = self.near4(near, addr, slice, r % 4, r / 4);
                let ctx_cat = if mb.kind == Kind::I16x16 { 1 } else { 2 };
                let ctx = 85 + [0, 4, 8][ctx_cat] + Reader::cbf_inc(ns, intra, luma_flag);
                if c.decision(ctx) == 1 {
                    near.cbf[r] = true;
                    if ctx_cat == 1 {
                        Reader::block(c, 1, 1, &mut mb.luma4x4[blk])?;
                    } else {
                        Reader::block(c, 2, 0, &mut mb.luma4x4[blk])?;
                    }
                }
            }
        }
        if mb.cbp_chroma != 0 {
            for comp in 0..2 {
                let flag = |n: &Near, _: usize| -> Option<bool> { (!n.kind.skip() && n.cbp_chroma != 0).then_some(n.cbf_cdc[comp]) };
                let ctx = 85 + 12 + Reader::cbf_inc([(left.as_ref(), 0), (top.as_ref(), 0)], intra, flag);
                if c.decision(ctx) == 1 {
                    near.cbf_cdc[comp] = true;
                    Reader::block(c, 3, 0, &mut mb.chroma_dc[comp])?;
                }
            }
        }
        if mb.cbp_chroma == 2 {
            for comp in 0..2 {
                for blk in 0..4 {
                    let (x, y) = (blk & 1, blk >> 1);
                    let cur = *near;
                    let a = if x > 0 { (Some(&cur), y * 2) } else { (left.as_ref(), y * 2 + 1) };
                    let b = if y > 0 { (Some(&cur), x) } else { (top.as_ref(), 2 + x) };
                    let flag = |n: &Near, i: usize| -> Option<bool> { (!n.kind.skip() && n.cbp_chroma == 2).then_some(n.cbf_cac[comp][i]) };
                    let ctx = 85 + 16 + Reader::cbf_inc([a, b], intra, flag);
                    if c.decision(ctx) == 1 {
                        near.cbf_cac[comp][blk] = true;
                        Reader::block(c, 4, 1, &mut mb.chroma_ac[comp * 4 + blk])?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The macroblock as x264 is to write it again: `side` is what the decoder found of the
/// frame it is in (its vectors). None for what cannot be written as it was: partitions
/// smaller than the whole macroblock, or vectors that differ within an 8x8 block.
pub fn copy_of(mb: &Mb, side: &crate::decode::Side) -> Option<crate::x264::CopyMb> {
    use crate::x264::*;
    let kind = match mb.kind {
        Kind::I4x4 => CopyMb::I4X4,
        Kind::I8x8 => CopyMb::I8X8,
        Kind::I16x16 => CopyMb::I16X16,
        Kind::P16x16 => CopyMb::P16X16,
        Kind::PSkip => CopyMb::P_SKIP,
        Kind::BDirect => CopyMb::B_DIRECT,
        Kind::BSkip => CopyMb::B_SKIP,
        Kind::B(1..=3) => CopyMb::B16X16,
        _ => return None,
    };
    let mut c = CopyMb {
        kind,
        t8x8: mb.t8x8 as u8,
        qp: mb.qp,
        cbp_luma: mb.cbp_luma,
        cbp_chroma: mb.cbp_chroma,
        i16_mode: mb.i16_mode,
        chroma_mode: mb.chroma_mode,
        intra_modes: mb.intra_modes,
        ref_idx: [[-1; 4]; 2],
        mv: [[[0; 2]; 4]; 2],
        luma: [0; 256],
        luma_dc: mb.luma_dc,
        chroma_dc: mb.chroma_dc,
        chroma_ac: mb.chroma_ac,
    };
    if !mb.kind.intra() {
        let a = mb.addr as usize;
        for b in 0..4 {
            for l in 0..2 {
                let k = (a * 4 + b) * 2 + l;
                match *side.mv_use.get(k)? {
                    0 => {}
                    1 => {
                        c.ref_idx[l][b] = 0;
                        c.mv[l][b] = [side.mv[k * 2], side.mv[k * 2 + 1]];
                    }
                    _ => return None,
                }
            }
        }
        if c.ref_idx.iter().flatten().all(|&r| r < 0) {
            return None;
        }
    }
    if mb.t8x8 && !matches!(mb.kind, Kind::I4x4 | Kind::I16x16) {
        for (b, block) in mb.luma8x8.iter().enumerate() {
            c.luma[b * 64..b * 64 + 64].copy_from_slice(block);
        }
    } else {
        for (b, block) in mb.luma4x4.iter().enumerate() {
            c.luma[b * 16..b * 16 + 16].copy_from_slice(block);
        }
    }
    Some(c)
}
