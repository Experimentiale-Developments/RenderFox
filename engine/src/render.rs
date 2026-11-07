//! The frame. Every quad in view becomes a prim with its plane equations over pixel
//! coordinates, rasterised into a visibility buffer (nearest prim per pixel), then each
//! pixel is shaded once. A full static render happens every few frames; the frames in
//! between warp the last one to the new camera. Players are drawn over it every frame.
//! Then bloom, temporal antialiasing, tonemap and grade, and dithered sRGB, all at the
//! output resolution.
//!
//! `fmax`/`fmin` behave like C++'s `std::max`/`std::min`, which differ from `f32::max`
//! on NaN.

use crate::assets::{Assets, CUTOUT, MAT_EMISSIVE, MAT_ENTITY, MAT_LAVA, MAT_LEAVES, MAT_NO_SHADOW, MAT_PLANT, MAT_WATER,
                    SOLID, TRANSLUCENT, Texture, load_png_rgba, lut};
use crate::jobs::Jobs;
use crate::mesh::{QUAD_FLAT, QUAD_TRI, Quad, SectionMesh};
use crate::parts::Parts;
use crate::sys;
use crate::vmath::*;
use crate::wide::{F8, I8};
use std::cell::RefCell;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const NO_PRIM: u32 = 0xffff_ffff;
const NEAR: f32 = 0.08;
const FAR: f32 = 1200.0;
// Rasterised in full-width bands of rows; each worker keeps its own prims and its own
// per-band index lists, so a band walks memory forwards.
const BAND: i32 = 16;
const HIZ: i32 = 8;
const SKY_UP: usize = 128;
const SKY_SUN: usize = 256;

fn since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn halton(mut i: i32, b: i32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

#[inline]
fn luminance(c: Vec3) -> f32 {
    c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722
}

// std::clamp: no panic when the bounds cross, unlike the std methods.
#[inline]
fn fclamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

#[inline]
fn iclamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

// 12-bit linear to sRGB in 8.4 fixed point: the four fraction bits are what the ordered
// dither rounds away.
// 4x4 ordered dither, removes banding from the sky gradient after 8-bit quantisation.
const BAYER: [u16; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

/// linear_to_srgb(c) * 255 * 16, rounded, for c in [0, 1). Above the linear toe, c^(1/2.4)
/// is its octave's power, from a register, times a polynomial in the mantissa, within
/// 1.4e-5: a twentieth of the 1/16 steps the dither works in.
struct Srgb16 {
    octaves: F8, // K 2^(e / 2.4) for e = -8 ..= -1
    below: F8,   // and for e = -9, the one octave above the toe the register cannot hold
}

impl Srgb16 {
    const K: f32 = 1.055 * 255.0 * 16.0;

    fn new() -> Srgb16 {
        let octave = |e: i32| Self::K * 2f32.powf(e as f32 / 2.4);
        Srgb16 { octaves: F8::load(&std::array::from_fn(|k| octave(k as i32 - 8))), below: F8::splat(octave(-9)) }
    }

    #[inline(always)]
    fn encode(&self, c: F8) -> I8 {
        let s = F8::splat;
        let bits = c.bits();
        let m = ((bits & I8::splat(0x7f_ffff)) | I8::splat(0x3f80_0000)).as_f8();
        let h = s(-0.010_483_21).mul_add(m, s(0.086_953_54)).mul_add(m, s(-0.313_670_5)).mul_add(m, s(0.824_403_6)).mul_add(m, s(0.412_81));
        let scale = F8::select(c.lt(s(1.0 / 256.0)), self.below, self.octaves.permute(bits.shr(I8::splat(23)) - I8::splat(119)));
        let v = F8::select(c.le(s(0.003_130_8)), c * s(12.92 * 255.0 * 16.0), scale.mul_add(h, s(-0.055 * 255.0 * 16.0)));
        (v + s(0.5)).to_int()
    }
}

/// The four Catmull-Rom weights for a fraction t in [0, 1).
#[inline(always)]
fn catmull_rom(t: F8) -> [F8; 4] {
    let t2 = t * t;
    let s = F8::splat;
    [
        t * (s(-0.5) + t * (s(1.0) - s(0.5) * t)),
        s(1.0) + t2 * (s(-2.5) + s(1.5) * t),
        t * (s(0.5) + t * (s(2.0) - s(1.5) * t)),
        t2 * (s(-0.5) + s(0.5) * t),
    ]
}

/// Eight consecutive floats from `s` at `i`.
#[inline(always)]
fn at8(s: &[f32], i: usize) -> F8 {
    F8::load(s[i..i + 8].try_into().unwrap())
}

/// Lanes that are two runs of neighbours, `idx` being a + lane in the lanes returned and
/// b + lane in the rest. That is what a group becomes where the motion changes across it,
/// one pixel skipped or repeated, and two plain loads and a blend cost much less than a
/// gather, about 13 cycles on this CPU.
#[inline(always)]
fn two_runs(idx: I8) -> Option<I8> {
    let d = idx - I8::lanes();
    let in_a = d.eq(d.first());
    (in_a | d.eq(d.permute(I8::splat(7)))).all().then_some(in_a)
}

/// `table[idx]` per lane: one load when the lanes are eight neighbours, as they nearly
/// always are here, a gather when not. Not the two-run loads of the filters: the warp is
/// one chain of dependent steps, and testing for two runs lengthened it by more than the
/// gather costs.
#[inline(always)]
fn pick(table: &[f32], idx: I8) -> F8 {
    let first = idx.first();
    if idx.eq(first + I8::lanes()).all() {
        at8(table, first.to_array()[0] as usize)
    } else {
        F8::gather(table, idx)
    }
}

/// Top left corners of the 4 x 4 filter windows of eight lanes at (col, row) that are two
/// runs of neighbours: lane 0's window, the window lane 0 would have in the second run,
/// and which lanes are in the first. None if either reaches past the plane's edge.
#[inline(always)]
fn two_windows(col: I8, row: I8, w: i32, h: i32) -> Option<(usize, usize, I8)> {
    let in_a = two_runs(row * I8::splat(w) + col)?;
    let (c, r) = (col.to_array(), row.to_array());
    let corner = |c: i32, r: i32| (c >= 1 && c + 10 <= w && r >= 1 && r + 2 < h).then(|| ((r - 1) * w + c - 1) as usize);
    // Lane 0 starts the first run and lane 7 ends the second.
    Some((corner(c[0], r[0])?, corner(c[7] - 7, r[7])?, in_a))
}

/// Catmull-Rom over a 4 x 4 neighbourhood, `tap(row, column)`, clamped to the range of
/// the middle 2 x 2 so edges do not ring. The four corner taps are left out, their
/// weights never reach 0.006, and the rest renormalised: three quarters of the reads.
#[inline(always)]
fn filter_cross(tap: impl Fn(usize, usize) -> F8, wx: &[F8; 4], wy: &[F8; 4]) -> F8 {
    let mid = |r: usize| {
        let (a, b) = (tap(r, 1), tap(r, 2));
        (a, b, tap(r, 0).mul_add(wx[0], a.mul_add(wx[1], b.mul_add(wx[2], tap(r, 3) * wx[3]))))
    };
    let ((a1, b1, l1), (a2, b2, l2)) = (mid(1), mid(2));
    let lo = a1.min(b1).min(a2.min(b2));
    let hi = a1.max(b1).max(a2.max(b2));
    let (top, bot) = (tap(0, 1).mul_add(wx[1], tap(0, 2) * wx[2]), tap(3, 1).mul_add(wx[1], tap(3, 2) * wx[2]));
    let acc = top.mul_add(wy[0], l1.mul_add(wy[1], l2.mul_add(wy[2], bot * wy[3])));
    let weight = (wy[0] + wy[3]).mul_add(wx[1] + wx[2], wy[1] + wy[2]);
    (acc * weight.recip()).max(lo).min(hi)
}

type Planes = [Vec<f32>; 3];

/// Which of the frame-sized buffers a renderer holds.
#[derive(Clone, Copy, PartialEq)]
enum Held {
    None,
    Overlay,
    All,
}

fn planes(len: usize) -> Planes {
    std::array::from_fn(|_| vec![0.0; len])
}

/// Colour planes in half floats, for the full-frame images read and written every
/// frame: half the memory traffic, and at 1080p memory is what those passes wait on.
/// One element of padding for `F8::gather_f16`.
type Halves = [Vec<u16>; 3];

fn halves(len: usize) -> Halves {
    std::array::from_fn(|_| vec![0; len + 1])
}

/// One value as a half float, rounded to nearest, and back.
#[inline]
fn to_half(v: f32) -> u16 {
    F8::splat(v).to_f16()[0]
}

#[inline]
fn from_half(v: u16) -> f32 {
    let mut out = [0.0f32; 8];
    F8::load_f16(&[v; 8]).store(&mut out);
    out[0]
}

#[inline]
fn fetch(t: &Texture, frame: i32, level: i32, u: f32, v: f32) -> u32 {
    let level = iclamp(level, 0, t.levels - 1);
    let size = (t.w >> level).max(1);
    let size_h = (t.h >> level).max(1);
    // The offset keeps the conversion a floor for the negative uvs of merged far quads;
    // it is a multiple of every power-of-two size, so wrapping is a mask.
    let mut x = (u * size as f32 + 65536.0) as i32;
    let mut y = (v * size_h as f32 + 65536.0) as i32;
    if size & (size - 1) == 0 && size_h & (size_h - 1) == 0 {
        x &= size - 1;
        y &= size_h - 1;
    } else {
        x = x.rem_euclid(size);
        y = y.rem_euclid(size_h);
    }
    t.frames[frame as usize][(t.level_offset[level as usize] + y * size + x) as usize]
}

#[inline]
fn ifloor(f: f32) -> i32 {
    let f = fclamp(f, -1e6, 1e6);
    let i = f as i32;
    i - (f < i as f32) as i32
}

#[inline]
fn iceil(f: f32) -> i32 {
    let f = fclamp(f, -1e6, 1e6);
    let i = f as i32;
    i + (f > i as f32) as i32
}

// floor and ceil without a libm call (SSE2 has no roundss). Exact for every input the
// renderer compares or converts; only the sign of a zero result can differ.
#[inline]
fn ffloor(x: f32) -> f32 {
    if x.abs() < 8388608.0 {
        let t = x as i32 as f32;
        if t > x { t - 1.0 } else { t }
    } else {
        x
    }
}

#[inline]
fn fceil(x: f32) -> f32 {
    if x.abs() < 8388608.0 {
        let t = x as i32 as f32;
        if t < x { t + 1.0 } else { t }
    } else {
        x
    }
}

#[inline]
fn fract(v: f32) -> f32 {
    v - ifloor(v) as f32
}

#[inline]
fn fast_log2(v: f32) -> f32 {
    let bits = v.to_bits();
    let e = (((bits >> 23) & 255) as i32 - 128) as f32;
    let m = f32::from_bits((bits & 0x7fffff) | 0x3f800000);
    e + (-0.34484843 * m + 2.02466578) * m - 0.67487759
}

#[inline]
fn tex_color(p: u32) -> Vec3 {
    let lut = lut();
    Vec3::new(lut[(p & 0xff) as usize], lut[((p >> 8) & 0xff) as usize], lut[((p >> 16) & 0xff) as usize])
}

#[inline]
fn tint_color(tint: u32) -> Vec3 {
    let lut = lut();
    Vec3::new(lut[((tint >> 16) & 0xff) as usize], lut[((tint >> 8) & 0xff) as usize], lut[(tint & 0xff) as usize])
}

/// Corner k of the box [lo, hi], bit 0 picking x, bit 1 y, bit 2 z.
#[inline]
fn corner(lo: Vec3, hi: Vec3, k: i32) -> Vec3 {
    Vec3::new(
        if k & 1 != 0 { hi.x } else { lo.x },
        if k & 2 != 0 { hi.y } else { lo.y },
        if k & 4 != 0 { hi.z } else { lo.z },
    )
}

#[inline]
fn row(m: &Mat4, r: usize) -> Vec4 {
    Vec4::new(m.m[0][r], m.m[1][r], m.m[2][r], m.m[3][r])
}

#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub sun_dir: Vec3, // towards the sun
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    pub ambient: Vec3,
    pub ambient_intensity: f32,
    pub block_light: Vec3,
    pub fog_start: f32,
    pub fog_end: f32,
    pub exposure: f32,
    pub saturation: f32,
    pub cloud_height: f32,
}

impl Default for Look {
    fn default() -> Look {
        Look {
            sun_dir: normalize(Vec3::new(0.55, 0.78, 0.30)),
            sun_color: Vec3::new(1.00, 0.90, 0.78),
            sun_intensity: 3.2,
            sky_zenith: Vec3::new(0.16, 0.34, 0.85),
            sky_horizon: Vec3::new(0.62, 0.76, 0.95),
            ambient: Vec3::new(0.42, 0.55, 0.85),
            ambient_intensity: 0.85,
            block_light: Vec3::new(1.0, 0.62, 0.32),
            fog_start: 110.0,
            fog_end: 230.0,
            exposure: 1.0,
            saturation: 1.12,
            cloud_height: 192.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
}

impl Default for CameraPose {
    fn default() -> CameraPose {
        CameraPose { eye: Vec3::ZERO, target: Vec3::ZERO, fov_y: 70.0 * PI / 180.0 }
    }
}

/// A player standing somewhere, for its shadow.
#[derive(Clone, Copy)]
pub struct Footprint {
    pub pos: Vec3,
}

/// Passes timed one by one, for `--bench`.
pub const PASSES: [&str; 8] = ["setup", "raster", "hiz", "sky", "shade", "dynamic", "compose", "bloom"];

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub passes: [f64; PASSES.len()],
    pub setup: f64,
    pub raster: f64,
    pub shade: f64,
    pub post: f64,
    pub prims: usize,
    pub sections: usize,
    pub occluded: usize,
    pub full: bool,
}

// ---------------------------------------------------------------------------------
// Shadows
//
// The whole scene is lit from one fixed sun: a shadow map for terrain, rebuilt only when
// terrain changes. Orthographic from the sun. Depth is distance towards the sun in 1/32
// blocks, stored in 8x8 texel tiles so a screen neighbourhood reads a compact memory area.

#[derive(Clone, Copy, Default)]
struct Basis {
    size: i32, // texels per side, a multiple of 8
    right: Vec3, // u = dot(p, right), v = dot(p, up), d = dot(p, to_sun)
    up: Vec3,
    to_sun: Vec3,
    u0: f32, // world u, v at texel 0
    v0: f32,
    d0: f32,          // world depth at stored value 0
    texel_world: f32, // world units per texel
}

impl Basis {
    #[inline]
    fn index(&self, x: i32, y: i32) -> usize {
        ((y >> 3) as usize * (self.size >> 3) as usize + (x >> 3) as usize) * 64 + ((y & 7) * 8 + (x & 7)) as usize
    }
}

#[derive(Clone)]
pub struct ShadowMap {
    b: Basis,
    depth: Vec<u16>,
}

/// Blocks the world worker has just published changed meshes for: a world box, and when
/// the change reached the engine (engine clock).
#[derive(Clone, Copy, Debug)]
pub struct Changed {
    pub lo: Vec3,
    pub hi: Vec3,
    pub arrived: f64,
}

/// What the world worker hands the renderer between frames.
#[derive(Default)]
struct Handover {
    maps: Option<Arc<Vec<ShadowMap>>>,
    changes: Vec<Changed>,
}

type ShadowSlot = Arc<Mutex<Handover>>;

/// Depth towards the sun for every texel in [x0, x1) x [y0, y1), keeping the nearest
/// caster. `depth` holds the map's texels from index `base` on. Whole sections that
/// project outside the rectangle are skipped, which is what keeps both banded full
/// builds and small updates cheap.
#[allow(clippy::too_many_arguments)]
fn raster_shadow(assets: &Assets, b: &Basis, depth: &mut [u16], base: usize, meshes: &[Arc<SectionMesh>], x0: i32,
                 y0: i32, x1: i32, y1: i32) {
    let inv = 1.0 / b.texel_world;
    let project = |p: Vec3| {
        Vec3::new((dot(p, b.right) - b.u0) * inv, (dot(p, b.up) - b.v0) * inv, (dot(p, b.to_sun) - b.d0) * 32.0)
    };
    for mesh in meshes {
        if mesh.quads.is_empty() {
            continue;
        }
        let (mut sx0, mut sy0, mut sx1, mut sy1) = (1e30f32, 1e30f32, -1e30f32, -1e30f32);
        for k in 0..8 {
            let t = project(corner(mesh.lo, mesh.hi, k));
            sx0 = fmin(sx0, t.x);
            sx1 = fmax(sx1, t.x);
            sy0 = fmin(sy0, t.y);
            sy1 = fmax(sy1, t.y);
        }
        if sx1 < (x0 - 1) as f32 || sx0 > (x1 + 1) as f32 || sy1 < (y0 - 1) as f32 || sy0 > (y1 + 1) as f32 {
            continue;
        }
        for q in &mesh.quads {
            if q.mat & MAT_NO_SHADOW != 0 {
                continue;
            }
            let v0 = project(q.p);
            let e1 = project(q.p + q.e1) - v0;
            let e2 = project(q.p + q.e2) - v0;
            let tri = q.flags & QUAD_TRI != 0;
            let min_x = v0.x + fmin(0.0, e1.x) + fmin(0.0, e2.x);
            let max_x = v0.x + fmax(0.0, e1.x) + fmax(0.0, e2.x);
            let min_y = v0.y + fmin(0.0, e1.y) + fmin(0.0, e2.y);
            let max_y = v0.y + fmax(0.0, e1.y) + fmax(0.0, e2.y);
            if max_y < y0 as f32 || min_y > y1 as f32 || max_x < x0 as f32 || min_x > x1 as f32 {
                continue;
            }
            let py0 = y0.max(fceil(min_y - 0.5) as i32);
            let py1 = (y1 - 1).min(ffloor(max_y - 0.5) as i32);
            let px0 = x0.max(fceil(min_x - 0.5) as i32);
            let px1 = (x1 - 1).min(ffloor(max_x - 0.5) as i32);
            if py0 > py1 || px0 > px1 {
                // Thinner than a texel: still casts, at its centre.
                let cx = ((min_x + max_x) * 0.5) as i32;
                let cy = ((min_y + max_y) * 0.5) as i32;
                if cy >= y0 && cy < y1 && cx >= x0 && cx < x1 && q.layer == SOLID {
                    let d = v0.z + (e1.z + e2.z) * 0.5;
                    let dv = fclamp(d, 0.0, 65535.0) as u16;
                    let z = &mut depth[b.index(cx, cy) - base];
                    *z = (*z).max(dv);
                }
                continue;
            }
            let det = e1.x * e2.y - e1.y * e2.x;
            if det.abs() < 1e-9 {
                continue;
            }
            let idet = 1.0 / det;
            let tex = assets.tex(q.tex);
            let cutout = q.layer != SOLID;
            for py in py0..=py1 {
                let fy = py as f32 + 0.5 - v0.y;
                for px in px0..=px1 {
                    let fx = px as f32 + 0.5 - v0.x;
                    let s = (fx * e2.y - fy * e2.x) * idet;
                    let t = (fy * e1.x - fx * e1.y) * idet;
                    const EPS: f32 = 1e-3;
                    if s < -EPS || t < -EPS {
                        continue;
                    }
                    if if tri { s + t > 1.0 + EPS } else { s > 1.0 + EPS || t > 1.0 + EPS } {
                        continue;
                    }
                    let d = v0.z + s * e1.z + t * e2.z;
                    let dv = fclamp(d, 0.0, 65535.0) as u16;
                    let z = &mut depth[b.index(px, py) - base];
                    if dv <= *z {
                        continue;
                    }
                    if cutout {
                        let u = q.uv.x + s * q.uv_s.x + t * q.uv_t.x;
                        let vv = q.uv.y + s * q.uv_s.y + t * q.uv_t.y;
                        if (fetch(tex, 0, 0, u, vv) >> 24) < 128 {
                            continue;
                        }
                    }
                    *z = dv;
                }
            }
        }
    }
}

/// Builds terrain shadow maps off the frame path, from any thread, and hands them to the
/// renderer, which swaps them in at its next frame.
#[derive(Clone)]
pub struct Shadows {
    assets: Arc<Assets>,
    to_sun: Vec3,
    pending: ShadowSlot,
}

impl Shadows {
    /// Full rebuild from the current meshes. Heavy.
    pub fn build(&self, meshes: &[Arc<SectionMesh>], focus: Vec3, radius: f32) -> Arc<Vec<ShadowMap>> {
        let cascades = [(48.0f32, 2560i32), (radius + 16.0, 2048)];
        let to_sun = self.to_sun;
        let right =
            normalize(cross(if to_sun.y.abs() > 0.99 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(0.0, 1.0, 0.0) }, to_sun));
        let up = cross(to_sun, right);
        let mut maps = Vec::new();
        for (half, size) in cascades {
            let b = Basis {
                size,
                right,
                up,
                to_sun,
                texel_world: 2.0 * half / size as f32,
                u0: dot(focus, right) - half,
                v0: dot(focus, up) - half,
                d0: dot(focus, to_sun) - 1000.0,
            };
            // One texel of padding for `I8::gather_u16`.
            let mut depth = vec![0u16; (size * size) as usize + 1];
            // Runs on its own threads: this happens in the background while frames render.
            // Bands are whole rows of tiles, so each is one run of memory.
            let band_h = ((size + 31) / 32 + 7) & !7;
            let count = (size + band_h - 1) / band_h;
            let bands = Parts::chunks(&mut depth, (band_h * size) as usize);
            let next = AtomicI32::new(0);
            std::thread::scope(|s| {
                for _ in 0..2 {
                    s.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= count {
                                break;
                            }
                            let mut band = bands.take(i as usize);
                            let y0 = i * band_h;
                            raster_shadow(&self.assets, &b, &mut band, (y0 * size) as usize, meshes, 0, y0, size,
                                          size.min(y0 + band_h));
                        }
                    });
                }
            });
            drop(bands);
            maps.push(ShadowMap { b, depth });
        }
        Arc::new(maps)
    }

    /// A copy of prev with only the texels under the world boxes redrawn: what a block
    /// change costs, instead of a full rebuild.
    pub fn update(&self, prev: &[ShadowMap], meshes: &[Arc<SectionMesh>], boxes: &[(Vec3, Vec3)]) -> Arc<Vec<ShadowMap>> {
        let mut maps = prev.to_vec();
        for sm in &mut maps {
            for &(lo, hi) in boxes {
                // Texels depend only on position across the sun's direction, so the box's
                // projection is exactly the area whose nearest caster may have changed.
                let b = sm.b;
                let inv = 1.0 / b.texel_world;
                let (mut x0, mut y0, mut x1, mut y1) = (1e30f32, 1e30f32, -1e30f32, -1e30f32);
                for k in 0..8 {
                    let c = corner(lo, hi, k);
                    let u = (dot(c, b.right) - b.u0) * inv;
                    let v = (dot(c, b.up) - b.v0) * inv;
                    x0 = fmin(x0, u);
                    x1 = fmax(x1, u);
                    y0 = fmin(y0, v);
                    y1 = fmax(y1, v);
                }
                let ix0 = 0.max(x0 as i32 - 2);
                let iy0 = 0.max(y0 as i32 - 2);
                let ix1 = b.size.min(x1 as i32 + 3);
                let iy1 = b.size.min(y1 as i32 + 3);
                if ix0 >= ix1 || iy0 >= iy1 {
                    continue;
                }
                for y in iy0..iy1 {
                    for x in ix0..ix1 {
                        sm.depth[b.index(x, y)] = 0;
                    }
                }
                raster_shadow(&self.assets, &b, &mut sm.depth, 0, meshes, ix0, iy0, ix1, iy1);
            }
        }
        Arc::new(maps)
    }

    pub fn to_sun(&self) -> Vec3 {
        self.to_sun
    }

    pub fn install(&self, maps: Arc<Vec<ShadowMap>>) {
        let mut p = self.pending.lock().unwrap();
        p.maps = Some(maps);
    }

    /// Takes back maps installed and not yet swapped in.
    pub fn forget(&self) {
        self.pending.lock().unwrap().maps = None;
    }

    /// Tells the renderer which blocks just changed, so it draws them without delay.
    pub fn changed(&self, changes: &[Changed]) {
        let mut p = self.pending.lock().unwrap();
        p.changes.extend_from_slice(changes);
        // Taken every frame; this only bounds what piles up while nothing renders.
        let excess = p.changes.len().saturating_sub(1024);
        p.changes.drain(..excess);
    }
}

// ---------------------------------------------------------------------------------
// Scene: everything shading reads, fixed while a frame's jobs run.

struct Scene {
    assets: Arc<Assets>,
    look: Look,
    sky_lut: Vec<Vec3>, // sky gradient by (elevation, angle to sun)
    sky_lut_planes: Planes, // the same, one plane per channel
    cloud_map: Vec<u32>, // clouds.png alpha, 256 x 256
    cloud_soft: Vec<f32>, // blurred coverage for cloud shadows
    shadows: Option<Arc<Vec<ShadowMap>>>,
    anim_frame: Vec<i32>, // per texture, this frame
    frame: u32,
    cloud_drift: Vec2,
    ray_origin: Vec3,
}

impl Scene {
    fn build_sky_lut(&mut self) {
        let l = self.look;
        self.sky_lut.resize(SKY_UP * SKY_SUN, Vec3::ZERO);
        for j in 0..SKY_SUN {
            let k = j as f32 / (SKY_SUN - 1) as f32; // sqrt((1 - cosSun) / 2)
            let cos_sun = 1.0 - 2.0 * k * k;
            for i in 0..SKY_UP {
                let up = i as f32 / (SKY_UP - 1) as f32 * 2.0 - 1.0;
                let t = clamp01(up).powf(0.45);
                let mut sky = mix3(l.sky_horizon, l.sky_zenith, t);
                if up < 0.0 {
                    sky = l.sky_horizon * (1.0 - 0.25 * clamp01(-up * 4.0));
                }
                let glow = fmax(0.0, cos_sun);
                sky += l.sun_color * (glow.powf(6.0) * 0.35 + glow.powf(60.0) * 0.6);
                sky += Vec3::new(0.25, 0.14, 0.05) * (glow.powf(3.0) * (1.0 - clamp01(up.abs())).powf(6.0));
                self.sky_lut[j * SKY_UP + i] = sky;
            }
        }
        self.sky_lut_planes = std::array::from_fn(|ch| self.sky_lut.iter().map(|c| [c.x, c.y, c.z][ch]).collect());
    }

    fn sky_gradient(&self, dir: Vec3) -> Vec3 {
        let cos_sun = dot(dir, self.look.sun_dir);
        let k = fmax(0.0, (1.0 - cos_sun) * 0.5).sqrt() * (SKY_SUN - 1) as f32;
        let u = clamp01(dir.y * 0.5 + 0.5) * (SKY_UP - 1) as f32;
        let ik = (k as i32).min(SKY_SUN as i32 - 2);
        let iu = (u as i32).min(SKY_UP as i32 - 2);
        let fk = k - ik as f32;
        let fu = u - iu as f32;
        let r0 = (ik as usize) * SKY_UP + iu as usize;
        let r1 = r0 + SKY_UP;
        let l = &self.sky_lut;
        mix3(mix3(l[r0], l[r0 + 1], fu), mix3(l[r1], l[r1 + 1], fu), fk)
    }

    fn sky_color(&self, dir: Vec3, clouds: bool) -> Vec3 {
        let l = &self.look;
        let up = dir.y;
        let mut sky = self.sky_gradient(dir);
        let cos_sun = dot(dir, l.sun_dir);

        // Square sun like the game's, with a bright core that feeds the bloom.
        let sun_right = normalize(cross(l.sun_dir, Vec3::new(0.0, 0.0, 1.0)));
        let sun_up = cross(sun_right, l.sun_dir);
        if cos_sun > 0.99 {
            let u = dot(dir, sun_right) / cos_sun;
            let v = dot(dir, sun_up) / cos_sun;
            let size = 0.045f32;
            let edge = fmax(u.abs(), v.abs());
            if edge < size {
                sky = l.sun_color * 30.0;
            } else if edge < size * 1.6 {
                sky += l.sun_color * (2.0 * (1.0 - (edge - size) / (size * 0.6)));
            }
        }

        if clouds && up > 0.02 && !self.cloud_map.is_empty() {
            // Blocky clouds: 12 x 12 block cells, 4 blocks thick, drifting slowly.
            let base = l.cloud_height;
            let top = base + 4.0;
            let drift = self.cloud_drift;
            let o = self.ray_origin;
            if o.y < base {
                let t_enter = (base - o.y) / up;
                let t_exit = (top - o.y) / up;
                let enter = o + dir * t_enter;
                let cell_x = (enter.x + drift.x) / 12.0;
                let cell_z = (enter.z + drift.y) / 12.0;
                let mut ix = ffloor(cell_x) as i32;
                let mut iz = ffloor(cell_z) as i32;
                let dx = dir.x / 12.0;
                let dz = dir.z / 12.0;
                let t_max = t_exit - t_enter;
                let mut t_cur = 0.0f32;
                let step_x = if dx > 0.0 { 1 } else { -1 };
                let step_z = if dz > 0.0 { 1 } else { -1 };
                let t_delta_x = if dx != 0.0 { (1.0 / dx).abs() } else { 1e30 };
                let t_delta_z = if dz != 0.0 { (1.0 / dz).abs() } else { 1e30 };
                let mut t_next_x = if dx != 0.0 {
                    (if dx > 0.0 { (ix + 1) as f32 - cell_x } else { cell_x - ix as f32 }) * t_delta_x
                } else {
                    1e30
                };
                let mut t_next_z = if dz != 0.0 {
                    (if dz > 0.0 { (iz + 1) as f32 - cell_z } else { cell_z - iz as f32 }) * t_delta_z
                } else {
                    1e30
                };
                let mut face = 0; // 0 bottom, 1 side x, 2 side z
                let mut hit = false;
                let mut step = 0;
                while step < 10 && t_cur <= t_max {
                    if self.cloud_map[((iz & 255) * 256 + (ix & 255)) as usize] != 0 {
                        hit = true;
                        break;
                    }
                    if t_next_x < t_next_z {
                        t_cur = t_next_x;
                        t_next_x += t_delta_x;
                        ix += step_x;
                        face = 1;
                    } else {
                        t_cur = t_next_z;
                        t_next_z += t_delta_z;
                        iz += step_z;
                        face = 2;
                    }
                    step += 1;
                }
                if hit && t_cur <= t_max {
                    let dist = t_enter + t_cur;
                    let sun_term = match face {
                        0 => 0.0,
                        1 => l.sun_dir.x.abs(),
                        _ => l.sun_dir.z.abs(),
                    };
                    let lit = Vec3::new(0.78, 0.82, 0.92) * (if face == 0 { 0.82 } else { 0.95 })
                        + l.sun_color * (0.35 * sun_term);
                    let fade = (-dist / 1400.0).exp() * smoothstep(0.02, 0.12, up);
                    sky = mix3(sky, lit, 0.92 * fade);
                }
            }
        }
        sky
    }

    /// Water, or another translucent surface, over colour `below` whose surface is `dist`
    /// away (sky if `over_sky`), for the pixel at (px, py) seen along the unit `ray`.
    #[allow(clippy::too_many_arguments)]
    fn water_over(&self, lists: &[Lists], wid: u32, px: f32, py: f32, ray: Vec3, below: Vec3, dist: f32, over_sky: bool, waves: [f32; 4]) -> Vec3 {
        let l = &self.look;
        let sun_light = l.sun_color * l.sun_intensity;
        let ambient = l.ambient * l.ambient_intensity;
        let p = prim(lists, wid);
        let q = p.q;
        let a = p.a[0] * px + p.a[1] * py + p.a[2];
        let wv = 1.0 / a;
        let s = clamp01((p.s[0] * px + p.s[1] * py + p.s[2]) * wv);
        let t = clamp01((p.t[0] * px + p.t[1] * py + p.t[2]) * wv);
        let wpos = p.at(s, t);
        let wdist = length(wpos - self.ray_origin);
        let tint = tint_color(q.tint);
        let u = q.uv.x + s * q.uv_s.x + t * q.uv_t.x;
        let v = q.uv.y + s * q.uv_s.y + t * q.uv_t.y;
        let texel = fetch(self.assets.tex(q.tex), self.anim_frame[q.tex as usize], 0, u, v);
        let mut c;
        if q.mat & MAT_WATER != 0 {
            // Gentle procedural waves for the normal; the texture barely shows through, as
            // in shader packs.
            let mut n = q.n;
            if n.y > 0.5 {
                let nx = (wpos.x * 1.3 + waves[0]).sin() * 0.05 + ((wpos.x + wpos.z) * 2.7 + waves[1]).sin() * 0.03;
                let nz = (wpos.z * 1.1 + waves[2]).sin() * 0.05 + ((wpos.x - wpos.z) * 2.3 + waves[3]).sin() * 0.03;
                n = normalize(Vec3::new(nx, 1.0, nz));
            }
            let cosv = fmax(0.02, -dot(ray, n));
            let fres = 0.02 + 0.98 * (1.0 - cosv).powf(5.0);
            let mut refl = ray - n * (2.0 * dot(ray, n));
            if refl.y < 0.02 {
                refl.y = 0.02;
            }
            let mut refl_col = self.sky_color(normalize(refl), true);
            let sky_l = q.sky[0] as f32 / 255.0;
            refl_col = refl_col * (0.35 + 0.65 * sky_l * sky_l);
            let depth_below = if over_sky { 64.0 } else { fmax(0.0, dist - wdist) };
            let absorb = Vec3::new(0.45, 0.12, 0.08);
            let trans =
                Vec3::new((-absorb.x * depth_below).exp(), (-absorb.y * depth_below).exp(), (-absorb.z * depth_below).exp());
            let deep = tint * Vec3::new(0.10, 0.16, 0.22) * (0.4 + 0.6 * sky_l);
            let mut under = below * trans + deep * (Vec3::new(1.0, 1.0, 1.0) - trans);
            let tex_lum = luminance(tex_color(texel));
            under = under * (0.85 + 0.3 * tex_lum);
            // A tight glint, capped: the waves scatter it across many pixels and an
            // uncapped one blooms into a white cloud over the whole pond.
            let spec = fmin(3.0, fmax(0.0, dot(refl, l.sun_dir)).powf(900.0) * 12.0);
            let glint = self.shadow_at(wpos, Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0) * self.cloud_shadow(wpos);
            c = mix3(under, refl_col, fres) + l.sun_color * (spec * glint);
        } else {
            let alpha = (texel >> 24) as f32 / 255.0;
            let albedo = tex_color(texel) * tint;
            let sky_l = q.sky[0] as f32 / 255.0;
            let lit = albedo * (ambient * (sky_l * sky_l) + sun_light * fmax(0.0, dot(q.n, l.sun_dir)) * 0.6);
            c = mix3(below, lit, alpha);
        }
        let fog = smoothstep(l.fog_start, l.fog_end, wdist);
        if fog > 0.0 {
            c = mix3(c, self.sky_gradient(ray), fog);
        }
        c
    }

    fn cloud_shadow(&self, p: Vec3) -> f32 {
        if self.cloud_soft.is_empty() {
            return 1.0;
        }
        let l = &self.look;
        let k = (l.cloud_height - p.y) / fmax(0.2, l.sun_dir.y);
        let x = (p.x + l.sun_dir.x * k + self.cloud_drift.x) / 12.0 - 0.5;
        let z = (p.z + l.sun_dir.z * k + self.cloud_drift.y) / 12.0 - 0.5;
        let ix = ifloor(x);
        let iz = ifloor(z);
        let fx = x - ix as f32;
        let fz = z - iz as f32;
        let c = |a: i32, b: i32| self.cloud_soft[((b & 255) * 256 + (a & 255)) as usize];
        let v = mix(mix(c(ix, iz), c(ix + 1, iz), fx), mix(c(ix, iz + 1), c(ix + 1, iz + 1), fx), fz);
        1.0 - 0.45 * v
    }

    fn shadow_at(&self, p: Vec3, n: Vec3, jx: f32, jy: f32) -> f32 {
        let Some(maps) = &self.shadows else { return 1.0 };
        for sm in maps.iter() {
            let b = &sm.b;
            // Normal offset of a texel and a half keeps self-shadowing acne away without a
            // large depth bias, so contact shadows stay attached.
            let q = p + n * (b.texel_world * 1.5);
            let inv = 1.0 / b.texel_world;
            // One bilinear 2x2 lookup, offset by per-pixel per-frame noise: the temporal
            // resolve integrates the offsets into a soft penumbra at a quarter of the cost
            // of a wide filter.
            let spread = if b.size > 2300 { 2.0 } else { 1.2 };
            let u = (dot(q, b.right) - b.u0) * inv - 0.5 + jx * spread;
            let v = (dot(q, b.up) - b.v0) * inv - 0.5 + jy * spread;
            if u < 1.0 || v < 1.0 || u > (b.size - 2) as f32 || v > (b.size - 2) as f32 {
                continue;
            }
            let d = (dot(q, b.to_sun) - b.d0) * 32.0 + 2.0; // 1/16 block bias
            let iu = u as i32;
            let iv = v as i32;
            let fu = u - iu as f32;
            let fv = v - iv as f32;
            let lit = |x: i32, y: i32| if d >= sm.depth[b.index(x, y)] as f32 { 1.0 } else { 0.0 };
            let (a, bb, c, e) = (lit(iu, iv), lit(iu + 1, iv), lit(iu, iv + 1), lit(iu + 1, iv + 1));
            return mix(mix(a, bb, fu), mix(c, e, fu), fv);
        }
        1.0
    }

    /// Colour of a surface point, and how much direct sun reaches it.
    #[allow(clippy::too_many_arguments)]
    fn lit_surface(&self, p: &Prim, s: f32, t: f32, w: f32, pos: Vec3, x: i32, y: i32) -> (Vec3, f32) {
        let l = &self.look;
        let lut = lut();
        let q = p.q;
        let sun_light = l.sun_color * l.sun_intensity;
        let ambient = l.ambient * l.ambient_intensity;
        let u = q.uv.x + s * q.uv_s.x + t * q.uv_t.x;
        let v = q.uv.y + s * q.uv_s.y + t * q.uv_t.y;
        let tex = self.assets.tex(q.tex);
        // Analytic screen derivatives of s and t pick the mip level.
        let dsx = (p.s[0] - s * p.a[0]) * w;
        let dtx = (p.t[0] - t * p.a[0]) * w;
        let dsy = (p.s[1] - s * p.a[1]) * w;
        let dty = (p.t[1] - t * p.a[1]) * w;
        let dux = q.uv_s.x * dsx + q.uv_t.x * dtx;
        let dvx = q.uv_s.y * dsx + q.uv_t.y * dtx;
        let duy = q.uv_s.x * dsy + q.uv_t.x * dty;
        let dvy = q.uv_s.y * dsy + q.uv_t.y * dty;
        let rho = fmax(dux * dux + dvx * dvx, duy * duy + dvy * dvy) * (tex.w * tex.w) as f32;
        let level = fmax(0.0, 0.5 * fast_log2(fmax(rho, 1e-6)) + MIP_ROUND) as i32;
        let texel = fetch(tex, self.anim_frame[q.tex as usize], level, u, v);
        let mut albedo =
            Vec3::new(lut[(texel & 0xff) as usize], lut[((texel >> 8) & 0xff) as usize], lut[((texel >> 16) & 0xff) as usize]);
        if q.tint != 0xffffff {
            albedo = albedo * tint_color(q.tint);
        }

        let (sky_l, blk_l, ao);
        if q.flags & QUAD_FLAT != 0 {
            sky_l = q.sky[0] as f32 / 255.0;
            blk_l = q.blk[0] as f32 / 255.0;
            ao = q.ao[0] as f32 / 255.0;
        } else {
            let w00 = (1.0 - s) * (1.0 - t);
            let w10 = s * (1.0 - t);
            let w01 = (1.0 - s) * t;
            let w11 = s * t;
            let blend =
                |c: [u8; 4]| (c[0] as f32 * w00 + c[1] as f32 * w10 + c[2] as f32 * w01 + c[3] as f32 * w11) / 255.0;
            sky_l = blend(q.sky);
            blk_l = blend(q.blk);
            ao = blend(q.ao);
        }
        let n = q.n;
        let plant = q.mat & MAT_PLANT != 0;
        let mut ndl = if plant { 0.72 } else { dot(n, l.sun_dir) };
        if q.mat & MAT_LEAVES != 0 {
            ndl = 0.35 + 0.65 * fmax(0.0, ndl);
        }
        let mut sun = 0.0;
        if ndl > 0.0 {
            let vis = smoothstep(0.45, 0.85, sky_l);
            if vis > 0.0 {
                // Interleaved gradient noise, rotated each frame.
                let g = fract(52.9829189 * fract(0.06711056 * (x as f32 + 5.588238 * (self.frame & 63) as f32) + 0.00583715 * y as f32));
                let g2 = fract(g * 7.123 + 0.37);
                let normal = if plant { Vec3::new(0.0, 1.0, 0.0) } else { n };
                sun = ndl * vis * self.shadow_at(pos, normal, g - 0.5, g2 - 0.5) * self.cloud_shadow(pos);
            }
        }
        let sky_amb = sky_l * sky_l * (0.7 + 0.3 * sky_l);
        let dir_amb = if plant { 0.9 } else { 0.72 + 0.28 * n.y };
        let mut light = sun_light * sun + ambient * (sky_amb * ao * dir_amb);
        // Light bounced off the ground and nearby surfaces, warmest where the sun lands.
        light += Vec3::new(0.30, 0.26, 0.20) * (sky_amb * ao * 0.35 * (1.0 - fmax(0.0, n.y)));
        let bl = blk_l * blk_l * blk_l;
        light += l.block_light * (bl * 2.2 * ao);
        light += Vec3::new(0.02, 0.02, 0.025); // never pure black
        let mut c = albedo * light;
        if q.mat & MAT_EMISSIVE != 0 {
            let lum = luminance(albedo);
            let k = if q.mat & MAT_LAVA != 0 { 2.5 } else { smoothstep(0.35, 0.75, lum) * 3.0 };
            c += albedo * k;
        }
        (c, sun)
    }
}

// ---------------------------------------------------------------------------------
// Prims

struct Prim<'a> {
    q: &'a Quad,
    a: [f32; 3], // plane coefficients over pixel coords: v = c0 * x + c1 * y + c2
    s: [f32; 3],
    t: [f32; 3],
    eps: f32,
    sway: f32,
    x0: i16,
    y0: i16,
    x1: i16,
    y1: i16,
    level: u8, // mip estimate for alpha tests in raster
    swayed: bool,
}

/// Position and second edge of a swaying plant quad this frame.
#[inline]
fn swayed(q: &Quad, sway: f32) -> (Vec3, Vec3) {
    let off = Vec3::new(sway, 0.0, sway * 0.6);
    if q.wave == 2 { (q.p + off, q.e2 + off) } else { (q.p, q.e2 + off) }
}

impl Prim<'_> {
    /// The surface point at (s, t).
    #[inline]
    fn at(&self, s: f32, t: f32) -> Vec3 {
        let q = self.q;
        if self.swayed {
            let (p, e2) = swayed(q, self.sway);
            p + q.e1 * s + e2 * t
        } else {
            q.p + q.e1 * s + q.e2 * t
        }
    }
}

/// One worker's prims and, per band of rows, the prims touching it.
struct Lists<'a> {
    prims: Vec<Prim<'a>>,
    bands: Vec<Vec<u32>>,
    water: Vec<Vec<u32>>,
}

impl<'a> Lists<'a> {
    fn new(bands: usize) -> Lists<'a> {
        Lists { prims: Vec::new(), bands: vec![Vec::new(); bands], water: vec![Vec::new(); bands] }
    }

    fn clear(&mut self) {
        self.prims.clear();
        self.bands.iter_mut().for_each(Vec::clear);
        self.water.iter_mut().for_each(Vec::clear);
    }

    /// Empties the lists and lets them outlive the meshes the prims pointed into. The
    /// collect reuses the allocation, since the element layout does not change.
    fn recycle(mut self) -> Lists<'static> {
        self.clear();
        Lists {
            prims: self.prims.into_iter().map(|_| -> Prim<'static> { unreachable!() }).collect(),
            bands: self.bands,
            water: self.water,
        }
    }
}

/// What prim setup needs to know about the camera.
#[derive(Clone, Copy)]
struct Setup {
    // Pixel-homogeneous transform: (xp, yp, w) = S * viewProj * p.
    mx: [f32; 4],
    my: [f32; 4],
    mw: [f32; 4],
    eye: Vec3,
    time: f64,
    sway: bool,
    w: i32,
    h: i32,
    // Prims wholly outside this pixel rectangle [x0, y0, x1, y1) are left out.
    clip: [f32; 4],
}

fn add_prim<'a>(c: &Setup, assets: &Assets, q: &'a Quad, water: bool, out: &mut Lists<'a>) {
    let hpoint = |p: Vec3| {
        Vec3::new(
            c.mx[0] * p.x + c.mx[1] * p.y + c.mx[2] * p.z + c.mx[3],
            c.my[0] * p.x + c.my[1] * p.y + c.my[2] * p.z + c.my[3],
            c.mw[0] * p.x + c.mw[1] * p.y + c.mw[2] * p.z + c.mw[3],
        )
    };
    let hdir = |d: Vec3| {
        Vec3::new(
            c.mx[0] * d.x + c.mx[1] * d.y + c.mx[2] * d.z,
            c.my[0] * d.x + c.my[1] * d.y + c.my[2] * d.z,
            c.mw[0] * d.x + c.mw[1] * d.y + c.mw[2] * d.z,
        )
    };
    let (mut sway, mut is_swayed) = (0.0f32, false);
    let (mut qp, mut qe2) = (q.p, q.e2);
    if c.sway && q.wave != 0 && (q.p.x - c.eye.x).abs() + (q.p.z - c.eye.z).abs() < 56.0 {
        let base = q.p;
        let phase = c.time as f32 * 1.6 + base.x * 0.37 + base.z * 0.29;
        sway = ((phase).sin() * 0.6 + (phase * 1.9 + 1.3).sin() * 0.4) * 0.045;
        is_swayed = true;
        (qp, qe2) = swayed(q, sway);
    }
    let to_eye = c.eye - qp;
    if dot(q.n, to_eye) <= 0.0 && !water {
        return;
    }
    let h0 = hpoint(qp);
    let d1 = hdir(q.e1);
    let d2 = hdir(qe2);
    let h1 = h0 + d1;
    let h2 = h0 + d2;
    let h3 = h1 + d2;
    let tri = q.flags & QUAD_TRI != 0;
    let nv = if tri { 3 } else { 4 };
    let mut hs = [h0, h1, h3, h2];
    if tri {
        hs[2] = h2;
    }
    // Screen bounds, clipping against the near plane only for the bound.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (1e30f32, 1e30f32, -1e30f32, -1e30f32);
    let behind = hs[..nv].iter().filter(|h| h.z < NEAR).count();
    if behind == nv {
        return;
    }
    let mut extend = |h: Vec3| {
        let x = h.x / h.z;
        let y = h.y / h.z;
        min_x = fmin(min_x, x);
        max_x = fmax(max_x, x);
        min_y = fmin(min_y, y);
        max_y = fmax(max_y, y);
    };
    if behind == 0 {
        hs[..nv].iter().for_each(|&h| extend(h));
    } else {
        for i in 0..nv {
            let a = hs[i];
            let b = hs[(i + 1) % nv];
            if a.z >= NEAR {
                extend(a);
            }
            if (a.z >= NEAR) != (b.z >= NEAR) {
                let t = (NEAR - a.z) / (b.z - a.z);
                extend(a + (b - a) * t);
            }
        }
    }
    let (w, h) = (c.w, c.h);
    if max_x < c.clip[0] || max_y < c.clip[1] || min_x > c.clip[2] || min_y > c.clip[3] {
        return;
    }
    // Covers no pixel centre this frame: nothing to draw (the jitter moves the grid, so
    // tiny detail still appears through the temporal resolve).
    if ffloor(max_x - 0.5) < fceil(min_x - 0.5) || ffloor(max_y - 0.5) < fceil(min_y - 0.5) {
        return;
    }
    // Plane equations via the inverse of [h0 h1 h2].
    let (m00, m01, m02) = (h0.x, h1.x, h2.x);
    let (m10, m11, m12) = (h0.y, h1.y, h2.y);
    let (m20, m21, m22) = (h0.z, h1.z, h2.z);
    let c00 = m11 * m22 - m12 * m21;
    let c01 = m12 * m20 - m10 * m22;
    let c02 = m10 * m21 - m11 * m20;
    let det = m00 * c00 + m01 * c01 + m02 * c02;
    if det.abs() < 1e-9 {
        return;
    }
    let id = 1.0 / det;
    // Rows of the inverse.
    let r0 = [c00 * id, (m02 * m21 - m01 * m22) * id, (m01 * m12 - m02 * m11) * id];
    let r1 = [c01 * id, (m00 * m22 - m02 * m20) * id, (m02 * m10 - m00 * m12) * id];
    let r2 = [c02 * id, (m01 * m20 - m00 * m21) * id, (m00 * m11 - m01 * m10) * id];
    let extent = fmax(max_x - min_x, max_y - min_y);
    let mut p = Prim {
        q,
        a: [r0[0] + r1[0] + r2[0], r0[1] + r1[1] + r2[1], r0[2] + r1[2] + r2[2]],
        s: r1,
        t: r2,
        eps: fclamp(0.45 / fmax(extent, 1.0), 1e-4, 0.05),
        sway,
        x0: iclamp(ffloor(min_x) as i32 - 1, 0, w - 1) as i16,
        y0: iclamp(ffloor(min_y) as i32 - 1, 0, h - 1) as i16,
        x1: iclamp(fceil(max_x) as i32 + 1, 0, w - 1) as i16,
        y1: iclamp(fceil(max_y) as i32 + 1, 0, h - 1) as i16,
        level: 0,
        swayed: is_swayed,
    };
    if q.layer == CUTOUT {
        let tex = assets.tex(q.tex);
        let texels = tex.w as f32 * fmax(q.uv_s.x.hypot(q.uv_s.y), q.uv_t.x.hypot(q.uv_t.y));
        let ratio = texels / fmax(extent, 0.5);
        p.level = iclamp(fast_log2(fmax(ratio, 1.0)) as i32, 0, tex.levels - 1) as u8;
    }
    let idx = out.prims.len() as u32;
    let (y0, y1) = (p.y0 as i32, p.y1 as i32);
    out.prims.push(p);
    let bands = if q.layer == TRANSLUCENT { &mut out.water } else { &mut out.bands };
    for b in y0 / BAND..=y1 / BAND {
        bands[b as usize].push(idx);
    }
}

#[inline]
fn prim<'l, 'a>(lists: &'l [Lists<'a>], id: u32) -> &'l Prim<'a> {
    &lists[(id >> 26) as usize].prims[(id & 0x3ff_ffff) as usize]
}

// ---------------------------------------------------------------------------------
// Renderer

/// The camera of the frame being drawn.
#[derive(Clone, Copy, Default)]
struct View {
    eye: Vec3,
    waves: [f32; 4], // the water's wave phases this frame
    ray_x: Vec3, // world ray for pixel: base + x * ray_x + y * ray_y
    ray_y: Vec3,
    ray_base: Vec3,
    jitter_x: f32,
    jitter_y: f32,
    view_proj: Mat4,
    prev_view_proj: Mat4,
    unjittered: Mat4,
    time: f64,
}

/// The camera the last full frame was taken with. In-between frames warp its image;
/// full frames reuse it sparsely.
#[derive(Clone, Copy, Default)]
struct StaticView {
    jitter_x: f32,
    jitter_y: f32,
    ray_base: Vec3,
    ray_x: Vec3,
    ray_y: Vec3,
    eye: Vec3,
}

pub struct Renderer {
    jobs: Arc<Jobs>,
    scene: Scene,
    w: i32,
    h: i32,
    bands: i32,
    view: View,
    st: StaticView,
    history_valid: bool,
    hiz_valid: bool,
    static_valid: bool,
    // The full frame before the last one: its camera, and whether there was one. Its depths
    // are in `shaded_depth` until the next full frame shades.
    prev_st: StaticView,
    prev_static: bool,
    last_full: u32,
    static_every: i32,
    // Full renders owed or saved, in frames: one comes every static_every frames, and a
    // change may bring the next one forward while this stays at least static_every.
    credit: i32,
    // Changes waiting for the next full render, and when those just drawn had arrived.
    changes: Vec<Changed>,
    shown: Vec<f64>,
    sway: bool,
    loop_seconds: f32,
    casters: Vec<Footprint>,
    // One set of lists per worker for the static world, plus one more for the dynamic layer.
    pool: Vec<Lists<'static>>,
    table: PrimTable,
    texels: TexPool,

    // A full frame's visibility: nearest opaque and nearest water surface.
    depth: Vec<f32>,
    depth_water: Vec<f32>,
    id: Vec<u32>,
    id_water: Vec<u32>,
    // What a full frame shades, before it becomes the static image below.
    shaded: Halves,
    shaded_sun: Vec<f32>,
    shaded_depth: Vec<f32>,
    // Players and other moving things, drawn every frame by themselves and laid over the
    // static image in the compose pass. Only bands with something in them are cleared
    // and read.
    dyn_depth: Vec<f32>,
    dyn_id: Vec<u32>,
    dyn_color: Halves, // linear, in half floats: the players' pixels only, far below a step of the output
    dyn_motion: [Vec<f32>; 2],
    dyn_band: Vec<u8>,
    hiz: Vec<f32>,    // farthest 1/w per HIZ block, last frame
    hiz_w: i32,
    hiz_h: i32,
    hiz_view_proj: Mat4,
    // The last full frame's static image: colour, direct sun and nearest-surface 1/w
    // (0 = sky).
    s_color: Halves,
    s_sun: Vec<f32>, // direct sun reaching each pixel, 0..1 (for player shadows)
    s_depth: Vec<f32>,
    // The nearest and farthest surface (1/w) within WARP_REACH of each group of eight
    // pixels of the static image, for the warp; worked out once per full frame rather than
    // by every in-between one.
    s_nearest: Vec<f32>,
    s_farthest: Vec<f32>,
    sky: Planes, // sky and clouds, quarter resolution
    sky_w: i32,
    sky_h: i32,
    history: Halves,
    history_next: Halves,
    // The static layer between full frames, on the last full frame's pixel centres: that
    // frame's fresh image (`centre`), what the resolve settled on there (`past`, without
    // players and bloom), and this frame's mix of the two (`blend`). In-between frames
    // warp `blend`, so a pixel on last frame's surface needs no history read at all: its
    // history is already in it, resampled once per full frame instead of every frame.
    centre: Halves,
    past: Halves,
    blend: Halves,
    // One byte per 8 x 8 tile: how many frames ago a player or its shadow was there (255:
    // long ago), for last frame and this one. The history there is not trusted unclamped.
    dyn_tiles: Vec<u8>,
    dyn_tiles_next: Vec<u8>,
    dyn_rows: Vec<u8>, // per row of tiles, 1 if any tile in it is recent
    dyn_rows_next: Vec<u8>,
    edges: Vec<u8>, // scratch for marking the static image's depth edges
    vig_x: Vec<f32>,
    // The bright pass the compose pass leaves, the blur's scratch, and the blurred
    // result, which the next frame adds in.
    bloom_a: Planes,
    bloom_b: Planes,
    bloom: Planes,
    bloom_w: i32,
    bloom_h: i32,
    out: [Vec<u8>; 3],
    pending: ShadowSlot,
    pub stats: Stats,
    // Pixels [x0, y0, x1, y1) the static raster covers: the whole frame, but for overlay mode.
    clip: [i32; 4],
    // Which frame-sized buffers are there: replaying, nothing is drawn, and they go.
    allocated: Held,
    // Overlay mode's per-rectangle buffers, kept from frame to frame.
    overlay_scratch: Vec<f32>,
    overlay_cover: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static MASKS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}


// Mip levels are rounded, as GPUs pick them, not truncated: truncated, a texture showing up
// to two texels a pixel still sampled its full-size level, and at a distance grass and
// flowers sparkled from one full render to the next.
const MIP_ROUND: f32 = 0.5;

// Frames a tile stays untrusted after a player or its shadow left it. Remembering only
// last frame, a pixel a player had left two frames before blended unclamped on the next
// full frame while its history still held some of the player, and that faint figure was
// then shown for the whole cycle after.
const DYN_MEMORY: u8 = 16;

const ALPHA_FULL: f32 = 0.2;
const ALPHA_BETWEEN: f32 = 0.1;
// A pixel away from depth edges and players blends gently and unclamped, at the same
// weight on every frame. Fine texture and shadow edges come out slightly different from
// each full render; blended at the weights above and clamped to the newest one, each
// new version replaced the last in a single step, so detail shimmered every 6th frame.
// Only next to a depth edge can a surface have just come into view, so everywhere else
// the history is of the same surface and needs no clamp.
const ALPHA_STILL: f32 = 0.06;
// Relative 1/w spread within two pixels that makes a depth edge.
const EDGE_SPREAD: f32 = 0.04;

// Rows per band of the compose pass. A multiple of 8, so each band owns whole rows of
// the eighth-size bloom it feeds.
const COMPOSE_ROWS: i32 = 24;

/// A player's body quad, for tracing its shadow.
#[derive(Clone, Copy)]
struct Blocker {
    p: Vec3,
    n: Vec3,
    u1: Vec3,
    u2: Vec3,
}

/// Where one change can show: the changed blocks grown by one, the sun-space rectangle
/// and top of their shadow, and the tiles (inclusive) that can hold either.
struct Zone {
    lo: Vec3,
    hi: Vec3,
    su: [f32; 2],
    sv: [f32; 2],
    top: f32,
    tx: (i32, i32),
    ty: (i32, i32),
}

/// Whether the segment from a to b passes through the box.
fn segment_hits(a: Vec3, b: Vec3, lo: Vec3, hi: Vec3) -> bool {
    let (a, d, lo, hi) = ([a.x, a.y, a.z], [b.x - a.x, b.y - a.y, b.z - a.z], [lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for k in 0..3 {
        if d[k].abs() < 1e-9 {
            if a[k] < lo[k] || a[k] > hi[k] {
                return false;
            }
            continue;
        }
        let (e, f) = ((lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]);
        t0 = fmax(t0, fmin(e, f));
        t1 = fmin(t1, fmax(e, f));
        if t0 > t1 {
            return false;
        }
    }
    true
}

/// Where one player's shadow can land this frame: a world box and its screen rectangle.
struct Caster {
    lo: Vec3,
    hi: Vec3,
    // The body's own box, grown by the most the per-pixel jitter of the sun direction
    // can move a ray over its length.
    body_lo: Vec3,
    body_hi: Vec3,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    near: Vec<Blocker>,
}

impl Renderer {
    pub fn new(assets: Arc<Assets>, jobs: Arc<Jobs>, width: i32, height: i32) -> Result<Renderer, String> {
        let (w, h) = (width, height);
        if w % 16 != 0 || h % 8 != 0 {
            return Err("the width must be a multiple of 16 and the height of 8".into());
        }
        let px = (w * h) as usize;
        let bands = (h + BAND - 1) / BAND;
        let (sky_w, sky_h) = (w / 4 + 3, h / 4 + 3);
        let (hiz_w, hiz_h) = ((w + HIZ - 1) / HIZ, (h + HIZ - 1) / HIZ);
        // An eighth of the frame: at 1080p the glow keeps the size on screen it had when
        // frames were rendered at 540p with a quarter-size bloom.
        let (bloom_w, bloom_h) = (w / 8, h / 8);
        let vig_x = (0..w)
            .map(|x| {
                let vx = (x as f32 + 0.5) / w as f32 - 0.5;
                vx * vx
            })
            .collect();
        let (mut cloud_map, mut cloud_soft) = (Vec::new(), Vec::new());
        if let Some((cw, ch, rgba)) = load_png_rgba(&assets.path("assets/minecraft/textures/environment/clouds.png"))
            && cw == 256
            && ch == 256
        {
            cloud_map = (0..256 * 256).map(|i| (rgba[i * 4 + 3] > 0) as u32).collect();
            // Soft coverage for shadows: a wide box blur so shadows drift as gentle patches.
            let mut a: Vec<f32> = cloud_map.iter().map(|&c| c as f32).collect();
            let mut b = vec![0.0f32; 256 * 256];
            for _ in 0..3 {
                for y in 0..256 {
                    for x in 0..256 {
                        let mut s = 0.0f32;
                        for k in -2i32..=2 {
                            s += a[y * 256 + ((x as i32 + k) & 255) as usize];
                        }
                        b[y * 256 + x] = s / 5.0;
                    }
                }
                for y in 0..256 {
                    for x in 0..256 {
                        let mut s = 0.0f32;
                        for k in -2i32..=2 {
                            s += b[((y as i32 + k) & 255) as usize * 256 + x];
                        }
                        a[y * 256 + x] = s / 5.0;
                    }
                }
            }
            cloud_soft = a;
        }
        let slots = jobs.size() as usize + 1;
        let mut scene = Scene {
            assets,
            look: Look::default(),
            sky_lut: Vec::new(),
            sky_lut_planes: planes(0),
            cloud_map,
            cloud_soft,
            shadows: None,
            anim_frame: Vec::new(),
            frame: 0,
            cloud_drift: Vec2::new(0.0, 0.0),
            ray_origin: Vec3::ZERO,
        };
        scene.build_sky_lut();
        Ok(Renderer {
            jobs,
            scene,
            w,
            h,
            bands,
            view: View::default(),
            st: StaticView::default(),
            history_valid: false,
            hiz_valid: false,
            static_valid: false,
            prev_st: StaticView::default(),
            prev_static: false,
            last_full: 0,
            static_every: 2,
            credit: 4,
            changes: Vec::new(),
            shown: Vec::new(),
            table: PrimTable::default(),
            texels: TexPool::default(),
            sway: false,
            loop_seconds: 360.0,
            casters: Vec::new(),
            pool: (0..slots).map(|_| Lists::new(bands as usize)).collect(),
            depth: vec![0.0; px],
            depth_water: vec![0.0; px],
            id: vec![0; px],
            id_water: vec![0; px],
            shaded: halves(px),
            shaded_sun: vec![0.0; px],
            shaded_depth: vec![0.0; px],
            dyn_depth: vec![0.0; px],
            dyn_id: vec![NO_PRIM; px],
            dyn_color: halves(px),
            dyn_motion: [vec![0.0; px], vec![0.0; px]],
            dyn_band: vec![0; bands as usize],
            hiz: vec![0.0; (hiz_w * hiz_h) as usize],
            hiz_w,
            hiz_h,
            hiz_view_proj: Mat4::default(),
            s_color: halves(px),
            s_sun: vec![0.0; px],
            s_depth: vec![0.0; px],
            s_nearest: vec![0.0; px / 8],
            s_farthest: vec![0.0; px / 8],
            sky: planes((sky_w * sky_h) as usize),
            sky_w,
            sky_h,
            history: halves(px),
            history_next: halves(px),
            centre: halves(px),
            past: halves(px),
            blend: halves(px),
            dyn_tiles: vec![255; px / 64],
            dyn_tiles_next: vec![255; px / 64],
            dyn_rows: vec![0; h as usize / 8],
            dyn_rows_next: vec![0; h as usize / 8],
            edges: vec![0; px],
            vig_x,
            bloom_a: planes((bloom_w * bloom_h) as usize),
            bloom_b: planes((bloom_w * bloom_h) as usize),
            bloom: planes((bloom_w * bloom_h) as usize),
            bloom_w,
            bloom_h,
            out: planes(px).map(|_| vec![0; px]),
            pending: Arc::new(Mutex::new(Handover::default())),
            stats: Stats { full: true, ..Stats::default() },
            clip: [0, 0, w, h],
            allocated: Held::All,
            overlay_scratch: Vec::new(),
            overlay_cover: Vec::new(),
        })
    }

    /// Gives back the frame-sized buffers, a few hundred MB at 1080p, while nothing is drawn;
    /// the next frame drawn takes them again, starting afresh.
    pub fn release(&mut self) {
        if self.allocated == Held::None {
            return;
        }
        for v in [&mut self.depth, &mut self.depth_water, &mut self.shaded_sun, &mut self.shaded_depth, &mut self.dyn_depth,
                  &mut self.s_sun, &mut self.s_depth] {
            *v = Vec::new();
        }
        for v in [&mut self.id, &mut self.id_water, &mut self.dyn_id] {
            *v = Vec::new();
        }
        for p in [&mut self.shaded, &mut self.s_color, &mut self.history, &mut self.history_next, &mut self.centre, &mut self.past,
                  &mut self.blend] {
            *p = Default::default();
        }
        self.dyn_color = Default::default();
        self.dyn_motion = Default::default();
        self.edges = Vec::new();
        self.out = Default::default();
        self.allocated = Held::None;
    }

    /// Takes the frame-sized buffers back if `release` gave them away.
    fn ensure(&mut self) {
        if self.allocated == Held::All {
            return;
        }
        self.ensure_overlay();
        let px = (self.w * self.h) as usize;
        for v in [&mut self.shaded_sun, &mut self.shaded_depth, &mut self.s_sun, &mut self.s_depth] {
            *v = vec![0.0; px];
        }
        self.dyn_motion = [vec![0.0; px], vec![0.0; px]];
        for p in [&mut self.shaded, &mut self.s_color, &mut self.history, &mut self.history_next, &mut self.centre, &mut self.past,
                  &mut self.blend] {
            *p = halves(px);
        }
        self.edges = vec![0; px];
        self.allocated = Held::All;
        self.restart();
    }

    /// Gives back what overlay mode does not draw with, if it is held: drawing over
    /// recordings only, the whole frame's shading, history and resolve buffers wait idle.
    pub fn keep_overlay_only(&mut self) {
        if self.allocated != Held::All {
            return;
        }
        for v in [&mut self.shaded_sun, &mut self.shaded_depth, &mut self.s_sun, &mut self.s_depth] {
            *v = Vec::new();
        }
        self.dyn_motion = Default::default();
        for p in [&mut self.shaded, &mut self.s_color, &mut self.history, &mut self.history_next, &mut self.centre, &mut self.past,
                  &mut self.blend] {
            *p = Default::default();
        }
        self.edges = Vec::new();
        self.allocated = Held::Overlay;
    }

    /// Drawing over recordings, the frame-sized buffers are only drawn in (and read) around
    /// the players, and every pass writes what it reads first: the pages of them drawn in so
    /// far go back to the system, to come back where the players are next.
    pub fn release_overlay_pages(&mut self) {
        if self.allocated != Held::Overlay {
            return;
        }
        for v in [&mut self.depth, &mut self.depth_water, &mut self.dyn_depth] {
            sys::release_pages(v);
        }
        for v in [&mut self.id, &mut self.id_water, &mut self.dyn_id] {
            sys::release_pages(v);
        }
        for v in self.dyn_color.iter_mut() {
            sys::release_pages(v);
        }
        for v in self.out.iter_mut() {
            sys::release_pages(v);
        }
    }

    /// Only the buffers overlay mode draws with: the static and dynamic layers' depth and
    /// prims, the players' colour and the output, half of all of them. The shading, history
    /// and resolve planes are left out until a frame is rendered whole.
    fn ensure_overlay(&mut self) {
        if self.allocated != Held::None {
            return;
        }
        let px = (self.w * self.h) as usize;
        for v in [&mut self.depth, &mut self.depth_water, &mut self.dyn_depth] {
            *v = vec![0.0; px];
        }
        self.id = vec![0; px];
        self.id_water = vec![0; px];
        self.dyn_id = vec![NO_PRIM; px];
        self.dyn_color = halves(px);
        self.out = planes(px).map(|_| vec![0; px]);
        self.allocated = Held::Overlay;
    }

    pub fn set_look(&mut self, look: Look) {
        self.scene.look = look;
        self.scene.build_sky_lut();
    }

    /// Shadow maps for this look, built on other threads and swapped in between frames.
    pub fn shadows(&self) -> Shadows {
        Shadows { assets: self.scene.assets.clone(), to_sun: self.scene.look.sun_dir, pending: self.pending.clone() }
    }

    /// The frame as 8-bit sRGB planes, red, green and blue, width x height.
    pub fn output(&self) -> &[Vec<u8>; 3] {
        &self.out
    }

    /// The same, for the overlay to draw on before it goes out.
    pub fn output_mut(&mut self) -> &mut [Vec<u8>; 3] {
        &mut self.out
    }

    pub fn width(&self) -> i32 {
        self.w
    }

    pub fn height(&self) -> i32 {
        self.h
    }

    pub fn set_static_every(&mut self, n: i32) {
        self.static_every = n.max(1);
        self.credit = 2 * self.static_every;
    }

    /// Takes what the world worker handed over: new shadow maps and the changes they came
    /// with. Called together with taking the meshes, so the three always match.
    /// Lets the shadow maps go, until the next ones are installed and adopted.
    pub fn forget_shadows(&mut self) {
        self.scene.shadows = None;
    }

    pub fn adopt(&mut self) {
        let mut p = self.pending.lock().unwrap();
        if let Some(maps) = p.maps.take() {
            self.scene.shadows = Some(maps);
        }
        self.changes.append(&mut p.changes);
    }

    /// Where a world point lands in the frame just rendered: pixel x and y, and its
    /// distance along the view. None behind the camera.
    pub fn project(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let c = self.view.unjittered * Vec4::new(p.x, p.y, p.z, 1.0);
        (c.w >= NEAR).then(|| ((c.x / c.w * 0.5 + 0.5) * self.w as f32, (0.5 - c.y / c.w * 0.5) * self.h as f32, c.w))
    }

    /// Screen pixels a block spans, face on to the camera, at a distance along the view.
    pub fn pixels_per_block(&self, dist: f32) -> f32 {
        1.0 / (length(self.view.ray_x) * dist)
    }

    /// Where the camera of the frame just rendered stands.
    pub fn eye(&self) -> Vec3 {
        self.view.eye
    }

    /// What the last full frame showed, for telling later whether a change would show in
    /// it (sight.rs). Its depth pyramid is exact for that frame's camera.
    pub fn sight(&self) -> crate::sight::View {
        crate::sight::View::new(self.hiz_view_proj, self.w, self.h, &self.hiz, self.hiz_w as usize, self.hiz_h as usize,
                                HIZ as usize)
    }

    /// When the changes drawn since the last call reached the engine (engine clock).
    pub fn take_shown(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.shown)
    }

    // Swaying plants live in the static layer, so at a low static refresh they step rather
    // than sway; off unless the static layer is drawn every frame or two.
    pub fn set_sway(&mut self, on: bool) {
        self.sway = on;
    }

    pub fn set_casters(&mut self, casters: Vec<Footprint>) {
        self.casters = casters;
    }

    /// The seconds after which everything that moves by itself repeats: the orbit's period.
    pub fn set_loop(&mut self, seconds: f32) {
        self.loop_seconds = seconds;
    }

    /// Forgets every frame drawn so far, as after a pause: the next one is a full render
    /// with no history to blend or reproject.
    pub fn restart(&mut self) {
        self.history_valid = false;
        self.hiz_valid = false;
        self.static_valid = false;
        self.prev_static = false;
        self.credit = 2 * self.static_every;
        for t in [&mut self.dyn_tiles, &mut self.dyn_tiles_next] {
            t.fill(255);
        }
        for r in [&mut self.dyn_rows, &mut self.dyn_rows_next] {
            r.fill(0);
        }
        for p in self.bloom.iter_mut() {
            p.fill(0.0);
        }
    }

    /// Render one frame. `time` is seconds into the loop (set_loop) and drives everything
    /// that moves by itself, all of it back where it started when the loop comes round:
    /// the stream replays an unchanged orbit rather than render it again, and a replayed
    /// frame must meet a rendered one without a jump.
    pub fn render<'a>(&mut self, meshes: &'a [Arc<SectionMesh>], dynamic: &'a [Quad], pose: &CameraPose, time: f64) {
        self.ensure();
        let t0 = Instant::now();
        // Sub-pixel jitter, 16-sample Halton; the temporal resolve turns it into AA.
        let k = (self.scene.frame % 16) as i32 + 1;
        let (jx, jy) = (halton(k, 2) - 0.5, halton(k, 3) - 0.5);
        let setup = self.camera(pose, time, jx, jy);
        self.render_static(t0, meshes, dynamic, setup, jx, jy);
    }

    /// The camera, clouds, waves and animated textures for a frame at `time`, the
    /// projection jittered by (jx, jy) pixels; what prim setup needs.
    fn camera(&mut self, pose: &CameraPose, time: f64, jx: f32, jy: f32) -> Setup {
        let (w, h) = (self.w, self.h);
        let period = self.loop_seconds as f64;
        let turn = time / period * std::f64::consts::TAU;
        self.view.time = time;
        // The clouds circle at 0.9 blocks a second instead of drifting off east.
        let r = 0.9 * period / std::f64::consts::TAU;
        self.scene.cloud_drift = Vec2::new((r * turn.cos()) as f32, (r * turn.sin()) as f32);
        // Wave speeds rounded to whole cycles a loop.
        let phase = |rate: f64| ((rate * period / std::f64::consts::TAU).round() * turn % std::f64::consts::TAU) as f32;
        self.view.waves = [phase(1.1), -phase(1.7), -phase(0.9), phase(1.3)];
        self.view.eye = pose.eye;
        self.scene.ray_origin = pose.eye;
        let view = look_at(pose.eye, pose.target, Vec3::new(0.0, 1.0, 0.0));
        let mut proj = perspective(pose.fov_y, w as f32 / h as f32, NEAR, FAR);
        let unjittered = proj * view;
        self.view.jitter_x = jx;
        self.view.jitter_y = jy;
        proj.m[2][0] += jx * 2.0 / w as f32;
        proj.m[2][1] -= jy * 2.0 / h as f32;
        self.view.view_proj = proj * view;

        // Per-pixel world rays from the jittered projection.
        {
            let f = (pose.fov_y / 2.0).tan();
            let aspect = w as f32 / h as f32;
            let fwd = normalize(pose.target - pose.eye);
            let right = normalize(cross(fwd, Vec3::new(0.0, 1.0, 0.0)));
            let up = cross(right, fwd);
            self.view.ray_x = right * (2.0 * f * aspect / w as f32);
            self.view.ray_y = up * (-2.0 * f / h as f32);
            // The jittered raster samples unjittered pixel (px + jx, py + jy).
            self.view.ray_base =
                fwd + right * (f * aspect * (2.0 * jx / w as f32 - 1.0)) + up * (f * (1.0 - 2.0 * jy / h as f32));
        }

        {
            let assets = &self.scene.assets;
            let anim = &mut self.scene.anim_frame;
            anim.clear();
            anim.resize(assets.texture_count(), 0);
            // 64-bit: game ticks since 1970 no longer fit an int.
            let ticks = (time * 20.0) as i64;
            for (i, frame) in anim.iter_mut().enumerate() {
                let t = assets.tex(i as u16);
                if t.frames.len() > 1 && !t.frame_order.is_empty() {
                    let step = (ticks / t.frame_time.max(1) as i64) % t.frame_order.len() as i64;
                    *frame = iclamp(t.frame_order[step as usize], 0, t.frames.len() as i32 - 1);
                }
            }
        }

        let vp = self.view.view_proj;
        let mut setup = Setup {
            mx: [0.0; 4],
            my: [0.0; 4],
            mw: [0.0; 4],
            eye: pose.eye,
            time,
            sway: self.sway,
            w,
            h,
            clip: [0.0, 0.0, w as f32, h as f32],
        };
        for c in 0..4 {
            setup.mx[c] = vp.m[c][0] * w as f32 * 0.5 + vp.m[c][3] * w as f32 * 0.5;
            setup.my[c] = -vp.m[c][1] * h as f32 * 0.5 + vp.m[c][3] * h as f32 * 0.5;
            setup.mw[c] = vp.m[c][3];
        }
        self.view.unjittered = unjittered;
        setup
    }

    #[allow(clippy::too_many_arguments)]
    fn render_static<'a>(&mut self, t0: Instant, meshes: &'a [Arc<SectionMesh>], dynamic: &'a [Quad], setup: Setup, jx: f32, jy: f32) {
        let mut lists: Vec<Lists<'a>> = std::mem::take(&mut self.pool);
        let workers = self.jobs.size() as usize;
        let age = self.scene.frame.wrapping_sub(self.last_full) as i32;
        self.credit = (self.credit + 1).min(2 * self.static_every);
        let due = !self.static_valid || age >= self.static_every;
        // A block someone placed is drawn now rather than at the next scheduled full
        // render, as long as full renders stay one in static_every on average: a burst of
        // building brings one or two forward, then the cadence holds.
        let early = !self.changes.is_empty() && age >= 2 && self.credit >= self.static_every;
        let full = due || early;
        if full {
            self.credit -= self.static_every;
        }
        self.stats.passes = [0.0; PASSES.len()];
        self.stats.full = full;
        let mut lap = Instant::now();
        let (t1, t2);
        if full {
            self.setup_prims(meshes, &setup, &mut lists[..workers]);
            self.lap(0, &mut lap);
            let t = Instant::now();
            t1 = t;
            self.stats.setup = since(t0);
            self.raster(&lists, false, false);
            self.raster(&lists, true, false);
            self.lap(1, &mut lap);
            self.build_hiz();
            self.lap(2, &mut lap);
            t2 = Instant::now();
            self.stats.raster = since(t1);
            self.draw_sky();
            self.lap(3, &mut lap);
            self.shade(&lists);
            // The frame just shaded is the new static image; the old one's buffers take
            // the next shading pass.
            std::mem::swap(&mut self.shaded, &mut self.s_color);
            std::mem::swap(&mut self.shaded_depth, &mut self.s_depth);
            std::mem::swap(&mut self.shaded_sun, &mut self.s_sun);
            self.mark_edges();
            self.reach_depths();
            self.lap(4, &mut lap);
            (self.prev_st, self.prev_static) = (self.st, self.static_valid);
            self.st = StaticView {
                jitter_x: jx,
                jitter_y: jy,
                ray_base: self.view.ray_base,
                ray_x: self.view.ray_x,
                ray_y: self.view.ray_y,
                eye: self.view.eye,
            };
            self.static_valid = true;
            self.last_full = self.scene.frame;
            self.seed_changes();
        } else {
            self.stats.setup = 0.0;
            t1 = Instant::now();
            t2 = t1;
            self.stats.raster = 0.0;
        }
        // Players and other moving things, every frame.
        {
            let dyn_lists = &mut lists[workers];
            dyn_lists.clear();
            for q in dynamic {
                add_prim(&setup, &self.scene.assets, q, q.layer == TRANSLUCENT, dyn_lists);
            }
        }
        self.raster(&lists, false, true);
        self.shade_dynamic(&lists);
        let casters = self.casters(dynamic);
        self.lap(5, &mut lap);
        let t3 = Instant::now();
        self.stats.shade = since(t2);
        self.compose(full, &casters);
        self.lap(6, &mut lap);
        self.blur_bloom();
        self.lap(7, &mut lap);
        std::mem::swap(&mut self.history, &mut self.history_next);
        std::mem::swap(&mut self.dyn_tiles, &mut self.dyn_tiles_next);
        std::mem::swap(&mut self.dyn_rows, &mut self.dyn_rows_next);
        self.history_valid = true;
        self.view.prev_view_proj = self.view.unjittered;
        self.stats.post = since(t3);
        self.scene.frame = self.scene.frame.wrapping_add(1);
        self.pool = lists.into_iter().map(Lists::recycle).collect();
    }

    /// Marks the 8 x 8 tiles where the changes just drawn show: pixels whose view ray
    /// passes through a changed block, or near one, and those in its shadow. Those tiles
    /// take the new render at once, as around players, instead of easing into it over a
    /// second as still pixels do.
    fn seed_changes(&mut self) {
        if self.changes.is_empty() {
            return;
        }
        let mut changes = std::mem::take(&mut self.changes);
        self.shown.extend(changes.iter().map(|c| c.arrived));
        if changes.len() > 64 {
            let lo = changes.iter().fold(Vec3::splat(1e30), |a, c| vmin(a, c.lo));
            let hi = changes.iter().fold(Vec3::splat(-1e30), |a, c| vmax(a, c.hi));
            changes = vec![Changed { lo, hi, arrived: 0.0 }];
        }
        let (w, h) = (self.w, self.h);
        let sun = self.scene.look.sun_dir;
        let right = normalize(cross(if sun.y.abs() > 0.99 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(0.0, 1.0, 0.0) }, sun));
        let up = cross(sun, right);
        let vp = self.view.view_proj;
        let zones: Vec<Zone> = changes
            .iter()
            .map(|c| {
                // A block's faces, and the ambient occlusion and light of its neighbours'.
                let (lo, hi) = (c.lo - Vec3::splat(1.0), c.hi + Vec3::splat(1.0));
                let (mut su, mut sv, mut top) = ([1e30f32, -1e30f32], [1e30f32, -1e30f32], -1e30f32);
                let (mut min_x, mut min_y, mut max_x, mut max_y) = (1e30f32, 1e30f32, -1e30f32, -1e30f32);
                let mut behind = false;
                for k in 0..16 {
                    // The box, and the box moved away from the sun: its shadow lies between, on
                    // ground up to 56 blocks below it.
                    let v = corner(lo, hi, k & 7) - if k < 8 { Vec3::ZERO } else { sun * 96.0 };
                    if k < 8 {
                        let (u, t) = (dot(v, right), dot(v, up));
                        su = [fmin(su[0], u), fmax(su[1], u)];
                        sv = [fmin(sv[0], t), fmax(sv[1], t)];
                        top = fmax(top, dot(v, sun));
                    }
                    let cl = vp * Vec4::new(v.x, v.y, v.z, 1.0);
                    if cl.w < NEAR {
                        behind = true;
                        continue;
                    }
                    min_x = fmin(min_x, (cl.x / cl.w * 0.5 + 0.5) * w as f32);
                    max_x = fmax(max_x, (cl.x / cl.w * 0.5 + 0.5) * w as f32);
                    min_y = fmin(min_y, (0.5 - cl.y / cl.w * 0.5) * h as f32);
                    max_y = fmax(max_y, (0.5 - cl.y / cl.w * 0.5) * h as f32);
                }
                let (tiles_x, tiles_y) = (w / 8, h / 8);
                let span = |a: f32, b: f32, n: i32| {
                    if behind { (0, n - 1) } else { (iclamp(a as i32 / 8 - 1, 0, n - 1), iclamp(b as i32 / 8 + 1, 0, n - 1)) }
                };
                // The shadow map's texels, the sun's per-pixel jitter and the edge its
                // filter softens, as a margin across the sun.
                let m = 0.6;
                Zone { lo, hi, su: [su[0] - m, su[1] + m], sv: [sv[0] - m, sv[1] + m], top, tx: span(min_x, max_x, tiles_x), ty: span(min_y, max_y, tiles_y) }
            })
            .collect();
        let st = self.st;
        let tw = (w / 8) as usize;
        let rows = Parts::chunks(&mut self.dyn_tiles, tw);
        let marked = Parts::chunks(&mut self.dyn_rows, 1);
        let depth = &self.s_depth;
        self.jobs.run(h / 8, |ty, _| {
            let mut tiles = rows.take(ty as usize);
            let mut any = false;
            for z in zones.iter().filter(|z| (z.ty.0..=z.ty.1).contains(&ty)) {
                for tx in z.tx.0..=z.tx.1 {
                    let hit = (0..16).any(|k| {
                        let (x, y) = (tx * 8 + 1 + 2 * (k & 3), ty * 8 + 1 + 2 * (k >> 2));
                        let a = depth[(y * w + x) as usize];
                        let ray = st.ray_base + st.ray_x * (x as f32 + 0.5) + st.ray_y * (y as f32 + 0.5);
                        let p = st.eye + ray * if a > 0.0 { 1.0 / a } else { FAR };
                        if segment_hits(st.eye, p, z.lo, z.hi) {
                            return true;
                        }
                        let (u, v) = (dot(p, right), dot(p, up));
                        a > 0.0 && dot(p, sun) < z.top && (z.su[0]..=z.su[1]).contains(&u) && (z.sv[0]..=z.sv[1]).contains(&v)
                    });
                    if hit {
                        tiles[tx as usize] = 0;
                        any = true;
                    }
                }
            }
            if any {
                marked.take(ty as usize)[0] = 1;
            }
        });
    }

    fn lap(&mut self, pass: usize, since_last: &mut Instant) {
        let now = Instant::now();
        self.stats.passes[pass] += (now - *since_last).as_secs_f64() * 1000.0;
        *since_last = now;
    }

    fn setup_prims<'a>(&mut self, meshes: &'a [Arc<SectionMesh>], setup: &Setup, lists: &mut [Lists<'a>]) {
        let (w, h) = (self.w, self.h);
        // Frustum planes from the (unjittered is fine) view-projection.
        let m = &self.view.view_proj;
        let (r0, r1, r2, r3) = (row(m, 0), row(m, 1), row(m, 2), row(m, 3));
        let add = |a: Vec4, b: Vec4, s: f32| Vec4::new(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z, a.w + s * b.w);
        let planes = [add(r3, r0, 1.0), add(r3, r0, -1.0), add(r3, r1, 1.0), add(r3, r1, -1.0), add(r3, r2, 1.0),
                      add(r3, r2, -1.0)];
        let visible = |lo: Vec3, hi: Vec3| {
            planes.iter().all(|p| {
                let v = Vec3::new(if p.x > 0.0 { hi.x } else { lo.x }, if p.y > 0.0 { hi.y } else { lo.y },
                                  if p.z > 0.0 { hi.z } else { lo.z });
                p.x * v.x + p.y * v.y + p.z * v.z + p.w >= 0.0
            })
        };

        let hw = w as f32 * 0.5;
        let hh = h as f32 * 0.5;
        // Occlusion against last frame's depth: the camera moves well under a pixel per
        // frame, so what hid a section then still hides it now. The test is done entirely
        // in last frame's view, where the depth pyramid is exact, with a margin of a block.
        let hm = &self.hiz_view_proj.m;
        let (mut pmx, mut pmy, mut pmw) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        for c in 0..4 {
            pmx[c] = hm[c][0] * hw + hm[c][3] * hw;
            pmy[c] = -hm[c][1] * hh + hm[c][3] * hh;
            pmw[c] = hm[c][3];
        }
        let (hiz, hiz_w, hiz_h, hiz_valid) = (&self.hiz, self.hiz_w, self.hiz_h, self.hiz_valid);
        let occluded = |m: &SectionMesh| {
            if !hiz_valid {
                return false;
            }
            let (mut min_x, mut min_y, mut max_x, mut max_y, mut nearest) = (1e30f32, 1e30f32, -1e30f32, -1e30f32, 0.0f32);
            for k in 0..8 {
                let c = corner(m.lo, m.hi, k);
                let wv = pmw[0] * c.x + pmw[1] * c.y + pmw[2] * c.z + pmw[3];
                if wv < NEAR * 4.0 {
                    return false;
                }
                let iw = 1.0 / wv;
                let x = (pmx[0] * c.x + pmx[1] * c.y + pmx[2] * c.z + pmx[3]) * iw;
                let y = (pmy[0] * c.x + pmy[1] * c.y + pmy[2] * c.z + pmy[3]) * iw;
                min_x = fmin(min_x, x);
                max_x = fmax(max_x, x);
                min_y = fmin(min_y, y);
                max_y = fmax(max_y, y);
                nearest = fmax(nearest, iw);
            }
            let bx0 = 0.max(min_x as i32 / HIZ - 1);
            let bx1 = (hiz_w - 1).min(max_x as i32 / HIZ + 1);
            let by0 = 0.max(min_y as i32 / HIZ - 1);
            let by1 = (hiz_h - 1).min(max_y as i32 / HIZ + 1);
            if bx0 > bx1 || by0 > by1 || min_x < 0.0 || min_y < 0.0 || max_x >= w as f32 || max_y >= h as f32 {
                return false;
            }
            nearest *= 1.03;
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    if hiz[(by * hiz_w + bx) as usize] <= nearest {
                        return false;
                    }
                }
            }
            true
        };

        // Overlay mode draws only a rectangle: sections wholly in front of the camera that
        // project outside it are left out.
        let clipped = setup.clip != [0.0, 0.0, w as f32, h as f32];
        let outside = |lo: Vec3, hi: Vec3| {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for k in 0..8 {
                let c = corner(lo, hi, k);
                let wv = setup.mw[0] * c.x + setup.mw[1] * c.y + setup.mw[2] * c.z + setup.mw[3];
                if wv < NEAR {
                    return false;
                }
                let x = (setup.mx[0] * c.x + setup.mx[1] * c.y + setup.mx[2] * c.z + setup.mx[3]) / wv;
                let y = (setup.my[0] * c.x + setup.my[1] * c.y + setup.my[2] * c.z + setup.my[3]) / wv;
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
            x1 < setup.clip[0] || y1 < setup.clip[1] || x0 > setup.clip[2] || y0 > setup.clip[3]
        };
        let mut vis: Vec<&'a SectionMesh> = Vec::with_capacity(meshes.len());
        let mut hidden = 0;
        for m in meshes {
            if !visible(m.lo, m.hi) || clipped && outside(m.lo, m.hi) {
                continue;
            }
            if occluded(m) {
                hidden += 1;
                continue;
            }
            vis.push(m);
        }
        self.stats.occluded = hidden;
        self.stats.sections = vis.len();

        lists.iter_mut().for_each(Lists::clear);
        let assets = &*self.scene.assets;
        let eye = self.view.eye;
        let slots: Vec<Mutex<&mut Lists<'a>>> = lists.iter_mut().map(Mutex::new).collect();
        self.jobs.run(vis.len() as i32, |ci, worker| {
            let mut out = slots[worker as usize].try_lock().expect("a worker's lists are its own");
            let m = vis[ci as usize];
            for g in 0..7 {
                if g < 6 {
                    let axis = match g / 2 {
                        0 => 1,
                        1 => 2,
                        _ => 0,
                    };
                    let seen = if g & 1 != 0 { eye[axis] > m.group_plane[g] } else { eye[axis] < m.group_plane[g] };
                    if !seen {
                        continue;
                    }
                }
                for q in &m.quads[m.group_start[g] as usize..m.group_start[g + 1] as usize] {
                    add_prim(setup, assets, q, false, &mut out);
                }
            }
            for q in &m.water {
                add_prim(setup, assets, q, true, &mut out);
            }
        });
        drop(slots);
        self.stats.prims = lists.iter().map(|l| l.prims.len()).sum();
    }

    fn raster(&mut self, lists: &[Lists], translucent: bool, dynamic: bool) {
        let Renderer {
            ref jobs,
            ref scene,
            w,
            h,
            bands,
            ref mut depth,
            ref mut depth_water,
            ref mut id,
            ref mut id_water,
            ref mut dyn_depth,
            ref mut dyn_id,
            ref mut dyn_band,
            clip,
            ..
        } = *self;
        let (zbuf, ibuf, opaque) = if dynamic {
            (&mut dyn_depth[..], &mut dyn_id[..], None)
        } else if translucent {
            (&mut depth_water[..], &mut id_water[..], Some(&depth[..]))
        } else {
            (&mut depth[..], &mut id[..], None)
        };
        let slot = jobs.size() as usize;
        let workers = if dynamic { slot..=slot } else { 0..=slot - 1 };
        if dynamic {
            for (b, flag) in dyn_band.iter_mut().enumerate() {
                *flag = !lists[slot].bands[b].is_empty() as u8;
            }
        }
        let zs = Parts::chunks(zbuf, (BAND * w) as usize);
        let is = Parts::chunks(ibuf, (BAND * w) as usize);
        let assets = &*scene.assets;
        let anim = &scene.anim_frame[..];
        let dyn_band = &dyn_band[..];
        let clip = clip;
        jobs.run(bands, |b, _| {
            if dynamic && dyn_band[b as usize] == 0 || (b + 1) * BAND <= clip[1] || b * BAND >= clip[3] {
                return;
            }
            let mut z = zs.take(b as usize);
            let mut i = is.take(b as usize);
            raster_band(b, lists, workers.clone(), &mut z, &mut i, opaque, translucent, assets, anim, w, h, (clip[0], clip[2]));
        });
    }

    fn build_hiz(&mut self) {
        // Depth pyramid level for the next occlusion test: the farthest opaque depth in
        // each HIZ x HIZ block (sky counts as infinitely far, so it never occludes).
        let Renderer { ref jobs, w, h, ref depth, ref mut hiz, hiz_w, hiz_h, .. } = *self;
        let rows = Parts::chunks(hiz, hiz_w as usize);
        jobs.run(hiz_h, |by, _| {
            let mut out = rows.take(by as usize);
            for bx in 0..hiz_w {
                let mut far = 1e30f32;
                for y in by * HIZ..h.min((by + 1) * HIZ) {
                    let row = &depth[(y * w) as usize..];
                    for x in bx * HIZ..w.min((bx + 1) * HIZ) {
                        far = fmin(far, row[x as usize]);
                    }
                }
                out[bx as usize] = far;
            }
        });
        self.hiz_view_proj = self.view.view_proj;
        self.hiz_valid = true;
    }

    fn draw_sky(&mut self) {
        // Sky and clouds at a quarter of the render size; sky pixels interpolate it.
        let Renderer { ref jobs, ref scene, ref view, ref mut sky, sky_w, sky_h, .. } = *self;
        let rows = sky.each_mut().map(|p| Parts::chunks(p, sky_w as usize));
        jobs.run(sky_h, |j, _| {
            let mut out = rows.each_ref().map(|p| p.take(j as usize));
            for i in 0..sky_w {
                let px = (i as f32 - 0.5) * 4.0;
                let py = (j as f32 - 0.5) * 4.0;
                let c = scene.sky_color(normalize(view.ray_base + view.ray_x * px + view.ray_y * py), true);
                (out[0][i as usize], out[1][i as usize], out[2][i as usize]) = (c.x, c.y, c.z);
            }
        });
    }

    fn shade(&mut self, lists: &[Lists]) {
        let workers = self.jobs.size() as usize;
        self.table.build(&lists[..workers], &self.jobs);
        self.texels.update(&self.scene.assets, &self.scene.anim_frame);
        let Renderer {
            ref jobs,
            ref scene,
            ref view,
            w,
            h,
            ref id,
            ref id_water,
            ref depth,
            ref depth_water,
            ref sky,
            sky_w,
            ref table,
            ref texels,
            ref mut shaded,
            ref mut shaded_sun,
            ref mut shaded_depth,
            ..
        } = *self;
        let bands = h / 8;
        let (wu, hu, nb) = (w as usize, h as usize, bands as usize);
        let colors = shaded.each_mut().map(|p| Parts::bands(p, wu, hu, nb));
        let suns = Parts::bands(shaded_sun, wu, hu, nb);
        let depths = Parts::bands(shaded_depth, wu, hu, nb);
        let ctx = ShadeCtx {
            table,
            pool: texels,
            lut: lut(),
            maps: scene.shadows.as_deref().map(|m| &m[..]).unwrap_or(&[]),
            cloud: &scene.cloud_soft,
            sky_lut: &scene.sky_lut_planes,
            sky,
            sky_w,
            look: scene.look,
            eye: view.eye,
            ray_base: view.ray_base,
            ray_x: view.ray_x,
            ray_y: view.ray_y,
            frame: scene.frame,
            cloud_drift: scene.cloud_drift,
        };
        jobs.run(bands, |b, _| {
            let (y0, y1) = (b * h / bands, (b + 1) * h / bands);
            let mut color = colors.each_ref().map(|p| p.take(b as usize));
            let (mut sun_vis, mut depth_out) = (suns.take(b as usize), depths.take(b as usize));
            let mut rgb = vec![0.0f32; 4 * wu];
            for y in y0..y1 {
                let (row, orow) = (y as usize * wu, (y - y0) as usize * wu);
                let (cr, rest) = rgb.split_at_mut(wu);
                let (cg, rest) = rest.split_at_mut(wu);
                let (cb, dist) = rest.split_at_mut(wu);
                let sun = &mut sun_vis[orow..orow + wu];
                let near = &mut depth_out[orow..orow + wu];
                let ids = &id[row..row + wu];
                let water = &id_water[row..row + wu];
                shade_row(&ctx, y, ids, water, &depth[row..row + wu], &depth_water[row..row + wu], [&mut *cr, &mut *cg, &mut *cb], sun, near,
                          dist);
                // Water over whatever is below it, one pixel at a time: it is a small part
                // of the frame, and its shading is its own.
                for x in 0..wu {
                    if water[x] == NO_PRIM {
                        continue;
                    }
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let ray = normalize(view.ray_base + view.ray_x * px + view.ray_y * py);
                    let below = Vec3::new(cr[x], cg[x], cb[x]);
                    let c = scene.water_over(lists, water[x], px, py, ray, below, dist[x], ids[x] == NO_PRIM, view.waves);
                    (cr[x], cg[x], cb[x]) = (c.x, c.y, c.z);
                    sun[x] = 0.5;
                }
                for (plane, src) in color.iter_mut().zip([&*cr, &*cg, &*cb]) {
                    let dst = plane[orow..orow + wu].as_chunks_mut::<8>().0;
                    for (d, v) in dst.iter_mut().zip(src.as_chunks::<8>().0) {
                        *d = F8::load(v).to_f16();
                    }
                }
            }
        });
    }

    fn shade_dynamic(&mut self, lists: &[Lists]) {
        let Renderer {
            ref jobs,
            ref scene,
            ref view,
            w,
            h,
            bands,
            ref dyn_id,
            ref dyn_band,
            ref mut dyn_color,
            ref mut dyn_motion,
            clip,
            ..
        } = *self;
        let slot = jobs.size();
        let wu = w as usize;
        let band_len = (BAND * w) as usize;
        let colors = dyn_color.each_mut().map(|p| Parts::chunks(p, band_len));
        // Overlay mode keeps no motion planes: only the full frame's resolve reads them.
        let motions = (!dyn_motion[0].is_empty()).then(|| dyn_motion.each_mut().map(|p| Parts::chunks(p, band_len)));
        let l = &scene.look;
        jobs.run(bands, |b, _| {
            if dyn_band[b as usize] == 0 {
                return;
            }
            let mut color = colors.each_ref().map(|p| p.take(b as usize));
            let mut motion = motions.as_ref().map(|m| m.each_ref().map(|p| p.take(b as usize)));
            let y0 = b * BAND;
            // Overlay mode rasterises (and clears) only the clip's columns.
            for y in y0..h.min(y0 + BAND) {
                let py = y as f32 + 0.5;
                for x in clip[0]..clip[2] {
                    let i = (y * w + x) as usize;
                    let pid = dyn_id[i];
                    if pid == NO_PRIM || (pid >> 26) != slot {
                        continue;
                    }
                    let o = (y - y0) as usize * wu + x as usize;
                    let px = x as f32 + 0.5;
                    let p = prim(lists, pid);
                    let a = p.a[0] * px + p.a[1] * py + p.a[2];
                    let wv = 1.0 / a;
                    let s = clamp01((p.s[0] * px + p.s[1] * py + p.s[2]) * wv);
                    let t = clamp01((p.t[0] * px + p.t[1] * py + p.t[2]) * wv);
                    let pos = p.at(s, t);
                    let (mut c, _) = scene.lit_surface(p, s, t, wv, pos, x, y);
                    let to_pos = pos - view.eye;
                    let dist = length(to_pos);
                    let fog = smoothstep(l.fog_start, l.fog_end, dist);
                    if fog > 0.0 {
                        c = mix3(c, scene.sky_gradient(to_pos / dist), fog);
                    }
                    (color[0][o], color[1][o], color[2][o]) = (to_half(c.x), to_half(c.y), to_half(c.z));
                    let Some(motion) = motion.as_mut() else { continue };
                    let m = view.prev_view_proj * Vec4::new(pos.x, pos.y, pos.z, 1.0);
                    (motion[0][o], motion[1][o]) = if m.w > 0.0 {
                        let inv = 1.0 / m.w;
                        ((m.x * inv * 0.5 + 0.5) * w as f32 - 0.5, (0.5 - m.y * inv * 0.5) * h as f32 - 0.5)
                    } else {
                        (-1.0, -1.0)
                    };
                }
            }
        });
    }

    /// Players shadow the ground in their true silhouette: for each, the world box its
    /// shadow can fall in, that box's screen rectangle, and the body quads to trace.
    fn casters(&self, dynamic: &[Quad]) -> Vec<Caster> {
        if self.casters.is_empty() {
            return Vec::new();
        }
        let mut quads = Vec::new();
        for q in dynamic {
            if q.mat & MAT_ENTITY == 0 || q.layer != SOLID {
                continue;
            }
            let n = cross(q.e1, q.e2);
            let a = cross(q.e2, n);
            let b = cross(n, q.e1);
            let da = dot(q.e1, a);
            let db = dot(q.e2, b);
            if da.abs() < 1e-12 || db.abs() < 1e-12 {
                continue;
            }
            let corners = [q.p, q.p + q.e1, q.p + q.e2, q.p + q.e1 + q.e2];
            let lo = corners.into_iter().fold(Vec3::splat(1e30), vmin);
            let hi = corners.into_iter().fold(Vec3::splat(-1e30), vmax);
            quads.push((Blocker { p: q.p, n, u1: a / da, u2: b / db }, lo, hi));
        }
        let (view, w, h) = (&self.view, self.w, self.h);
        let sun = self.scene.look.sun_dir;
        let mut out = Vec::new();
        for c in &self.casters {
            let mut lo = c.pos - Vec3::new(0.6, 3.0, 0.6);
            let mut hi = c.pos + Vec3::new(0.6, 2.1, 0.6);
            let reach = 5.0 / fmax(0.2, sun.y);
            let far = sun * -reach;
            lo = vmin(lo, lo + far);
            hi = vmax(hi, hi + far);
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (1e30f32, 1e30f32, -1e30f32, -1e30f32);
            let mut behind = false;
            for k in 0..8 {
                let v = corner(lo, hi, k);
                let cl = view.view_proj * Vec4::new(v.x, v.y, v.z, 1.0);
                if cl.w < NEAR {
                    behind = true;
                    break;
                }
                let x = (cl.x / cl.w * 0.5 + 0.5) * w as f32;
                let y = (0.5 - cl.y / cl.w * 0.5) * h as f32;
                min_x = fmin(min_x, x);
                max_x = fmax(max_x, x);
                min_y = fmin(min_y, y);
                max_y = fmax(max_y, y);
            }
            if behind {
                continue;
            }
            let (x0, x1) = (0.max(min_x as i32), (w - 1).min(max_x as i32 + 1));
            let (y0, y1) = (0.max(min_y as i32), (h - 1).min(max_y as i32 + 1));
            let body: Vec<_> = quads
                .iter()
                .filter(|(q, _, _)| (q.p.x - c.pos.x).abs() < 1.5 && (q.p.z - c.pos.z).abs() < 1.5 && q.p.y > c.pos.y - 0.5)
                .collect();
            if x0 <= x1 && y0 <= y1 && !body.is_empty() {
                // The jitter turns the sun by up to 0.035 either way, 0.42 over the 12
                // blocks a shadow ray reaches.
                let grow = Vec3::splat(0.45);
                let body_lo = body.iter().fold(Vec3::splat(1e30), |a, b| vmin(a, b.1)) - grow;
                let body_hi = body.iter().fold(Vec3::splat(-1e30), |a, b| vmax(a, b.2)) + grow;
                let near = body.iter().map(|b| b.0).collect();
                out.push(Caster { lo, hi, body_lo, body_hi, x0, x1, y0, y1, near });
            }
        }
        out
    }

    /// This in-between frame's static layer: the full frame's image, with what the resolve
    /// had there fading out at the rate a still pixel blends, (1 - ALPHA_STILL) a frame.
    /// The pass reads and writes 36 MB, so after the first in-between frame each one redoes
    /// only every second row, alternating: a pixel steps every second frame, but the picture
    /// as a whole changes a little every frame instead of ticking at half the frame rate.
    fn mix_static(&mut self) {
        let k = self.scene.frame.wrapping_sub(self.last_full) as i32;
        let (first, step) = if k == 1 { (0, 1) } else { ((k % 2) as usize, 2) };
        let f = F8::splat((1.0 - ALPHA_STILL).powi(k + 1));
        let Renderer { ref jobs, ref centre, ref past, ref mut blend, w, h, .. } = *self;
        let (w, len) = (w as usize, COMPOSE_ROWS as usize * w as usize);
        let parts = blend.each_mut().map(|p| Parts::chunks(p, len));
        jobs.run((h + COMPOSE_ROWS - 1) / COMPOSE_ROWS, |b, _| {
            let at = b as usize * len;
            for ((part, c), p) in parts.iter().zip(centre).zip(past) {
                let mut out = part.take(b as usize);
                let rows = out.len() / w;
                for r in (first..rows).step_by(step) {
                    let (c, p) = (&c[at + r * w..][..w], &p[at + r * w..][..w]);
                    for (i, o) in out[r * w..][..w].as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let c = F8::load_f16(c[8 * i..][..8].try_into().unwrap());
                        let p = F8::load_f16(p[8 * i..][..8].try_into().unwrap());
                        *o = (p - c).mul_add(f, c).to_f16();
                    }
                }
            }
        });
    }

    /// Everything after shading, one band of rows at a time so it stays in cache: the
    /// static image (as shaded on a full frame, warped to this camera otherwise), players
    /// and their shadows over it, bloom, the temporal resolve, tonemap, grade and dithered
    /// sRGB out. It also leaves this frame's bright pass for the next frame's bloom: a
    /// frame late, which at this camera speed nothing shows.
    fn compose(&mut self, full: bool, casters: &[Caster]) {
        if !full {
            self.mix_static();
        }
        let Renderer {
            ref jobs,
            ref scene,
            ref view,
            ref st,
            ref prev_st,
            w,
            h,
            history_valid,
            ref s_color,
            ref s_depth,
            ref shaded_depth,
            prev_static,
            ref s_nearest,
            ref s_farthest,
            ref s_sun,
            ref dyn_depth,
            ref dyn_id,
            ref dyn_color,
            ref dyn_motion,
            ref dyn_band,
            ref bloom,
            bloom_w,
            bloom_h,
            ref history,
            ref dyn_tiles,
            ref dyn_rows,
            ref vig_x,
            ref blend,
            ref mut centre,
            ref mut past,
            ref mut history_next,
            ref mut dyn_tiles_next,
            ref mut dyn_rows_next,
            ref mut out,
            ref mut bloom_a,
            ..
        } = *self;
        let l = &scene.look;
        let (wu, bw) = (w as usize, bloom_w as usize);
        let cs = wu + 2; // a padded row: the edge pixel copied either side
        let bands = (h + COMPOSE_ROWS - 1) / COMPOSE_ROWS;
        let band_len = COMPOSE_ROWS as usize * wu;
        let nexts = history_next.each_mut().map(|p| Parts::chunks(p, band_len));
        // The static image before this one, to tell which surfaces it showed.
        let before = (full && history_valid && prev_static).then_some(&shaded_depth[..]);
        let centres = centre.each_mut().map(|p| Parts::chunks(p, band_len));
        let pasts = past.each_mut().map(|p| Parts::chunks(p, band_len));
        let tiles_next = Parts::chunks(dyn_tiles_next, COMPOSE_ROWS as usize / 8 * (wu / 8));
        let tile_rows_next = Parts::chunks(dyn_rows_next, COMPOSE_ROWS as usize / 8);
        let outs = out.each_mut().map(|p| Parts::chunks(p, band_len));
        let brights = bloom_a.each_mut().map(|p| Parts::chunks(p, COMPOSE_ROWS as usize / 8 * bw));
        let scaled = |r: Vec4, k: f32| Vec4::new(r.x * k, r.y * k, r.z * k, r.w * k);
        let (hw, hh) = (0.5 * w as f32, 0.5 * h as f32);
        let cv = ComposeView {
            w,
            h,
            eye: view.eye,
            ray_base: view.ray_base,
            ray_x: view.ray_x,
            ray_y: view.ray_y,
            s_ray_base: st.ray_base,
            s_ray_x: st.ray_x,
            s_ray_y: st.ray_y,
            s_jitter_x: st.jitter_x,
            s_jitter_y: st.jitter_y,
            parallax: length(view.eye - st.eye) / length(view.ray_x),
            b_parallax: length(view.eye - prev_st.eye) / length(view.ray_x),
            b_eye: prev_st.eye,
            bx: prev_st.ray_x * (1.0 / dot(prev_st.ray_x, prev_st.ray_x)),
            by: prev_st.ray_y * (1.0 / dot(prev_st.ray_y, prev_st.ray_y)),
            bz: prev_st.ray_base - prev_st.ray_x * (dot(prev_st.ray_base, prev_st.ray_x) / dot(prev_st.ray_x, prev_st.ray_x))
                - prev_st.ray_y * (dot(prev_st.ray_base, prev_st.ray_y) / dot(prev_st.ray_y, prev_st.ray_y)),
            bcx: dot(prev_st.ray_base, prev_st.ray_x) / dot(prev_st.ray_x, prev_st.ray_x),
            bcy: dot(prev_st.ray_base, prev_st.ray_y) / dot(prev_st.ray_y, prev_st.ray_y),
            p0: scaled(row(&view.prev_view_proj, 0), hw),
            p1: scaled(row(&view.prev_view_proj, 1), -hh),
            p3: row(&view.prev_view_proj, 3),
            u0: Through::new(scaled(row(&view.unjittered, 0), hw), st),
            u1: Through::new(scaled(row(&view.unjittered, 1), -hh), st),
            u3: Through::new(row(&view.unjittered, 3), st),
            m0: Through::new(scaled(row(&view.prev_view_proj, 0), hw), st),
            m1: Through::new(scaled(row(&view.prev_view_proj, 1), -hh), st),
            m3: Through::new(row(&view.prev_view_proj, 3), st),
        };
        // A full frame's motion is for its jittered samples, an in-between frame's for the
        // pixel centres the warp solved for.
        let (jx, jy) = if full { (view.jitter_x, view.jitter_y) } else { (0.0, 0.0) };
        let tc = TaaConst {
            w,
            h,
            jx,
            jy,
            // Only a full frame brings new samples; an in-between frame is the same ones
            // resampled, and weighs less.
            alpha: if full { ALPHA_FULL } else { ALPHA_BETWEEN },
            exposure: l.exposure,
            saturation: l.saturation,
            history_valid,
            full,
            srgb: Srgb16::new(),
        };
        let bloom_row = |y: i32| {
            let fy = fclamp((y as f32 + 0.5) * bloom_h as f32 / h as f32 - 0.5, 0.0, bloom_h as f32 - 1.001);
            (fy as usize, fy - (fy as i32) as f32)
        };

        jobs.run(bands, |band, _| {
            let (y0, y1) = (band * COMPOSE_ROWS, h.min((band + 1) * COMPOSE_ROWS));
            let own = (y1 - y0) as usize;
            let rows = own + 2; // one row of margin above and below
            let mut next = nexts.each_ref().map(|p| p.take(band as usize));
            let mut centre = centres.each_ref().map(|p| p.take(band as usize));
            let mut past = pasts.each_ref().map(|p| p.take(band as usize));
            let (mut tiles, mut tile_rows) = (tiles_next.take(band as usize), tile_rows_next.take(band as usize));
            let tiles_at = band as usize * tiles.len();
            for (a, last) in tiles.iter_mut().zip(&dyn_tiles[tiles_at..]) {
                *a = last.saturating_add(1);
            }
            let mut outs = outs.each_ref().map(|p| p.take(band as usize));
            let mut bright = brights.each_ref().map(|p| p.take(band as usize));
            let mut line = Vec::with_capacity(bw + 9);
            let (mut reach, mut reach_row) = (vec![0u8; wu / 8], usize::MAX);
            SCRATCH.with_borrow_mut(|buf| {
                MASKS.with_borrow_mut(|flags| {
                    // The band's rows of input (static image, players, shadows, bloom) as
                    // three padded planes, their motion, rows of depth, sun and warp
                    // scratch, a mask per row (1 for a player, 2 for its shadow) and, for
                    // its own rows, which pixels may blend unclamped. Anything more per row
                    // and four bands stop fitting in the cache together.
                    buf.resize(3 * rows * cs + 2 * rows * wu + 5 * wu, 0.0);
                    let (input, rest) = buf.split_at_mut(3 * rows * cs);
                    let (motion, rest) = rest.split_at_mut(2 * rows * wu);
                    let (near, rest) = rest.split_at_mut(wu);
                    let (inner, rest) = rest.split_at_mut(wu);
                    let (sun, rest) = rest.split_at_mut(wu);
                    let (q_x, q_y) = rest.split_at_mut(wu);
                    let (mo_x, mo_y) = motion.split_at_mut(rows * wu);
                    // Only the masks start cleared: still_row writes every flag it is asked for.
                    flags.resize(rows * wu + own * wu, 0);
                    let (masks, still) = flags.split_at_mut(rows * wu);
                    masks.fill(0);
                    let mut flagged = [false; COMPOSE_ROWS as usize + 2];
                    for i in 0..rows {
                        let y = iclamp(y0 - 1 + i as i32, 0, h - 1);
                        let row = y as usize * wu;
                        let mask = &mut masks[i * wu..(i + 1) * wu];
                        let shadowed = casters.iter().any(|c| (c.y0..=c.y1).contains(&y));
                        let mut cur: [&mut [f32]; 3] = {
                            let (pr, rest) = input.split_at_mut(rows * cs);
                            let (pg, pb) = rest.split_at_mut(rows * cs);
                            [pr, pg, pb].map(|p| &mut p[i * cs..(i + 1) * cs])
                        };
                        let (mx, my) = (&mut mo_x[i * wu..(i + 1) * wu], &mut mo_y[i * wu..(i + 1) * wu]);
                        {
                            let [cr, cg, cb] = &mut cur;
                            let dst = [&mut cr[1..=wu], &mut cg[1..=wu], &mut cb[1..=wu]];
                            // Colour is always taken at the pixel's centre. A full frame's
                            // samples sit the jitter away from it: taken as they are, fine
                            // detail hopped back and forth with the jitter every frame.
                            if full {
                                let groups = row / 8..(row + wu) / 8;
                                identity_row(&cv, y, &s_depth[row..row + wu], before.map(|b| (b, &s_nearest[groups.clone()], &s_farthest[groups])),
                                             near, inner, mx, my);
                                if shadowed {
                                    sun.copy_from_slice(&s_sun[row..row + wu]);
                                }
                                for (x, (qx, qy)) in q_x.iter_mut().zip(q_y.iter_mut()).enumerate() {
                                    (*qx, *qy) = (x as f32 - cv.s_jitter_x, y as f32 - cv.s_jitter_y);
                                }
                            } else {
                                warp_row(&cv, y, s_depth, (s_nearest, s_farthest), if shadowed { Some((&s_sun[..], &mut *sun)) } else { None }, near, inner,
                                         q_x, q_y, mx, my);
                                // `blend` is on pixel centres, the static image's samples
                                // the jitter away from them.
                                for (qx, qy) in q_x.iter_mut().zip(q_y.iter_mut()) {
                                    (*qx, *qy) = (*qx + cv.s_jitter_x, *qy + cv.s_jitter_y);
                                }
                            }
                            filter_row(w, h, if full { s_color } else { blend }, q_x, q_y, dst);
                        }
                        if full && (1..=own).contains(&i) {
                            let o = (i - 1) * wu;
                            for (plane, c) in centre.iter_mut().zip(&cur) {
                                let dst = plane[o..o + wu].as_chunks_mut::<8>().0;
                                for (m, d) in dst.iter_mut().enumerate() {
                                    *d = F8::load(c[1 + 8 * m..][..8].try_into().unwrap()).to_f16();
                                }
                            }
                        }
                        // Players, where they are nearer than the static surface.
                        if dyn_band[y as usize / BAND as usize] != 0 {
                            for x in 0..wu {
                                let k = row + x;
                                if dyn_id[k] != NO_PRIM && dyn_depth[k] > near[x] {
                                    for ch in 0..3 {
                                        cur[ch][x + 1] = from_half(dyn_color[ch][k]);
                                    }
                                    mx[x] = dyn_motion[0][k];
                                    my[x] = dyn_motion[1][k];
                                    mask[x] = 1;
                                    flagged[i] = true;
                                }
                            }
                        }
                        if shadowed && player_shadows(&cv, y, scene, casters, near, sun, mask, &mut cur) {
                            flagged[i] = true;
                        }
                        if (1..=own).contains(&i) {
                            let o = (i - 1) * wu;
                            if history_valid {
                                // Recent player tiles, grown by one tile (nothing moves
                                // a tile a frame), for this row of tiles: one byte per group
                                // of eight, which is one tile wide.
                                let ty = y as usize / 8;
                                let (t0, t1) = (ty.saturating_sub(1), (ty + 1).min(dyn_rows.len() - 1));
                                let near_dyn = dyn_rows[t0..=t1].contains(&1);
                                let recent = |a: &u8| *a < DYN_MEMORY;
                                if near_dyn && reach_row != ty {
                                    let tw = wu / 8;
                                    for (tx, r) in reach.iter_mut().enumerate() {
                                        let (x0, x1) = (tx.saturating_sub(1), (tx + 1).min(tw - 1));
                                        *r = (t0..=t1).any(|row| dyn_tiles[row * tw + x0..=row * tw + x1].iter().any(recent)) as u8;
                                    }
                                    reach_row = ty;
                                }
                                still_row(near_dyn.then_some(&reach[..]), inner, mask, &mut still[o..o + wu]);
                            }
                            if flagged[i] {
                                let t = (i - 1) / 8 * (wu / 8);
                                for x in (0..wu).filter(|&x| mask[x] != 0) {
                                    tiles[t + x / 8] = 0;
                                }
                            }
                        }
                        // This band's share of the next frame's bright pass. A full frame's
                        // is taken after the resolve, below.
                        if !full && i >= 1 && i <= own && (y & 1) == 1 {
                            bright_row([&cur[0][1..], &cur[1][1..], &cur[2][1..]], y, y0, bw, bright.each_mut().map(|b| &mut b[..]));
                        }
                        // Last frame's bloom goes in before the temporal blend rather than
                        // after, so it is smoothed with everything else.
                        let (by, fy) = bloom_row(y);
                        for (ch, dst) in cur.iter_mut().enumerate() {
                            let rows = (&bloom[ch][by * bw..(by + 1) * bw], &bloom[ch][(by + 1) * bw..(by + 2) * bw]);
                            add_bloom(rows, fy, &mut line, &mut dst[1..=wu]);
                            dst[0] = dst[1];
                            dst[wu + 1] = dst[wu];
                        }
                    }
                    for (flag, row) in tile_rows.iter_mut().zip(tiles.chunks(wu / 8)) {
                        *flag = row.iter().any(|a| *a < DYN_MEMORY) as u8;
                    }
                    let vig = &vig_x[..wu];
                    let mut glow = if full { vec![0.0f32; 3 * wu] } else { Vec::new() };
                    for r in 0..own {
                        let y = y0 + r as i32;
                        let o = r * wu;
                        let near: [[&[f32]; 3]; 3] =
                            std::array::from_fn(|ch| std::array::from_fn(|k| &input[(ch * rows + r + k) * cs..][..cs]));
                        let dyn_near = flagged[r] | flagged[r + 1] | flagged[r + 2];
                        let dyn_masks = dyn_near.then(|| std::array::from_fn(|k| &masks[(r + k) * wu..][..wu]));
                        let [nr, ng, nb] = &mut next;
                        let [or, og, ob] = &mut outs;
                        taa_row(
                            &tc,
                            y,
                            near,
                            (&mo_x[(r + 1) * wu..][..wu], &mo_y[(r + 1) * wu..][..wu]),
                            &still[o..o + wu],
                            dyn_masks,
                            history,
                            full.then(|| past.each_mut().map(|p| &mut p[o..o + wu])),
                            [&mut nr[o..o + wu], &mut ng[o..o + wu], &mut nb[o..o + wu]],
                            [&mut or[o..o + wu], &mut og[o..o + wu], &mut ob[o..o + wu]],
                            vig,
                        );
                        if full {
                            // `past` is kept without what this frame added to the static
                            // image (bloom, and a player where one stands): the resolve left
                            // its history there, or the image itself, and the difference
                            // between the image and `centre` is taken back out.
                            let mask = masks[(r + 1) * wu..][..wu].as_chunks::<8>().0;
                            for ch in 0..3 {
                                let img = &input[(ch * rows + r + 1) * cs + 1..][..wu];
                                let (c, p) = (centre[ch][o..o + wu].as_chunks::<8>().0, past[ch][o..o + wu].as_chunks_mut::<8>().0);
                                for (m, p) in p.iter_mut().enumerate() {
                                    let (cv, now) = (F8::load_f16(&c[m]), F8::load(img[8 * m..][..8].try_into().unwrap()));
                                    let free = I8::load_u8(&mask[m]).eq(I8::splat(0));
                                    *p = F8::select(free, F8::load_f16(p) - now + cv, cv).to_f16();
                                }
                            }
                        }
                        // A full frame's bright pass is taken from what the next in-between
                        // frame will show, the settled picture nudged towards this one, not
                        // from this frame's fresh render. From the fresh render, the sun's glint
                        // on water, sharp and at its brightest there, bloomed into a glow over
                        // the pond for the one frame after every full frame: a light
                        // flickering on the water a few times a second.
                        if full && (y & 1) == 1 {
                            let k = 1.0 - ALPHA_STILL;
                            for ch in 0..3 {
                                let (c, p) = (&centre[ch][o..o + wu], &past[ch][o..o + wu]);
                                for (m, g) in glow[ch * wu..(ch + 1) * wu].as_chunks_mut::<8>().0.iter_mut().enumerate() {
                                    let cv = F8::load_f16(c[8 * m..][..8].try_into().unwrap());
                                    (F8::load_f16(p[8 * m..][..8].try_into().unwrap()) - cv).mul_add(F8::splat(k), cv).store(g);
                                }
                            }
                            let (gr, rest) = glow.split_at(wu);
                            let (gg, gb) = rest.split_at(wu);
                            bright_row([gr, gg, gb], y, y0, bw, bright.each_mut().map(|b| &mut b[..]));
                        }
                    }
                });
            });
        });
    }

    /// Marks the static image's depth edges in its depths: the lowest bit of a surface's 1/w
    /// set, and sky's zero made -0.0, where 1/w varies by more than EDGE_SPREAD within two
    /// pixels. One ulp of 1/w changes nothing, and the warp reads these depths anyway, so
    /// in-between frames learn it for free.
    fn mark_edges(&mut self) {
        let Renderer { ref jobs, w, h, ref mut s_depth, ref mut edges, .. } = *self;
        let (wu, rows) = (w as usize, 24);
        let bands = (h + rows - 1) / rows;
        {
            let flags = Parts::chunks(edges, rows as usize * wu);
            let depth = &*s_depth;
            // Separable: each row's 5-wide minimum and maximum once, kept for the five output
            // rows that need it, then the 5-tall ones; a 5 x 5 window per pixel read every
            // depth 25 times.
            let across = |row: &[f32], span: &mut [f32]| {
                let (lo, hi) = span.split_at_mut(wu);
                let mut x = 2;
                while x + 10 <= wu {
                    let (mut l, mut u) = (F8::splat(f32::MAX), F8::splat(0.0));
                    for k in 0..5 {
                        let v = F8::load(row[x - 2 + k..][..8].try_into().unwrap()).max(F8::splat(0.0));
                        (l, u) = (l.min(v), u.max(v));
                    }
                    l.store((&mut lo[x..x + 8]).try_into().unwrap());
                    u.store((&mut hi[x..x + 8]).try_into().unwrap());
                    x += 8;
                }
                for x in (0..2).chain(x..wu) {
                    let v = &row[x.saturating_sub(2)..(x + 3).min(wu)];
                    lo[x] = v.iter().fold(f32::MAX, |a, v| a.min(v.max(0.0)));
                    hi[x] = v.iter().fold(0.0f32, |a, v| a.max(v.max(0.0)));
                }
            };
            jobs.run(bands, |b, _| {
                let mut flags = flags.take(b as usize);
                let row = |y: i32| {
                    let r = y.clamp(0, h - 1) as usize * wu;
                    &depth[r..r + wu]
                };
                let (y0, y1) = (b * rows, h.min((b + 1) * rows));
                let mut ring: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0; 2 * wu]);
                for y in y0 - 2..y0 + 2 {
                    across(row(y), &mut ring[y.rem_euclid(5) as usize]);
                }
                for y in y0..y1 {
                    across(row(y + 2), &mut ring[(y + 2).rem_euclid(5) as usize]);
                    let out = &mut flags[(y - y0) as usize * wu..][..wu];
                    for (m, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let (mut lo, mut hi) = (F8::splat(f32::MAX), F8::splat(0.0));
                        for span in &ring {
                            lo = lo.min(F8::load(span[8 * m..][..8].try_into().unwrap()));
                            hi = hi.max(F8::load(span[wu + 8 * m..][..8].try_into().unwrap()));
                        }
                        *o = ((hi - lo).gt(hi * F8::splat(EDGE_SPREAD)) & I8::splat(1)).to_bytes();
                    }
                }
            });
        }
        let depths = Parts::chunks(s_depth, rows as usize * wu);
        let edges = &*edges;
        jobs.run(bands, |b, _| {
            let mut depth = depths.take(b as usize);
            let at = b as usize * rows as usize * wu;
            let (d, e) = (depth.as_chunks_mut::<8>().0, edges[at..].as_chunks::<8>().0);
            for (d, e) in d.iter_mut().zip(e) {
                let (v, edge) = (F8::load(d), I8::load_u8(e));
                let surface = (v.bits() & I8::splat(!1)) | edge;
                I8::select(v.le(F8::splat(0.0)), edge * I8::splat(i32::MIN), surface).as_f8().store(d);
            }
        });
    }

    /// Fills s_nearest and s_farthest from the static depths.
    fn reach_depths(&mut self) {
        let Renderer { ref jobs, w, h, ref s_depth, ref mut s_nearest, ref mut s_farthest, .. } = *self;
        let (wu, gw) = (w as usize, w as usize / 8);
        let (nears, fars) = (Parts::chunks(s_nearest, gw), Parts::chunks(s_farthest, gw));
        jobs.run(h, |y, _| {
            let (mut nearest, mut farthest) = (nears.take(y as usize), fars.take(y as usize));
            reach_row(&s_depth[y as usize * wu..][..wu], &mut nearest, &mut farthest);
        });
    }

    fn blur_bloom(&mut self) {
        // The bright pass the compose pass left, blurred in two widening passes for the
        // next frame, eight texels at a time.
        const KERNEL: [f32; 5] = [0.227, 0.1945, 0.1216, 0.054, 0.0162];
        let Renderer { ref jobs, ref mut bloom_a, ref mut bloom_b, ref mut bloom, bloom_w, bloom_h, .. } = *self;
        let bw = bloom_w as usize;
        let k = KERNEL.map(F8::splat);
        for pass in 0..2 {
            // Across: src a, out b.
            {
                let rows = bloom_b.each_mut().map(|p| Parts::chunks(p, bw));
                let src = &*bloom_a;
                jobs.run(bloom_h, |by, _| {
                    let mut out = rows.each_ref().map(|p| p.take(by as usize));
                    for (plane, o) in src.iter().zip(out.iter_mut()) {
                        let row = &plane[by as usize * bw..(by as usize + 1) * bw];
                        let at = |x: i32| row[x.clamp(0, bloom_w - 1) as usize];
                        let reach = 4 * (pass + 1);
                        for (m, o) in o.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                            let x = 8 * m as i32;
                            if x >= reach && x + 7 + reach < bloom_w {
                                let mut sum = at8(row, x as usize) * k[0];
                                for (i, w) in k.iter().enumerate().skip(1) {
                                    let step = i as i32 * (pass + 1);
                                    sum = (at8(row, (x - step) as usize) + at8(row, (x + step) as usize)).mul_add(*w, sum);
                                }
                                sum.store(o);
                            } else {
                                *o = std::array::from_fn(|j| {
                                    let x = x + j as i32;
                                    let mut sum = at(x) * KERNEL[0];
                                    for (i, w) in KERNEL.iter().enumerate().skip(1) {
                                        let step = i as i32 * (pass + 1);
                                        sum += (at(x - step) + at(x + step)) * w;
                                    }
                                    sum
                                });
                            }
                        }
                    }
                });
            }
            // Down: src b, out a (or, the second time, the result).
            let dst = if pass == 0 { &mut *bloom_a } else { &mut *bloom };
            let rows = dst.each_mut().map(|p| Parts::chunks(p, bw));
            let src = &*bloom_b;
            jobs.run(bloom_h, |by, _| {
                let mut out = rows.each_ref().map(|p| p.take(by as usize));
                for (plane, o) in src.iter().zip(out.iter_mut()) {
                    let line = |y: i32| &plane[y.clamp(0, bloom_h - 1) as usize * bw..][..bw];
                    let taps: [(&[f32], &[f32]); 4] = std::array::from_fn(|i| {
                        let step = (i as i32 + 1) * (pass + 1);
                        (line(by - step), line(by + step))
                    });
                    let centre = line(by);
                    for (m, o) in o.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let x = 8 * m;
                        let mut sum = at8(centre, x) * k[0];
                        for (i, (a, b)) in taps.iter().enumerate() {
                            sum = (at8(a, x) + at8(b, x)).mul_add(k[i + 1], sum);
                        }
                        sum.store(o);
                    }
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------------
// Overlay mode: players drawn over a recorded frame of the orbit instead of rendering the
// world again. The static scene is rasterised only where players, their shadows and their
// name tags fall, for its depth (what hides a player) and the sun on it (for shadows); the
// players are drawn four times at sub-pixel offsets instead of through the temporal
// resolve; and the result is laid over the recorded picture, which is brought back to
// linear light for it by inverting the grade, tonemap and sRGB of `taa_row`.

/// Rectangles [x0, y0, x1, y1) clipped to the frame, on even pixels, those that overlap
/// or nearly touch merged into one.
pub fn merge_rects(mut rects: Vec<[i32; 4]>, w: i32, h: i32) -> Vec<[i32; 4]> {
    const NEAR_BY: i32 = 16;
    rects.retain_mut(|r| {
        *r = [r[0].max(0) & !1, r[1].max(0) & !1, (r[2].min(w) + 1) & !1, (r[3].min(h) + 1) & !1];
        r[0] < r[2] && r[1] < r[3]
    });
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let (a, b) = (rects[i], rects[j]);
                if a[0] <= b[2] + NEAR_BY && b[0] <= a[2] + NEAR_BY && a[1] <= b[3] + NEAR_BY && b[1] <= a[3] + NEAR_BY {
                    rects[i] = [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])];
                    rects.swap_remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    rects
}

/// Sub-pixel offsets of the four player samples: a rotated grid.
const OVERLAY_SAMPLES: [(f32, f32); 4] = [(-0.125, -0.375), (0.375, -0.125), (0.125, 0.375), (-0.375, 0.125)];

#[inline]
fn aces(v: f32) -> f32 {
    fclamp(v * (2.51 * v + 0.03) / (v * (2.43 * v + 0.59) + 0.14), 0.0, 1.0)
}

/// The inverse of `aces` over [0, 1): the larger root of its quadratic.
#[inline]
fn aces_inverse(y: f32) -> f32 {
    let y = fclamp(y, 0.0, 0.995);
    let (a, b, c) = (2.43 * y - 2.51, 0.59 * y - 0.03, 0.14 * y);
    let d = fmax(0.0, b * b - 4.0 * a * c);
    fmax(0.0, (-b - d.sqrt()) / (2.0 * a))
}

#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

impl Renderer {
    /// Sets up the camera for an overlay frame at `time` and returns the rectangles of
    /// pixels the players and their shadows can touch, one per group of them.
    pub fn overlay_reach(&mut self, dynamic: &[Quad], pose: &CameraPose, time: f64) -> Vec<[i32; 4]> {
        self.ensure_overlay();
        self.camera(pose, time, 0.0, 0.0);
        let (w, h) = (self.w, self.h);
        let vp = self.view.view_proj;
        let mut rects = Vec::new();
        for q in dynamic {
            let corners = [q.p, q.p + q.e1, q.p + q.e2, q.p + q.e1 + q.e2];
            let (mut a, mut b, mut c, mut d) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            let mut behind = false;
            for p in corners {
                let cl = vp * Vec4::new(p.x, p.y, p.z, 1.0);
                if cl.w < NEAR {
                    behind = true;
                    break;
                }
                let (x, y) = ((cl.x / cl.w * 0.5 + 0.5) * w as f32, (0.5 - cl.y / cl.w * 0.5) * h as f32);
                (a, b, c, d) = (a.min(x), b.min(y), c.max(x), d.max(y));
            }
            if behind {
                rects.push([0, 0, w, h]);
            } else if c >= 0.0 && d >= 0.0 && a < w as f32 && b < h as f32 {
                rects.push([a as i32 - 2, b as i32 - 2, c as i32 + 3, d as i32 + 3]);
            }
        }
        for c in self.casters(dynamic) {
            rects.push([c.x0, c.y0, c.x1 + 1, c.y1 + 1]);
        }
        merge_rects(rects, w, h)
    }

    /// Draws the players over the frame in `out` (8-bit sRGB, as a recorded frame decodes
    /// to) within `rect`, which overlay_reach found, grown to hold anything else drawn later.
    pub fn overlay_draw<'a>(&mut self, meshes: &'a [Arc<SectionMesh>], dynamic: &'a [Quad], pose: &CameraPose, time: f64,
                            rect: [i32; 4]) {
        let (w, h) = (self.w, self.h);
        let [rx0, ry0, rx1, ry1] = rect;
        if rx0 >= rx1 || ry0 >= ry1 {
            return;
        }
        self.ensure_overlay();
        let workers = self.jobs.size() as usize;
        let mut lists: Vec<Lists<'a>> = std::mem::take(&mut self.pool);
        // The static scene in the rectangle, unjittered: depth and which prim is where.
        let mut setup = self.camera(pose, time, 0.0, 0.0);
        setup.clip = [rx0 as f32, ry0 as f32, rx1 as f32, ry1 as f32];
        self.clip = rect;
        // Last frame's depth pyramid is from another camera.
        self.hiz_valid = false;
        self.setup_prims(meshes, &setup, &mut lists[..workers]);
        self.raster(&lists, false, false);
        self.raster(&lists, true, false);
        let casters = self.casters(dynamic);
        let (rw, rh) = ((rx1 - rx0) as usize, (ry1 - ry0) as usize);
        let at = |x: i32, y: i32| (y * w + x) as usize;
        // Per pixel of the rectangle: the nearest static surface (1/w, 0 for sky), and the
        // players' summed colour and how many samples they cover.
        let mut scratch = std::mem::take(&mut self.overlay_scratch);
        scratch.resize(4 * rw * rh, 0.0);
        let (near, colour) = scratch.split_at_mut(rw * rh);
        colour.fill(0.0);
        let colour: [&mut [f32]; 3] = {
            let (r, rest) = colour.split_at_mut(rw * rh);
            let (g, b) = rest.split_at_mut(rw * rh);
            [r, g, b]
        };
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                let (i, o) = (at(x, y), (y - ry0) as usize * rw + (x - rx0) as usize);
                let (d, dw) = (self.depth[i], self.depth_water[i]);
                near[o] = if self.id_water[i] != NO_PRIM && dw > d { dw } else if self.id[i] != NO_PRIM { d } else { 0.0 };
            }
        }
        // The players, at four sub-pixel offsets.
        let mut cover = std::mem::take(&mut self.overlay_cover);
        cover.clear();
        cover.resize(rw * rh, 0);
        if !dynamic.is_empty() {
            for (sx, sy) in OVERLAY_SAMPLES {
                let mut s = self.camera(pose, time, sx, sy);
                s.clip = setup.clip;
                let dyn_lists = &mut lists[workers];
                dyn_lists.clear();
                for q in dynamic {
                    add_prim(&s, &self.scene.assets, q, q.layer == TRANSLUCENT, dyn_lists);
                }
                self.raster(&lists, false, true);
                self.shade_dynamic(&lists);
                for y in ry0..ry1 {
                    if self.dyn_band[(y / BAND) as usize] == 0 {
                        continue;
                    }
                    let band = (y / BAND * BAND * w) as usize;
                    for x in rx0..rx1 {
                        let (i, o) = (at(x, y), (y - ry0) as usize * rw + (x - rx0) as usize);
                        if self.dyn_id[i] == NO_PRIM || self.dyn_depth[i] <= near[o] {
                            continue;
                        }
                        let k = i - band;
                        for ch in 0..3 {
                            colour[ch][o] += from_half(self.dyn_color[ch][band + k]);
                        }
                        cover[o] += 1;
                    }
                }
            }
        }
        self.camera(pose, time, 0.0, 0.0);
        // Composite, through the inverse of the frame's grade and back.
        let l = self.scene.look;
        let (eye, base, rx, ry) = (self.view.eye, self.view.ray_base, self.view.ray_x, self.view.ray_y);
        let sun_dir = l.sun_dir;
        let frame = self.scene.frame;
        let inv = |d: f32| 1.0 / if d.abs() < 1e-6 { 1e-6f32.copysign(d) } else { d };
        let inv_sun = Vec3::new(inv(sun_dir.x), inv(sun_dir.y), inv(sun_dir.z));
        for y in ry0..ry1 {
            let py = y as f32 + 0.5;
            let vy = (y as f32 + 0.5) / h as f32 - 0.5;
            for x in rx0..rx1 {
                let (i, o) = (at(x, y), (y - ry0) as usize * rw + (x - rx0) as usize);
                // How much of the sun the players take from this pixel.
                let mut shade = 1.0f32;
                let a = near[o];
                // The sun on this pixel, worked out only where a body's shadow can fall.
                let mut sun: Option<f32> = None;
                if a > 0.0 && cover[o] < 4 {
                    let p = eye + (base + rx * (x as f32 + 0.5) + ry * py) * (1.0 / a);
                    for c in casters.iter().filter(|c| x >= c.x0 && x <= c.x1 && y >= c.y0 && y <= c.y1) {
                        if p.x < c.lo.x || p.y < c.lo.y || p.z < c.lo.z || p.x > c.hi.x || p.y > c.hi.y || p.z > c.hi.z {
                            continue;
                        }
                        let (t0, t1) = ((c.body_lo - p) * inv_sun, (c.body_hi - p) * inv_sun);
                        let (near_t, far_t) = (vmin(t0, t1), vmax(t0, t1));
                        if fmin(far_t.x, fmin(far_t.y, far_t.z)) < fmax(0.0, fmax(near_t.x, fmax(near_t.y, near_t.z))) {
                            continue;
                        }
                        let lit = *sun.get_or_insert_with(|| {
                            let i = at(x, y);
                            let pid = self.id[i];
                            if pid == NO_PRIM {
                                0.0
                            } else if self.id_water[i] != NO_PRIM && self.depth_water[i] > self.depth[i] {
                                0.5
                            } else {
                                let px = x as f32 + 0.5;
                                let pr = prim(&lists, pid);
                                let aa = pr.a[0] * px + pr.a[1] * py + pr.a[2];
                                let wv = 1.0 / aa;
                                let s = clamp01((pr.s[0] * px + pr.s[1] * py + pr.s[2]) * wv);
                                let t = clamp01((pr.t[0] * px + pr.t[1] * py + pr.t[2]) * wv);
                                self.scene.lit_surface(pr, s, t, wv, pr.at(s, t), x, y).1
                            }
                        });
                        if lit <= 0.01 {
                            break;
                        }
                        // Four rays, the sun turned slightly each way, for a soft edge.
                        let mut hits = 0;
                        for k in 0..4 {
                            let g = fract(52.9829189 * fract(0.06711056 * (x as f32 + 5.588238 * ((frame + k) & 63) as f32) + 0.00583715 * y as f32));
                            let dir = normalize(sun_dir + Vec3::new((g - 0.5) * 0.05, 0.0, (fract(g * 7.123) - 0.5) * 0.05));
                            let hit = c.near.iter().any(|q| {
                                let denom = dot(q.n, dir);
                                if denom.abs() < 1e-9 {
                                    return false;
                                }
                                let lambda = dot(q.n, q.p - p) / denom;
                                if !(0.03..=12.0).contains(&lambda) {
                                    return false;
                                }
                                let hp = p + dir * lambda - q.p;
                                let (s, t) = (dot(hp, q.u1), dot(hp, q.u2));
                                (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t)
                            });
                            hits += hit as i32;
                        }
                        shade = fmin(shade, 1.0 - 0.62 * fmin(1.0, lit) * hits as f32 / 4.0);
                    }
                }
                if shade >= 1.0 && cover[o] == 0 {
                    continue;
                }
                // Back to the linear light the resolve held, before its grade.
                let v = 1.0 - 0.44 * (self.vig_x[x as usize] + vy * vy);
                let shown = [self.out[0][i], self.out[1][i], self.out[2][i]].map(|b| srgb_to_linear(b as f32 / 255.0) / v);
                let lum = shown[0] * 0.2126 + shown[1] * 0.7152 + shown[2] * 0.0722;
                let lin = shown.map(|c| aces_inverse((c - lum) / l.saturation + lum) / l.exposure);
                let k = cover[o] as f32 / 4.0;
                let mixed: [f32; 3] = std::array::from_fn(|ch| lin[ch] * shade * (1.0 - k) + colour[ch][o] * 0.25);
                let toned = mixed.map(|c| aces(c * l.exposure));
                let lum = toned[0] * 0.2126 + toned[1] * 0.7152 + toned[2] * 0.0722;
                let dither = BAYER[(y as usize & 3) * 4 + (x as usize & 3)] as f32 / 16.0 - 0.5;
                for ch in 0..3 {
                    let c = fclamp(((toned[ch] - lum) * l.saturation + lum) * v, 0.0, 0.999_999_94);
                    self.out[ch][i] = fclamp(linear_to_srgb(c) * 255.0 + 0.5 + dither / 16.0, 0.0, 255.0) as u8;
                }
            }
        }
        self.clip = [0, 0, w, h];
        self.overlay_scratch = scratch;
        self.overlay_cover = cover;
        self.pool = lists.into_iter().map(Lists::recycle).collect();
    }

    /// Ends an overlay frame, whose rectangles overlay_draw has drawn.
    pub fn overlay_done(&mut self) {
        self.scene.frame = self.scene.frame.wrapping_add(1);
    }
}

/// Per prim, what shading reads, 32 values to a prim (two cache lines), so that eight
/// lanes can each read their own prim's field with one gather and the next field of the
/// same prims is already in cache. Rebuilt every full frame.
#[derive(Default)]
struct PrimTable {
    v: Vec<f32>,
    // Where each worker's prims start; a prim id is (worker << 26) | its index there.
    first: [i32; 8],
}

// Floats: plane equations a, s and t (three each), texture coordinates (uv, uv_s, uv_t),
// linear tint, normal, the quad's two edges. Then integers, as bits: texture, the four
// corners' sky light, block light and occlusion packed a byte each, and mat | flags << 16.
const F_A: usize = 0;
const F_S: usize = 3;
const F_T: usize = 6;
const F_UV: usize = 9;
const F_TINT: usize = 15;
const F_N: usize = 18;
const F_E: usize = 21;
const I_TEX: usize = 27;
const I_SKY: usize = 28;
const I_BLK: usize = 29;
const I_AO: usize = 30;
const I_MAT: usize = 31;
const PRIM_STRIDE: usize = 32;

impl PrimTable {
    fn build(&mut self, lists: &[Lists], jobs: &Jobs) {
        let counts: Vec<usize> = lists.iter().map(|l| l.prims.len()).collect();
        // At least one entry, so a gather has somewhere to go even with nothing drawn.
        let n = counts.iter().sum::<usize>().max(1);
        self.v.resize(PRIM_STRIDE * n, 0.0);
        let mut ends = Vec::new();
        let mut at = 0;
        for (w, c) in counts.iter().enumerate() {
            self.first[w] = at as i32;
            at += c;
            ends.push(at * PRIM_STRIDE);
        }
        let parts = Parts::split(&mut self.v, ends);
        let lut = lut();
        jobs.run(lists.len() as i32, |w, _| {
            let mut out = parts.take(w as usize);
            for (p, o) in lists[w as usize].prims.iter().zip(out.as_chunks_mut::<PRIM_STRIDE>().0) {
                let q = p.q;
                let tint = if q.tint == 0xffffff {
                    [1.0; 3]
                } else {
                    [lut[((q.tint >> 16) & 0xff) as usize], lut[((q.tint >> 8) & 0xff) as usize], lut[(q.tint & 0xff) as usize]]
                };
                let bits = |v: i32| f32::from_bits(v as u32);
                *o = [
                    p.a[0], p.a[1], p.a[2], p.s[0], p.s[1], p.s[2], p.t[0], p.t[1], p.t[2], q.uv.x, q.uv.y, q.uv_s.x, q.uv_s.y,
                    q.uv_t.x, q.uv_t.y, tint[0], tint[1], tint[2], q.n.x, q.n.y, q.n.z, q.e1.x, q.e1.y, q.e1.z, q.e2.x, q.e2.y,
                    q.e2.z, bits(q.tex as i32), bits(i32::from_le_bytes(q.sky)), bits(i32::from_le_bytes(q.blk)),
                    bits(i32::from_le_bytes(q.ao)), bits(q.mat as i32 | (q.flags as i32) << 16),
                ];
            }
        });
    }
}

/// Every texture's texels in one array, so that eight lanes can each fetch from a
/// different texture with one gather. Textures are only ever added, so this only grows.
#[derive(Default)]
struct TexPool {
    texels: Vec<u32>,
    first: Vec<i32>,     // per texture, where its first frame starts
    frame_len: Vec<i32>, // per texture, texels per frame, every mip level
    size: Vec<i32>,      // per texture, the (square, power of two) side of level 0
    levels: Vec<i32>,
    level_offset: Vec<i32>, // per texture, eight entries
    base: Vec<i32>,         // per texture, where this frame's animation frame starts
}

impl TexPool {
    fn update(&mut self, assets: &Assets, anim_frame: &[i32]) {
        for id in self.first.len()..assets.texture_count() {
            let t = assets.tex(id as u16);
            // Block textures are all 16 or 32 square; anything else (only entity skins,
            // which static geometry never uses) keeps the fields but samples as level 0.
            let square = t.w == t.h && t.w > 0 && (t.w & (t.w - 1)) == 0 && t.levels <= 8;
            self.first.push(self.texels.len() as i32);
            self.frame_len.push(t.frames.first().map_or(0, |f| f.len()) as i32);
            for f in &t.frames {
                self.texels.extend_from_slice(f);
            }
            self.size.push(if square { t.w } else { 1 });
            self.levels.push(if square { t.levels } else { 1 });
            self.level_offset.extend((0..8).map(|l| if (l as i32) < t.levels { t.level_offset[l] } else { 0 }));
        }
        if self.texels.is_empty() {
            self.texels.push(0);
        }
        self.base.resize(self.first.len(), 0);
        for (b, ((first, len), frame)) in self.base.iter_mut().zip(self.first.iter().zip(&self.frame_len).zip(anim_frame)) {
            *b = first + frame * len;
        }
    }
}

/// Everything the shading of one row reads, fixed for the frame.
struct ShadeCtx<'a> {
    table: &'a PrimTable,
    pool: &'a TexPool,
    lut: &'a [f32; 256],
    maps: &'a [ShadowMap],
    cloud: &'a [f32],
    sky_lut: &'a Planes,
    sky: &'a Planes,
    sky_w: i32,
    look: Look,
    eye: Vec3,
    ray_base: Vec3,
    ray_x: Vec3,
    ray_y: Vec3,
    frame: u32,
    cloud_drift: Vec2,
}

#[inline(always)]
fn smooth8(e0: f32, e1: f32, x: F8) -> F8 {
    let s = F8::splat;
    let t = ((x - s(e0)) * s(1.0 / (e1 - e0))).max(s(0.0)).min(s(1.0));
    t * t * (s(3.0) - s(2.0) * t)
}

#[inline(always)]
fn fract8(v: F8) -> F8 {
    v - v.floor()
}

/// Eight lanes of `Scene::sky_gradient`.
#[inline(always)]
fn sky_gradient8(c: &ShadeCtx, dx: F8, dy: F8, dz: F8) -> [F8; 3] {
    let s = F8::splat;
    let l = &c.look;
    let cos_sun = s(l.sun_dir.x).mul_add(dx, s(l.sun_dir.y).mul_add(dy, s(l.sun_dir.z) * dz));
    let k = ((s(1.0) - cos_sun) * s(0.5)).max(s(0.0)).sqrt() * s((SKY_SUN - 1) as f32);
    let u = dy.mul_add(s(0.5), s(0.5)).max(s(0.0)).min(s(1.0)) * s((SKY_UP - 1) as f32);
    let ik = k.floor().min(s((SKY_SUN - 2) as f32));
    let iu = u.floor().min(s((SKY_UP - 2) as f32));
    let (fk, fu) = (k - ik, u - iu);
    let r0 = ik.mul_add(s(SKY_UP as f32), iu).to_int();
    let r1 = r0 + I8::splat(SKY_UP as i32);
    let one = I8::splat(1);
    c.sky_lut.each_ref().map(|p| {
        let (a, b, cc, d) = (F8::gather(p, r0), F8::gather(p, r0 + one), F8::gather(p, r1), F8::gather(p, r1 + one));
        let top = (b - a).mul_add(fu, a);
        let bot = (d - cc).mul_add(fu, cc);
        (bot - top).mul_add(fk, top)
    })
}

/// Eight lanes of `Scene::shadow_at`: the first cascade that holds each point.
#[inline(always)]
fn shadow8(c: &ShadeCtx, p: [F8; 3], n: [F8; 3], jx: F8, jy: F8) -> F8 {
    let s = F8::splat;
    let mut out = s(1.0);
    let mut done = I8::splat(0);
    for sm in c.maps {
        let b = &sm.b;
        let off = s(b.texel_world * 1.5);
        let q = [n[0].mul_add(off, p[0]), n[1].mul_add(off, p[1]), n[2].mul_add(off, p[2])];
        let dot = |v: Vec3| s(v.x).mul_add(q[0], s(v.y).mul_add(q[1], s(v.z) * q[2]));
        let inv = s(1.0 / b.texel_world);
        let spread = s(if b.size > 2300 { 2.0 } else { 1.2 });
        let u = (dot(b.right) - s(b.u0)).mul_add(inv, jx.mul_add(spread, s(-0.5)));
        let v = (dot(b.up) - s(b.v0)).mul_add(inv, jy.mul_add(spread, s(-0.5)));
        let hi = s((b.size - 2) as f32);
        let inside = u.ge(s(1.0)) & v.ge(s(1.0)) & hi.ge(u) & hi.ge(v);
        let take = inside & done.eq(I8::splat(0));
        if !take.any() {
            continue;
        }
        let d = (dot(b.to_sun) - s(b.d0)).mul_add(s(32.0), s(2.0)); // 1/16 block bias
        let (u, v) = (F8::select(take, u, s(1.0)), F8::select(take, v, s(1.0)));
        let (fu, fv) = (u.floor(), v.floor());
        let (iu, iv) = (fu.to_int(), fv.to_int());
        let tiles = I8::splat(b.size >> 3);
        let (seven, three) = (I8::splat(7), I8::splat(3));
        // Texel (x, y) in the map's 8 x 8 tiles.
        let index = |x: I8, y: I8| (y.shr(three) * tiles + x.shr(three)) * I8::splat(64) + (y & seven) * I8::splat(8) + (x & seven);
        let one = I8::splat(1);
        let lit = |x: I8, y: I8| {
            let z = I8::gather_u16(&sm.depth, index(x, y)).to_float();
            F8::select(d.ge(z), s(1.0), s(0.0))
        };
        let (a, bb, cc, dd) = (lit(iu, iv), lit(iu + one, iv), lit(iu, iv + one), lit(iu + one, iv + one));
        let (fx, fy) = (u - fu, v - fv);
        let top = (bb - a).mul_add(fx, a);
        let bot = (dd - cc).mul_add(fx, cc);
        out = F8::select(take, (bot - top).mul_add(fy, top), out);
        done = done | take;
        if done.all() {
            break;
        }
    }
    out
}

/// Eight lanes of `Scene::cloud_shadow`.
#[inline(always)]
fn cloud8(c: &ShadeCtx, p: [F8; 3]) -> F8 {
    let s = F8::splat;
    if c.cloud.is_empty() {
        return s(1.0);
    }
    let l = &c.look;
    let k = (s(l.cloud_height) - p[1]) * s(1.0 / fmax(0.2, l.sun_dir.y));
    let x = (s(l.sun_dir.x).mul_add(k, p[0]) + s(c.cloud_drift.x)).mul_add(s(1.0 / 12.0), s(-0.5));
    let z = (s(l.sun_dir.z).mul_add(k, p[2]) + s(c.cloud_drift.y)).mul_add(s(1.0 / 12.0), s(-0.5));
    let (fx0, fz0) = (x.floor(), z.floor());
    let (fx, fz) = (x - fx0, z - fz0);
    let (ix, iz) = (fx0.to_int(), fz0.to_int());
    let (mask, one) = (I8::splat(255), I8::splat(1));
    let at = |a: I8, b: I8| F8::gather(c.cloud, (b & mask) * I8::splat(256) + (a & mask));
    let (a, b, cc, d) = (at(ix, iz), at(ix + one, iz), at(ix, iz + one), at(ix + one, iz + one));
    let top = (b - a).mul_add(fx, a);
    let bot = (d - cc).mul_add(fx, cc);
    s(1.0) - s(0.45) * (bot - top).mul_add(fz, top)
}

/// Eight lanes of `fast_log2`.
#[inline(always)]
fn fast_log2_8(v: F8) -> F8 {
    let s = F8::splat;
    let bits = v.bits();
    let e = (bits.shr(I8::splat(23)) & I8::splat(255)) - I8::splat(128);
    let m = ((bits & I8::splat(0x7fffff)) | I8::splat(0x3f800000)).as_f8();
    s(-0.34484843).mul_add(m, s(2.02466578)).mul_add(m, e.to_float() - s(0.67487759))
}

/// One row of a full frame, eight pixels at a time: every opaque surface lit as
/// `Scene::lit_surface` does, then fogged; sky from the quarter-size sky image. Also each
/// pixel's direct sun, its nearest surface's 1/w (water included) and its distance, which
/// the water pass over it needs.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn shade_row(c: &ShadeCtx, y: i32, ids: &[u32], water: &[u32], depth: &[f32], depth_water: &[f32], color: [&mut [f32]; 3],
             sun_out: &mut [f32], near_out: &mut [f32], dist_out: &mut [f32]) {
    let s = F8::splat;
    let zero = s(0.0);
    let one = s(1.0);
    let l = &c.look;
    let t = c.table;
    let lanes = I8::lanes().to_float();
    let no_prim = I8::splat(NO_PRIM as i32);
    let first = I8::load(&c.table.first);
    let py = s(y as f32 + 0.5);
    let sun_light = l.sun_color * l.sun_intensity;
    let ambient = l.ambient * l.ambient_intensity;
    let (ids, water) = (ids.as_chunks::<8>().0, water.as_chunks::<8>().0);
    let (depth, depth_water) = (depth.as_chunks::<8>().0, depth_water.as_chunks::<8>().0);
    let [cr, cg, cb] = color.map(|p| p.as_chunks_mut::<8>().0);
    let (sun_out, near_out, dist_out) = (sun_out.as_chunks_mut::<8>().0, near_out.as_chunks_mut::<8>().0, dist_out.as_chunks_mut::<8>().0);
    // Frame noise for the shadow lookups: interleaved gradient noise, rotated each frame.
    let noise_x = 5.588238 * (c.frame & 63) as f32;
    for m in 0..ids.len() {
        let id = I8::load_u32(&ids[m]);
        let dry = I8::load_u32(&water[m]).eq(no_prim);
        let px = s((8 * m) as f32 + 0.5) + lanes;
        let sky = id.eq(no_prim);
        let ray = [
            s(c.ray_x.x).mul_add(px, s(c.ray_y.x).mul_add(py, s(c.ray_base.x))),
            s(c.ray_x.y).mul_add(px, s(c.ray_y.y).mul_add(py, s(c.ray_base.y))),
            s(c.ray_x.z).mul_add(px, s(c.ray_y.z).mul_add(py, s(c.ray_base.z))),
        ];
        let len = ray[2].mul_add(ray[2], ray[1].mul_add(ray[1], ray[0] * ray[0])).sqrt();
        let mut rgb = [zero; 3];
        let mut sun = zero;
        let mut dist = s(1e9);
        if !sky.all() {
            let id = I8::select(sky, I8::splat(0), id);
            let g = (first.permute(id.shr(I8::splat(26))) + (id & I8::splat(0x3ff_ffff))) * I8::splat(PRIM_STRIDE as i32);
            // Most groups of eight lie on one quad, whose fields are then plain loads: a
            // gather costs this CPU about 13 cycles.
            let one_prim = g.eq(g.first()).all();
            let g0 = g.to_array()[0] as usize;
            let tf = |k: usize| if one_prim { s(t.v[g0 + k]) } else { F8::gather(&t.v, g + I8::splat(k as i32)) };
            let ti = |k: usize| tf(k).bits();
            let a = tf(F_A).mul_add(px, tf(F_A + 1).mul_add(py, tf(F_A + 2)));
            let wv = a.recip();
            let (s0, s1, t0, t1) = (tf(F_S), tf(F_S + 1), tf(F_T), tf(F_T + 1));
            let sv = s0.mul_add(px, s1.mul_add(py, tf(F_S + 2)));
            let tv = t0.mul_add(px, t1.mul_add(py, tf(F_T + 2)));
            let (s_raw, t_raw) = (sv * wv, tv * wv);
            let ss = s_raw.max(zero).min(one);
            let tt = t_raw.max(zero).min(one);
            let mut pos = [ray[0].mul_add(wv, s(c.eye.x)), ray[1].mul_add(wv, s(c.eye.y)), ray[2].mul_add(wv, s(c.eye.z))];
            dist = len * wv;
            // Pixels on a quad's silhouette are covered with a little margin; their point is
            // clamped onto the quad's edge, as the scalar path does, so their shadow lookup
            // does not fall just beyond it.
            let (ds, dt) = (ss - s_raw, tt - t_raw);
            if (ds.ge(s(1e-7)) | ds.le(s(-1e-7)) | dt.ge(s(1e-7)) | dt.le(s(-1e-7))).any() {
                for (k, p) in pos.iter_mut().enumerate() {
                    *p = tf(F_E + k).mul_add(ds, tf(F_E + 3 + k).mul_add(dt, *p));
                }
            }

            // Texture, at the mip level the analytic screen derivatives of s and t pick.
            let (usx, usy, utx, uty) = (tf(F_UV + 2), tf(F_UV + 3), tf(F_UV + 4), tf(F_UV + 5));
            let u = usx.mul_add(ss, utx.mul_add(tt, tf(F_UV)));
            let v = usy.mul_add(ss, uty.mul_add(tt, tf(F_UV + 1)));
            let dsx = (s0 - ss * tf(F_A)) * wv;
            let dtx = (t0 - tt * tf(F_A)) * wv;
            let dsy = (s1 - ss * tf(F_A + 1)) * wv;
            let dty = (t1 - tt * tf(F_A + 1)) * wv;
            let (dux, dvx) = (usx.mul_add(dsx, utx * dtx), usy.mul_add(dsx, uty * dtx));
            let (duy, dvy) = (usx.mul_add(dsy, utx * dty), usy.mul_add(dsy, uty * dty));
            let tex = ti(I_TEX);
            let tex0 = tex.to_array()[0] as usize;
            let per_tex = |table: &[i32]| if one_prim { I8::splat(table[tex0]) } else { I8::gather(table, tex) };
            let size = per_tex(&c.pool.size);
            let sizef = size.to_float();
            let rho = dux.mul_add(dux, dvx * dvx).max(duy.mul_add(duy, dvy * dvy)) * sizef * sizef;
            let level = s(0.5).mul_add(fast_log2_8(rho.max(s(1e-6))), s(MIP_ROUND)).max(zero).to_int();
            let level = level.max(I8::splat(0)).min(per_tex(&c.pool.levels) - I8::splat(1));
            let sz = size.shr(level).max(I8::splat(1));
            let szf = sz.to_float();
            let mask = sz - I8::splat(1);
            let tx = u.mul_add(szf, s(65536.0)).to_int() & mask;
            let ty = v.mul_add(szf, s(65536.0)).to_int() & mask;
            let offset = if one_prim {
                I8::load(c.pool.level_offset[8 * tex0..][..8].try_into().unwrap()).permute(level)
            } else {
                I8::gather(&c.pool.level_offset, tex * I8::splat(8) + level)
            };
            let at = per_tex(&c.pool.base) + offset + ty * sz + tx;
            let texel = I8::gather_u32(&c.pool.texels, at);
            let byte = |k: i32| texel.shr(I8::splat(k)) & I8::splat(255);
            let albedo = [
                F8::gather(c.lut, byte(0)) * tf(F_TINT),
                F8::gather(c.lut, byte(8)) * tf(F_TINT + 1),
                F8::gather(c.lut, byte(16)) * tf(F_TINT + 2),
            ];

            // Light: corners interpolated, or the first corner on a flat quad.
            let mat = ti(I_MAT);
            let flat = (mat & I8::splat((QUAD_FLAT as i32) << 16)).eq(I8::splat(0)).eq(I8::splat(0));
            let w00 = F8::select(flat, one, (one - ss) * (one - tt));
            let w10 = F8::select(flat, zero, ss * (one - tt));
            let w01 = F8::select(flat, zero, (one - ss) * tt);
            let w11 = F8::select(flat, zero, ss * tt);
            let corners = |k: usize| {
                let p = ti(k);
                let b = |sh: i32| (p.shr(I8::splat(sh)) & I8::splat(255)).to_float();
                b(0).mul_add(w00, b(8).mul_add(w10, b(16).mul_add(w01, b(24) * w11))) * s(1.0 / 255.0)
            };
            let (sky_l, blk_l, ao) = (corners(I_SKY), corners(I_BLK), corners(I_AO));
            let n = [tf(F_N), tf(F_N + 1), tf(F_N + 2)];
            let has = |bit: u16| (mat & I8::splat(bit as i32)).eq(I8::splat(0)).eq(I8::splat(0));
            let (plant, leaves) = (has(MAT_PLANT), has(MAT_LEAVES));
            let mut ndl = F8::select(plant, s(0.72), s(l.sun_dir.x).mul_add(n[0], s(l.sun_dir.y).mul_add(n[1], s(l.sun_dir.z) * n[2])));
            ndl = F8::select(leaves, ndl.max(zero).mul_add(s(0.65), s(0.35)), ndl);
            let vis = smooth8(0.45, 0.85, sky_l);
            let lit = ndl.gt(zero) & vis.gt(zero) & sky.eq(I8::splat(0));
            if lit.any() {
                let gx = fract8((px - s(0.5) + s(noise_x)) * s(0.06711056) + s(0.00583715 * y as f32));
                let g = fract8(s(52.9829189) * gx);
                let g2 = fract8(g.mul_add(s(7.123), s(0.37)));
                let normal = [F8::select(plant, zero, n[0]), F8::select(plant, one, n[1]), F8::select(plant, zero, n[2])];
                let sh = shadow8(c, pos, normal, g - s(0.5), g2 - s(0.5)) * cloud8(c, pos);
                sun = F8::select(lit, ndl * vis * sh, zero);
            }
            let sky_amb = sky_l * sky_l * s(0.3).mul_add(sky_l, s(0.7));
            let dir_amb = F8::select(plant, s(0.9), s(0.28).mul_add(n[1], s(0.72)));
            let amb = sky_amb * ao * dir_amb;
            let bounce = sky_amb * ao * s(0.35) * (one - n[1].max(zero));
            let bl = blk_l * blk_l * blk_l * s(2.2) * ao;
            let emissive = has(MAT_EMISSIVE);
            let mut k = zero;
            if emissive.any() {
                let lum = albedo[0] * s(0.2126) + albedo[1] * s(0.7152) + albedo[2] * s(0.0722);
                k = F8::select(emissive, F8::select(has(MAT_LAVA), s(2.5), smooth8(0.35, 0.75, lum) * s(3.0)), zero);
            }
            let chan = |ch: usize, sl: f32, am: f32, bo: f32, bk: f32, floor: f32| {
                let light = s(sl).mul_add(sun, s(am).mul_add(amb, s(bo).mul_add(bounce, s(bk).mul_add(bl, s(floor)))));
                albedo[ch] * light + albedo[ch] * k
            };
            rgb = [
                chan(0, sun_light.x, ambient.x, 0.30, l.block_light.x, 0.02),
                chan(1, sun_light.y, ambient.y, 0.26, l.block_light.y, 0.02),
                chan(2, sun_light.z, ambient.z, 0.20, l.block_light.z, 0.025),
            ];
            let fog = smooth8(l.fog_start, l.fog_end, dist);
            if fog.gt(zero).any() {
                let inv = len.recip();
                let g = sky_gradient8(c, ray[0] * inv, ray[1] * inv, ray[2] * inv);
                for ch in 0..3 {
                    rgb[ch] = (g[ch] - rgb[ch]).mul_add(fog, rgb[ch]);
                }
            }
        }
        if sky.any() {
            // Sky and clouds, from the quarter-size image.
            let (sx, sy) = (px.mul_add(s(0.25), s(0.5)), py.mul_add(s(0.25), s(0.5)));
            let (ix, iy) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - ix, sy - iy);
            let r0 = iy.mul_add(s(c.sky_w as f32), ix).to_int();
            let r1 = r0 + I8::splat(c.sky_w);
            let o = I8::splat(1);
            for (ch, p) in c.sky.iter().enumerate() {
                let (a, b, cc, d) = (F8::gather(p, r0), F8::gather(p, r0 + o), F8::gather(p, r1), F8::gather(p, r1 + o));
                let top = (b - a).mul_add(fx, a);
                let bot = (d - cc).mul_add(fx, cc);
                rgb[ch] = F8::select(sky, (bot - top).mul_add(fy, top), rgb[ch]);
            }
        }
        rgb[0].store(&mut cr[m]);
        rgb[1].store(&mut cg[m]);
        rgb[2].store(&mut cb[m]);
        sun.store(&mut sun_out[m]);
        dist.store(&mut dist_out[m]);
        // Nearest surface, water included: what the reprojection of this frame sees.
        F8::select(dry, F8::select(sky, zero, F8::load(&depth[m])), F8::load(&depth_water[m])).store(&mut near_out[m]);
    }
}

/// The cameras, as each row of the compose pass needs them.
struct ComposeView {
    w: i32,
    h: i32,
    eye: Vec3,
    ray_base: Vec3,
    ray_x: Vec3,
    ray_y: Vec3,
    // The last full frame's camera.
    s_ray_base: Vec3,
    s_ray_x: Vec3,
    s_ray_y: Vec3,
    s_jitter_x: f32,
    s_jitter_y: f32,
    // Pixels a surface slides against another per unit of 1/w between them, the camera
    // having moved as far as it has since the last full frame.
    parallax: f32,
    // The same for the camera's move since the full frame before this one.
    b_parallax: f32,
    // The full frame before, as a projection: pixel x + 0.5 = dot(d, bx) / dot(d, bz) - cx
    // for d = p - b_eye, and the same for y; dot(d, bz) is the point's w there.
    b_eye: Vec3,
    bx: Vec3,
    by: Vec3,
    bz: Vec3,
    bcx: f32,
    bcy: f32,
    // Last frame's projection, the viewport folded into x and y:
    // pixel x = dot(p0, p) / dot(p3, p) + w / 2.
    p0: Vec4,
    p1: Vec4,
    p3: Vec4,
    // This frame's and last frame's projections of the last full frame's pixels.
    u0: Through,
    u1: Through,
    u3: Through,
    m0: Through,
    m1: Through,
    m3: Through,
}

/// A row of a projection applied to the last full frame's pixel (fx, fy) at 1/w = d,
/// times d: fx a + fy b + c + d e. The world point in between cancels out, and with it
/// a division and most of the work of each warp step.
#[derive(Clone, Copy)]
struct Through {
    a: f32,
    b: f32,
    c: f32,
    e: f32,
}

impl Through {
    fn new(r: Vec4, st: &StaticView) -> Through {
        let n = Vec3::new(r.x, r.y, r.z);
        Through { a: dot(n, st.ray_x), b: dot(n, st.ray_y), c: dot(n, st.ray_base), e: dot(n, st.eye) + r.w }
    }

    #[inline(always)]
    fn at(self, fx: F8, fy: F8, d: F8) -> F8 {
        let s = F8::splat;
        s(self.e).mul_add(d, s(self.a).mul_add(fx, s(self.b).mul_add(fy, s(self.c))))
    }
}

#[inline(always)]
fn dot4(r: Vec4, x: F8, y: F8, z: F8) -> F8 {
    let s = F8::splat;
    s(r.x).mul_add(x, s(r.y).mul_add(y, s(r.z).mul_add(z, s(r.w))))
}

/// World points along the rays of pixels (fx, fy) at 1/w = d, or far out for sky (d = 0).
#[inline(always)]
fn along(eye: Vec3, base: Vec3, rx: Vec3, ry: Vec3, fx: F8, fy: F8, d: F8) -> (F8, F8, F8) {
    let s = F8::splat;
    let x = s(rx.x).mul_add(fx, s(ry.x).mul_add(fy, s(base.x)));
    let y = s(rx.y).mul_add(fx, s(ry.y).mul_add(fy, s(base.y)));
    let z = s(rx.z).mul_add(fx, s(ry.z).mul_add(fy, s(base.z)));
    let sky = d.le(s(0.0));
    let mut scale = F8::select(sky, s(1.0), d).recip();
    if sky.any() {
        scale = F8::select(sky, s(4000.0) * z.mul_add(z, y.mul_add(y, x * x)).rsqrt(), scale);
    }
    (x.mul_add(scale, s(eye.x)), y.mul_add(scale, s(eye.y)), z.mul_add(scale, s(eye.z)))
}

/// Where points were last frame, in pixels; (-1, -1) behind the camera.
#[inline(always)]
fn motion_of(v: &ComposeView, px: F8, py: F8, pz: F8, mx: &mut [f32; 8], my: &mut [f32; 8]) {
    let s = F8::splat;
    let mw = dot4(v.p3, px, py, pz);
    let minv = mw.max(s(1e-6)).recip();
    let bad = mw.le(s(0.0));
    F8::select(bad, s(-1.0), dot4(v.p0, px, py, pz).mul_add(minv, s(0.5 * v.w as f32 - 0.5))).store(mx);
    F8::select(bad, s(-1.0), dot4(v.p1, px, py, pz).mul_add(minv, s(0.5 * v.h as f32 - 0.5))).store(my);
}

/// Whether the full frame before this one showed the surface at (px, py, pz), rather than
/// something nearer that has since moved off it; `before` holds that frame's depths.
/// Ground a bamboo stalk has just uncovered is no still pixel, whatever the depth edges
/// say: the history there is the stalk, and blended in unclamped the stalk stayed on it
/// as a ghost for over a second. Between full frames still pixels take no history, so
/// only full frames ask.
#[inline(always)]
fn shown_before(v: &ComposeView, before: &[f32], px: F8, py: F8, pz: F8) -> I8 {
    let s = F8::splat;
    let zero = s(0.0);
    let (dx, dy, dz) = (px - s(v.b_eye.x), py - s(v.b_eye.y), pz - s(v.b_eye.z));
    let along = |a: Vec3| s(a.x).mul_add(dx, s(a.y).mul_add(dy, s(a.z) * dz));
    let z = along(v.bz);
    let inv = z.max(s(NEAR)).recip();
    let x = along(v.bx).mul_add(inv, s(-v.bcx)).floor();
    let y = along(v.by).mul_add(inv, s(-v.bcy)).floor();
    let inside = z.gt(s(NEAR)) & x.ge(zero) & y.ge(zero) & x.lt(s(v.w as f32)) & y.lt(s(v.h as f32));
    let idx = y.max(zero).min(s((v.h - 1) as f32)).mul_add(s(v.w as f32), x.max(zero).min(s((v.w - 1) as f32))).to_int();
    let was = pick(before, idx).max(zero);
    // Outside that frame, or behind its camera: nothing to go by, so as before.
    was.le(inv * s(1.0 + EDGE_SPREAD)) | inside.eq(I8::splat(0))
}

/// A full frame's row: its own shading, at its own jittered rays. `inner` gets 1 where
/// the pixel is away from depth edges and the full frame before showed the same surface:
/// `before` holds that frame's depths, and this row's nearest and farthest surfaces near
/// each pixel, which tell where a thin thing can have uncovered anything.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn identity_row(v: &ComposeView, y: i32, depth: &[f32], before: Option<(&[f32], &[f32], &[f32])>, near: &mut [f32], inner: &mut [f32],
                mx: &mut [f32], my: &mut [f32]) {
    let s = F8::splat;
    let lanes = I8::lanes().to_float();
    let fy = s(y as f32 + 0.5);
    let depth = depth.as_chunks::<8>().0;
    let (near, inner) = (near.as_chunks_mut::<8>().0, inner.as_chunks_mut::<8>().0);
    let (mx, my) = (mx.as_chunks_mut::<8>().0, my.as_chunks_mut::<8>().0);
    for m in 0..depth.len() {
        let d = F8::load(&depth[m]);
        let (px, py, pz) = along(v.eye, v.ray_base, v.ray_x, v.ray_y, s((8 * m) as f32 + 0.5) + lanes, fy, d);
        d.store(&mut near[m]);
        motion_of(v, px, py, pz, &mut mx[m], &mut my[m]);
        let mut ok = off_edge(d);
        if let Some((before, nearest, farthest)) = before
            && (nearest[m] - farthest[m]) * v.b_parallax > 0.5
        {
            ok = ok & shown_before(v, before, px, py, pz);
        }
        F8::select(ok, s(1.0), s(0.0)).store(&mut inner[m]);
    }
}

/// How far along a static row, either way, the warp looks for a thin near thing that has
/// slid off a pixel, or for the ground behind one that has slid off it. Six in-between
/// frames at the fastest parallax on the orbit, a bamboo stalk near the camera against
/// ground well behind it, come to about three pixels.
const WARP_REACH: i32 = 8;

/// An in-between frame's row: for each pixel, the sub-pixel point (qx, qy) in the last
/// full frame whose surface projects here, found by fixed-point iteration on the static
/// depth; the camera moves under a pixel per frame, so two steps converge. Everything
/// is then taken at that exact point: snapping to the nearest static pixel made each
/// region hold still and then jump a pixel, at different moments for different depths,
/// which read as heat haze.
///
/// Started from the surface under the pixel, the iteration cannot find a thing thinner
/// than the distance it has slid against what is behind it: a bamboo stalk stayed where
/// the last full frame had it, then jumped at the next one, and left a fading copy at
/// every place it had stood. So where a nearer surface is close by, a second search holds
/// that depth, and wins where the static image does have that surface at the point it
/// finds, as in a depth test. Where the first search does not settle, a third holds the
/// farthest nearby depth, the ground a thin thing has uncovered; and if the static image
/// never saw that ground, it is filled from the nearest ground pixel along the row.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn warp_row(v: &ComposeView, y: i32, s_depth: &[f32], reach: (&[f32], &[f32]), mut sun: Option<(&[f32], &mut [f32])>, near: &mut [f32],
            inner: &mut [f32], qx_out: &mut [f32], qy_out: &mut [f32], mx: &mut [f32], my: &mut [f32]) {
    let s = F8::splat;
    let zero = s(0.0);
    let lanes = I8::lanes().to_float();
    let (w, wu) = (v.w, v.w as usize);
    let (wf, hw, hh) = (s(w as f32), s(0.5 * w as f32), s(0.5 * v.h as f32));
    let (max_x, max_y) = (s((w - 1) as f32), s((v.h - 1) as f32));
    let uy = s(y as f32 + 0.5);
    let (half, spread) = (s(0.5), s(EDGE_SPREAD));
    let abs = |x: F8| x.max(zero - x);
    let none = I8::splat(0);
    let index = |fx: F8, fy: F8| fy.max(zero).min(max_y).floor().mul_add(wf, fx.max(zero).min(max_x)).to_int();
    // A static 1/w as the warp uses it: sky is taken 4000 blocks out along the ray.
    let distance = |d: F8, fx: F8, fy: F8| {
        let sky = d.le(zero);
        if !sky.any() {
            return d;
        }
        let r = |c: fn(Vec3) -> f32| s(c(v.s_ray_x)).mul_add(fx, s(c(v.s_ray_y)).mul_add(fy, s(c(v.s_ray_base))));
        let (rx, ry, rz) = (r(|a| a.x), r(|a| a.y), r(|a| a.z));
        F8::select(sky, rz.mul_add(rz, ry.mul_add(ry, rx * rx)).sqrt() * s(1.0 / 4000.0), d)
    };
    // How far from the pixel the static point at (fx, fy), taken at 1/w dd, lands now; and
    // its w here.
    let miss = |fx: F8, fy: F8, dd: F8, ux: F8| {
        let cw = v.u3.at(fx, fy, dd);
        let inv = cw.max(s(NEAR) * dd).recip();
        (ux - v.u0.at(fx, fy, dd).mul_add(inv, hw), uy - v.u1.at(fx, fy, dd).mul_add(inv, hh), cw)
    };
    // The search holding one depth: where it ends, the depth the static image has there,
    // and whether that is the depth held.
    let hold = |dk: F8, qx0: F8, qy0: F8, ux: F8| {
        let (mut qx, mut qy) = (qx0, qy0);
        for _ in 0..2 {
            let (rx, ry, _) = miss(qx + half, qy + half, dk, ux);
            (qx, qy) = (qx + rx, qy + ry);
        }
        let d = pick(s_depth, index(qx + half, qy + half));
        (qx, qy, d, dk.gt(zero) & abs(d.max(zero) - dk).le(dk * spread))
    };
    let row0 = (y as f32 + 0.5 - v.s_jitter_y).floor().clamp(0.0, (v.h - 1) as f32) as usize * wu;
    let (nearest_row, farthest_row) = (&reach.0[row0 / 8..(row0 + wu) / 8], &reach.1[row0 / 8..(row0 + wu) / 8]);
    let (near, inner) = (near.as_chunks_mut::<8>().0, inner.as_chunks_mut::<8>().0);
    let (qx_out, qy_out) = (qx_out.as_chunks_mut::<8>().0, qy_out.as_chunks_mut::<8>().0);
    let (mx, my) = (mx.as_chunks_mut::<8>().0, my.as_chunks_mut::<8>().0);
    // Groups of eight pixels searched side by side: each search is one long chain of
    // dependent steps, and two in flight keep the core busy.
    const PAIR: usize = 2;
    for m0 in (0..near.len()).step_by(PAIR) {
        let ux2: [F8; PAIR] = std::array::from_fn(|g| s((8 * (m0 + g)) as f32 + 0.5) + lanes);
        let (mut qx2, mut qy2) = (ux2.map(|u| u - s(0.5 + v.s_jitter_x)), [uy - s(0.5 + v.s_jitter_y); PAIR]);
        let (mut d2, mut dd2, mut settled2, mut under2) = ([zero; PAIR], [zero; PAIR], [zero; PAIR], [zero; PAIR]);
        // The nearest and farthest surfaces along the static row around each pixel, and
        // whether the camera can have moved one against the other by half a pixel since the
        // last full frame. Nowhere else can the plain search go wrong, and early in each
        // cycle that is nowhere at all.
        let reach2: [(F8, F8); PAIR] = std::array::from_fn(|g| (s(nearest_row[m0 + g]), s(farthest_row[m0 + g])));
        let gate: [bool; PAIR] = std::array::from_fn(|g| (nearest_row[m0 + g] - farthest_row[m0 + g]) * v.parallax > 0.5);
        for iter in 0..3 {
            for g in 0..PAIR {
                let (fx, fy) = (qx2[g] + half, qy2[g] + half);
                d2[g] = pick(s_depth, index(fx, fy));
                dd2[g] = distance(d2[g], fx, fy);
                if iter == 0 {
                    under2[g] = d2[g].max(zero);
                }
                if iter < 2 {
                    let (rx, ry, _) = miss(fx, fy, dd2[g], ux2[g]);
                    (qx2[g], qy2[g]) = (qx2[g] + rx, qy2[g] + ry);
                } else if gate[g] {
                    let (rx, ry, _) = miss(fx, fy, dd2[g], ux2[g]);
                    settled2[g] = abs(rx).max(abs(ry));
                }
            }
        }
        for g in 0..PAIR {
            let m = m0 + g;
            let ux = ux2[g];
            let (qx0, qy0) = (ux - s(0.5 + v.s_jitter_x), uy - s(0.5 + v.s_jitter_y));
            let (mut qx, mut qy, mut d, mut dd, settled, under) = (qx2[g], qy2[g], d2[g], dd2[g], settled2[g], under2[g]);
            let mut ok = settled.lt(half);
            let mut moved = none; // taken from another search: not a still pixel
            let (nearest, far) = reach2[g];
            if gate[g] && (nearest - under).gt(s(0.5 / v.parallax.max(1e-6))).any() {
                let (bx, by, bd, found) = hold(nearest, qx0, qy0, ux);
                let take = found & (ok.eq(none) | nearest.gt(dd * s(1.0 + EDGE_SPREAD)));
                if take.any() {
                    (qx, qy, d, dd) = (F8::select(take, bx, qx), F8::select(take, by, qy), F8::select(take, bd, d), F8::select(take, nearest, dd));
                    ok = ok | take;
                    moved = moved | take;
                }
            }
            let lost = ok.eq(none) & far.gt(zero);
            if gate[g] && lost.any() {
                let (cx, cy, cd, found) = hold(far, qx0, qy0, ux);
                // Never seen: the nearest pixel along the row that shows that ground.
                let mut fill = found;
                let mut fx = cx;
                let mut fd = cd;
                let mut k = 1;
                while (lost & fill.eq(none)).any() && k <= WARP_REACH {
                    for step in [k as f32, -(k as f32)] {
                        let tx = cx + s(step);
                        let td = pick(s_depth, index(tx + half, cy + half));
                        let hit = fill.eq(none) & abs(td.max(zero) - far).le(far * spread);
                        fx = F8::select(hit, tx, fx);
                        fd = F8::select(hit, td, fd);
                        fill = fill | hit;
                    }
                    k += 1;
                }
                let take = lost & fill;
                (qx, qy, d, dd) = (F8::select(take, fx, qx), F8::select(take, cy, qy), F8::select(take, fd, d), F8::select(take, far, dd));
                moved = moved | take;
            }
            let (fx, fy) = (qx + half, qy + half);
            let idx = index(fx, fy);
            let cw = v.u3.at(fx, fy, dd);
            let near_d = s(NEAR) * dd;
            F8::select(d.le(zero) | cw.le(near_d), zero, dd * cw.max(near_d).recip()).store(&mut near[m]);
            if let Some((s_sun, sun)) = sun.as_mut() {
                pick(s_sun, idx).store((&mut sun[8 * m..8 * m + 8]).try_into().unwrap());
            }
            let mw = v.m3.at(fx, fy, dd);
            let minv = mw.max(s(1e-6) * dd).recip();
            let bad = mw.le(zero);
            let hx = F8::select(bad, s(-1.0), v.m0.at(fx, fy, dd).mul_add(minv, s(0.5 * w as f32 - 0.5)));
            let hy = F8::select(bad, s(-1.0), v.m1.at(fx, fy, dd).mul_add(minv, s(0.5 * v.h as f32 - 0.5)));
            F8::select(off_edge(d) & moved.eq(none), s(1.0), zero).store(&mut inner[m]);
            hx.store(&mut mx[m]);
            hy.store(&mut my[m]);
            qx.store(&mut qx_out[m]);
            qy.store(&mut qy_out[m]);
        }
    }
}

/// The static colour at each warped point. Catmull-Rom, not bilinear: bilinear softened
/// the in-between frames against the full ones, a 5 % sharpness pulse at 6 Hz.
#[inline(never)]
fn filter_row(w: i32, h: i32, s_color: &Halves, qx: &[f32], qy: &[f32], cur: [&mut [f32]; 3]) {
    let s = F8::splat;
    let zero = s(0.0);
    let wu = w as usize;
    let lanes = I8::lanes();
    let (max_x, max_y, zero_i, wi) = (I8::splat(w - 1), I8::splat(h - 1), I8::splat(0), I8::splat(w));
    let (qx, qy) = (qx.as_chunks::<8>().0, qy.as_chunks::<8>().0);
    let [cr, cg, cb] = cur.map(|c| c.as_chunks_mut::<8>().0);
    for m in 0..qx.len() {
        let bx = F8::load(&qx[m]).max(zero).min(s((w - 1) as f32 - 1e-3));
        let by = F8::load(&qy[m]).max(zero).min(s((h - 1) as f32 - 1e-3));
        let (fxl, fyl) = (bx.floor(), by.floor());
        let wx = catmull_rom(bx - fxl);
        let wy = catmull_rom(by - fyl);
        let (jx, jy) = (fxl.to_int(), fyl.to_int());
        let first = jx.first();
        let (left, top) = (first.to_array()[0], jy.to_array()[0]);
        let straight = jy.eq(jy.first()).all() && jx.eq(first + lanes).all() && left >= 1 && left + 10 <= w && top >= 1 && top + 2 < h;
        let dst = [&mut cr[m], &mut cg[m], &mut cb[m]];
        if straight {
            let base = ((top - 1) * w + left - 1) as usize;
            for (plane, d) in s_color.iter().zip(dst) {
                // The four rows of eleven texels the taps read.
                let t: [&[u16; 11]; 4] = std::array::from_fn(|r| plane[base + r * wu..][..11].try_into().unwrap());
                filter_cross(|r, k| F8::load_f16(t[r][k..k + 8].try_into().unwrap()), &wx, &wy).store(d);
            }
        } else if let Some((a, b, in_a)) = two_windows(jx, jy, w, h) {
            for (plane, d) in s_color.iter().zip(dst) {
                let load = |at: usize| F8::load_f16(plane[at..][..8].try_into().unwrap());
                filter_cross(|r, k| F8::select(in_a, load(a + r * wu + k), load(b + r * wu + k)), &wx, &wy).store(d);
            }
        } else {
            // Taps at rows jy - 1 ..= jy + 2 and columns jx - 1 ..= jx + 2, clamped to the
            // image; jx <= w - 2 and jy <= h - 2 already.
            let rows: [I8; 4] = std::array::from_fn(|r| (jy + I8::splat(r as i32 - 1)).max(zero_i).min(max_y) * wi);
            let cols: [I8; 4] = std::array::from_fn(|k| (jx + I8::splat(k as i32 - 1)).max(zero_i).min(max_x));
            for (plane, d) in s_color.iter().zip(dst) {
                filter_cross(|r, k| F8::gather_f16(plane, rows[r] + cols[k]), &wx, &wy).store(d);
            }
        }
    }
}

/// Players shadow the ground in their true silhouette: each receiving pixel near a
/// player is traced towards the sun against that player's body quads. The sun direction
/// is jittered per pixel and frame, which the resolve turns into a soft edge. True if
/// any pixel of the row was shadowed.
#[allow(clippy::too_many_arguments)]
fn player_shadows(v: &ComposeView, y: i32, scene: &Scene, casters: &[Caster], near: &[f32], sun: &[f32], mask: &mut [u8],
                  cur: &mut [&mut [f32]; 3]) -> bool {
    let py = y as f32 + 0.5;
    let sun_dir = scene.look.sun_dir;
    let inv = |d: f32| 1.0 / if d.abs() < 1e-6 { 1e-6f32.copysign(d) } else { d };
    let inv_sun = Vec3::new(inv(sun_dir.x), inv(sun_dir.y), inv(sun_dir.z));
    let mut any = false;
    for c in casters.iter().filter(|c| (c.y0..=c.y1).contains(&y)) {
        for x in c.x0..=c.x1 {
            let xu = x as usize;
            let a = near[xu];
            if a <= 0.0 || sun[xu] <= 0.01 || mask[xu] != 0 {
                continue;
            }
            let p = v.eye + (v.ray_base + v.ray_x * (x as f32 + 0.5) + v.ray_y * py) * (1.0 / a);
            if p.x < c.lo.x || p.y < c.lo.y || p.z < c.lo.z || p.x > c.hi.x || p.y > c.hi.y || p.z > c.hi.z {
                continue;
            }
            // Most of the rectangle is ground the shadow never reaches: a ray towards the
            // sun that misses the body's box misses every quad in it.
            let (t0, t1) = ((c.body_lo - p) * inv_sun, (c.body_hi - p) * inv_sun);
            let (near_t, far_t) = (vmin(t0, t1), vmax(t0, t1));
            if fmin(far_t.x, fmin(far_t.y, far_t.z)) < fmax(0.0, fmax(near_t.x, fmax(near_t.y, near_t.z))) {
                continue;
            }
            let g = fract(52.9829189 * fract(0.06711056 * (x as f32 + 5.588238 * (scene.frame & 63) as f32) + 0.00583715 * y as f32));
            let dir = normalize(sun_dir + Vec3::new((g - 0.5) * 0.05, 0.0, (fract(g * 7.123) - 0.5) * 0.05));
            for q in &c.near {
                let denom = dot(q.n, dir);
                if denom.abs() < 1e-9 {
                    continue;
                }
                let lambda = dot(q.n, q.p - p) / denom;
                if !(0.03..=12.0).contains(&lambda) {
                    continue;
                }
                let hit = p + dir * lambda - q.p;
                let (s, t) = (dot(hit, q.u1), dot(hit, q.u2));
                if s < 0.0 || t < 0.0 || s > 1.0 || t > 1.0 {
                    continue;
                }
                let k = 1.0 - 0.62 * fmin(1.0, sun[xu]);
                for p in cur.iter_mut() {
                    p[xu + 1] *= k;
                }
                mask[xu] = 2;
                any = true;
                break;
            }
        }
    }
    any
}

/// Adds a row of linear colour to the band's share of the next frame's bright pass, an
/// eighth of the frame's size: only what is brighter than sunlit white, so wool and snow
/// stay crisp while the sun, its glint and lamps glow. Rows come in pairs, odd ones taken.
fn bright_row(src: [&[f32]; 3], y: i32, y0: i32, bw: usize, mut bright: [&mut [f32]; 3]) {
    let by = (y / 8 - y0 / 8) as usize;
    // Eight texels at a time: each takes the odd columns of its eight
    // pixels, and a tree of pair sums adds up all eight at once.
    let sp = F8::splat;
    let odd = F8::load(&[0.0, 0.0625, 0.0, 0.0625, 0.0, 0.0625, 0.0, 0.0625]);
    let groups = bw / 8;
    for g in 0..groups {
        let part: [[F8; 3]; 8] = std::array::from_fn(|k| {
            let x = 64 * g + 8 * k;
            let c = [at8(src[0], x), at8(src[1], x), at8(src[2], x)];
            let lum = c[0].mul_add(sp(0.2126), c[1].mul_add(sp(0.7152), c[2] * sp(0.0722)));
            let k = (lum - sp(1.15)).max(sp(0.0)).min(sp(3.0)) * lum.max(sp(1e-4)).recip() * odd;
            c.map(|v| v * k)
        });
        for (ch, p) in bright.iter_mut().enumerate() {
            let pair = |a: usize| F8::pair_sums(part[a][ch], part[a + 1][ch]);
            let sums = F8::pair_sums(F8::pair_sums(pair(0), pair(2)), F8::pair_sums(pair(4), pair(6)));
            let o: &mut [f32; 8] = (&mut p[by * bw + 8 * g..][..8]).try_into().unwrap();
            if y & 7 == 1 { sums } else { F8::load(o) + sums }.store(o);
        }
    }
    for bx in 8 * groups..bw {
        let mut sum = Vec3::ZERO;
        for x in [8 * bx + 1, 8 * bx + 3, 8 * bx + 5, 8 * bx + 7] {
            let c = Vec3::new(src[0][x], src[1][x], src[2][x]);
            let lum = luminance(c);
            let over = fmin(3.0, fmax(0.0, lum - 1.15));
            sum += c * (over / fmax(lum, 1e-4));
        }
        let o = by * bw + bx;
        for (p, v) in bright.iter_mut().zip([sum.x, sum.y, sum.z]) {
            p[o] = if y & 7 == 1 { v * 0.0625 } else { p[o] + v * 0.0625 };
        }
    }
}

/// Adds 0.6 of the bloom, bilinear from two of its rows at fraction fy, to a row.
#[inline(never)]
fn add_bloom(rows: (&[f32], &[f32]), fy: f32, line: &mut Vec<f32>, dst: &mut [f32]) {
    // Output pixel 8m + j sits between bloom texels m - 1 + LO[j] and the next, at a
    // fraction that repeats every eight pixels. The two rows are blended into `line`
    // first, shifted by one and with the edge texels repeated past both ends, which is
    // where the clamped samples at the image's edges land.
    const LO: [i32; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    const FRAC: [f32; 8] = [0.5625, 0.6875, 0.8125, 0.9375, 0.0625, 0.1875, 0.3125, 0.4375];
    let s = F8::splat;
    let (b0, b1) = rows;
    let bw = b0.len();
    line.clear();
    line.push(mix(b0[0], b1[0], fy));
    line.extend(b0.iter().zip(b1).map(|(&a, &b)| (b - a).mul_add(fy, a)));
    line.resize(bw + 9, line[bw]);
    let (lo_idx, hi_idx, frac) = (I8::load(&LO), I8::load(&LO) + I8::splat(1), F8::load(&FRAC));
    for (m, d) in dst.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let v = at8(line, m);
        let (lo, hi) = (v.permute(lo_idx), v.permute(hi_idx));
        (hi - lo).mul_add(frac, lo).mul_add(s(0.6), F8::load(d)).store(d);
    }
}

struct TaaConst {
    w: i32,
    h: i32,
    jx: f32,
    jy: f32,
    alpha: f32,
    full: bool,
    exposure: f32,
    saturation: f32,
    history_valid: bool,
    srgb: Srgb16,
}

/// The nearest and farthest surface (largest and smallest 1/w, sky counting as neither,
/// 0 if there is only sky) within WARP_REACH of each group of eight pixels of a row of
/// static depths, whose length is a multiple of eight. Per group rather than per pixel:
/// the warp decides per group anyway, and a full-size pair of planes cost a millisecond
/// or two of memory traffic every full frame.
fn reach_row(row: &[f32], nearest: &mut [f32], farthest: &mut [f32]) {
    const _: () = assert!(WARP_REACH == 8, "the windows below are a group either side");
    let s = F8::splat;
    let groups = row.len() / 8;
    for g in 0..groups {
        let (mut near, mut far) = (s(0.0), s(f32::MAX));
        for k in [g.max(1) - 1, g, (g + 1).min(groups - 1)] {
            let t = at8(row, 8 * k).max(s(0.0));
            near = near.max(t);
            far = F8::select(t.gt(s(0.0)), far.min(t), far);
        }
        let (mut n, mut f) = ([0.0f32; 8], [0.0f32; 8]);
        near.store(&mut n);
        far.store(&mut f);
        nearest[g] = n.into_iter().fold(0.0, f32::max);
        let far = f.into_iter().fold(f32::MAX, f32::min);
        farthest[g] = if far < f32::MAX { far } else { 0.0 };
    }
}

/// Static depths away from a depth edge: `mark_edges` leaves the lowest bit of a surface's
/// 1/w clear there, and sky's zero positive.
#[inline(always)]
fn off_edge(d: F8) -> I8 {
    (d.bits() & I8::splat(i32::MIN | 1)).eq(I8::splat(0))
}

/// Which of a row's pixels may blend unclamped: away from depth edges (`inner`), from
/// players and their shadows now (`mask`), and from where one was lately (`reach`,
/// per group of eight; none when no player was near this row).
fn still_row(reach: Option<&[u8]>, inner: &[f32], mask: &[u8], still: &mut [u8]) {
    let n = inner.len() / 8;
    let (inner, mask, still) = (&inner.as_chunks::<8>().0[..n], &mask.as_chunks::<8>().0[..n], &mut still.as_chunks_mut::<8>().0[..n]);
    for m in 0..n {
        if reach.is_some_and(|r| r[m] != 0) {
            still[m] = [0; 8];
            continue;
        }
        let free = I8::load_u8(&mask[m]).eq(I8::splat(0));
        let ok = free & F8::load(&inner[m]).gt(F8::splat(0.5));
        still[m] = (ok & I8::splat(1)).to_bytes();
    }
}

/// The temporal resolve of one row, eight pixels at a time: this frame blended into the
/// reprojected history, clamped to this frame's neighbourhood unless `still` says the
/// pixel shows last frame's surface. Then exposure, tonemap, grade and dithered sRGB.
/// `near` holds each channel's rows above, at and below, padded; `masks` the same rows'
/// player (1) and shadow (2) marks, where there are any.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn taa_row(t: &TaaConst, y: i32, near: [[&[f32]; 3]; 3], motion: (&[f32], &[f32]), still: &[u8], masks: Option<[&[u8]; 3]>,
           history: &Halves, past: Option<[&mut [u16]; 3]>, next: [&mut [u16]; 3], out: [&mut [u8]; 3], vig: &[f32]) {
    let s = F8::splat;
    let zero = s(0.0);
    let (w, wu) = (t.w, t.w as usize);
    let (wf, max_x, max_y) = (s(w as f32), s((w - 1) as f32), s((t.h - 1) as f32));
    let lanes = I8::lanes();
    let aces = |v: F8| {
        let v = v * s(t.exposure);
        let n = v * s(2.51).mul_add(v, s(0.03));
        let d = v.mul_add(s(2.43).mul_add(v, s(0.59)), s(0.14));
        (n * d.recip()).max(zero).min(s(1.0))
    };
    let vy = (y as f32 + 0.5) / t.h as f32 - 0.5;
    let vy2 = s(vy * vy);
    let bayer = &BAYER[(y as usize & 3) * 4..][..4];
    let bayer = I8::load(&std::array::from_fn(|k| bayer[k & 3] as i32));
    // Every stream as groups of eight cut to the same length, so indexing them needs no
    // bounds checks: with some thirty streams, those were a third of the instructions.
    let n = wu / 8;
    let cross = near.map(|[up, mid, dn]| [&up[1..], mid, &mid[1..], &mid[2..], &dn[1..]].map(|r| &r.as_chunks::<8>().0[..n]));
    let (mxr, myr) = (&motion.0.as_chunks::<8>().0[..n], &motion.1.as_chunks::<8>().0[..n]);
    let still = &still.as_chunks::<8>().0[..n];
    let vig = &vig.as_chunks::<8>().0[..n];
    let mut next = next.map(|p| &mut p.as_chunks_mut::<8>().0[..n]);
    let mut past = past.map(|p| p.map(|p| &mut p.as_chunks_mut::<8>().0[..n]));
    let mut out = out.map(|o| &mut o.as_chunks_mut::<8>().0[..n]);
    for m in 0..n {
        let x = 8 * m;
        let cur: [F8; 3] = std::array::from_fn(|ch| F8::load(&cross[ch][2][m]));
        let mut res = cur;
        let mut settled = cur;
        // Between full frames a still pixel's history is in `blend` already, so eight of
        // them need nothing more.
        let all_still = !t.full && masks.is_none() && u64::from_ne_bytes(still[m]) == 0x0101_0101_0101_0101;
        if t.history_valid && !all_still {
            // Where this pixel's centre was last frame: the motion is for the jittered
            // sample.
            let mut hx = F8::load(&mxr[m]) - s(t.jx);
            let mut hy = F8::load(&myr[m]) - s(t.jy);
            let valid = hx.ge(zero) & hy.ge(zero) & hx.lt(max_x) & hy.lt(max_y);
            if valid.any() {
                let mut mn = [zero; 3];
                let mut mx = [zero; 3];
                for ch in 0..3 {
                    let [u, lf, c, rt, d] = cross[ch].map(|r| F8::load(&r[m]));
                    mn[ch] = c.min(lf).min(rt.min(u)).min(d);
                    mx[ch] = c.max(lf).max(rt.max(u)).max(d);
                }
                hx = F8::select(valid, hx, zero);
                hy = F8::select(valid, hy, zero);
                let (fxl, fyl) = (hx.floor(), hy.floor());
                let (fx, fy) = (hx - fxl, hy - fyl);
                let i00 = fyl.mul_add(wf, fxl).to_int();
                let first = i00.first();
                let straight = i00.eq(first + lanes).all();
                let at = first.to_array()[0] as usize;
                let mut same = I8::load_u8(&still[m]).eq(I8::splat(1));
                let mut alpha = F8::select(same, s(ALPHA_STILL), s(t.alpha));
                // Moving players and their shadows: trust this frame, or they smear.
                if let Some([up, mid, dn]) = masks {
                    let by = I8::load(&std::array::from_fn(|k| {
                        let xi = x + k;
                        (mid[xi] | mid[xi.max(1) - 1] | mid[(wu - 1).min(xi + 1)] | up[xi] | dn[xi] != 0) as i32
                    }));
                    alpha = F8::select(by.eq(I8::splat(1)), s(0.65), alpha);
                    same = same & by.eq(I8::splat(0));
                }
                // Catmull-Rom, like the static image: resampled bilinearly every frame, the
                // history softened a little more each time and the picture never looked
                // its resolution.
                let (wx, wy) = (catmull_rom(fx), catmull_rom(fy));
                let (fxi, fyi) = (fxl.to_int(), fyl.to_int());
                let (col, row) = (fxi.to_array()[0], fyi.to_array()[0]);
                let inner = straight && col >= 1 && col + 10 <= w && row >= 1 && row + 2 < t.h;
                let two = if inner { None } else { two_windows(fxi, fyi, w, t.h) };
                let edge = (!inner && two.is_none()).then(|| {
                    let rows: [I8; 4] =
                        std::array::from_fn(|r| (fyi + I8::splat(r as i32 - 1)).max(I8::splat(0)).min(I8::splat(t.h - 1)) * I8::splat(w));
                    let cols: [I8; 4] = std::array::from_fn(|k| (fxi + I8::splat(k as i32 - 1)).max(I8::splat(0)).min(I8::splat(w - 1)));
                    (rows, cols)
                });
                for ch in 0..3 {
                    let hp = &history[ch][..];
                    let load = |at: usize| F8::load_f16(hp[at..][..8].try_into().unwrap());
                    let hist = match (two, edge) {
                        (Some((a, b, in_a)), _) => filter_cross(|r, k| F8::select(in_a, load(a + r * wu + k), load(b + r * wu + k)), &wx, &wy),
                        (_, Some((rows, cols))) => filter_cross(|r, k| F8::gather_f16(hp, rows[r] + cols[k]), &wx, &wy),
                        _ => {
                            let base = at - wu - 1;
                            filter_cross(|r, k| load(base + r * wu + k), &wx, &wy)
                        }
                    };
                    let hist = hist.max(zero).min(s(1e4));
                    let hist = F8::select(same, hist, hist.min(mx[ch]).max(mn[ch]));
                    let blended = F8::select(valid, (cur[ch] - hist).mul_add(alpha, hist), cur[ch]);
                    res[ch] = if t.full { blended } else { F8::select(same, cur[ch], blended) };
                    settled[ch] = F8::select(valid & same, hist, cur[ch]);
                }
            }
        }
        for (n, r) in next.iter_mut().zip(res) {
            n[m] = r.to_f16();
        }
        if let Some(past) = past.as_mut() {
            for (p, v) in past.iter_mut().zip(settled) {
                p[m] = v.to_f16();
            }
        }
        let (cr, cg, cb) = (aces(res[0]), aces(res[1]), aces(res[2]));
        let lum = cr * s(0.2126) + cg * s(0.7152) + cb * s(0.0722);
        let v = s(1.0) - s(0.44) * (F8::load(&vig[m]) + vy2);
        for (o, c) in out.iter_mut().zip([cr, cg, cb]) {
            let c = ((c - lum).mul_add(s(t.saturation), lum) * v).max(zero).min(s(0.999_999_94));
            o[m] = (t.srgb.encode(c) + bayer).shr4().to_bytes();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn raster_band(band: i32, lists: &[Lists], workers: std::ops::RangeInclusive<usize>, zb: &mut [f32], ib: &mut [u32],
               opaque: Option<&[f32]>, translucent: bool, assets: &Assets, anim: &[i32], w: i32, h: i32, cols: (i32, i32)) {
    let (bx0, by0) = (cols.0, band * BAND);
    let (bx1, by1) = (cols.1 - 1, h.min(by0 + BAND) - 1);
    if cols == (0, w) {
        zb.fill(0.0);
        ib.fill(NO_PRIM);
    } else {
        // Overlay mode draws a few columns: only those are cleared.
        let wu = w as usize;
        for r in 0..(by1 - by0 + 1) as usize {
            zb[r * wu + bx0 as usize..=r * wu + bx1 as usize].fill(0.0);
            ib[r * wu + bx0 as usize..=r * wu + bx1 as usize].fill(NO_PRIM);
        }
    }
    let inv_near = 1.0 / NEAR;
    let wu = w as usize;
    let s = F8::splat;
    let (zero, one, lanes) = (s(0.0), s(1.0), I8::lanes());
    for wk in workers {
        let l = &lists[wk];
        let indices = if translucent { &l.water[band as usize] } else { &l.bands[band as usize] };
        for &local in indices {
            let p = &l.prims[local as usize];
            let pi = (wk as u32) << 26 | local;
            let q = p.q;
            let x0 = bx0.max(p.x0 as i32);
            let x1 = bx1.min(p.x1 as i32);
            let mut y0 = by0.max(p.y0 as i32);
            let mut y1 = by1.min(p.y1 as i32);
            if x0 > x1 || y0 > y1 {
                continue;
            }
            let e = p.eps;
            // Coverage constraints c(x, y) = cx * x + cy * y + c0 >= 0 over pixel centres.
            let mut cons = [[0.0f32; 3]; 5];
            let mut nc;
            for k in 0..3 {
                cons[0][k] = p.s[k] + e * p.a[k];
                cons[1][k] = p.t[k] + e * p.a[k];
            }
            if q.flags & QUAD_TRI != 0 {
                for k in 0..3 {
                    cons[2][k] = (1.0 + e) * p.a[k] - p.s[k] - p.t[k];
                }
                nc = 3;
            } else {
                for k in 0..3 {
                    cons[2][k] = (1.0 + e) * p.a[k] - p.s[k];
                    cons[3][k] = (1.0 + e) * p.a[k] - p.t[k];
                }
                nc = 4;
            }
            for k in 0..3 {
                cons[nc][k] = -p.a[k];
            }
            cons[nc][2] += inv_near;
            nc += 1;
            // Constraints with no x term only clip rows; the rest bound each row's span on
            // one side, and their boundary moves linearly down the rows, so a row costs one
            // add per constraint instead of a divide.
            let mut reject = false;
            for c in &cons[..nc] {
                let (cx, cy, c0) = (c[0], c[1], c[2]);
                if cx.abs() > 1e-9 {
                    continue;
                }
                if cy.abs() > 1e-12 {
                    let yb = -c0 / cy - 0.5;
                    if cy > 0.0 {
                        y0 = y0.max(iceil(yb));
                    } else {
                        y1 = y1.min(ifloor(yb));
                    }
                } else if c0 < 0.0 {
                    reject = true;
                }
            }
            if reject || y0 > y1 {
                continue;
            }
            let (mut low_b, mut low_step, mut hi_b, mut hi_step) = ([0.0f32; 5], [0.0f32; 5], [0.0f32; 5], [0.0f32; 5]);
            let (mut n_low, mut n_hi) = (0, 0);
            for c in &cons[..nc] {
                let (cx, cy, c0) = (c[0], c[1], c[2]);
                if cx.abs() <= 1e-9 {
                    continue;
                }
                let b0 = -(cy * (y0 as f32 + 0.5) + c0) / cx - 0.5;
                let step = -cy / cx;
                if cx > 0.0 {
                    low_b[n_low] = b0;
                    low_step[n_low] = step;
                    n_low += 1;
                } else {
                    hi_b[n_hi] = b0;
                    hi_step[n_hi] = step;
                    n_hi += 1;
                }
            }
            // Cutout quads keep only the pixels whose texel is opaque, sampled as shading
            // will sample it: this prim's mip level of the current animation frame.
            let cutout = (q.layer == CUTOUT).then(|| {
                let t = assets.tex(q.tex);
                let level = iclamp(p.level as i32, 0, t.levels - 1);
                let texels = &t.frames[anim[q.tex as usize] as usize][t.level_offset[level as usize] as usize..];
                ((t.w >> level).max(1), (t.h >> level).max(1), texels)
            });
            let id8 = I8::splat(pi as i32);
            for y in y0..=y1 {
                let (mut lo, mut hi) = (x0, x1);
                for k in 0..n_low {
                    lo = lo.max(iceil(low_b[k]));
                    low_b[k] += low_step[k];
                }
                for k in 0..n_hi {
                    hi = hi.min(ifloor(hi_b[k]));
                    hi_b[k] += hi_step[k];
                }
                if lo > hi {
                    continue;
                }
                let py = y as f32 + 0.5;
                let row = (y - by0) as usize * wu;
                let zrow = &mut zb[row..row + wu];
                let irow = &mut ib[row..row + wu];
                let opaque_row = opaque.map(|o| &o[(y * w) as usize..(y * w) as usize + wu]);
                let (a_row, s_row, t_row) = (s(p.a[1] * py), s(p.s[1] * py), s(p.t[1] * py));
                let (first, last) = (I8::splat(lo - 1), I8::splat(hi + 1));
                // Eight pixels at a time, masked to the span; the group that would run past
                // the end of the row ends on it instead.
                let mut x = lo as usize;
                while x <= hi as usize {
                    let start = x.min(wu - 8);
                    let at = lanes + I8::splat(start as i32);
                    let xs = at.to_float() + s(0.5);
                    // The same operations in the same order as one pixel at a time.
                    let a = (s(p.a[0]) * xs + a_row) + s(p.a[2]);
                    let z = at8(zrow, start);
                    let mut keep = a.gt(z) & at.gt(first) & last.gt(at);
                    if let Some(o) = opaque_row {
                        keep = keep & a.gt(at8(o, start));
                    }
                    if let Some((size, size_h, texels)) = cutout
                        && keep.any()
                    {
                        let wv = one / a;
                        let st = ((s(p.s[0]) * xs + s_row) + s(p.s[2])) * wv;
                        let tt = ((s(p.t[0]) * xs + t_row) + s(p.t[2])) * wv;
                        let (st, tt) = (st.max(zero).min(one), tt.max(zero).min(one));
                        let u = (s(q.uv.x) + st * s(q.uv_s.x)) + tt * s(q.uv_t.x);
                        let v = (s(q.uv.y) + st * s(q.uv_s.y)) + tt * s(q.uv_t.y);
                        // The offset keeps the conversion a floor for the negative uvs of
                        // merged far quads; it is a multiple of every power-of-two size.
                        let tx = (u * s(size as f32) + s(65536.0)).to_int();
                        let ty = (v * s(size_h as f32) + s(65536.0)).to_int();
                        let (tx, ty) = if size & (size - 1) == 0 && size_h & (size_h - 1) == 0 {
                            (tx & I8::splat(size - 1), ty & I8::splat(size_h - 1))
                        } else {
                            (I8::load(&tx.to_array().map(|v| v.rem_euclid(size))), I8::load(&ty.to_array().map(|v| v.rem_euclid(size_h))))
                        };
                        let texel = I8::gather_u32(texels, ty * I8::splat(size) + tx);
                        keep = keep & texel.shr(I8::splat(24)).gt(I8::splat(25));
                    }
                    if keep.any() {
                        F8::select(keep, a, z).store((&mut zrow[start..start + 8]).try_into().unwrap());
                        let ids: &mut [u32; 8] = (&mut irow[start..start + 8]).try_into().unwrap();
                        I8::select(keep, id8, I8::load_u32(ids)).store_u32(ids);
                    }
                    x = start + 8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reach_row_matches_every_window() {
        let mut seed = 11u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        for wu in [8usize, 16, 64, 1920] {
            // Surfaces, sky (0 and -0.0) and edge-marked values, in runs and alone.
            let row: Vec<f32> = (0..wu)
                .map(|_| match next() % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => (next() % 1000) as f32 / 1000.0 + 0.001,
                })
                .collect();
            let (mut near, mut far) = (vec![0.0; wu / 8], vec![0.0; wu / 8]);
            reach_row(&row, &mut near, &mut far);
            let r = WARP_REACH as i64;
            for g in 0..wu as i64 / 8 {
                let window: Vec<f32> = (8 * g - r..8 * g + 8 + r).map(|i| row[i.clamp(0, wu as i64 - 1) as usize].max(0.0)).collect();
                let want_near = window.iter().fold(0.0f32, |a, &v| a.max(v));
                let want_far = window.iter().filter(|&&v| v > 0.0).fold(f32::MAX, |a, &v| a.min(v));
                assert_eq!(near[g as usize], want_near, "nearest at group {g} of {wu}");
                assert_eq!(far[g as usize], if want_far == f32::MAX { 0.0 } else { want_far }, "farthest at group {g} of {wu}");
            }
        }
    }
}
