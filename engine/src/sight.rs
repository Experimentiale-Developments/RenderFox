//! What part of the world a frame shows, so a recorded part of the orbit is thrown away
//! only when something changes that it could show, rather than whenever anything in the
//! mirrored area changes: a block placed behind the camera, a player walking about out of
//! frame or someone digging underground leaves every picture exactly as it was.
//!
//! A view is a full frame's camera and the farthest surface in each 32 x 32 tile of it.
//! Something is out of view if it projects outside the frame (with a margin), or lies
//! wholly behind the farthest surface of every tile it covers.

use crate::persist::{self, Reader, Writer};
use crate::vmath::*;
use std::sync::Mutex;

const TILE: usize = 32;
// Pixels around the frame that still count as in view. Views are taken every few frames
// and the camera turns about a degree a second, a few pixels between two of them.
pub const WORLD_MARGIN: f32 = 64.0;
// Players are tested without occlusion, since their name tags show through blocks, and
// with a margin that holds the widest tag around a point at the frame's edge.
pub const PLAYER_MARGIN: f32 = 240.0;
const NEAR: f32 = 0.08;

pub struct View {
    view_proj: Mat4,
    w: f32,
    h: f32,
    tiles_x: usize,
    tiles_y: usize,
    // The farthest opaque surface (1/w, 0 where sky shows) in each tile and the eight
    // around it: the camera moves a little between views, so a tile's neighbours may
    // come to show what the tile itself does not. As `narrow` keeps it.
    far: Vec<u16>,
}

/// An f32's top half (sign, exponent and 7 bits of mantissa): half the memory, and never
/// larger than the value, so a tile's farthest surface only ever comes nearer the camera
/// and a view sees a little more rather than less. Infinity and 0 stay what they are.
fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn widen(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

impl View {
    /// From the renderer's depth pyramid: farthest 1/w in each `block` x `block` square.
    pub fn new(view_proj: Mat4, w: i32, h: i32, pyramid: &[f32], pw: usize, ph: usize, block: usize) -> View {
        let per = TILE / block;
        let (tiles_x, tiles_y) = (pw.div_ceil(per), ph.div_ceil(per));
        let mut tiles = vec![f32::INFINITY; tiles_x * tiles_y];
        for y in 0..ph {
            for x in 0..pw {
                let t = &mut tiles[(y / per) * tiles_x + x / per];
                *t = t.min(pyramid[y * pw + x]);
            }
        }
        let mut far = vec![0u16; tiles.len()];
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut v = f32::INFINITY;
                for y in ty.saturating_sub(1)..(ty + 2).min(tiles_y) {
                    for x in tx.saturating_sub(1)..(tx + 2).min(tiles_x) {
                        v = v.min(tiles[y * tiles_x + x]);
                    }
                }
                far[ty * tiles_x + tx] = narrow(v);
            }
        }
        View { view_proj, w: w as f32, h: h as f32, tiles_x, tiles_y, far }
    }

    pub fn save(&self, w: &mut Writer) {
        for row in &self.view_proj.m {
            for &v in row {
                w.f32(v);
            }
        }
        w.f32(self.w);
        w.f32(self.h);
        w.len(self.tiles_x);
        w.len(self.tiles_y);
        for &f in &self.far {
            w.f32(widen(f));
        }
    }

    pub fn load(r: &mut Reader) -> persist::Result<View> {
        let mut view_proj = Mat4::default();
        for row in &mut view_proj.m {
            for v in row.iter_mut() {
                *v = r.f32()?;
            }
        }
        let (w, h) = (r.f32()?, r.f32()?);
        let (tiles_x, tiles_y) = (r.len(4)?, r.len(4)?);
        let n = tiles_x * tiles_y;
        if n > 1 << 20 {
            return Err("view too large".into());
        }
        let far = (0..n).map(|_| r.f32().map(narrow)).collect::<persist::Result<Vec<u16>>>()?;
        Ok(View { view_proj, w, h, tiles_x, tiles_y, far })
    }

    /// Whether any part of the convex hull of `pts` can show in this view.
    pub fn sees(&self, pts: &[Vec3], margin: f32, occlusion: bool) -> bool {
        let (mut x0, mut y0, mut x1, mut y1, mut nearest) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN, 0.0f32);
        let mut behind = 0;
        for &p in pts {
            let c = self.view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            if c.w < NEAR {
                behind += 1;
                continue;
            }
            let iw = 1.0 / c.w;
            let (x, y) = ((c.x * iw * 0.5 + 0.5) * self.w, (0.5 - c.y * iw * 0.5) * self.h);
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            nearest = nearest.max(iw);
        }
        if behind == pts.len() {
            return false;
        }
        // Reaching round the camera: projections are no longer bounds.
        if behind > 0 {
            return true;
        }
        if x1 < -margin || y1 < -margin || x0 > self.w + margin || y0 > self.h + margin {
            return false;
        }
        if !occlusion || x0 < 0.0 || y0 < 0.0 || x1 >= self.w || y1 >= self.h {
            return true;
        }
        let t = TILE as f32;
        let (tx0, tx1) = ((x0 / t) as usize, ((x1 / t) as usize).min(self.tiles_x - 1));
        let (ty0, ty1) = ((y0 / t) as usize, ((y1 / t) as usize).min(self.tiles_y - 1));
        // The renderer's own occlusion test allows the same 3 %.
        let nearest = nearest * 1.03;
        (ty0..=ty1).any(|ty| self.far[ty * self.tiles_x + tx0..=ty * self.tiles_x + tx1].iter().any(|&f| widen(f) <= nearest))
    }
}

/// A change in what the world looks like, published at `at` on the engine clock: the
/// region it can show in, as a convex hull, or everything.
#[derive(Clone)]
pub struct Change {
    pub at: f64,
    pub hull: Option<Vec<Vec3>>,
}

impl Change {
    pub fn seen_by(&self, view: &View) -> bool {
        self.hull.as_ref().is_none_or(|h| view.sees(h, WORLD_MARGIN, true))
    }
}

/// What the world worker changed, until the stream takes it.
#[derive(Default)]
pub struct Changes {
    list: Mutex<Vec<Change>>,
}

pub fn corners(lo: Vec3, hi: Vec3) -> [Vec3; 8] {
    std::array::from_fn(|k| {
        Vec3::new(if k & 1 != 0 { hi.x } else { lo.x }, if k & 2 != 0 { hi.y } else { lo.y }, if k & 4 != 0 { hi.z } else { lo.z })
    })
}

impl Changes {
    /// Boxes whose look changed: sections remeshed, new light in them.
    pub fn boxes(&self, at: f64, boxes: impl IntoIterator<Item = (Vec3, Vec3)>) {
        let mut l = self.list.lock().unwrap();
        l.extend(boxes.into_iter().map(|(lo, hi)| Change { at, hull: Some(corners(lo, hi).to_vec()) }));
    }

    /// Boxes whose shadows were redrawn. A shadow can fall anywhere along the sun's rays
    /// from the box, down to where they enter the ground (`floor`, the lowest surface under
    /// a point), so that whole stretch changes: in pieces, so the occlusion test can drop
    /// what lies underground.
    pub fn shadows(&self, at: f64, boxes: &[(Vec3, Vec3)], to_sun: Vec3, floor: impl Fn(Vec3, Vec3) -> f32) {
        // A texel either side is redrawn too; a block's worth covers the largest cascade's.
        let pad = Vec3::splat(1.0);
        let step = to_sun * (-16.0 / to_sun.y.max(0.05));
        let mut l = self.list.lock().unwrap();
        for &(lo, hi) in boxes {
            let (mut a, mut b) = (lo - pad, hi + pad);
            for _ in 0..8 {
                let (na, nb) = (a + step, b + step);
                let mut hull = corners(a, b).to_vec();
                hull.extend(corners(na, nb));
                l.push(Change { at, hull: Some(hull) });
                if nb.y < floor(na, nb) {
                    break;
                }
                (a, b) = (na, nb);
            }
        }
    }

    pub fn everything(&self, at: f64) {
        self.list.lock().unwrap().push(Change { at, hull: None });
    }

    pub fn take(&self) -> Vec<Change> {
        std::mem::take(&mut self.list.lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(far: f32) -> View {
        let (w, h) = (640, 360);
        let proj = perspective(70.0f32.to_radians(), w as f32 / h as f32, NEAR, 1200.0);
        let cam = look_at(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, 10.0, -1.0), Vec3::new(0.0, 1.0, 0.0));
        let (pw, ph) = (w as usize / 8, h as usize / 8);
        View::new(proj * cam, w, h, &vec![far; pw * ph], pw, ph, 8)
    }

    fn cube(c: Vec3, r: f32) -> Vec<Vec3> {
        corners(c - Vec3::splat(r), c + Vec3::splat(r)).to_vec()
    }

    #[test]
    fn in_front_is_seen_and_behind_the_camera_is_not() {
        let v = view(0.0);
        assert!(v.sees(&cube(Vec3::new(0.0, 10.0, -30.0), 1.0), WORLD_MARGIN, true));
        assert!(!v.sees(&cube(Vec3::new(0.0, 10.0, 30.0), 1.0), WORLD_MARGIN, true));
        // Far off to the side, outside the frame and its margin.
        assert!(!v.sees(&cube(Vec3::new(200.0, 10.0, -30.0), 1.0), WORLD_MARGIN, true));
        // Straddling the camera plane counts as seen.
        assert!(v.sees(&cube(Vec3::new(0.0, 10.0, 0.0), 2.0), WORLD_MARGIN, true));
    }

    #[test]
    fn behind_a_wall_is_hidden_only_with_occlusion() {
        // Every tile's farthest surface 10 blocks away: a cube at 30 is behind it.
        let v = view(1.0 / 10.0);
        let hidden = cube(Vec3::new(0.0, 10.0, -30.0), 1.0);
        assert!(!v.sees(&hidden, WORLD_MARGIN, true));
        assert!(v.sees(&hidden, PLAYER_MARGIN, false));
        assert!(v.sees(&cube(Vec3::new(0.0, 10.0, -6.0), 1.0), WORLD_MARGIN, true));
    }

    #[test]
    fn a_shadow_reaches_down_the_sun_to_the_ground() {
        let c = Changes::default();
        let to_sun = normalize(Vec3::new(0.3, 0.6, 0.2));
        c.shadows(1.0, &[(Vec3::new(0.0, 100.0, 0.0), Vec3::new(1.0, 101.0, 1.0))], to_sun, |_, _| 60.0);
        let pieces = c.take();
        assert_eq!(pieces.len(), 3, "100 down to 60 in steps of 16");
        let lowest = pieces.last().unwrap().hull.as_ref().unwrap().iter().map(|p| p.y).fold(f32::MAX, f32::min);
        assert!(lowest < 60.0);
        // Away from the sun: x and z fall as y does.
        assert!(pieces.last().unwrap().hull.as_ref().unwrap().iter().all(|p| p.x < 2.0 && p.z < 2.0));
    }
}
