use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Vec3 { x: v, y: v, z: v }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f32) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

pub fn length(v: Vec3) -> f32 {
    dot(v, v).sqrt()
}

pub fn normalize(v: Vec3) -> Vec3 {
    let l = length(v);
    if l > 0.0 { v / l } else { v }
}

pub fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn clamp01(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp01((x - a) / (b - a));
    t * t * (3.0 - 2.0 * t)
}

// std::max(a, b) is `a < b ? b : a`; f32::max differs for NaN and signed zeros.
#[inline]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

// std::min(a, b) is `b < a ? b : a`.
#[inline]
pub fn fmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

pub fn vmax(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z))
}

pub fn vmin(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }
}

/// Column-major like GL: m[col][row]; transforms column vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Mat4 {
        let mut r = Mat4::default();
        for i in 0..4 {
            r.m[i][i] = 1.0;
        }
        r
    }

}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        let m = &self.m;
        Vec4::new(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w,
            m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w,
        )
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Mat4) -> Mat4 {
        let mut r = Mat4::default();
        for c in 0..4 {
            for rr in 0..4 {
                let mut s = 0.0f32;
                for k in 0..4 {
                    s += self.m[k][rr] * o.m[c][k];
                }
                r.m[c][rr] = s;
            }
        }
        r
    }
}

pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = normalize(target - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    let mut r = Mat4::identity();
    r.m[0][0] = s.x;
    r.m[1][0] = s.y;
    r.m[2][0] = s.z;
    r.m[0][1] = u.x;
    r.m[1][1] = u.y;
    r.m[2][1] = u.z;
    r.m[0][2] = -f.x;
    r.m[1][2] = -f.y;
    r.m[2][2] = -f.z;
    r.m[3][0] = -dot(s, eye);
    r.m[3][1] = -dot(u, eye);
    r.m[3][2] = dot(f, eye);
    r
}

pub fn perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    let mut r = Mat4::default();
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (z_far + z_near) / (z_near - z_far);
    r.m[2][3] = -1.0;
    r.m[3][2] = 2.0 * z_far * z_near / (z_near - z_far);
    r
}

/// sRGB <-> linear via a cheap but accurate enough curve; textures are sRGB.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

// Facing directions, in the vanilla order.
pub const DOWN: u8 = 0;
pub const UP: u8 = 1;
pub const NORTH: u8 = 2;
pub const SOUTH: u8 = 3;
pub const WEST: u8 = 4;
pub const EAST: u8 = 5;
pub const NONE: u8 = 255;
pub const DIR_DX: [i32; 6] = [0, 0, 0, 0, -1, 1];
pub const DIR_DY: [i32; 6] = [-1, 1, 0, 0, 0, 0];
pub const DIR_DZ: [i32; 6] = [0, 0, -1, 1, 0, 0];

pub fn dir_normal(d: usize) -> Vec3 {
    Vec3::new(DIR_DX[d] as f32, DIR_DY[d] as f32, DIR_DZ[d] as f32)
}
