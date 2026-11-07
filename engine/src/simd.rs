//! Four-wide float and int vectors on SSE2, for CPUs with SSE2 and nothing newer,
//! which these compile to directly; anything past SSE2 (blendv, roundps, gathers,
//! pmulld) is spelled out by hand here.

// SSE2 is part of the x86-64 baseline, so every intrinsic below is always available;
// the unsafe blocks only acknowledge that the compiler cannot see this.
use std::arch::x86_64::*;
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Shl, Sub};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F4(pub __m128);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I4(pub __m128i);

impl F4 {
    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> F4 {
        unsafe { F4(_mm_setr_ps(a, b, c, d)) }
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// The same bits as integers, like a C cast between vector types.
    #[inline(always)]
    pub fn bits(self) -> I4 {
        unsafe { I4(_mm_castps_si128(self.0)) }
    }
    #[inline(always)]
    pub fn lt(self, o: F4) -> I4 {
        unsafe { I4(_mm_castps_si128(_mm_cmplt_ps(self.0, o.0))) }
    }
    #[inline(always)]
    pub fn le(self, o: F4) -> I4 {
        unsafe { I4(_mm_castps_si128(_mm_cmple_ps(self.0, o.0))) }
    }
    #[inline(always)]
    pub fn gt(self, o: F4) -> I4 {
        unsafe { I4(_mm_castps_si128(_mm_cmpgt_ps(self.0, o.0))) }
    }
    #[inline(always)]
    pub fn ge(self, o: F4) -> I4 {
        unsafe { I4(_mm_castps_si128(_mm_cmpge_ps(self.0, o.0))) }
    }
}

impl I4 {
    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
    #[inline(always)]
    pub fn as_f4(self) -> F4 {
        unsafe { F4(_mm_castsi128_ps(self.0)) }
    }
    #[inline(always)]
    pub fn lt(self, o: I4) -> I4 {
        unsafe { I4(_mm_cmplt_epi32(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn gt(self, o: I4) -> I4 {
        unsafe { I4(_mm_cmpgt_epi32(self.0, o.0)) }
    }
    /// Any lane set, as `_mm_movemask_ps` on the mask.
    #[inline(always)]
    pub fn any(self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) != 0 }
    }
}

#[inline(always)]
pub fn splat(v: f32) -> F4 {
    unsafe { F4(_mm_set1_ps(v)) }
}

#[inline(always)]
pub fn splati(v: i32) -> I4 {
    unsafe { I4(_mm_set1_epi32(v)) }
}

/// From a four-element array, the form `as_chunks` hands out: no length to check.
#[inline(always)]
pub fn load4a(p: &[f32; 4]) -> F4 {
    F4(unsafe { _mm_loadu_ps(p.as_ptr()) })
}

#[inline(always)]
pub fn store4a(p: &mut [f32; 4], v: F4) {
    unsafe { _mm_storeu_ps(p.as_mut_ptr(), v.0) }
}

#[inline(always)]
pub fn load4u(p: &[u32]) -> I4 {
    assert!(p.len() >= 4);
    I4(unsafe { _mm_loadu_si128(p.as_ptr() as *const __m128i) })
}

#[inline(always)]
pub fn vmin4(a: F4, b: F4) -> F4 {
    unsafe { F4(_mm_min_ps(a.0, b.0)) }
}

#[inline(always)]
pub fn vmax4(a: F4, b: F4) -> F4 {
    unsafe { F4(_mm_max_ps(a.0, b.0)) }
}

#[inline(always)]
pub fn sqrt4(v: F4) -> F4 {
    unsafe { F4(_mm_sqrt_ps(v.0)) }
}

/// mask ? a : b, with mask lanes all-ones or all-zero.
#[inline(always)]
pub fn select4(mask: I4, a: F4, b: F4) -> F4 {
    ((mask & a.bits()) | (!mask & b.bits())).as_f4()
}

#[inline(always)]
pub fn select4i(mask: I4, a: I4, b: I4) -> I4 {
    (mask & a) | (!mask & b)
}

/// Truncates towards zero.
#[inline(always)]
pub fn to_int(v: F4) -> I4 {
    unsafe { I4(_mm_cvttps_epi32(v.0)) }
}

#[inline(always)]
pub fn to_float(v: I4) -> F4 {
    unsafe { F4(_mm_cvtepi32_ps(v.0)) }
}

impl Add for F4 {
    type Output = F4;
    #[inline(always)]
    fn add(self, o: F4) -> F4 {
        unsafe { F4(_mm_add_ps(self.0, o.0)) }
    }
}

impl Sub for F4 {
    type Output = F4;
    #[inline(always)]
    fn sub(self, o: F4) -> F4 {
        unsafe { F4(_mm_sub_ps(self.0, o.0)) }
    }
}

impl Mul for F4 {
    type Output = F4;
    #[inline(always)]
    fn mul(self, o: F4) -> F4 {
        unsafe { F4(_mm_mul_ps(self.0, o.0)) }
    }
}

impl Div for F4 {
    type Output = F4;
    #[inline(always)]
    fn div(self, o: F4) -> F4 {
        unsafe { F4(_mm_div_ps(self.0, o.0)) }
    }
}

impl Add for I4 {
    type Output = I4;
    #[inline(always)]
    fn add(self, o: I4) -> I4 {
        unsafe { I4(_mm_add_epi32(self.0, o.0)) }
    }
}

impl Sub for I4 {
    type Output = I4;
    #[inline(always)]
    fn sub(self, o: I4) -> I4 {
        unsafe { I4(_mm_sub_epi32(self.0, o.0)) }
    }
}

impl Mul for I4 {
    type Output = I4;
    /// Low 32 bits of each product: SSE2 has no pmulld, so two pmuludq and a shuffle.
    #[inline(always)]
    fn mul(self, o: I4) -> I4 {
        unsafe {
            let even = _mm_mul_epu32(self.0, o.0);
            let odd = _mm_mul_epu32(_mm_srli_si128::<4>(self.0), _mm_srli_si128::<4>(o.0));
            I4(_mm_unpacklo_epi32(_mm_shuffle_epi32::<0b1000>(even), _mm_shuffle_epi32::<0b1000>(odd)))
        }
    }
}

impl BitAnd for I4 {
    type Output = I4;
    #[inline(always)]
    fn bitand(self, o: I4) -> I4 {
        unsafe { I4(_mm_and_si128(self.0, o.0)) }
    }
}

impl BitOr for I4 {
    type Output = I4;
    #[inline(always)]
    fn bitor(self, o: I4) -> I4 {
        unsafe { I4(_mm_or_si128(self.0, o.0)) }
    }
}

impl Not for I4 {
    type Output = I4;
    #[inline(always)]
    fn not(self) -> I4 {
        unsafe { I4(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }
}

impl Shl<i32> for I4 {
    type Output = I4;
    #[inline(always)]
    fn shl(self, n: i32) -> I4 {
        unsafe { I4(_mm_sll_epi32(self.0, _mm_cvtsi32_si128(n))) }
    }
}
