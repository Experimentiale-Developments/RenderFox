//! Eight-lane float and int vectors for the passes that touch every pixel of every frame.
//! On a CPU with AVX2 and FMA (any Intel since Haswell, or a VM whose CPU type passes
//! them through, such as Proxmox's "host") they are one register each; built for anything
//! older, the same code runs on two SSE2 halves, so a VM with a generic CPU type still
//! works after a rebuild. The build targets the CPU it runs on (`.cargo/config.toml`), and
//! `main` refuses to start on a CPU missing what the binary was built for.

#[cfg(target_feature = "avx2")]
mod imp {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct F8(pub __m256);

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct I8(pub __m256i);

    // AVX2 and FMA are compile-time target features here, so every intrinsic is
    // available; the unsafe blocks only acknowledge that.
    impl F8 {
        #[inline(always)]
        pub fn splat(v: f32) -> F8 {
            unsafe { F8(_mm256_set1_ps(v)) }
        }
        #[inline(always)]
        pub fn load(p: &[f32; 8]) -> F8 {
            unsafe { F8(_mm256_loadu_ps(p.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, p: &mut [f32; 8]) {
            unsafe { _mm256_storeu_ps(p.as_mut_ptr(), self.0) }
        }
        /// Eight half floats.
        #[inline(always)]
        pub fn load_f16(p: &[u16; 8]) -> F8 {
            unsafe { F8(_mm256_cvtph_ps(_mm_loadu_si128(p.as_ptr() as *const __m128i))) }
        }
        /// As half floats, rounded to nearest.
        #[inline(always)]
        pub fn to_f16(self) -> [u16; 8] {
            unsafe { std::mem::transmute(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(self.0)) }
        }
        /// `table[idx]` per lane from a table of half floats. Each lane reads the element
        /// after its own too, so `table` must reach one element past every index.
        #[inline(always)]
        pub fn gather_f16(table: &[u16], idx: I8) -> F8 {
            debug_assert!(idx.to_array().iter().all(|&i| i >= 0 && (i as usize) + 1 < table.len()));
            unsafe {
                let words = _mm256_i32gather_epi32::<2>(table.as_ptr() as *const i32, idx.0);
                let low = _mm256_and_si256(words, _mm256_set1_epi32(0xffff));
                let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(low, low));
                F8(_mm256_cvtph_ps(_mm256_castsi256_si128(packed)))
            }
        }
        /// `table[idx]` per lane; every index must be in range, which the callers
        /// establish by clamping and which is asserted in debug builds.
        #[inline(always)]
        pub fn gather(table: &[f32], idx: I8) -> F8 {
            debug_assert!(idx.to_array().iter().all(|&i| i >= 0 && (i as usize) < table.len()));
            unsafe { F8(_mm256_i32gather_ps::<4>(table.as_ptr(), idx.0)) }
        }
        /// self * a + b, fused.
        #[inline(always)]
        pub fn mul_add(self, a: F8, b: F8) -> F8 {
            unsafe { F8(_mm256_fmadd_ps(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: F8) -> F8 {
            unsafe { F8(_mm256_min_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: F8) -> F8 {
            unsafe { F8(_mm256_max_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sqrt(self) -> F8 {
            unsafe { F8(_mm256_sqrt_ps(self.0)) }
        }
        /// 1 / self to about 23 bits: the estimate and one Newton step, several times the
        /// throughput of a divide on Haswell.
        #[inline(always)]
        pub fn recip(self) -> F8 {
            unsafe {
                let x = _mm256_rcp_ps(self.0);
                F8(_mm256_mul_ps(x, _mm256_fnmadd_ps(self.0, x, _mm256_set1_ps(2.0))))
            }
        }
        /// 1 / sqrt(self) the same way.
        #[inline(always)]
        pub fn rsqrt(self) -> F8 {
            unsafe {
                let x = _mm256_rsqrt_ps(self.0);
                let hx = _mm256_mul_ps(_mm256_mul_ps(self.0, _mm256_set1_ps(0.5)), x);
                F8(_mm256_mul_ps(x, _mm256_fnmadd_ps(hx, x, _mm256_set1_ps(1.5))))
            }
        }
        #[inline(always)]
        pub fn floor(self) -> F8 {
            unsafe { F8(_mm256_floor_ps(self.0)) }
        }
        /// Truncates towards zero.
        #[inline(always)]
        pub fn to_int(self) -> I8 {
            unsafe { I8(_mm256_cvttps_epi32(self.0)) }
        }
        #[inline(always)]
        pub fn lt(self, o: F8) -> I8 {
            unsafe { I8(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, o.0))) }
        }
        #[inline(always)]
        pub fn gt(self, o: F8) -> I8 {
            unsafe { I8(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, o.0))) }
        }
        #[inline(always)]
        pub fn bits(self) -> I8 {
            unsafe { I8(_mm256_castps_si256(self.0)) }
        }
        #[inline(always)]
        pub fn le(self, o: F8) -> I8 {
            unsafe { I8(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(self.0, o.0))) }
        }
        #[inline(always)]
        pub fn ge(self, o: F8) -> I8 {
            unsafe { I8(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(self.0, o.0))) }
        }
        /// mask ? a : b, lanes of mask all-ones or all-zero.
        #[inline(always)]
        pub fn select(mask: I8, a: F8, b: F8) -> F8 {
            unsafe { F8(_mm256_blendv_ps(b.0, a.0, _mm256_castsi256_ps(mask.0))) }
        }
        /// Lane i of the result is lane idx[i] of self.
        #[inline(always)]
        pub fn permute(self, idx: I8) -> F8 {
            unsafe { F8(_mm256_permutevar8x32_ps(self.0, idx.0)) }
        }
        /// Sums of neighbouring lanes: a0 + a1, a2 + a3, ..., b6 + b7.
        #[inline(always)]
        pub fn pair_sums(a: F8, b: F8) -> F8 {
            unsafe {
                let h = _mm256_hadd_ps(a.0, b.0);
                F8(_mm256_permutevar8x32_ps(h, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7)))
            }
        }
    }

    impl I8 {
        #[inline(always)]
        pub fn splat(v: i32) -> I8 {
            unsafe { I8(_mm256_set1_epi32(v)) }
        }
        #[inline(always)]
        pub fn lanes() -> I8 {
            unsafe { I8(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)) }
        }
        #[inline(always)]
        pub fn load(p: &[i32; 8]) -> I8 {
            unsafe { I8(_mm256_loadu_si256(p.as_ptr() as *const __m256i)) }
        }
        /// The same bits from u32s.
        #[inline(always)]
        pub fn load_u32(p: &[u32; 8]) -> I8 {
            unsafe { I8(_mm256_loadu_si256(p.as_ptr() as *const __m256i)) }
        }
        #[inline(always)]
        pub fn load_u8(p: &[u8; 8]) -> I8 {
            unsafe { I8(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p.as_ptr() as *const __m128i))) }
        }
        #[inline(always)]
        pub fn store_u32(self, p: &mut [u32; 8]) {
            unsafe { _mm256_storeu_si256(p.as_mut_ptr() as *mut __m256i, self.0) }
        }
        #[inline(always)]
        pub fn gather(table: &[i32], idx: I8) -> I8 {
            debug_assert!(idx.to_array().iter().all(|&i| i >= 0 && (i as usize) < table.len()));
            unsafe { I8(_mm256_i32gather_epi32::<4>(table.as_ptr(), idx.0)) }
        }
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            unsafe { std::mem::transmute(self.0) }
        }
        #[inline(always)]
        pub fn to_float(self) -> F8 {
            unsafe { F8(_mm256_cvtepi32_ps(self.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: I8) -> I8 {
            unsafe { I8(_mm256_min_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: I8) -> I8 {
            unsafe { I8(_mm256_max_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn eq(self, o: I8) -> I8 {
            unsafe { I8(_mm256_cmpeq_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn all(self) -> bool {
            unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) == 0xff }
        }
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) != 0 }
        }
        /// Lane 0 in every lane.
        #[inline(always)]
        pub fn first(self) -> I8 {
            unsafe { I8(_mm256_permutevar8x32_epi32(self.0, _mm256_setzero_si256())) }
        }
        /// Lane i of the result is lane idx[i] of self.
        #[inline(always)]
        pub fn permute(self, idx: I8) -> I8 {
            unsafe { I8(_mm256_permutevar8x32_epi32(self.0, idx.0)) }
        }
        #[inline(always)]
        pub fn gt(self, o: I8) -> I8 {
            unsafe { I8(_mm256_cmpgt_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn select(mask: I8, a: I8, b: I8) -> I8 {
            unsafe { I8(_mm256_blendv_epi8(b.0, a.0, mask.0)) }
        }
        /// Logical shift right, each lane by its own count.
        #[inline(always)]
        pub fn shr(self, n: I8) -> I8 {
            unsafe { I8(_mm256_srlv_epi32(self.0, n.0)) }
        }
        #[inline(always)]
        pub fn as_f8(self) -> F8 {
            unsafe { F8(_mm256_castsi256_ps(self.0)) }
        }
        /// `table[idx]` per lane from a table of u16. Each lane reads the element after
        /// its own too, so `table` must reach one element past every index.
        #[inline(always)]
        pub fn gather_u16(table: &[u16], idx: I8) -> I8 {
            debug_assert!(idx.to_array().iter().all(|&i| i >= 0 && (i as usize) + 1 < table.len()));
            unsafe {
                let words = _mm256_i32gather_epi32::<2>(table.as_ptr() as *const i32, idx.0);
                I8(_mm256_and_si256(words, _mm256_set1_epi32(0xffff)))
            }
        }
        /// `table[idx]` per lane from a table of u32.
        #[inline(always)]
        pub fn gather_u32(table: &[u32], idx: I8) -> I8 {
            debug_assert!(idx.to_array().iter().all(|&i| i >= 0 && (i as usize) < table.len()));
            unsafe { I8(_mm256_i32gather_epi32::<4>(table.as_ptr() as *const i32, idx.0)) }
        }
        /// Arithmetic shift right by 4.
        #[inline(always)]
        pub fn shr4(self) -> I8 {
            unsafe { I8(_mm256_srai_epi32::<4>(self.0)) }
        }
        /// The lanes as bytes, saturating to 0..=255.
        #[inline(always)]
        pub fn to_bytes(self) -> [u8; 8] {
            unsafe {
                let w = _mm256_packus_epi32(self.0, self.0); // u16: 0 1 2 3 0 1 2 3 | 4 5 6 7 4 5 6 7
                let b = _mm256_packus_epi16(w, w); // u8: 0 1 2 3 ... | 4 5 6 7 ...
                let lo = _mm256_permutevar8x32_epi32(b, _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));
                (_mm_cvtsi128_si64(_mm256_castsi256_si128(lo)) as u64).to_le_bytes()
            }
        }
    }

    macro_rules! ops {
        ($t:ident, $($tr:ident $f:ident $i:ident),*) => {$(
            impl std::ops::$tr for $t {
                type Output = $t;
                #[inline(always)]
                fn $f(self, o: $t) -> $t {
                    unsafe { $t($i(self.0, o.0)) }
                }
            }
        )*};
    }
    ops!(F8, Add add _mm256_add_ps, Sub sub _mm256_sub_ps, Mul mul _mm256_mul_ps, Div div _mm256_div_ps);
    ops!(I8, Add add _mm256_add_epi32, Sub sub _mm256_sub_epi32, Mul mul _mm256_mullo_epi32, BitAnd bitand _mm256_and_si256,
         BitOr bitor _mm256_or_si256);
}

#[cfg(not(target_feature = "avx2"))]
mod imp {
    use crate::simd::*;

    #[derive(Clone, Copy)]
    pub struct F8(F4, F4);

    #[derive(Clone, Copy)]
    pub struct I8(I4, I4);

    fn halves<T: Copy>(p: &[T; 8]) -> (&[T; 4], &[T; 4]) {
        let (a, b) = p.split_at(4);
        (a.try_into().unwrap(), b.try_into().unwrap())
    }

    impl F8 {
        #[inline(always)]
        pub fn splat(v: f32) -> F8 {
            F8(splat(v), splat(v))
        }
        #[inline(always)]
        pub fn load(p: &[f32; 8]) -> F8 {
            let (a, b) = halves(p);
            F8(load4a(a), load4a(b))
        }
        #[inline(always)]
        pub fn store(self, p: &mut [f32; 8]) {
            let (a, b) = p.split_at_mut(4);
            store4a(a.try_into().unwrap(), self.0);
            store4a(b.try_into().unwrap(), self.1);
        }
        #[inline(always)]
        pub fn load_f16(p: &[u16; 8]) -> F8 {
            F8::from_array(p.map(super::half_to_f32))
        }
        #[inline(always)]
        pub fn to_f16(self) -> [u16; 8] {
            self.to_array().map(super::f32_to_half)
        }
        #[inline(always)]
        pub fn gather_f16(table: &[u16], idx: I8) -> F8 {
            let i = idx.to_array();
            F8::from_array(std::array::from_fn(|k| super::half_to_f32(table[i[k] as usize])))
        }
        #[inline(always)]
        pub fn gather(table: &[f32], idx: I8) -> F8 {
            let i = idx.to_array();
            F8::from_array(std::array::from_fn(|k| table[i[k] as usize]))
        }
        #[inline(always)]
        fn from_array(v: [f32; 8]) -> F8 {
            F8(F4::new(v[0], v[1], v[2], v[3]), F4::new(v[4], v[5], v[6], v[7]))
        }
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let (a, b) = (self.0.to_array(), self.1.to_array());
            std::array::from_fn(|k| if k < 4 { a[k] } else { b[k - 4] })
        }
        #[inline(always)]
        pub fn mul_add(self, a: F8, b: F8) -> F8 {
            self * a + b
        }
        #[inline(always)]
        pub fn min(self, o: F8) -> F8 {
            F8(vmin4(self.0, o.0), vmin4(self.1, o.1))
        }
        #[inline(always)]
        pub fn max(self, o: F8) -> F8 {
            F8(vmax4(self.0, o.0), vmax4(self.1, o.1))
        }
        #[inline(always)]
        pub fn sqrt(self) -> F8 {
            F8(sqrt4(self.0), sqrt4(self.1))
        }
        #[inline(always)]
        pub fn recip(self) -> F8 {
            F8::splat(1.0) / self
        }
        #[inline(always)]
        pub fn rsqrt(self) -> F8 {
            F8::splat(1.0) / self.sqrt()
        }
        #[inline(always)]
        pub fn floor(self) -> F8 {
            F8::from_array(self.to_array().map(f32::floor))
        }
        #[inline(always)]
        pub fn to_int(self) -> I8 {
            I8(to_int(self.0), to_int(self.1))
        }
        #[inline(always)]
        pub fn lt(self, o: F8) -> I8 {
            I8(self.0.lt(o.0), self.1.lt(o.1))
        }
        #[inline(always)]
        pub fn gt(self, o: F8) -> I8 {
            I8(self.0.gt(o.0), self.1.gt(o.1))
        }
        #[inline(always)]
        pub fn bits(self) -> I8 {
            I8(self.0.bits(), self.1.bits())
        }
        #[inline(always)]
        pub fn le(self, o: F8) -> I8 {
            I8(self.0.le(o.0), self.1.le(o.1))
        }
        #[inline(always)]
        pub fn ge(self, o: F8) -> I8 {
            I8(self.0.ge(o.0), self.1.ge(o.1))
        }
        #[inline(always)]
        pub fn select(mask: I8, a: F8, b: F8) -> F8 {
            F8(select4(mask.0, a.0, b.0), select4(mask.1, a.1, b.1))
        }
        /// Only an index's low three bits count, as in vpermd: callers permute with
        /// out-of-range indices in lanes they then select away.
        #[inline(always)]
        pub fn permute(self, idx: I8) -> F8 {
            let (v, i) = (self.to_array(), idx.to_array());
            F8::from_array(std::array::from_fn(|k| v[(i[k] & 7) as usize]))
        }
        #[inline(always)]
        pub fn pair_sums(a: F8, b: F8) -> F8 {
            let (a, b) = (a.to_array(), b.to_array());
            F8::from_array(std::array::from_fn(|k| if k < 4 { a[2 * k] + a[2 * k + 1] } else { b[2 * k - 8] + b[2 * k - 7] }))
        }
    }

    impl I8 {
        #[inline(always)]
        pub fn splat(v: i32) -> I8 {
            I8(splati(v), splati(v))
        }
        #[inline(always)]
        pub fn lanes() -> I8 {
            I8::load(&[0, 1, 2, 3, 4, 5, 6, 7])
        }
        #[inline(always)]
        pub fn load(p: &[i32; 8]) -> I8 {
            let (a, b) = halves(p);
            let q = |h: &[i32; 4]| load4u(&h.map(|v| v as u32));
            I8(q(a), q(b))
        }
        #[inline(always)]
        pub fn load_u32(p: &[u32; 8]) -> I8 {
            let (a, b) = halves(p);
            I8(load4u(a), load4u(b))
        }
        #[inline(always)]
        pub fn load_u8(p: &[u8; 8]) -> I8 {
            I8::load(&p.map(|v| v as i32))
        }
        #[inline(always)]
        pub fn store_u32(self, p: &mut [u32; 8]) {
            *p = self.to_array().map(|v| v as u32);
        }
        #[inline(always)]
        pub fn gather(table: &[i32], idx: I8) -> I8 {
            let i = idx.to_array();
            I8::load(&std::array::from_fn(|k| table[i[k] as usize]))
        }
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let (a, b) = (self.0.to_array(), self.1.to_array());
            std::array::from_fn(|k| if k < 4 { a[k] } else { b[k - 4] })
        }
        #[inline(always)]
        pub fn to_float(self) -> F8 {
            F8(to_float(self.0), to_float(self.1))
        }
        #[inline(always)]
        pub fn min(self, o: I8) -> I8 {
            I8(select4i(self.0.lt(o.0), self.0, o.0), select4i(self.1.lt(o.1), self.1, o.1))
        }
        #[inline(always)]
        pub fn max(self, o: I8) -> I8 {
            I8(select4i(self.0.gt(o.0), self.0, o.0), select4i(self.1.gt(o.1), self.1, o.1))
        }
        #[inline(always)]
        pub fn eq(self, o: I8) -> I8 {
            let e = |a: I4, b: I4| !(a.lt(b) | a.gt(b));
            I8(e(self.0, o.0), e(self.1, o.1))
        }
        #[inline(always)]
        pub fn all(self) -> bool {
            self.to_array().iter().all(|&v| v != 0)
        }
        #[inline(always)]
        pub fn any(self) -> bool {
            self.0.any() || self.1.any()
        }
        #[inline(always)]
        pub fn first(self) -> I8 {
            I8::splat(self.to_array()[0])
        }
        #[inline(always)]
        pub fn permute(self, idx: I8) -> I8 {
            let (v, i) = (self.to_array(), idx.to_array());
            I8::load(&std::array::from_fn(|k| v[(i[k] & 7) as usize]))
        }
        #[inline(always)]
        pub fn gt(self, o: I8) -> I8 {
            I8(self.0.gt(o.0), self.1.gt(o.1))
        }
        #[inline(always)]
        pub fn select(mask: I8, a: I8, b: I8) -> I8 {
            I8(select4i(mask.0, a.0, b.0), select4i(mask.1, a.1, b.1))
        }
        #[inline(always)]
        pub fn shr(self, n: I8) -> I8 {
            let (v, n) = (self.to_array(), n.to_array());
            I8::load(&std::array::from_fn(|k| ((v[k] as u32) >> n[k]) as i32))
        }
        #[inline(always)]
        pub fn as_f8(self) -> F8 {
            F8(self.0.as_f4(), self.1.as_f4())
        }
        #[inline(always)]
        pub fn gather_u16(table: &[u16], idx: I8) -> I8 {
            let i = idx.to_array();
            I8::load(&std::array::from_fn(|k| table[i[k] as usize] as i32))
        }
        #[inline(always)]
        pub fn gather_u32(table: &[u32], idx: I8) -> I8 {
            let i = idx.to_array();
            I8::load(&std::array::from_fn(|k| table[i[k] as usize] as i32))
        }
        #[inline(always)]
        pub fn shr4(self) -> I8 {
            I8::load(&self.to_array().map(|v| v >> 4))
        }
        #[inline(always)]
        pub fn to_bytes(self) -> [u8; 8] {
            self.to_array().map(|v| v.clamp(0, 255) as u8)
        }
    }

    macro_rules! ops {
        ($t:ident, $($tr:ident $f:ident),*) => {$(
            impl std::ops::$tr for $t {
                type Output = $t;
                #[inline(always)]
                fn $f(self, o: $t) -> $t {
                    $t(std::ops::$tr::$f(self.0, o.0), std::ops::$tr::$f(self.1, o.1))
                }
            }
        )*};
    }
    ops!(F8, Add add, Sub sub, Mul mul, Div div);
    ops!(I8, Add add, Sub sub, Mul mul, BitAnd bitand, BitOr bitor);
}

pub use imp::{F8, I8};


/// A half float's value, in software for the portable eight-lane forms; the AVX2 ones use
/// F16C.
#[cfg(not(target_feature = "avx2"))]
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: normalise the mantissa.
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((man << shift) & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// The nearest half float, ties to even; values past the range become infinity.
#[cfg(not(target_feature = "avx2"))]
fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = (rest > mid || (rest == mid && half & 1 != 0)) as u32;
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let round = (rest > 0x1000 || (rest == 0x1000 && half & 1 != 0)) as u32;
    sign | (half + round) as u16
}

/// What the binary was built to use and this CPU lacks, if anything.
pub fn missing_cpu_features() -> Vec<&'static str> {
    let mut missing = Vec::new();
    if cfg!(target_feature = "avx2") && !std::arch::is_x86_feature_detected!("avx2") {
        missing.push("avx2");
    }
    if cfg!(target_feature = "fma") && !std::arch::is_x86_feature_detected!("fma") {
        missing.push("fma");
    }
    missing
}
