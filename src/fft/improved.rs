use super::DFTBase;
use num_complex::Complex32;
use std::f32::consts::PI;
use std::ops::{Add, Mul, Sub};
use std::simd::prelude::*;
use std::simd::LaneCount;
use std::simd::SupportedLaneCount;

/// SIMD batch of L complex numbers stored in SoA layout.
/// Exposes real and imaginary SIMD vectors for kernel operations.
#[derive(Clone, Copy)]
pub struct BatchComplex<const L: usize>
where
    LaneCount<L>: SupportedLaneCount,
{
    pub re: Simd<f32, L>,
    pub im: Simd<f32, L>,
}

impl<const L: usize> BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    /// Returns a batch with all lanes set to the same complex value.
    #[inline(always)]
    fn splat(c: Complex32) -> Self {
        Self {
            re: Simd::splat(c.re),
            im: Simd::splat(c.im),
        }
    }

    /// Returns a batch initialized to zero.
    #[inline(always)]
    fn zero() -> Self {
        Self {
            re: Simd::splat(0.0),
            im: Simd::splat(0.0),
        }
    }
}

impl<const L: usize> Add for BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = Self;

    /// SIMD complex addition.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<const L: usize> Sub for BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = Self;

    /// SIMD complex subtraction.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<const L: usize> Mul<f32> for BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = Self;

    /// Multiplies each lane by a scalar.
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        let s = Simd::splat(rhs);
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl<const L: usize> Mul<Complex32> for BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = Self;

    /// SIMD complex multiplication with a scalar complex value.
    #[inline(always)]
    fn mul(self, rhs: Complex32) -> Self {
        let cre = Simd::splat(rhs.re);
        let cim = Simd::splat(rhs.im);
        Self {
            re: self.re * cre - self.im * cim,
            im: self.re * cim + self.im * cre,
        }
    }
}

impl<const L: usize> Mul<BatchComplex<L>> for BatchComplex<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = Self;

    /// SIMD complex multiplication with another batch.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Computes sin(pi * a / b).
#[inline(always)]
fn sab(a: f32, b: f32) -> f32 {
    (PI * a / b).sin()
}

/// Computes cos(pi * a / b).
#[inline(always)]
fn cab(a: f32, b: f32) -> f32 {
    (PI * a / b).cos()
}

/// Returns the imaginary-unit complex constant.
#[inline(always)]
fn i_c() -> Complex32 {
    Complex32::new(0.0, 1.0)
}

/// DFT kernel interface for SIMD-capable transform implementations.
pub trait DftKernel {
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount;
}

/// Kernel for size-1 DFT.
pub struct Kernel1;
impl DftKernel for Kernel1 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        X[0] = x[0];
    }
}

/// Kernel for size-2 DFT.
pub struct Kernel2;
impl DftKernel for Kernel2 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        X[0] = x[0] + x[1];
        X[1] = x[0] - x[1];
    }
}

/// Kernel for size-3 DFT.
pub struct Kernel3;
impl DftKernel for Kernel3 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let t0 = (x[1] - x[2]) * sab(1.0, 3.0) * i_c();
        let u0 = x[1] + x[2];
        let u1 = x[0] - u0 * 0.5;
        X[0] = x[0] + u0;
        X[1] = u1 - t0;
        X[2] = u1 + t0;
    }
}

/// Kernel for size-4 DFT.
pub struct Kernel4;
impl DftKernel for Kernel4 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let t0 = x[0] + x[2];
        let t1 = x[3] + x[1];
        let u0 = x[0] - x[2];
        let u1 = (x[3] - x[1]) * i_c();
        X[0] = t0 + t1;
        X[1] = u0 + u1;
        X[2] = t0 - t1;
        X[3] = u0 - u1;
    }
}

/// Kernel for size-5 DFT.
pub struct Kernel5;
impl DftKernel for Kernel5 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let a = 0.25;
        let b = sab(2.0, 5.0);
        let c = sab(1.0, 5.0);
        let d = cab(1.0, 5.0) - a;

        let t0 = x[1] + x[4];
        let t1 = x[2] + x[3];
        let t2 = (t0 - t1) * d;
        let u0 = x[1] - x[4];
        let u1 = x[2] - x[3];
        let u2 = t0 + t1;
        let u3 = x[0] - u2 * a;
        let t4 = u3 + t2;
        let t5 = (u0 * b + u1 * c) * i_c();

        X[0] = x[0] + u2;

        let u4 = u3 - t2;
        let u5 = (u1 * b - u0 * c) * i_c();

        X[1] = t4 - t5;
        X[2] = u4 + u5;
        X[4] = t4 + t5;
        X[3] = u4 - u5;
    }
}

/// Kernel for size-6 DFT.
pub struct Kernel6;
impl DftKernel for Kernel6 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let a = 0.5;
        let b = sab(1.0, 3.0);

        let t0 = x[0] + x[3];
        let t1 = x[4] + x[1];
        let t2 = x[2] + x[5];
        let t3 = t0 - (t1 + t2) * a;
        let t4 = (t1 - t2) * i_c();

        let u0 = x[0] - x[3];
        let u1 = x[4] - x[1];
        let u2 = x[2] - x[5];
        let u3 = u0 - (u1 + u2) * a * i_c();
        let u4 = (u1 - u2) * i_c();

        X[0] = t0 + t1 + t2;
        X[1] = u3 + u4 * b;
        X[4] = t3 + t4 * b;
        X[3] = u0 + u1 + u2;
        X[5] = u3 - u4 * b;
        X[2] = t3 - t4 * b;
    }
}

/// Kernel for size-8 DFT.
pub struct Kernel8;
impl DftKernel for Kernel8 {
    #[inline(always)]
    fn transform<const L: usize>(x: &mut [BatchComplex<L>], X: &mut [BatchComplex<L>])
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let a = sab(1.0, 4.0);
        let t0 = x[7] - x[3];
        let t1 = x[1] - x[5];
        let t2 = x[0] + x[4];
        let t3 = x[2] + x[6];
        let t4 = (t0 + t1) * a;
        let u0 = x[7] + x[3];
        let u1 = x[1] + x[5];
        let u2 = x[0] - x[4];
        let u3 = x[2] - x[6];
        let u4 = (t0 - t1) * a;
        let t5 = t2 + t3;
        let t6 = u2 + t4;
        let t7 = u0 + u1;
        let t8 = (u4 - u3) * i_c();
        let u5 = t2 - t3;
        let u6 = u2 - t4;
        let u7 = (u0 - u1) * i_c();
        let u8 = (u4 + u3) * i_c();

        X[0] = t5 + t7;
        X[1] = t6 + t8;
        X[2] = u5 + u7;
        X[3] = u6 + u8;
        X[4] = t5 - t7;
        X[7] = t6 - t8;
        X[6] = u5 - u7;
        X[5] = u6 - u8;
    }
}

/// DFT implementation using SIMD gather/transform/scatter.
/// Parameterized by kernel type and transform size.
pub struct DFTImproved<K: DftKernel, const N: usize> {
    _marker: std::marker::PhantomData<K>,
}

impl<K: DftKernel + Send + Sync + 'static, const N: usize> DFTImproved<K, N> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    /// Gathers strided input into SIMD batches, applies the kernel, and scatters results back.
    #[inline(always)]
    fn dosimd3<const L: usize>(
        &self,
        input: &[Complex32],
        istep: usize,
        istep2: usize,
        output: &mut [Complex32],
        ostep: usize,
        ostep2: usize,
    ) where
        LaneCount<L>: SupportedLaneCount,
    {
        let mut x = [BatchComplex::<L>::zero(); N];
        let mut X = [BatchComplex::<L>::zero(); N];

        for a in 0..N {
            let mut re_arr = [0.0f32; L];
            let mut im_arr = [0.0f32; L];

            for b in 0..L {
                let idx = a * istep + b * istep2;
                let c = input[idx];
                re_arr[b] = c.re;
                im_arr[b] = c.im;
            }

            x[a].re = Simd::from_array(re_arr);
            x[a].im = Simd::from_array(im_arr);
        }

        K::transform(&mut x, &mut X);

        for a in 0..N {
            let re_arr = X[a].re.to_array();
            let im_arr = X[a].im.to_array();

            for b in 0..L {
                let idx = a * ostep + b * ostep2;
                output[idx] = Complex32::new(re_arr[b], im_arr[b]);
            }
        }
    }

    /// Attempts SIMD processing in widths 8, then 4, then scalar width 1.
    fn dosimd2(
        &self,
        mut input: &[Complex32],
        istep: usize,
        mut istep2: usize,
        mut output: &mut [Complex32],
        ostep: usize,
        mut ostep2: usize,
        num: usize,
    ) {
        if num == 1 {
            istep2 = 0;
            ostep2 = 0;
        }

        let mut n = 0;

        while n + 8 <= num {
            self.dosimd3::<8>(input, istep, istep2, output, ostep, ostep2);
            input = &input[8 * istep2..];
            output = &mut output[8 * ostep2..];
            n += 8;
        }

        while n + 4 <= num {
            self.dosimd3::<4>(input, istep, istep2, output, ostep, ostep2);
            input = &input[4 * istep2..];
            output = &mut output[4 * ostep2..];
            n += 4;
        }

        while n < num {
            self.dosimd3::<1>(input, istep, istep2, output, ostep, ostep2);
            input = &input[1 * istep2..];
            output = &mut output[1 * ostep2..];
            n += 1;
        }
    }
}

impl<K: DftKernel + Send + Sync + 'static, const N: usize> DFTBase for DFTImproved<K, N> {
    fn name(&self) -> String {
        format!("Improved_{}", N)
    }

    fn size(&self) -> usize {
        N
    }

    fn is_inplace(&self) -> bool {
        true
    }

    fn xform_many(
        &self,
        input: &[Complex32],
        output: &mut [Complex32],
        istep: usize,
        istep2: usize,
        ostep: usize,
        ostep2: usize,
        count: usize,
    ) {
        self.dosimd2(input, istep, istep2, output, ostep, ostep2, count);
    }
}
