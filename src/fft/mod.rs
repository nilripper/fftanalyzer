#[cfg(feature = "use_fftw")]
pub mod fftw;
pub mod improved;
pub mod orig;
pub mod prime_cache;

use lazy_static::lazy_static;
use num_complex::Complex32;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Base interface for all DFT implementations.
pub trait DFTBase: Send + Sync {
    /// Single transform using contiguous input/output.
    fn xform(&self, input: &[Complex32], output: &mut [Complex32]) {
        self.xform_many(input, output, 1, 0, 1, 0, 1);
    }

    /// Transform `count` sequences.
    /// `istep`  = stride between elements in one transform.
    /// `istep2` = stride between distinct transforms.
    /// `ostep`  = output element stride.
    /// `ostep2` = output transform stride.
    fn xform_many(
        &self,
        input: &[Complex32],
        output: &mut [Complex32],
        istep: usize,
        istep2: usize,
        ostep: usize,
        ostep2: usize,
        count: usize,
    );

    /// Default in-place transform: temporary buffer copy.
    fn xform_inplace(&self, buffer: &mut [Complex32]) {
        let temp = buffer.to_vec();
        self.xform_many(&temp, buffer, 1, 0, 1, 0, 1);
    }

    fn name(&self) -> String;
    fn size(&self) -> usize;
    fn is_inplace(&self) -> bool;
}

lazy_static! {
    static ref PLAN_CACHE: Mutex<HashMap<usize, Arc<dyn DFTBase>>> = Mutex::new(HashMap::new());
}

/// Returns a DFT plan for size `n`, using caching and heuristic selection.
pub fn find_dft(n: usize) -> Arc<dyn DFTBase> {
    // Cached plan lookup.
    {
        let cache = PLAN_CACHE.lock();
        if let Some(plan) = cache.get(&n) {
            return plan.clone();
        }
    }

    // Strategy selection.
    let plan: Arc<dyn DFTBase> = if cfg!(feature = "use_fftw") {
        #[cfg(feature = "use_fftw")]
        {
            Arc::new(fftw::DFT_FFTW::new(n))
        }
        #[cfg(not(feature = "use_fftw"))]
        {
            unreachable!()
        }
    } else {
        match n {
            1 => Arc::new(improved::DFTImproved::<improved::Kernel1, 1>::new()),
            2 => Arc::new(improved::DFTImproved::<improved::Kernel2, 2>::new()),
            3 => Arc::new(improved::DFTImproved::<improved::Kernel3, 3>::new()),
            4 => Arc::new(improved::DFTImproved::<improved::Kernel4, 4>::new()),
            5 => Arc::new(improved::DFTImproved::<improved::Kernel5, 5>::new()),
            6 => Arc::new(improved::DFTImproved::<improved::Kernel6, 6>::new()),
            8 => Arc::new(improved::DFTImproved::<improved::Kernel8, 8>::new()),
            _ => {
                let (_factors, count) = prime_cache::get_factors_all(n);

                if count >= 2 {
                    Arc::new(orig::DFTRadix::new(n))
                } else {
                    let nb = (2 * n - 1).next_power_of_two();
                    if count == 0 {
                        Arc::new(orig::DFTRader::new(n))
                    } else {
                        Arc::new(orig::DFTBluestein::new(n, nb))
                    }
                }
            }
        }
    };

    // Cache the plan.
    let mut cache = PLAN_CACHE.lock();
    cache.insert(n, plan.clone());
    plan
}
