use super::{find_dft, prime_cache, DFTBase};
use num_complex::Complex32;
use std::f32::consts::PI;
use std::sync::Arc;

fn w(k: usize, n: usize) -> Complex32 {
    let angle = -2.0 * PI * (k as f32) / (n as f32);
    Complex32::from_polar(1.0, angle)
}

//
// Radix-P (Cooley–Tukey) implementation.
//
pub struct DFTRadix {
    n: usize,
    p: usize,
    q: usize,
    wtable: Vec<Complex32>,
    dft_p: Option<Arc<dyn DFTBase>>,
    dft_q: Option<Arc<dyn DFTBase>>,
}

impl DFTRadix {
    pub fn new(n: usize) -> Self {
        //
        // Select radix factor p and compute q = n / p.
        //
        let (factors, count) = prime_cache::get_factors_all(n);
        let p = if count > 0 { factors[0] } else { n };
        let q = n / p;

        let mut wtable = Vec::with_capacity(n);

        //
        // Precompute twiddle values for each index.
        //
        for a in 0..n {
            wtable.push(w((a % q) * (a / q), n));
        }

        //
        // Initialize sub-transforms for p and q sizes.
        //
        let dft_p = if p > 1 { Some(find_dft(p)) } else { None };
        let dft_q = if q > 1 { Some(find_dft(q)) } else { None };

        Self {
            n,
            p,
            q,
            wtable,
            dft_p,
            dft_q,
        }
    }
}

impl DFTBase for DFTRadix {
    fn name(&self) -> String {
        format!("RadixP<{}>({})", self.p, self.n)
    }
    fn size(&self) -> usize {
        self.n
    }
    fn is_inplace(&self) -> bool {
        false
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
        //
        // Compute p transforms, each of length q.
        //
        if let Some(dq) = &self.dft_q {
            for i in 0..count {
                let in_base = i * istep2;
                let out_base = i * ostep2;

                dq.xform_many(
                    &input[in_base..],
                    &mut output[out_base..],
                    self.p * istep,
                    istep,
                    ostep,
                    ostep * self.q,
                    self.p,
                );
            }
        }

        //
        // Apply twiddle factors to intermediate output.
        //
        for i in 0..count {
            let out_base = i * ostep2;
            for b in 1..self.p {
                for a in 1..self.q {
                    let idx = ostep * (b * self.q + a);
                    output[out_base + idx] *= self.wtable[b * self.q + a];
                }
            }
        }

        //
        // Compute q transforms, each of length p.
        //
        if let Some(dp) = &self.dft_p {
            for i in 0..count {
                let out_base = i * ostep2;

                let mut temp_col = vec![Complex32::default(); self.n];

                //
                // Copy block into column-major buffer.
                //
                for k in 0..self.n {
                    temp_col[k] = output[out_base + k * ostep];
                }

                //
                // Execute q transforms of length p.
                //
                let temp_in = temp_col.clone();
                dp.xform_many(&temp_in, &mut temp_col, self.q, 1, self.q, 1, self.q);

                //
                // Store transformed block back to output.
                //
                for k in 0..self.n {
                    output[out_base + k * ostep] = temp_col[k];
                }
            }
        }
    }
}

//
// Rader’s algorithm for prime-length DFT.
//
pub struct DFTRader {
    n: usize,
    g: usize,
    g_inv: usize,
    omega: Vec<Complex32>,
    dft_n1: Arc<dyn DFTBase>,
}

impl DFTRader {
    pub fn new(n: usize) -> Self {
        //
        // Find generator g for multiplicative group mod n.
        //
        let (factors, count) = prime_cache::get_factors_all(n - 1);
        let mut g = 2;

        loop {
            let mut is_gen = true;
            for i in 0..count {
                if powermod(g, (n - 1) / factors[i], n) == 1 {
                    is_gen = false;
                    break;
                }
            }
            if is_gen {
                break;
            }
            g += 1;
        }

        let g_inv = powermod(g, n - 2, n);

        //
        // Build reordered twiddle sequence.
        //
        let mut omega = vec![Complex32::default(); n - 1];
        let mut gp = 1;
        for i in 0..n - 1 {
            omega[i] = w(gp, n);
            gp = (gp * g_inv) % n;
        }

        //
        // Compute transformed kernel vector.
        //
        let dft_n1 = find_dft(n - 1);
        dft_n1.xform_inplace(&mut omega);

        //
        // Normalize kernel coefficients.
        //
        for x in &mut omega {
            *x /= (n - 1) as f32;
        }

        Self {
            n,
            g,
            g_inv,
            omega,
            dft_n1,
        }
    }
}

fn powermod(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    result
}

impl DFTBase for DFTRader {
    fn name(&self) -> String {
        format!("Rader({})", self.n)
    }
    fn size(&self) -> usize {
        self.n
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
        let n_minus_1 = self.n - 1;

        //
        // Allocate working buffer for all transforms.
        //
        let mut buf = vec![Complex32::default(); count + n_minus_1 * count * 2];

        //
        // Apply Rader permutation and extract DC terms.
        //
        for i in 0..count {
            buf[i] = input[i * istep2];

            let mut gp = 1;
            for k in 0..n_minus_1 {
                let in_idx = i * istep2 + gp * istep;
                let buf_idx = count + k + i * n_minus_1;
                buf[buf_idx] = input[in_idx];
                gp = (gp * self.g) % self.n;
            }
        }

        //
        // Forward DFT of permuted blocks.
        //
        let split_idx = count + n_minus_1 * count;
        let (buf_lower, buf_upper) = buf.split_at_mut(split_idx);

        self.dft_n1.xform_many(
            &buf_lower[count..],
            buf_upper,
            1,
            n_minus_1,
            1,
            n_minus_1,
            count,
        );

        //
        // Multiply by kernel and compute DC correction.
        //
        for i in 0..count {
            let dc_term = buf[i] + buf[count + n_minus_1 * (i + count)];
            output[i * ostep2] = dc_term;

            for k in 0..n_minus_1 {
                let idx_src = count + k + n_minus_1 * (i + count);
                let val = buf[idx_src];
                let w = self.omega[k];
                let correction = if k == 0 { buf[i] } else { Complex32::default() };

                buf[idx_src] = (val * w + correction).conj();
            }
        }

        //
        // Inverse DFT of modified sequence.
        //
        let (buf_lower, buf_upper) = buf.split_at_mut(split_idx);

        self.dft_n1.xform_many(
            buf_upper,
            &mut buf_lower[count..],
            1,
            n_minus_1,
            1,
            n_minus_1,
            count,
        );

        //
        // Apply inverse permutation and write result.
        //
        for i in 0..count {
            let mut gp = 1;
            for k in 0..n_minus_1 {
                let buf_idx = count + k + i * n_minus_1;
                let out_idx = i * ostep2 + gp * ostep;
                output[out_idx] = buf[buf_idx].conj();
                gp = (gp * self.g_inv) % self.n;
            }
        }
    }
}

//
// Bluestein’s algorithm for arbitrary sizes.
//
pub struct DFTBluestein {
    n: usize,
    nb: usize,
    w0: Vec<Complex32>,
    w1: Vec<Complex32>,
    dft_nb: Arc<dyn DFTBase>,
}

impl DFTBluestein {
    pub fn new(n: usize, nb: usize) -> Self {
        //
        // Generate chirp sequence w0.
        //
        let mut w0 = Vec::with_capacity(n);
        for k in 0..n {
            w0.push(w(k * k, 2 * n));
        }

        //
        // Build convolution kernel w1 padded to nb.
        //
        let mut w1 = vec![Complex32::default(); nb];
        for k in 0..n {
            w1[k] = w0[k] / (nb as f32);
        }
        for k in 1..n {
            w1[nb - k] = w1[k];
        }

        //
        // Transform kernel in frequency domain.
        //
        let dft_nb = find_dft(nb);
        dft_nb.xform_inplace(&mut w1);

        Self {
            n,
            nb,
            w0,
            w1,
            dft_nb,
        }
    }
}

impl DFTBase for DFTBluestein {
    fn name(&self) -> String {
        format!("Bluestein({})", self.n)
    }
    fn size(&self) -> usize {
        self.n
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
        //
        // Allocate contiguous buffer for all transforms.
        //
        let mut buf = vec![Complex32::default(); self.nb * count * 2];
        let (slice1, slice2) = buf.split_at_mut(self.nb * count);

        //
        // Apply initial modulation using chirp sequence.
        //
        for i in 0..count {
            for k in 0..self.n {
                slice1[k + i * self.nb] = input[k * istep + i * istep2] * self.w0[k];
            }
        }

        //
        // Forward DFT of modulated blocks.
        //
        self.dft_nb
            .xform_many(slice1, slice2, 1, self.nb, 1, self.nb, count);

        //
        // Multiply by precomputed kernel in frequency domain.
        //
        for i in 0..count {
            for j in 0..self.nb {
                let idx = j + i * self.nb;
                slice2[idx] = slice2[idx].conj() * self.w1[j];
            }
        }

        //
        // Inverse DFT of product.
        //
        self.dft_nb
            .xform_many(slice2, slice1, 1, self.nb, 1, self.nb, count);

        //
        // Final modulation and output write-back.
        //
        for i in 0..count {
            for k in 0..self.n {
                let val = slice1[k + i * self.nb].conj();
                output[k * ostep + i * ostep2] = val * self.w0[k];
            }
        }
    }
}
