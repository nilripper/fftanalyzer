use super::DFTBase;
use fftw::plan::*;
use fftw::types::*;
use num_complex::Complex32;
use std::sync::Arc;
use std::sync::Mutex;

/// FFTW3 wrapper providing a dedicated internal buffer and plan.
/// The internal state is guarded by a mutex because FFTW plan/buffer
/// combinations are not thread-safe under concurrent writes.
pub struct DFT_FFTW {
    n: usize,
    //
    // Internal plan and dedicated buffer.
    //
    state: Mutex<InternalState>,
}

struct InternalState {
    //
    // Plan stored in Arc to allow moving the wrapper safely.
    //
    plan: Arc<C2CPlan32>,

    //
    // Internal buffer used for both input and output.
    //
    data: Vec<Complex32>,
}

impl DFT_FFTW {
    pub fn new(n: usize) -> Self {
        let n_i32 = n as i32;

        //
        // Build a plan using temporary buffers.
        //
        let mut data = vec![Complex32::default(); n];
        let mut out = vec![Complex32::default(); n];

        //
        // Create a forward FFT plan with MEASURE.
        //
        let plan = C2CPlan::new(&[n], &mut data, &mut out, Sign::Forward, Flag::MEASURE)
            .expect("Failed to create FFTW plan");

        //
        // Allocate a buffer twice the size of the transform length.
        //
        let mut double_buffer = vec![Complex32::default(); n * 2];

        Self {
            n,
            state: Mutex::new(InternalState {
                plan: Arc::new(plan),
                data: double_buffer,
            }),
        }
    }
}

impl DFTBase for DFT_FFTW {
    fn name(&self) -> String {
        format!("FFTW({})", self.n)
    }

    fn size(&self) -> usize {
        self.n
    }

    fn is_inplace(&self) -> bool {
        //
        // Externally behaves as an in-place transform due to internal buffering.
        //
        true
    }

    fn xform(&self, input: &[Complex32], output: &mut [Complex32]) {
        self.xform_many(input, output, 1, 1, 1, 1, 1);
    }

    fn xform_inplace(&self, buffer: &mut [Complex32]) {
        //
        // Perform transform using the internal buffer.
        //
        self.xform(buffer, buffer);
    }

    //
    // Main transform function supporting batching and custom strides.
    //
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
        let mut state = self.state.lock().unwrap();
        let n = self.n;

        for k in 0..count {
            //
            // Copy input slice into the internal buffer.
            //
            for i in 0..n {
                state.data[i] = input[k * istep2 + i * istep];
            }

            //
            // Execute FFT into the second half of the buffer.
            //
            let (in_buf, out_buf) = state.data.split_at_mut(n);
            state
                .plan
                .reprocess(&mut in_buf[0..n], &mut out_buf[0..n])
                .expect("Exec failed");

            //
            // Copy FFT output to the destination slice.
            //
            for i in 0..n {
                output[k * ostep2 + i * ostep] = out_buf[i];
            }
        }
    }
}
