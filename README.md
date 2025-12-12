# fftanaylzer

This project details the architecture of a high-performance Fast Fourier Transform (FFT) engine implemented in Rust, utilizing explicit SIMD vectorization for real-time signal processing. It demonstrates the efficacy of combining recursive decomposition algorithms (Cooley-Tukey, Rader, Bluestein) with hardware-accelerated kernels using Rust's experimental `portable_simd` module.

## DSP Architecture and Algorithms

The core of the developed spectrum analyzer is a polymorphic FFT engine designed to handle any input size $N$ efficiently. Unlike basic implementations that require $N$ to be a power of two, this engine employs a heuristic planner to select the optimal decomposition strategy based on the prime factorization of the input size.

### Recursive Decomposition strategies
The DSP logic follows a hierarchical structure implemented via a `DFTBase` trait:

1.  **Cooley-Tukey (Radix-P):** Selected when $N$ is a composite number with small prime factors. The algorithm recursively divides the DFT into $P$ smaller transforms. This implementation handles "Twiddle Factor" multiplication efficiently via pre-calculated trigonometric tables.
2.  **Rader's Algorithm:** Selected when $N$ is a prime number. Since primes cannot be factored for Cooley-Tukey, this algorithm transforms the DFT computation into a cyclic convolution, solved via an internal FFT of size $N-1$, enabling $O(N \log N)$ performance.
3.  **Bluestein's Algorithm (Chirp-Z):** Utilized as a fallback for "awkward" composite numbers lacking small prime factors. It modulates the signal with a chirp sequence to perform convolution via an FFT padded to a power of two.

### SIMD Acceleration
The critical performance optimization lies in the "leaf nodes" of the recursion. When the decomposition reaches small sizes (e.g., $N \in \{2, 3, 4, 5, 6, 8\}$), the engine dispatches execution to hardware-accelerated kernels.

Implemented using Rust's `portable_simd` (Nightly), these kernels utilize a **Structure-of-Arrays (SoA)** layout. This allows the CPU to perform complex arithmetic operations using fused multiply-add instructions across parallel frequency bins, significantly reducing the instruction cycle count compared to scalar execution.

## Visualization Interface

The visualization pipeline acts as the final stage of the DSP chain, rendered via a custom GUI using `egui` with a MacOS 9 "Platinum" theme.

1.  **Windowing:** A Hann window is applied to the time-domain PCM data to reduce spectral leakage.
2.  **Normalization:** Complex magnitudes are converted to Decibels (dB) and normalized to a $0.0-1.0$ range mapped to a $-100\text{dB}$ floor.
3.  **Rendering:** Data is rendered as both an instantaneous line plot and a scrolling spectrogram (waterfall) texture.

## Compilation Methodology

The implementation relies on experimental Rust features for SIMD. Therefore, the **Nightly** toolchain is required.

```bash
rustup override set nightly
```

To ensure the DSP loop meets real-time latency requirements, the artifact must be compiled with optimizations enabled:

```bash
cargo run --release
```

### Algorithmic Verification
The heuristic planner's behavior can be verified by modifying the `DFT_SIZE` constant in `src/main.rs`.

*   **Cooley-Tukey:** Set $N$ to a power of two (e.g., 2048).
*   **Rader:** Set $N$ to a prime number (e.g., 2053).
*   **Bluestein:** Set $N$ to a composite with large prime factors (e.g., 2000).

## Audio Input Configuration

The audio input device is pre-selected in the source code. The application is programmed to automatically attach to the operating system's default recording device. To specify a different hardware interface, the device selection logic within src/audio/mod.rs must be modified directly.

## References

**[1]** Joel Yliluoma. *Nopea Fourier-muunnos – teoria ja toteutus modernilla C++:lla*. Master's thesis, University of Helsinki, Department of Mathematics and Statistics, 2024.
