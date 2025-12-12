#![feature(portable_simd)]
mod audio;
mod fft;
mod gui;

use fft::find_dft;
use gui::AnalyzerApp;

// Configuration constants.
const SAMPLE_RATE: u32 = 44100; // Standard audio sample rate.
const DFT_SIZE: usize = 2048; // FFT size balancing resolution and latency.

fn main() -> Result<(), eframe::Error> {
    //
    // Initialize logging with default filter set to "info".
    //
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting real-time audio spectrum analyzer...");

    //
    // Initialize FFT/DSP plan.
    //
    log::info!("Initializing FFT plan for N={}", DFT_SIZE);
    let fft_plan = find_dft(DFT_SIZE);

    //
    // Initialize audio capture subsystem.
    //
    log::info!("Initializing audio apture...");
    let (audio_stream, audio_consumer) = audio::start_capture(DFT_SIZE);

    //
    // Initialize GUI configuration.
    //
    log::info!("Initializing GUI...");
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 480.0])
            .with_title("fftanalyzer"),
        ..Default::default()
    };

    //
    // Launch GUI application.
    //
    eframe::run_native(
        "fftanalyzer",
        options,
        Box::new(move |cc| {
            gui::theme::setup_global_style(&cc.egui_ctx);

            //
            // Construct and return the analyzer application instance.
            //
            Ok(Box::new(AnalyzerApp::new(
                cc,
                audio_consumer,
                audio_stream,
                fft_plan,
                DFT_SIZE,
            )))
        }),
    )
}
