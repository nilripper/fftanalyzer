pub mod theme;

use crate::fft::DFTBase;
use eframe::egui;
use num_complex::Complex32;
use ringbuf::Consumer;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct AnalyzerApp {
    //
    // Audio input and processing components.
    //
    audio_consumer: Consumer<f32, Arc<ringbuf::HeapRb<f32>>>,
    _audio_stream: cpal::Stream,
    fft_plan: Arc<dyn DFTBase>,

    //
    // DSP buffers for time-domain and frequency-domain processing.
    //
    dft_size: usize,
    time_domain_buf: VecDeque<f32>,
    freq_domain_buf: Vec<f32>,

    //
    // Waterfall visualization buffers and texture handle.
    //
    waterfall_buf: Vec<u8>,
    waterfall_height: usize,
    texture: Option<egui::TextureHandle>,

    //
    // Statistics and diagnostic information.
    //
    last_stats_time: Instant,
    samples_processed: usize,
    max_input_peak: f32,
    max_fft_peak: f32,

    //
    // Silence detection state.
    //
    no_signal_timer: Instant,
    is_silence: bool,
}

impl AnalyzerApp {
    pub fn new(
        _cc: &eframe::CreationContext,
        audio_consumer: Consumer<f32, Arc<ringbuf::HeapRb<f32>>>,
        audio_stream: cpal::Stream,
        fft_plan: Arc<dyn DFTBase>,
        dft_size: usize,
    ) -> Self {
        let waterfall_height = 256;

        Self {
            audio_consumer,
            _audio_stream: audio_stream,
            fft_plan,
            dft_size,

            //
            // Initialize DSP buffers.
            //
            time_domain_buf: VecDeque::from(vec![0.0; dft_size]),
            freq_domain_buf: vec![0.0; dft_size / 2],

            //
            // Allocate waterfall buffer (RGBA).
            //
            waterfall_buf: vec![0; (dft_size / 2) * waterfall_height * 4],
            waterfall_height,
            texture: None,

            //
            // Initialize statistics and silence state.
            //
            last_stats_time: Instant::now(),
            samples_processed: 0,
            max_input_peak: 0.0,
            max_fft_peak: 0.0,
            no_signal_timer: Instant::now(),
            is_silence: true,
        }
    }

    fn update_dsp(&mut self) {
        let mut max_in_batch = 0.0;

        //
        // Ingest audio samples from ring buffer.
        //
        while let Some(sample) = self.audio_consumer.pop() {
            self.time_domain_buf.pop_front();
            self.time_domain_buf.push_back(sample);
            self.samples_processed += 1;

            let abs_sample = sample.abs();
            if abs_sample > self.max_input_peak {
                self.max_input_peak = abs_sample;
            }
            if abs_sample > max_in_batch {
                max_in_batch = abs_sample;
            }
        }

        //
        // Silence detection (−80 dB threshold, 2-second timeout).
        //
        if max_in_batch > 0.0001 {
            self.no_signal_timer = Instant::now();
            self.is_silence = false;
        } else if self.no_signal_timer.elapsed() > Duration::from_secs(2) {
            self.is_silence = true;
        }

        //
        // Apply window function and prepare complex FFT input.
        //
        let mut complex_in: Vec<Complex32> = self
            .time_domain_buf
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (self.dft_size - 1) as f32)
                            .cos());
                Complex32::new(x * window, 0.0)
            })
            .collect();

        //
        // Execute FFT.
        //
        self.fft_plan.xform_inplace(&mut complex_in);

        //
        // Convert magnitudes to normalized dB values.
        //
        let width = self.dft_size / 2;
        let min_db = -100.0;
        let max_db = 0.0;

        for i in 0..width {
            let mag = complex_in[i].norm();
            if mag > self.max_fft_peak {
                self.max_fft_peak = mag;
            }

            let db = 20.0 * mag.max(1e-9).log10();
            let range = max_db - min_db;
            let norm = ((db - min_db) / range).clamp(0.0, 1.0);

            self.freq_domain_buf[i] = norm;
        }

        //
        // Periodic DSP statistics logging.
        //
        if self.last_stats_time.elapsed() > Duration::from_secs(1) {
            log::info!(
                "DSP | Processed: {} | Max Peak: {:.5} | Silence: {}",
                self.samples_processed,
                self.max_input_peak,
                self.is_silence
            );
            self.samples_processed = 0;
            self.max_input_peak = 0.0;
            self.max_fft_peak = 0.0;
            self.last_stats_time = Instant::now();
        }

        //
        // Update waterfall: scroll up one row and write new spectrum colors.
        //
        let row_size = width * 4;
        let buf_len = self.waterfall_buf.len();
        self.waterfall_buf
            .copy_within(0..buf_len - row_size, row_size);

        for i in 0..width {
            let val = self.freq_domain_buf[i];
            let (r, g, b) = theme::get_heatmap_color(val);
            self.waterfall_buf[i * 4] = r;
            self.waterfall_buf[i * 4 + 1] = g;
            self.waterfall_buf[i * 4 + 2] = b;
            self.waterfall_buf[i * 4 + 3] = 255;
        }
    }
}

impl eframe::App for AnalyzerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        //
        // Run DSP update and request GUI repaint.
        //
        self.update_dsp();
        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            //
            // Draw top menu bar.
            //
            theme::draw_menu_bar(ui, &self.fft_plan.name());
            ui.add_space(4.0);

            //
            // Frequency-domain visualization window.
            //
            theme::draw_platinum_window(ui, "Frequency Domain", |ui| {
                ui.heading("Spectrogram");

                //
                // Upload waterfall buffer to texture each frame.
                //
                let width = self.dft_size / 2;
                let height = self.waterfall_height;
                let image =
                    egui::ColorImage::from_rgba_unmultiplied([width, height], &self.waterfall_buf);

                if let Some(texture) = &mut self.texture {
                    texture.set(image, egui::TextureOptions::NEAREST);
                } else {
                    self.texture = Some(ui.ctx().load_texture(
                        "waterfall",
                        image,
                        egui::TextureOptions::NEAREST,
                    ));
                }

                //
                // Draw waterfall texture and overlay silence warning.
                //
                if let Some(tex) = &self.texture {
                    let r = ui.image((tex.id(), egui::vec2(ui.available_width(), 200.0)));

                    if self.is_silence {
                        ui.painter().text(
                            r.rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "NO SIGNAL\nCheck Privacy Settings\nAllow Desktop Apps Access",
                            egui::FontId::proportional(20.0),
                            egui::Color32::RED,
                        );
                    }
                }

                ui.separator();
                ui.heading("Instantaneous");

                //
                // Draw instantaneous spectrum plot.
                //
                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    let (_rect, response) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 100.0),
                        egui::Sense::hover(),
                    );

                    ui.painter().rect_stroke(
                        response.rect,
                        egui::Rounding::ZERO,
                        egui::Stroke::new(1.0, egui::Color32::GRAY),
                    );

                    let points: Vec<egui::Pos2> = self
                        .freq_domain_buf
                        .iter()
                        .enumerate()
                        .map(|(i, &val)| {
                            let x = response.rect.min.x
                                + (i as f32 / width as f32) * response.rect.width();
                            let y = response.rect.max.y - (val * response.rect.height());
                            egui::Pos2::new(x, y)
                        })
                        .collect();

                    ui.painter().add(egui::Shape::line(
                        points,
                        egui::Stroke::new(1.0, egui::Color32::DARK_BLUE),
                    ));
                });
            });
        });
    }
}
