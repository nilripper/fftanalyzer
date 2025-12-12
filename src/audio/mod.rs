use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use ringbuf::{Consumer, HeapRb};
use std::sync::Arc;

/// Starts audio capture on the default input device.
/// Supports f32, i16, and u16 formats and performs stereo-to-mono downmixing.
pub fn start_capture(buffer_size: usize) -> (cpal::Stream, Consumer<f32, Arc<HeapRb<f32>>>) {
    let host = cpal::default_host();

    //
    // Log all available input devices for debugging.
    //
    log::info!("--- AVAILABLE INPUT DEVICS ---");
    if let Ok(devices) = host.input_devices() {
        for (i, dev) in devices.enumerate() {
            let name = dev.name().unwrap_or("Unknown".into());
            log::info!("  [{}]: {}", i, name);
        }
    }
    log::info!("-------------------------------");

    //
    // Select the default audio input device.
    //
    let device = host
        .default_input_device()
        .expect("No audio input device found. Please check system settings.");

    log::info!(
        "Selected audio device: {}",
        device.name().unwrap_or("Unknown".into())
    );

    //
    // Create ring buffer (4× buffer size to reduce risk of underruns).
    //
    let (mut producer, consumer) = HeapRb::<f32>::new(buffer_size * 4).split();

    //
    // Retrieve and log the device's default input configuration.
    //
    let supported_config = device
        .default_input_config()
        .expect("Failed to get default input config");

    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let channels = config.channels as usize;

    log::info!(
        "Audio config: {:?} @ {}Hz, Channels: {}",
        sample_format,
        config.sample_rate.0,
        channels
    );

    let err_fn = |err| eprintln!("Audio input error: {}", err);

    //
    // Push mono samples into the buffer (downmix if necessary).
    //
    let mut push_mono = move |data: &[f32]| {
        if channels == 1 {
            let _ = producer.push_slice(data);
        } else if channels == 2 {
            //
            // Downmix stereo to mono using averaged samples.
            //
            for chunk in data.chunks_exact(2) {
                let mono = (chunk[0] + chunk[1]) * 0.5;
                let _ = producer.push(mono);
            }
        } else {
            //
            // Downmix multi-channel audio by selecting the first channel.
            //
            for chunk in data.chunks_exact(channels) {
                if let Some(&sample) = chunk.first() {
                    let _ = producer.push(sample);
                }
            }
        }
    };

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                push_mono(data);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &_| {
                //
                // Convert i16 samples to f32 before processing.
                //
                let f32_data: Vec<f32> = data.iter().map(|&s| (s as f32) / 32768.0).collect();
                push_mono(&f32_data);
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _: &_| {
                //
                // Convert u16 samples to signed f32 before processing.
                //
                let f32_data: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 - 32768.0) / 32768.0)
                    .collect();
                push_mono(&f32_data);
            },
            err_fn,
            None,
        ),
        _ => panic!("Unsupported audio sample format: {:?}", sample_format),
    }
    .expect("Failed to build audio stream");

    //
    // Start audio input stream.
    //
    stream.play().expect("Failed to start audio stream");

    (stream, consumer)
}
