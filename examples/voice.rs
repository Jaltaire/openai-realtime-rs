use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FrameCount, StreamConfig};
use openai_realtime_types::audio::{
    Base64EncodedAudioBytes, ServerVadTurnDetection, TurnDetection,
};
use openai_realtime_utils as utils;
use openai_realtime_utils::audio::REALTIME_API_PCM16_SAMPLE_RATE;
use ringbuf::traits::{Consumer, Producer, Split};
use rubato::Resampler;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::Level;
use tracing_subscriber::fmt::time::ChronoLocal;

const INPUT_CHUNK_SIZE: usize = 1024;
const OUTPUT_CHUNK_SIZE: usize = 1024;
const OUTPUT_LATENCY_MS: usize = 1000;
const INTERRUPT_COOLDOWN_MS: u64 = 500;

pub enum Input {
    Audio(Vec<f32>),
    Initialize(),
    Initialized(),
    AISpeaking(),
    AISpeakingDone(),
    InterruptionDetected(),
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv_override().ok();

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_timer(ChronoLocal::rfc_3339())
        .init();

    let (input_tx, mut input_rx) = tokio::sync::mpsc::channel::<Input>(1024);

    // Setup audio input device
    let input = utils::device::get_or_default_input(None).expect("failed to get input device");
    println!("input: {:?}", &input.name().unwrap());

    let input_config = input
        .default_input_config()
        .expect("failed to get default input config");
    let input_config = StreamConfig {
        channels: input_config.channels(),
        sample_rate: input_config.sample_rate(),
        buffer_size: cpal::BufferSize::Fixed(FrameCount::from(INPUT_CHUNK_SIZE as u32)),
    };
    let input_channel_count = input_config.channels as usize;

    println!(
        "input: device={:?}, config={:?}",
        &input.name().unwrap(),
        &input_config
    );

    let audio_input = input_tx.clone();
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let audio = if input_channel_count > 1 {
            data.chunks(input_channel_count)
                .map(|c| c.iter().sum::<f32>() / input_channel_count as f32)
                .collect::<Vec<f32>>()
        } else {
            data.to_vec()
        };
        if let Err(e) = audio_input.try_send(Input::Audio(audio)) {
            eprintln!("Failed to send audio data to buffer: {:?}", e);
        }
    };

    let input_stream = input
        .build_input_stream(
            &input_config,
            input_data_fn,
            move |err| eprintln!("an error occurred on input stream: {}", err),
            None,
        )
        .expect("failed to build input stream");
    input_stream.play().expect("failed to play input stream");
    let input_sample_rate = input_config.sample_rate.0 as f32;

    // Setup audio output device
    let output = utils::device::get_or_default_output(None).expect("failed to get output device");
    let output_config = output
        .default_output_config()
        .expect("failed to get default output config");
    let output_config = StreamConfig {
        channels: output_config.channels(),
        sample_rate: output_config.sample_rate(),
        buffer_size: cpal::BufferSize::Fixed(FrameCount::from(OUTPUT_CHUNK_SIZE as u32)),
    };
    let output_channel_count = output_config.channels as usize;
    let output_sample_rate = output_config.sample_rate.0 as f32;

    let audio_out_buffer =
        utils::audio::shared_buffer(output_sample_rate as usize * OUTPUT_LATENCY_MS);
    let (mut audio_out_tx, mut audio_out_rx) = audio_out_buffer.split();

    let client_ctrl = input_tx.clone();
    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let mut sample_index = 0;
        let mut silence = 0;
        while sample_index < data.len() {
            let sample = audio_out_rx.try_pop().unwrap_or(0.0);

            if sample == 0.0 {
                silence += 1;
            }

            if sample_index < data.len() {
                data[sample_index] = sample;
                sample_index += 1;
            }
            if output_channel_count > 1 && sample_index < data.len() {
                data[sample_index] = sample;
                sample_index += 1;
            }
            sample_index += output_channel_count.saturating_sub(2);
        }

        let client_ctrl = client_ctrl.clone();
        if silence == (data.len() / output_channel_count) {
            if let Err(e) = client_ctrl.try_send(Input::AISpeakingDone()) {
                eprintln!("Failed to send speaking done event to client: {:?}", e);
            }
        } else {
            if let Err(e) = client_ctrl.try_send(Input::AISpeaking()) {
                eprintln!("Failed to send speaking event to client: {:?}", e);
            }
        }
    };

    let output_stream = output
        .build_output_stream(
            &output_config,
            output_data_fn,
            move |err| eprintln!("an error occurred on output stream: {}", err),
            None,
        )
        .expect("failed to build output stream");

    output_stream.play().expect("failed to play output stream");

    // OpenAI Realtime API setup
    let mut realtime_api = openai_realtime::connect_with_config(
        1024,
        openai_realtime::config::ConfigBuilder::new()
            .with_model("gpt-4o-mini-realtime-preview-2024-12-17")
            .build(),
    )
    .await
    .expect("failed to connect to OpenAI Realtime API");

    let mut out_resampler = utils::audio::create_resampler(
        REALTIME_API_PCM16_SAMPLE_RATE,
        output_sample_rate as f64,
        100,
    )
    .expect("failed to create resampler for output");

    let (post_tx, mut post_rx) = tokio::sync::mpsc::channel::<Base64EncodedAudioBytes>(100);

    let post_process = tokio::spawn(async move {
        while let Some(audio) = post_rx.recv().await {
            let audio_bytes = utils::audio::decode(&audio);
            let chunk_size = out_resampler.input_frames_next();
            for samples in utils::audio::split_for_chunks(&audio_bytes, chunk_size) {
                if let Ok(resamples) = out_resampler.process(&[samples.as_slice()], None) {
                    if let Some(resamples) = resamples.first() {
                        for resample in resamples {
                            if let Err(e) = audio_out_tx.try_push(*resample) {
                                eprintln!("Failed to push samples to buffer: {:?}", e);
                            }
                        }
                    }
                }
            }
        }
    });

    let client_ctrl2 = input_tx.clone();
    let mut server_events = realtime_api
        .server_events()
        .await
        .expect("failed to get server events");
    let server_handle = tokio::spawn(async move {
        while let Ok(e) = server_events.recv().await {
            match e {
                openai_realtime::types::events::ServerEvent::SessionCreated(data) => {
                    println!("session created: {:?}", data.session());
                    if let Err(e) = client_ctrl2.try_send(Input::Initialize()) {
                        eprintln!("Failed to send initialized event to client: {:?}", e);
                    }
                }
                openai_realtime::types::events::ServerEvent::SessionUpdated(data) => {
                    println!("session updated: {:?}", data.session());
                    if let Err(e) = client_ctrl2.try_send(Input::Initialized()) {
                        eprintln!("Failed to send initialized event to client: {:?}", e);
                    }
                }
                openai_realtime::types::events::ServerEvent::InputAudioBufferSpeechStarted(data) => {
                    println!("speech started: {:?}", data);
                    if let Err(e) = client_ctrl2.try_send(Input::InterruptionDetected()) {
                        eprintln!("Failed to send interruption event: {:?}", e);
                    }
                }
                openai_realtime::types::events::ServerEvent::ResponseAudioDelta(data) => {
                    if let Err(e) = post_tx.send(data.delta().to_string()).await {
                        eprintln!("Failed to send audio data to resampler: {:?}", e);
                    }
                }
                openai_realtime::types::events::ServerEvent::ConversationItemInputAudioTranscriptionCompleted(data) => {
                    println!("Human: {:?}", data.transcript().trim());
                }
                openai_realtime::types::events::ServerEvent::ResponseAudioTranscriptDone(data) => {
                    println!("AI: {:?}", data.transcript());
                }
                openai_realtime::types::events::ServerEvent::Close { reason } => {
                    println!("close: {:?}", reason);
                    break;
                }
                _ => {}
            }
        }
    });

    let mut in_resampler = utils::audio::create_resampler(
        input_sample_rate as f64,
        REALTIME_API_PCM16_SAMPLE_RATE,
        INPUT_CHUNK_SIZE,
    )
    .expect("failed to create resampler for input");

    let client_handle = tokio::spawn(async move {
        let mut ai_speaking = false;
        let mut initialized = false;
        let mut can_interrupt = true;
        let mut last_interrupt_time = Instant::now();
        let mut buffer: VecDeque<f32> = VecDeque::with_capacity(INPUT_CHUNK_SIZE * 2);

        while let Some(i) = input_rx.recv().await {
            match i {
                Input::Initialize() => {
                    println!("initializing...");
                    let session = openai_realtime::types::Session::new()
                        .with_modalities_enable_audio()
                        .with_voice(openai_realtime::types::audio::Voice::Alloy)
                        .with_input_audio_transcription_enable(
                            openai_realtime::types::audio::TranscriptionModel::Whisper,
                        )
                        .with_turn_detection_enable(TurnDetection::ServerVad(
                            ServerVadTurnDetection::default(),
                        ))
                        .build();
                    println!(
                        "session config: {:?}",
                        serde_json::to_string(&session).unwrap()
                    );
                    realtime_api
                        .update_session(session)
                        .await
                        .expect("failed to init session");
                }
                Input::Initialized() => {
                    println!("initialized");
                    initialized = true;
                }
                Input::AISpeaking() => {
                    if !ai_speaking {
                        println!("AI speaking...");
                    }
                    // buffer.clear();
                    ai_speaking = true;
                }
                Input::AISpeakingDone() => {
                    if ai_speaking {
                        println!("AI speaking done");
                    }
                    ai_speaking = false;
                }
                Input::InterruptionDetected() => {
                    println!("Interruption detected, AI speaking: {}", ai_speaking);
                    if ai_speaking
                        && can_interrupt
                        && last_interrupt_time.elapsed().as_millis() > INTERRUPT_COOLDOWN_MS as u128
                    {
                        println!("Attempting to stop AI response...");
                        realtime_api
                            .stop_response()
                            .await
                            .expect("failed to stop response");
                        last_interrupt_time = Instant::now();
                        println!("AI response stopped");
                    }
                }
                Input::Audio(audio) => {
                    if initialized {
                        for sample in audio {
                            buffer.push_back(sample);
                        }
                        let mut resampled: Vec<f32> = vec![];
                        while buffer.len() >= INPUT_CHUNK_SIZE {
                            let audio: Vec<f32> = buffer.drain(..INPUT_CHUNK_SIZE).collect();
                            if let Ok(resamples) = in_resampler.process(&[audio.as_slice()], None) {
                                if let Some(resamples) = resamples.first() {
                                    resampled.extend(resamples.iter().cloned());
                                }
                            }
                        }
                        if resampled.is_empty() {
                            continue;
                        }
                        let audio_bytes = utils::audio::encode(&resampled);
                        let audio_bytes = Base64EncodedAudioBytes::from(audio_bytes);
                        realtime_api
                            .append_input_audio_buffer(audio_bytes.clone())
                            .await
                            .expect("failed to send audio");
                    }
                }
            }
        }
    });

    tokio::select! {
        _ = post_process => {},
        _ = server_handle => {},
        _ = client_handle => {},
        _ = tokio::signal::ctrl_c() => {
            println!("Received Ctrl-C, shutting down...");
        }
    }
    println!("Shutting down...");
}
