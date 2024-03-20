use ort::{Environment, InMemorySession, LoggingLevel};
use std::{fs, sync::Arc};
mod model_file;

// finished porting the following:
// params.{c,h}
// model_file.{c,h}
// april_model.{c,h}
// ort_util.{c,h}

// WONTIMPLEMENT: transfer_strings_and_free_model, *_free
// WONTPORT:
// init.c
// log.h
// proc_thread.{c,h}

// TODO:
// april_session.c
// common.h
// fbank.c
// audio_provider.c

pub struct FBankOptions {
    sample_freq: i32,

    // Stride in milliseconds, for example 10
    frame_shift_ms: i32,

    // Window length in milliseconds, for example 25
    frame_length_ms: i32,

    // Number of mel bins, for example 80
    num_bins: i32,

    // Whether or not to round window size to pow2
    round_pow2: bool,

    // Mel low frequency, for example 20 Hz
    mel_low: i32,

    // Mel high frequency. If 0, sample_freq/2 is assumed
    mel_high: i32,

    // ???????????
    snip_edges: bool,

    // How many segments to pull in fbank_pull_segments.
    // For example, if this is equal to 9, then you should call
    // fbank_pull_segments with a tensor of size (1, 9, num_bins)
    pull_segment_count: i32,

    // How many segments to step over in fbank_pull_segments.
    // For example, if this is set to 4, then each call to fbank_pull_segments
    // will step over 4 segments
    pull_segment_step: i32,

    // If false, speed feature will be unavailable
    use_sonic: bool,
}

pub struct AprilASRModel<'a> {
    env: Arc<Environment>,
    encoder: InMemorySession<'a>,
    decoder: InMemorySession<'a>,
    joiner: InMemorySession<'a>,

    // same as eout_dim
    x_dim: Vec<usize>,
    h_dim: Vec<usize>,
    c_dim: Vec<usize>,
    dout_dim: Vec<usize>,
    eout_dim: Vec<usize>,
    context_dim: Vec<usize>,
    logits_dim: Vec<usize>,
    fbank_opts: FBankOptions,
    // ModelParameters params;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cursor = fs::File::open("model.april")?;
    let model_file = model_file::ModelFile::new(&mut cursor)?;
    assert_eq!(
        model_file.model_type,
        model_file::ModelType::ModelLstmTransducerStateless
    );

    let environment = Environment::builder()
        .with_name("aam")
        .with_log_level(LoggingLevel::Verbose)
        .build()?
        .into_arc();

    // Here's some repeated boilerplate because we can't clone a session builder
    let encoder = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&model_file.params.networks[0])?;

    assert_eq!(encoder.inputs.len(), 3);
    assert_eq!(encoder.outputs.len(), 3);

    let decoder = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&model_file.params.networks[1])?;

    assert_eq!(decoder.inputs.len(), 1);
    assert_eq!(decoder.outputs.len(), 1);

    let joiner = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&model_file.params.networks[2])?;

    assert_eq!(joiner.inputs.len(), 2);
    assert_eq!(joiner.outputs.len(), 1);

    let x_dim = encoder.inputs[0]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();
    let h_dim = encoder.inputs[1]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();
    let c_dim = encoder.inputs[2]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();
    let eout_dim = encoder.outputs[0]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();

    let context_dim = decoder.inputs[0]
        .dimensions()
        .take(2)
        .map(|c| c.unwrap())
        .collect();
    let dout_dim = decoder.outputs[0]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();

    let logits_dim = joiner.outputs[0]
        .dimensions()
        .take(3)
        .map(|c| c.unwrap())
        .collect();

    let model = AprilASRModel {
        env: environment,
        encoder,
        decoder,
        joiner,
        x_dim,
        h_dim,
        c_dim,
        dout_dim,
        eout_dim,
        context_dim,
        logits_dim,
        fbank_opts: FBankOptions {
            sample_freq: model_file.params.sample_rate,
            frame_shift_ms: model_file.params.frame_shift_ms,
            frame_length_ms: model_file.params.frame_length_ms,
            num_bins: model_file.params.mel_features,
            round_pow2: model_file.params.round_pow2,
            mel_low: model_file.params.mel_low,
            mel_high: model_file.params.mel_high,
            snip_edges: true,
            pull_segment_count: model_file.params.segment_size,
            pull_segment_step: model_file.params.segment_step,
            use_sonic: false,
        },
    };

    assert_eq!(model.x_dim[0], model_file.params.batch_size as usize);
    // assert_eq!(model.x_dim[0], model_file.params.batch_size as usize);
    Ok(())
}
