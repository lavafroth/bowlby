use binread::{BinRead, BinReaderExt};
use ort::{Environment, LoggingLevel};
use std::{
    error::Error,
    fs,
    io::{Read, Seek},
};

// finished porting the following:
// params.{c,h}
// model_file.{c,h}

// currently working on ort_util.c, april_model.c
// WONTIMPLEMENT: transfer_strings_and_free_model

const MAX_NETWORKS: u64 = 8;

#[repr(C)]
#[derive(BinRead)]
pub struct ModelFileHeader {
    // APRILMDL
    aprilmdl: [u8; 8],
    version: u32,
    header_size: u64,
}

#[repr(C)]
#[derive(BinRead)]
pub struct LanguageHeader {
    language: [u8; 8],
}

#[derive(Debug, Clone, Copy, BinRead, PartialEq, Eq)]
#[br(repr = u32)]
pub enum ModelType {
    ModelUnknown = 0,
    ModelLstmTransducerStateless = 1,
    ModelMax = 2,
}

pub fn read_string<T>(cursor: &mut T) -> Result<String, Box<dyn Error>>
where
    T: Read + Seek,
{
    let size: u64 = cursor.read_ne()?;
    let mut buf = vec![0; size as usize];
    cursor.read_exact(buf.as_mut_slice())?;
    Ok(String::from_utf8(buf)?)
}

#[derive(Debug, Clone, Copy, BinRead)]
pub struct ParamMetadata {
    offset: u64,
    size: u64,
    num_networks: u64,
}

impl ParamMetadata {
    pub fn parse<T>(self, cursor: &mut T) -> Result<Params, Box<dyn Error>>
    where
        T: Read + Seek,
    {
        assert!(self.num_networks <= MAX_NETWORKS);

        let networks: Vec<Network> = (0..self.num_networks)
            .map(|_| -> Result<Network, Box<dyn Error>> { Ok(cursor.read_ne()?) })
            .collect::<Result<Vec<Network>, Box<dyn Error>>>()?;

        cursor.seek(std::io::SeekFrom::Start(self.offset))?;
        let partial_params: PartialParams = cursor.read_ne()?;

        assert!(partial_params.batch_size == 1);
        assert!((partial_params.segment_size > 0) && (partial_params.segment_size < 100));
        assert!(
            (partial_params.segment_step > 0)
                && (partial_params.segment_step < 100)
                && (partial_params.segment_step <= partial_params.segment_size)
        );
        assert!((partial_params.mel_features > 0) && (partial_params.mel_features < 256));
        assert!((partial_params.sample_rate > 0) && (partial_params.sample_rate < 144000));
        assert!((partial_params.token_count > 0) && (partial_params.token_count < 16384));
        assert!(
            (partial_params.blank_id >= 0)
                && (partial_params.blank_id < partial_params.token_count)
        );

        assert!(
            (partial_params.frame_shift_ms > 0)
                && (partial_params.frame_shift_ms <= partial_params.frame_length_ms)
        );
        assert!((partial_params.frame_length_ms > 0) && (partial_params.frame_length_ms <= 5000));
        assert!(
            (partial_params.mel_low > 0) && (partial_params.mel_low < partial_params.sample_rate)
        );
        assert!(
            (partial_params.mel_high == 0) || (partial_params.mel_high > partial_params.mel_low)
        );

        let mut params = Params {
            batch_size: partial_params.batch_size,
            segment_size: partial_params.segment_size,
            segment_step: partial_params.segment_step,
            mel_features: partial_params.mel_features,
            sample_rate: partial_params.sample_rate,

            frame_shift_ms: partial_params.frame_shift_ms,
            frame_length_ms: partial_params.frame_length_ms,
            round_pow2: partial_params.round_pow2 != 0,
            mel_low: partial_params.mel_low,
            mel_high: partial_params.mel_high,
            snip_edges: partial_params.snip_edges != 0,

            blank_id: partial_params.blank_id,
            networks,
            tokens: vec![],
        };

        params.tokens = Vec::with_capacity(partial_params.token_count as usize);

        for _ in 0..partial_params.token_count {
            let token_len: i32 = cursor.read_ne()?;
            let mut buf = vec![0; token_len as usize];
            cursor.read_exact(buf.as_mut_slice())?;
            let token = std::str::from_utf8(&buf)?.trim_matches('\x00').to_string();
            params.tokens.push(token);
        }

        Ok(params)
    }
}

#[derive(Default, BinRead)]
pub struct PartialParams {
    magic: [u8; 8],
    batch_size: i32,
    segment_size: i32,
    segment_step: i32,
    mel_features: i32,
    sample_rate: i32,

    frame_shift_ms: i32,
    frame_length_ms: i32,
    round_pow2: i32,
    mel_low: i32,
    mel_high: i32,
    snip_edges: i32,

    token_count: i32,
    blank_id: i32,
}

#[derive(Debug)]
pub struct Params {
    batch_size: i32,
    segment_size: i32,
    segment_step: i32,
    mel_features: i32,
    sample_rate: i32,

    frame_shift_ms: i32,
    frame_length_ms: i32,
    round_pow2: bool,
    mel_low: i32,
    mel_high: i32,
    snip_edges: bool,

    blank_id: i32,

    networks: Vec<Network>,
    tokens: Vec<String>,
}

#[derive(Debug, Clone, Copy, BinRead)]
pub struct Network {
    offset: u64,
    size: u64,
}
impl Network {
    pub fn read<T>(self, cursor: &mut T) -> Result<Vec<u8>, Box<dyn Error>>
    where
        T: Read + Seek,
    {
        cursor.seek(std::io::SeekFrom::Start(self.offset))?;
        let mut buf = vec![0; self.size as usize];
        cursor.read_exact(&mut buf)?;
        Ok(buf)
    }
}

#[derive(Debug)]
pub struct ModelFile {
    version: u32,
    language: String,
    name: String,
    description: String,
    model_type: ModelType,
    params: Params,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cursor = fs::File::open("model.april")?;
    let file_header: ModelFileHeader = cursor.read_ne()?;
    let lang_header: LanguageHeader = cursor.read_ne()?;
    let language = std::str::from_utf8(&lang_header.language)?
        .trim_matches('\x00')
        .to_string();
    let name = read_string(&mut cursor)?;
    let description = read_string(&mut cursor)?;

    let model_type: ModelType = cursor.read_ne()?;
    let params: ParamMetadata = cursor.read_ne()?;
    let params = params.parse(&mut cursor)?;

    let model_file = ModelFile {
        version: file_header.version,
        language,
        name,
        description,
        model_type,
        params,
    };

    assert_eq!(
        model_file.model_type,
        ModelType::ModelLstmTransducerStateless
    );

    let environment = Environment::builder()
        .with_name("aam")
        .with_log_level(LoggingLevel::Verbose)
        .build()?
        .into_arc();

    // Here's some repeated boilerplate because we can't clone a session builder
    let encoder = model_file.params.networks[0].read(&mut cursor)?;
    let encoder = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&encoder)?;

    assert_eq!(encoder.inputs.len(), 3);
    assert_eq!(encoder.outputs.len(), 3);

    let decoder = model_file.params.networks[1].read(&mut cursor)?;
    let decoder = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&decoder)?;

    assert_eq!(decoder.inputs.len(), 1);
    assert_eq!(decoder.outputs.len(), 1);

    let joiner = model_file.params.networks[2].read(&mut cursor)?;
    let joiner = ort::SessionBuilder::new(&environment)?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_model_from_memory(&joiner)?;

    assert_eq!(joiner.inputs.len(), 2);
    assert_eq!(joiner.outputs.len(), 1);

    // assert_eq!(encoder.inputs[0].dimensions, 3);
    // assert_eq!(encoder.outputs.len(), 3);

    Ok(())
}
