// cpal backend device
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

//! Host audio backend built on [`cpal`], used where PipeWire is not available
//! (macOS, where cpal talks to CoreAudio).
//!
//! Playback only. The guest hands us PCM through the TX virtqueue; the virtio
//! stream machinery in [`super::super::worker`] turns each descriptor into a
//! [`Buffer`](super::super::stream::Buffer) queued on the stream. This backend
//! opens one cpal output stream per prepared virtio stream and, from the host
//! audio callback, pulls those buffers, converts the samples to the host
//! format and writes them into the callback's output slice. Dropping a
//! consumed `Buffer` is what completes the guest's request, so the audio clock
//! — not the worker thread — paces the guest.
//!
//! Capture (RX) is not implemented: `read` warns once and does nothing.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// `::cpal` — this module is itself named `cpal`.
use ::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ::cpal::{
    Device, FromSample, Sample, SampleFormat, SizedSample, StreamConfig, SupportedStreamConfig,
};

use super::super::stream::{Error as StreamError, PCMState};
use super::super::virtio_sound::{
    VirtioSndPcmSetParams, VIRTIO_SND_PCM_FMT_S16, VIRTIO_SND_PCM_FMT_S24, VIRTIO_SND_PCM_FMT_S32,
    VIRTIO_SND_PCM_FMT_U8, VIRTIO_SND_PCM_RATE_11025, VIRTIO_SND_PCM_RATE_16000,
    VIRTIO_SND_PCM_RATE_22050, VIRTIO_SND_PCM_RATE_32000, VIRTIO_SND_PCM_RATE_44100,
    VIRTIO_SND_PCM_RATE_48000, VIRTIO_SND_PCM_RATE_8000,
};
use super::super::{Direction, Error, Result, Stream};
use super::AudioBackend;

/// `MSB_SND_STATS=1` logs a line per second per stream with the number of
/// frames handed to the host audio device.
const STATS_ENV: &str = "MSB_SND_STATS";

//--------------------------------------------------------------------------------------------------
// Guest sample formats
//--------------------------------------------------------------------------------------------------

/// The PCM encodings this backend can decode out of guest memory.
///
/// These are the formats `snd::defs::SUPPORTED_FORMATS` advertises to the
/// guest; anything else is rejected in `set_parameters` with
/// `VIRTIO_SND_S_NOT_SUPP` rather than reaching the audio callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuestFormat {
    /// `VIRTIO_SND_PCM_FMT_U8`: unsigned 8-bit, 128 is silence.
    U8,
    /// `VIRTIO_SND_PCM_FMT_S16`: signed 16-bit little endian.
    S16,
    /// `VIRTIO_SND_PCM_FMT_S24`: signed 24-bit little endian in a 4-byte
    /// container (ALSA's `S24_LE`), value in the low 24 bits.
    S24,
    /// `VIRTIO_SND_PCM_FMT_S32`: signed 32-bit little endian.
    S32,
}

impl GuestFormat {
    /// Maps a `virtio_snd` format code, or `None` when we cannot decode it.
    pub(crate) fn from_virtio(format: u8) -> Option<Self> {
        Some(match format {
            VIRTIO_SND_PCM_FMT_U8 => Self::U8,
            VIRTIO_SND_PCM_FMT_S16 => Self::S16,
            VIRTIO_SND_PCM_FMT_S24 => Self::S24,
            VIRTIO_SND_PCM_FMT_S32 => Self::S32,
            _ => return None,
        })
    }

    /// Bytes one sample occupies in the guest buffer.
    pub(crate) const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16 => 2,
            Self::S24 | Self::S32 => 4,
        }
    }

    /// Decodes one sample into the `-1.0..=1.0` range.
    ///
    /// `raw` must be exactly [`Self::bytes`] long; callers slice it with
    /// `chunks_exact`, so the array conversions below cannot fail.
    pub(crate) fn decode(self, raw: &[u8]) -> f32 {
        debug_assert_eq!(raw.len(), self.bytes());
        match self {
            Self::U8 => (f32::from(raw[0]) - 128.0) / 128.0,
            Self::S16 => {
                let v = i16::from_le_bytes([raw[0], raw[1]]);
                f32::from(v) / 32_768.0
            }
            Self::S24 => {
                // Low 24 bits carry the sample; sign-extend through the top byte.
                let v = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                ((v << 8) >> 8) as f32 / 8_388_608.0
            }
            Self::S32 => {
                i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f32 / 2_147_483_648.0
            }
        }
    }
}

/// Maps a `virtio_snd` rate code to Hz, or `None` when we do not advertise it.
pub(crate) fn rate_from_virtio(rate: u8) -> Option<u32> {
    Some(match rate {
        VIRTIO_SND_PCM_RATE_8000 => 8_000,
        VIRTIO_SND_PCM_RATE_11025 => 11_025,
        VIRTIO_SND_PCM_RATE_16000 => 16_000,
        VIRTIO_SND_PCM_RATE_22050 => 22_050,
        VIRTIO_SND_PCM_RATE_32000 => 32_000,
        VIRTIO_SND_PCM_RATE_44100 => 44_100,
        VIRTIO_SND_PCM_RATE_48000 => 48_000,
        _ => return None,
    })
}

//--------------------------------------------------------------------------------------------------
// Playback: the state the host audio callback owns
//--------------------------------------------------------------------------------------------------

/// Per-second frame counters, enabled with `MSB_SND_STATS=1`.
struct Stats {
    stream_id: u32,
    window_start: Instant,
    frames: u64,
    silence: u64,
}

impl Stats {
    fn new(stream_id: u32) -> Self {
        Self {
            stream_id,
            window_start: Instant::now(),
            frames: 0,
            silence: 0,
        }
    }

    fn record(&mut self, frames: u64, silence: u64) {
        self.frames += frames;
        self.silence += silence;
        let elapsed = self.window_start.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let secs = elapsed.as_secs_f64();
            log::info!(
                "virtio-snd cpal: stream {}: {:.0} frames/s to the host device ({} silence frames \
                 in {:.2}s)",
                self.stream_id,
                self.frames as f64 / secs,
                self.silence,
                secs
            );
            self.window_start = Instant::now();
            self.frames = 0;
            self.silence = 0;
        }
    }
}

/// Everything the host audio callback needs to turn guest buffers into host
/// samples. Lives inside the callback closure, so it is only ever touched from
/// the audio thread.
struct Playback {
    streams: Arc<RwLock<Vec<Stream>>>,
    stream_id: u32,
    format: GuestFormat,
    src_channels: usize,
    dst_channels: usize,
    /// Source frames consumed per output frame (`guest_rate / host_rate`).
    step: f64,
    /// Whether `step != 1.0`, i.e. whether we interpolate between two frames.
    resampling: bool,
    /// Position of the next output sample within `src`, in source frames.
    pos: f64,
    /// Interleaved source samples decoded from guest memory, not yet played.
    src: VecDeque<f32>,
    /// Guest bytes read but not yet forming a whole frame.
    raw: Vec<u8>,
    /// Set once a guest memory read fails, to keep the log from flooding.
    read_failed: bool,
    stats: Option<Stats>,
}

impl Playback {
    /// Decodes at least `frames` source frames into `src`, pulling from the
    /// guest's queued buffers.
    ///
    /// Buffers that run dry are popped; dropping them is what marks the guest's
    /// request complete. The `streams` write lock is taken here and released
    /// before returning — the same order (`streams` then the vring mutex, taken
    /// inside `read_output`) the worker thread uses.
    fn pull(&mut self, frames: usize) {
        let sample_bytes = self.format.bytes();
        let frame_bytes = sample_bytes * self.src_channels;
        let have = self.src.len() / self.src_channels;
        if have >= frames {
            return;
        }
        // `raw` holds at most a partial frame, so it never exceeds what we want.
        let mut want = ((frames - have) * frame_bytes).saturating_sub(self.raw.len());

        {
            let mut streams = match self.streams.write() {
                Ok(streams) => streams,
                // A poisoned lock means another thread panicked while holding
                // it; play silence rather than propagating the panic into the
                // host audio callback.
                Err(poisoned) => poisoned.into_inner(),
            };
            let Some(stream) = streams.get_mut(self.stream_id as usize) else {
                return;
            };
            while want > 0 {
                let Some(buffer) = stream.buffers.front_mut() else {
                    break;
                };
                let avail = (buffer.desc_len() as usize).saturating_sub(buffer.pos);
                if avail == 0 {
                    stream.buffers.pop_front();
                    continue;
                }
                let n = want.min(avail);
                let at = self.raw.len();
                self.raw.resize(at + n, 0);
                match buffer.read_output(&mut self.raw[at..]) {
                    Ok(read) => {
                        let read = read as usize;
                        self.raw.truncate(at + read);
                        buffer.pos += read;
                        want -= read;
                        if buffer.pos >= buffer.desc_len() as usize {
                            stream.buffers.pop_front();
                        }
                        if read == 0 {
                            // Guest memory gave us nothing; stop rather than spin.
                            break;
                        }
                    }
                    Err(err) => {
                        self.raw.truncate(at);
                        if !self.read_failed {
                            self.read_failed = true;
                            log::error!(
                                "virtio-snd cpal: stream {}: reading guest PCM failed: {err}",
                                self.stream_id
                            );
                        }
                        stream.buffers.pop_front();
                        break;
                    }
                }
            }
        }

        let whole = self.raw.len() - self.raw.len() % frame_bytes;
        for sample in self.raw[..whole].chunks_exact(sample_bytes) {
            self.src.push_back(self.format.decode(sample));
        }
        self.raw.drain(..whole);
    }

    /// Source channel feeding host channel `dst`.
    fn map_channel(&self, dst: usize) -> usize {
        if self.src_channels == 1 {
            0
        } else {
            dst.min(self.src_channels - 1)
        }
    }

    /// Fills one host callback's worth of samples, padding with silence on
    /// underrun.
    fn render<T: Sample + FromSample<f32>>(&mut self, out: &mut [T]) {
        if self.dst_channels == 0 {
            return;
        }
        let out_frames = out.len() / self.dst_channels;
        // One extra frame so interpolation always has its right-hand sample.
        let needed = (self.pos + self.step * out_frames as f64).ceil() as usize + 1;
        self.pull(needed);

        let lookahead = usize::from(self.resampling);
        let mut produced = 0;
        for frame in 0..out_frames {
            let index = self.pos as usize;
            if index + lookahead >= self.src.len() / self.src_channels {
                break;
            }
            let t = (self.pos - index as f64) as f32;
            for channel in 0..self.dst_channels {
                let src = self.map_channel(channel);
                let a = self.src[index * self.src_channels + src];
                let sample = if self.resampling {
                    let b = self.src[(index + 1) * self.src_channels + src];
                    a + (b - a) * t
                } else {
                    a
                };
                out[frame * self.dst_channels + channel] = T::from_sample(sample);
            }
            self.pos += self.step;
            produced += 1;
        }
        for sample in out[produced * self.dst_channels..].iter_mut() {
            *sample = T::EQUILIBRIUM;
        }

        // When the host rate is less than half the guest's, one output frame
        // can step past everything `pull` returned; clamp so the drain stays
        // inside `src` (a panic here would unwind into the host callback).
        let consumed = (self.pos as usize).min(self.src.len() / self.src_channels);
        self.src.drain(..consumed * self.src_channels);
        self.pos -= consumed as f64;

        if let Some(stats) = self.stats.as_mut() {
            stats.record(produced as u64, (out_frames - produced) as u64);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Backend
//--------------------------------------------------------------------------------------------------

pub struct CpalBackend {
    streams: Arc<RwLock<Vec<Stream>>>,
    /// Host streams by virtio stream id. Only output streams appear here.
    host_streams: RwLock<HashMap<u32, ::cpal::Stream>>,
    /// `MSB_SND_STATS=1` was set when the backend was created.
    stats: bool,
    /// Whether the "capture is unimplemented" warning has been logged.
    capture_warned: AtomicBool,
}

impl CpalBackend {
    pub fn new(streams: Arc<RwLock<Vec<Stream>>>) -> Self {
        let stats = std::env::var(STATS_ENV).is_ok_and(|v| v == "1");
        log::info!("virtio-snd: using the cpal host audio backend (stats: {stats})");
        Self {
            streams,
            host_streams: RwLock::new(HashMap::new()),
            stats,
            capture_warned: AtomicBool::new(false),
        }
    }

    /// Opens a host output stream matching the guest's negotiated parameters.
    fn open_output(&self, stream_id: u32) -> Result<::cpal::Stream> {
        let (format, guest_rate, guest_channels) = {
            let streams = self.streams.read().map_err(lock_poisoned)?;
            let stream = streams
                .get(stream_id as usize)
                .ok_or(Error::StreamWithIdNotFound(stream_id))?;
            let params = &stream.params;
            let format = GuestFormat::from_virtio(params.format)
                .ok_or(Error::UnexpectedAudioBackendConfiguration)?;
            let rate =
                rate_from_virtio(params.rate).ok_or(Error::UnexpectedAudioBackendConfiguration)?;
            if params.channels == 0 {
                return Err(Error::ChannelNotSupported(params.channels));
            }
            (format, rate, params.channels)
        };

        let host = ::cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| {
            Error::UnexpectedAudioBackendError("no default host output device".to_owned())
        })?;
        let device_name = device
            .description()
            .map_or_else(|_| "<unknown>".to_owned(), |d| d.name().to_owned());

        let supported = choose_output_config(&device, guest_rate, u16::from(guest_channels))?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.config();
        let host_rate = config.sample_rate;
        let dst_channels = config.channels as usize;
        let step = f64::from(guest_rate) / f64::from(host_rate);

        log::info!(
            "virtio-snd cpal: stream {stream_id} -> \"{device_name}\": guest {guest_rate} Hz \
             {guest_channels} ch {format:?}, host {host_rate} Hz {dst_channels} ch \
             {sample_format:?}{}",
            if step == 1.0 { "" } else { " (resampling)" }
        );
        if dst_channels != usize::from(guest_channels) {
            log::warn!(
                "virtio-snd cpal: stream {stream_id}: host device has {dst_channels} channels but \
                 the guest asked for {guest_channels}; channels are mapped by index"
            );
        }

        let playback = Playback {
            streams: Arc::clone(&self.streams),
            stream_id,
            format,
            src_channels: usize::from(guest_channels),
            dst_channels,
            step,
            resampling: step != 1.0,
            pos: 0.0,
            src: VecDeque::new(),
            raw: Vec::new(),
            read_failed: false,
            stats: self.stats.then(|| Stats::new(stream_id)),
        };

        build_output_stream(&device, &config, sample_format, playback)
    }
}

/// Turns a poisoned lock into a backend error instead of a panic.
fn lock_poisoned<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::UnexpectedAudioBackendError("virtio-snd stream state lock was poisoned".to_owned())
}

/// Picks a host output configuration for the guest's rate and channel count.
///
/// Preference order: the guest's rate and channel count, then the guest's rate
/// with whatever channel count the device offers, then the device default
/// (which makes [`Playback`] resample). Float formats are preferred because the
/// mixer works in `f32`.
fn choose_output_config(
    device: &Device,
    rate: u32,
    channels: u16,
) -> Result<SupportedStreamConfig> {
    let ranges: Vec<_> = device
        .supported_output_configs()
        .map_err(|e| {
            Error::UnexpectedAudioBackendError(format!("no supported output configs: {e}"))
        })?
        .collect();

    let fits_rate = |r: &&::cpal::SupportedStreamConfigRange| {
        r.min_sample_rate() <= rate && rate <= r.max_sample_rate()
    };
    let best = ranges
        .iter()
        .filter(|r| r.channels() == channels)
        .filter(fits_rate)
        .max_by_key(|r| format_rank(r.sample_format()))
        .or_else(|| {
            ranges
                .iter()
                .filter(fits_rate)
                .max_by_key(|r| format_rank(r.sample_format()))
        });

    match best {
        Some(range) => Ok((*range).with_sample_rate(rate)),
        None => device.default_output_config().map_err(|e| {
            Error::UnexpectedAudioBackendError(format!("no default output config: {e}"))
        }),
    }
}

/// Ranks host sample formats; higher is preferred.
fn format_rank(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::F32 => 4,
        SampleFormat::I16 => 3,
        SampleFormat::I32 => 2,
        SampleFormat::F64 => 1,
        _ => 0,
    }
}

/// Builds the host output stream for `sample_format`, monomorphising
/// [`Playback::render`] over the host's sample type.
fn build_output_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    playback: Playback,
) -> Result<::cpal::Stream> {
    fn build<T: SizedSample + FromSample<f32>>(
        device: &Device,
        config: &StreamConfig,
        mut playback: Playback,
    ) -> Result<::cpal::Stream> {
        device
            .build_output_stream::<T, _, _>(
                *config,
                move |data: &mut [T], _| playback.render(data),
                |err| log::error!("virtio-snd cpal: host output stream error: {err}"),
                None,
            )
            .map_err(|e| {
                Error::UnexpectedAudioBackendError(format!("could not build output stream: {e}"))
            })
    }

    match sample_format {
        SampleFormat::I8 => build::<i8>(device, config, playback),
        SampleFormat::I16 => build::<i16>(device, config, playback),
        SampleFormat::I32 => build::<i32>(device, config, playback),
        SampleFormat::I64 => build::<i64>(device, config, playback),
        SampleFormat::U8 => build::<u8>(device, config, playback),
        SampleFormat::U16 => build::<u16>(device, config, playback),
        SampleFormat::U32 => build::<u32>(device, config, playback),
        SampleFormat::U64 => build::<u64>(device, config, playback),
        SampleFormat::F32 => build::<f32>(device, config, playback),
        SampleFormat::F64 => build::<f64>(device, config, playback),
        other => Err(Error::UnexpectedAudioBackendError(format!(
            "unsupported host sample format {other:?}"
        ))),
    }
}

impl AudioBackend for CpalBackend {
    fn write(&self, stream_id: u32) -> Result<()> {
        // Buffers were queued by the worker; the host audio callback picks them
        // up on its own clock. Only the stream state is checked here.
        let streams = self.streams.read().map_err(lock_poisoned)?;
        let stream = streams
            .get(stream_id as usize)
            .ok_or(Error::StreamWithIdNotFound(stream_id))?;
        if !matches!(stream.state, PCMState::Start | PCMState::Prepare) {
            return Err(Error::Stream(StreamError::InvalidState(
                "write",
                stream.state,
            )));
        }
        Ok(())
    }

    fn read(&self, stream_id: u32) -> Result<()> {
        if !self.capture_warned.swap(true, Ordering::Relaxed) {
            log::warn!(
                "virtio-snd cpal: capture is not implemented; guest recording from stream \
                 {stream_id} will read silence and its buffers stay pending"
            );
        }
        Ok(())
    }

    fn set_parameters(&self, stream_id: u32, request: VirtioSndPcmSetParams) -> Result<()> {
        let mut streams = self.streams.write().map_err(lock_poisoned)?;
        let stream = streams
            .get_mut(stream_id as usize)
            .ok_or(Error::StreamWithIdNotFound(stream_id))?;
        if let Err(err) = stream.state.set_parameters() {
            log::error!("virtio-snd cpal: stream {stream_id} set_parameters: {err}");
            return Err(Error::Stream(err));
        }
        if !stream.supports_format(request.format) || !stream.supports_rate(request.rate) {
            return Err(Error::UnexpectedAudioBackendConfiguration);
        }
        // Reject anything this backend cannot decode before the audio callback
        // has to deal with it: the worker turns this into VIRTIO_SND_S_NOT_SUPP
        // and the guest driver picks another format.
        if GuestFormat::from_virtio(request.format).is_none()
            || rate_from_virtio(request.rate).is_none()
        {
            log::warn!(
                "virtio-snd cpal: stream {stream_id}: guest asked for format {} rate {}, which \
                 this backend cannot convert",
                request.format,
                request.rate
            );
            return Err(Error::UnexpectedAudioBackendConfiguration);
        }
        if request.channels == 0 {
            return Err(Error::ChannelNotSupported(request.channels));
        }
        stream.params.features = request.features;
        stream.params.buffer_bytes = request.buffer_bytes;
        stream.params.period_bytes = request.period_bytes;
        stream.params.channels = request.channels;
        stream.params.format = request.format;
        stream.params.rate = request.rate;
        Ok(())
    }

    fn prepare(&self, stream_id: u32) -> Result<()> {
        let direction = {
            let mut streams = self.streams.write().map_err(lock_poisoned)?;
            let stream = streams
                .get_mut(stream_id as usize)
                .ok_or(Error::StreamWithIdNotFound(stream_id))?;
            if let Err(err) = stream.state.prepare() {
                log::error!("virtio-snd cpal: stream {stream_id} prepare: {err}");
                return Err(Error::Stream(err));
            }
            stream.direction
        };

        // Replacing a prepared stream: take the old handle out of the map and
        // drop it with no lock held — dropping a cpal stream waits for the host
        // audio callback, which itself takes the `streams` lock.
        let previous = self
            .host_streams
            .write()
            .map_err(lock_poisoned)?
            .remove(&stream_id);
        drop(previous);

        if matches!(direction, Direction::Input) {
            if !self.capture_warned.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "virtio-snd cpal: capture is not implemented; guest recording from stream \
                     {stream_id} will read silence and its buffers stay pending"
                );
            }
            return Ok(());
        }

        let host_stream = self.open_output(stream_id)?;
        self.host_streams
            .write()
            .map_err(lock_poisoned)?
            .insert(stream_id, host_stream);
        Ok(())
    }

    fn release(&self, stream_id: u32) -> Result<()> {
        {
            let mut streams = self.streams.write().map_err(lock_poisoned)?;
            let stream = streams
                .get_mut(stream_id as usize)
                .ok_or(Error::StreamWithIdNotFound(stream_id))?;
            if let Err(err) = stream.state.release() {
                log::error!("virtio-snd cpal: stream {stream_id} release: {err}");
                return Err(Error::Stream(err));
            }
        }

        // Stop the host callback first (no lock held, see `prepare`), then hand
        // back whatever the guest still had queued: dropping those buffers
        // completes their requests.
        let host_stream = self
            .host_streams
            .write()
            .map_err(lock_poisoned)?
            .remove(&stream_id);
        drop(host_stream);

        let pending = {
            let mut streams = self.streams.write().map_err(lock_poisoned)?;
            streams
                .get_mut(stream_id as usize)
                .map(|stream| std::mem::take(&mut stream.buffers))
        };
        drop(pending);
        Ok(())
    }

    fn start(&self, stream_id: u32) -> Result<()> {
        {
            let mut streams = self.streams.write().map_err(lock_poisoned)?;
            let stream = streams
                .get_mut(stream_id as usize)
                .ok_or(Error::StreamWithIdNotFound(stream_id))?;
            if let Err(err) = stream.state.start() {
                log::error!("virtio-snd cpal: stream {stream_id} start: {err}");
                return Err(Error::Stream(err));
            }
        }
        let host_streams = self.host_streams.read().map_err(lock_poisoned)?;
        match host_streams.get(&stream_id) {
            Some(host) => host.play().map_err(|e| {
                Error::UnexpectedAudioBackendError(format!("could not start output stream: {e}"))
            }),
            // Capture streams have no host stream; starting them is a no-op.
            None => Ok(()),
        }
    }

    fn stop(&self, stream_id: u32) -> Result<()> {
        {
            let mut streams = self.streams.write().map_err(lock_poisoned)?;
            let stream = streams
                .get_mut(stream_id as usize)
                .ok_or(Error::StreamWithIdNotFound(stream_id))?;
            if let Err(err) = stream.state.stop() {
                log::error!("virtio-snd cpal: stream {stream_id} stop: {err}");
                return Err(Error::Stream(err));
            }
        }
        let host_streams = self.host_streams.read().map_err(lock_poisoned)?;
        match host_streams.get(&stream_id) {
            Some(host) => host.pause().map_err(|e| {
                Error::UnexpectedAudioBackendError(format!("could not stop output stream: {e}"))
            }),
            None => Ok(()),
        }
    }

    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    #[test]
    fn test_guest_format_from_virtio() {
        assert_eq!(
            GuestFormat::from_virtio(VIRTIO_SND_PCM_FMT_U8),
            Some(GuestFormat::U8)
        );
        assert_eq!(
            GuestFormat::from_virtio(VIRTIO_SND_PCM_FMT_S16),
            Some(GuestFormat::S16)
        );
        assert_eq!(
            GuestFormat::from_virtio(VIRTIO_SND_PCM_FMT_S24),
            Some(GuestFormat::S24)
        );
        assert_eq!(
            GuestFormat::from_virtio(VIRTIO_SND_PCM_FMT_S32),
            Some(GuestFormat::S32)
        );
        // VIRTIO_SND_PCM_FMT_FLOAT is not in SUPPORTED_FORMATS.
        assert_eq!(
            GuestFormat::from_virtio(super::super::super::virtio_sound::VIRTIO_SND_PCM_FMT_FLOAT),
            None
        );
    }

    #[test]
    fn test_guest_format_bytes() {
        assert_eq!(GuestFormat::U8.bytes(), 1);
        assert_eq!(GuestFormat::S16.bytes(), 2);
        assert_eq!(GuestFormat::S24.bytes(), 4);
        assert_eq!(GuestFormat::S32.bytes(), 4);
    }

    #[test]
    fn test_decode_silence_and_extremes() {
        approx(GuestFormat::U8.decode(&[128]), 0.0);
        approx(GuestFormat::U8.decode(&[0]), -1.0);
        approx(GuestFormat::S16.decode(&[0, 0]), 0.0);
        approx(GuestFormat::S16.decode(&0i16.to_le_bytes()), 0.0);
        approx(GuestFormat::S16.decode(&i16::MIN.to_le_bytes()), -1.0);
        approx(GuestFormat::S16.decode(&16_384i16.to_le_bytes()), 0.5);
        approx(GuestFormat::S32.decode(&i32::MIN.to_le_bytes()), -1.0);
        approx(GuestFormat::S32.decode(&0i32.to_le_bytes()), 0.0);
    }

    #[test]
    fn test_decode_s24_sign_extends() {
        // -(1 << 23) is full negative scale in a 24-bit container.
        approx(GuestFormat::S24.decode(&0x0080_0000u32.to_le_bytes()), -1.0);
        // A negative sample that is not full scale still sign-extends.
        approx(
            GuestFormat::S24.decode(&0x00FF_8000u32.to_le_bytes()),
            -0.003_906_25,
        );
        approx(GuestFormat::S24.decode(&0x0000_0000u32.to_le_bytes()), 0.0);
        // The top byte is padding and must not reach the sample value.
        approx(GuestFormat::S24.decode(&0xFF00_0000u32.to_le_bytes()), 0.0);
        approx(GuestFormat::S24.decode(&0x0040_0000u32.to_le_bytes()), 0.5);
    }

    #[test]
    fn test_rate_from_virtio() {
        assert_eq!(rate_from_virtio(VIRTIO_SND_PCM_RATE_48000), Some(48_000));
        assert_eq!(rate_from_virtio(VIRTIO_SND_PCM_RATE_44100), Some(44_100));
        assert_eq!(rate_from_virtio(VIRTIO_SND_PCM_RATE_8000), Some(8_000));
        // 96 kHz is outside SUPPORTED_RATES.
        assert_eq!(
            rate_from_virtio(super::super::super::virtio_sound::VIRTIO_SND_PCM_RATE_96000),
            None
        );
    }

    /// A `Playback` with no guest buffers: it must emit silence, not panic.
    fn silent_playback(step: f64, dst_channels: usize) -> Playback {
        Playback {
            streams: Arc::new(RwLock::new(vec![Stream::default()])),
            stream_id: 0,
            format: GuestFormat::S16,
            src_channels: 2,
            dst_channels,
            step,
            resampling: step != 1.0,
            pos: 0.0,
            src: VecDeque::new(),
            raw: Vec::new(),
            read_failed: false,
            stats: None,
        }
    }

    #[test]
    fn test_render_underrun_is_silence() {
        let mut playback = silent_playback(1.0, 2);
        let mut out = [1.0f32; 8];
        playback.render(&mut out);
        assert_eq!(out, [0.0; 8]);
    }

    #[test]
    fn test_render_passthrough() {
        let mut playback = silent_playback(1.0, 2);
        playback.src.extend([0.25, -0.25, 0.5, -0.5]);
        let mut out = [1.0f32; 6];
        playback.render(&mut out);
        // Two frames of audio, then silence.
        assert_eq!(out, [0.25, -0.25, 0.5, -0.5, 0.0, 0.0]);
        assert!(playback.src.is_empty());
        approx(playback.pos as f32, 0.0);
    }

    #[test]
    fn test_render_interpolates_when_resampling() {
        // Guest 24 kHz into a 48 kHz host: one source frame per two output ones.
        let mut playback = silent_playback(0.5, 2);
        playback.src.extend([0.0, 0.0, 1.0, -1.0, 1.0, -1.0]);
        let mut out = [9.0f32; 6];
        playback.render(&mut out);
        approx(out[0], 0.0);
        approx(out[1], 0.0);
        approx(out[2], 0.5);
        approx(out[3], -0.5);
        approx(out[4], 1.0);
        approx(out[5], -1.0);
    }

    #[test]
    fn test_render_does_not_overrun_source_when_downsampling() {
        // 48 kHz guest into a 16 kHz host: three source frames per output one,
        // so a single output frame can step past everything that is buffered.
        let mut playback = silent_playback(3.0, 2);
        playback.pos = 0.9;
        playback.src.extend([0.5, -0.5, 0.25, -0.25]);
        let mut out = [9.0f32; 8];
        playback.render(&mut out);
        // One frame is produced, the rest is silence, and nothing panics.
        assert_eq!(out[2..], [0.0; 6]);
        assert!(playback.src.is_empty());
    }

    #[test]
    fn test_render_mono_guest_fills_stereo_host() {
        let mut playback = silent_playback(1.0, 2);
        playback.src_channels = 1;
        playback.src.extend([0.5, -0.5]);
        let mut out = [9.0f32; 4];
        playback.render(&mut out);
        assert_eq!(out, [0.5, 0.5, -0.5, -0.5]);
    }
}
