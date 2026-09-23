// Manos Pitsidianakis <manos.pitsidianakis@linaro.org>
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

#[cfg(target_os = "macos")]
mod cpal;
#[cfg(target_os = "linux")]
mod pipewire;

use std::sync::{Arc, RwLock};

#[cfg(target_os = "macos")]
use self::cpal::CpalBackend;
#[cfg(target_os = "linux")]
use self::pipewire::PwBackend;
use super::{stream::Stream, BackendType, Error, Result, VirtioSndPcmSetParams};

pub trait AudioBackend {
    fn write(&self, stream_id: u32) -> Result<()>;

    #[allow(dead_code)]
    fn read(&self, stream_id: u32) -> Result<()>;

    fn set_parameters(&self, _stream_id: u32, _: VirtioSndPcmSetParams) -> Result<()> {
        Ok(())
    }

    fn prepare(&self, _stream_id: u32) -> Result<()> {
        Ok(())
    }

    fn release(&self, _stream_id: u32) -> Result<()> {
        Ok(())
    }

    fn start(&self, _stream_id: u32) -> Result<()> {
        Ok(())
    }

    fn stop(&self, _stream_id: u32) -> Result<()> {
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any;
}

pub fn alloc_audio_backend(
    backend: BackendType,
    streams: Arc<RwLock<Vec<Stream>>>,
) -> Result<Box<dyn AudioBackend + Send + Sync>> {
    log::trace!("allocating audio backend {backend:?}");
    match backend {
        #[cfg(target_os = "linux")]
        BackendType::Pipewire => Ok(Box::new(PwBackend::new(streams))),
        #[cfg(target_os = "macos")]
        BackendType::Cpal => Ok(Box::new(CpalBackend::new(streams))),
        other => {
            log::error!("no host audio backend for {other:?} is compiled in on this platform");
            Err(Error::AudioBackendNotSupported)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_matches_target() {
        if cfg!(target_os = "linux") {
            assert_eq!(BackendType::default(), BackendType::Pipewire);
        } else {
            assert_eq!(BackendType::default(), BackendType::Cpal);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_alloc_cpal_backend() {
        use std::any::TypeId;

        let value = alloc_audio_backend(BackendType::Cpal, Default::default()).unwrap();
        assert_eq!(TypeId::of::<CpalBackend>(), value.as_any().type_id());
    }
}
