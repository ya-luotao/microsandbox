//! Host side of the experimental virtio-gpu display.
//!
//! The VMM hands every flushed scanout frame to a `krun_display` backend on
//! the gpu worker thread. Two backends live here:
//!
//! * [`SharedFrameBackend`] (via [`DisplayServer`]) copies frames into a
//!   memory-mapped file per scanout and tells a viewer connected to the
//!   sandbox's `display.sock` which slot is fresh; the viewer's keyboard and
//!   pointer events come back over the same socket and are queued into two
//!   virtio-input devices. `msb display <sandbox>` is that viewer.
//! * [`dump::FrameDumpBackend`] keeps the latest frame on disk for debugging.
//!
//! A guest that drives the cursor plane also sends its pointer image and
//! position through [`SharedFrameBackend`], which spares it a full scanout
//! flush per pointer move.
//!
//! `present_frame` must never wait for a viewer: the guest's FLUSH is only
//! answered once it returns, and the same holds for the cursor methods.

pub mod clipboard;
pub mod dump;
pub mod input;
pub mod protocol;

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use memmap2::{MmapMut, MmapOptions};
use msb_krun::krun_display::{
    CursorImage, DisplayBackend, DisplayBackendBasicFramebuffer, DisplayBackendCursor,
    DisplayBackendError, DisplayBackendNew, IntoDisplayBackendWithCursor, Rect, ResourceFormat,
    MAX_DISPLAYS,
};
use msb_krun::krun_input::{InputConfigBackend, InputEvent, InputEventProviderBackend};
use msb_krun_utils::pollable_channel::PollableChannelSender;

use clipboard::ClipboardPortBackend;
pub use dump::frame_dump_backend;
use input::{event, syn, InputDevice};
use protocol::evdev::*;
use protocol::{ServerMsg, ViewerMsg, SLOTS};

/// A viewer connection; only one is served at a time.
struct Viewer {
    id: u64,
    stream: UnixStream,
}

struct ScanoutInfo {
    width: u32,
    height: u32,
    format: String,
    path: PathBuf,
}

/// State shared between the gpu worker thread (frames), the accept thread
/// and the per-connection reader threads (input).
struct Shared {
    sandbox: String,
    fb_dir: PathBuf,
    viewer: Mutex<Option<Viewer>>,
    scanouts: Mutex<Vec<Option<ScanoutInfo>>>,
    keyboard_tx: PollableChannelSender<InputEvent>,
    pointer_tx: PollableChannelSender<InputEvent>,
    clipboard: Arc<ClipboardPortBackend>,
    /// The latest [`ServerMsg::Cursor`] per scanout, so a viewer that attaches
    /// after the guest set its cursor still gets the image.
    cursors: Mutex<Vec<Option<ServerMsg>>>,
}

impl Shared {
    fn send(&self, msg: &ServerMsg) {
        let mut guard = self.viewer.lock().unwrap_or_else(|e| e.into_inner());
        let Some(viewer) = guard.as_mut() else { return };
        let mut line = serde_json::to_vec(msg).expect("protocol messages serialize");
        line.push(b'\n');
        if let Err(e) = viewer.stream.write_all(&line) {
            tracing::info!(viewer = viewer.id, error = %e, "gpu display: dropping viewer");
            *guard = None;
        }
    }

    fn attach(&self, id: u64, stream: UnixStream) {
        // A stalled viewer must not stall the guest's FLUSH.
        let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
        {
            let mut guard = self.viewer.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(Viewer { id, stream });
        }
        tracing::info!(viewer = id, "gpu display: viewer attached");
        self.send(&ServerMsg::Hello {
            sandbox: self.sandbox.clone(),
        });
        {
            let scanouts = self.scanouts.lock().unwrap_or_else(|e| e.into_inner());
            for (scanout, info) in scanouts.iter().enumerate() {
                if let Some(info) = info {
                    self.send(&configure_msg(scanout as u32, info));
                }
            }
        }
        {
            let cursors = self.cursors.lock().unwrap_or_else(|e| e.into_inner());
            for msg in cursors.iter().flatten() {
                self.send(msg);
            }
        }
        // A viewer that attached after the guest copied still gets that value.
        if let Some(msg) = self.clipboard.last_guest() {
            self.send(&msg);
        }
    }

    fn detach(&self, id: u64) {
        let mut guard = self.viewer.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_some_and(|v| v.id == id) {
            *guard = None;
            tracing::info!(viewer = id, "gpu display: viewer detached");
        }
    }

    fn handle(&self, msg: ViewerMsg) {
        let result = match msg {
            ViewerMsg::Key { code, down } => self
                .keyboard_tx
                .send_many([event(EV_KEY, code, u32::from(down)), syn()]),
            ViewerMsg::Abs { x, y } => self.pointer_tx.send_many([
                event(EV_ABS, ABS_X, x),
                event(EV_ABS, ABS_Y, y),
                syn(),
            ]),
            ViewerMsg::Btn { code, down } => self
                .pointer_tx
                .send_many([event(EV_KEY, code, u32::from(down)), syn()]),
            ViewerMsg::Rel { code, value } => self
                .pointer_tx
                .send_many([event(EV_REL, code, value as u32), syn()]),
            ViewerMsg::Clipboard { mime, data } => {
                self.clipboard.send_to_guest(mime, data);
                return;
            }
        };
        if let Err(e) = result {
            tracing::warn!(error = %e, "gpu display: input event dropped");
        }
    }
}

fn configure_msg(scanout: u32, info: &ScanoutInfo) -> ServerMsg {
    ServerMsg::Configure {
        scanout,
        width: info.width,
        height: info.height,
        format: info.format.clone(),
        path: info.path.to_string_lossy().into_owned(),
        slots: SLOTS,
    }
}

/// Serves scanout frames and accepts input on the sandbox's display socket.
pub struct DisplayServer {
    shared: &'static &'static Shared,
    keyboard: Option<InputDevice>,
    pointer: Option<InputDevice>,
}

impl DisplayServer {
    /// Listen on `socket`; frame files go under `fb_dir`.
    pub fn start(sandbox: &str, fb_dir: &Path, socket: &Path) -> io::Result<Self> {
        let keyboard = input::keyboard()?;
        let pointer = input::pointer()?;
        let clipboard = Arc::new(ClipboardPortBackend::new());
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            sandbox: sandbox.to_string(),
            fb_dir: fb_dir.to_path_buf(),
            viewer: Mutex::new(None),
            scanouts: Mutex::new((0..MAX_DISPLAYS).map(|_| None).collect()),
            keyboard_tx: keyboard.tx.clone(),
            pointer_tx: pointer.tx.clone(),
            clipboard: Arc::clone(&clipboard),
            cursors: Mutex::new((0..MAX_DISPLAYS).map(|_| None).collect()),
        }));
        // `Shared` had to exist first; the backend only forwards after this.
        clipboard.set_shared(shared);
        fs::create_dir_all(fb_dir)?;
        let _ = fs::remove_file(socket);
        let listener = UnixListener::bind(socket)?;
        std::thread::Builder::new()
            .name("gpu display".into())
            .spawn(move || accept_loop(shared, listener))?;
        tracing::info!(socket = %socket.display(), "gpu display: listening");
        Ok(Self {
            shared: Box::leak(Box::new(shared)),
            keyboard: Some(keyboard),
            pointer: Some(pointer),
        })
    }

    /// The display backend to hand to the VM builder.
    pub fn display_backend(&self) -> DisplayBackend<'static> {
        SharedFrameBackend::into_display_backend_with_cursor(Some(self.shared))
    }

    /// The vsock backend serving the guest clipboard agent.
    ///
    /// Register it on [`protocol::CLIPBOARD_VSOCK_PORT`] whenever the display
    /// server runs, independent of any user-configured vsock routes.
    pub fn clipboard_backend(&self) -> Arc<ClipboardPortBackend> {
        Arc::clone(&self.shared.clipboard)
    }

    /// The virtual keyboard's backends (once).
    pub fn take_keyboard(
        &mut self,
    ) -> Option<(InputConfigBackend<'static>, InputEventProviderBackend<'static>)> {
        self.keyboard.take().map(|d| (d.config, d.events))
    }

    /// The virtual pointer's backends (once).
    pub fn take_pointer(
        &mut self,
    ) -> Option<(InputConfigBackend<'static>, InputEventProviderBackend<'static>)> {
        self.pointer.take().map(|d| (d.config, d.events))
    }
}

fn accept_loop(shared: &'static Shared, listener: UnixListener) {
    let mut next_id = 1u64;
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                tracing::warn!(error = %e, "gpu display: accept failed");
                continue;
            }
        };
        let id = next_id;
        next_id += 1;
        let writer = match stream.try_clone() {
            Ok(writer) => writer,
            Err(e) => {
                tracing::warn!(error = %e, "gpu display: clone failed");
                continue;
            }
        };
        shared.attach(id, writer);
        if let Err(e) = std::thread::Builder::new()
            .name(format!("gpu display viewer {id}"))
            .spawn(move || read_loop(shared, id, stream))
        {
            tracing::warn!(error = %e, "gpu display: reader thread failed");
            shared.detach(id);
        }
    }
}

fn read_loop(shared: &'static Shared, id: u64, stream: UnixStream) {
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { break };
        match serde_json::from_str::<ViewerMsg>(&line) {
            Ok(msg) => shared.handle(msg),
            Err(e) => tracing::debug!(error = %e, "gpu display: bad viewer message"),
        }
    }
    shared.detach(id);
}

struct Frames {
    mmap: MmapMut,
    frame_size: usize,
    format: ResourceFormat,
    seq: u64,
    slot: usize,
}

/// Display backend writing frames into per-scanout shared files.
pub struct SharedFrameBackend {
    shared: &'static Shared,
    frames: Vec<Option<Frames>>,
}

impl DisplayBackendNew<&'static Shared> for SharedFrameBackend {
    fn new(userdata: Option<&&'static Shared>) -> Self {
        Self {
            shared: *userdata.expect("DisplayServer passes its state as userdata"),
            frames: (0..MAX_DISPLAYS).map(|_| None).collect(),
        }
    }
}

fn map_frames(path: &Path, frame_size: usize) -> io::Result<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len((frame_size * SLOTS as usize) as u64)?;
    // SAFETY: the file is private to the sandbox runtime dir and only ever
    // remapped read-only by the viewer.
    unsafe { MmapOptions::new().map_mut(&file) }
}

impl DisplayBackendBasicFramebuffer for SharedFrameBackend {
    fn configure_scanout(
        &mut self,
        scanout_id: u32,
        display_width: u32,
        display_height: u32,
        width: u32,
        height: u32,
        format: ResourceFormat,
    ) -> Result<(), DisplayBackendError> {
        if scanout_id as usize >= self.frames.len() {
            return Err(DisplayBackendError::InvalidScanoutId);
        }
        let frame_size = width as usize * height as usize * ResourceFormat::BYTES_PER_PIXEL;
        // The guest re-issues SET_SCANOUT on every page flip; keep the
        // mapping (and the viewer's) when nothing about it changed.
        if let Some(frames) = &self.frames[scanout_id as usize] {
            if frames.frame_size == frame_size && frames.format == format {
                return Ok(());
            }
        }
        let path = self.shared.fb_dir.join(format!("scanout{scanout_id}.fb"));
        let mmap = map_frames(&path, frame_size).map_err(|e| {
            tracing::warn!(scanout_id, error = %e, "gpu display: cannot map frame file");
            DisplayBackendError::InternalError
        })?;
        tracing::info!(
            scanout_id,
            display_width,
            display_height,
            width,
            height,
            ?format,
            "gpu display: scanout configured"
        );
        self.frames[scanout_id as usize] = Some(Frames {
            mmap,
            frame_size,
            format,
            seq: 0,
            slot: 0,
        });
        let info = ScanoutInfo {
            width,
            height,
            format: format!("{format:?}"),
            path,
        };
        let msg = configure_msg(scanout_id, &info);
        {
            let mut scanouts = self.shared.scanouts.lock().unwrap_or_else(|e| e.into_inner());
            scanouts[scanout_id as usize] = Some(info);
        }
        self.shared.send(&msg);
        Ok(())
    }

    fn disable_scanout(&mut self, scanout_id: u32) -> Result<(), DisplayBackendError> {
        let slot = self
            .frames
            .get_mut(scanout_id as usize)
            .ok_or(DisplayBackendError::InvalidScanoutId)?;
        *slot = None;
        let path = {
            let mut scanouts = self.shared.scanouts.lock().unwrap_or_else(|e| e.into_inner());
            scanouts[scanout_id as usize].take().map(|info| info.path)
        };
        if let Some(path) = path {
            let _ = fs::remove_file(path);
        }
        tracing::info!(scanout_id, "gpu display: scanout disabled");
        self.shared.send(&ServerMsg::Disable {
            scanout: scanout_id,
        });
        Ok(())
    }

    fn alloc_frame(&mut self, scanout_id: u32) -> Result<(u32, &mut [u8]), DisplayBackendError> {
        let frames = self
            .frames
            .get_mut(scanout_id as usize)
            .and_then(Option::as_mut)
            .ok_or(DisplayBackendError::InvalidScanoutId)?;
        frames.slot = ((frames.seq + 1) % u64::from(SLOTS)) as usize;
        let start = frames.slot * frames.frame_size;
        let end = start + frames.frame_size;
        Ok((frames.slot as u32, &mut frames.mmap[start..end]))
    }

    fn present_frame(
        &mut self,
        scanout_id: u32,
        frame_id: u32,
        rect: Option<&Rect>,
    ) -> Result<(), DisplayBackendError> {
        let frames = self
            .frames
            .get_mut(scanout_id as usize)
            .and_then(Option::as_mut)
            .ok_or(DisplayBackendError::InvalidScanoutId)?;
        frames.seq += 1;
        self.shared.send(&ServerMsg::Frame {
            scanout: scanout_id,
            slot: frame_id,
            seq: frames.seq,
            rect: rect.map(|r| [r.x, r.y, r.width, r.height]),
        });
        Ok(())
    }
}

/// The cursor image as RGBA8888, which is what the viewer's custom cursor
/// wants.
///
/// virtio-gpu names a format by its bytes in memory order, so `BGRA` is
/// `[b, g, r, a]`; the `X` variants carry no alpha and become opaque.
fn to_rgba(format: ResourceFormat, data: &[u8]) -> Vec<u8> {
    let (r, g, b, a) = match format {
        ResourceFormat::BGRA => (2, 1, 0, Some(3)),
        ResourceFormat::BGRX => (2, 1, 0, None),
        ResourceFormat::ARGB => (1, 2, 3, Some(0)),
        ResourceFormat::XRGB => (1, 2, 3, None),
        ResourceFormat::RGBA => (0, 1, 2, Some(3)),
        ResourceFormat::RGBX => (0, 1, 2, None),
        ResourceFormat::ABGR => (3, 2, 1, Some(0)),
        ResourceFormat::XBGR => (3, 2, 1, None),
    };
    data.chunks_exact(ResourceFormat::BYTES_PER_PIXEL)
        .flat_map(|px| [px[r], px[g], px[b], a.map_or(0xff, |a| px[a])])
        .collect()
}

/// The guest's cursor plane. `SharedFrameBackend` only forwards it; the viewer
/// turns it into the window's cursor, so the pointer moves without the guest
/// re-flushing the whole scanout.
impl DisplayBackendCursor for SharedFrameBackend {
    fn set_cursor(
        &mut self,
        scanout_id: u32,
        image: Option<CursorImage<'_>>,
    ) -> Result<(), DisplayBackendError> {
        if scanout_id as usize >= self.frames.len() {
            return Err(DisplayBackendError::InvalidScanoutId);
        }
        // An empty image is how the guest hides its cursor.
        let msg = match image {
            Some(image) => ServerMsg::Cursor {
                scanout: scanout_id,
                width: image.width,
                height: image.height,
                hot_x: image.hot_x,
                hot_y: image.hot_y,
                rgba: base64::engine::general_purpose::STANDARD
                    .encode(to_rgba(image.format, image.data)),
            },
            None => ServerMsg::Cursor {
                scanout: scanout_id,
                width: 0,
                height: 0,
                hot_x: 0,
                hot_y: 0,
                rgba: String::new(),
            },
        };
        {
            let mut cursors = self.shared.cursors.lock().unwrap_or_else(|e| e.into_inner());
            cursors[scanout_id as usize] = Some(msg.clone());
        }
        self.shared.send(&msg);
        Ok(())
    }

    fn move_cursor(&mut self, scanout_id: u32, x: i32, y: i32) -> Result<(), DisplayBackendError> {
        if scanout_id as usize >= self.frames.len() {
            return Err(DisplayBackendError::InvalidScanoutId);
        }
        self.shared.send(&ServerMsg::CursorPos {
            scanout: scanout_id,
            x,
            y,
        });
        Ok(())
    }
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// One pixel whose bytes are distinguishable in every position.
    const PIXEL: [u8; 4] = [0x10, 0x20, 0x30, 0x40];

    #[test]
    fn bgra_pixels_become_rgba() {
        // [b, g, r, a] -> [r, g, b, a]
        assert_eq!(
            to_rgba(ResourceFormat::BGRA, &PIXEL),
            vec![0x30, 0x20, 0x10, 0x40]
        );
        // [a, r, g, b] -> [r, g, b, a]
        assert_eq!(
            to_rgba(ResourceFormat::ARGB, &PIXEL),
            vec![0x20, 0x30, 0x40, 0x10]
        );
        // [a, b, g, r] -> [r, g, b, a]
        assert_eq!(
            to_rgba(ResourceFormat::ABGR, &PIXEL),
            vec![0x40, 0x30, 0x20, 0x10]
        );
        assert_eq!(to_rgba(ResourceFormat::RGBA, &PIXEL), PIXEL.to_vec());
    }

    #[test]
    fn formats_without_alpha_are_opaque() {
        assert_eq!(
            to_rgba(ResourceFormat::BGRX, &PIXEL),
            vec![0x30, 0x20, 0x10, 0xff]
        );
        assert_eq!(
            to_rgba(ResourceFormat::XRGB, &PIXEL),
            vec![0x20, 0x30, 0x40, 0xff]
        );
        assert_eq!(
            to_rgba(ResourceFormat::RGBX, &PIXEL),
            vec![0x10, 0x20, 0x30, 0xff]
        );
        assert_eq!(
            to_rgba(ResourceFormat::XBGR, &PIXEL),
            vec![0x40, 0x30, 0x20, 0xff]
        );
    }

    #[test]
    fn every_pixel_is_converted_and_a_short_tail_is_dropped() {
        let mut data = PIXEL.to_vec();
        data.extend_from_slice(&[1, 2, 3, 4]);
        // A trailing partial pixel cannot be converted and is left out.
        data.extend_from_slice(&[9, 9]);
        assert_eq!(
            to_rgba(ResourceFormat::BGRA, &data),
            vec![0x30, 0x20, 0x10, 0x40, 3, 2, 1, 4]
        );
    }
}
