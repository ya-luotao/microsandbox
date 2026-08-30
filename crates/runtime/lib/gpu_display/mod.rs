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
//! answered once it returns, and the same holds for the cursor methods. So the
//! gpu worker never touches the socket — it queues onto a bounded channel that
//! a per-connection writer thread drains. A viewer that stops reading fills the
//! queue and its messages are dropped rather than blocking the guest. A dropped
//! frame is superseded by the next one and a dropped clipboard is recoverable
//! from the backend, but a dropped `Configure`/`Disable` or cursor image would
//! leave the viewer acting on a scanout or a cursor that no longer exists, so
//! the writer replays the current state of those once it catches up.

pub mod clipboard;
pub mod dump;
pub mod input;
pub mod protocol;

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
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

/// Messages queued for one viewer before the slowest are dropped.
///
/// A cursor image is ~22 KB of base64 and the socket buffers hold 8 KB, so a
/// write on the gpu worker thread would wait for the viewer's reader. The queue
/// absorbs that; 256 messages is a few frames' worth of slack.
const VIEWER_QUEUE: usize = 256;

/// State a viewer must be told again after its queue overflowed.
///
/// Most dropped messages need no follow-up: a `Frame` is superseded by the next
/// one, and a `Clipboard` is recoverable from the backend's `last_guest`. These
/// two are not — the viewer would be left acting on a scanout or a cursor that
/// no longer exists.
#[derive(Default)]
struct Dirty {
    /// Scanouts whose `Configure` or `Disable` was dropped, as a bitmask.
    ///
    /// Losing one is not cosmetic: the viewer mmaps the scanout's frame file at
    /// the size the last `Configure` gave it, and `configure_scanout` truncates
    /// and reopens the same path. A mode shrink whose `Configure` was dropped
    /// leaves the viewer reading past the new end of file, which is a SIGBUS.
    config: AtomicU64,
    /// A cursor image was dropped; the viewer would keep showing a stale one.
    cursor: AtomicBool,
}

impl Dirty {
    /// Note that `dropped` never reached the viewer.
    fn mark(&self, dropped: &ServerMsg) {
        match *dropped {
            ServerMsg::Configure { scanout, .. } | ServerMsg::Disable { scanout } => {
                if scanout < MAX_DISPLAYS as u32 {
                    self.config.fetch_or(1 << scanout, Ordering::Relaxed);
                }
            }
            ServerMsg::Cursor { .. } => self.cursor.store(true, Ordering::Relaxed),
            _ => {}
        }
    }
}

/// A viewer connection; only one is served at a time.
///
/// The socket is owned by a writer thread, so nothing the gpu worker does can
/// block on the viewer reading.
struct Viewer {
    id: u64,
    tx: SyncSender<ServerMsg>,
    dirty: Arc<Dirty>,
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
    /// Queue a message for the viewer. Never blocks: the gpu worker calls this
    /// from `present_frame` and the cursor methods, and the guest's FLUSH is
    /// waiting on it.
    fn send(&self, msg: ServerMsg) {
        let guard = self.viewer.lock().unwrap_or_else(|e| e.into_inner());
        let Some(viewer) = guard.as_ref() else { return };
        match viewer.tx.try_send(msg) {
            Ok(()) => {}
            Err(TrySendError::Full(dropped)) => {
                viewer.dirty.mark(&dropped);
                tracing::debug!(
                    viewer = viewer.id,
                    "gpu display: viewer queue full, dropping"
                );
            }
            // The writer thread is gone; it detaches on its way out.
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// The latest cursor image of every configured scanout.
    fn latest_cursors(&self) -> Vec<ServerMsg> {
        let cursors = self.cursors.lock().unwrap_or_else(|e| e.into_inner());
        cursors.iter().flatten().cloned().collect()
    }

    fn attach(&'static self, id: u64, stream: UnixStream) {
        // A stalled viewer must not stall its writer thread forever either.
        let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
        let (tx, rx) = sync_channel(VIEWER_QUEUE);
        let dirty = Arc::new(Dirty::default());
        {
            let mut guard = self.viewer.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(Viewer {
                id,
                tx,
                dirty: Arc::clone(&dirty),
            });
        }
        if let Err(e) = std::thread::Builder::new()
            .name(format!("gpu display writer {id}"))
            .spawn(move || writer_loop(self, id, stream, rx, dirty))
        {
            tracing::warn!(error = %e, "gpu display: writer thread failed");
            self.detach(id);
            return;
        }
        tracing::info!(viewer = id, "gpu display: viewer attached");
        // Everything below goes through the same queue, so the viewer sees
        // Hello, then Configure, then the cursor, in that order.
        self.send(ServerMsg::Hello {
            sandbox: self.sandbox.clone(),
        });
        {
            let scanouts = self.scanouts.lock().unwrap_or_else(|e| e.into_inner());
            for (scanout, info) in scanouts.iter().enumerate() {
                if let Some(info) = info {
                    self.send(configure_msg(scanout as u32, info));
                }
            }
        }
        for msg in self.latest_cursors() {
            self.send(msg);
        }
        // A viewer that attached after the guest copied still gets that value.
        if let Some(msg) = self.clipboard.last_guest() {
            self.send(msg);
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

/// What a viewer must be re-sent after its queue overflowed: the current state
/// of every scanout whose `Configure`/`Disable` was dropped, then the latest
/// cursor images. Config first — a cursor means nothing to a viewer that has
/// the wrong idea of the scanout it belongs to.
///
/// The state is read now rather than replaying what was dropped, so a scanout
/// that changed twice while the queue was full still lands on its real size.
fn overflow_replay(
    dirty: &Dirty,
    scanouts: &Mutex<Vec<Option<ScanoutInfo>>>,
    cursors: &Mutex<Vec<Option<ServerMsg>>>,
) -> Vec<ServerMsg> {
    let mut replay = Vec::new();
    let config = dirty.config.swap(0, Ordering::Relaxed);
    if config != 0 {
        let scanouts = scanouts.lock().unwrap_or_else(|e| e.into_inner());
        for (scanout, info) in scanouts.iter().enumerate() {
            if config & (1 << scanout) == 0 {
                continue;
            }
            let scanout = scanout as u32;
            replay.push(match info {
                Some(info) => configure_msg(scanout, info),
                None => ServerMsg::Disable { scanout },
            });
        }
    }
    if dirty.cursor.swap(false, Ordering::Relaxed) {
        let cursors = cursors.lock().unwrap_or_else(|e| e.into_inner());
        replay.extend(cursors.iter().flatten().cloned());
    }
    replay
}

/// Owns the viewer's socket so no write ever happens on the gpu worker thread.
fn writer_loop(
    shared: &'static Shared,
    id: u64,
    mut stream: UnixStream,
    rx: Receiver<ServerMsg>,
    dirty: Arc<Dirty>,
) {
    fn write_msg(stream: &mut UnixStream, msg: &ServerMsg) -> io::Result<()> {
        let mut line = serde_json::to_vec(msg).expect("protocol messages serialize");
        line.push(b'\n');
        stream.write_all(&line)
    }

    for msg in rx {
        if let Err(e) = write_msg(&mut stream, &msg) {
            tracing::info!(viewer = id, error = %e, "gpu display: dropping viewer");
            break;
        }
        // Caught up after dropping something the viewer cannot recover on its
        // own; put it back in step before carrying on.
        for msg in overflow_replay(&dirty, &shared.scanouts, &shared.cursors) {
            if let Err(e) = write_msg(&mut stream, &msg) {
                tracing::info!(viewer = id, error = %e, "gpu display: dropping viewer");
                return shared.detach(id);
            }
        }
    }
    shared.detach(id);
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
    /// Hash of the cursor image last sent per scanout. Compositors re-set the
    /// same cursor fb on some pointer paths, and the image is ~22 KB of base64.
    cursor_hashes: Vec<Option<u64>>,
}

impl DisplayBackendNew<&'static Shared> for SharedFrameBackend {
    fn new(userdata: Option<&&'static Shared>) -> Self {
        Self {
            shared: *userdata.expect("DisplayServer passes its state as userdata"),
            frames: (0..MAX_DISPLAYS).map(|_| None).collect(),
            cursor_hashes: (0..MAX_DISPLAYS).map(|_| None).collect(),
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
        self.shared.send(msg);
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
        self.shared.send(ServerMsg::Disable {
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
        self.shared.send(ServerMsg::Frame {
            scanout: scanout_id,
            slot: frame_id,
            seq: frames.seq,
            rect: rect.map(|r| [r.x, r.y, r.width, r.height]),
        });
        Ok(())
    }
}

/// Identity of a cursor image, so an unchanged one is not re-encoded and re-sent.
fn hash_cursor(image: &CursorImage<'_>) -> u64 {
    let mut hasher = DefaultHasher::new();
    (image.width, image.height, image.hot_x, image.hot_y).hash(&mut hasher);
    (image.format as u32).hash(&mut hasher);
    image.data.hash(&mut hasher);
    hasher.finish()
}

/// The cursor image as straight-alpha RGBA8888, which is what the viewer's
/// custom cursor wants.
///
/// virtio-gpu names a format by its bytes in memory order, so `BGRA` is
/// `[b, g, r, a]`; the `X` variants carry no alpha and become opaque.
///
/// The pixels arrive premultiplied — a compositor renders its cursor through
/// the same GL pipeline as everything else, and the cursor plane carries the
/// result untouched (measured on Hyprland's arrow: of 182 antialiased edge
/// pixels, none had a colour channel above its alpha). winit's
/// `CustomCursor::from_rgba` documents the opposite, so undo the multiply or
/// every soft edge is darkened toward black.
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
        .flat_map(|px| {
            let alpha = a.map_or(0xff, |a| px[a]);
            [
                unpremultiply(px[r], alpha),
                unpremultiply(px[g], alpha),
                unpremultiply(px[b], alpha),
                alpha,
            ]
        })
        .collect()
}

/// `colour / alpha`, rounded, for one premultiplied channel.
fn unpremultiply(colour: u8, alpha: u8) -> u8 {
    match alpha {
        // Nothing to recover, and the colour is meaningless at zero coverage.
        0 => 0,
        0xff => colour,
        alpha => {
            let (colour, alpha) = (u32::from(colour), u32::from(alpha));
            ((colour * 255 + alpha / 2) / alpha).min(255) as u8
        }
    }
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
        let (msg, hash) = match image {
            Some(image) => {
                let hash = hash_cursor(&image);
                // The same image again changes nothing on the viewer, and it is
                // ~22 KB of base64 per send.
                if self.cursor_hashes[scanout_id as usize] == Some(hash) {
                    return Ok(());
                }
                (
                    ServerMsg::Cursor {
                        scanout: scanout_id,
                        width: image.width,
                        height: image.height,
                        hot_x: image.hot_x,
                        hot_y: image.hot_y,
                        rgba: base64::engine::general_purpose::STANDARD
                            .encode(to_rgba(image.format, image.data)),
                    },
                    Some(hash),
                )
            }
            None => (
                ServerMsg::Cursor {
                    scanout: scanout_id,
                    width: 0,
                    height: 0,
                    hot_x: 0,
                    hot_y: 0,
                    rgba: String::new(),
                },
                None,
            ),
        };
        self.cursor_hashes[scanout_id as usize] = hash;
        {
            let mut cursors = self.shared.cursors.lock().unwrap_or_else(|e| e.into_inner());
            cursors[scanout_id as usize] = Some(msg.clone());
        }
        self.shared.send(msg);
        Ok(())
    }

    fn move_cursor(&mut self, scanout_id: u32, x: i32, y: i32) -> Result<(), DisplayBackendError> {
        if scanout_id as usize >= self.frames.len() {
            return Err(DisplayBackendError::InvalidScanoutId);
        }
        self.shared.send(ServerMsg::CursorPos {
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

    /// One opaque pixel whose bytes are distinguishable in every position.
    /// Opaque so the channel-order tests are not perturbed by unpremultiply.
    const PIXEL: [u8; 4] = [0x10, 0x20, 0x30, 0xff];

    #[test]
    fn bgra_pixels_become_rgba() {
        // [b, g, r, a] -> [r, g, b, a]
        assert_eq!(
            to_rgba(ResourceFormat::BGRA, &PIXEL),
            vec![0x30, 0x20, 0x10, 0xff]
        );
        // [a, r, g, b] -> [r, g, b, a]; alpha 0x10 scales the rest up
        assert_eq!(
            to_rgba(ResourceFormat::ARGB, &PIXEL),
            vec![0xff, 0xff, 0xff, 0x10]
        );
        // [a, b, g, r] -> [r, g, b, a]
        assert_eq!(
            to_rgba(ResourceFormat::ABGR, &PIXEL),
            vec![0xff, 0xff, 0xff, 0x10]
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
            vec![0x20, 0x30, 0xff, 0xff]
        );
        assert_eq!(
            to_rgba(ResourceFormat::RGBX, &PIXEL),
            vec![0x10, 0x20, 0x30, 0xff]
        );
        assert_eq!(
            to_rgba(ResourceFormat::XBGR, &PIXEL),
            vec![0xff, 0x30, 0x20, 0xff]
        );
    }

    #[test]
    fn every_pixel_is_converted_and_a_short_tail_is_dropped() {
        let mut data = PIXEL.to_vec();
        data.extend_from_slice(&[10, 20, 30, 40]);
        // A trailing partial pixel cannot be converted and is left out.
        data.extend_from_slice(&[9, 9]);
        assert_eq!(
            to_rgba(ResourceFormat::BGRA, &data),
            vec![
                0x30,
                0x20,
                0x10,
                0xff,
                unpremultiply(30, 40),
                unpremultiply(20, 40),
                unpremultiply(10, 40),
                40,
            ]
        );
    }

    /// Cursor pixels arrive premultiplied; winit wants straight alpha.
    #[test]
    fn premultiplied_pixels_are_restored_to_straight_alpha() {
        // A half-covered white edge pixel: premultiplied it is (128,128,128,128).
        assert_eq!(
            to_rgba(ResourceFormat::RGBA, &[128, 128, 128, 128]),
            vec![255, 255, 255, 128]
        );
        // Fully transparent carries no colour at all.
        assert_eq!(
            to_rgba(ResourceFormat::RGBA, &[0, 0, 0, 0]),
            vec![0, 0, 0, 0]
        );
        // Opaque pixels are untouched.
        assert_eq!(
            to_rgba(ResourceFormat::RGBA, &[10, 20, 30, 255]),
            vec![10, 20, 30, 255]
        );
        // A colour channel above its alpha would not be premultiplied; clamp
        // rather than overflow.
        assert_eq!(
            to_rgba(ResourceFormat::RGBA, &[200, 0, 0, 100]),
            vec![255, 0, 0, 100]
        );
    }

    fn cursor_msg(scanout: u32, rgba: &str) -> ServerMsg {
        ServerMsg::Cursor {
            scanout,
            width: 1,
            height: 1,
            hot_x: 0,
            hot_y: 0,
            rgba: rgba.to_string(),
        }
    }

    fn scanout_info(width: u32, height: u32) -> ScanoutInfo {
        ScanoutInfo {
            width,
            height,
            format: "BGRX".to_string(),
            path: PathBuf::from("/tmp/scanout0.fb"),
        }
    }

    /// A viewer that stops reading must not stall the gpu worker, and must not
    /// be left acting on state that was dropped on the way.
    #[test]
    fn a_full_queue_drops_rather_than_blocking() {
        let (tx, rx) = sync_channel::<ServerMsg>(1);
        let dirty = Arc::new(Dirty::default());
        let viewer = Viewer {
            id: 1,
            tx,
            dirty: Arc::clone(&dirty),
        };

        // One message fits; everything after it is dropped rather than blocking.
        assert!(matches!(viewer.tx.try_send(cursor_msg(0, "first")), Ok(())));
        let Err(TrySendError::Full(dropped)) = viewer.tx.try_send(cursor_msg(0, "second")) else {
            panic!("expected a full queue");
        };
        // This is what `Shared::send` does with the message it could not queue.
        viewer.dirty.mark(&dropped);
        assert!(dirty.cursor.load(Ordering::Relaxed));
        assert!(matches!(rx.recv(), Ok(ServerMsg::Cursor { .. })));
    }

    /// A dropped `Configure` is the dangerous one: the viewer mmaps the frame
    /// file at the size it last heard, and `configure_scanout` truncates and
    /// reopens the same path, so a mode shrink it missed is a SIGBUS.
    #[test]
    fn a_dropped_configure_is_replayed_from_current_state() {
        let dirty = Dirty::default();
        let scanouts = Mutex::new(vec![Some(scanout_info(1920, 1080)), None]);
        let cursors: Mutex<Vec<Option<ServerMsg>>> = Mutex::new(vec![None, None]);

        // Nothing was dropped, so there is nothing to put right.
        assert!(overflow_replay(&dirty, &scanouts, &cursors).is_empty());

        // The guest shrank the mode while the viewer was behind, and the
        // Configure never made it. The replay carries the size it is at *now*,
        // not the one that was dropped.
        dirty.mark(&configure_msg(0, &scanout_info(1280, 720)));
        *scanouts.lock().unwrap() = vec![Some(scanout_info(800, 600)), None];
        let replay = overflow_replay(&dirty, &scanouts, &cursors);
        assert!(
            matches!(
                replay.as_slice(),
                [ServerMsg::Configure {
                    scanout: 0,
                    width: 800,
                    height: 600,
                    ..
                }]
            ),
            "got {replay:?}"
        );
        // The flag is consumed, so a caught-up viewer is not spammed.
        assert!(overflow_replay(&dirty, &scanouts, &cursors).is_empty());

        // A scanout the guest has since turned off replays the disable, not a
        // stale size.
        dirty.mark(&configure_msg(1, &scanout_info(640, 480)));
        assert!(matches!(
            overflow_replay(&dirty, &scanouts, &cursors).as_slice(),
            [ServerMsg::Disable { scanout: 1 }]
        ));

        // Config comes before the cursor: a cursor means nothing to a viewer
        // that has the wrong idea of the scanout it belongs to.
        *cursors.lock().unwrap() = vec![Some(cursor_msg(0, "img")), None];
        dirty.mark(&ServerMsg::Disable { scanout: 0 });
        dirty.mark(&cursor_msg(0, "dropped"));
        assert!(matches!(
            overflow_replay(&dirty, &scanouts, &cursors).as_slice(),
            [
                ServerMsg::Configure { scanout: 0, .. },
                ServerMsg::Cursor { scanout: 0, .. }
            ]
        ));
    }

    /// A dropped frame is superseded by the next one, and a dropped clipboard
    /// is recoverable from the backend's `last_guest`.
    #[test]
    fn dropped_frames_and_clipboard_need_no_replay() {
        let dirty = Dirty::default();
        dirty.mark(&ServerMsg::Frame {
            scanout: 0,
            slot: 0,
            seq: 1,
            rect: None,
        });
        dirty.mark(&ServerMsg::Clipboard {
            mime: protocol::TEXT_MIME.to_string(),
            data: String::new(),
        });
        assert_eq!(dirty.config.load(Ordering::Relaxed), 0);
        assert!(!dirty.cursor.load(Ordering::Relaxed));
    }

    /// The same image again is not worth ~22 KB of base64 on the wire.
    #[test]
    fn an_unchanged_cursor_image_hashes_the_same() {
        let pixels = [1u8, 2, 3, 4];
        let image = |hot_x| CursorImage {
            width: 1,
            height: 1,
            format: ResourceFormat::BGRA,
            data: &pixels,
            hot_x,
            hot_y: 0,
        };
        assert_eq!(hash_cursor(&image(0)), hash_cursor(&image(0)));
        // A moved hotspot is a different cursor even with the same pixels.
        assert_ne!(hash_cursor(&image(0)), hash_cursor(&image(1)));
        let other = [1u8, 2, 3, 5];
        assert_ne!(
            hash_cursor(&image(0)),
            hash_cursor(&CursorImage {
                width: 1,
                height: 1,
                format: ResourceFormat::BGRA,
                data: &other,
                hot_x: 0,
                hot_y: 0,
            })
        );
    }

    #[test]
    fn unpremultiply_rounds_and_saturates() {
        assert_eq!(unpremultiply(0, 0), 0);
        assert_eq!(unpremultiply(255, 255), 255);
        assert_eq!(unpremultiply(64, 128), 128);
        assert_eq!(unpremultiply(128, 128), 255);
        assert_eq!(unpremultiply(200, 100), 255);
    }
}
