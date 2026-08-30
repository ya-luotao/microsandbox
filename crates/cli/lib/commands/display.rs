//! `msb display` — show a sandbox's virtio-gpu scanout in a native window and
//! feed keyboard/pointer events back to it (macOS only).
//!
//! The sandbox process serves frames through a memory-mapped file per
//! scanout and a JSON-lines socket (see
//! `microsandbox_runtime::gpu_display::protocol`); this command is the viewer.
//! It runs the window event loop on the process's main thread, before the
//! Tokio runtime starts.

use clap::Args;

/// Open a native window on a running sandbox's display.
///
/// Clipboard text is kept in sync with the Mac pasteboard while the window is
/// open. Set `MSB_DISPLAY_CLIPBOARD=0` to turn that off in both directions.
///
/// A guest driving the virtio-gpu cursor plane has its pointer drawn as the
/// window's own cursor, so moving the mouse costs no frames at all; a guest
/// that renders its pointer into the scanout instead keeps the Mac's cursor
/// hidden over the window.
#[derive(Debug, Args)]
pub struct DisplayArgs {
    /// Sandbox whose display to show.
    pub name: String,
}

/// Execute `msb display`. Never returns.
#[cfg(not(target_os = "macos"))]
pub fn run(_args: DisplayArgs) -> ! {
    eprintln!("msb display is only available on macOS in this build");
    std::process::exit(2);
}

/// Execute `msb display`. Never returns.
#[cfg(target_os = "macos")]
pub fn run(args: DisplayArgs) -> ! {
    match macos::run(&args.name) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::fs::File;
    use std::hash::{Hash, Hasher};
    use std::io::{BufRead, BufReader, Write};
    use std::num::NonZeroU32;
    use std::os::unix::net::UnixStream;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, Context};
    use base64::Engine as _;
    use memmap2::Mmap;
    use microsandbox_runtime::gpu_display::protocol::evdev::*;
    use microsandbox_runtime::gpu_display::protocol::{ABS_RANGE, ServerMsg, TEXT_MIME, ViewerMsg};
    use microsandbox_runtime::ipc::{display_socket_path_for, sandbox_socket_paths};
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Cursor, CustomCursor, Window, WindowId};

    /// Only the first scanout is shown.
    const SCANOUT: u32 = 0;

    /// Set `MSB_DISPLAY_STATS=1` for a one-line frame/copy summary per second.
    const STATS_ENV: &str = "MSB_DISPLAY_STATS";

    /// Clipboard sync is on unless this is set to something other than `1`.
    ///
    /// The pasteboard is pushed into the sandbox whenever the window takes
    /// focus or a key is pressed, and the guest can replace what the Mac
    /// holds, so an untrusted image needs a way to opt out of both.
    const CLIPBOARD_ENV: &str = "MSB_DISPLAY_CLIPBOARD";

    /// Whether to sync the clipboard at all.
    fn clipboard_enabled() -> bool {
        std::env::var_os(CLIPBOARD_ENV).is_none_or(|v| v == "1")
    }

    enum UserEvent {
        Server(ServerMsg),
        Disconnected,
    }

    /// What the guest's cursor plane last asked for.
    ///
    /// With a hardware cursor the host cursor *is* the guest's cursor: the
    /// guest stops drawing its pointer into the scanout, so the window shows
    /// the guest's image instead of hiding the Mac's own arrow.
    enum GuestCursor {
        /// No `Cursor` message yet — the guest renders its pointer into the
        /// scanout, so the host cursor must stay hidden or there are two.
        Software,
        /// The guest hid its cursor.
        Hidden,
        /// The guest's cursor image.
        Image(CustomCursor),
    }

    struct Scanout {
        width: u32,
        height: u32,
        frame_size: usize,
        mmap: Mmap,
        slot: usize,
        scaler: Scaler,
    }

    /// Nearest-neighbour scaler for windows that are not the scanout size, with
    /// the x-index table cached across redraws so a resize costs one table
    /// build rather than a division per pixel.
    #[derive(Default)]
    struct Scaler {
        xmap: Vec<u32>,
        /// `(source width, destination width)` `xmap` was built for.
        built_for: (usize, usize),
        /// The source row currently expanded to `0RGB`.
        row: Vec<u32>,
    }

    impl Scaler {
        /// Scale a `sw` x `sh` BGRX frame into a `dw` x `dh` `0RGB` buffer.
        /// Integer factors duplicate pixels with slice fills, other sizes gather
        /// through the x-index table; either way each source row is expanded
        /// once and destination rows sampling the same source row are copied
        /// rather than rebuilt.
        fn scale_into(
            &mut self,
            src: &[u8],
            (sw, sh): (usize, usize),
            dst: &mut [u32],
            (dw, dh): (usize, usize),
        ) {
            if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
                return;
            }
            if src.len() < sw * sh * 4 || dst.len() < dw * dh {
                return;
            }
            let factor = (dw % sw == 0 && dh % sh == 0 && dw / sw == dh / sh).then(|| dw / sw);
            if factor.is_none() && self.built_for != (sw, dw) {
                self.xmap = (0..dw).map(|x| (x * sw / dw) as u32).collect();
                self.built_for = (sw, dw);
            }
            // The source row currently in `self.row`, and the destination row
            // holding it — consecutive destination rows usually share one.
            let mut built: Option<usize> = None;
            let mut prev = 0usize;
            for y in 0..dh {
                let sy = y * sh / dh;
                let (done, rest) = dst.split_at_mut(y * dw);
                let out = &mut rest[..dw];
                if built == Some(sy) {
                    out.copy_from_slice(&done[prev * dw..prev * dw + dw]);
                    continue;
                }
                self.row.clear();
                self.row.extend(
                    src[sy * sw * 4..(sy + 1) * sw * 4]
                        .chunks_exact(4)
                        .map(|px| u32::from_le_bytes([px[0], px[1], px[2], 0])),
                );
                match factor {
                    Some(k) => {
                        for (x, &px) in self.row.iter().enumerate() {
                            out[x * k..(x + 1) * k].fill(px);
                        }
                    }
                    None => {
                        for (o, &sx) in out.iter_mut().zip(&self.xmap) {
                            *o = self.row[sx as usize];
                        }
                    }
                }
                built = Some(sy);
                prev = y;
            }
        }
    }

    /// Byte view of a pixel slice. softbuffer's macOS backend takes `0RGB`
    /// `u32`s and renders them with `NoneSkipFirst` + little-endian byte order
    /// (`softbuffer-0.4.8/src/backends/cg.rs:326`), so a pixel is `[b, g, r, x]`
    /// in memory — exactly the guest's BGRX layout, with the top byte ignored.
    /// A same-size frame therefore reaches the window as one `memcpy`, with no
    /// per-pixel conversion at all.
    fn pixels_as_bytes_mut(pixels: &mut [u32]) -> &mut [u8] {
        const _: () = assert!(cfg!(target_endian = "little"), "0RGB pixels assume LE");
        // SAFETY: `u32` has no padding and no invalid bit patterns, and its
        // alignment is stricter than `u8`'s, so the same memory is a valid
        // `[u8]` four times as long, for the same lifetime.
        unsafe {
            std::slice::from_raw_parts_mut(pixels.as_mut_ptr().cast::<u8>(), pixels.len() * 4)
        }
    }

    /// Per-second counters printed when `MSB_DISPLAY_STATS=1`.
    struct Stats {
        since: Instant,
        frames: u64,
        redraws: u64,
        bytes: u64,
        draw: Duration,
        /// Cursor plane updates: new images, then moves.
        cursors: u64,
        cursor_moves: u64,
    }

    impl Stats {
        fn new() -> Stats {
            Stats {
                since: Instant::now(),
                frames: 0,
                redraws: 0,
                bytes: 0,
                draw: Duration::ZERO,
                cursors: 0,
                cursor_moves: 0,
            }
        }

        /// Print and reset once a second has passed.
        fn tick(&mut self) {
            let elapsed = self.since.elapsed();
            if elapsed < Duration::from_secs(1) {
                return;
            }
            let secs = elapsed.as_secs_f64();
            let per_redraw = if self.redraws == 0 {
                0.0
            } else {
                self.draw.as_secs_f64() * 1e3 / self.redraws as f64
            };
            eprintln!(
                "stats: {:.1} frames/s, {:.1} redraws/s, {:.1} MB/s copied, \
                 {per_redraw:.2} ms/redraw, {:.1} cursors/s, {:.1} cursor moves/s",
                self.frames as f64 / secs,
                self.redraws as f64 / secs,
                self.bytes as f64 / secs / 1e6,
                self.cursors as f64 / secs,
                self.cursor_moves as f64 / secs,
            );
            *self = Stats::new();
        }
    }

    struct Sender(Mutex<UnixStream>);

    impl Sender {
        fn send(&self, msg: &ViewerMsg) {
            let mut line = serde_json::to_vec(msg).expect("protocol messages serialize");
            line.push(b'\n');
            let mut stream = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let _ = stream.write_all(&line);
        }
    }

    struct App {
        sandbox: String,
        sender: Arc<Sender>,
        window: Option<Rc<Window>>,
        surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
        scanout: Option<Scanout>,
        wheel_carry: (f32, f32),
        stats: Option<Stats>,
        /// `None` when the pasteboard is unavailable; clipboard sync is then off.
        clipboard: Option<arboard::Clipboard>,
        /// Last text this viewer put on, or read from, the Mac pasteboard.
        last_host_text: Option<String>,
        /// Last text the guest sent, so it is never echoed back to the guest.
        last_guest_text: Option<String>,
        /// MIME types already reported as unsupported.
        warned_mimes: HashSet<String>,
        /// What the guest's cursor plane last asked for.
        cursor: GuestCursor,
        /// Hash of the image behind [`GuestCursor::Image`]: compositors re-send
        /// the same cursor on some pointer paths, and rebuilding the
        /// `CustomCursor` would allocate a new image every time.
        cursor_hash: u64,
        /// Whether the pointer is over the window.
        pointer_inside: bool,
    }

    impl App {
        fn ensure_window(&mut self, event_loop: &ActiveEventLoop, width: u32, height: u32) {
            if let Some(window) = &self.window {
                let _ = window.request_inner_size(PhysicalSize::new(width, height));
                return;
            }
            let attrs = Window::default_attributes()
                .with_title(format!("{} — microsandbox", self.sandbox))
                .with_inner_size(PhysicalSize::new(width, height));
            let window = match event_loop.create_window(attrs) {
                Ok(window) => Rc::new(window),
                Err(e) => {
                    eprintln!("error: cannot create window: {e}");
                    event_loop.exit();
                    return;
                }
            };
            let context = softbuffer::Context::new(window.clone()).expect("softbuffer context");
            let surface = softbuffer::Surface::new(&context, window.clone()).expect("surface");
            let size = window.inner_size();
            eprintln!(
                "window: {}x{} physical, scale {}",
                size.width,
                size.height,
                window.scale_factor()
            );
            self.window = Some(window);
            self.surface = Some(surface);
        }

        /// Send the Mac pasteboard to the guest when it holds something new.
        ///
        /// Cheap enough to call on focus and on every key press: reading the
        /// pasteboard is a local call and nothing is sent unless the text
        /// actually changed.
        fn sync_host_clipboard(&mut self) {
            let Some(clipboard) = &mut self.clipboard else {
                return;
            };
            // An empty or non-text pasteboard reads as an error; nothing to do.
            let Ok(text) = clipboard.get_text() else {
                return;
            };
            if text.is_empty()
                || self.last_host_text.as_deref() == Some(text.as_str())
                || self.last_guest_text.as_deref() == Some(text.as_str())
            {
                return;
            }
            self.sender.send(&ViewerMsg::Clipboard {
                mime: TEXT_MIME.to_string(),
                data: base64::engine::general_purpose::STANDARD.encode(&text),
            });
            self.last_host_text = Some(text);
        }

        /// Put the guest's selection on the Mac pasteboard.
        fn apply_guest_clipboard(&mut self, mime: String, data: String) {
            // Disabled, or no pasteboard: drop it without even decoding.
            if self.clipboard.is_none() {
                return;
            }
            if !mime.starts_with("text/plain") {
                if self.warned_mimes.insert(mime.clone()) {
                    eprintln!("warning: ignoring clipboard type {mime}");
                }
                return;
            }
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&data) else {
                eprintln!("warning: bad clipboard payload from sandbox");
                return;
            };
            let Ok(text) = String::from_utf8(bytes) else {
                eprintln!("warning: clipboard text from sandbox is not UTF-8");
                return;
            };
            let Some(clipboard) = &mut self.clipboard else {
                return;
            };
            if let Err(e) = clipboard.set_text(&text) {
                eprintln!("warning: cannot set the clipboard: {e}");
                return;
            }
            // Both, so neither the next poll nor a focus event bounces it back.
            self.last_guest_text = Some(text.clone());
            self.last_host_text = Some(text);
        }

        fn redraw(&mut self) {
            let started = Instant::now();
            let Some(window) = self.window.as_ref() else {
                return;
            };
            let (Some(surface), Some(scanout)) = (self.surface.as_mut(), self.scanout.as_mut())
            else {
                return;
            };
            let size = window.inner_size();
            let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
            else {
                return;
            };
            if surface.resize(w, h).is_err() {
                return;
            }
            let Scanout {
                width,
                height,
                frame_size,
                mmap,
                slot,
                scaler,
            } = scanout;
            let Some(src) = mmap.get(*slot * *frame_size..(*slot + 1) * *frame_size) else {
                return;
            };
            let bytes = src.len();
            // The slot always holds a complete frame, and softbuffer's macOS
            // surface buffer never persists — `buffer_mut` hands out a fresh
            // zeroed `Vec` and `present` moves it into a `CGDataProvider`
            // (cg.rs:261, 298), while `present_with_damage` ignores its damage
            // (cg.rs:364). Every redraw therefore writes the whole window out
            // of the whole slot; a viewer-side copy of the frame saves nothing.
            let Ok(mut buffer) = surface.buffer_mut() else {
                return;
            };
            let (sw, sh) = (*width as usize, *height as usize);
            let (dw, dh) = (size.width as usize, size.height as usize);
            if (sw, sh) == (dw, dh) && buffer.len() * 4 == bytes {
                pixels_as_bytes_mut(&mut buffer).copy_from_slice(src);
            } else {
                scaler.scale_into(src, (sw, sh), &mut buffer, (dw, dh));
            }
            let _ = buffer.present();
            if let Some(stats) = self.stats.as_mut() {
                stats.redraws += 1;
                stats.bytes += bytes as u64;
                stats.draw += started.elapsed();
                stats.tick();
            }
        }

        /// Reconcile the window's cursor with what the guest last asked for.
        ///
        /// Called on every change of the three inputs: the guest's cursor, the
        /// scanout, and whether the pointer is over the window.
        fn apply_cursor(&self) {
            let Some(window) = &self.window else {
                return;
            };
            // Away from a live scanout the Mac's own cursor is the user's.
            if !self.pointer_inside || self.scanout.is_none() {
                window.set_cursor(Cursor::default());
                window.set_cursor_visible(true);
                return;
            }
            match &self.cursor {
                // The guest already drew a pointer into the frame, or wants no
                // cursor at all; either way the host must not add one.
                GuestCursor::Software | GuestCursor::Hidden => window.set_cursor_visible(false),
                GuestCursor::Image(cursor) => {
                    window.set_cursor(cursor.clone());
                    window.set_cursor_visible(true);
                }
            }
        }

        /// Take a new cursor image from the guest. An empty image hides it.
        fn set_guest_cursor(
            &mut self,
            event_loop: &ActiveEventLoop,
            width: u32,
            height: u32,
            hot_x: u32,
            hot_y: u32,
            rgba: &str,
        ) {
            if let Some(stats) = self.stats.as_mut() {
                stats.cursors += 1;
                stats.tick();
            }
            if width == 0 || height == 0 || rgba.is_empty() {
                self.cursor = GuestCursor::Hidden;
                self.cursor_hash = 0;
                self.apply_cursor();
                return;
            }
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(rgba) else {
                eprintln!("warning: bad cursor image from sandbox");
                return;
            };
            let hash = hash_cursor(width, height, hot_x, hot_y, &bytes);
            if hash == self.cursor_hash && matches!(self.cursor, GuestCursor::Image(_)) {
                return;
            }
            // `from_rgba` rejects an oversized image, a byte count that does
            // not match, and a hotspot outside the image, so the casts below
            // are the only thing left to check.
            let (Ok(width), Ok(height), Ok(hot_x), Ok(hot_y)) = (
                u16::try_from(width),
                u16::try_from(height),
                u16::try_from(hot_x),
                u16::try_from(hot_y),
            ) else {
                eprintln!("warning: cursor image {width}x{height} from sandbox is out of range");
                return;
            };
            let source = match CustomCursor::from_rgba(bytes, width, height, hot_x, hot_y) {
                Ok(source) => source,
                Err(e) => {
                    eprintln!("warning: cannot use the sandbox's cursor image: {e}");
                    return;
                }
            };
            self.cursor = GuestCursor::Image(event_loop.create_custom_cursor(source));
            self.cursor_hash = hash;
            self.apply_cursor();
        }

        fn pointer_abs(&self, x: f64, y: f64) -> Option<(u32, u32)> {
            let size = self.window.as_ref()?.inner_size();
            if size.width == 0 || size.height == 0 {
                return None;
            }
            let scale = |v: f64, max: u32| -> u32 {
                ((v / f64::from(max)).clamp(0.0, 1.0) * f64::from(ABS_RANGE)).round() as u32
            };
            Some((scale(x, size.width), scale(y, size.height)))
        }
    }

    impl ApplicationHandler<UserEvent> for App {
        fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

        fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
            match event {
                UserEvent::Server(ServerMsg::Hello { sandbox }) => {
                    eprintln!("connected to {sandbox}");
                }
                UserEvent::Server(ServerMsg::Configure {
                    scanout,
                    width,
                    height,
                    format,
                    path,
                    slots,
                }) => {
                    if scanout != SCANOUT {
                        return;
                    }
                    let frame_size = width as usize * height as usize * 4;
                    let mmap = File::open(&path)
                        .and_then(|f| unsafe { Mmap::map(&f) })
                        .ok()
                        .filter(|m| m.len() >= frame_size * slots as usize);
                    let Some(mmap) = mmap else {
                        eprintln!("error: cannot map {path}");
                        return;
                    };
                    eprintln!("scanout {scanout}: {width}x{height} {format}");
                    self.scanout = Some(Scanout {
                        width,
                        height,
                        frame_size,
                        mmap,
                        slot: 0,
                        scaler: Scaler::default(),
                    });
                    self.ensure_window(event_loop, width, height);
                    self.apply_cursor();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                // `rect` is deliberately ignored. Linux's virtio-gpu driver
                // sets `ignore_damage_clips` whenever the plane's framebuffer
                // object changes, because uploads are done per buffer (v6.12
                // `drivers/gpu/drm/virtio/virtgpu_plane.c:91-97`), and a
                // compositor page-flips between buffers on every frame — so the
                // guest's FLUSH always covers the whole scanout. It stays in the
                // protocol for logging and for guests that do send real damage.
                UserEvent::Server(ServerMsg::Frame { scanout, slot, .. }) => {
                    if scanout != SCANOUT {
                        return;
                    }
                    let Some(s) = &mut self.scanout else { return };
                    s.slot = slot as usize;
                    if let Some(stats) = self.stats.as_mut() {
                        stats.frames += 1;
                        stats.tick();
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                UserEvent::Server(ServerMsg::Disable { scanout }) => {
                    if scanout == SCANOUT {
                        self.scanout = None;
                        self.apply_cursor();
                    }
                }
                UserEvent::Server(ServerMsg::Cursor {
                    scanout,
                    width,
                    height,
                    hot_x,
                    hot_y,
                    rgba,
                }) => {
                    if scanout != SCANOUT {
                        return;
                    }
                    self.set_guest_cursor(event_loop, width, height, hot_x, hot_y, &rgba);
                }
                // Deliberately not used for positioning. The host pointer is
                // what moves the guest's, through the absolute tablet, so
                // warping the Mac's cursor to the position the guest echoes
                // back would fight the user's own hand — and a round trip
                // behind it. It is counted in the stats, which is what tells
                // you the cursor plane is live.
                UserEvent::Server(ServerMsg::CursorPos { .. }) => {
                    if let Some(stats) = self.stats.as_mut() {
                        stats.cursor_moves += 1;
                        stats.tick();
                    }
                }
                UserEvent::Server(ServerMsg::Clipboard { mime, data }) => {
                    self.apply_guest_clipboard(mime, data);
                }
                UserEvent::Disconnected => {
                    eprintln!("sandbox display closed");
                    event_loop.exit();
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::RedrawRequested => self.redraw(),
                WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::Focused(true) => self.sync_host_clipboard(),
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.repeat {
                        return;
                    }
                    if event.state == ElementState::Pressed {
                        // Catches a copy made on the Mac without leaving the window.
                        self.sync_host_clipboard();
                    }
                    let PhysicalKey::Code(code) = event.physical_key else {
                        return;
                    };
                    let Some(code) = keycode_to_evdev(code) else {
                        return;
                    };
                    self.sender.send(&ViewerMsg::Key {
                        code,
                        down: event.state == ElementState::Pressed,
                    });
                }
                WindowEvent::CursorMoved { position, .. } => {
                    // `CursorEntered` does not fire when the window regains
                    // focus under a pointer that never left it.
                    if !self.pointer_inside {
                        self.pointer_inside = true;
                        self.apply_cursor();
                    }
                    if let Some((x, y)) = self.pointer_abs(position.x, position.y) {
                        self.sender.send(&ViewerMsg::Abs { x, y });
                    }
                }
                WindowEvent::CursorEntered { .. } => {
                    self.pointer_inside = true;
                    self.apply_cursor();
                }
                // winit keeps the cursor hidden while the window is focused
                // even once the pointer has left it, so restore it explicitly.
                WindowEvent::CursorLeft { .. } | WindowEvent::Focused(false) => {
                    self.pointer_inside = false;
                    self.apply_cursor();
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let code = match button {
                        MouseButton::Left => BTN_LEFT,
                        MouseButton::Right => BTN_RIGHT,
                        MouseButton::Middle => BTN_MIDDLE,
                        _ => return,
                    };
                    self.sender.send(&ViewerMsg::Btn {
                        code,
                        down: state == ElementState::Pressed,
                    });
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (x, y),
                        MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 40.0, p.y as f32 / 40.0),
                    };
                    self.wheel_carry.0 += dx;
                    self.wheel_carry.1 += dy;
                    let steps_x = self.wheel_carry.0.trunc();
                    let steps_y = self.wheel_carry.1.trunc();
                    self.wheel_carry.0 -= steps_x;
                    self.wheel_carry.1 -= steps_y;
                    if steps_y != 0.0 {
                        self.sender.send(&ViewerMsg::Rel {
                            code: REL_WHEEL,
                            value: steps_y as i32,
                        });
                    }
                    if steps_x != 0.0 {
                        self.sender.send(&ViewerMsg::Rel {
                            code: REL_HWHEEL,
                            value: steps_x as i32,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    fn reader(stream: UnixStream, proxy: EventLoopProxy<UserEvent>) {
        // A sandbox newer than this viewer sends variants it does not know, and
        // a cursor-plane guest sends them per pointer move. Warn once per tag.
        let mut warned: HashSet<String> = HashSet::new();
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            match serde_json::from_str::<ServerMsg>(&line) {
                Ok(msg) => {
                    if proxy.send_event(UserEvent::Server(msg)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let tag = serde_json::from_str::<serde_json::Value>(&line)
                        .ok()
                        .and_then(|v| v.get("t").and_then(|t| t.as_str()).map(str::to_owned))
                        .unwrap_or_else(|| "?".to_string());
                    if warned.insert(tag.clone()) {
                        eprintln!("warning: ignoring unknown message \"{tag}\" from sandbox: {e}");
                    }
                }
            }
        }
        let _ = proxy.send_event(UserEvent::Disconnected);
    }

    pub fn run(name: &str) -> anyhow::Result<()> {
        let run_dir = microsandbox_utils::resolve_home().join(microsandbox_utils::RUN_SUBDIR);
        let paths = sandbox_socket_paths(&run_dir, name);
        let socket = display_socket_path_for(&paths.agent);
        let stream = UnixStream::connect(&socket).with_context(|| {
            format!(
                "cannot connect to {}; is `{name}` running with MSB_GPU=1?",
                socket.display()
            )
        })?;
        let reader_stream = stream.try_clone().context("clone socket")?;
        let event_loop = EventLoop::<UserEvent>::with_user_event()
            .build()
            .map_err(|e| anyhow!("event loop: {e}"))?;
        let proxy = event_loop.create_proxy();
        std::thread::Builder::new()
            .name("display reader".into())
            .spawn(move || reader(reader_stream, proxy))
            .context("spawn reader")?;
        let mut app = App {
            sandbox: name.to_string(),
            sender: Arc::new(Sender(Mutex::new(stream))),
            window: None,
            surface: None,
            scanout: None,
            wheel_carry: (0.0, 0.0),
            stats: std::env::var_os(STATS_ENV)
                .is_some_and(|v| v == "1")
                .then(Stats::new),
            clipboard: match clipboard_enabled().then(arboard::Clipboard::new) {
                None => None,
                Some(Ok(clipboard)) => Some(clipboard),
                Some(Err(e)) => {
                    eprintln!("warning: clipboard sync disabled: {e}");
                    None
                }
            },
            last_host_text: None,
            last_guest_text: None,
            warned_mimes: HashSet::new(),
            cursor: GuestCursor::Software,
            cursor_hash: 0,
            pointer_inside: false,
        };
        event_loop
            .run_app(&mut app)
            .map_err(|e| anyhow!("event loop: {e}"))
    }

    /// Identity of a cursor image, so an unchanged one is not rebuilt.
    fn hash_cursor(width: u32, height: u32, hot_x: u32, hot_y: u32, rgba: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        (width, height, hot_x, hot_y).hash(&mut hasher);
        rgba.hash(&mut hasher);
        hasher.finish()
    }

    /// winit physical key → Linux `KEY_*` code.
    fn keycode_to_evdev(code: KeyCode) -> Option<u16> {
        use KeyCode::*;
        Some(match code {
            Escape => KEY_ESC,
            Digit1 => KEY_1,
            Digit2 => KEY_2,
            Digit3 => KEY_3,
            Digit4 => KEY_4,
            Digit5 => KEY_5,
            Digit6 => KEY_6,
            Digit7 => KEY_7,
            Digit8 => KEY_8,
            Digit9 => KEY_9,
            Digit0 => KEY_0,
            Minus => KEY_MINUS,
            Equal => KEY_EQUAL,
            Backspace => KEY_BACKSPACE,
            Tab => KEY_TAB,
            KeyQ => KEY_Q,
            KeyW => KEY_W,
            KeyE => KEY_E,
            KeyR => KEY_R,
            KeyT => KEY_T,
            KeyY => KEY_Y,
            KeyU => KEY_U,
            KeyI => KEY_I,
            KeyO => KEY_O,
            KeyP => KEY_P,
            BracketLeft => KEY_LEFTBRACE,
            BracketRight => KEY_RIGHTBRACE,
            Enter => KEY_ENTER,
            ControlLeft => KEY_LEFTCTRL,
            KeyA => KEY_A,
            KeyS => KEY_S,
            KeyD => KEY_D,
            KeyF => KEY_F,
            KeyG => KEY_G,
            KeyH => KEY_H,
            KeyJ => KEY_J,
            KeyK => KEY_K,
            KeyL => KEY_L,
            Semicolon => KEY_SEMICOLON,
            Quote => KEY_APOSTROPHE,
            Backquote => KEY_GRAVE,
            ShiftLeft => KEY_LEFTSHIFT,
            Backslash => KEY_BACKSLASH,
            KeyZ => KEY_Z,
            KeyX => KEY_X,
            KeyC => KEY_C,
            KeyV => KEY_V,
            KeyB => KEY_B,
            KeyN => KEY_N,
            KeyM => KEY_M,
            Comma => KEY_COMMA,
            Period => KEY_DOT,
            Slash => KEY_SLASH,
            ShiftRight => KEY_RIGHTSHIFT,
            NumpadMultiply => KEY_KPASTERISK,
            AltLeft => KEY_LEFTALT,
            Space => KEY_SPACE,
            CapsLock => KEY_CAPSLOCK,
            F1 => KEY_F1,
            F2 => KEY_F2,
            F3 => KEY_F3,
            F4 => KEY_F4,
            F5 => KEY_F5,
            F6 => KEY_F6,
            F7 => KEY_F7,
            F8 => KEY_F8,
            F9 => KEY_F9,
            F10 => KEY_F10,
            NumLock => KEY_NUMLOCK,
            ScrollLock => KEY_SCROLLLOCK,
            Numpad7 => KEY_KP7,
            Numpad8 => KEY_KP8,
            Numpad9 => KEY_KP9,
            NumpadSubtract => KEY_KPMINUS,
            Numpad4 => KEY_KP4,
            Numpad5 => KEY_KP5,
            Numpad6 => KEY_KP6,
            NumpadAdd => KEY_KPPLUS,
            Numpad1 => KEY_KP1,
            Numpad2 => KEY_KP2,
            Numpad3 => KEY_KP3,
            Numpad0 => KEY_KP0,
            NumpadDecimal => KEY_KPDOT,
            IntlBackslash => KEY_102ND,
            F11 => KEY_F11,
            F12 => KEY_F12,
            NumpadEnter => KEY_KPENTER,
            ControlRight => KEY_RIGHTCTRL,
            NumpadDivide => KEY_KPSLASH,
            PrintScreen => KEY_SYSRQ,
            AltRight => KEY_RIGHTALT,
            Home => KEY_HOME,
            ArrowUp => KEY_UP,
            PageUp => KEY_PAGEUP,
            ArrowLeft => KEY_LEFT,
            ArrowRight => KEY_RIGHT,
            End => KEY_END,
            ArrowDown => KEY_DOWN,
            PageDown => KEY_PAGEDOWN,
            Insert => KEY_INSERT,
            Delete => KEY_DELETE,
            AudioVolumeMute => KEY_MUTE,
            AudioVolumeDown => KEY_VOLUMEDOWN,
            AudioVolumeUp => KEY_VOLUMEUP,
            NumpadEqual => KEY_KPEQUAL,
            Pause => KEY_PAUSE,
            SuperLeft => KEY_LEFTMETA,
            SuperRight => KEY_RIGHTMETA,
            ContextMenu => KEY_COMPOSE,
            _ => return None,
        })
    }

    //--------------------------------------------------------------------------------------------------
    // Tests
    //--------------------------------------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A `w` x `h` BGRX frame whose pixels encode their own index.
        fn frame(w: usize, h: usize) -> Vec<u8> {
            (0..w * h)
                .flat_map(|i| [i as u8, (i >> 8) as u8, 0, 0xff])
                .collect()
        }

        /// The `0RGB` pixel [`frame`] puts at index `i` — note the BGRX `x`
        /// byte is dropped, which is what softbuffer's `NoneSkipFirst` ignores.
        fn pixel(i: usize) -> u32 {
            u32::from_le_bytes([i as u8, (i >> 8) as u8, 0, 0])
        }

        #[test]
        fn integer_scale_duplicates_pixels_and_rows() {
            let (sw, sh) = (3usize, 2usize);
            let (dw, dh) = (sw * 2, sh * 2);
            let mut dst = vec![u32::MAX; dw * dh];
            Scaler::default().scale_into(&frame(sw, sh), (sw, sh), &mut dst, (dw, dh));
            let want: Vec<u32> = [
                0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5,
            ]
            .iter()
            .map(|&i| pixel(i))
            .collect();
            assert_eq!(dst, want);
        }

        #[test]
        fn non_integer_scale_matches_nearest_neighbour() {
            let (sw, sh) = (4usize, 3usize);
            let (dw, dh) = (7usize, 5usize);
            let src = frame(sw, sh);
            let mut dst = vec![u32::MAX; dw * dh];
            let mut scaler = Scaler::default();
            scaler.scale_into(&src, (sw, sh), &mut dst, (dw, dh));
            for y in 0..dh {
                for x in 0..dw {
                    let want = pixel((y * sh / dh) * sw + x * sw / dw);
                    assert_eq!(dst[y * dw + x], want, "at {x},{y}");
                }
            }
            // The table is cached and the second pass reuses it unchanged.
            assert_eq!(scaler.built_for, (sw, dw));
            let mut again = vec![u32::MAX; dw * dh];
            scaler.scale_into(&src, (sw, sh), &mut again, (dw, dh));
            assert_eq!(again, dst);
        }

        #[test]
        fn scale_into_ignores_buffers_that_are_too_small() {
            let mut dst = vec![0u32; 3];
            Scaler::default().scale_into(&frame(2, 2), (2, 2), &mut dst, (2, 2));
            assert_eq!(dst, vec![0, 0, 0]);
            let mut dst = vec![0u32; 4];
            Scaler::default().scale_into(&frame(2, 1), (2, 2), &mut dst, (2, 2));
            assert_eq!(dst, vec![0, 0, 0, 0]);
        }
    }
}
