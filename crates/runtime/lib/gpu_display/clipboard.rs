//! Guest clipboard relay for the display server.
//!
//! The guest agent connects to [`CLIPBOARD_VSOCK_PORT`] and speaks
//! newline-delimited [`GuestClipboardMsg`]; this backend forwards each line to
//! the viewer as [`ServerMsg::Clipboard`] and queues the viewer's
//! [`ViewerMsg::Clipboard`] back to the guest. Nothing here ever blocks, and
//! nothing here runs on the gpu worker thread.
//!
//! Loop prevention is deliberately absent: the guest agent and the viewer each
//! remember what they last received and refuse to echo it back.

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use msb_krun::backends::vsock::{
    VsockConnectRequest, VsockNotifier, VsockPortBackend, VsockShutdown, VsockStreamBackend,
};

use super::Shared;
use super::protocol::{GuestClipboardMsg, ServerMsg};

/// Most bytes queued in either direction before payloads are dropped.
const MAX_QUEUE: usize = 16 * 1024 * 1024;

/// One guest connection's buffers.
#[derive(Default)]
struct ConnState {
    /// Bytes from the guest that do not yet form a complete line.
    inbound: Vec<u8>,
    /// Bytes waiting to be delivered to the guest.
    outbound: VecDeque<u8>,
    /// A newer connection replaced this one; report EOF and discard writes.
    stale: bool,
}

struct Conn {
    id: u64,
    notifier: VsockNotifier,
    state: Mutex<ConnState>,
}

impl Conn {
    fn lock(&self) -> std::sync::MutexGuard<'_, ConnState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// State shared by the port backend and every stream it hands out.
struct ClipboardState {
    /// Set once, right after `DisplayServer` leaks its `Shared`.
    shared: OnceLock<&'static Shared>,
    /// Only the newest guest connection is served, like viewers.
    conn: Mutex<Option<Arc<Conn>>>,
    /// Last selection seen from the guest, replayed to a late viewer.
    last_guest: Mutex<Option<(String, String)>>,
    next_id: AtomicU64,
}

impl ClipboardState {
    fn conn(&self) -> Option<Arc<Conn>> {
        self.conn
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(Arc::clone)
    }
}

/// Serves the guest clipboard agent on [`CLIPBOARD_VSOCK_PORT`].
///
/// [`CLIPBOARD_VSOCK_PORT`]: super::protocol::CLIPBOARD_VSOCK_PORT
pub struct ClipboardPortBackend {
    state: Arc<ClipboardState>,
}

impl ClipboardPortBackend {
    /// Create a backend with no connection and no viewer yet.
    pub fn new() -> Self {
        Self {
            state: Arc::new(ClipboardState {
                shared: OnceLock::new(),
                conn: Mutex::new(None),
                last_guest: Mutex::new(None),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Point the backend at the display server's state (once).
    pub(super) fn set_shared(&self, shared: &'static Shared) {
        let _ = self.state.shared.set(shared);
    }

    /// The last selection the guest reported, for a viewer that just attached.
    pub(super) fn last_guest(&self) -> Option<ServerMsg> {
        let guard = self
            .state
            .last_guest
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(|(mime, data)| ServerMsg::Clipboard {
            mime: mime.clone(),
            data: data.clone(),
        })
    }

    /// Queue the host's selection for the guest agent.
    pub(super) fn send_to_guest(&self, mime: String, data: String) {
        let Some(conn) = self.state.conn() else {
            tracing::debug!("gpu display: clipboard set with no guest agent connected");
            return;
        };
        let mut line = match serde_json::to_vec(&GuestClipboardMsg::Set { mime, data }) {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!(error = %e, "gpu display: cannot encode clipboard message");
                return;
            }
        };
        line.push(b'\n');
        {
            let mut state = conn.lock();
            if state.stale {
                return;
            }
            if state.outbound.len() + line.len() > MAX_QUEUE {
                tracing::warn!(
                    conn = conn.id,
                    queued = state.outbound.len(),
                    len = line.len(),
                    "gpu display: clipboard queue full, dropping host selection"
                );
                return;
            }
            state.outbound.extend(line);
        }
        // Wake libkrun so it retries the read that was blocked on an empty queue.
        if let Err(e) = conn.notifier.notify() {
            tracing::warn!(conn = conn.id, error = %e, "gpu display: clipboard notify failed");
        }
    }
}

impl Default for ClipboardPortBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VsockPortBackend for ClipboardPortBackend {
    fn connect(
        &self,
        request: VsockConnectRequest,
        notifier: VsockNotifier,
    ) -> io::Result<Box<dyn VsockStreamBackend>> {
        let id = self.state.next_id.fetch_add(1, Ordering::Relaxed);
        let conn = Arc::new(Conn {
            id,
            notifier,
            state: Mutex::new(ConnState::default()),
        });
        let previous = {
            let mut guard = self.state.conn.lock().unwrap_or_else(|e| e.into_inner());
            guard.replace(Arc::clone(&conn))
        };
        // libkrun still owns the replaced stream; retire it so it sees EOF.
        if let Some(previous) = previous {
            previous.lock().stale = true;
            let _ = previous.notifier.notify();
            tracing::info!(conn = previous.id, "gpu display: clipboard agent replaced");
        }
        tracing::info!(
            conn = id,
            guest_port = request.guest_port,
            "gpu display: clipboard agent connected"
        );
        Ok(Box::new(ClipboardStream {
            state: Arc::clone(&self.state),
            conn,
        }))
    }
}

/// One guest connection's byte stream.
struct ClipboardStream {
    state: Arc<ClipboardState>,
    conn: Arc<Conn>,
}

impl ClipboardStream {
    /// Forward one complete line from the guest to the viewer.
    ///
    /// Called with no lock held: `Shared::send` takes the viewer lock, which
    /// `Shared::attach` already holds while reading `last_guest`.
    fn deliver(&self, line: &[u8]) {
        let msg = match serde_json::from_slice::<GuestClipboardMsg>(line) {
            Ok(msg) => msg,
            Err(e) => {
                tracing::debug!(error = %e, "gpu display: bad clipboard message from guest");
                return;
            }
        };
        let GuestClipboardMsg::Set { mime, data } = msg;
        {
            let mut last = self
                .state
                .last_guest
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *last = Some((mime.clone(), data.clone()));
        }
        // No viewer attached is normal; `last_guest` replays on the next attach.
        if let Some(shared) = self.state.shared.get() {
            shared.send(ServerMsg::Clipboard { mime, data });
        }
    }

    /// Drop this connection if it is still the current one.
    fn retire(&self) {
        let mut guard = self.state.conn.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_some_and(|c| c.id == self.conn.id) {
            *guard = None;
        }
        drop(guard);
        self.conn.lock().stale = true;
    }
}

impl VsockStreamBackend for ClipboardStream {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut state = self.conn.lock();
        if state.stale {
            // A newer agent took over: EOF, not an empty read.
            return Ok(0);
        }
        if state.outbound.is_empty() || buf.is_empty() {
            // `Ok(0)` here would close the connection.
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let n = state.outbound.len().min(buf.len());
        for (slot, byte) in buf.iter_mut().zip(state.outbound.drain(..n)) {
            *slot = byte;
        }
        Ok(n)
    }

    fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let lines = {
            let mut state = self.conn.lock();
            if state.stale {
                // Accept and discard so the guest is not wedged mid-shutdown.
                return Ok(buf.len());
            }
            if state.inbound.len() + buf.len() > MAX_QUEUE {
                tracing::warn!(
                    conn = self.conn.id,
                    buffered = state.inbound.len(),
                    "gpu display: clipboard line too long, dropping"
                );
                state.inbound.clear();
                return Ok(buf.len());
            }
            state.inbound.extend_from_slice(buf);
            let mut lines = Vec::new();
            while let Some(end) = state.inbound.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = state.inbound.drain(..=end).collect();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            lines
        };
        for line in lines {
            self.deliver(&line);
        }
        Ok(buf.len())
    }

    fn shutdown(&self, _how: VsockShutdown) -> io::Result<()> {
        self.retire();
        tracing::info!(conn = self.conn.id, "gpu display: clipboard agent gone");
        Ok(())
    }
}

impl Drop for ClipboardStream {
    fn drop(&mut self) {
        self.retire();
    }
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> VsockConnectRequest {
        VsockConnectRequest {
            guest_cid: 3,
            guest_port: 4000,
            host_port: super::super::protocol::CLIPBOARD_VSOCK_PORT,
        }
    }

    fn backend() -> ClipboardPortBackend {
        ClipboardPortBackend::new()
    }

    fn set_line(text: &str) -> Vec<u8> {
        let mut line = serde_json::to_vec(&GuestClipboardMsg::Set {
            mime: super::super::protocol::TEXT_MIME.to_string(),
            data: text.to_string(),
        })
        .unwrap();
        line.push(b'\n');
        line
    }

    fn last_text(backend: &ClipboardPortBackend) -> Option<String> {
        match backend.last_guest() {
            Some(ServerMsg::Clipboard { data, .. }) => Some(data),
            _ => None,
        }
    }

    #[test]
    fn empty_queue_reads_would_block_rather_than_eof() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        let mut buf = [0u8; 64];
        let err = stream.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    }

    #[test]
    fn host_selection_is_queued_as_one_json_line() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        backend.send_to_guest(
            super::super::protocol::TEXT_MIME.to_string(),
            "aGVsbG8=".to_string(),
        );
        let mut buf = [0u8; 256];
        let n = stream.read(&mut buf).unwrap();
        let line = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(line.ends_with('\n'), "{line:?}");
        let msg: GuestClipboardMsg = serde_json::from_str(line.trim_end()).unwrap();
        let GuestClipboardMsg::Set { mime, data } = msg;
        assert_eq!(mime, super::super::protocol::TEXT_MIME);
        assert_eq!(data, "aGVsbG8=");
        // Drained: the next read blocks again.
        assert_eq!(
            stream.read(&mut buf).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn guest_writes_are_split_on_newlines_across_chunks() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        let first = set_line("b25l");
        let second = set_line("dHdv");
        // A single write carrying two lines.
        let mut both = first.clone();
        both.extend_from_slice(&second);
        assert_eq!(stream.write(&both).unwrap(), both.len());
        assert_eq!(last_text(&backend).as_deref(), Some("dHdv"));

        // A line split across two writes is only delivered once complete.
        let third = set_line("dGhyZWU=");
        let (head, tail) = third.split_at(5);
        assert_eq!(stream.write(head).unwrap(), head.len());
        assert_eq!(last_text(&backend).as_deref(), Some("dHdv"));
        assert_eq!(stream.write(tail).unwrap(), tail.len());
        assert_eq!(last_text(&backend).as_deref(), Some("dGhyZWU="));
    }

    #[test]
    fn malformed_and_blank_lines_are_skipped() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        stream
            .write(b"\n{ not json }\n{\"t\":\"future\"}\n")
            .unwrap();
        assert!(backend.last_guest().is_none());
        stream.write(&set_line("b2s=")).unwrap();
        assert_eq!(last_text(&backend).as_deref(), Some("b2s="));
    }

    #[test]
    fn a_new_connection_replaces_and_retires_the_old_one() {
        let backend = backend();
        let first = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        let second = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();

        // The retired stream reports EOF and swallows further guest bytes.
        let mut buf = [0u8; 64];
        assert_eq!(first.read(&mut buf).unwrap(), 0);
        assert_eq!(
            first.write(&set_line("b2xk")).unwrap(),
            set_line("b2xk").len()
        );
        assert!(
            backend.last_guest().is_none(),
            "stale writes must be dropped"
        );

        // Host selections now go to the new connection only.
        backend.send_to_guest(
            super::super::protocol::TEXT_MIME.to_string(),
            "bmV3".to_string(),
        );
        assert_eq!(first.read(&mut buf).unwrap(), 0);
        let n = second.read(&mut buf).unwrap();
        assert!(std::str::from_utf8(&buf[..n]).unwrap().contains("bmV3"));
    }

    #[test]
    fn an_oversized_guest_line_is_dropped_without_growing_forever() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        let chunk = vec![b'x'; 1024 * 1024];
        // No newline anywhere: the buffer must be capped, not unbounded.
        for _ in 0..20 {
            assert_eq!(stream.write(&chunk).unwrap(), chunk.len());
        }
        assert!(backend.last_guest().is_none());
        // The connection still works once a real line arrives.
        stream.write(b"\n").unwrap();
        stream.write(&set_line("YWZ0ZXI=")).unwrap();
        assert_eq!(last_text(&backend).as_deref(), Some("YWZ0ZXI="));
    }

    #[test]
    fn the_host_queue_is_capped() {
        let backend = backend();
        let stream = backend
            .connect(request(), VsockNotifier::new().unwrap())
            .unwrap();
        let big = "a".repeat(4 * 1024 * 1024);
        for _ in 0..8 {
            backend.send_to_guest(super::super::protocol::TEXT_MIME.to_string(), big.clone());
        }
        let queued = {
            let state = stream_conn_len(&stream);
            state
        };
        assert!(queued <= MAX_QUEUE, "queue grew to {queued}");
    }

    /// Bytes currently queued for the guest on `stream`.
    fn stream_conn_len(stream: &Box<dyn VsockStreamBackend>) -> usize {
        let mut total = 0;
        let mut buf = [0u8; 64 * 1024];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(_) => break,
            }
        }
        total
    }
}
