// The Background-role thread that turns read-back frames into PNG files, so
// decoding, compressing and writing never run on the thread that presents.

use std::sync::{Arc, Condvar, Mutex, mpsc};

use concinnity_core::render::error::RenderResult;
use concinnity_host::thread::{ThreadRole, set_current_thread_role};

use super::image::CaptureImage;
use super::readback::ReadbackLease;
use crate::png_encode::encode_png;

// Captures recorded but not yet written, so the end of a run can wait them out.
#[derive(Default)]
struct Outstanding {
    count: Mutex<usize>,
    drained: Condvar,
}

impl Outstanding {
    fn wait(&self) {
        let mut count = self
            .count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *count > 0 {
            count = self
                .drained
                .wait(count)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

// One outstanding capture. Dropping it, written or abandoned, is what lets a
// waiting flush go on.
struct Ticket(Arc<Outstanding>);

impl Ticket {
    fn new(outstanding: &Arc<Outstanding>) -> Self {
        *outstanding
            .count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        Self(Arc::clone(outstanding))
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut count = self
            .0
            .count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.0.drained.notify_all();
        }
    }
}

/// A capture whose copy is recorded into a frame that has not yet finished on
/// the GPU.
pub(crate) struct PendingCapture {
    path: String,
    image: CaptureImage,
    lease: ReadbackLease,
    _ticket: Ticket,
}

impl PendingCapture {
    /// Give up on the capture, saying why.
    pub(crate) fn abandon(self, why: &str) {
        tracing::warn!("frame capture {} failed: {why}", self.path);
    }
}

/// A capture whose frame has retired, ready to decode and write.
pub(crate) struct RetiredCapture(PendingCapture);

impl RetiredCapture {
    /// # Safety
    ///
    /// The GPU work that copies the frame into `pending`'s buffer must have
    /// completed.
    pub(crate) unsafe fn new(pending: PendingCapture) -> Self {
        Self(pending)
    }
}

// Writes opaque RGBA8 of the given size to a path.
type Encode = fn(&str, u32, u32, &[u8]) -> RenderResult<()>;

/// Where retired captures go to be written. Cloned into whatever learns that a
/// frame retired, such as a Metal completion handler.
#[derive(Clone)]
pub(crate) struct CaptureSender(mpsc::Sender<RetiredCapture>);

impl CaptureSender {
    pub(crate) fn send(&self, capture: RetiredCapture) {
        if let Err(mpsc::SendError(capture)) = self.0.send(capture) {
            capture.0.abandon("the capture writer has stopped");
        }
    }
}

struct Running {
    sender: CaptureSender,
    thread: std::thread::JoinHandle<()>,
}

/// The capture writer. Its thread starts on the first capture and, on drop,
/// writes every capture already sent before it is joined.
pub(crate) struct CaptureWriter {
    outstanding: Arc<Outstanding>,
    encode: Encode,
    running: Option<Running>,
}

impl Default for CaptureWriter {
    fn default() -> Self {
        Self::with_encode(encode_png)
    }
}

impl CaptureWriter {
    fn with_encode(encode: Encode) -> Self {
        Self {
            outstanding: Arc::default(),
            encode,
            running: None,
        }
    }

    /// Track a capture whose copy is being recorded into the current frame.
    pub(crate) fn pending(
        &self,
        path: String,
        image: CaptureImage,
        lease: ReadbackLease,
    ) -> PendingCapture {
        PendingCapture {
            path,
            image,
            lease,
            _ticket: Ticket::new(&self.outstanding),
        }
    }

    /// The sender retired captures are written through, starting the thread on
    /// first use.
    pub(crate) fn sender(&mut self) -> CaptureSender {
        if self.running.is_none() {
            self.running = spawn(self.encode);
        }
        match &self.running {
            Some(running) => running.sender.clone(),
            // The thread could not start; a sender to nowhere abandons each
            // capture with a warning.
            None => CaptureSender(mpsc::channel().0),
        }
    }

    /// Block until every capture tracked so far is written or abandoned.
    pub(crate) fn flush(&self) {
        self.outstanding.wait();
    }
}

impl Drop for CaptureWriter {
    fn drop(&mut self) {
        if let Some(Running { sender, thread }) = self.running.take() {
            drop(sender);
            if thread.join().is_err() {
                tracing::warn!("frame capture: the writer thread panicked");
            }
        }
    }
}

fn spawn(encode: Encode) -> Option<Running> {
    let (tx, rx) = mpsc::channel::<RetiredCapture>();
    let thread = std::thread::Builder::new()
        .name("frame-capture".to_string())
        .spawn(move || {
            set_current_thread_role(ThreadRole::Background);
            for RetiredCapture(capture) in rx {
                write(capture, encode);
            }
        });
    match thread {
        Ok(thread) => Some(Running {
            sender: CaptureSender(tx),
            thread,
        }),
        Err(e) => {
            tracing::warn!("frame capture writer did not start: {e}");
            None
        }
    }
}

// Decode the capture, hand its buffer back, then compress and write the PNG.
fn write(capture: PendingCapture, encode: Encode) {
    let PendingCapture {
        path,
        image,
        lease,
        _ticket,
    } = capture;
    // SAFETY: a `RetiredCapture` is only built once its copy completed.
    let rgba = image.to_rgba8(unsafe { lease.bytes() });
    drop(lease);
    if let Some(dir) = std::path::Path::new(&path).parent()
        && !dir.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        tracing::warn!("frame capture {path}: cannot create {}: {e}", dir.display());
    }
    match rgba.and_then(|rgba| encode(&path, image.width, image.height, &rgba)) {
        Ok(()) => tracing::info!("frame capture saved: {path}"),
        Err(e) => tracing::warn!("frame capture {path} failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_capture::readback::{Mapped, ReadbackPool};
    use concinnity_core::gfx::image_decode::PixelLayout;
    use concinnity_core::render::error::RenderError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Host bytes standing in for one mapped read-back buffer.
    fn pool_over(bytes: &'static [u8]) -> ReadbackPool<()> {
        let mut pool = ReadbackPool::default();
        pool.reserve(bytes.len(), |_| {
            Ok(Mapped {
                buffer: (),
                mapped: bytes.as_ptr(),
            })
        })
        .unwrap();
        pool
    }

    fn pending(writer: &CaptureWriter, pool: &mut ReadbackPool<()>, path: &str) -> PendingCapture {
        let image = CaptureImage::tight(2, 1, PixelLayout::Bgra8);
        // A second buffer, when the reserved one is still lent, reads zeroes.
        static SPARE: [u8; 8] = [0; 8];
        let ((), lease) = pool
            .lend(image.byte_len(), |_| {
                Ok(Mapped {
                    buffer: (),
                    mapped: SPARE.as_ptr(),
                })
            })
            .unwrap();
        writer.pending(path.to_string(), image, lease)
    }

    #[test]
    fn a_retired_capture_is_written_as_a_png_by_the_time_flush_returns() {
        static BYTES: [u8; 8] = [10, 20, 30, 0, 40, 50, 60, 0];
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("shot.png");
        let mut pool = pool_over(&BYTES);
        let mut writer = CaptureWriter::default();
        let capture = pending(&writer, &mut pool, path.to_str().unwrap());
        // SAFETY: nothing writes the host stand-in's bytes.
        writer
            .sender()
            .send(unsafe { RetiredCapture::new(capture) });
        writer.flush();

        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut pixels).unwrap();
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(
            &pixels[..frame.buffer_size()],
            [30, 20, 10, 255, 60, 50, 40, 255]
        );
        drop(writer);
    }

    #[test]
    fn a_capture_into_a_missing_directory_creates_it() {
        static BYTES: [u8; 8] = [10, 20, 30, 0, 40, 50, 60, 0];
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("shots").join("day").join("shot.png");
        let mut pool = pool_over(&BYTES);
        let mut writer = CaptureWriter::default();
        let capture = pending(&writer, &mut pool, path.to_str().unwrap());
        // SAFETY: nothing writes the host stand-in's bytes.
        writer
            .sender()
            .send(unsafe { RetiredCapture::new(capture) });
        writer.flush();
        assert!(path.is_file());
        drop(writer);
    }

    #[test]
    fn flush_waits_for_a_capture_still_on_the_writer_thread() {
        static BYTES: [u8; 8] = [0; 8];
        static SAVED: AtomicUsize = AtomicUsize::new(0);
        fn slow_save(_: &str, _: u32, _: u32, _: &[u8]) -> RenderResult<()> {
            std::thread::sleep(std::time::Duration::from_millis(30));
            SAVED.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        let mut pool = pool_over(&BYTES);
        let mut writer = CaptureWriter::with_encode(slow_save);
        let capture = pending(&writer, &mut pool, "slow.png");
        // SAFETY: nothing writes the host stand-in's bytes.
        writer
            .sender()
            .send(unsafe { RetiredCapture::new(capture) });
        writer.flush();
        assert_eq!(SAVED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_the_writer_finishes_captures_already_sent() {
        static BYTES: [u8; 8] = [0; 8];
        static SAVED: AtomicUsize = AtomicUsize::new(0);
        fn slow_save(_: &str, _: u32, _: u32, _: &[u8]) -> RenderResult<()> {
            std::thread::sleep(std::time::Duration::from_millis(10));
            SAVED.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        let mut pool = pool_over(&BYTES);
        let mut writer = CaptureWriter::with_encode(slow_save);
        let sender = writer.sender();
        for path in ["a.png", "b.png"] {
            let capture = pending(&writer, &mut pool, path);
            // SAFETY: nothing writes the host stand-in's bytes.
            sender.send(unsafe { RetiredCapture::new(capture) });
        }
        drop(sender);
        drop(writer);
        assert_eq!(SAVED.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn an_abandoned_capture_no_longer_holds_up_flush() {
        static BYTES: [u8; 8] = [0; 8];
        let mut pool = pool_over(&BYTES);
        let writer = CaptureWriter::default();
        pending(&writer, &mut pool, "lost.png").abandon("the frame was not submitted");
        writer.flush();
    }

    #[test]
    fn a_failed_save_still_returns_the_buffer_and_releases_flush() {
        static BYTES: [u8; 8] = [0; 8];
        fn failing_save(_: &str, _: u32, _: u32, _: &[u8]) -> RenderResult<()> {
            Err(RenderError::Other("disk full".into()))
        }
        let mut pool = pool_over(&BYTES);
        let mut writer = CaptureWriter::with_encode(failing_save);
        let capture = pending(&writer, &mut pool, "full.png");
        // SAFETY: nothing writes the host stand-in's bytes.
        writer
            .sender()
            .send(unsafe { RetiredCapture::new(capture) });
        writer.flush();
        // The buffer came back, so lending it again allocates nothing.
        let image = CaptureImage::tight(2, 1, PixelLayout::Bgra8);
        assert!(
            pool.lend(image.byte_len(), |_| unreachable!("returned"))
                .is_ok()
        );
    }
}
