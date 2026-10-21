//! Frame captures that cost the presenting thread nothing it can see. A frame
//! asked to be captured records a copy of its presented image into a reused
//! host-visible buffer inside its own submission; once that frame retires on
//! the GPU, a background thread decodes the bytes and writes the PNG. The
//! backends supply the copy and say when a frame retired; everything else is
//! here, the same on every backend.

mod image;
mod readback;
mod writer;

pub(crate) use image::{CaptureImage, save_png};
pub(crate) use readback::Mapped;
pub(crate) use writer::{PendingCapture, RetiredCapture};

use concinnity_engine::render::backend::error::RenderResult;

use readback::ReadbackPool;
use writer::CaptureWriter;

/// One backend's frame captures. `B` is its read-back buffer handle and `T`
/// what it tells a retired frame by (a fence value, a frame slot).
pub(crate) struct FrameCaptures<B, T> {
    // Whether this run can capture at all, so a read-back buffer is kept ready.
    enabled: bool,
    // The capture asked of the frame being drawn, until its copy is recorded.
    armed: Option<String>,
    // Declared ahead of the writer and the pool so a teardown abandons these
    // first, then lets the writer finish, and only then frees the buffers.
    in_flight: Vec<(T, PendingCapture)>,
    writer: CaptureWriter,
    pool: ReadbackPool<B>,
}

impl<B, T> FrameCaptures<B, T> {
    /// Captures for a run that `enabled` frame capture, or did not. A run
    /// that did starts its writer thread now, so no capture starts one.
    pub(crate) fn new(enabled: bool) -> Self {
        let mut writer = CaptureWriter::default();
        if enabled {
            writer.start();
        }
        Self {
            enabled,
            armed: None,
            in_flight: Vec::new(),
            writer,
            pool: ReadbackPool::default(),
        }
    }
}

impl<B: Clone, T> FrameCaptures<B, T> {
    /// Whether this run enabled frame capture.
    #[cfg(backend_metal)]
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Start a frame, asked to capture to `path` or not. A capture the last
    /// frame was asked for but never recorded is reported missed.
    pub(crate) fn arm(&mut self, path: Option<&str>) {
        if let Some(missed) = self.armed.take() {
            tracing::warn!("frame capture {missed} skipped: its frame presented nothing");
        }
        self.armed = path.map(str::to_string);
    }

    /// The capture this frame should record, taken so it is recorded once.
    pub(crate) fn take_armed(&mut self) -> Option<String> {
        self.armed.take()
    }

    /// Keep a `len`-byte read-back buffer ready, when this run captures, so a
    /// capture never allocates on the frame it is taken. A failure resurfaces
    /// on the capture itself, which allocates again.
    #[expect(
        clippy::let_underscore_must_use,
        reason = "a failure resurfaces on the capture itself, which allocates again"
    )]
    pub(crate) fn reserve(
        &mut self,
        len: usize,
        alloc: impl FnOnce(usize) -> RenderResult<Mapped<B>>,
    ) {
        if self.enabled {
            let _ = self.pool.reserve(len, alloc);
        }
    }

    /// Lend a buffer for `image` and track the capture to `path` its copy is
    /// about to be recorded for.
    pub(crate) fn begin(
        &mut self,
        path: String,
        image: CaptureImage,
        alloc: impl FnOnce(usize) -> RenderResult<Mapped<B>>,
    ) -> RenderResult<(B, PendingCapture)> {
        let (buffer, lease) = self.pool.lend(image.byte_len(), alloc)?;
        Ok((buffer, self.writer.pending(path, image, lease)))
    }

    /// Hold a submitted capture until the frame `token` names retires.
    #[cfg(any(backend_dx, backend_vk, test))]
    pub(crate) fn park(&mut self, token: T, capture: PendingCapture) {
        self.in_flight.push((token, capture));
    }

    /// Hand every capture whose frame `retired` says has finished to the
    /// writer.
    ///
    /// # Safety
    ///
    /// `retired` may only answer true for a token whose frame's GPU work has
    /// completed.
    pub(crate) unsafe fn retire(&mut self, retired: impl Fn(&T) -> bool) {
        if self.in_flight.is_empty() {
            return;
        }
        let sender = self.writer.sender();
        let mut index = 0;
        while index < self.in_flight.len() {
            if retired(&self.in_flight[index].0) {
                let (_, capture) = self.in_flight.remove(index);
                // SAFETY: the caller vouches this token's frame completed.
                sender.send(unsafe { RetiredCapture::new(capture) });
            } else {
                index += 1;
            }
        }
    }

    /// Where a backend that learns of retirement off this thread sends
    /// captures.
    #[cfg(backend_metal)]
    pub(crate) fn sender(&mut self) -> writer::CaptureSender {
        self.writer.sender()
    }

    /// Block until every capture recorded so far is written or abandoned.
    ///
    /// # Safety
    ///
    /// The GPU must be idle: every parked capture is handed to the writer.
    pub(crate) unsafe fn finish(&mut self) {
        if let Some(missed) = self.armed.take() {
            tracing::warn!("frame capture {missed} skipped: the run ended before its frame");
        }
        // SAFETY: the caller vouches the GPU is idle.
        unsafe { self.retire(|_| true) };
        self.writer.flush();
    }

    #[cfg(test)]
    fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::gfx::image_decode::PixelLayout;

    static BYTES: [u8; 8] = [0; 8];

    #[expect(
        clippy::unnecessary_wraps,
        reason = "stands in for the staging allocator, which can fail"
    )]
    fn host(_: usize) -> RenderResult<Mapped<()>> {
        Ok(Mapped {
            buffer: (),
            mapped: BYTES.as_ptr(),
        })
    }

    fn begin(captures: &mut FrameCaptures<(), u64>, path: &str) -> PendingCapture {
        let image = CaptureImage::tight(2, 1, PixelLayout::Rgba8);
        captures.begin(path.to_string(), image, host).unwrap().1
    }

    #[test]
    fn a_run_that_captures_starts_its_writer_up_front() {
        assert!(FrameCaptures::<(), u64>::new(true).writer.is_running());
        assert!(!FrameCaptures::<(), u64>::new(false).writer.is_running());
    }

    #[test]
    fn an_armed_capture_is_taken_once() {
        let mut captures = FrameCaptures::<(), u64>::new(true);
        captures.arm(Some("a.png"));
        assert_eq!(captures.take_armed().as_deref(), Some("a.png"));
        assert_eq!(captures.take_armed(), None);
    }

    #[test]
    fn a_capture_its_frame_never_recorded_does_not_carry_over() {
        let mut captures = FrameCaptures::<(), u64>::new(true);
        captures.arm(Some("a.png"));
        captures.arm(None);
        assert_eq!(captures.take_armed(), None);
    }

    #[test]
    fn only_captures_whose_frames_retired_leave_for_the_writer() {
        let tree = concinnity_testing::TempTree::new();
        let mut captures = FrameCaptures::<(), u64>::new(true);
        for fence in [3, 4, 5] {
            let path = tree.join(&format!("{fence}.png"));
            let capture = begin(&mut captures, path.to_str().unwrap());
            captures.park(fence, capture);
        }
        // SAFETY: the host stand-in has no GPU work.
        unsafe { captures.retire(|&fence| fence <= 4) };
        assert_eq!(captures.in_flight(), 1);
        // A flush waits on every capture still tracked, the parked one too.
        captures.in_flight.clear();
        captures.writer.flush();
        assert!(tree.join("3.png").exists());
        assert!(tree.join("4.png").exists());
        assert!(!tree.join("5.png").exists());
    }

    #[test]
    fn finishing_writes_every_parked_capture() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("last.png");
        let mut captures = FrameCaptures::<(), u64>::new(true);
        let capture = begin(&mut captures, path.to_str().unwrap());
        captures.park(9, capture);
        // SAFETY: the host stand-in has no GPU work.
        unsafe { captures.finish() };
        assert_eq!(captures.in_flight(), 0);
        assert!(path.exists());
    }

    #[test]
    fn a_capture_dropped_unsubmitted_does_not_block_finishing() {
        let mut captures = FrameCaptures::<(), u64>::new(true);
        let capture = begin(&mut captures, "never.png");
        drop(capture);
        // SAFETY: the host stand-in has no GPU work.
        unsafe { captures.finish() };
    }
}
