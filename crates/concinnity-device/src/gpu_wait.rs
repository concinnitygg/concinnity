// src/gpu_wait.rs
//
// The CPU's blocked-on-GPU time within one frame's draw call. Every backend
// blocks in the same two places: the frames-in-flight fence or semaphore that
// paces the CPU against GPU retirement, and the swapchain / drawable acquire.
// Both sit inside the per-frame `draw_frame` the engine times its graphics
// system around, so without a separate reading a GPU-bound frame reports as a
// CPU-bound one -- the graphics system's span tracks the GPU frame time because
// it is mostly waiting for it.
//
// The accumulator is a frame-local value the backend hands to
// `RenderStats::gpu_wait_us`, so the shape is identical across Metal, Vulkan
// and DirectX and the field means the same thing on each.

/// Blocked-on-GPU microseconds accumulated across one frame's waits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuWait {
    micros: u32,
}

impl GpuWait {
    /// A frame that has not waited yet.
    pub(crate) const fn none() -> Self {
        Self { micros: 0 }
    }

    /// Run `f`, adding the wall time it blocked for to the running total.
    /// Saturates rather than wrapping: a pathological stall reports the
    /// ceiling, never a small number.
    pub(crate) fn measure<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let start = std::time::Instant::now();
        let value = f();
        let elapsed = start.elapsed().as_micros().min(u32::MAX as u128) as u32;
        self.micros = self.micros.saturating_add(elapsed);
        value
    }

    /// Total blocked microseconds this frame.
    pub(crate) const fn micros(self) -> u32 {
        self.micros
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_that_never_waits_reports_nothing() {
        assert_eq!(GpuWait::none().micros(), 0);
    }

    #[test]
    fn measure_returns_the_inner_value_and_accumulates() {
        let mut wait = GpuWait::none();
        let a = wait.measure(|| 7u32);
        let b = wait.measure(|| {
            std::thread::sleep(std::time::Duration::from_millis(2));
            "slot"
        });
        assert_eq!((a, b), (7, "slot"));
        // The sleep is the only bound worth asserting: a 2 ms block cannot
        // report as a sub-millisecond one.
        assert!(wait.micros() >= 1_000, "{} us", wait.micros());
    }

    #[test]
    fn the_total_saturates_instead_of_wrapping() {
        let mut wait = GpuWait {
            micros: u32::MAX - 1,
        };
        wait.measure(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        assert_eq!(wait.micros(), u32::MAX);
    }
}
