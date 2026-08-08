// Frame-loop recovery policy for a failed draw_frame. Each RenderError class
// resolves to an action, and consecutive failures are bounded per class so no
// error can skip frames forever: any successful frame resets the streaks, and
// an exhausted bound escalates to a controlled stop naming the class.

use crate::gfx::error::RenderError;

// What the frame loop does with a failed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FrameAction {
    // Drop this frame's output and try again next tick.
    SkipFrame,
    // Controlled stop. The device still services work, so the loop may drain
    // in-flight frames before tearing down.
    Shutdown,
    // Controlled stop with the device gone: draining would block on a queue
    // that can never signal, so teardown must not wait for the GPU.
    ShutdownDeviceLost,
}

// Consecutive frames a swapchain mismatch may skip. The backend recreates the
// swapchain itself, so one skip normally suffices; two seconds of failures at
// 60 Hz means recreation is not converging.
const SWAPCHAIN_BOUND: u32 = 120;
// Consecutive frames device-memory exhaustion may skip while the pressure
// signal waits for a consumer to free memory.
const OOM_BOUND: u32 = 5;
// Consecutive frames an unclassified error may skip.
const OTHER_BOUND: u32 = 3;

#[derive(Debug, Default)]
pub(super) struct FramePolicy {
    swapchain_streak: u32,
    oom_streak: u32,
    other_streak: u32,
}

impl FramePolicy {
    // A frame drew successfully; every failure streak resets.
    pub(super) fn frame_succeeded(&mut self) {
        *self = Self::default();
    }

    pub(super) fn on_frame_error(&mut self, error: &RenderError) -> FrameAction {
        match error {
            RenderError::DeviceLost { .. } => FrameAction::ShutdownDeviceLost,
            RenderError::SwapchainOutOfDate => {
                bump(&mut self.swapchain_streak, SWAPCHAIN_BOUND, error)
            }
            RenderError::OutOfDeviceMemory(_) => bump(&mut self.oom_streak, OOM_BOUND, error),
            RenderError::ShaderCompile(_) | RenderError::Other(_) => {
                bump(&mut self.other_streak, OTHER_BOUND, error)
            }
        }
    }
}

// Count one failure against `bound`. Skips until the bound is reached, then
// escalates. Logged once per streak (at the first failure) plus once at
// escalation, so a long streak cannot spam at frame rate.
fn bump(streak: &mut u32, bound: u32, error: &RenderError) -> FrameAction {
    *streak += 1;
    if *streak >= bound {
        tracing::error!(
            "GraphicsSystem: {} consecutive frames failed, stopping: {}",
            streak,
            error
        );
        FrameAction::Shutdown
    } else {
        if *streak == 1 {
            tracing::warn!("GraphicsSystem: frame skipped: {}", error);
        }
        FrameAction::SkipFrame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::error::DeviceLostReason;

    fn device_lost() -> RenderError {
        RenderError::DeviceLost {
            reason: DeviceLostReason::Removed,
            detail: "test".to_string(),
        }
    }

    #[test]
    fn device_lost_stops_immediately_without_gpu_drain() {
        let mut policy = FramePolicy::default();
        assert_eq!(
            policy.on_frame_error(&device_lost()),
            FrameAction::ShutdownDeviceLost
        );
    }

    #[test]
    fn swapchain_mismatch_skips_then_escalates_at_bound() {
        let mut policy = FramePolicy::default();
        for _ in 0..SWAPCHAIN_BOUND - 1 {
            assert_eq!(
                policy.on_frame_error(&RenderError::SwapchainOutOfDate),
                FrameAction::SkipFrame
            );
        }
        assert_eq!(
            policy.on_frame_error(&RenderError::SwapchainOutOfDate),
            FrameAction::Shutdown
        );
    }

    #[test]
    fn oom_skips_then_escalates_at_bound() {
        let mut policy = FramePolicy::default();
        let oom = RenderError::OutOfDeviceMemory("test".to_string());
        for _ in 0..OOM_BOUND - 1 {
            assert_eq!(policy.on_frame_error(&oom), FrameAction::SkipFrame);
        }
        assert_eq!(policy.on_frame_error(&oom), FrameAction::Shutdown);
    }

    #[test]
    fn unclassified_errors_share_the_other_bound_with_shader_compile() {
        let mut policy = FramePolicy::default();
        let other = RenderError::Other("test".to_string());
        let shader = RenderError::ShaderCompile("test".to_string());
        assert_eq!(policy.on_frame_error(&other), FrameAction::SkipFrame);
        assert_eq!(policy.on_frame_error(&shader), FrameAction::SkipFrame);
        assert_eq!(policy.on_frame_error(&other), FrameAction::Shutdown);
    }

    #[test]
    fn success_resets_every_streak() {
        let mut policy = FramePolicy::default();
        let other = RenderError::Other("test".to_string());
        for _ in 0..OTHER_BOUND - 1 {
            assert_eq!(policy.on_frame_error(&other), FrameAction::SkipFrame);
        }
        policy.frame_succeeded();
        assert_eq!(policy.on_frame_error(&other), FrameAction::SkipFrame);
    }
}
