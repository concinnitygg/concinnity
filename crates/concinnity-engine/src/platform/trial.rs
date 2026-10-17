//! The graphics API chosen in the settings menu, tried on a launch and kept
//! only once that launch draws. A launch that dies before its first frame
//! leaves the trial marked, and the next launch starts on the native API.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use concinnity_core::platform::Platform;
use concinnity_core::settings::GraphicsApiTrial;

static CONFIRM_PENDING: AtomicBool = AtomicBool::new(false);
static REVERTED: OnceLock<Platform> = OnceLock::new();

// What a launch does with the persisted choice before rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Launch {
    pub(super) request: Option<Platform>,
    pub(super) step: Option<Step>,
}

// The settings write a launch makes before it renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    // First launch on a new choice: mark it in progress, to be cleared once a
    // frame is drawn.
    MarkTrying,
    // The last launch on the choice never drew a frame: drop the choice.
    Revert(Platform),
}

// How a launch treats `choice` given where its trial stands.
pub(super) fn plan(choice: Option<Platform>, trial: Option<GraphicsApiTrial>) -> Launch {
    match (choice, trial) {
        (Some(platform), Some(GraphicsApiTrial::Trying)) => Launch {
            request: None,
            step: Some(Step::Revert(platform)),
        },
        (Some(platform), Some(GraphicsApiTrial::Untried)) => Launch {
            request: Some(platform),
            step: Some(Step::MarkTrying),
        },
        (choice, _) => Launch {
            request: choice,
            step: None,
        },
    }
}

// Apply `step` to the persisted graphics settings, recording what the frame
// path and the settings menu need to know about it.
pub(super) fn apply(step: Step, graphics: &mut concinnity_core::settings::UserGraphicsSettings) {
    match step {
        Step::MarkTrying => {
            graphics.graphics_api_trial = Some(GraphicsApiTrial::Trying);
            CONFIRM_PENDING.store(true, Ordering::Relaxed);
        }
        Step::Revert(platform) => {
            tracing::warn!(
                "the last launch on graphics API {} never drew a frame; starting on the \
                 native API instead",
                platform.key()
            );
            graphics.graphics_api = None;
            graphics.graphics_api_trial = None;
            let _ = REVERTED.set(platform);
        }
    }
}

/// Whether this launch is trying a newly chosen graphics API that has not yet
/// drawn, taken once: the caller clears the persisted trial after a frame.
pub(crate) fn take_confirmation() -> bool {
    CONFIRM_PENDING.swap(false, Ordering::Relaxed)
}

/// The graphics API this launch gave up on because the last launch on it
/// never drew a frame.
pub(crate) fn reverted() -> Option<Platform> {
    REVERTED.get().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_or_absent_choice_is_requested_as_is() {
        for choice in [None, Some(Platform::Vulkan)] {
            assert_eq!(
                plan(choice, None),
                Launch {
                    request: choice,
                    step: None
                }
            );
        }
    }

    #[test]
    fn a_new_choice_is_tried_and_marked() {
        assert_eq!(
            plan(Some(Platform::Vulkan), Some(GraphicsApiTrial::Untried)),
            Launch {
                request: Some(Platform::Vulkan),
                step: Some(Step::MarkTrying)
            }
        );
    }

    #[test]
    fn a_choice_left_trying_reverts_to_native() {
        assert_eq!(
            plan(Some(Platform::Vulkan), Some(GraphicsApiTrial::Trying)),
            Launch {
                request: None,
                step: Some(Step::Revert(Platform::Vulkan))
            }
        );
    }

    #[test]
    fn a_revert_clears_the_choice_and_its_trial() {
        let mut graphics = concinnity_core::settings::UserGraphicsSettings {
            graphics_api: Some(Platform::Vulkan),
            graphics_api_trial: Some(GraphicsApiTrial::Trying),
            ..Default::default()
        };
        apply(Step::Revert(Platform::Vulkan), &mut graphics);
        assert_eq!(graphics.graphics_api, None);
        assert_eq!(graphics.graphics_api_trial, None);
        assert_eq!(reverted(), Some(Platform::Vulkan));
    }

    #[test]
    fn marking_a_trial_arms_one_confirmation() {
        let mut graphics = concinnity_core::settings::UserGraphicsSettings::default();
        apply(Step::MarkTrying, &mut graphics);
        assert_eq!(graphics.graphics_api_trial, Some(GraphicsApiTrial::Trying));
        assert!(take_confirmation());
        assert!(!take_confirmation());
    }
}
