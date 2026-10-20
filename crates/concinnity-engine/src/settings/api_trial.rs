//! The graphics API chosen in the settings menu, tried on a launch and kept
//! only once that launch draws. A launch that dies before its first frame
//! leaves the trial marked, and the next launch starts on the native API.

use crate::platform::{GraphicsApiStatus, Platform};
use crate::settings::{GraphicsApiTrial, UserGraphicsSettings};

/// What a launch does with the persisted choice before rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    /// The API to ask for, or `None` for the native one.
    pub request: Option<Platform>,
    /// The settings write to make first, if any.
    pub step: Option<Step>,
}

/// The settings write a launch makes before it renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// First launch on a new choice: mark it in progress, to be cleared once a
    /// frame is drawn.
    MarkTrying,
    /// The last launch on the choice never drew a frame: drop the choice.
    Revert(Platform),
}

impl Step {
    /// The API this step gave up on, for a revert.
    pub fn reverted(self) -> Option<Platform> {
        match self {
            Self::Revert(platform) => Some(platform),
            Self::MarkTrying => None,
        }
    }
}

/// How a launch treats `choice` given where its trial stands.
pub fn plan(choice: Option<Platform>, trial: Option<GraphicsApiTrial>) -> Launch {
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

/// Apply `step` to the persisted graphics settings.
pub fn apply(step: Step, graphics: &mut UserGraphicsSettings) {
    match step {
        Step::MarkTrying => {
            graphics.graphics_api_trial = Some(GraphicsApiTrial::Trying);
        }
        Step::Revert(platform) => {
            tracing::warn!(
                "the last launch on graphics API {} never drew a frame; starting on the \
                 native API instead",
                platform.key()
            );
            graphics.graphics_api = None;
            graphics.graphics_api_trial = None;
        }
    }
}

impl GraphicsApiStatus {
    /// The status of a launch rendering with `current` that took `step` on
    /// the persisted choice, if any.
    pub fn from_launch(current: Platform, step: Option<Step>) -> Self {
        Self {
            current,
            reverted: step.and_then(Step::reverted),
            confirm_pending: step == Some(Step::MarkTrying),
        }
    }
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
        let mut graphics = UserGraphicsSettings {
            graphics_api: Some(Platform::Vulkan),
            graphics_api_trial: Some(GraphicsApiTrial::Trying),
            ..Default::default()
        };
        apply(Step::Revert(Platform::Vulkan), &mut graphics);
        assert_eq!(graphics.graphics_api, None);
        assert_eq!(graphics.graphics_api_trial, None);
        assert_eq!(
            Step::Revert(Platform::Vulkan).reverted(),
            Some(Platform::Vulkan)
        );
    }

    #[test]
    fn marking_a_trial_marks_it_trying_and_reverts_nothing() {
        let mut graphics = UserGraphicsSettings::default();
        apply(Step::MarkTrying, &mut graphics);
        assert_eq!(graphics.graphics_api_trial, Some(GraphicsApiTrial::Trying));
        assert_eq!(Step::MarkTrying.reverted(), None);
    }

    #[test]
    fn a_launch_status_reads_its_step() {
        let trying = GraphicsApiStatus::from_launch(Platform::Vulkan, Some(Step::MarkTrying));
        assert!(trying.confirm_pending && trying.reverted.is_none());
        let reverted =
            GraphicsApiStatus::from_launch(Platform::Metal, Some(Step::Revert(Platform::Vulkan)));
        assert_eq!(reverted.reverted, Some(Platform::Vulkan));
        assert!(!reverted.confirm_pending);
        let settled = GraphicsApiStatus::from_launch(Platform::Metal, None);
        assert!(!settled.confirm_pending && settled.reverted.is_none());
    }
}
