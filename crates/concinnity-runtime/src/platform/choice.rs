//! The rules of the process's graphics API choice, apart from the statics that
//! hold it: made once by an explicit select, never by a read.

use std::sync::OnceLock;

use concinnity_core::platform::Platform;

/// One process's graphics API choice.
pub(super) struct ApiChoice {
    chosen: OnceLock<Chosen>,
    // What the first read before the choice was answered.
    early_answer: OnceLock<Platform>,
}

// The chosen API and what the select that chose asked for, so a later select
// repeating the request passes quietly.
#[derive(Clone, Copy)]
struct Chosen {
    platform: Platform,
    requested: Option<Platform>,
}

impl ApiChoice {
    pub(super) const fn new() -> Self {
        Self {
            chosen: OnceLock::new(),
            early_answer: OnceLock::new(),
        }
    }

    /// The chosen API, or `unchosen` before any select. A read before the
    /// choice chooses nothing: the select that follows still decides, and the
    /// first such read is reported.
    pub(super) fn current(&self, unchosen: Platform) -> Platform {
        if let Some(chosen) = self.chosen.get() {
            return chosen.platform;
        }
        if self.early_answer.set(unchosen).is_ok() {
            tracing::warn!(
                "the graphics API was read before this process chose one; answering {} until it does",
                unchosen.key()
            );
        }
        unchosen
    }

    /// Choose by `resolve(requested)` unless an earlier select already chose,
    /// returning the API the process renders with. A later select asking for
    /// another API keeps the first answer and warns, and a choice that differs
    /// from what an earlier read was answered warns too.
    ///
    /// `resolve` runs while the choice is being made, so it must not select.
    pub(super) fn select(
        &self,
        requested: Option<Platform>,
        resolve: impl FnOnce(Option<Platform>) -> Platform,
    ) -> Platform {
        let mut chose = false;
        let chosen = *self.chosen.get_or_init(|| {
            chose = true;
            Chosen {
                platform: resolve(requested),
                requested,
            }
        });
        if chose {
            if let Some(early) = self.mismatched_early_read(chosen.platform) {
                tracing::warn!(
                    "an earlier read was answered {}; anything built from it will not match {}",
                    early.key(),
                    chosen.platform.key()
                );
            }
        } else if let Some(asked) = overruled(chosen, requested) {
            tracing::warn!(
                "graphics API {} was already chosen for this process; keeping it over {}",
                chosen.platform.key(),
                asked.key()
            );
        }
        chosen.platform
    }

    /// Whether a select has chosen.
    pub(super) fn is_chosen(&self) -> bool {
        self.chosen.get().is_some()
    }

    // What an earlier read was answered, when it is not `chosen`.
    fn mismatched_early_read(&self, chosen: Platform) -> Option<Platform> {
        self.early_answer
            .get()
            .copied()
            .filter(|&early| early != chosen)
    }
}

// The API a later select asks for in vain: another than `chosen`, and not the
// request that chose it.
fn overruled(chosen: Chosen, requested: Option<Platform>) -> Option<Platform> {
    requested.filter(|&asked| asked != chosen.platform && requested != chosen.requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(platform: Platform) -> impl FnOnce(Option<Platform>) -> Platform {
        move |_| platform
    }

    // A read before the choice answers what it is told to and leaves the
    // choice to the select that follows.
    #[test]
    fn an_early_read_does_not_choose() {
        let choice = ApiChoice::new();
        assert_eq!(choice.current(Platform::Metal), Platform::Metal);
        assert!(!choice.is_chosen());
        let chosen = choice.select(Some(Platform::Vulkan), answer(Platform::Vulkan));
        assert_eq!(chosen, Platform::Vulkan);
        assert_eq!(choice.current(Platform::Metal), Platform::Vulkan);
    }

    #[test]
    fn the_first_select_stands() {
        let choice = ApiChoice::new();
        choice.select(Some(Platform::Vulkan), answer(Platform::Vulkan));
        let again = choice.select(Some(Platform::Metal), answer(Platform::Metal));
        assert_eq!(again, Platform::Vulkan);
    }

    // The select resolves the request; a fallback is what the process keeps.
    #[test]
    fn a_select_keeps_what_the_request_resolved_to() {
        let choice = ApiChoice::new();
        let chosen = choice.select(Some(Platform::Vulkan), answer(Platform::Metal));
        assert_eq!(chosen, Platform::Metal);
        assert_eq!(choice.current(Platform::Vulkan), Platform::Metal);
    }

    #[test]
    fn the_select_is_handed_the_request() {
        let choice = ApiChoice::new();
        let chosen = choice.select(None, |requested| {
            assert_eq!(requested, None);
            Platform::Metal
        });
        assert_eq!(chosen, Platform::Metal);
    }

    // A read that the choice then contradicts is what can leave something
    // built for the wrong API, so it is the one reported at the choice.
    #[test]
    fn an_early_read_the_choice_contradicts_is_found() {
        let choice = ApiChoice::new();
        choice.current(Platform::Metal);
        choice.select(None, answer(Platform::Vulkan));
        assert_eq!(
            choice.mismatched_early_read(Platform::Vulkan),
            Some(Platform::Metal)
        );

        let agreed = ApiChoice::new();
        agreed.current(Platform::Metal);
        agreed.select(None, answer(Platform::Metal));
        assert_eq!(agreed.mismatched_early_read(Platform::Metal), None);
    }

    #[test]
    fn only_another_request_is_overruled() {
        let chosen = Chosen {
            platform: Platform::Metal,
            requested: Some(Platform::Vulkan),
        };
        assert_eq!(
            overruled(chosen, Some(Platform::DirectX)),
            Some(Platform::DirectX)
        );
        assert_eq!(overruled(chosen, Some(Platform::Vulkan)), None);
        assert_eq!(overruled(chosen, Some(Platform::Metal)), None);
        assert_eq!(overruled(chosen, None), None);
    }
}
