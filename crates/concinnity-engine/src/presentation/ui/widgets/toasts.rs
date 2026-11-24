// A stack of short-lived message cards, and the event that adds one.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{LayoutAnchor, Screen, SpriteFit, Vocabulary};
use crate::ecs::Ref;

use super::WidgetStyle;

/// What a [`Toast`] reports, which colors its card's edge.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum ToastTone {
    /// Something to know.
    #[default]
    #[vocab("info")]
    Info,
    /// Something finished.
    #[vocab("success")]
    Success,
    /// Something to look at.
    #[vocab("warning")]
    Warning,
    /// Something failed.
    #[vocab("error")]
    Error,
}

/// How far along the work a [`Toast`] reports is, drawn as a bar under its
/// message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToastProgress {
    /// This share of the work is done, from 0 to 1.
    Fraction(f32),
    /// The work is under way but cannot be counted yet.
    Ongoing,
}

/// A message for every [Toasts](#toasts) stack to show: sent by the app's
/// systems as an event.
#[derive(Debug, Clone, PartialEq)]
pub struct Toast {
    /// The message.
    pub text: String,
    /// What it reports.
    pub tone: ToastTone,
    /// Seconds it shows for; 0 takes the stack's `seconds`, and infinity
    /// holds it until it is clicked away.
    pub seconds: f32,
    /// A name for the card: a toast with the key of a shown card replaces it
    /// where it stands. Empty never replaces.
    pub key: String,
    /// A bar under the message. A card with one is not clicked away.
    pub progress: Option<ToastProgress>,
}

impl Toast {
    /// A message of `tone` shown for the stack's usual time.
    pub fn new(text: &str, tone: ToastTone) -> Self {
        Self {
            text: text.into(),
            tone,
            seconds: 0.0,
            key: String::new(),
            progress: None,
        }
    }
}

/// One card a [Toasts](#toasts) stack shows, with the seconds it has left.
#[derive(Debug, Clone, PartialEq)]
pub struct ShownToast {
    /// The message.
    pub text: String,
    /// What it reports.
    pub tone: ToastTone,
    /// Seconds before it is gone.
    pub left: f32,
    /// The card's name, which a click away reports.
    pub key: String,
    /// The bar under the message.
    pub progress: Option<ToastProgress>,
    /// Seconds it has shown, which an ongoing bar animates by.
    pub age: f32,
}

/// A stack of short-lived message cards in a corner of its
/// [Screen](#screen), newest nearest the corner, drawn over the screen's
/// other elements.
///
/// The app's systems send a [`Toast`] event for each message. A card shows
/// for its seconds and fades out; a click dismisses it and sends `on_dismiss`
/// with the card's key. Past `max_visible` cards, a line counts the ones
/// waiting, and a click on it sends `on_more`.
///
/// ```rust
/// # use concinnity_engine::components::{LayoutAnchor, Toasts};
/// let mut toasts = Toasts::default();
/// toasts.anchor = LayoutAnchor::TopRight;
/// toasts.max_visible = 3;
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Toasts {
    /// The corner or edge the stack grows from.
    #[asset(default = LayoutAnchor::BottomRight)]
    pub anchor: LayoutAnchor,
    /// Gap from the screen's edges to the stack.
    #[asset(default = [16.0, 16.0])]
    pub margin: [f32; 2],
    /// Width of each card.
    #[asset(default = 320.0)]
    pub width: f32,
    /// Height of each card.
    #[asset(default = 44.0)]
    pub card_height: f32,
    /// The most cards shown at once.
    #[asset(default = 4)]
    pub max_visible: u32,
    /// Seconds a card shows when its toast names none.
    #[asset(default = 4.0)]
    pub seconds: f32,
    /// The event a click sends as it dismisses a card, carrying the card's
    /// key.
    pub on_dismiss: String,
    /// The event a click on the line counting the waiting cards sends.
    pub on_more: String,
    /// When false the stack is not drawn and takes no clicks; its cards keep
    /// counting down.
    #[asset(default = true)]
    pub visible: bool,
    /// How the cards draw.
    pub style: WidgetStyle,
    /// How a screen-owned stack maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the stack draws over. Unset, it joins the world's HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime cards, oldest first. Not authored.
    #[serde(skip)]
    pub shown: Vec<ShownToast>,
}

// Gap between two cards of a stack.
pub(crate) const CARD_GAP: f32 = 8.0;

impl Toasts {
    // How many cards show, newest first.
    pub(crate) fn visible_count(&self) -> usize {
        self.shown
            .len()
            .min(usize::try_from(self.max_visible).unwrap_or(usize::MAX))
    }

    // The rectangle of the card `slot` places from the anchor (0 nearest it),
    // in a screen of size `area`.
    pub(crate) fn card_rect(&self, slot: usize, area: [f32; 2]) -> [f32; 4] {
        let [fx, fy] = self.anchor.fractions();
        let [mx, my] = self.margin;
        let (w, h) = (self.width, self.card_height);
        let x = mx + (area[0] - 2.0 * mx - w) * fx;
        let step = (h + CARD_GAP) * slot as f32;
        // A stack anchored at the top grows down; any other grows up.
        let y = if fy == 0.0 {
            my + step
        } else {
            my + (area[1] - 2.0 * my - h) * fy - step
        };
        [x, y, w, h]
    }

    // The card under `(x, y)` in a screen of size `area`, as an index into
    // `shown`.
    pub(crate) fn card_at(&self, x: f32, y: f32, area: [f32; 2]) -> Option<usize> {
        let newest = self.shown.len().checked_sub(1)?;
        (0..self.visible_count())
            .find(|&slot| super::point_in(x, y, self.card_rect(slot, area)))
            .map(|slot| newest - slot)
    }

    /// Whether `(x, y)` lands on a shown card or the line counting the waiting
    /// ones, in a screen of size `area`: where the stack takes a click.
    pub fn covers(&self, x: f32, y: f32, area: [f32; 2]) -> bool {
        self.visible
            && (self.card_at(x, y, area).is_some()
                || self
                    .more_rect(area)
                    .is_some_and(|r| super::point_in(x, y, r)))
    }

    // The line counting the waiting cards, in a screen of size `area`, while
    // any wait.
    pub(crate) fn more_rect(&self, area: [f32; 2]) -> Option<[f32; 4]> {
        let visible = self.visible_count();
        (self.shown.len() > visible).then(|| self.card_rect(visible, area))
    }

    // Add `toast` as the newest card, or in place of the shown card with its
    // key.
    pub(crate) fn push(&mut self, toast: &Toast) {
        let left = if toast.seconds > 0.0 {
            toast.seconds
        } else {
            self.seconds
        };
        let keyed = (!toast.key.is_empty())
            .then(|| self.shown.iter_mut().find(|t| t.key == toast.key))
            .flatten();
        if let Some(card) = keyed {
            card.text.clone_from(&toast.text);
            card.tone = toast.tone;
            card.left = left;
            card.progress = toast.progress;
            return;
        }
        self.shown.push(ShownToast {
            text: toast.text.clone(),
            tone: toast.tone,
            left,
            key: toast.key.clone(),
            progress: toast.progress,
            age: 0.0,
        });
    }

    // Count `dt` seconds off every card, dropping the spent ones.
    pub(crate) fn tick(&mut self, dt: f32) {
        for t in &mut self.shown {
            t.left -= dt;
            t.age += dt;
        }
        self.shown.retain(|t| t.left > 0.0);
    }
}

impl ToastTone {
    // The RGBA of a card's edge stripe for this tone.
    pub(crate) fn accent(self) -> [f32; 4] {
        match self {
            Self::Info => [0.58, 0.72, 0.95, 1.0],
            Self::Success => [0.42, 0.72, 0.46, 1.0],
            Self::Warning => [0.95, 0.78, 0.45, 1.0],
            Self::Error => [0.90, 0.42, 0.42, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AREA: [f32; 2] = [1280.0, 720.0];

    #[test]
    fn a_bottom_stack_grows_up_and_a_top_one_down() {
        let mut t = Toasts::default();
        let first = t.card_rect(0, AREA);
        assert_eq!(
            first,
            [1280.0 - 16.0 - 320.0, 720.0 - 16.0 - 44.0, 320.0, 44.0]
        );
        assert_eq!(t.card_rect(1, AREA)[1], first[1] - 44.0 - CARD_GAP);
        t.anchor = LayoutAnchor::Top;
        assert_eq!(t.card_rect(0, AREA)[1], 16.0);
        assert_eq!(t.card_rect(1, AREA)[1], 16.0 + 44.0 + CARD_GAP);
    }

    #[test]
    fn cards_count_down_and_cap_what_shows() {
        let mut t = Toasts {
            max_visible: 2,
            ..Default::default()
        };
        for i in 0..3 {
            t.push(&Toast {
                seconds: 1.0 + i as f32,
                ..Toast::new("x", ToastTone::Info)
            });
        }
        assert_eq!(t.visible_count(), 2);
        t.tick(1.5);
        assert_eq!(t.shown.len(), 2, "the first ran out");
        t.tick(10.0);
        assert!(t.shown.is_empty());
    }

    #[test]
    fn a_keyed_toast_replaces_its_card_and_infinity_holds() {
        let mut t = Toasts::default();
        let keyed = |text: &str, progress| Toast {
            key: "cook".into(),
            seconds: f32::INFINITY,
            progress,
            ..Toast::new(text, ToastTone::Info)
        };
        t.push(&keyed("Cooking", Some(ToastProgress::Ongoing)));
        t.push(&Toast::new("saved", ToastTone::Success));
        t.push(&keyed("Cooking 2/4", Some(ToastProgress::Fraction(0.5))));
        assert_eq!(t.shown.len(), 2, "replaced, not stacked");
        assert_eq!(t.shown[0].text, "Cooking 2/4");
        assert_eq!(t.shown[0].progress, Some(ToastProgress::Fraction(0.5)));
        t.tick(1000.0);
        assert_eq!(t.shown.len(), 1, "the held card stays");
        assert_eq!(t.shown[0].age, 1000.0);
    }

    #[test]
    fn the_waiting_line_sits_past_the_last_card() {
        let mut t = Toasts {
            max_visible: 1,
            ..Default::default()
        };
        t.push(&Toast::new("a", ToastTone::Info));
        assert_eq!(t.more_rect(AREA), None);
        t.push(&Toast::new("b", ToastTone::Info));
        let more = t.card_rect(1, AREA);
        assert_eq!(t.more_rect(AREA), Some(more));
        assert!(t.covers(more[0] + 1.0, more[1] + 1.0, AREA));
        assert!(!t.covers(1.0, 1.0, AREA));
        t.visible = false;
        assert!(!t.covers(more[0] + 1.0, more[1] + 1.0, AREA), "hidden");
    }
}
