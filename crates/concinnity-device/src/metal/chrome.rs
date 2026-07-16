// src/metal/chrome.rs
//
// NSWindow chrome styling, shared by window creation (`init::window`) and the
// settings-menu mode switch (`input::set_window_mode`) so both agree on what a
// windowed window looks like.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2_app_kit::{NSWindow, NSWindowButton, NSWindowStyleMask, NSWindowTitleVisibility};

// Style mask for a windowed window. `FullSizeContentView` extends the content
// under the title bar area; paired with `apply_title_bar` it leaves the
// traffic-light buttons floating over the content, so a title-bar-less window
// can still be moved, zoomed and closed.
pub(super) fn windowed_style_mask(title_bar: bool) -> NSWindowStyleMask {
    let standard = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;
    if title_bar {
        standard
    } else {
        standard | NSWindowStyleMask::FullSizeContentView
    }
}

// Show or hide the title bar strip, keeping the traffic-light buttons visible
// either way. `Titled` stays in the mask even when hidden: a non-titled window
// cannot become key, which would kill keyboard input.
pub(super) fn apply_title_bar(window: &NSWindow, title_bar: bool) {
    window.setTitlebarAppearsTransparent(!title_bar);
    window.setTitleVisibility(if title_bar {
        NSWindowTitleVisibility::Visible
    } else {
        NSWindowTitleVisibility::Hidden
    });
    set_window_buttons_hidden(window, false);
}

// Show or hide the close / minimize / zoom traffic-light buttons.
pub(super) fn set_window_buttons_hidden(window: &NSWindow, hidden: bool) {
    for kind in [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ] {
        if let Some(button) = window.standardWindowButton(kind) {
            button.setHidden(hidden);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_style_mask_keeps_the_standard_controls() {
        for title_bar in [true, false] {
            let mask = windowed_style_mask(title_bar);
            // Titled is what makes the window key-window-eligible, and the
            // three control bits back the traffic lights; none may drop when
            // the title bar is turned off.
            for bit in [
                NSWindowStyleMask::Titled,
                NSWindowStyleMask::Closable,
                NSWindowStyleMask::Miniaturizable,
                NSWindowStyleMask::Resizable,
            ] {
                assert!(mask.contains(bit), "title_bar={title_bar} dropped {bit:?}");
            }
        }
    }

    #[test]
    fn full_size_content_view_tracks_the_title_bar() {
        assert!(
            !windowed_style_mask(true).contains(NSWindowStyleMask::FullSizeContentView),
            "a titled window must not extend content under the title bar"
        );
        assert!(
            windowed_style_mask(false).contains(NSWindowStyleMask::FullSizeContentView),
            "a title-bar-less window must fill the frame"
        );
    }
}
