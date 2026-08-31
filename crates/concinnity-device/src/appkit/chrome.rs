// src/appkit/chrome.rs
//
// NSWindow creation and chrome styling, shared by both macOS backends and by
// the settings-menu mode switch (`window::set_window_mode`), so all three agree
// on what a windowed window looks like.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2_app_kit::{
    NSBackingStoreType, NSWindow, NSWindowButton, NSWindowStyleMask, NSWindowTitleVisibility,
};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

// Style mask for a windowed window. `FullSizeContentView` extends the content
// under the title bar area; paired with `apply_title_bar` it leaves the
// traffic-light buttons floating over the content, so a title-bar-less window
// can still be moved, zoomed and closed.
pub(crate) fn windowed_style_mask(title_bar: bool) -> NSWindowStyleMask {
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
pub(crate) fn apply_title_bar(window: &NSWindow, title_bar: bool) {
    window.setTitlebarAppearsTransparent(!title_bar);
    window.setTitleVisibility(if title_bar {
        NSWindowTitleVisibility::Visible
    } else {
        NSWindowTitleVisibility::Hidden
    });
    set_window_buttons_hidden(window, false);
}

// Show or hide the close / minimize / zoom traffic-light buttons.
pub(crate) fn set_window_buttons_hidden(window: &NSWindow, hidden: bool) {
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

// Create the engine's NSWindow at the requested content size, styled from the
// world's authored title-bar choice. Shared by both macOS backends: the Metal
// backend fills it with an MTKView, the Vulkan backend with a CAMetalLayer-
// hosting NSView.
pub(crate) fn create_window(
    mtm: objc2::MainThreadMarker,
    title: &str,
    width: u32,
    height: u32,
    title_bar: bool,
) -> Result<Retained<NSWindow>, String> {
    concinnity_core::window_policy::assert_windows_allowed("the AppKit window");

    let content_rect = NSRect::new(
        NSPoint::new(0.0, 0.0),
        NSSize::new(width as f64, height as f64),
    );
    // SAFETY: `mtm` proves this is the main thread, which is where AppKit
    // requires NSWindow to be created; the designated initializer consumes the
    // fresh allocation exactly once.
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            content_rect,
            windowed_style_mask(title_bar),
            NSBackingStoreType::Buffered,
            false,
        )
    };
    window.setTitle(&NSString::from_str(title));
    apply_title_bar(&window, title_bar);
    window.center();
    // Prevent AppKit from releasing the window when it is closed. The default
    // is YES for alloc/init-created windows, which causes AppKit to release
    // (and possibly deallocate) the window on close while Rust's Retained<NSWindow>
    // still holds a reference, leading to EXC_BAD_ACCESS in objc_release.
    // SAFETY: main-thread AppKit property setter on a live NSWindow.
    unsafe { window.setReleasedWhenClosed(false) };
    // Receive mouse-moved events even when the cursor is outside the window
    // area (necessary when CGAssociateMouseAndMouseCursorPosition decouples
    // cursor position from hardware movement).
    window.setAcceptsMouseMovedEvents(true);
    Ok(window)
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
