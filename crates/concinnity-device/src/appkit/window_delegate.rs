//! NSWindowDelegate that tracks native-fullscreen state and the window's close.
//!
//! macOS native fullscreen is an animated, asynchronous transition: the
//! NSWindow `FullScreen` style-mask bit lags it, so reading the bit right after
//! issuing `toggleFullScreen:` (or stepping the settings menu's Window Mode row
//! faster than the ~1s animation) can momentarily report the wrong state and
//! toggle in the wrong direction. This delegate observes the will / did enter /
//! exit fullscreen notifications and keeps a shared flag in sync, which
//! `set_window_mode` / `set_window_size` read instead of the lagging style mask.
//! It also captures OS-driven transitions (the green traffic-light button,
//! Mission Control) that never go through the settings menu.
//!
//! Close is observed the same way, through `windowWillClose:`. Visibility cannot
//! stand in for it: hiding the app (Cmd-H) or minimizing the window also clears
//! `isVisible`.

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{NSWindow, NSWindowDelegate, NSWindowStyleMask};
use objc2_foundation::NSNotification;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// The window state the delegate observes, shared with the window layer that
// reads it each frame. Atomic because the delegate stores into it from AppKit's
// notification callbacks.
#[derive(Debug, Default)]
pub(crate) struct WindowSignals {
    fullscreen: AtomicBool,
    closed: AtomicBool,
}

impl WindowSignals {
    fn new(fullscreen: bool) -> Self {
        Self {
            fullscreen: AtomicBool::new(fullscreen),
            closed: AtomicBool::new(false),
        }
    }

    // Whether native fullscreen is active.
    pub(crate) fn is_fullscreen(&self) -> bool {
        self.fullscreen.load(Ordering::Relaxed)
    }

    pub(super) fn set_fullscreen(&self, fullscreen: bool) {
        self.fullscreen.store(fullscreen, Ordering::Relaxed);
    }

    // Whether the window has closed. Latched: a closed window never reopens.
    pub(crate) fn closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    fn mark_closed(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct DelegateIvars {
    signals: Arc<WindowSignals>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ConcinnityWindowDelegate"]
    #[ivars = DelegateIvars]
    pub(crate) struct WindowDelegate;

    unsafe impl NSObjectProtocol for WindowDelegate {}

    unsafe impl NSWindowDelegate for WindowDelegate {
        // Fired at the start of the enter-fullscreen animation, so the flag is
        // correct as soon as a transition begins rather than at its end.
        #[unsafe(method(windowWillEnterFullScreen:))]
        fn window_will_enter_full_screen(&self, _notification: &NSNotification) {
            self.ivars().signals.set_fullscreen(true);
        }
        // Fired at the start of the exit-fullscreen animation.
        #[unsafe(method(windowWillExitFullScreen:))]
        fn window_will_exit_full_screen(&self, _notification: &NSNotification) {
            self.ivars().signals.set_fullscreen(false);
        }
        // Re-affirm at the end of each transition in case a will-callback was
        // never delivered (e.g. a transition the system canceled and reversed).
        #[unsafe(method(windowDidEnterFullScreen:))]
        fn window_did_enter_full_screen(&self, _notification: &NSNotification) {
            self.ivars().signals.set_fullscreen(true);
        }
        #[unsafe(method(windowDidExitFullScreen:))]
        fn window_did_exit_full_screen(&self, _notification: &NSNotification) {
            self.ivars().signals.set_fullscreen(false);
        }
        // Fired by every close: the title-bar button, `performClose:`, and a
        // programmatic `close`.
        #[unsafe(method(windowWillClose:))]
        fn window_will_close(&self, _notification: &NSNotification) {
            self.ivars().signals.mark_closed();
        }
    }
);

impl WindowDelegate {
    fn new(mtm: objc2::MainThreadMarker, signals: Arc<WindowSignals>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DelegateIvars { signals });
        // SAFETY: `this` is a freshly allocated instance with its ivars set,
        // and NSObject's `init` is the superclass designated initializer,
        // which consumes the allocation and returns the same instance.
        unsafe { msg_send![super(this), init] }
    }
}

// Create the window delegate, attach it to `window`, and return the delegate
// (which the caller must keep alive: NSWindow holds its delegate as a zeroing
// weak reference) plus the signals both it and the renderer read. The
// fullscreen flag is seeded from the window's current style mask; a freshly
// created window is not fullscreen, so this is normally false.
pub(crate) fn attach_window_delegate(
    mtm: objc2::MainThreadMarker,
    window: &NSWindow,
) -> (Retained<WindowDelegate>, Arc<WindowSignals>) {
    let signals = Arc::new(WindowSignals::new(
        window.styleMask().contains(NSWindowStyleMask::FullScreen),
    ));
    let delegate = WindowDelegate::new(mtm, Arc::clone(&signals));
    window.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    (delegate, signals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_window_is_open() {
        assert!(!WindowSignals::new(false).closed());
        assert!(!WindowSignals::default().closed());
    }

    #[test]
    fn close_latches() {
        let signals = WindowSignals::new(false);
        signals.mark_closed();
        signals.mark_closed();
        assert!(signals.closed());
    }

    #[test]
    fn fullscreen_transitions_never_read_as_a_close() {
        let signals = WindowSignals::new(true);
        assert!(signals.is_fullscreen());
        signals.set_fullscreen(false);
        signals.set_fullscreen(true);
        assert!(signals.is_fullscreen());
        assert!(!signals.closed());
    }

    #[test]
    fn a_close_leaves_the_fullscreen_flag_alone() {
        let signals = WindowSignals::new(true);
        signals.mark_closed();
        assert!(signals.is_fullscreen());
    }
}
