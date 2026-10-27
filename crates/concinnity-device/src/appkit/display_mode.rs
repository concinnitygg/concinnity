//! Display-mode enumeration and fullscreen mode switching via CoreGraphics.
//! `enumerate` lists the modes (pixel resolution + refresh rate) the window's
//! display supports, feeding the Resolution settings row. While the window is
//! in native fullscreen, `FullscreenDisplayMode` holds the display to the
//! user's chosen mode with `CGDisplaySetDisplayMode` and restores the desktop's
//! original mode when the window leaves fullscreen (including OS-driven exits:
//! the reconcile runs once per frame off the delegate-tracked flag) or the
//! context is dropped. Outside fullscreen the choice is only remembered; the
//! windowed resize path is unaffected.

use concinnity_core::math::{SaturatingFrom, narrow_f32};
use concinnity_engine::window::display_mode::{DisplayMode, DisplayProbe, DisplayRefresh};
use objc2_app_kit::{NSScreen, NSWindow};
use objc2_core_foundation::{CFDictionary, CFRetained, Type, kCFBooleanTrue};
use objc2_core_graphics::{
    CGDirectDisplayID, CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode, CGDisplayMode,
    CGDisplaySetDisplayMode, CGError, CGMainDisplayID, kCGDisplayShowDuplicateLowResolutionModes,
};
use objc2_foundation::{NSNumber, ns_string};

// The id of the display `window` sits on, falling back to the main display
// (also the embedded-mode answer, where no engine window exists).
fn display_id(window: Option<&NSWindow>) -> CGDirectDisplayID {
    window
        .and_then(NSWindow::screen)
        .and_then(|s| screen_display_id(&s))
        .unwrap_or_else(|| CGMainDisplayID())
}

// NSScreen's display id, read from its device description ("NSScreenNumber").
fn screen_display_id(screen: &NSScreen) -> Option<CGDirectDisplayID> {
    let desc = screen.deviceDescription();
    let obj = desc.objectForKey(ns_string!("NSScreenNumber"))?;
    let num = obj.downcast_ref::<NSNumber>()?;
    Some(num.unsignedIntValue())
}

// The modes the window's display supports, as raw (pixel width, pixel height,
// refresh Hz) values; the caller dedups + sorts. Duplicate low-resolution
// (HiDPI) modes are requested so the common game resolutions appear on a
// Retina panel; modes CoreGraphics marks unusable for a desktop GUI (e.g.
// stretched) are skipped. Empty when CoreGraphics returns nothing.
pub(super) fn enumerate(window: Option<&NSWindow>) -> Vec<DisplayMode> {
    let display = display_id(window);
    let mut out = Vec::new();
    for mode in copy_all_modes(display) {
        if !CGDisplayMode::is_usable_for_desktop_gui(Some(&mode)) {
            continue;
        }
        if let Some(dm) = display_mode_of(&mode) {
            out.push(dm);
        }
    }
    out
}

// The mode the window's display is currently running.
pub(super) fn current(window: Option<&NSWindow>) -> Option<DisplayMode> {
    let mode = CGDisplayCopyDisplayMode(display_id(window))?;
    display_mode_of(&mode)
}

// How fast the window's screen can refresh, from the shortest and longest
// refresh intervals NSScreen reports. `None` before the window is on a screen.
pub(super) fn refresh(window: &NSWindow) -> Option<DisplayRefresh> {
    let screen = window.screen()?;
    refresh_from_intervals(
        screen.minimumRefreshInterval(),
        screen.maximumRefreshInterval(),
    )
}

// The refresh range for a screen's shortest and longest frame intervals in
// seconds, equal on a fixed-rate display. `None` for an interval that is not
// a positive time.
fn refresh_from_intervals(shortest: f64, longest: f64) -> Option<DisplayRefresh> {
    if !(shortest > 0.0 && longest >= shortest) {
        return None;
    }
    Some(DisplayRefresh {
        max_hz: narrow_f32(1.0 / shortest),
        min_hz: Some(narrow_f32(1.0 / longest)),
    })
}

// The main display's mode and its pixels per point, read before any window
// exists: a window opens on the main display at its size in points.
pub(crate) fn main_display_probe() -> Option<DisplayProbe> {
    let mode = CGDisplayCopyDisplayMode(CGMainDisplayID())?;
    let points = CGDisplayMode::width(Some(&mode));
    let pixels = display_mode_of(&mode)?;
    let scale = if points == 0 {
        1.0
    } else {
        pixels.width as f32 / points as f32
    };
    Some(DisplayProbe {
        mode: pixels,
        scale,
        fullscreen_scale: 1.0,
    })
}

// Every CGDisplayMode of `display` (including HiDPI duplicates), retained.
fn copy_all_modes(display: CGDirectDisplayID) -> Vec<CFRetained<CGDisplayMode>> {
    // Ask for the duplicate low-resolution modes too; without the option the
    // scaled (HiDPI) resolutions a game typically offers are omitted.
    // SAFETY: `kCFBooleanTrue` and `kCGDisplayShowDuplicateLowResolutionModes`
    // are framework-owned statics that outlive the dictionary built from them,
    // and that dictionary is the options type CGDisplayCopyAllDisplayModes
    // documents; `display` came from a live CGDirectDisplayID.
    let options = unsafe {
        kCFBooleanTrue.map(|yes| {
            CFDictionary::from_slices(&[kCGDisplayShowDuplicateLowResolutionModes], &[yes])
        })
    };
    // SAFETY: as above -- `options` is either null or the documented options
    // dictionary, and it outlives the call.
    let array = unsafe {
        CGDisplayCopyAllDisplayModes(display, options.as_deref().map(CFDictionary::as_opaque))
    };
    let Some(array) = array else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in 0..array.count() {
        // SAFETY: the array's values are CGDisplayModes (documented by
        // CGDisplayCopyAllDisplayModes) and `i` is in bounds, so the raw
        // accessor's requirements hold.
        let ptr = unsafe { array.value_at_index(i) };
        if ptr.is_null() {
            continue;
        }
        // SAFETY: non-null per the check above, and the array owns the mode for
        // the rest of this loop body; `retain` then detaches the reference from
        // the array's lifetime.
        let mode: &CGDisplayMode = unsafe { &*ptr.cast() };
        out.push(mode.retain());
    }
    out
}

// The (pixel width, pixel height, rounded refresh Hz) of a CGDisplayMode, or
// `None` for a degenerate mode. A refresh of 0 stays 0 (unknown; some built-in
// panels report none).
fn display_mode_of(mode: &CGDisplayMode) -> Option<DisplayMode> {
    let width = u32::saturating_from(CGDisplayMode::pixel_width(Some(mode)));
    let height = u32::saturating_from(CGDisplayMode::pixel_height(Some(mode)));
    if width == 0 || height == 0 {
        return None;
    }
    let refresh_hz = u32::saturating_from(CGDisplayMode::refresh_rate(Some(mode)).round().max(0.0));
    Some(DisplayMode {
        width,
        height,
        refresh_hz,
    })
}

// The CGDisplayMode of `display` matching `want`. An exact (resolution, rate)
// match wins; a `want` with an unknown rate (0) takes the matching resolution's
// highest rate. `None` when the display has no mode at that resolution (e.g. a
// stale persisted choice from another monitor).
fn find_native_mode(
    display: CGDirectDisplayID,
    want: DisplayMode,
) -> Option<CFRetained<CGDisplayMode>> {
    let mut best: Option<(u32, CFRetained<CGDisplayMode>)> = None;
    for mode in copy_all_modes(display) {
        let Some(dm) = display_mode_of(&mode) else {
            continue;
        };
        if dm.width != want.width || dm.height != want.height {
            continue;
        }
        if want.refresh_hz != 0 && dm.refresh_hz == want.refresh_hz {
            return Some(mode);
        }
        if best.as_ref().is_none_or(|(hz, _)| dm.refresh_hz > *hz) {
            best = Some((dm.refresh_hz, mode));
        }
    }
    // With a known wanted rate but no exact rate match, the resolution's
    // nearest-available (highest) rate still honors the resolution choice.
    best.map(|(_, mode)| mode)
}

// Holds the display to the user's chosen mode while the window is in native
// fullscreen. Owned by `AppKitWindow`; `reconcile` runs once per frame from the
// delegate-tracked fullscreen flag, so entry (after the animation starts),
// menu-driven exit, and OS-driven exit (green traffic-light button, Mission
// Control) all converge on the right display state.
pub(super) struct FullscreenDisplayMode {
    // The user's chosen mode, or `None` to leave the display alone.
    desired: Option<DisplayMode>,
    // The desktop mode captured before the first switch, restored on exit.
    original: Option<CFRetained<CGDisplayMode>>,
    // The display being held, resolved from the window when first switching
    // (a fullscreen window cannot change displays mid-hold).
    display: Option<CGDirectDisplayID>,
    // The mode currently applied, so the per-frame reconcile is a no-op
    // comparison while nothing changes.
    applied: Option<DisplayMode>,
}

impl std::fmt::Debug for FullscreenDisplayMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullscreenDisplayMode")
            .field("desired", &self.desired)
            .field("applied", &self.applied)
            .field("holds_original", &self.original.is_some())
            .finish_non_exhaustive()
    }
}

impl FullscreenDisplayMode {
    pub(super) fn new() -> Self {
        Self {
            desired: None,
            original: None,
            display: None,
            applied: None,
        }
    }

    // Remember the mode to hold while fullscreen. Applied (or re-applied) by
    // the next reconcile; outside fullscreen it only updates the memory.
    pub(super) fn set_desired(&mut self, mode: DisplayMode) {
        self.desired = Some(mode);
    }

    // Converge the display on the current fullscreen state: switch to the
    // desired mode while fullscreen, restore the desktop mode otherwise.
    // Cheap when nothing changed (two field compares).
    pub(super) fn reconcile(&mut self, window: Option<&NSWindow>, fullscreen: bool) {
        if !fullscreen {
            self.restore();
            return;
        }
        let Some(desired) = self.desired else {
            return;
        };
        if self.applied == Some(desired) {
            return;
        }
        let display = *self.display.get_or_insert_with(|| display_id(window));
        let Some(mode) = find_native_mode(display, desired) else {
            // No mode at that resolution on this display (a stale persisted
            // choice): leave the display alone and stop retrying.
            tracing::warn!(
                "display has no {}x{} mode; keeping the current mode",
                desired.width,
                desired.height
            );
            self.desired = None;
            return;
        };
        if self.original.is_none() {
            self.original = CGDisplayCopyDisplayMode(display);
        }
        // SAFETY: `mode` was just looked up from this same display's mode
        // list, so it is a valid mode for `display`; no options are passed.
        let err = unsafe { CGDisplaySetDisplayMode(display, Some(&mode), None) };
        if err == CGError::Success {
            self.applied = Some(desired);
        } else {
            tracing::warn!(
                "CGDisplaySetDisplayMode({}x{}@{}Hz) failed: {:?}",
                desired.width,
                desired.height,
                desired.refresh_hz,
                err
            );
            self.desired = None;
        }
    }

    // Put the display back on the desktop mode captured before the first
    // switch. Idempotent, and run from this type's own `Drop`, so quitting while
    // fullscreen never strands the desktop on the game's mode.
    pub(super) fn restore(&mut self) {
        let Some(original) = self.original.take() else {
            return;
        };
        let Some(display) = self.display else {
            return;
        };
        // SAFETY: `original` was copied from this same display before the
        // first switch, so it is a valid mode for `display`.
        let err = unsafe { CGDisplaySetDisplayMode(display, Some(&original), None) };
        if err != CGError::Success {
            tracing::warn!("restoring the desktop display mode failed: {:?}", err);
        }
        self.applied = None;
    }
}

impl Drop for FullscreenDisplayMode {
    fn drop(&mut self) {
        self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_promotion_screen_reads_as_variable() {
        let refresh = refresh_from_intervals(1.0 / 120.0, 1.0 / 24.0).unwrap();
        assert_eq!(refresh.label(), "120 Hz, variable down to 24 Hz");
    }

    #[test]
    fn equal_intervals_read_as_fixed() {
        let refresh = refresh_from_intervals(1.0 / 60.0, 1.0 / 60.0).unwrap();
        assert_eq!(refresh.label(), "60 Hz fixed");
    }

    #[test]
    fn a_missing_interval_reads_as_unknown() {
        assert_eq!(refresh_from_intervals(0.0, 0.0), None);
        assert_eq!(refresh_from_intervals(1.0 / 60.0, 1.0 / 120.0), None);
    }
}
