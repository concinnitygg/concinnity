// src/win32/chrome.rs
//
// Window style selection, shared by window creation and the settings-menu mode
// switch (`window::do_set_window_mode`) so both agree on what a windowed window
// looks like. Mirrors `metal::chrome`.

use windows::Win32::UI::WindowsAndMessaging::{
    WINDOW_STYLE, WS_MAXIMIZEBOX, WS_MINIMIZEBOX, WS_OVERLAPPEDWINDOW, WS_POPUP, WS_SYSMENU,
    WS_THICKFRAME,
};

// Style for a windowed window. Dropping the title bar means dropping
// `WS_CAPTION`, and Windows draws the close / minimize / maximize buttons
// inside it, so they go too -- unlike macOS, where the traffic lights float
// over the content and survive. `WS_THICKFRAME` keeps the resize border and
// `WS_SYSMENU` keeps the taskbar / Alt+Space close route, so the window is
// still resizable and closable even though it cannot be dragged by a caption
// it no longer has.
pub(crate) fn windowed_style(title_bar: bool) -> WINDOW_STYLE {
    if title_bar {
        WS_OVERLAPPEDWINDOW
    } else {
        WS_POPUP | WS_THICKFRAME | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::WS_CAPTION;

    #[test]
    fn title_bar_controls_the_caption_bit() {
        assert_eq!(
            windowed_style(true).0 & WS_CAPTION.0,
            WS_CAPTION.0,
            "a titled window must keep its caption"
        );
        assert_eq!(
            windowed_style(false).0 & WS_CAPTION.0,
            0,
            "a title-bar-less window must drop its caption"
        );
    }

    #[test]
    fn both_styles_stay_resizable() {
        for title_bar in [true, false] {
            assert_eq!(
                windowed_style(title_bar).0 & WS_THICKFRAME.0,
                WS_THICKFRAME.0,
                "title_bar={title_bar} dropped the resize border"
            );
        }
    }
}
