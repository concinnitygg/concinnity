// Application window schema.

use alloc::string::String;
use alloc::string::ToString;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
/// How the application window is presented.
pub enum WindowMode {
    /// A resizable desktop window.
    #[default]
    Windowed,
    /// Exclusive fullscreen at the display's mode.
    Fullscreen,
    /// A borderless window filling the display.
    Borderless,
}

/// Declares the application window.
///
/// ```json
/// {
///   "name": "main_window",
///   "type": "Window",
///   "args": {
///     "title": "Game",
///     "width": 1280,
///     "height": 720,
///     "mode": "windowed",
///     "resizable": true,
///     "title_bar": true
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Window {
    /// Window title shown in the title bar.
    pub title: String,
    /// Initial window width in pixels.
    pub width: u32,
    /// Initial window height in pixels.
    pub height: u32,
    /// How the window is displayed.
    pub mode: WindowMode,
    /// Whether the user can resize the window.
    pub resizable: bool,
    /// Whether the title bar is drawn, letting content fill the frame when it
    /// is not. Only applies to `windowed` mode: `borderless` has no title bar
    /// by definition, and `fullscreen` leaves the chrome to the OS.
    ///
    /// Platforms differ in what survives the title bar. macOS keeps the close /
    /// minimize / zoom buttons floating over the content, so the window stays
    /// movable and closable. Windows and Linux draw their controls *in* the
    /// title bar, so turning it off there also removes them: the window can
    /// still be resized from its border, but offers no close button and cannot
    /// be dragged.
    pub title_bar: bool,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            title: "Concinnity".to_string(),
            width: 1024,
            height: 768,
            mode: WindowMode::Windowed,
            resizable: false,
            title_bar: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_bar_defaults_on() {
        // A world that says nothing about its chrome keeps a title bar: on
        // Windows and Linux dropping it also drops the close button and the
        // drag handle, which is not something to hand a world by default.
        assert!(Window::default().title_bar);
        let w: Window = serde_json::from_str(r#"{"title":"Game"}"#).unwrap();
        assert!(w.title_bar);
    }

    #[test]
    fn title_bar_round_trips_through_json() {
        let w: Window = serde_json::from_str(r#"{"title_bar":false}"#).unwrap();
        assert!(!w.title_bar);
        // Serializing is the authored-args path (`cn add` writes normalized
        // args back to world.jsonl), so the key has to survive a round trip.
        let back: Window = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert!(!back.title_bar);
    }
}
