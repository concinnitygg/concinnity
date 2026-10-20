// Application window schema.

use crate::components::Vocabulary;
use alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default, Vocabulary)]
/// How the application window is presented.
pub enum WindowMode {
    /// A resizable desktop window.
    #[default]
    #[vocab("windowed")]
    Windowed,
    /// Exclusive fullscreen at the display's mode.
    #[vocab("fullscreen")]
    Fullscreen,
    /// A borderless window filling the display.
    #[vocab("borderless")]
    Borderless,
}

/// Declares the application window.
///
/// ```json
/// ["Window", {
///   "title": "Game",
///   "width": 1280,
///   "height": 720,
///   "mode": "windowed",
///   "resizable": true,
///   "title_bar": true
/// }]
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
pub struct Window {
    /// Window title shown in the title bar.
    #[asset(default = "Concinnity")]
    pub title: String,
    /// Initial window width in pixels.
    #[asset(default = 1024)]
    pub width: u32,
    /// Initial window height in pixels.
    #[asset(default = 768)]
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
    #[asset(default = true)]
    pub title_bar: bool,
}
