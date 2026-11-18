// Widgets: overlay controls the engine draws and drives whole, from their own
// asset and the values they bind, rather than from elements an author places.

mod button;
pub(crate) mod canvas;
mod checkbox;
pub(crate) mod grid;
mod icon;
pub(crate) mod list;
mod menu;
pub(crate) mod modal;
pub(crate) mod scroll_view;
mod style;
mod tabs;
mod text_area;
mod toasts;
pub(crate) mod value_slider;

pub use button::Button;
pub use canvas::Canvas;
pub use checkbox::Checkbox;
pub use grid::Grid;
pub use icon::{Icon, IconPart};
pub use list::{List, ListToggle, MAX_LIST_TOGGLES};
pub use menu::{EntryMark, Menu, MenuEntry};
pub use modal::{Modal, ModalButton, ModalRects};
pub use style::WidgetStyle;
pub use tabs::Tabs;
pub use text_area::{LineMark, TextArea, TextLanguage};
pub use toasts::{ShownToast, Toast, ToastProgress, ToastTone, Toasts};
pub use value_slider::ValueSlider;

// Whether `(x, y)` lies inside the `[x, y, width, height]` rectangle.
pub(crate) fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}
