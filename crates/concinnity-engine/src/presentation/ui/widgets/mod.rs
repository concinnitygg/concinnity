// Widgets: overlay controls the engine draws and drives whole, from their own
// asset and the values they bind, rather than from elements an author places.

mod checkbox;
pub(crate) mod list;
mod menu;
pub(crate) mod modal;
mod style;
mod tabs;
mod text_area;
mod toasts;

pub use checkbox::Checkbox;
pub use list::List;
pub use menu::{Menu, MenuEntry};
pub use modal::{Modal, ModalButton, ModalRects};
pub use style::WidgetStyle;
pub use tabs::Tabs;
pub use text_area::{TextArea, TextLanguage};
pub use toasts::{ShownToast, Toast, ToastTone, Toasts};

// Whether `(x, y)` lies inside the `[x, y, width, height]` rectangle.
pub(crate) fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}
