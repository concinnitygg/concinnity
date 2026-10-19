// A standalone window that reports a fatal startup error and offers a Quit
// button, for failures the user can act on but the log cannot reach them with.
//
// It deliberately depends on nothing but the device layer and an embedded glyph
// atlas: no world, no schedule, no blob, no compiled assets. The errors it
// reports are failures to load exactly that data, so anything it needed from
// there would be unavailable at the moment it is needed. When the window cannot
// be created the caller keeps its original console-and-exit path.

use concinnity_core::components::Window;
use concinnity_core::ecs::FontHandle;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::backend_init::BackendInit;
use concinnity_core::render::error_screen::{self, dismissed};
use concinnity_core::render::text::{FontSet, build_text_calls};

// The single atlas slot the embedded face occupies: this screen uploads it
// alone, with no world fonts beside it.
const FONT_HANDLE: FontHandle = FontHandle::new(0);

/// Show `message` in a window with a Quit button, returning once the user
/// dismisses it. `false` means no window could be stood up (no embedded font,
/// or the backend failed to initialize), in which case nothing was displayed
/// and the caller should fall back to reporting the error on the console.
///
/// Blocks the calling thread until dismissed, and must run on the main thread:
/// it creates the process's window.
pub(crate) fn show(title: &str, message: &str) -> bool {
    if !crate::platform::device::AVAILABLE {
        return false;
    }
    let Some(builtin) = crate::render::builtin_font::load(FONT_HANDLE) else {
        return false;
    };
    let atlases = vec![builtin.atlas];
    let mut fonts = FontSet::default();
    fonts.insert(FONT_HANDLE, builtin.loaded);

    let window = Window {
        title: title.to_string(),
        resizable: true,
        ..Window::default()
    };

    // Must precede window creation so AppKit will display it.
    #[cfg(target_os = "macos")]
    crate::app::runloop::activate_app_macos();

    let init = BackendInit::minimal(crate::platform::current(), &window, atlases);
    let mut backend = match crate::platform::device::init_backend(init) {
        Ok(backend) => backend,
        Err(e) => {
            tracing::error!(
                "error screen: no render backend ({e}); reporting on the console instead"
            );
            return false;
        }
    };

    run_loop(backend.as_mut(), message, &fonts);
    backend.wait_idle();
    true
}

// Draw until the user dismisses the screen: a click on Quit, Escape, Return, or
// closing the window.
fn run_loop(backend: &mut dyn RenderBackend, message: &str, fonts: &FontSet) {
    let started = std::time::Instant::now();
    // Hover is resolved against the rect the previous frame laid out, since the
    // cursor position arriving now was sampled against what the user last saw.
    let mut hovered = false;

    loop {
        if backend.window_closed() {
            return;
        }

        let (win_w, win_h) = backend.logical_size();
        let screen = error_screen::build(message, win_w, win_h, fonts, FONT_HANDLE, hovered);
        let text_calls = build_text_calls(&screen.labels, fonts, [win_w, win_h]);

        // The screen draws overlay text only, so no camera is involved.
        backend.update_view(concinnity_core::transform::IDENTITY);
        // Metal pumps its window events inside `draw_frame`, so the draw comes
        // before the input sample, exactly as the world's frame path orders it.
        if let Err(e) = backend.draw_frame(error_screen::frame_params(
            &text_calls,
            started.elapsed().as_secs_f64(),
        )) {
            // The screen cannot report its own failure to draw; the log is all
            // that is left, and staying in the loop would spin on it.
            tracing::error!("error screen: draw failed, closing: {e}");
            return;
        }

        let input = backend.take_input();
        hovered = screen.quit_hit(input.mouse_x, input.mouse_y);
        if dismissed(&input, hovered) {
            return;
        }
    }
}
