// Scaffolding shared by the facade's own tests.

// Start `app`'s world on the headless loop, and assert it starts.
//
// The cook injects a GraphicsConfig for anything renderable, and that component
// is what gates the render stack into a windowed run: a cooked world started as
// it stands opens a window and takes a GPU on whatever host runs `cargo test`.
// The headless loop has no render system for it to gate in.
pub(crate) fn assert_starts_headless(app: crate::App) {
    let mut app = app.into_headless();
    assert_eq!(app.inner_mut().start(), Ok(()));
}
