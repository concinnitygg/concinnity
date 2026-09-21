// Scaffolding shared by the facade's own tests.

// Start `app`'s world on the headless loop, and assert it starts.
//
// A world holding anything renderable resolves to a windowed run, so a cooked
// world started as it stands opens a window and takes a GPU on whatever host
// runs `cargo test`. The headless loop has no render system to gate in.
//
// The window ban is armed first, so a regression that puts the windowed driver
// back here panics naming the backend instead of blocking on an event loop the
// harness cannot end.
pub(crate) fn assert_starts_headless(app: crate::App) {
    concinnity_testing::forbid_windows();
    let mut app = app.into_headless();
    app.driver_mut().start().expect("the world starts");
}
