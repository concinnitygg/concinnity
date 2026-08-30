// Scaffolding shared by the facade's own tests.

// Start `app`'s world with no renderer behind it, and assert it starts.
//
// The cook injects a GraphicsConfig for anything renderable, and that component
// is what gates the render stack into the schedule: a cooked world started as
// it stands opens a window and takes a GPU on whatever host runs `cargo test`.
// Dropping the column leaves every other system. The `no_std` tier has no
// renderer to drop.
pub(crate) fn assert_starts_headless(app: &mut crate::App) {
    #[cfg(feature = "std")]
    app.inner_mut()
        .world_mut()
        .remove_all::<crate::components::GraphicsConfig>();
    assert_eq!(app.inner_mut().start(), Ok(()));
}
