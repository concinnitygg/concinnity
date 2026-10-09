// What a behavior system tells the author about without stopping the world.

/// Where a [`BehaviorSystem`](super::BehaviorSystem) reports what an author
/// should know but that does not stop the world, such as a write another
/// system undoes the next tick. A world whose host installs none reports
/// nothing.
pub trait BehaviorReporter: core::fmt::Debug + Send {
    /// Report `message` as a warning.
    fn warn(&self, message: &str);
}
