// src/components/prop_instance.rs

/// Marks an entity that was authored as a `Prop`.
///
/// Runtime-only zero-size tag. The `Prop` column is drained once its
/// placement has been decomposed into per-instance components, so this is
/// what identifies a prop's entity from the first tick onward. A behavior
/// scoped to `"Prop"` resolves to this tag, matching model- and mesh-backed
/// props alike.
#[derive(Debug, Clone, Copy, Default)]
pub struct PropInstance;
