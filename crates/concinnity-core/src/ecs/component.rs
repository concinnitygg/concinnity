//! The runtime-facing component contract: the `Component` trait a registered
//! type implements, and the two plain metadata enums the registry and the blob
//! format are built from.
//!
//! All authoring metadata (reference fields, args schema, validators) lives in
//! the build-side registry in concinnity-world, derived from the
//! `for_each_component!` metadata blocks in [`crate::ecs::registry`].

use crate::ecs::asset_id::AssetId;
use crate::ecs::{ComponentAsset, PayloadLocator};
use crate::result::CnResult;

/// A component a world can hold after the cook: every type an authored world
/// declares that survives into a blob, plus every type only the runtime mints.
///
/// The bound on [`World::add_component`](crate::ecs::World::add_component).
/// Exactly the registry's `stored` group, which is also the group with a
/// `ComponentTag`, a `ComponentAsset` variant, and a column; the `build_only`
/// group implements [`BuildOnlyAsset`] instead. The two are mutually exclusive
/// by construction: an entry is in one group or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be added to a world",
    label = "not a runtime component",
    note = "`{Self}` is a build-time asset: either the cook expands it into the components it stands for (`BuildOnlyAsset`), or it compiles into the blob's resource stream and is reached by handle (`ResourceAsset`). Either way it never reaches a world as a component. Declare it in a world.jsonl and build."
)]
pub trait RuntimeComponent: Into<ComponentAsset> {}

/// An asset the cook consumes and never hands to the runtime.
///
/// A world declares one of these, cook expands it into the components it stands
/// for, and nothing of it reaches a blob. Exactly the registry's `build_only`
/// group: no tag, no `ComponentAsset` variant, no column, no `Component` impl.
/// Carries no methods: it exists so the groups are checkable at compile time and
/// so the list is discoverable in the docs.
pub trait BuildOnlyAsset {}

/// An asset compiled into the blob's resource stream rather than stored as a
/// component.
///
/// A world declares one of these, cook compiles its payload and assigns it a
/// dense per-kind handle, and the runtime keeps it in the resource table owned
/// by the system that reads it. Exactly the registry's `resource` group. Like
/// [`BuildOnlyAsset`], a marker only: the three groups partition the registry,
/// so a type carries exactly one of these.
pub trait ResourceAsset {}

/// Where an asset comes from and whether it persists to a blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetOrigin {
    /// Authored in a world and persisted to the blob.
    External,
    /// Created by the runtime; never persisted.
    RuntimeOnly,
    /// Consumed by the build; never reaches the runtime.
    BuildOnly,
}

/// Whether the asset has a compiled binary payload packed into a .cnb blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetPayload {
    /// No compiled payload; the component is its own data.
    None,
    /// A compiled binary payload packed into the blob.
    Compiled,
}

/// Component -- pure serializable data, no behavior. The runtime-facing surface
/// only: a component loads from its baked blob bytes and receives its injected
/// identity/payload hooks. All authoring metadata (origin, payload kind,
/// reference fields, args schema, validators) lives in the build-side registry
/// (concinnity-world), derived from the `for_each_component!` metadata blocks.
pub trait Component: Sized + Send + core::fmt::Debug + 'static {
    /// The registry name a world authors this component under.
    const NAME: &'static str;

    /// Reconstruct this component from a blob record, whose bytes are the
    /// serialized runtime component (cook already ran the asset -> component
    /// translation). The default rejects: runtime-only components are never
    /// stored in a blob, so only loadable types provide an implementation.
    fn from_baked(_bytes: &[u8]) -> Result<Self, CnResult> {
        Err(CnResult::AssetInvalidType)
    }

    /// Called after construction to inject the payload locator from the blob def.
    /// Only meaningful for components with a compiled payload.
    /// The default implementation does nothing (correct for most components).
    fn inject_locator(&mut self, _locator: PayloadLocator) {}

    /// Called after construction to inject the asset's identity from the blob
    /// def. Only meaningful for components that look themselves up by id at
    /// runtime. The default implementation does nothing.
    fn inject_name(&mut self, _id: AssetId) {}
}

#[cfg(test)]
mod tests {
    use super::{BuildOnlyAsset, RuntimeComponent};
    use crate::components::{CharacterSchema, MainMenu, Prefab, TextLabel, Transform};

    fn runtime<C: RuntimeComponent>() {}
    fn build_only<A: BuildOnlyAsset>() {}

    // One representative per group. The calls are the assertion: each fails to
    // compile if an entry moves between the registry's two groups. An entry
    // cannot be in neither, and the storage half is generated from the same
    // grouping, so a type reaching the wrong marker also loses (or gains) its
    // column.
    #[test]
    fn origin_markers_follow_the_registry() {
        runtime::<TextLabel>();
        runtime::<Transform>();

        build_only::<Prefab>();
        build_only::<MainMenu>();
        build_only::<CharacterSchema>();
    }
}
