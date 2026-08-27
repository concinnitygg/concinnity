//! The macros that build the runtime half of the component registry.
//!
//! `define_components!` is invoked in [`crate::ecs::registry`] over the shared
//! `for_each_component!` list: it emits the `ComponentTag` discriminants, the
//! `ComponentAsset` value enum (via the `__define_asset_kind!` helper), the blob
//! loader, and the `ComponentStorage` / `ComponentSlot` pair, whose storage half
//! comes from [`define_component_storage!`](crate::define_component_storage).
//!
//! The authoring `RegisteredType` registry is built from the same list in
//! concinnity-world. Systems are registered separately, client-side, by the
//! `define_systems!` table.

// Internal helper. Resolves an entry's `consumed` flag into the tag that still
// holds its entities once the world has started: the entry's own tag when no
// load-time pass drains it, the `consumed: <Type>` substitute when one does but
// leaves a runtime marker behind, and `None` when nothing survives.
#[macro_export]
#[doc(hidden)]
macro_rules! __cn_surviving_tag {
    ($variant:ident;) => { Some($crate::ecs::ComponentTag::$variant) };
    ($variant:ident; consumed: $surviving:ident $($rest:tt)*) => {
        Some($crate::ecs::ComponentTag::$surviving)
    };
    ($variant:ident; consumed $($rest:tt)*) => { None };
    ($variant:ident; $skip:tt $($rest:tt)*) => { $crate::__cn_surviving_tag!($variant; $($rest)*) };
}

// Internal helper. Emits the runtime `<Kind>Asset` value enum the ECS stores
// and the `From<$ty>` conversions. The authoring metadata registry (`<Kind>Type`)
// is emitted separately so the two can live in different crates.
#[macro_export]
#[doc(hidden)]
macro_rules! __define_asset_kind {
    (
        asset_enum: $asset_enum:ident,
        asset_kind: $kind_variant:ident,
        $( $variant:ident => $ty:path, $disc:expr_2021 ),+ $(,)?
    ) => {
        // One variant per component type; each is named for the component it
        // wraps, so the list itself is the documentation.
        /// A loaded component of any registered type.
        #[derive(Debug)]
        #[expect(missing_docs, reason = "one variant per component type, each named for the component it wraps")]
        pub enum $asset_enum {
            $( $variant($ty) ),+
        }

        $( impl From<$ty> for $asset_enum { fn from(c: $ty) -> Self { $asset_enum::$variant(c) } } )+
    };
}

/// Generate the runtime component registry from the engine's component list:
/// the `ComponentTag` discriminants, the `ComponentAsset` value enum, and the
/// `ComponentStorage` / `ComponentSlot` pair the ECS stores rows in.
///
/// Each list entry carries a `{ ... }` metadata block the authoring registry
/// consumes; this macro captures and ignores it.
#[macro_export]
macro_rules! define_components {
    // Only the `stored` group gets a tag, an enum variant, and a column; the
    // `resource` group is named here solely to mark it, since a resource is
    // reached by handle rather than stored in one. Each entry's `{ ... }`
    // metadata block is authoring metadata for `cn_impl_components!` and the
    // world-side registry; this macro captures and ignores it.
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        /// The component type tag: one fieldless variant per component, in list
        /// order, so each variant's `#[repr(u8)]` discriminant is its list
        /// position (0, 1, 2, ...). `ComponentTag::$variant as u8` is that tag,
        /// used both as the on-disk blob discriminant and as the in-memory ECS
        /// `ComponentId`. The tag is assigned by position, not hand-written, and
        /// is not a stable on-disk contract: a build regenerates the blob, so the
        /// blob and the engine that loads it always agree. The authoring
        /// `RegisteredType` registry derives the same tag from this enum.
        // One variant per component type, named for that component.
        #[repr(u8)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[expect(missing_docs, reason = "one variant per component type, named for that component")]
        pub enum ComponentTag {
            $( $variant ),+
        }

        impl ComponentTag {
            /// The registry name of this tag, as a world authors it.
            pub fn as_str(self) -> &'static str {
                match self {
                    $( ComponentTag::$variant => stringify!($variant) ),+
                }
            }


            /// The tag that still holds this component's entities once the
            /// world has started, or `None` if nothing does.
            ///
            /// A load-time pass drains some columns during `World::start`, so
            /// they match nothing from the first tick onward. Where such a pass
            /// leaves a runtime marker behind, this returns that marker's tag:
            /// `Prop` resolves to `PropInstance`, which decomposition puts on
            /// every prop's entity. Anything not drained resolves to itself.
            pub fn surviving_tag(self) -> Option<ComponentTag> {
                match self {
                    $( ComponentTag::$variant => $crate::__cn_surviving_tag!($variant; $($meta)*) ),+
                }
            }

            /// The tag a component name denotes. Resolves the component names a
            /// Behavior declares in its `scope` and `queries`.
            pub fn parse(name: &str) -> Option<ComponentTag> {
                $(
                    if name == stringify!($variant) {
                        return Some(ComponentTag::$variant);
                    }
                )+
                None
            }
        }

        $crate::__define_asset_kind! {
            asset_enum: ComponentAsset,
            asset_kind: Component,
            $( $variant => $ty, ComponentTag::$variant as u8 ),+
        }

        impl ComponentAsset {
            /// Reconstruct a component from a blob def: dispatch on the tag and
            /// deserialize the runtime component via `Component::from_baked`
            /// (every record is baked -- cook already ran the asset -> component
            /// translation).
            pub fn from_baked(def: &BlobAssetDef) -> Result<Self, CnResult> {
                $(
                    if def.discriminant == ComponentTag::$variant as u8 {
                        let mut c = <$ty as Component>::from_baked(&def.args_bytes)?;
                        if let Some(id) = def.name {
                            <$ty as Component>::inject_name(&mut c, id);
                        }
                        return Ok(ComponentAsset::$variant(c));
                    }
                )+
                Err(CnResult::AssetInvalidType)
            }

            /// Inject a payload locator into the component after construction.
            /// Delegates to `Component::inject_locator`; a no-op for types
            /// that don't override that method.
            pub fn inject_locator(&mut self, locator: PayloadLocator) {
                match self {
                    $( ComponentAsset::$variant(c) => c.inject_locator(locator) ),+
                }
            }
        }

        // Per-type runtime storage. The `Column`-backed storage struct, the
        // `ComponentSlot` access trait, and the generic storage operations
        // (typed push, drain, mutable access, counts) are generated by
        // `define_component_storage!` -- shared and engine-agnostic. The
        // asset-enum dispatch (`push`, `all_defs`) is engine-specific and
        // added in the impl below.
        $crate::define_component_storage! {
            storage: ComponentStorage,
            slot: ComponentSlot,
            $( $variant => $ty, ComponentTag::$variant as u8 ),+
        }

        impl ComponentStorage {
            /// Dispatch a `ComponentAsset` variant into its typed column via the
            /// generic typed push (which mints the Entity and stamps the tick).
            /// Returns the minted Entity so loaders can index it by name.
            pub fn push(&mut self, asset: ComponentAsset) -> $crate::ecs::Entity {
                match asset {
                    $( ComponentAsset::$variant(c) => self.push_typed(c), )+
                }
            }

            /// Overwrite the component the asset's variant addresses on
            /// `entity`, keeping the entity and its other components, and
            /// stamping the change tick so the frame's readers see it.
            /// `false` when the entity holds no component of that type.
            pub fn replace(&mut self, entity: $crate::ecs::Entity, asset: ComponentAsset) -> bool {
                match asset {
                    $(
                        ComponentAsset::$variant(c) => match self.get_mut::<$ty>(entity) {
                            Some(slot) => {
                                *slot = c;
                                true
                            }
                            None => false,
                        },
                    )+
                }
            }

            /// Every entity carrying the component with this tag, in column
            /// order. Serves the declared-query resolution in BehaviorSystem,
            /// which selects components by authored name rather than by type.
            pub fn entities_with_tag(&self, tag: u8) -> &[$crate::ecs::Entity] {
                $(
                    if tag == ComponentTag::$variant as u8 {
                        return self.$variant.entities();
                    }
                )+
                &[]
            }

            /// How many components of each type are stored: one `(tag, count)`
            /// entry per populated type, in tag order. The debug WS snapshot
            /// reports these; nothing re-serializes stored components back to
            /// defs. Counted rather than listed per instance, so the snapshot
            /// is sized by the number of component types rather than by the
            /// world.
            pub fn component_census(&self) -> ::alloc::vec::Vec<(u8, u32)> {
                let mut out = ::alloc::vec::Vec::new();
                $(
                    let count = self.$variant.len();
                    if count > 0 {
                        out.push((ComponentTag::$variant as u8, count as u32));
                    }
                )+
                out
            }
        }

        // Which group an entry is in decides whether a world can hold it.
        $( impl $crate::ecs::RuntimeComponent for $ty {} )+
        $( impl $crate::ecs::ResourceAsset for $rty {} )+
    };
}
