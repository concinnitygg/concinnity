//! The authoring-only group of the asset registry: the types a world declares
//! and the cook expands into the components they stand for, gone before a blob
//! is written.
//!
//! Their list lives here rather than in concinnity-core's `for_each_component!`
//! because nothing in the runtime can hold one: they have no `ComponentTag`, no
//! `ComponentAsset` variant, no column, and no `Component` impl. What they do
//! have is a [`RegisteredType`](super::RegisteredType) variant, so the
//! authoring registry composes this group with core's two.
//!
//! Being in the group is the origin, which is why the entries carry no origin
//! flag; the schemas themselves are [`concinnity_asset::cook`].

pub use concinnity_asset::cook::{
    CameraShot, CharacterModel, CharacterSchema, EngineDefaults, LightRig, MainMenu,
    MaterialPalette, OptionSelect, Panel, Prefab, SceneImport, Slider, StoryImport,
};

/// An asset the cook consumes and never hands to the runtime.
///
/// A world declares one of these, cook expands it into the components it stands
/// for, and nothing of it reaches a blob. Exactly the list below: no tag, no
/// `ComponentAsset` variant, no column, no `Component` impl. Carries no methods
/// -- it exists so the registry's groups are checkable at compile time and so
/// the list is discoverable in the docs. The stored group carries
/// `concinnity_core::ecs::RuntimeComponent` instead, and the resource group
/// `concinnity_core::ecs::ResourceAsset`.
pub trait BuildOnlyAsset {}

/// The authoring-only list. `$cb` receives it as a `build_only:` group shaped
/// exactly like concinnity-core's groups, so one callback serves both.
///
/// The `$cb, $prefix` form prepends tokens to what `$cb` receives, which is how
/// `for_each_authored_type!` hands a callback core's groups and this one
/// together.
#[macro_export]
macro_rules! for_each_build_only_type {
    ($cb:ident) => { $crate::for_each_build_only_type!($cb,); };
    ($cb:ident, $($prefix:tt)*) => {
        $cb! {
            $($prefix)*
            build_only: {
                LightRig          => $crate::registry::build_only::LightRig { },
                MaterialPalette   => $crate::registry::build_only::MaterialPalette { },
                CameraShot        => $crate::registry::build_only::CameraShot { },
                Prefab            => $crate::registry::build_only::Prefab { },
                SceneImport       => $crate::registry::build_only::SceneImport { },
                MainMenu          => $crate::registry::build_only::MainMenu { renders },
                OptionSelect      => $crate::registry::build_only::OptionSelect { },
                Slider            => $crate::registry::build_only::Slider { },
                EngineDefaults    => $crate::registry::build_only::EngineDefaults { },
                StoryImport       => $crate::registry::build_only::StoryImport { },
                Panel             => $crate::registry::build_only::Panel { },
                CharacterSchema   => $crate::registry::build_only::CharacterSchema { },
                CharacterModel    => $crate::registry::build_only::CharacterModel { },
            },
        }
    };
}

// The marker impls, generated from the list so the group and the trait cannot
// disagree.
macro_rules! __impl_build_only {
    (build_only: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? } $(,)?) => {
        $( impl BuildOnlyAsset for $ty {} )+
    };
}

crate::for_each_build_only_type!(__impl_build_only);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{AssetOrigin, RegisteredType, ScopeResolution};

    // The composition is what keeps this group in the one authoring registry:
    // the list lives in this crate while the other two come from
    // concinnity-core, and a callback that saw only core's would silently drop
    // every name below. Derived from the list, so an entry added or removed
    // updates the check with it.
    macro_rules! assert_registered_as_build_only {
        (build_only: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? } $(,)?) => {
            #[test]
            fn every_build_only_type_reaches_the_authoring_registry() {
                let mut count = 0;
                $(
                    let name = stringify!($variant);
                    let ty = RegisteredType::parse(name)
                        .unwrap_or_else(|| panic!("{name} is not a registered type"));
                    assert_eq!(ty, RegisteredType::$variant);
                    assert!(
                        RegisteredType::all().contains(&ty),
                        "{name} is missing from all()"
                    );
                    // A world declares it and the cook expands it, so no
                    // record of it is written: no tag, no column. `addable` is
                    // the External-origin predicate behind the authoring
                    // tools' add list, which offers these through their own
                    // flow instead, so it stays false here.
                    assert_eq!(ty.registration().origin, AssetOrigin::BuildOnly);
                    assert!(!ty.addable(), "{name} is not externally addable");
                    assert_eq!(ty.discriminant(), None, "{name} carries a blob tag");
                    assert_eq!(ty.scope_resolution(), ScopeResolution::Expanded);
                    assert!(!ty.is_resource(), "{name} is not a resource");
                    count += 1;
                )+
                // The group is exactly this list: nothing else in the registry
                // reports the build-only origin.
                let registered = RegisteredType::all()
                    .iter()
                    .filter(|t| t.registration().origin == AssetOrigin::BuildOnly)
                    .count();
                assert_eq!(registered, count);
            }
        };
    }

    crate::for_each_build_only_type!(assert_registered_as_build_only);

    // The marker is the compile-time half of the same fact.
    fn expanded_by_the_cook<T: BuildOnlyAsset>() {}

    #[test]
    fn the_group_carries_its_marker() {
        expanded_by_the_cook::<Prefab>();
        expanded_by_the_cook::<MainMenu>();
        expanded_by_the_cook::<CharacterSchema>();
    }
}
