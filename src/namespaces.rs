// The partition check for the two vocabulary namespaces, derived from the
// component registry: `components` is its `stored` group, `cook` its
// `build_only` and `resource` groups. A type added to the registry, or moved
// between its groups, fails here until the namespace it is reached through
// follows.
//
// The groups come from two lists: the runtime's (`stored` and `resource`, in
// concinnity-core) and the authoring-only one (`build_only`, in
// concinnity-world, which the runtime tier never links). One derived check per
// list, so each name is checked for being both reachable and reachable through
// the right namespace. The entries' metadata blocks are captured and ignored.

// Bind one runtime-registry entry at a time to the marker trait its group
// carries.
macro_rules! assert_runtime_groups_are_partitioned {
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        fn a_world_holds<T: concinnity_core::ecs::RuntimeComponent>() {}

        #[test]
        fn the_components_namespace_is_the_stored_group() {
            $( a_world_holds::<crate::components::$variant>(); )+
        }

        #[cfg(feature = "cook")]
        fn the_cook_compiles<T: concinnity_core::ecs::ResourceAsset>() {}

        // The five types whose authored form diverges are in both namespaces by
        // design, and are checked above through their runtime form; a bare name
        // here is the authored one, which is the only form the cook sees.
        #[cfg(feature = "cook")]
        #[test]
        fn the_cook_namespace_holds_the_compiled_resources() {
            $( the_cook_compiles::<crate::cook::$rvariant>(); )+
        }
    };
}

concinnity_core::for_each_component!(assert_runtime_groups_are_partitioned);

// The same, for the group the authoring registry owns. Gated whole: without
// `cook` there is no `cook` namespace to reach these through, and no
// concinnity-world to read the list from.
#[cfg(feature = "cook")]
macro_rules! assert_the_build_only_group_is_reachable {
    (build_only: { $( $bvariant:ident => $bty:path { $($bmeta:tt)* } ),+ $(,)? } $(,)?) => {
        fn the_cook_expands<T: concinnity_world::registry::BuildOnlyAsset>() {}

        #[test]
        fn the_cook_namespace_holds_the_expanded_types() {
            $( the_cook_expands::<crate::cook::$bvariant>(); )+
        }
    };
}

#[cfg(feature = "cook")]
concinnity_world::for_each_build_only_type!(assert_the_build_only_group_is_reachable);
