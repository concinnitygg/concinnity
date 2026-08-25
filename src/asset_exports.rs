// Coverage check for the curated `assets` list: every asset type the registries
// know must be reachable through it. A type added to a registry and not to the
// list fails to compile here.

// Bind one anonymous const per registry entry, naming the type through the
// facade. The list arrives in three groups; every one of them is a type the
// registries know, so all three must be reachable. The entries' metadata blocks
// are captured and ignored.
macro_rules! assert_exported {
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        build_only: { $( $bvariant:ident => $bty:path { $($bmeta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        $( const _: Option<crate::assets::$variant> = None; )+
        $( const _: Option<crate::assets::$bvariant> = None; )+
        $( const _: Option<crate::assets::$rvariant> = None; )+
    };
}

concinnity_core::for_each_component!(assert_exported);
