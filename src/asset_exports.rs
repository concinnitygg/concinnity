// Coverage check for the curated `assets` list: every asset type the registries
// know must be reachable through it. A type added to a registry and not to the
// list fails to compile here.

// Bind one anonymous const per registry entry, naming the type through the
// facade. The entries' metadata blocks are captured and ignored.
macro_rules! assert_exported {
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        $( const _: Option<crate::assets::$variant> = None; )+
    };
}

concinnity_core::for_each_component!(assert_exported);
concinnity_core::for_each_resource_asset!(assert_exported);
