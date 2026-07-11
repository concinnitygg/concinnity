// Per-kind resource handles.
//
// A resource (a mesh, texture, material, ...) is shared, compiled data the
// runtime addresses by a dense integer index into a per-kind resource table.
// Each kind has its own `0..N` index space, assigned by cook in declaration
// order. The handle is a newtype per kind so a `TextureHandle` cannot be passed
// where a `MeshHandle` is expected. Like `AssetId`, a handle serializes as a
// bare `u32`.
//
// These are the runtime replacement for the per-reference `AssetId` a component
// carries today: cook resolves the name to the resource's handle at build time,
// so the runtime never scans to resolve a reference.

use serde::{Deserialize, Serialize};

macro_rules! resource_handles {
    ( $( $(#[$m:meta])* $name:ident ),+ $(,)? ) => {
        $(
            $(#[$m])*
            #[derive(
                Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default,
                Serialize, Deserialize,
            )]
            #[serde(transparent)]
            pub struct $name(pub u32);

            impl $name {
                // The handle's index into its per-kind resource table.
                pub fn index(self) -> usize {
                    self.0 as usize
                }
            }
        )+
    };
}

resource_handles! {
    MeshHandle,
    TextureHandle,
    MaterialHandle,
    FontHandle,
    AudioClipHandle,
    CubemapTextureHandle,
    EnvironmentMapHandle,
    ColorLutHandle,
    SkinnedMeshHandle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_serializes_as_a_bare_u32() {
        // Same wire form as AssetId: a bare integer, not a one-tuple.
        let json = serde_json::to_string(&TextureHandle(7)).unwrap();
        assert_eq!(json, "7");
        let back: TextureHandle = serde_json::from_str("7").unwrap();
        assert_eq!(back, TextureHandle(7));
    }

    #[test]
    fn index_is_the_inner_value() {
        assert_eq!(MeshHandle(0).index(), 0);
        assert_eq!(MeshHandle(42).index(), 42);
    }

    #[test]
    fn per_kind_handles_are_distinct_types_with_independent_values() {
        // A round-trip through a small table keyed by the raw index works the
        // same for each kind; the types just keep the spaces from mixing.
        let table = ["a", "b", "c"];
        assert_eq!(table[TextureHandle(1).index()], "b");
        assert_eq!(table[MeshHandle(2).index()], "c");
    }
}
