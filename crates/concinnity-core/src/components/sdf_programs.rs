//! The compiled form of an `SdfVolume`'s distance field: what the cook puts in
//! the volume's payload and the renderer takes out of it.
//!
//! An `SdfVolume` is the one asset whose shader source is only complete once a
//! world is loaded, because the world authors the field that goes in the middle
//! of the engine's template. Every other engine shader is a build-time artifact.
//! Making this one a build-time artifact too is what keeps a shipped player from
//! needing a shader compiler: the cook runs dxc and stores what it emitted.
//!
//! The field rides along with the artifacts because a compiled artifact is only
//! usable while the template it was built against still matches. The renderer
//! assembles the source it expects, digests it, and takes the stored artifact
//! only on a match; a hot-reload edit to the engine template misses every entry
//! and recompiles, which is the behavior that makes editing one possible at
//! all. The field's path is part of that source, fenced around the text so a
//! compiler names the author's file, so it rides along too.

use alloc::vec::Vec;

use super::compiled_programs::CompiledProgram;
use super::shader::ShaderSource;

/// An `SdfVolume`'s payload: the authored field plus every entry the cook
/// compiled from it.
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
pub struct SdfPrograms {
    /// The authored distance field and the path it was compiled under, spliced
    /// at `{SDF_BODY}`. Kept so a renderer can assemble the source the stored
    /// artifacts were built from, and compile when it cannot use them.
    pub field: ShaderSource,
    /// Compiled entries, in the order the cook emitted them.
    pub programs: Vec<CompiledProgram>,
}

impl SdfPrograms {
    /// Serialize the payload for the blob.
    pub fn encode(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Read a payload back out of the blob.
    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// The artifact holding `entry`, if one was compiled from source matching
    /// `digest`. A mismatch is a stale artifact and reads as absent.
    pub fn artifact(&self, entry: &str, digest: u64) -> Option<&[u8]> {
        super::compiled_programs::artifact(&self.programs, entry, digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn field(text: &str) -> ShaderSource {
        ShaderSource {
            path: "shaders/blob.hlsl".to_string(),
            text: text.to_string(),
        }
    }

    fn programs() -> SdfPrograms {
        SdfPrograms {
            field: field("float map() { return 1.0; }"),
            programs: vec![
                CompiledProgram {
                    entry: "raymarch_vertex".to_string(),
                    source_digest: 7,
                    artifact: vec![1, 2, 3],
                },
                CompiledProgram {
                    entry: "raymarch_volumetric_vertex".to_string(),
                    source_digest: 9,
                    artifact: vec![4, 5],
                },
                CompiledProgram {
                    entry: "raymarch_volumetric_fragment".to_string(),
                    source_digest: 9,
                    artifact: vec![6],
                },
            ],
        }
    }

    #[test]
    fn an_entry_resolves_only_against_the_digest_it_was_built_from() {
        let p = programs();
        assert_eq!(p.artifact("raymarch_vertex", 7), Some(&[1u8, 2, 3][..]));
        // Two entries under one digest each resolve to their own artifact.
        assert_eq!(
            p.artifact("raymarch_volumetric_vertex", 9),
            Some(&[4u8, 5][..])
        );
        assert_eq!(
            p.artifact("raymarch_volumetric_fragment", 9),
            Some(&[6u8][..])
        );
        // The template moved under the artifact: the renderer has to compile.
        assert_eq!(p.artifact("raymarch_vertex", 8), None);
        // An entry the cook never emitted, for instance a shadow caster on a
        // volume that did not declare one.
        assert_eq!(p.artifact("raymarch_shadow_vertex", 7), None);
    }

    #[test]
    fn the_field_and_the_artifacts_round_trip_through_postcard() {
        let bytes = postcard::to_allocvec(&programs()).unwrap();
        assert_eq!(
            postcard::from_bytes::<SdfPrograms>(&bytes).unwrap(),
            programs()
        );
    }

    // A volume whose payload predates the compiled form, or one the cook could
    // not compile, still carries its field: the renderer falls back to
    // compiling every entry rather than drawing nothing.
    #[test]
    fn a_payload_with_no_artifacts_still_carries_the_field() {
        let p = SdfPrograms {
            field: field("float map() { return 0.0; }"),
            programs: Vec::new(),
        };
        assert!(p.artifact("raymarch_fragment", 0).is_none());
        assert!(!p.field.text.is_empty());
    }
}
