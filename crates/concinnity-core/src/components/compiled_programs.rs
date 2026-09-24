//! One compiled shader artifact and the digest-keyed lookup over a set of
//! them, shared by every asset whose shader source is only complete once a
//! world is loaded: an `SdfVolume`'s distance field and a `Shader`'s hooks.
//!
//! The cook runs dxc and stores what it emitted; the renderer assembles the
//! source it expects, digests it, and takes a stored artifact only on a match.
//! A hot-reload edit to an engine template misses every entry and recompiles,
//! which is the behavior that makes editing one possible at all.

use alloc::string::String;
use alloc::vec::Vec;

/// One compiled artifact, the entry point it holds, and the source it came from.
///
/// The cook compiles one entry point per artifact on every target: a variant
/// binds only the resources it reads, so each entry is its own translation.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
pub struct CompiledProgram {
    /// The entry point this artifact holds, as the shader source spells it.
    pub entry: String,
    /// `shader_source::source_digest` of the assembled source this artifact was
    /// built from. A renderer whose assembly digests differently has a template
    /// the artifact predates and must compile rather than load.
    pub source_digest: u64,
    /// The emitted artifact: SPIR-V, a signed DXIL container, or MSL text.
    pub artifact: Vec<u8>,
}

/// The artifact holding `entry`, if one was compiled from source matching
/// `digest`. A mismatch is a stale artifact and reads as absent.
pub fn artifact<'a>(programs: &'a [CompiledProgram], entry: &str, digest: u64) -> Option<&'a [u8]> {
    programs
        .iter()
        .find(|p| p.source_digest == digest && p.entry == entry)
        .map(|p| p.artifact.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn programs() -> Vec<CompiledProgram> {
        vec![
            CompiledProgram {
                entry: "vertex_main".to_string(),
                source_digest: 7,
                artifact: vec![1, 2, 3],
            },
            // Both stages of one file under one digest, each its own artifact.
            CompiledProgram {
                entry: "vertex_main_bindless".to_string(),
                source_digest: 9,
                artifact: vec![4, 5],
            },
            CompiledProgram {
                entry: "fragment_main_bindless".to_string(),
                source_digest: 9,
                artifact: vec![6],
            },
        ]
    }

    #[test]
    fn an_entry_is_found_under_its_own_digest_only() {
        let p = programs();
        assert_eq!(artifact(&p, "vertex_main", 7), Some(&[1u8, 2, 3][..]));
        assert_eq!(artifact(&p, "vertex_main", 8), None, "stale digest");
        assert_eq!(artifact(&p, "no_such_entry", 7), None);
    }

    #[test]
    fn entries_sharing_a_digest_each_find_their_own_artifact() {
        let p = programs();
        assert_eq!(artifact(&p, "vertex_main_bindless", 9), Some(&[4u8, 5][..]));
        assert_eq!(artifact(&p, "fragment_main_bindless", 9), Some(&[6u8][..]));
    }

    #[test]
    fn a_program_round_trips_through_postcard() {
        let p = programs();
        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: Vec<CompiledProgram> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, p);
    }
}
