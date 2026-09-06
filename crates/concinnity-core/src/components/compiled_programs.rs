//! One compiled shader artifact and the digest-keyed lookup over a set of
//! them, shared by every asset whose shader source is only complete once a
//! world is loaded: an `SdfVolume`'s distance field and a `Shader`'s hooks.
//!
//! The cook runs slangc and stores what it emitted; the renderer assembles the
//! source it expects, digests it, and takes a stored artifact only on a match.
//! A hot-reload edit to an engine template misses every entry and recompiles,
//! which is the behaviour that makes editing one possible at all.

use alloc::string::String;
use alloc::vec::Vec;

/// One compiled artifact, the entries it holds, and the source it came from.
///
/// An artifact carries more than one entry where the target allows it: slangc
/// emits one MSL translation unit for a pair of stages, and the Metal runtime
/// wants both in one library. DXIL has no such form, so a container there
/// holds exactly one.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompiledProgram {
    /// Entry point names this artifact holds, as the shader source spells them.
    pub entries: Vec<String>,
    /// `slang_source::source_digest` of the assembled source this artifact was
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
        .find(|p| p.source_digest == digest && p.entries.iter().any(|e| e == entry))
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
                entries: vec!["vertex_main".to_string()],
                source_digest: 7,
                artifact: vec![1, 2, 3],
            },
            // One artifact holding both stages, the shape the Metal target
            // takes: a library the runtime pulls two functions out of.
            CompiledProgram {
                entries: vec![
                    "vertex_main_bindless".to_string(),
                    "fragment_main_bindless".to_string(),
                ],
                source_digest: 9,
                artifact: vec![4, 5],
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
    fn a_shared_artifact_is_found_under_either_entry() {
        let p = programs();
        assert_eq!(artifact(&p, "vertex_main_bindless", 9), Some(&[4u8, 5][..]));
        assert_eq!(
            artifact(&p, "fragment_main_bindless", 9),
            Some(&[4u8, 5][..])
        );
    }

    #[test]
    fn a_program_round_trips_through_postcard() {
        let p = programs();
        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: Vec<CompiledProgram> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, p);
    }
}
