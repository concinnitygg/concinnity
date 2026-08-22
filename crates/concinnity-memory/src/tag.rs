// concinnity-memory/src/tag.rs
//
// The vocabulary subsystems account against: what a block of memory is for, and
// which memory it sits in.
//
// It is deliberately a small closed set rather than an open string registry. A
// fixed set indexes straight into a flat array of counters, which is what lets
// the ledger stay allocation-free and readable from a global allocator's
// neighbourhood; it also keeps a readout's rows stable frame to frame instead of
// appearing and reordering as strings are interned.

/// Which memory a report is about. The two are counted separately because they
/// are separately budgeted and separately exhausted: a host allocation and a
/// device allocation for the same texture are two different costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Realm {
    /// Process memory: what the CPU allocates and the Rust heap holds.
    Host,
    /// Device memory: what a GPU backend allocates, whether that is discrete
    /// VRAM or a unified-memory working set.
    Device,
}

impl Realm {
    /// Number of realms.
    pub const COUNT: usize = 2;
    /// Every realm, in readout order.
    pub const ALL: [Realm; Self::COUNT] = [Realm::Host, Realm::Device];

    /// The realm's position in a per-realm table.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// How a readout names the realm.
    pub const fn name(self) -> &'static str {
        match self {
            Realm::Host => "RAM",
            Realm::Device => "VRAM",
        }
    }
}

/// What a block of memory is for. `Other` is the honest bucket for a reporter
/// that has no better answer; it is not a catch-all for everything unreported,
/// since the ledger only ever holds what someone reports into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemTag {
    /// Texture images.
    Textures,
    /// Mesh geometry.
    Meshes,
    /// Streamed world chunks.
    Chunks,
    /// Compiled shader binaries and pipeline state.
    Shaders,
    /// Decoded audio clips and mixer buffers.
    Audio,
    /// Physics bodies, colliders, and broad-phase structures.
    Physics,
    /// Overlay and HUD geometry.
    Ui,
    /// Per-frame working memory: arenas and pools that are reset or reused
    /// rather than freed.
    Scratch,
    /// Anything with no better bucket.
    Other,
}

impl MemTag {
    /// Number of tags.
    pub const COUNT: usize = 9;
    /// Every tag, in the order a readout lists them. Fixed, so rows never
    /// reorder under a reader as the numbers move.
    pub const ALL: [MemTag; Self::COUNT] = [
        MemTag::Textures,
        MemTag::Meshes,
        MemTag::Chunks,
        MemTag::Shaders,
        MemTag::Audio,
        MemTag::Physics,
        MemTag::Ui,
        MemTag::Scratch,
        MemTag::Other,
    ];

    /// The tag's position in a per-tag table.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// How a readout names the tag.
    pub const fn name(self) -> &'static str {
        match self {
            MemTag::Textures => "Textures",
            MemTag::Meshes => "Meshes",
            MemTag::Chunks => "Chunks",
            MemTag::Shaders => "Shaders",
            MemTag::Audio => "Audio",
            MemTag::Physics => "Physics",
            MemTag::Ui => "UI",
            MemTag::Scratch => "Scratch",
            MemTag::Other => "Other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `index` is what selects a counter out of the ledger's flat array, so the
    // discriminants must cover 0..COUNT exactly once.
    #[test]
    fn tag_indices_are_dense_and_match_their_position() {
        assert_eq!(MemTag::ALL.len(), MemTag::COUNT);
        for (i, tag) in MemTag::ALL.iter().enumerate() {
            assert_eq!(tag.index(), i);
        }
    }

    #[test]
    fn realm_indices_are_dense_and_match_their_position() {
        assert_eq!(Realm::ALL.len(), Realm::COUNT);
        for (i, realm) in Realm::ALL.iter().enumerate() {
            assert_eq!(realm.index(), i);
        }
    }

    #[test]
    fn every_tag_names_itself_distinctly() {
        for (i, a) in MemTag::ALL.iter().enumerate() {
            assert!(!a.name().is_empty());
            for b in &MemTag::ALL[i + 1..] {
                assert_ne!(a.name(), b.name());
            }
        }
    }
}
