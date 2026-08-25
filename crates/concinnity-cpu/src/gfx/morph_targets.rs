//! Sparse morph-target storage for skinned payloads.
//!
//! A morph target moves a small region of the mesh and is zero elsewhere, so
//! the payload stores only the non-zero deltas. The storage is vertex-major
//! (compressed sparse row): `offsets[v]..offsets[v + 1]` is the run of
//! [`MorphEntry`]s touching vertex `v`, each naming its target. That is the
//! order the deform kernels want: one vertex reads its own run and skips every
//! target that does not move it.
//!
//! The GPU consumes both tables through one buffer, see [`PayloadMorphs::packed_words`].

/// One morph-target vertex delta in dense form: position and normal offsets
/// added to the bind pose before skinning, scaled by the target's weight.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct MorphDelta {
    /// Bind-space position offset.
    pub position: [f32; 3],
    /// Normal offset; the deformed normal is re-normalised.
    pub normal: [f32; 3],
}

/// One sparse morph entry as the GPU consumes it: the target it belongs to
/// plus the position and normal offsets. Plain tightly packed 4-byte fields;
/// the shader-side struct uses packed types so the 28-byte stride matches.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct MorphEntry {
    /// Morph target this delta belongs to.
    pub target: u32,
    /// Bind-space position offset.
    pub position: [f32; 3],
    /// Normal offset.
    pub normal: [f32; 3],
}

/// Deltas whose every component is at or below this magnitude are dropped
/// when a dense target is sparsified: a micron of position or a 1e-6 normal
/// tilt is invisible, and imported targets carry that much float noise.
pub const MORPH_DELTA_EPSILON: f32 = 1e-6;

/// Morph-target block of a skinned payload: target names plus the sparse
/// vertex-major entries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadMorphs {
    /// Morph-target names, in target order.
    pub names: Vec<String>,
    /// `vertex_count + 1` entry offsets; vertex `v` owns
    /// `entries[offsets[v]..offsets[v + 1]]`. Empty when there are no targets.
    pub offsets: Vec<u32>,
    /// Sparse entries, grouped by vertex, targets ascending within a vertex.
    pub entries: Vec<MorphEntry>,
}

impl PayloadMorphs {
    /// Whether the mesh declares no morph targets.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Morph targets on the mesh.
    pub fn target_count(&self) -> usize {
        self.names.len()
    }

    /// Vertices the offsets table covers (0 without targets).
    pub fn vertex_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Build the sparse block from dense target-major deltas
    /// (`deltas[t * vertex_count + v]`), keeping every delta with a component
    /// above [`MORPH_DELTA_EPSILON`].
    pub fn from_dense(
        names: Vec<String>,
        vertex_count: usize,
        deltas: &[MorphDelta],
    ) -> Result<Self, String> {
        if deltas.len() != names.len() * vertex_count {
            return Err(format!(
                "morph_deltas has {} entries; {} target(s) x {} vertices requires {}",
                deltas.len(),
                names.len(),
                vertex_count,
                names.len() * vertex_count,
            ));
        }
        if names.is_empty() {
            return Ok(Self::default());
        }
        let mut offsets = Vec::with_capacity(vertex_count + 1);
        let mut entries = Vec::new();
        offsets.push(0u32);
        for v in 0..vertex_count {
            for (t, name_block) in deltas.chunks_exact(vertex_count).enumerate() {
                let d = name_block[v];
                if is_significant(&d) {
                    entries.push(MorphEntry {
                        target: t as u32,
                        position: d.position,
                        normal: d.normal,
                    });
                }
            }
            offsets.push(entries.len() as u32);
        }
        Ok(Self {
            names,
            offsets,
            entries,
        })
    }

    /// Expand back to dense target-major deltas (`[t * vertex_count + v]`),
    /// with zeros wherever no entry exists.
    pub fn to_dense(&self) -> Vec<MorphDelta> {
        let n = self.vertex_count();
        let mut out = vec![MorphDelta::default(); self.target_count() * n];
        for (v, e) in self.vertex_entries() {
            out[e.target as usize * n + v] = MorphDelta {
                position: e.position,
                normal: e.normal,
            };
        }
        out
    }

    /// Every entry paired with the vertex it belongs to.
    pub(crate) fn vertex_entries(&self) -> impl Iterator<Item = (usize, &MorphEntry)> {
        self.offsets.windows(2).enumerate().flat_map(move |(v, w)| {
            self.entries[w[0] as usize..w[1] as usize]
                .iter()
                .map(move |e| (v, e))
        })
    }

    /// Check the tables agree: offsets start at 0, never decrease, end at the
    /// entry count, and every entry names a declared target.
    pub fn validate(&self) -> Result<(), String> {
        if self.is_empty() {
            if !self.offsets.is_empty() || !self.entries.is_empty() {
                return Err("morph block has entries but no targets".to_string());
            }
            return Ok(());
        }
        if self.offsets.first() != Some(&0) {
            return Err("morph offsets must start at 0".to_string());
        }
        if self.offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err("morph offsets must not decrease".to_string());
        }
        if self.offsets.last().copied().unwrap_or(0) as usize != self.entries.len() {
            return Err(format!(
                "morph offsets end at {} but there are {} entries",
                self.offsets.last().copied().unwrap_or(0),
                self.entries.len()
            ));
        }
        let targets = self.target_count() as u32;
        if let Some(e) = self.entries.iter().find(|e| e.target >= targets) {
            return Err(format!(
                "morph entry names target {} of {targets}",
                e.target
            ));
        }
        Ok(())
    }

    /// The single GPU buffer the deform kernels read: the offsets table, then
    /// the entries, which start at the first 16-byte-aligned word past it.
    /// Each entry is seven 4-byte words laid out as [`MorphEntry`].
    pub fn packed_words(&self) -> Vec<u32> {
        if self.is_empty() {
            return Vec::new();
        }
        let base = entry_word_base(self.vertex_count());
        let mut words = Vec::with_capacity(base + self.entries.len() * MORPH_ENTRY_WORDS);
        words.extend_from_slice(&self.offsets);
        words.resize(base, 0);
        words.extend_from_slice(bytemuck::cast_slice::<MorphEntry, u32>(&self.entries));
        words
    }
}

/// Words per [`MorphEntry`] in the packed buffer.
pub(crate) const MORPH_ENTRY_WORDS: usize = 7;

/// Word index where the entries begin in [`PayloadMorphs::packed_words`]: the
/// `vertex_count + 1` offsets rounded up to a 16-byte boundary. The shaders
/// compute the same value from their `vertex_count` parameter.
pub(crate) fn entry_word_base(vertex_count: usize) -> usize {
    (vertex_count + 1 + 3) & !3
}

fn is_significant(d: &MorphDelta) -> bool {
    d.position
        .iter()
        .chain(d.normal.iter())
        .any(|x| x.abs() > MORPH_DELTA_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(p: f32) -> MorphDelta {
        MorphDelta {
            position: [p, 0.0, 0.0],
            normal: [0.0, 0.0, 0.0],
        }
    }

    fn sample() -> PayloadMorphs {
        // 2 targets x 3 vertices: target 0 moves v0 and v2, target 1 moves v2.
        let dense = vec![
            delta(1.0),
            delta(0.0),
            delta(2.0),
            delta(0.0),
            delta(0.0),
            MorphDelta {
                position: [0.0; 3],
                normal: [0.0, 0.5, 0.0],
            },
        ];
        PayloadMorphs::from_dense(vec!["a".into(), "b".into()], 3, &dense).expect("dense")
    }

    #[test]
    fn sparsifies_and_expands_to_the_same_dense_block() {
        let m = sample();
        assert_eq!(m.offsets, vec![0, 1, 1, 3]);
        assert_eq!(m.entries.len(), 3);
        assert_eq!(m.entries[1].target, 0);
        assert_eq!(m.entries[2].target, 1);
        assert_eq!(m.entries[2].normal, [0.0, 0.5, 0.0]);
        m.validate().expect("valid");
        let dense = m.to_dense();
        assert_eq!(dense.len(), 6);
        assert_eq!(dense[2], delta(2.0));
        assert_eq!(dense[5].normal, [0.0, 0.5, 0.0]);
        assert_eq!(dense[1], MorphDelta::default());
        let again = PayloadMorphs::from_dense(m.names.clone(), 3, &dense).expect("dense");
        assert_eq!(again, m);
    }

    #[test]
    fn deltas_at_the_epsilon_are_dropped_but_any_component_above_it_is_kept() {
        let dense = vec![
            MorphDelta {
                position: [MORPH_DELTA_EPSILON; 3],
                normal: [0.0; 3],
            },
            MorphDelta {
                position: [0.0; 3],
                normal: [0.0, 0.0, -MORPH_DELTA_EPSILON * 2.0],
            },
        ];
        let m = PayloadMorphs::from_dense(vec!["t".into()], 2, &dense).expect("dense");
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.offsets, vec![0, 0, 1]);
    }

    #[test]
    fn no_targets_is_the_empty_block() {
        let m = PayloadMorphs::from_dense(Vec::new(), 5, &[]).expect("dense");
        assert!(m.is_empty());
        assert_eq!(m, PayloadMorphs::default());
        assert!(m.to_dense().is_empty());
        assert!(m.packed_words().is_empty());
    }

    #[test]
    fn a_dense_block_of_the_wrong_length_is_refused() {
        let err = PayloadMorphs::from_dense(vec!["t".into()], 3, &[delta(1.0)]).unwrap_err();
        assert!(err.contains("1 target(s) x 3 vertices requires 3"), "{err}");
    }

    #[test]
    fn validate_catches_every_table_disagreement() {
        let mut m = sample();
        m.offsets[0] = 1;
        assert!(m.validate().unwrap_err().contains("start at 0"));
        let mut m = sample();
        m.offsets[2] = 0;
        assert!(m.validate().unwrap_err().contains("not decrease"));
        let mut m = sample();
        m.offsets[3] = 2;
        assert!(m.validate().unwrap_err().contains("end at 2"));
        let mut m = sample();
        m.entries[0].target = 2;
        assert!(m.validate().unwrap_err().contains("target 2 of 2"));
        let mut m = sample();
        m.names.clear();
        assert!(m.validate().unwrap_err().contains("no targets"));
    }

    #[test]
    fn packed_words_place_entries_at_the_aligned_base() {
        let m = sample();
        // 4 offsets round up to 4 words; 3 entries x 7 words follow.
        assert_eq!(entry_word_base(3), 4);
        assert_eq!(entry_word_base(4), 8);
        assert_eq!(entry_word_base(0), 4);
        let words = m.packed_words();
        assert_eq!(words.len(), 4 + 3 * MORPH_ENTRY_WORDS);
        assert_eq!(&words[..4], &[0, 1, 1, 3]);
        assert_eq!(words[4], 0, "entry 0 target");
        assert_eq!(f32::from_bits(words[5]), 1.0, "entry 0 position.x");
        assert_eq!(words[4 + 2 * MORPH_ENTRY_WORDS], 1, "entry 2 target");
        assert_eq!(
            f32::from_bits(words[4 + 2 * MORPH_ENTRY_WORDS + 5]),
            0.5,
            "entry 2 normal.y"
        );
    }

    #[test]
    fn morph_entry_layout_matches_shaders() {
        // `MorphEntry` is read through a raw pointer by the deform kernels
        // (`MorphEntry` in rt_skin.metal / main.metal, and by word offset in
        // rt_skin.comp / rt_skin.hlsl): uint target at 0, two packed float3s
        // at 4 and 16, 28-byte stride.
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<MorphEntry>(), MORPH_ENTRY_WORDS * 4);
        assert_eq!(offset_of!(MorphEntry, target), 0);
        assert_eq!(offset_of!(MorphEntry, position), 4);
        assert_eq!(offset_of!(MorphEntry, normal), 16);
    }
}
