// src/shader_layout/byte_offsets.rs
//
// The layout lock for a kernel that byte-addresses its buffers.
//
// Reflection cannot reach these. `slangc -reflection-json` reports the layout of
// a struct a shader declares, and a `ByteAddressBuffer` declares none: the
// strides and field offsets live in the kernel as plain constants. So the check
// reads those constants back out of the same `.slang` text the renderer
// compiles and compares them to the `#[repr(C)]` mirror, which locks both sides
// the way the reflected mirrors do.
//
// A byte-addressed buffer is not a workaround here, it is the only shape that
// survives all three targets: a structured-buffer `float3` packs to 12 bytes on
// Metal and DXIL but pads to 16 on SPIR-V, which would stride `SkinnedVertex` at
// 96 instead of 80.

// Value of a `static const uint <name> = <value>;` declaration in `source`.
// Panics when the constant is missing or unparsable, because a renamed constant
// is exactly the drift this check exists to catch -- silently skipping it would
// leave the mirror unguarded.
pub(super) fn shader_const(source: &str, name: &str) -> usize {
    let decl = format!("static const uint {name} = ");
    let rest = source
        .split_once(&decl)
        .unwrap_or_else(|| panic!("shader declares no `{decl}...`"))
        .1;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits
        .parse()
        .unwrap_or_else(|e| panic!("`{name}` is not a uint literal: {e}"))
}

#[cfg(test)]
mod tests {
    use super::shader_const;
    use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
    use crate::gfx::morph_targets::MorphEntry;
    use concinnity_core::render::shaders::RT_SKIN;
    use std::mem::{offset_of, size_of};

    // Every offset the skin kernel reads or writes a mesh payload at, against
    // the Rust struct the CPU fills. The kernel walks three payloads: the
    // bind-pose vertices it reads, the deformed vertices it writes (which the
    // trace's fetchers read back at the same offsets), and the sparse morph
    // deltas it folds in before the skin matrix.
    #[test]
    fn mesh_payload_offsets_match_the_kernel() {
        let s = RT_SKIN;

        assert_eq!(
            shader_const(s, "SKINNED_STRIDE"),
            size_of::<SkinnedVertex>()
        );
        assert_eq!(
            shader_const(s, "SKINNED_POS"),
            offset_of!(SkinnedVertex, pos)
        );
        assert_eq!(
            shader_const(s, "SKINNED_NORMAL"),
            offset_of!(SkinnedVertex, normal)
        );
        assert_eq!(
            shader_const(s, "SKINNED_TANGENT"),
            offset_of!(SkinnedVertex, tangent)
        );
        assert_eq!(
            shader_const(s, "SKINNED_COLOR"),
            offset_of!(SkinnedVertex, color)
        );
        assert_eq!(shader_const(s, "SKINNED_UV"), offset_of!(SkinnedVertex, uv));
        assert_eq!(
            shader_const(s, "SKINNED_JOINTS"),
            offset_of!(SkinnedVertex, joints)
        );
        assert_eq!(
            shader_const(s, "SKINNED_WEIGHTS"),
            offset_of!(SkinnedVertex, weights)
        );

        assert_eq!(shader_const(s, "VERTEX_STRIDE"), size_of::<Vertex>());
        assert_eq!(shader_const(s, "VERTEX_POS"), offset_of!(Vertex, pos));
        assert_eq!(shader_const(s, "VERTEX_NORMAL"), offset_of!(Vertex, normal));
        assert_eq!(
            shader_const(s, "VERTEX_TANGENT"),
            offset_of!(Vertex, tangent)
        );
        assert_eq!(shader_const(s, "VERTEX_COLOR"), offset_of!(Vertex, color));
        assert_eq!(shader_const(s, "VERTEX_UV"), offset_of!(Vertex, uv));

        assert_eq!(shader_const(s, "MORPH_STRIDE"), size_of::<MorphEntry>());
        assert_eq!(
            shader_const(s, "MORPH_TARGET"),
            offset_of!(MorphEntry, target)
        );
        assert_eq!(
            shader_const(s, "MORPH_POSITION"),
            offset_of!(MorphEntry, position)
        );
        assert_eq!(
            shader_const(s, "MORPH_NORMAL"),
            offset_of!(MorphEntry, normal)
        );
    }

    // The kernel reads the four u16 joint indices as two uints, so the pair must
    // tile the Rust field exactly.
    #[test]
    fn joint_indices_tile_the_two_words_the_kernel_reads() {
        assert_eq!(size_of::<[u16; 4]>(), 2 * size_of::<u32>());
        assert_eq!(
            offset_of!(SkinnedVertex, weights) - offset_of!(SkinnedVertex, joints),
            size_of::<[u16; 4]>()
        );
    }

    #[test]
    fn a_missing_constant_is_a_failure_not_a_skip() {
        let err = std::panic::catch_unwind(|| shader_const("static const uint A = 1;", "B"));
        assert!(err.is_err());
    }
}
