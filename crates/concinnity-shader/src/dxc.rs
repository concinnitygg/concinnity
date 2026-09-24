//! The `dxc` invocation.
//!
//! One argument list, assembled in one place, because the build script and the
//! renderer must produce byte-identical artifacts or the content-addressed
//! shader cache serves one path's bytes to the other.

use std::path::Path;
use std::process::Command;

use crate::{Stage, Warnings, run};

/// What one dxc run emits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Emit {
    /// A signed DXIL container.
    Dxil,
    /// SPIR-V for Vulkan, with the entry point renamed to `main` -- which is
    /// what pipeline stage creation asks for.
    SpirvForVulkan,
    /// SPIR-V as an intermediate for the MSL leg, keeping the entry point's own
    /// name: the Metal host looks a function up by it.
    SpirvForMsl,
    /// SPIR-V under Vulkan layout rules, for reading struct layouts back.
    SpirvWithVulkanLayout,
    /// SPIR-V laid out the way DXIL lays its buffers out, for reading the
    /// DirectX leg's struct layouts back without a D3D12 reflection interface.
    SpirvWithDxLayout,
}

impl Emit {
    // Whether every declared resource stays in the module, read or not. The MSL
    // leg needs it because Metal lays an argument buffer out by the members a
    // function declares, and the layout legs because a block the entry never
    // reads still has a layout to check. The shipped Vulkan artifact does not
    // take it: it would name bindings the pipeline layout it is created
    // against has no reason to carry.
    fn preserves_bindings(self) -> bool {
        match self {
            Emit::Dxil | Emit::SpirvForVulkan => false,
            Emit::SpirvForMsl => true,
            Emit::SpirvWithVulkanLayout | Emit::SpirvWithDxLayout => true,
        }
    }

    // Whether an entry of `stage` keeps every input it declares, read or not.
    // A Vulkan fragment stage has to: the vertex stage writes every varying
    // of the shared block, and one the fragment dropped is an output nothing
    // consumes. A vertex stage must not, since every input it kept would need
    // a vertex attribute the pipeline feeds.
    fn preserves_interface(self, stage: Stage) -> bool {
        self == Emit::SpirvForVulkan && stage == Stage::Pixel
    }
}

// The Vulkan environment the engine's SPIR-V targets, matching what the
// backend's instance asks for.
const SPIRV_TARGET_ENV: &str = "vulkan1.2";

/// The argument list after the source path and before `-Fo`.
///
/// `-Zpc` is mandatory rather than merely explicit: the engine uploads
/// column-major matrices, and a future dxc default of row-major would transpose
/// every one of them. `-Wno-ignored-attributes` rides the DXIL leg alone, where
/// every `vk::` annotation is correctly ignored rather than wrong; a misspelled
/// one is an unknown attribute on every leg and still warns. The engine's own
/// `cn::` attributes never reach dxc (see
/// [`strip_engine_attributes`](crate::declarations::strip_engine_attributes)).
pub(crate) fn command_args(
    stage: Stage,
    profile: &str,
    entry: &str,
    emit: Emit,
    warnings: Warnings,
) -> Vec<String> {
    let mut args = vec![
        "-T".to_string(),
        profile.to_string(),
        "-E".to_string(),
        entry.to_string(),
        "-Zpc".to_string(),
    ];
    if warnings == Warnings::Deny {
        args.push("-WX".to_string());
    }
    if emit == Emit::Dxil {
        args.push("-Wno-ignored-attributes".to_string());
    }
    if emit != Emit::Dxil {
        args.push("-spirv".to_string());
        args.push(format!("-fspv-target-env={SPIRV_TARGET_ENV}"));
    }
    if emit == Emit::SpirvForVulkan {
        args.push("-fspv-entrypoint-name=main".to_string());
    }
    if emit == Emit::SpirvWithDxLayout {
        args.push("-fvk-use-dx-layout".to_string());
    }
    if emit.preserves_bindings() {
        args.push("-fspv-preserve-bindings".to_string());
    }
    if emit.preserves_interface(stage) {
        args.push("-fspv-preserve-interface".to_string());
    }
    args
}

/// Compile `src_name` (already written under `scratch`) into `out_name`,
/// returning what dxc printed: its warnings, when the compile succeeded.
pub(crate) fn compile(
    dxc: &Path,
    scratch: &Path,
    src_name: &str,
    out_name: &str,
    args: &[String],
) -> Result<String, String> {
    run(
        Command::new(dxc)
            .current_dir(scratch)
            .arg(src_name)
            .args(args)
            .arg("-Fo")
            .arg(out_name),
        "dxc",
        src_name,
    )
}

/// Preprocess `src_name` into `out_name`, which is the text every later step
/// reads: a declaration behind an inactive `#if` is not a declaration.
pub(crate) fn preprocess(
    dxc: &Path,
    scratch: &Path,
    src_name: &str,
    out_name: &str,
) -> Result<(), String> {
    run(
        Command::new(dxc)
            .current_dir(scratch)
            .arg(src_name)
            .arg("-P")
            .arg("-Fi")
            .arg(out_name),
        "dxc",
        src_name,
    )
    .map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dxil_leg_asks_for_no_spirv_and_keeps_the_entry_name() {
        let args = command_args(
            Stage::Pixel,
            "ps_6_0",
            "text_fragment_main",
            Emit::Dxil,
            Warnings::Deny,
        );
        assert_eq!(args[..4], ["-T", "ps_6_0", "-E", "text_fragment_main"]);
        assert!(!args.iter().any(|a| a == "-spirv"));
        assert!(!args.iter().any(|a| a.contains("entrypoint-name")));
    }

    // Vulkan pipeline stage creation asks for `main`.
    #[test]
    fn the_vulkan_leg_renames_the_entry_point_to_main() {
        let args = command_args(
            Stage::Vertex,
            "vs_6_0",
            "text_vertex_main",
            Emit::SpirvForVulkan,
            Warnings::Deny,
        );
        assert!(args.iter().any(|a| a == "-spirv"));
        assert!(args.iter().any(|a| a == "-fspv-entrypoint-name=main"));
    }

    // The Metal host looks a function up by name, so the MSL leg's intermediate
    // must not be renamed.
    #[test]
    fn the_msl_leg_emits_spirv_under_the_entry_points_own_name() {
        let args = command_args(
            Stage::Pixel,
            "ps_6_0",
            "text_fragment_main",
            Emit::SpirvForMsl,
            Warnings::Deny,
        );
        assert!(args.iter().any(|a| a == "-spirv"));
        assert!(!args.iter().any(|a| a.contains("entrypoint-name")));
    }

    // A row-major default would transpose every matrix the engine uploads.
    #[test]
    fn every_leg_pins_the_matrix_layout() {
        for emit in EVERY_EMIT {
            assert!(asks(emit, "-Zpc"));
        }
    }

    // Only the reading of the DirectX layout asks for DirectX packing: the
    // Vulkan artifact keeps the layout a Vulkan driver expects.
    #[test]
    fn only_the_dx_layout_leg_asks_for_dx_packing() {
        let dx_layout = |emit| {
            command_args(Stage::Pixel, "ps_6_0", "f", emit, Warnings::Deny)
                .iter()
                .any(|a| a == "-fvk-use-dx-layout")
        };
        assert!(dx_layout(Emit::SpirvWithDxLayout));
        assert!(!dx_layout(Emit::SpirvForVulkan));
    }

    fn asks(emit: Emit, flag: &str) -> bool {
        asks_under(emit, Warnings::Deny, flag)
    }

    fn asks_under(emit: Emit, warnings: Warnings, flag: &str) -> bool {
        command_args(Stage::Pixel, "ps_6_0", "f", emit, warnings)
            .iter()
            .any(|a| a == flag)
    }

    const EVERY_EMIT: [Emit; 5] = [
        Emit::Dxil,
        Emit::SpirvForVulkan,
        Emit::SpirvForMsl,
        Emit::SpirvWithVulkanLayout,
        Emit::SpirvWithDxLayout,
    ];

    // A `vk::` annotation is ignored on the DXIL leg by design, and anywhere
    // else an ignored one is misplaced, so only DXIL silences the warning; no
    // leg silences an unknown attribute, which is what a misspelling is.
    #[test]
    fn only_the_dxil_leg_silences_ignored_attributes() {
        assert!(asks(Emit::Dxil, "-Wno-ignored-attributes"));
        for emit in EVERY_EMIT {
            assert!(!asks(emit, "-Wno-unknown-attributes"));
            if emit != Emit::Dxil {
                assert!(!asks(emit, "-Wno-ignored-attributes"));
            }
        }
    }

    #[test]
    fn a_denied_warning_is_an_error_and_a_reported_one_is_not() {
        for emit in EVERY_EMIT {
            assert!(asks_under(emit, Warnings::Deny, "-WX"));
            assert!(!asks_under(emit, Warnings::Report, "-WX"));
        }
    }

    // Metal lays an argument buffer out by the members its function declares,
    // so the MSL leg keeps the members an entry never reads; the layout legs
    // keep the blocks it never reads. The shipped artifacts keep only what the
    // entry uses, which is what their pipeline layouts are validated against.
    #[test]
    fn only_the_msl_and_layout_legs_preserve_unread_bindings() {
        let flag = "-fspv-preserve-bindings";
        assert!(asks(Emit::SpirvForMsl, flag));
        assert!(asks(Emit::SpirvWithVulkanLayout, flag));
        assert!(asks(Emit::SpirvWithDxLayout, flag));
        assert!(!asks(Emit::SpirvForVulkan, flag));
        assert!(!asks(Emit::Dxil, flag));
    }

    // Only a shipped Vulkan fragment keeps its unread inputs: Metal links a
    // fragment's inputs to its vertex by attribute, and a vertex stage's kept
    // inputs would each need an attribute the pipeline feeds.
    #[test]
    fn only_the_vulkan_fragment_preserves_its_interface() {
        let preserves = |stage, emit| {
            command_args(stage, "ps_6_0", "f", emit, Warnings::Deny)
                .iter()
                .any(|a| a == "-fspv-preserve-interface")
        };
        assert!(preserves(Stage::Pixel, Emit::SpirvForVulkan));
        assert!(!preserves(Stage::Vertex, Emit::SpirvForVulkan));
        assert!(!preserves(Stage::Compute, Emit::SpirvForVulkan));
        for emit in EVERY_EMIT {
            assert_eq!(preserves(Stage::Pixel, emit), emit == Emit::SpirvForVulkan);
        }
    }
}
