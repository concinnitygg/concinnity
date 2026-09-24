//! What each backend compiles from the single-source shaders.
//!
//! Declarations only: the file, the entry point, and the variant gates, as
//! plain `&'static` data. They live here rather than in the device crate
//! because both halves of the toolchain iterate them -- the renderer to compile
//! a program at init, and the device build script to compile the same programs
//! ahead of time -- and a build script cannot read its own crate's data. One
//! table is what keeps the two from disagreeing about what a program's source
//! is, which the content-addressed shader cache would otherwise paper over by
//! serving one path's bytes to the other.
//!
//! A program's stage is not declared here: the entry point's `[shader("...")]`
//! attribute states it. Neither is the backend define, which the source
//! assembler adds for whichever platform a program compiles for.
//!
//! Everything that needs a compiler, a cache, or a filesystem stays in the
//! device crate and reaches these through a trait.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::platform::Platform;

/// What the DirectX backend compiles to DXIL.
pub mod dx;

/// What the Metal backend compiles to metallibs.
pub mod metal;

/// What every backend compiles for the raymarched SDF volume pass.
pub mod raymarch;

/// The engine programs every backend compiles.
pub mod shared;

/// The single-pass Hi-Z downsampler Vulkan and DirectX build the depth pyramid
/// with.
pub mod spd;

/// What every backend compiles for a world `Shader`.
pub mod surface;

/// What the Vulkan backend compiles to SPIR-V.
pub mod vk;

#[cfg(test)]
mod declared;

/// One engine program: which shader file, which entry point, under which
/// variant gates.
pub struct ShaderProgram {
    /// File name under `src/render/shaders/`, also the embedded fallback's origin and
    /// the name compiler diagnostics use.
    pub file: &'static str,
    /// Entry point compiled out of that file.
    pub entry: &'static str,
    /// Lookup key for the precompiled artifact, and the diagnostic label.
    pub label: &'static str,
    /// Fixed variant gates, each injected as `#define <gate> 1`. More than one
    /// where a variant is the intersection of two, like the textured
    /// ray-traced glass fragment.
    pub gates: &'static [&'static str],
    /// Reads the main pass's depth, so it declares that source by the main
    /// pass's sample count and compiles under `USE_MSAA` once per sample count
    /// the backend renders at.
    pub msaa: bool,
}

impl ShaderProgram {
    /// This program as a host whose main pass is (`msaa`) or is not
    /// multisampled compiles it. A program that does not read the sample count
    /// has one variant, whatever the host.
    pub const fn at(&self, msaa: bool) -> Variant<'_> {
        Variant {
            program: self,
            msaa: msaa && self.msaa,
        }
    }
}

/// One program at one main-pass sample count: the unit a backend compiles and
/// files an artifact under.
#[derive(Clone, Copy)]
pub struct Variant<'a> {
    /// The declaration.
    pub program: &'a ShaderProgram,
    /// Compiled against the multisampled main-pass depth. Only ever set on a
    /// program that reads the sample count.
    pub msaa: bool,
}

impl Variant<'_> {
    /// The variant defines: each gate as `1`, then `USE_MSAA` for a program
    /// that reads the sample count.
    pub fn defines(&self) -> Vec<(&'static str, &'static str)> {
        let mut defines: Vec<(&'static str, &'static str)> =
            self.program.gates.iter().map(|g| (*g, "1")).collect();
        if self.program.msaa {
            defines.push(("USE_MSAA", if self.msaa { "1" } else { "0" }));
        }
        defines
    }

    /// The exact source text this variant compiles for `platform`, from the
    /// embedded shaders alone.
    pub fn assemble(&self, platform: Platform) -> String {
        crate::render::shader_source::assemble(self.program.file, platform, &self.defines())
    }

    /// The key the precompiled artifact is filed under. The multisampled
    /// variant of a program is a second artifact from the same label, since
    /// handing one to the other would sample the wrong depth resource.
    pub fn artifact_name(&self) -> String {
        if self.msaa {
            format!("{}.msaa", self.program.label)
        } else {
            String::from(self.program.label)
        }
    }
}

/// Everything one backend compiles, which the renderer and the build script
/// both iterate: one compiles it at init, the other ahead of time.
pub struct Table {
    /// The program lists the backend draws from.
    pub programs: &'static [&'static [&'static ShaderProgram]],
    /// The main-pass sample counts a program that reads one compiles for, as
    /// multisampled or not.
    pub msaa: &'static [bool],
}

impl Table {
    /// Every declared program.
    pub fn programs(&self) -> impl Iterator<Item = &'static ShaderProgram> {
        self.programs.iter().flat_map(|list| list.iter().copied())
    }

    /// Every variant the backend compiles: each program once, and once per
    /// sample count where it reads one.
    pub fn variants(&self) -> impl Iterator<Item = Variant<'static>> {
        let msaa = self.msaa;
        self.programs().flat_map(move |program| {
            let counts: &[bool] = if program.msaa { msaa } else { &[false] };
            counts.iter().map(move |m| program.at(*m))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::declared::{Row, assert_rows_are_sound};
    use super::*;

    // Every variant each backend compiles, as the build script and the renderer
    // expand it, so an artifact name is unique per backend at every sample count.
    #[test]
    fn every_backend_table_is_sound() {
        for (platform, table) in [
            (Platform::DirectX, &dx::TABLE),
            (Platform::Vulkan, &vk::TABLE),
            (Platform::Metal, &metal::TABLE),
        ] {
            let rows: Vec<Row> = table
                .variants()
                .map(|v| {
                    let defines = v.defines();
                    Row {
                        label: v.artifact_name(),
                        file: v.program.file,
                        entry: v.program.entry,
                        source: v.assemble(platform),
                        defines,
                    }
                })
                .collect();
            assert_rows_are_sound(&format!("{platform:?}"), &rows);
        }
    }

    static READS_DEPTH: ShaderProgram = ShaderProgram {
        file: "glass.hlsl",
        entry: "glass_rt_fragment",
        label: "glass_frag_rt.hlsl",
        gates: &["GLASS_RT", "RT_TEXTURED"],
        msaa: true,
    };
    static PLAIN: ShaderProgram = ShaderProgram {
        file: "glass.hlsl",
        entry: "glass_vertex",
        label: "glass_vert.hlsl",
        gates: &[],
        msaa: false,
    };

    #[test]
    fn the_gates_lead_and_the_sample_count_follows() {
        assert!(PLAIN.at(true).defines().is_empty());
        assert_eq!(
            READS_DEPTH.at(false).defines(),
            [("GLASS_RT", "1"), ("RT_TEXTURED", "1"), ("USE_MSAA", "0")]
        );
        assert_eq!(
            READS_DEPTH.at(true).defines(),
            [("GLASS_RT", "1"), ("RT_TEXTURED", "1"), ("USE_MSAA", "1")]
        );
    }

    // Only a program that reads the sample count has a second artifact; keying
    // the rest on it would miss the single one they do have.
    #[test]
    fn only_a_sample_count_reader_has_an_msaa_artifact() {
        assert_eq!(READS_DEPTH.at(false).artifact_name(), "glass_frag_rt.hlsl");
        assert_eq!(
            READS_DEPTH.at(true).artifact_name(),
            "glass_frag_rt.hlsl.msaa"
        );
        assert_eq!(PLAIN.at(true).artifact_name(), "glass_vert.hlsl");
        assert!(!PLAIN.at(true).msaa);
    }

    #[test]
    fn a_table_expands_only_the_sample_count_readers() {
        static BOTH: Table = Table {
            programs: &[&[&READS_DEPTH, &PLAIN]],
            msaa: &[false, true],
        };
        static SINGLE: Table = Table {
            programs: &[&[&READS_DEPTH], &[&PLAIN]],
            msaa: &[false],
        };
        let names: Vec<String> = BOTH.variants().map(|v| v.artifact_name()).collect();
        assert_eq!(
            names,
            [
                "glass_frag_rt.hlsl",
                "glass_frag_rt.hlsl.msaa",
                "glass_vert.hlsl"
            ]
        );
        let names: Vec<String> = SINGLE.variants().map(|v| v.artifact_name()).collect();
        assert_eq!(names, ["glass_frag_rt.hlsl", "glass_vert.hlsl"]);
        assert_eq!(SINGLE.programs().count(), 2);
    }
}
