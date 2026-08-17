// src/object_data_layout.rs
//
// Drift guard for the shared bindless object record. `GpuObjectData` is declared
// once per shading language (vulkan/shaders/object_common.glsl,
// directx/shaders/object_common.hlsl, metal/shaders/object_common.msl) and each
// fragment is spliced into its backend's passes at an `{OBJECT_DATA}` marker.
// `shaders/object_common.slang` is the fourth, spliced into every single-source
// pass that strides the buffer on all three backends at once, so it is checked
// alongside them.
//
// Only one backend compiles per build, so the fragments are checked here as
// plain text rather than through any backend module: a macOS build still catches
// an HLSL edit that would misread the record on a Windows GPU.

use concinnity_core::gfx::render_types::GpuObjectData;
use core::mem::{offset_of, size_of};

const VULKAN_SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vulkan/shaders");
const DIRECTX_SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/directx/shaders");
const METAL_SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/shaders");
const SLANG_SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");

// What a declared member contributes to the record's byte layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Mat4,
    Vec3,
    Scalar,
}

impl Kind {
    fn size(self) -> usize {
        match self {
            Kind::Mat4 => 64,
            Kind::Vec3 => 12,
            Kind::Scalar => 4,
        }
    }
}

// A fragment plus the type spellings its language uses for the record's members.
struct Fragment {
    label: &'static str,
    source: &'static str,
    mat4: &'static str,
    vec3: &'static str,
}

const FRAGMENTS: &[Fragment] = &[
    Fragment {
        label: "object_common.glsl",
        source: include_str!("vulkan/shaders/object_common.glsl"),
        mat4: "mat4",
        vec3: "vec3",
    },
    Fragment {
        label: "object_common.hlsl",
        source: include_str!("directx/shaders/object_common.hlsl"),
        mat4: "float4x4",
        vec3: "float3",
    },
    Fragment {
        label: "object_common.msl",
        source: include_str!("metal/shaders/object_common.msl"),
        mat4: "float4x4",
        vec3: "packed_float3",
    },
    Fragment {
        label: "object_common.slang",
        source: include_str!("shaders/object_common.slang"),
        mat4: "float4x4",
        vec3: "float3",
    },
];

// One member as declared in a fragment, with the offset tight packing puts it at.
#[derive(Debug, PartialEq, Eq)]
struct Member {
    name: String,
    kind: Kind,
    offset: usize,
}

impl Fragment {
    // Members of the fragment's `GpuObjectData`, in declaration order. Panics on
    // anything it cannot account for: an unparseable fragment is a drift the
    // test must fail on, never one it may skip past.
    fn members(&self) -> Vec<Member> {
        let body = struct_body(self.source, self.label);
        let mut members = Vec::new();
        let mut offset = 0usize;
        for decl in body.split(';').map(str::trim).filter(|d| !d.is_empty()) {
            let tokens: Vec<&str> = decl.split_whitespace().collect();
            let [qualifiers @ .., ty, name] = tokens.as_slice() else {
                panic!("{}: cannot parse member declaration `{decl}`", self.label);
            };
            assert!(
                qualifiers.iter().all(|q| *q == "column_major"),
                "{}: unexpected qualifier on `{decl}`",
                self.label
            );
            let kind = if *ty == self.mat4 {
                Kind::Mat4
            } else if *ty == self.vec3 {
                Kind::Vec3
            } else if *ty == "float" || *ty == "uint" {
                Kind::Scalar
            } else {
                panic!("{}: unhandled member type `{ty}` on `{decl}`", self.label);
            };
            members.push(Member {
                name: (*name).to_string(),
                kind,
                offset,
            });
            offset += kind.size();
        }
        members
    }

    fn size(&self) -> usize {
        self.members().iter().map(|m| m.kind.size()).sum()
    }
}

// Text between the braces of the fragment's `struct GpuObjectData`, with line
// comments stripped so commented-out members can never be counted.
fn struct_body(source: &str, label: &str) -> String {
    let uncommented: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let decl = uncommented
        .find("struct GpuObjectData")
        .unwrap_or_else(|| panic!("{label}: no `struct GpuObjectData` declaration"));
    let open = uncommented[decl..]
        .find('{')
        .unwrap_or_else(|| panic!("{label}: `struct GpuObjectData` has no body"))
        + decl;
    let close = uncommented[open..]
        .find('}')
        .unwrap_or_else(|| panic!("{label}: `struct GpuObjectData` body is unterminated"))
        + open;
    uncommented[open + 1..close].to_string()
}

// The Rust record's members, expanded so the three-float `_pad` tail lines up
// with the individual pad scalars the shading languages declare.
fn rust_members() -> Vec<Member> {
    let pad = offset_of!(GpuObjectData, _pad);
    let mut members: Vec<Member> = [
        ("model", Kind::Mat4, offset_of!(GpuObjectData, model)),
        ("tint", Kind::Vec3, offset_of!(GpuObjectData, tint)),
        (
            "roughness",
            Kind::Scalar,
            offset_of!(GpuObjectData, roughness),
        ),
        ("emissive", Kind::Vec3, offset_of!(GpuObjectData, emissive)),
        (
            "metallic",
            Kind::Scalar,
            offset_of!(GpuObjectData, metallic),
        ),
        (
            "albedo_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, albedo_index),
        ),
        (
            "normal_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, normal_index),
        ),
        (
            "macro_variation",
            Kind::Scalar,
            offset_of!(GpuObjectData, macro_variation),
        ),
        (
            "terrain_blend",
            Kind::Scalar,
            offset_of!(GpuObjectData, terrain_blend),
        ),
        ("bb_min", Kind::Vec3, offset_of!(GpuObjectData, bb_min)),
        (
            "cull_distance",
            Kind::Scalar,
            offset_of!(GpuObjectData, cull_distance),
        ),
        ("bb_max", Kind::Vec3, offset_of!(GpuObjectData, bb_max)),
        (
            "secondary_blend_sharpness",
            Kind::Scalar,
            offset_of!(GpuObjectData, secondary_blend_sharpness),
        ),
        (
            "albedo_secondary_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, albedo_secondary_index),
        ),
        (
            "normal_secondary_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, normal_secondary_index),
        ),
        (
            "emissive_map_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, emissive_map_index),
        ),
        (
            "orm_map_index",
            Kind::Scalar,
            offset_of!(GpuObjectData, orm_map_index),
        ),
        (
            "alpha_cutoff",
            Kind::Scalar,
            offset_of!(GpuObjectData, alpha_cutoff),
        ),
        ("_pad0", Kind::Scalar, pad),
        ("_pad1", Kind::Scalar, pad + 4),
        ("_pad2", Kind::Scalar, pad + 8),
    ]
    .into_iter()
    .map(|(name, kind, offset)| Member {
        name: name.to_string(),
        kind,
        offset,
    })
    .collect();
    members.sort_by_key(|m| m.offset);
    members
}

// Every fragment declares the same members, in the same order, at the same
// offsets as the Rust record the CPU uploads. A reordered, retyped, added or
// dropped member fails here rather than as garbage sampling on one backend.
#[test]
fn fragments_match_the_rust_record() {
    let expected = rust_members();
    for fragment in FRAGMENTS {
        assert_eq!(
            fragment.members(),
            expected,
            "{} drifted from the Rust GpuObjectData",
            fragment.label
        );
        assert_eq!(
            fragment.size(),
            size_of::<GpuObjectData>(),
            "{} strides at a different size",
            fragment.label
        );
    }
}

// The packing rules that make the tight Rust layout legal on the GPU. A matrix
// is 16-byte aligned in every one of these languages, and a vector may not
// straddle a 16-byte boundary; the record's size must be a multiple of 16 so an
// array of them keeps every matrix aligned.
#[test]
fn fragments_satisfy_gpu_packing_rules() {
    for fragment in FRAGMENTS {
        assert_eq!(
            fragment.size() % 16,
            0,
            "{}: size is not a multiple of 16",
            fragment.label
        );
        for member in fragment.members() {
            match member.kind {
                Kind::Mat4 => assert_eq!(
                    member.offset % 16,
                    0,
                    "{}: `{}` is a matrix at an unaligned offset {}",
                    fragment.label,
                    member.name,
                    member.offset
                ),
                Kind::Vec3 => assert_eq!(
                    member.offset / 16,
                    (member.offset + member.kind.size() - 1) / 16,
                    "{}: `{}` straddles a 16-byte boundary at offset {}",
                    fragment.label,
                    member.name,
                    member.offset
                ),
                Kind::Scalar => {}
            }
        }
    }
}

// std430 aligns `vec3` to 16 bytes, so a GLSL member that lands elsewhere would
// be silently padded forward and desync the whole record from the Rust upload.
#[test]
fn glsl_vectors_sit_on_std430_boundaries() {
    let glsl = &FRAGMENTS[0];
    assert_eq!(glsl.label, "object_common.glsl");
    for member in glsl.members().iter().filter(|m| m.kind == Kind::Vec3) {
        assert_eq!(
            member.offset % 16,
            0,
            "{}: `{}` is a vec3 at std430-unaligned offset {}",
            glsl.label,
            member.name,
            member.offset
        );
    }
}

// A bare MSL `float3` is 16-byte aligned, which would push `roughness` off
// offset 76 and misread every member after it.
#[test]
fn msl_three_float_members_are_packed() {
    let msl = &FRAGMENTS[2];
    assert_eq!(msl.label, "object_common.msl");
    for decl in struct_body(msl.source, msl.label)
        .split(';')
        .map(str::trim)
        .filter(|d| !d.is_empty())
    {
        assert!(
            !decl.starts_with("float3 "),
            "{}: `{decl}` must be packed_float3",
            msl.label
        );
    }
}

// The fragments are the only declaration of the record. A shader that grows its
// own copy is exactly the drift the splice exists to prevent, so it fails here
// even on a build whose backend never compiles that shader.
#[test]
fn no_shader_redeclares_the_record() {
    let fragments = [
        "object_common.glsl",
        "object_common.hlsl",
        "object_common.msl",
        "object_common.slang",
    ];
    let mut declarations = 0usize;
    for dir in [
        VULKAN_SHADERS,
        DIRECTX_SHADERS,
        METAL_SHADERS,
        SLANG_SHADERS,
    ] {
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {dir}: {e}")) {
            let path = entry.expect("dir entry").path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .expect("utf8 shader filename")
                .to_string();
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            if !source.contains("struct GpuObjectData") {
                continue;
            }
            assert!(
                fragments.contains(&name.as_str()),
                "{name} declares its own GpuObjectData; splice the shared fragment instead"
            );
            declarations += 1;
        }
    }
    assert_eq!(declarations, fragments.len(), "a fragment went missing");
}

// Each backend's shader directory still splices the record somewhere: a refactor
// that dropped every marker would otherwise leave the fragments orphaned and the
// passes silently undeclared.
#[test]
fn every_backend_splices_the_record() {
    for dir in [VULKAN_SHADERS, DIRECTX_SHADERS, METAL_SHADERS] {
        let spliced = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("read {dir}: {e}"))
            .filter(|entry| {
                let path = entry.as_ref().expect("dir entry").path();
                std::fs::read_to_string(&path).is_ok_and(|source| source.contains("{OBJECT_DATA}"))
            })
            .count();
        assert!(
            spliced > 0,
            "{dir}: no shader carries the OBJECT_DATA marker"
        );
    }
}
