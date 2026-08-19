// src/shader_layout/mirror.rs
//
// The Rust half of a shader-struct layout check, and the comparison itself.
//
// A `#[repr(C)]` mirror rarely names its fields the way the shader does: the
// shader packs `cam_pos` and `z_near` into one `float4` lane because MSL sizes a
// constant-buffer `float3` at 16 bytes, while the CPU keeps them as separate
// members. So a mirror is described as a list of lanes, each pairing a run of
// Rust fields with the run of shader fields covering the same bytes. Both runs
// must start at the same offset and span the same number of bytes; the Rust
// lanes together must tile the whole struct.
//
// Nothing here is a hand-written number. Every expected value comes from
// slangc's reflection of the shader that actually compiles.

use crate::shader_layout::programs::Target;
use crate::shader_layout::reflect::ShaderStruct;

// One Rust field: where `#[repr(C)]` puts it and how wide it is.
pub(super) struct RustField {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
}

// A run of Rust fields and the run of shader fields occupying the same bytes.
// An empty `shader` run marks Rust bytes this declaration does not cover -- tail
// padding, or a field a partial view leaves out, like the bloom prefilter's
// six-float prefix of the nine-float post block. Those bytes must lie beyond
// every field the shader does declare.
pub(super) struct Lane {
    pub rust: Vec<RustField>,
    pub shader: Vec<&'static str>,
}

// One `#[repr(C)]` struct and the shader struct it must match.
pub(super) struct Mirror {
    pub rust_name: &'static str,
    pub rust_size: usize,
    pub shader_name: &'static str,
    pub lanes: Vec<Lane>,
}

// A Rust field's name, offset and width, read from the type itself. The name may
// be a path (`post.exposure`) so a mirror can reach into a nested block the
// shader declares flat.
macro_rules! rust_field {
    ($ty:ty, $($field:ident).+) => {{
        fn width<F>(_: fn(&$ty) -> &F) -> usize {
            ::core::mem::size_of::<F>()
        }
        $crate::shader_layout::mirror::RustField {
            name: stringify!($($field).+),
            offset: ::core::mem::offset_of!($ty, $($field).+),
            size: width(|v| &v.$($field).+),
        }
    }};
}

// The lane list. `field,` is the shorthand for a lane whose Rust and shader
// names match; `[a, b] => ["x"],` spells out a packed lane, and `[p] => [],`
// marks Rust-only tail padding.
macro_rules! lanes {
    ($out:ident, $ty:ty,) => {};
    ($out:ident, $ty:ty, [$($($r:ident).+),+] => [$($s:literal),*], $($rest:tt)*) => {
        $out.push($crate::shader_layout::mirror::Lane {
            rust: vec![$($crate::shader_layout::mirror::rust_field!($ty, $($r).+)),+],
            shader: vec![$($s),*],
        });
        $crate::shader_layout::mirror::lanes!($out, $ty, $($rest)*);
    };
    ($out:ident, $ty:ty, $r:ident, $($rest:tt)*) => {
        $out.push($crate::shader_layout::mirror::Lane {
            rust: vec![$crate::shader_layout::mirror::rust_field!($ty, $r)],
            shader: vec![stringify!($r)],
        });
        $crate::shader_layout::mirror::lanes!($out, $ty, $($rest)*);
    };
}

// One mirror: the Rust type, the shader struct name, and the lane list.
macro_rules! mirror {
    ($ty:ty => $shader:literal { $($lanes:tt)* }) => {{
        let mut lanes = Vec::new();
        $crate::shader_layout::mirror::lanes!(lanes, $ty, $($lanes)*);
        $crate::shader_layout::mirror::Mirror {
            rust_name: stringify!($ty),
            rust_size: ::core::mem::size_of::<$ty>(),
            shader_name: $shader,
            lanes,
        }
    }};
}

pub(super) use {lanes, mirror, rust_field};

// Every way `mirror` disagrees with the layout slangc gave `shader`. Empty means
// the CPU struct and the compiled shader struct describe the same bytes.
pub(super) fn drift(mirror: &Mirror, shader: &ShaderStruct) -> Vec<String> {
    let mut out = Vec::new();
    check_tiling(mirror, &mut out);
    check_lanes(mirror, shader, &mut out);
    check_coverage(mirror, shader, &mut out);
    check_size(mirror, shader, &mut out);
    out
}

// The Rust fields the lanes name must tile the struct: no gap (an unlisted
// field, so an unchecked one), no overlap, nothing past the end.
fn check_tiling(mirror: &Mirror, out: &mut Vec<String>) {
    let mut cursor = 0;
    for field in mirror.lanes.iter().flat_map(|lane| &lane.rust) {
        if field.offset != cursor {
            out.push(format!(
                "{}.{} sits at {} but the previous field ends at {cursor}: the lane list \
                 must name every field in order",
                mirror.rust_name, field.name, field.offset,
            ));
        }
        cursor = field.offset + field.size;
    }
    if cursor != mirror.rust_size {
        out.push(format!(
            "{} is {} bytes but its lanes account for {cursor}",
            mirror.rust_name, mirror.rust_size,
        ));
    }
}

// Each lane's Rust run and shader run must start together and span the same
// bytes. This is where a shader-side move or resize surfaces.
fn check_lanes(mirror: &Mirror, shader: &ShaderStruct, out: &mut Vec<String>) {
    let declared = shader.extent();
    for lane in &mirror.lanes {
        let rust_offset = lane.rust[0].offset;
        let rust_size: usize = lane.rust.iter().map(|f| f.size).sum();
        let names = lane
            .rust
            .iter()
            .map(|f| f.name)
            .collect::<Vec<_>>()
            .join(" + ");
        let Some(first) = lane.shader.first() else {
            if rust_offset < declared {
                out.push(format!(
                    "{}.{names} is marked as bytes the shader does not cover, but {} declares \
                     members through byte {declared}",
                    mirror.rust_name, mirror.shader_name,
                ));
            }
            continue;
        };
        let mut shader_offset = None;
        let mut shader_size = 0;
        let mut cursor = None;
        for name in &lane.shader {
            let Some(field) = shader.fields.iter().find(|f| f.name == *name) else {
                out.push(format!(
                    "{}.{names} maps to `{name}`, which {} does not declare",
                    mirror.rust_name, mirror.shader_name,
                ));
                continue;
            };
            if let Some(end) = cursor
                && field.offset != end
            {
                out.push(format!(
                    "{} members `{first}`..`{name}` are not contiguous: `{name}` sits at {} \
                     after byte {end}",
                    mirror.shader_name, field.offset,
                ));
            }
            shader_offset.get_or_insert(field.offset);
            shader_size += field.size;
            cursor = Some(field.offset + field.size);
        }
        let Some(shader_offset) = shader_offset else {
            continue;
        };
        if shader_offset != rust_offset || shader_size != rust_size {
            out.push(format!(
                "{}.{names} covers bytes {rust_offset}..{} but {}.{} covers {shader_offset}..{}",
                mirror.rust_name,
                rust_offset + rust_size,
                mirror.shader_name,
                lane.shader.join(" + "),
                shader_offset + shader_size,
            ));
        }
    }
}

// Every shader member must be claimed by a lane, or a field added shader-side
// would go unchecked.
fn check_coverage(mirror: &Mirror, shader: &ShaderStruct, out: &mut Vec<String>) {
    for field in &shader.fields {
        if !mirror
            .lanes
            .iter()
            .any(|lane| lane.shader.contains(&field.name.as_str()))
        {
            out.push(format!(
                "{}.{} is declared in the shader but no {} field claims it",
                mirror.shader_name, field.name, mirror.rust_name,
            ));
        }
    }
}

// The CPU upload has to cover the block the shader binds. Two shapes pass: the
// sizes match, or the shader stops at its declared extent -- because the
// DirectX leg reports the block unrounded, or because the declaration is a
// partial view -- and the Rust struct's uncovered tail makes up the difference.
fn check_size(mirror: &Mirror, shader: &ShaderStruct, out: &mut Vec<String>) {
    let Some(block) = shader.block_size else {
        return;
    };
    let rounds_up = block == shader.extent() && mirror.rust_size > block;
    if mirror.rust_size != block && !rounds_up {
        out.push(format!(
            "{} is {} bytes but the shader binds {} as a {block}-byte block",
            mirror.rust_name, mirror.rust_size, mirror.shader_name,
        ));
    }
}

// One mirror and the targets whose reflection declares its shader struct. All
// three unless the declaration sits behind one backend's host-shape gate: the
// G-buffer model pair is a constant buffer on Metal and DirectX and a push
// constant on Vulkan, so each leg mirrors a different struct.
pub(super) struct Case {
    pub mirror: Mirror,
    pub targets: &'static [Target],
}

// A mirror every backend declares.
pub(super) fn everywhere(mirror: Mirror) -> Case {
    Case {
        mirror,
        targets: &Target::ALL,
    }
}

// A mirror only some backends declare.
pub(super) fn on(targets: &'static [Target], mirror: Mirror) -> Case {
    Case { mirror, targets }
}
