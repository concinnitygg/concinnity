// The host binding layouts checked against what the shared sources declare:
// the slots each Metal encoder writes against the MSL a program compiles to,
// and the registers each DirectX root signature binds against the program's
// declarations, as the engine's `programs::abi` tables state them. These need
// dxc, and skip without it.

#[cfg(any(backend_metal, backend_dx))]
use concinnity_core::platform::Platform;
#[cfg(backend_dx)]
use concinnity_engine::render::shader::programs::abi::{DXIL_ENTRY_ABI, DxilAbi};
#[cfg(backend_metal)]
use concinnity_engine::render::shader::programs::abi::{METAL_ENTRY_ABI, MetalAbi};
#[cfg(backend_metal)]
use concinnity_shader::HlslTarget;

// The Metal main-pass binding layout is what the encoders in `metal/draw/`
// write, and a world Shader's stages compile from the same declarations. The
// sources pin every slot with a register(), and the table that turns those into
// Metal indices is generated (concinnity-shader's `metal_bindings`), so the
// MSL each program compiles to is checked: a slot that moves fails here instead
// of binding garbage at draw time.
#[cfg(backend_metal)]
#[test]
fn every_metal_entry_binds_the_slots_its_encoder_writes() {
    concinnity_shader::require_dxc!();
    let work = crate::shader::compiler_work::dir().expect("compiler work dir");
    for abi in METAL_ENTRY_ABI {
        let program = abi.program;
        let variant = program.at(false);
        let job = concinnity_shader::HlslJob {
            source: &variant.assemble(Platform::Metal),
            file_name: program.file,
            entry: program.entry,
            target: HlslTarget::Msl,
        };
        let msl = concinnity_shader::compile(&job, work.path())
            .unwrap_or_else(|e| panic!("{}: {e}", program.file));
        let msl = String::from_utf8(msl).expect("MSL is text");
        let msl = msl.as_str();
        let params = msl_entry_params(msl, program.entry).unwrap_or_else(|e| panic!("{e}"));
        assert_every_param_is_attributed(abi, &params);
        assert_every_resource_is_listed(abi, &params);
        for binding in abi.slots() {
            let (param, attribute) = (binding.name, binding.to_string());
            assert!(
                msl_binds(msl, param, &attribute),
                "{}: Metal ABI drifted at `{}`: expected `{param}` on [[{attribute}]] in the \
                 emitted MSL. \
                 The Metal binding layout is what the encoder writes and what world shaders \
                 compile from; fix the shader's declarations or the slot assignment \
                 before shipping.",
                program.file,
                program.entry
            );
        }
        for (member, id) in abi.argument_ids {
            let found = concinnity_shader::msl_argument_id(msl, member)
                .and_then(|id| usize::try_from(id).ok());
            assert_eq!(
                found,
                Some(*id),
                "{}: argument-buffer member `{member}` is at id {found:?}, not {id}, which is \
                 where the host writes it. A member's register number is its [[id(n)]]; fix \
                 the declaration.",
                program.file,
            );
        }
    }
}

#[cfg(backend_metal)]
// An entry parameter with no binding attribute is placed by the Metal compiler
// at whatever slot happens to be unused -- a placement the emitted MSL cannot be
// read for, and one an unrelated declaration silently moves.
fn assert_every_param_is_attributed(abi: &MetalAbi, params: &[String]) {
    let unattributed: Vec<&str> = params
        .iter()
        .filter(|p| !p.contains("[["))
        .map(|p| msl_param_name(p))
        .collect();
    assert!(
        unattributed.is_empty(),
        "{} ({}): the emitted MSL has entry parameters with no binding attribute: \
         {unattributed:?}, so the Metal compiler places them at the next unused slot and no \
         host binding can be checked against them. Give each a register(), or a \
         [[cn::metal_argument_buffer(n)]] for a set that rides an argument buffer.",
        abi.program.file,
        abi.program.entry,
    );
}

#[cfg(backend_metal)]
// A resource parameter the row does not list is one its host encoder never
// binds, which Metal validation rejects and a release build reads as garbage.
fn assert_every_resource_is_listed(abi: &MetalAbi, params: &[String]) {
    let unlisted: Vec<&str> = params
        .iter()
        .filter(|p| msl_binds_a_resource(p))
        .map(|p| msl_param_name(p))
        .filter(|name| !abi.slots().any(|binding| binding.name == *name))
        .collect();
    assert!(
        unlisted.is_empty(),
        "{} ({}): the emitted MSL binds {unlisted:?}, which the row does not list and so no \
         host encoder writes. Drop the read that keeps it, or bind it and list it.",
        abi.program.file,
        abi.program.entry,
    );
}

// Every DirectX program's registers against the root signature its host
// builds. Each variant compiles alone, so a register only has to hold in the
// entry that declares it.
#[cfg(backend_dx)]
#[test]
fn every_dxil_entry_declares_the_registers_its_root_signature_binds() {
    concinnity_shader::require_dxc!();
    let work = crate::shader::compiler_work::dir().expect("compiler work dir");
    for abi in DXIL_ENTRY_ABI {
        let program = abi.program;
        let source = program.at(false).assemble(Platform::DirectX);
        let declared =
            concinnity_shader::preprocessed_declarations(&source, program.file, work.path());
        assert_hlsl_registers(abi, declared);
    }
}

#[cfg(backend_dx)]
// An `.hlsl` source states its registers outright, so the check is that the
// declarations the preprocessor leaves standing are the ones the root signature
// binds. Preprocessed rather than raw: a register behind an inactive `#if`, or
// spelled as a macro, is not the one the compile sees.
fn assert_hlsl_registers(
    abi: &DxilAbi,
    declared: Result<Vec<concinnity_shader::declarations::Declaration>, String>,
) {
    let program = abi.program;
    let declared = declared.unwrap_or_else(|e| panic!("DXIL ABI check ({}): {e}", program.entry));
    for binding in abi.registers {
        let param = binding.name;
        let found = declared.iter().find(|d| d.name == param);
        let actual = found.map(|d| dxil_register(d.register));
        let register = binding.to_string();
        assert_eq!(
            actual.as_deref(),
            Some(register.as_str()),
            "{}: `{param}` is declared at {actual:?}, not register({register}). Fix the \
             .hlsl declaration or the matching root signature under src/directx before \
             shipping.",
            program.file,
        );
    }
}

#[cfg(backend_dx)]
// A register annotation as a table row spells it: `t0`, or `t0, space1` where
// the declaration names a space. The space is part of the D3D slot, so a
// row naming one asserts it.
fn dxil_register(register: concinnity_shader::declarations::Register) -> String {
    match register.space {
        0 => format!("{}{}", register.class, register.index),
        space => format!("{}{}, space{space}", register.class, register.index),
    }
}

// Readers for the resource bindings spirv-cross writes into emitted MSL.

// Whether the emitted MSL binds `param` at `attribute`. spirv-cross emits the
// source name unchanged, so the name matches exactly.
fn msl_binds(msl: &str, param: &str, attribute: &str) -> bool {
    let marker = format!(" [[{attribute}]]");
    msl.match_indices(&marker).any(|(at, _)| {
        msl[..at]
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .is_some_and(|name| name == param)
    })
}

// The parameter list of `entry` in an emitted MSL translation unit, split at
// top-level commas.
//
// # Errors
//
// Returns an error if no function named `entry` carries a leading stage
// qualifier (`fragment`, `vertex` or `kernel`).
fn msl_entry_params(msl: &str, entry: &str) -> Result<Vec<String>, String> {
    let needle = format!("{entry}(");
    let open = msl
        .match_indices(&needle)
        .map(|(at, _)| at + needle.len() - 1)
        .find(|at| stage_qualified(msl, *at))
        .ok_or_else(|| format!("emitted MSL declares no entry point for `{entry}`"))?;

    let mut params = Vec::new();
    let mut depth = 0i32;
    let mut start = open + 1;
    for (at, ch) in msl[open..].char_indices().map(|(i, c)| (open + i, c)) {
        match ch {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => {
                depth -= 1;
                if depth == 0 {
                    params.push(msl[start..at].trim().to_string());
                    break;
                }
            }
            ',' if depth == 1 => {
                params.push(msl[start..at].trim().to_string());
                start = at + 1;
            }
            _ => {}
        }
    }
    Ok(params)
}

// Whether the declaration whose parameter list opens at `open` carries a stage
// qualifier, which is what separates an entry point from a call to it.
fn stage_qualified(msl: &str, open: usize) -> bool {
    let line = msl[..open].rfind('\n').map_or(0, |at| at + 1);
    let head = msl[line..open].trim_start();
    ["fragment ", "vertex ", "kernel "]
        .iter()
        .any(|tag| head.starts_with(tag))
}

// An emitted MSL parameter's name, with its type and any attribute dropped.
fn msl_param_name(param: &str) -> &str {
    let decl = param.split(" [[").next().unwrap_or(param);
    decl.rsplit(|c: char| c.is_whitespace())
        .next()
        .unwrap_or(decl)
}

// Whether an emitted MSL parameter takes a buffer, texture or sampler slot,
// which a host encoder has to fill -- as opposed to `[[stage_in]]` or a
// built-in such as `[[instance_id]]`, which the pipeline supplies.
fn msl_binds_a_resource(param: &str) -> bool {
    ["[[buffer(", "[[texture(", "[[sampler("]
        .iter()
        .any(|slot| param.contains(slot))
}

mod readers {
    use super::*;

    const FRAGMENT: &str = "struct VOut { float4 pos [[position]]; };\n\
        fragment float4 fragment_main(VOut in [[stage_in]], \
        array<texture2d<float>, 4> t [[texture(0)]], constant P& p [[buffer(1)]])\n\
        {\n    return float4(0.0);\n}\n";

    // A name that prefixes another's must not match it.
    #[test]
    fn msl_binds_matches_the_exact_name() {
        let msl = "texture2d<float> atlas [[texture(0)]], sampler atlas_sampler [[sampler(0)]]";
        assert!(msl_binds(msl, "atlas", "texture(0)"));
        assert!(msl_binds(msl, "atlas_sampler", "sampler(0)"));
        assert!(!msl_binds(msl, "atlas", "sampler(0)"));
        assert!(!msl_binds(msl, "atla", "texture(0)"));
    }

    #[test]
    fn msl_binds_rejects_the_right_name_on_another_attribute() {
        let msl = "    texture2d<float> albedo [[texture(2)]],";
        assert!(!msl_binds(msl, "albedo", "texture(3)"));
        assert!(!msl_binds(msl, "albedo", "buffer(2)"));
    }

    #[test]
    fn msl_param_name_drops_the_type_and_attribute() {
        assert_eq!(
            msl_param_name("array<texturecube<float>, 8> probe_cubes [[texture(1)]]"),
            "probe_cubes"
        );
        assert_eq!(msl_param_name("constant P& p_1"), "p_1");
        assert_eq!(msl_param_name("tex"), "tex");
    }

    #[test]
    fn msl_entry_params_splits_at_top_level_commas() {
        let params = msl_entry_params(FRAGMENT, "fragment_main").unwrap();
        assert_eq!(
            params,
            [
                "VOut in [[stage_in]]",
                "array<texture2d<float>, 4> t [[texture(0)]]",
                "constant P& p [[buffer(1)]]",
            ]
        );
    }

    #[test]
    fn msl_entry_params_keeps_an_unattributed_parameter() {
        let msl = "vertex VOut vertex_main(uint id [[vertex_id]], \
            array<texturecube<float>, 8> probes)";
        let params = msl_entry_params(msl, "vertex_main").unwrap();
        assert_eq!(params.len(), 2);
        assert!(!params[1].contains("[["));
        assert_eq!(msl_param_name(&params[1]), "probes");
    }

    #[test]
    fn msl_entry_params_errs_on_a_different_entry_name() {
        let err = msl_entry_params(FRAGMENT, "vertex_main").unwrap_err();
        assert!(err.contains("vertex_main"), "{err}");
    }

    // spirv-cross names the return struct after the entry point, so the name
    // alone appears twice on the line.
    #[test]
    fn msl_entry_params_reads_the_leading_qualifier() {
        let msl = "fragment text_frag_out text_frag(text_frag_in in [[stage_in]], \
            texture2d<float> atlas [[texture(0)]])\n{\n}\n";
        let params = msl_entry_params(msl, "text_frag").unwrap();
        assert_eq!(
            params,
            [
                "text_frag_in in [[stage_in]]",
                "texture2d<float> atlas [[texture(0)]]",
            ]
        );
    }

    // A call to an entry point is not its declaration.
    #[test]
    fn msl_entry_params_errs_for_an_unqualified_declaration() {
        let msl = "float4 helper(float2 uv)\n{\n    return shade(uv);\n}\n";
        assert!(msl_entry_params(msl, "shade").is_err());
    }

    #[test]
    fn msl_entry_params_errs_without_a_stage_tag() {
        let msl = "float4 fragment_main(VOut in [[stage_in]])";
        let err = msl_entry_params(msl, "fragment_main").unwrap_err();
        assert!(err.contains("no entry point"), "{err}");
    }

    #[test]
    fn msl_entry_params_errs_without_a_parameter_list() {
        let err = msl_entry_params("kernel void compute_main", "compute_main").unwrap_err();
        assert!(err.contains("compute_main"), "{err}");
    }

    #[test]
    fn only_a_buffer_texture_or_sampler_slot_is_a_resource() {
        assert!(msl_binds_a_resource("constant P& p [[buffer(1)]]"));
        assert!(msl_binds_a_resource("texture2d<float> t [[texture(0)]]"));
        assert!(msl_binds_a_resource("sampler s [[sampler(2)]]"));
        assert!(!msl_binds_a_resource("VOut in [[stage_in]]"));
        assert!(!msl_binds_a_resource(
            "uint gl_InstanceIndex [[instance_id]]"
        ));
    }
}
