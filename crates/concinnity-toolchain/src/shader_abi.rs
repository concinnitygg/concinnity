// Readers for the resource bindings spirv-cross writes into emitted MSL, so a
// build script can check them against the slots its host encoders write.

/// Whether the emitted MSL binds `param` at `attribute`. spirv-cross emits the
/// source name unchanged, so the name matches exactly.
pub fn msl_binds(msl: &str, param: &str, attribute: &str) -> bool {
    let marker = format!(" [[{attribute}]]");
    msl.match_indices(&marker).any(|(at, _)| {
        msl[..at]
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .is_some_and(|name| name == param)
    })
}

/// The parameter list of `entry` in an emitted MSL translation unit, split at
/// top-level commas. Errs when no function of that name carries a leading stage
/// qualifier (`fragment`, `vertex` or `kernel`).
pub fn msl_entry_params(msl: &str, entry: &str) -> Result<Vec<String>, String> {
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

/// An emitted MSL parameter's name, with its type and any attribute dropped.
pub fn msl_param_name(param: &str) -> &str {
    let decl = param.split(" [[").next().unwrap_or(param);
    decl.rsplit(|c: char| c.is_whitespace())
        .next()
        .unwrap_or(decl)
}

/// Whether an emitted MSL parameter takes a buffer, texture or sampler slot,
/// which a host encoder has to fill -- as opposed to `[[stage_in]]` or a
/// built-in such as `[[instance_id]]`, which the pipeline supplies.
pub fn msl_binds_a_resource(param: &str) -> bool {
    ["[[buffer(", "[[texture(", "[[sampler("]
        .iter()
        .any(|slot| param.contains(slot))
}

#[cfg(test)]
mod tests {
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
