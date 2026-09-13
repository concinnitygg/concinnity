// Readers for the resource bindings slangc writes into emitted MSL and HLSL, so
// a build script can check them against the slots its host encoders write.

/// Whether the emitted MSL binds `param` at `attribute`. Emitted parameter names
/// carry a `_<n>` suffix whose number is the compiler's own (`joints_3`), so the
/// name is matched up to it.
pub fn msl_binds(msl: &str, param: &str, attribute: &str) -> bool {
    let marker = format!(" [[{attribute}]]");
    msl.match_indices(&marker).any(|(at, _)| {
        msl[..at]
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .and_then(|name| name.rsplit_once('_'))
            .is_some_and(|(head, tail)| head == param && tail.chars().all(|c| c.is_ascii_digit()))
    })
}

/// The parameter list of the one entry point in an emitted MSL translation unit,
/// split at top-level commas. Errs when no stage-tagged entry point named `entry`
/// with a parameter list is found.
pub fn msl_entry_params(msl: &str, entry: &str) -> Result<Vec<String>, String> {
    let stage = ["[[fragment]]", "[[vertex]]", "[[kernel]]"]
        .iter()
        .find_map(|tag| msl.find(tag))
        .ok_or_else(|| format!("emitted MSL declares no entry point for `{entry}`"))?;
    let open = stage
        + msl[stage..]
            .find('(')
            .ok_or_else(|| format!("emitted MSL entry point `{entry}` has no parameter list"))?;
    if !msl[stage..open].trim_end().ends_with(entry) {
        return Err(format!(
            "emitted MSL entry point is not `{entry}`: {}",
            &msl[stage..open]
        ));
    }

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

/// An emitted MSL parameter's source name. The `_<n>` suffix is the compiler's
/// own (`probe_cubes_texture_1`), so it is trimmed off.
pub fn msl_param_name(param: &str) -> &str {
    let name = param
        .rsplit(|c: char| c.is_whitespace())
        .next()
        .unwrap_or(param);
    match name.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => head,
        _ => name,
    }
}

/// Whether the emitted HLSL declares `param` at `register`, either as
/// `name_0 : register(..)` or as a resource array `name_0[int(N)] : register(..)`.
pub fn dxil_register_declared(emitted: &str, param: &str, register: &str) -> bool {
    let expected = format!("{param}_0 : register({register})");
    let expected_array = format!("{param}_0[");
    let expected_register = format!("register({register})");
    emitted.contains(&expected)
        || emitted
            .lines()
            .any(|l| l.contains(&expected_array) && l.contains(&expected_register))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAGMENT: &str = "struct VOut { float4 pos [[position]]; };\n\
        [[fragment]] float4 fragment_main(VOut in [[stage_in]], \
        array<texture2d<float>, 4> t_0 [[texture(0)]], constant P& p_1 [[buffer(1)]])\n\
        {\n    return float4(0.0);\n}\n";

    #[test]
    fn msl_binds_matches_the_name_up_to_a_numeric_suffix() {
        let msl = "    texture2d<float> albedo_3 [[texture(2)]],";
        assert!(msl_binds(msl, "albedo", "texture(2)"));
    }

    #[test]
    fn msl_binds_rejects_a_non_numeric_suffix() {
        let msl = "    texture2d<float> albedo_2d [[texture(2)]],";
        assert!(!msl_binds(msl, "albedo", "texture(2)"));
    }

    #[test]
    fn msl_binds_rejects_the_right_name_on_another_attribute() {
        let msl = "    texture2d<float> albedo_3 [[texture(2)]],";
        assert!(!msl_binds(msl, "albedo", "texture(3)"));
        assert!(!msl_binds(msl, "albedo", "buffer(2)"));
    }

    #[test]
    fn msl_param_name_trims_only_a_numeric_suffix() {
        assert_eq!(
            msl_param_name("probe_cubes_texture_1"),
            "probe_cubes_texture"
        );
        assert_eq!(
            msl_param_name("array<texturecube<float>, 8> probe_cubes_texture_1"),
            "probe_cubes_texture"
        );
        assert_eq!(msl_param_name("tex_2d"), "tex_2d");
        assert_eq!(msl_param_name("name_"), "name_");
    }

    #[test]
    fn msl_entry_params_splits_at_top_level_commas() {
        let params = msl_entry_params(FRAGMENT, "fragment_main").unwrap();
        assert_eq!(
            params,
            [
                "VOut in [[stage_in]]",
                "array<texture2d<float>, 4> t_0 [[texture(0)]]",
                "constant P& p_1 [[buffer(1)]]",
            ]
        );
    }

    #[test]
    fn msl_entry_params_keeps_an_unattributed_parameter() {
        let msl = "[[vertex]] VOut vertex_main(uint id [[vertex_id]], \
            array<texturecube<float>, 8> probes_2)";
        let params = msl_entry_params(msl, "vertex_main").unwrap();
        assert_eq!(params.len(), 2);
        assert!(!params[1].contains("[["));
        assert_eq!(msl_param_name(&params[1]), "probes");
    }

    #[test]
    fn msl_entry_params_errs_on_a_different_entry_name() {
        let err = msl_entry_params(FRAGMENT, "vertex_main").unwrap_err();
        assert!(err.contains("is not `vertex_main`"), "{err}");
    }

    #[test]
    fn msl_entry_params_errs_without_a_stage_tag() {
        let msl = "float4 fragment_main(VOut in [[stage_in]])";
        let err = msl_entry_params(msl, "fragment_main").unwrap_err();
        assert!(err.contains("no entry point"), "{err}");
    }

    #[test]
    fn msl_entry_params_errs_without_a_parameter_list() {
        let err = msl_entry_params("[[kernel]] void compute_main", "compute_main").unwrap_err();
        assert!(err.contains("no parameter list"), "{err}");
    }

    #[test]
    fn dxil_register_declared_matches_scalar_and_array_forms() {
        let hlsl = "cbuffer view_cb_0 : register(b0)\n\
            Texture2D<float4 > tex_0[int(4)] : register(t3);\n";
        assert!(dxil_register_declared(hlsl, "view_cb", "b0"));
        assert!(dxil_register_declared(hlsl, "tex", "t3"));
    }

    #[test]
    fn dxil_register_declared_rejects_a_wrong_register() {
        let hlsl = "cbuffer view_cb_0 : register(b0)\n\
            Texture2D<float4 > tex_0[int(4)] : register(t3);\n";
        assert!(!dxil_register_declared(hlsl, "view_cb", "b1"));
        assert!(!dxil_register_declared(hlsl, "tex", "t2"));
    }
}
