use super::*;
use crate::compile::program::CompileFailure;
use concinnity_core::render::shader_source;

const PATH: &str = "shaders/blob.hlsl";

const MAP: &str = "float map(float3 p, SdfParams q, float t) { return sdSphere(p, 0.5); }\n";
const SHADE: &str = "SdfSurface shade(float3 p, float3 n, SdfParams q, float t, float2 uv) {\n\
    SdfSurface s; s.albedo = float3(1.0, 1.0, 1.0); s.roughness = 0.5;\n\
    s.metallic = 0.0; s.emissive = float3(0.0, 0.0, 0.0);\n\
    s.transmitted = float3(0.0, 0.0, 0.0); return s; }\n";

fn field(text: &str) -> SourceFile<'_> {
    SourceFile { path: PATH, text }
}

fn surface() -> String {
    format!("{MAP}{SHADE}")
}

fn failure(err: ProgramError) -> CompileFailure {
    match err {
        ProgramError::Compile(failed) => failed,
        other => panic!("not a compile failure: {other}"),
    }
}

// A casting surface field compiles all four of its entries on every host, each
// findable under the digest the renderer computes from the payload's own
// field. Metal's artifact is MSL text, emitted on every host: a world cooked
// on Windows or Linux still gives a Metal player its field without a compiler.
#[test]
fn every_host_cooks_one_artifact_per_entry_for_every_target() {
    concinnity_shader::require_dxc!();
    let text = surface();
    for platform in Platform::ALL {
        let compiled = compile_sdf_field("blob", field(&text), platform, false, true)
            .unwrap_or_else(|e| panic!("{platform:?}: {e}"));
        assert!(compiled.warnings.is_empty(), "{platform:?}");
        let programs = compiled.programs;
        assert_eq!(
            (programs.field.path.as_str(), programs.field.text.as_str()),
            (PATH, text.as_str())
        );
        let entries: Vec<&str> = programs.programs.iter().map(|p| p.entry.as_str()).collect();
        assert_eq!(
            entries,
            [
                "raymarch_vertex",
                "raymarch_fragment",
                "raymarch_shadow_vertex",
                "raymarch_shadow_fragment"
            ],
            "{platform:?}"
        );
        for program in raymarch::programs(false, true) {
            let source = raymarch::source(program.family, platform, programs.field.as_file());
            let digest = shader_source::source_digest(&source);
            let bytes = programs
                .artifact(program.entry, digest)
                .unwrap_or_else(|| panic!("{platform:?}: no artifact for {}", program.entry));
            if platform == Platform::Metal {
                let msl = std::str::from_utf8(bytes).expect("MSL is text");
                assert!(msl.contains(program.entry), "{}", program.entry);
            }
        }
    }
}

// Each compile runs in its own scratch directory, so a payload that differs
// between two compiles of the same field has stamped that directory in.
#[test]
fn a_volume_payload_is_the_same_on_every_compile() {
    concinnity_shader::require_dxc!();
    let text = surface();
    for platform in [Platform::Metal, Platform::Vulkan] {
        let compile = || {
            compile_sdf_field("blob", field(&text), platform, false, true)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
                .programs
        };
        assert_eq!(compile(), compile(), "{platform:?}");
    }
}

// The path names the field in diagnostics and nowhere else: the same text
// under another path compiles to the same bytes on every host.
#[test]
fn the_path_does_not_reach_the_artifact() {
    concinnity_shader::require_dxc!();
    let text = surface();
    let elsewhere = SourceFile {
        path: r"C:\worlds\lake\shaders\blob.hlsl",
        text: &text,
    };
    for platform in Platform::ALL {
        let artifacts = |file: SourceFile<'_>| -> Vec<Vec<u8>> {
            compile_sdf_field("blob", file, platform, false, true)
                .unwrap_or_else(|e| panic!("{platform:?}: {e}"))
                .programs
                .programs
                .into_iter()
                .map(|p| p.artifact)
                .collect()
        };
        assert_eq!(
            artifacts(field(&text)),
            artifacts(elsewhere),
            "{platform:?}"
        );
    }
}

// An error on a known line of the field is reported at that line of that
// file, once, although every entry compiled it.
#[test]
fn an_error_names_the_authors_file_and_line() {
    concinnity_shader::require_dxc!();
    let broken = format!(
        "// a comment\n\
         float map(float3 p, SdfParams q, float t)\n\
         {{\n\
         \x20   return sdSphere(p, undeclared_radius);\n\
         }}\n{SHADE}"
    );
    let failed = failure(
        compile_sdf_field("blob", field(&broken), Platform::Vulkan, false, true).unwrap_err(),
    );
    let errors: Vec<_> = failed.errors().collect();
    assert_eq!(errors.len(), 1, "{failed}");
    assert_eq!(
        (errors[0].path.as_str(), errors[0].line, errors[0].column),
        (PATH, 4, 24),
        "{failed}"
    );
    assert!(errors[0].message.contains("undeclared_radius"), "{failed}");
    assert_eq!(failed.failures.len(), 4, "every entry compiled the field");
    assert!(
        failed
            .to_string()
            .contains("shaders/blob.hlsl:4:24: error:"),
        "{failed}"
    );
}

// A `map` whose return type differs from the template's declaration errors at
// the author's line, with the declaration it clashes with as the note beneath.
#[test]
fn a_return_type_mismatch_names_the_authors_line_and_notes_the_template() {
    concinnity_shader::require_dxc!();
    let wrong = format!("{SHADE}float3 map(float3 p, SdfParams q, float t) {{ return 1.0; }}\n");
    let failed = failure(
        compile_sdf_field("blob", field(&wrong), Platform::DirectX, false, false).unwrap_err(),
    );
    let errors: Vec<_> = failed.errors().collect();
    assert_eq!(errors.len(), 1, "{failed}");
    assert_eq!(
        (errors[0].path.as_str(), errors[0].line),
        (PATH, 5),
        "{failed}"
    );
    let note = format!("{}:", raymarch::FILE);
    assert!(
        errors[0].context.contains(&note) && errors[0].context.contains("note:"),
        "{failed}"
    );
}

// An error the template reports after the field carries the author's line in
// its note, so the fence restores the template's own numbering and still lets
// the compiler point back into the field.
#[test]
fn a_template_error_notes_the_authors_line() {
    concinnity_shader::require_dxc!();
    let clash = format!(
        "{}// the proxy helper, again\nfloat3 proxy_world_pos(float3 pos) {{ return pos; }}\n",
        surface()
    );
    let failed = failure(
        compile_sdf_field("blob", field(&clash), Platform::Metal, false, false).unwrap_err(),
    );
    let errors: Vec<_> = failed.errors().collect();
    assert_eq!(errors.len(), 1, "{failed}");
    assert_eq!(errors[0].path, raymarch::FILE, "{failed}");
    assert!(
        errors[0]
            .context
            .contains("shaders/blob.hlsl:7:8: note: previous definition is here"),
        "{failed}"
    );
}

// A surface field that never defines `shade` is a call to an undefined
// function, and the message says what the field must define.
#[test]
fn a_missing_function_fails_naming_what_the_field_must_define() {
    concinnity_shader::require_dxc!();
    let err = compile_sdf_field("blob", field(MAP), Platform::Vulkan, false, false).unwrap_err();
    let message = err.to_string();
    assert!(
        message.starts_with("SdfVolume 'blob': compiling"),
        "{message}"
    );
    assert!(message.contains("found undefined function"), "{message}");
    assert!(message.contains("must define"), "{message}");
}

// A warning fails nothing and comes back located in the author's field, once.
#[test]
fn a_warning_is_returned_at_the_authors_line() {
    concinnity_shader::require_dxc!();
    let warns = format!(
        "float map(float3 p, SdfParams q, float t)\n\
         {{\n\
         \x20   int truncated = 3.5;\n\
         \x20   return sdSphere(p, 0.5);\n\
         }}\n{SHADE}"
    );
    let compiled = compile_sdf_field("blob", field(&warns), Platform::Metal, false, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let located: Vec<(&str, u32)> = compiled
        .warnings
        .iter()
        .map(|d| (d.path.as_str(), d.line))
        .collect();
    assert_eq!(located, [(PATH, 3)]);
}

// A volumetric field compiles the medium's pair and nothing else, even when
// the volume also asks to cast.
#[test]
fn a_volumetric_field_compiles_the_medium_pair() {
    concinnity_shader::require_dxc!();
    let medium = "VolumeSample sampleVolume(float3 p, SdfParams q, float t)\n\
                  {\n\
                  \x20   VolumeSample vs;\n\
                  \x20   vs.density = 0.5;\n\
                  \x20   vs.scattering = float3(0.8, 0.8, 0.85);\n\
                  \x20   vs.emission = float3(0.0, 0.0, 0.0);\n\
                  \x20   return vs;\n\
                  }\n";
    let programs = compile_sdf_field("cloud", field(medium), Platform::Vulkan, true, true)
        .unwrap_or_else(|e| panic!("{e}"))
        .programs;
    let entries: Vec<&str> = programs.programs.iter().map(|p| p.entry.as_str()).collect();
    assert_eq!(
        entries,
        ["raymarch_volumetric_vertex", "raymarch_volumetric_fragment"]
    );
}
