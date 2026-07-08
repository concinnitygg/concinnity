// Bridge that registers the Metal shader-layout validator with the core build
// pipeline. The Metal reflection itself lives in `concinnity_device::metal` (the
// only place that touches the Metal reflection API); this file is the thin
// adapter that plugs it into the cook `ShaderBuildValidator` seam and wraps a
// mismatch with the offending asset's name so `cn build` reports which shader to
// fix. It is the one spot that depends on both cook and the device backend.
//
// Compiled only under the Metal backend.

use std::sync::Once;

use concinnity_cook::shader::{ShaderBuildValidator, set_shader_build_validator};
use concinnity_device::metal::{ShaderLayoutIssue, validate_metal_shader_layout};

// Register the Metal shader-layout validator with the core build pipeline. Safe
// to call from every build entry point; only the first call installs it (the
// underlying registration is itself first-wins).
pub(crate) fn register_shader_layout_validator() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        set_shader_build_validator(Box::new(MetalShaderValidator));
    });
}

struct MetalShaderValidator;

impl ShaderBuildValidator for MetalShaderValidator {
    fn validate_metal(&self, source: &str, kind: &str, asset_name: &str) -> Result<(), String> {
        match validate_metal_shader_layout(source, kind) {
            Ok(()) => Ok(()),
            Err(ShaderLayoutIssue::Mismatch(msg)) => Err(format!(
                "shader asset '{asset_name}': {msg}\nThe shader declares an engine-provided buffer \
                 struct with a different memory layout than the engine's, so the GPU would read the \
                 engine's data through the wrong offsets. Match the documented layout (see the \
                 ShaderStage asset reference)."
            )),
            Err(ShaderLayoutIssue::Infra(reason)) => {
                // Fail open: never break a build over a reflection-infrastructure
                // problem. A missed check is recoverable; a spurious build break
                // erodes trust in the build.
                tracing::warn!("shader asset '{asset_name}': skipped layout validation ({reason})");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_device::metal::metal_device_available;

    // A correct user vertex shader: declares ViewUniforms exactly as the engine
    // does (packed_float3 cam_pos) and binds it at buffer(0).
    const GOOD_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            packed_float3 cam_pos;
            float prefilter_mip_count;
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + float3(view.cam_pos) * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // The same shader with `float3 cam_pos` (16-byte aligned) instead of
    // packed_float3: grows the struct stride past the engine's 160 bytes, so the
    // size check rejects it (the float3-vs-[f32;3] class of bug).
    const BAD_SIZE_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            float3 cam_pos;
            float prefilter_mip_count;
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + view.cam_pos * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    #[test]
    fn validator_fails_the_build_with_asset_context() {
        if !metal_device_available() {
            return;
        }
        // The build-facing entry point wraps the mismatch with the asset name so
        // `cn build` reports which shader to fix.
        let err = MetalShaderValidator
            .validate_metal(BAD_SIZE_VERTEX, "vertex", "my_custom_vert")
            .expect_err("a mismatched shader must fail the build");
        assert!(err.contains("my_custom_vert"), "names the asset: {err}");
        assert!(
            err.contains("ViewUniforms"),
            "names the engine struct: {err}"
        );
    }

    #[test]
    fn faithful_shader_passes_the_build_entry() {
        if !metal_device_available() {
            return;
        }
        MetalShaderValidator
            .validate_metal(GOOD_VERTEX, "vertex", "ok_vert")
            .expect("a faithful shader must pass the build");
    }
}
