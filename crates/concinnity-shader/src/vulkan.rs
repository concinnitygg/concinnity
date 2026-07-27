// The Vulkan toolchain: GLSL to SPIR-V in process via shaderc.

use concinnity_cook::shader::{ShaderCompileArgs, ShaderToolchain, set_shader_toolchain};

pub(crate) fn install() {
    set_shader_toolchain(Box::new(VulkanToolchain));
}

struct VulkanToolchain;

impl ShaderToolchain for VulkanToolchain {
    fn compile_glsl(&self, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
        compile_glsl(args)
    }
}

fn compile_glsl(args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use shaderc::{CompileOptions, Compiler, EnvVersion, OptimizationLevel, ShaderKind, TargetEnv};

    let source = std::fs::read_to_string(&args.source_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read '{}': {}", args.source_path, e),
        )
    })?;

    let compiler = Compiler::new()
        .map_err(|e| std::io::Error::other(format!("shaderc init failed: {}", e)))?;

    let mut options = CompileOptions::new()
        .map_err(|e| std::io::Error::other(format!("shaderc options init: {}", e)))?;
    options.set_optimization_level(OptimizationLevel::Performance);
    // Target Vulkan 1.2 so the emitted module is SPIR-V 1.5 (not the 1.6 a 1.3
    // target produces). The runtime instance is created at Vulkan 1.2 (see
    // `concinnity_device::vulkan::init`), and `vkCreateShaderModule` rejects a SPIR-V version
    // newer than the instance's, so a world-authored Shader stage compiled to
    // SPIR-V 1.6 fails to load. 1.5 loads cleanly on a 1.2-or-newer instance.
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);

    let kind = match args.kind.to_lowercase().as_str() {
        "vertex" | "vert" => ShaderKind::Vertex,
        "fragment" | "frag" => ShaderKind::Fragment,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported shader kind: '{}'", args.kind),
            ));
        }
    };

    let artifact = compiler
        .compile_into_spirv(&source, kind, &args.source_path, "main", Some(&options))
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shaderc failed for '{}':\n{}", args.asset_name, e),
            )
        })?;

    if artifact.get_num_warnings() > 0 {
        tracing::warn!(
            "Asset '{}' GLSL warnings:\n{}",
            args.asset_name,
            artifact.get_warning_messages()
        );
    }

    let spv_bytes: Vec<u8> = artifact
        .as_binary()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();

    Ok(spv_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(asset_name: &str, kind: &str, source_path: &str) -> ShaderCompileArgs {
        ShaderCompileArgs {
            source_path: source_path.to_string(),
            asset_name: asset_name.to_string(),
            kind: kind.to_string(),
            ..Default::default()
        }
    }

    fn write_source(dir: &std::path::Path, name: &str, text: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, text).expect("write source");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn a_missing_source_reports_the_path() {
        let err = compile_glsl(&args("stage", "fragment", "/no/such/user.glsl")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("/no/such/user.glsl"), "got: {err}");
    }

    #[test]
    fn an_unknown_compile_kind_is_rejected_before_compiling() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_source(dir.path(), "s.glsl", "#version 450\nvoid main() {}\n");
        let err = compile_glsl(&args("stage", "geometry", &path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("geometry"), "got: {err}");
    }

    #[test]
    fn a_source_that_does_not_compile_names_the_asset() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_source(dir.path(), "bad.glsl", "this is not valid GLSL");
        let err = compile_glsl(&args("bad_stage", "fragment", &path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("bad_stage"), "got: {err}");
    }

    // SPIR-V starts with the 0x07230203 magic word, little endian.
    #[test]
    fn a_valid_source_compiles_to_spirv() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_source(
            dir.path(),
            "ok.glsl",
            "#version 450\nlayout(location=0) out vec4 c;\nvoid main() { c = vec4(0); }\n",
        );
        let bytes = compile_glsl(&args("good_stage", "fragment", &path)).expect("valid GLSL");
        assert_eq!(&bytes[..4], &[0x03, 0x02, 0x23, 0x07]);
    }
}
