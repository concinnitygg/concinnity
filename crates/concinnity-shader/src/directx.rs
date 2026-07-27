// The DirectX toolchain: HLSL through the Direct3D `Fxc` compiler. Worlds
// targeting this backend author HLSL, so the GLSL arm carries a hint rather than
// falling to the generic unsupported message.

use concinnity_cook::shader::{ShaderCompileArgs, ShaderToolchain, set_shader_toolchain};

pub(crate) fn install() {
    set_shader_toolchain(Box::new(DirectXToolchain));
}

struct DirectXToolchain;

impl ShaderToolchain for DirectXToolchain {
    fn compile_hlsl(
        &self,
        source: &str,
        args: &ShaderCompileArgs,
    ) -> Result<Vec<u8>, std::io::Error> {
        compile_hlsl(source, args)
    }

    fn compile_glsl(&self, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!(
                "Asset '{}': GLSL/SPIR-V compilation is not supported by the DirectX backend (use HLSL)",
                args.asset_name
            ),
        ))
    }
}

fn compile_hlsl(source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use windows::Win32::Graphics::Direct3D::Fxc::{
        D3DCOMPILE_DEBUG, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR,
        D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompile,
    };

    let target = match args.kind.to_lowercase().as_str() {
        "fragment" | "frag" => "ps_5_1",
        _ => "vs_5_1",
    };

    let src_c = std::ffi::CString::new(source).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("hlsl src: {e}"))
    })?;
    let entry_c = std::ffi::CString::new("main").unwrap();
    let target_c = std::ffi::CString::new(target).unwrap();

    // Force column-major matrix storage so matrices inside `StructuredBuffer`
    // structs read as column-major (matching Rust's upload layout). FXC
    // silently ignores `#pragma pack_matrix(column_major)` for SRV-resident
    // matrices and defaults them to row_major; without this flag a custom
    // shader that reads e.g. an instance-matrix StructuredBuffer would see
    // every transform transposed. Mirrors the same flag in
    // `concinnity_device::directx::pipeline::compile_hlsl`.
    let flags = if cfg!(debug_assertions) {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
    } else {
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
    };

    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;

    let result = unsafe {
        D3DCompile(
            src_c.as_ptr() as *const std::ffi::c_void,
            source.len(),
            None,
            None,
            None,
            windows::core::PCSTR(entry_c.as_ptr() as *const u8),
            windows::core::PCSTR(target_c.as_ptr() as *const u8),
            flags,
            0,
            &mut blob,
            Some(&mut error),
        )
    };

    if result.is_err() {
        let msg = error
            .as_ref()
            .map(|e| {
                let ptr = unsafe { e.GetBufferPointer() } as *const u8;
                let len = unsafe { e.GetBufferSize() };
                // The error blob's size counts its NUL terminator; keeping it
                // would embed a NUL in the middle of the build error text.
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) })
                    .trim_end_matches(['\0', '\n'])
                    .to_string()
            })
            .unwrap_or_else(|| "unknown compile error".to_string());
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("FXC failed for '{}' ({target}):\n{msg}", args.asset_name),
        ));
    }

    let b = blob.ok_or_else(|| {
        std::io::Error::other(format!(
            "Asset '{}': D3DCompile returned no blob",
            args.asset_name
        ))
    })?;
    let ptr = unsafe { b.GetBufferPointer() } as *const u8;
    let len = unsafe { b.GetBufferSize() };
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(asset_name: &str, kind: &str) -> ShaderCompileArgs {
        ShaderCompileArgs {
            source_path: "user_frag.hlsl".to_string(),
            asset_name: asset_name.to_string(),
            kind: kind.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn glsl_points_at_hlsl_instead_of_the_generic_message() {
        let err = DirectXToolchain
            .compile_glsl(&args("stage", "fragment"))
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("use HLSL"), "got: {err}");
        assert!(err.to_string().contains("stage"), "names the asset: {err}");
    }

    #[test]
    fn a_source_that_does_not_compile_reports_the_asset_and_target() {
        let err =
            compile_hlsl("this is not valid HLSL", &args("bad_stage", "fragment")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad_stage"), "names the asset: {msg}");
        assert!(msg.contains("ps_5_1"), "names the target profile: {msg}");
        assert!(!msg.contains('\0'), "no NUL from the error blob: {msg:?}");
    }

    #[test]
    fn a_valid_source_compiles_to_bytecode() {
        let src = "float4 main() : SV_TARGET { return float4(0,0,0,1); }";
        let bytes = compile_hlsl(src, &args("good_stage", "fragment")).expect("valid HLSL");
        assert!(!bytes.is_empty(), "bytecode is never empty");
    }

    // A vertex stage selects the vertex profile, not the pixel one.
    #[test]
    fn the_compile_kind_selects_the_target_profile() {
        let err = compile_hlsl("not valid", &args("stage", "vertex")).unwrap_err();
        assert!(err.to_string().contains("vs_5_1"), "got: {err}");
    }
}
