// The `BuildAsset` impls of the resource assets whose payload is one call into
// their compiler: the decoders and generators in `crate::compile`.

use std::path::{Path, PathBuf};

use concinnity_engine::components::{
    AudioClip, ColorLut, CubemapTexture, EnvironmentMap, Font, Mesh, Texture,
};

use crate::asset::{BuildAsset, BuildCtx, SourceFiles};

// The files a resource keyed by `source` reads: the source itself, the files a
// text `.gltf` names beside it, and a character model's own source.
pub(super) fn source_files(args: &serde_json::Value, assets_dir: Option<&Path>) -> SourceFiles {
    let source = args
        .get("source")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let mut files: Vec<PathBuf> = source.map(PathBuf::from).into_iter().collect();
    if let Some(src) = source
        && src.to_lowercase().ends_with(".gltf")
    {
        files.extend(crate::import::gltf_source::referenced_files(
            src, assets_dir,
        ));
    }
    if let Some(arg) = args.get("character_model").and_then(|v| {
        serde_json::from_value::<crate::compile::character::import::CharacterModelArg>(v.clone())
            .ok()
    }) {
        files.extend(arg.source_files());
    }
    SourceFiles::Extra(files)
}

impl BuildAsset for AudioClip {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::audio_clip::compile_audio_clip_payload(args).map_err(std::io::Error::from)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }
}

impl BuildAsset for Texture {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::texture::compile_texture_payload(args, ctx.assets_dir)
            .map_err(std::io::Error::from)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }
}

impl BuildAsset for CubemapTexture {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::cubemap::compile_cubemap_payload(args).map_err(std::io::Error::from)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }
}

impl BuildAsset for EnvironmentMap {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::environment_map::compile_environment_map_payload(args, ctx.assets_dir)
            .map_err(std::io::Error::from)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }

    // The display fields (`background`) never reach the convolution, so the
    // payload keys on the bake inputs alone and editing one reuses it.
    fn payload_key_args(args: &serde_json::Value) -> std::borrow::Cow<'_, serde_json::Value> {
        crate::compile::environment_map::environment_map_payload_key_args(args)
            .map_or(std::borrow::Cow::Borrowed(args), std::borrow::Cow::Owned)
    }

    fn compile_data(_: &str, args: &serde_json::Value) -> std::io::Result<Option<Vec<u8>>> {
        crate::compile::environment_map::compile_environment_map_data(args)
            .map(Some)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

impl BuildAsset for ColorLut {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::color_lut::compile_color_lut_payload(args, ctx.assets_dir)
            .map_err(std::io::Error::from)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }
}

impl BuildAsset for Font {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::font::compile_font_payload(args).map_err(std::io::Error::from)
    }

    // A Font keys its TTF source under `path`, not `source`.
    fn source_files(args: &serde_json::Value, _: &BuildCtx<'_>) -> SourceFiles {
        SourceFiles::Extra(
            args.get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .into_iter()
                .collect(),
        )
    }
}

impl BuildAsset for Mesh {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        crate::compile::mesh::compile_mesh_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        source_files(args, ctx.assets_dir)
    }
}
