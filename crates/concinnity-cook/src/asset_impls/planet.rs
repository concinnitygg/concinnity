use concinnity_engine::components::Planet;
use concinnity_engine::planet::decode_edits;

use crate::asset::{BuildCtx, SourceFiles};

// The edits file a planet's args name, if any.
fn edits_path(args: &serde_json::Value) -> Option<&str> {
    args.get("edits")
        .and_then(serde_json::Value::as_str)
        .filter(|p| !p.is_empty())
}

// Where the edits file `raw` lies: the first place it exists among the build's
// asset root, the checkout's `assets/` and the working directory.
fn resolve(raw: &str, ctx: &BuildCtx<'_>) -> Option<String> {
    concinnity_host::store::source::find_existing(raw, ctx.assets_dir)
}

impl crate::asset::BuildAsset for Planet {
    // The edits file itself, checked to read as edits; nothing for a planet
    // with none.
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        let Some(raw) = edits_path(args) else {
            return Ok(Vec::new());
        };
        let path = resolve(raw, ctx).unwrap_or_else(|| raw.to_string());
        let bytes = std::fs::read(&path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Planet '{}': cannot read its edits '{path}': {e}", ctx.name),
            )
        })?;
        decode_edits(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Planet '{}': its edits '{path}' do not read: {e}", ctx.name),
            )
        })?;
        Ok(bytes)
    }

    // The edits file is the payload's only input beyond the args.
    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        SourceFiles::Extra(
            edits_path(args)
                .and_then(|raw| resolve(raw, ctx))
                .into_iter()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::BuildAsset;
    use concinnity_engine::planet::{
        CubeFace, EditPage, EditPages, PAGE_SAMPLES, TileId, encode_edits,
    };

    fn ctx(assets_dir: Option<&std::path::Path>) -> BuildCtx<'_> {
        BuildCtx {
            name: "world",
            targets: concinnity_engine::platform::PlatformSet::single(
                concinnity_engine::platform::Platform::Metal,
            ),
            assets_dir,
            all_assets: &[],
        }
    }

    fn edits_file() -> Vec<u8> {
        let page = EditPage::from_samples(&[2.0; PAGE_SAMPLES], &[0.5; PAGE_SAMPLES]).unwrap();
        encode_edits(&EditPages::new([(TileId::root(CubeFace(2)), page)]))
    }

    // A temporary asset root holding `bytes` as `name`.
    fn root_with(name: &str, bytes: &[u8]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(name), bytes).expect("write");
        dir
    }

    #[test]
    fn a_planet_without_edits_compiles_to_nothing() {
        let args = serde_json::json!({ "radius": 9000.0 });
        assert!(
            Planet::compile_payload(&args, &ctx(None))
                .unwrap()
                .is_empty()
        );
        let blank = serde_json::json!({ "edits": "" });
        assert!(
            Planet::compile_payload(&blank, &ctx(None))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            Planet::source_files(&args, &ctx(None)),
            SourceFiles::Extra(Vec::new())
        );
    }

    // The payload is the file as written, found under the asset root, and
    // the file is an input of the payload's cache key.
    #[test]
    fn a_planet_s_edits_compile_to_the_file() {
        let bytes = edits_file();
        let root = root_with("good.cnedits", &bytes);
        let args = serde_json::json!({ "edits": "good.cnedits" });
        let payload = Planet::compile_payload(&args, &ctx(Some(root.path()))).unwrap();
        assert_eq!(payload, bytes);
        let SourceFiles::Extra(files) = Planet::source_files(&args, &ctx(Some(root.path()))) else {
            panic!("the walk stays on");
        };
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("good.cnedits"), "{files:?}");
    }

    // A missing file and one that does not read as edits fail the build,
    // each naming the planet, the file and why.
    #[test]
    fn missing_or_bad_edits_fail_with_a_clear_message() {
        let missing = serde_json::json!({ "edits": "nowhere/none.cnedits" });
        let err = Planet::compile_payload(&missing, &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("nowhere/none.cnedits"), "{err}");
        let mut bad = edits_file();
        bad[4] = 7;
        let root = root_with("bad.cnedits", &bad);
        let args = serde_json::json!({ "edits": "bad.cnedits" });
        let err = Planet::compile_payload(&args, &ctx(Some(root.path()))).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let message = err.to_string();
        assert!(
            message.contains("Planet 'world'") && message.contains("version 7"),
            "{message}"
        );
    }
}
