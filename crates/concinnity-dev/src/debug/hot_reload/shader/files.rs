//! Which world Shaders a filesystem event touches. The catalog stores paths as
//! the build resolved them (often relative), while the watcher reports them
//! absolute and, on macOS, through the canonical `/private` prefix, so both
//! sides are normalized before they are compared.

use concinnity_core::ecs::asset_id::AssetId;
use concinnity_engine::gfx::system::shader_sources::ShaderSourceMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// Every Shader file under its normalized path, with the Shader reading it. A
// file two Shaders share appears once per Shader.
#[derive(Debug, Default)]
pub(in crate::debug::hot_reload) struct ShaderFileIndex(Vec<(PathBuf, AssetId)>);

impl ShaderFileIndex {
    pub(in crate::debug::hot_reload) fn new(catalog: &ShaderSourceMap) -> Self {
        Self(
            catalog
                .files()
                .map(|(path, id)| (normalize(Path::new(path)), id))
                .collect(),
        )
    }

    // The Shaders reading any of `paths`.
    pub(in crate::debug::hot_reload) fn shaders_touched<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a PathBuf>,
    ) -> BTreeSet<AssetId> {
        let mut touched = BTreeSet::new();
        for path in paths {
            let path = normalize(path);
            touched.extend(
                self.0
                    .iter()
                    .filter(|(file, _)| *file == path)
                    .map(|&(_, id)| id),
            );
        }
        touched
    }
}

// An absolute path with symlinks resolved as far as the filesystem allows. A
// file an atomic save just removed cannot be canonicalized, so its parent is
// instead; a path whose directory is gone stays as it was, made absolute.
fn normalize(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name())
        && let Ok(parent) = parent.canonicalize()
    {
        return parent.join(name);
    }
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::Shader;

    fn catalog(shaders: &[(u32, &str, Option<&str>)]) -> ShaderSourceMap {
        let shaders: Vec<(Option<AssetId>, Shader)> = shaders
            .iter()
            .map(|&(id, fragment, vertex)| {
                (
                    Some(AssetId(id)),
                    Shader {
                        fragment: fragment.to_string(),
                        vertex: vertex.map(str::to_string),
                        locator: None,
                    },
                )
            })
            .collect();
        ShaderSourceMap::build(
            shaders.iter().map(|(id, s)| (*id, s)),
            str::to_string,
            |_| None,
        )
    }

    fn touched(index: &ShaderFileIndex, paths: &[&str]) -> Vec<AssetId> {
        let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        index.shaders_touched(&paths).into_iter().collect()
    }

    // A save marks just the Shaders reading that file, and a file two Shaders
    // share marks both.
    #[test]
    fn a_save_marks_only_the_shaders_reading_the_file() {
        let index = ShaderFileIndex::new(&catalog(&[
            (1, "/cn-none/lit.hlsl", Some("/cn-none/sway.hlsl")),
            (2, "/cn-none/reeds.hlsl", Some("/cn-none/sway.hlsl")),
            (3, "/cn-none/rock.hlsl", None),
        ]));
        assert_eq!(touched(&index, &["/cn-none/rock.hlsl"]), [AssetId(3)]);
        assert_eq!(
            touched(&index, &["/cn-none/sway.hlsl"]),
            [AssetId(1), AssetId(2)]
        );
        assert_eq!(
            touched(&index, &["/cn-none/lit.hlsl", "/cn-none/rock.hlsl"]),
            [AssetId(1), AssetId(3)]
        );
        assert!(touched(&index, &["/cn-none/engine.hlsl"]).is_empty());
    }

    // A catalog path relative to the working directory matches the absolute
    // path the watcher reports for the same file.
    #[test]
    fn a_relative_catalog_path_matches_its_absolute_event_path() {
        let index = ShaderFileIndex::new(&catalog(&[(4, "cn-none/water.hlsl", None)]));
        let absolute = std::env::current_dir().unwrap().join("cn-none/water.hlsl");
        assert_eq!(
            index
                .shaders_touched(&[absolute])
                .into_iter()
                .collect::<Vec<_>>(),
            [AssetId(4)]
        );
    }

    #[test]
    fn an_empty_catalog_touches_nothing() {
        let index = ShaderFileIndex::default();
        assert!(touched(&index, &["/cn-none/lit.hlsl"]).is_empty());
    }
}
