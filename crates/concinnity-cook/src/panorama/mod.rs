// src/panorama/mod.rs
//
// Recognises the "panorama sphere" packaging a lot of downloadable environment
// images ship in: a `.glb` holding one inverted sphere you stand inside, its
// image painted on the emissive channel over a black base colour so nothing
// lights or shades it. Imported as geometry that file renders as a ball in the
// middle of the scene; what the author wanted was a sky.
//
// `detect` is the predicate and `equirect` the extractor. Detection is
// deliberately narrow: every ordinary `.glb` must keep importing as scene
// geometry, so a file that misses any one criterion is rejected with the
// reason rather than reinterpreted.

mod detect;
mod equirect;

pub use detect::detect;
pub(crate) use equirect::load_equirect;
// Panorama / ordinary-scene `.glb` bytes for tests in sibling modules that
// need one of each without rebuilding the container by hand.
#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) fn panorama_glb_bytes() -> Vec<u8> {
        super::detect::test_fixtures::panorama_glb()
    }

    pub(crate) fn ordinary_scene_glb_bytes() -> Vec<u8> {
        super::detect::test_fixtures::ordinary_scene_glb()
    }
}

use crate::gltf_source::GltfDoc;

/// Whether the `.glb` / `.gltf` at `path` is a panorama sphere, i.e. an
/// environment image rather than scene geometry. `path` is read as given. A
/// file that fails to parse is not a panorama; the geometry import path
/// reports the real parse error.
pub fn file_is_panorama_sphere(path: &str) -> bool {
    GltfDoc::parse_file(path)
        .ok()
        .map(|doc| detect(&doc).is_ok())
        .unwrap_or(false)
}

// Decode the panorama image embedded in the `.glb` / `.gltf` at `path` into a
// linear-light equirectangular image; `path` is read as given (the caller
// resolved the authored source). Errors when the file is not a panorama
// sphere, naming the criterion it missed.
pub(crate) fn load_panorama_file(path: &str) -> Result<crate::hdr::HdrImage, String> {
    let doc = GltfDoc::parse_file(path)?;
    let panorama = detect(&doc).map_err(|e| format!("'{}': {}", path, e))?;
    load_equirect(&doc, path, panorama.image_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama::detect::test_fixtures::{ordinary_scene_glb, panorama_glb};

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).expect("write glb");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn a_panorama_glb_on_disk_is_recognised_and_decodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "sky.glb", &panorama_glb());
        assert!(file_is_panorama_sphere(&path));

        let image = load_panorama_file(&path).expect("decode");
        assert_eq!((image.width, image.height), (4, 2));
    }

    #[test]
    fn an_ordinary_scene_glb_on_disk_is_not_a_panorama() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "scene.glb", &ordinary_scene_glb());
        assert!(!file_is_panorama_sphere(&path));

        let err = load_panorama_file(&path).unwrap_err();
        assert!(err.contains("scene.glb"), "got: {err}");
    }

    #[test]
    fn an_unparseable_file_is_not_a_panorama() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "junk.glb", b"not a glb at all");
        assert!(!file_is_panorama_sphere(&path));
        assert!(load_panorama_file(&path).is_err());
    }
}
