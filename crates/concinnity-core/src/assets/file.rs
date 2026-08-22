// src/assets/file.rs
//
// Runtime `File` component. Its authored args and `FileKind` live in the schema
// crate (concinnity_asset::file).

use alloc::string::String;

use crate::assets::{FileArgs, FileKind};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Component, PayloadLocator};

/// References a source file by path.
///
/// For supported kinds the build compiles the file into the world (an `.obj`
/// becomes mesh data); other kinds are path-only references.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct File {
    /// Assigned by the loader; not authored.
    pub asset_id: AssetId,
    /// Path to the source file, relative to the world.
    pub path: String,
    /// Content category, derived from the extension when not authored.
    pub kind: Option<FileKind>,
    /// Injected at load time for kinds that produce a compiled blob (e.g. obj → mesh payload).
    pub locator: Option<PayloadLocator>,
}

impl File {
    /// Translate the authored args into the runtime file reference: derive
    /// `kind` from the path extension when unset. Run by cook at build time
    /// (the baked blob record carries the result).
    pub fn bake(args: FileArgs) -> Self {
        let kind = args
            .kind
            .clone()
            .or_else(|| super::path_extension(&args.path).and_then(FileKind::from_ext));
        Self {
            asset_id: AssetId::default(),
            path: args.path,
            kind,
            locator: None,
        }
    }
}

impl Component for File {
    const NAME: &'static str = "File";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(postcard::from_bytes(bytes)?)
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_ext_maps_every_known_extension() {
        let cases = [
            ("obj", FileKind::Obj),
            ("png", FileKind::Png),
            ("jpg", FileKind::Jpg),
            ("jpeg", FileKind::Jpeg),
            ("bmp", FileKind::Bmp),
            ("tga", FileKind::Tga),
            ("gif", FileKind::Gif),
            ("ttf", FileKind::Ttf),
            ("otf", FileKind::Otf),
            ("txt", FileKind::Txt),
            ("md", FileKind::Md),
            ("mtl", FileKind::Mtl),
        ];
        for (ext, want) in cases {
            assert_eq!(FileKind::from_ext(ext), Some(want.clone()));
            // Matching is case-insensitive.
            assert_eq!(FileKind::from_ext(&ext.to_uppercase()), Some(want));
        }
        assert_eq!(FileKind::from_ext("zzz"), None);
    }

    #[test]
    fn from_args_infers_kind_from_the_extension() {
        // No explicit kind -> inferred from the path.
        let f = File::bake(FileArgs {
            path: "models/box.obj".into(),
            kind: None,
        });
        assert_eq!(f.kind, Some(FileKind::Obj));
        assert_eq!(f.path, "models/box.obj");
        // An explicit kind is kept even when it disagrees with the extension.
        let g = File::bake(FileArgs {
            path: "data.obj".into(),
            kind: Some(FileKind::Txt),
        });
        assert_eq!(g.kind, Some(FileKind::Txt));
        // An unknown extension leaves the kind unset.
        let h = File::bake(FileArgs {
            path: "notes.zzz".into(),
            kind: None,
        });
        assert_eq!(h.kind, None);
    }

    #[test]
    fn is_mesh_is_true_only_for_obj() {
        assert!(FileKind::Obj.is_mesh());
        assert!(!FileKind::Png.is_mesh());
        assert!(!FileKind::Ttf.is_mesh());
    }

    #[test]
    fn file_args_and_kind_round_trip_through_json() {
        let args = FileArgs {
            path: "x.png".into(),
            kind: Some(FileKind::Png),
        };
        let value = serde_json::to_value(&args).unwrap();
        let back: FileArgs = serde_json::from_value(value).unwrap();
        assert_eq!(back.path, "x.png");
        assert_eq!(back.kind, Some(FileKind::Png));
        // FileKind serializes to its lowercase name.
        assert_eq!(serde_json::to_string(&FileKind::Jpeg).unwrap(), "\"jpeg\"");
        // to_args mirrors the component fields.
        assert_eq!(File::bake(args).kind, Some(FileKind::Png));
    }
}
