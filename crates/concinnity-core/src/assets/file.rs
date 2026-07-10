// src/assets/file.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// References a source file by path.
///
/// For supported kinds the build compiles the file into the world (an `.obj`
/// becomes mesh data); other kinds are path-only references.
#[derive(Debug)]
pub struct File {
    pub asset_id: AssetId,
    pub path: String,
    pub kind: Option<FileKind>,
    /// Injected at load time for kinds that produce a compiled blob (e.g. obj → mesh payload).
    pub locator: Option<PayloadLocator>,
}

impl Component for File {
    const NAME: &'static str = "File";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;

    type Args = FileArgs;

    fn to_args(&self) -> FileArgs {
        FileArgs {
            path: self.path.clone(),
            kind: self.kind.clone(),
        }
    }

    fn from_args(args: FileArgs) -> Self {
        let kind = args.kind.clone().or_else(|| {
            std::path::Path::new(&args.path)
                .extension()
                .and_then(|e| e.to_str())
                .and_then(FileKind::from_ext)
        });
        Self {
            asset_id: AssetId::default(),
            path: args.path,
            kind,
            locator: None,
        }
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::build::SourceBacked for File {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

/// The category of file content, inferred from the extension when not supplied.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileKind {
    Obj,
    Png,
    Jpg,
    Jpeg,
    Bmp,
    Tga,
    Gif,
    Ttf,
    Otf,
    Txt,
    Md,
    Mtl,
}

impl FileKind {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "obj" => Some(Self::Obj),
            "png" => Some(Self::Png),
            "jpg" => Some(Self::Jpg),
            "jpeg" => Some(Self::Jpeg),
            "bmp" => Some(Self::Bmp),
            "tga" => Some(Self::Tga),
            "gif" => Some(Self::Gif),
            "ttf" => Some(Self::Ttf),
            "otf" => Some(Self::Otf),
            "txt" => Some(Self::Txt),
            "md" => Some(Self::Md),
            "mtl" => Some(Self::Mtl),
            _ => None,
        }
    }

    /// Returns true for kinds whose build output is a mesh blob compatible with the
    /// mesh payload format (vertex + index data readable by GraphicsSystem).
    pub fn is_mesh(&self) -> bool {
        matches!(self, Self::Obj)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct FileArgs {
    /// Path to the source file, relative to the project root.
    pub path: String,
    /// File category. Inferred from the path extension when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<FileKind>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{Platform, SourceBacked};
    use crate::ecs::Component;

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
        let f = File::from_args(FileArgs {
            path: "models/box.obj".into(),
            kind: None,
        });
        assert_eq!(f.kind, Some(FileKind::Obj));
        assert_eq!(f.path, "models/box.obj");
        // An explicit kind is kept even when it disagrees with the extension.
        let g = File::from_args(FileArgs {
            path: "data.obj".into(),
            kind: Some(FileKind::Txt),
        });
        assert_eq!(g.kind, Some(FileKind::Txt));
        // An unknown extension leaves the kind unset.
        let h = File::from_args(FileArgs {
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
    fn source_path_reads_the_path_field() {
        assert_eq!(
            File::source_path(&serde_json::json!({"path": "a.obj"}), Platform::Metal),
            Some("a.obj".to_string())
        );
        assert_eq!(
            File::source_path(&serde_json::json!({"path": ""}), Platform::Metal),
            None
        );
        assert_eq!(
            File::source_path(&serde_json::json!({}), Platform::Metal),
            None
        );
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
        assert_eq!(File::from_args(args).to_args().kind, Some(FileKind::Png));
    }
}
