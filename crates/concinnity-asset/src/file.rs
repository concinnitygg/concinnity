// File authoring schema. The runtime `File` component lives in core.

use alloc::string::String;

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

/// Authored fields of a `File`.
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

    #[test]
    fn every_kind_is_reachable_from_an_extension() {
        // The inference table is the only way a kind is assigned when the args
        // omit one, so a kind missing from it can never be produced.
        let all = [
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
        for (ext, kind) in all {
            assert_eq!(FileKind::from_ext(ext).as_ref(), Some(&kind), "{ext}");
            // Extensions are matched case-insensitively.
            assert_eq!(
                FileKind::from_ext(&ext.to_uppercase()).as_ref(),
                Some(&kind),
                "{ext}"
            );
        }
    }

    #[test]
    fn an_unknown_extension_has_no_kind() {
        assert_eq!(FileKind::from_ext("wav"), None);
        assert_eq!(FileKind::from_ext(""), None);
    }

    #[test]
    fn only_obj_builds_to_a_mesh_payload() {
        assert!(FileKind::Obj.is_mesh());
        for kind in [FileKind::Png, FileKind::Ttf, FileKind::Mtl, FileKind::Md] {
            assert!(!kind.is_mesh(), "{kind:?}");
        }
    }

    #[test]
    fn args_default_to_an_empty_path_and_inferred_kind() {
        let args = FileArgs::default();
        assert!(args.path.is_empty());
        assert_eq!(args.kind, None);
    }

    #[test]
    fn an_absent_kind_is_omitted_from_the_serialized_args() {
        let args: FileArgs = serde_json::from_str(r#"{"path":"assets/board.obj"}"#).unwrap();
        assert_eq!(args.path, "assets/board.obj");
        assert_eq!(args.kind, None);
        // `cn add` writes normalized args back, so an inferred kind stays absent
        // rather than being frozen into the world file.
        assert_eq!(
            serde_json::to_string(&args).unwrap(),
            r#"{"path":"assets/board.obj"}"#
        );
    }

    #[test]
    fn an_explicit_kind_round_trips_through_its_lowercase_name() {
        let args: FileArgs = serde_json::from_str(r#"{"path":"font.dat","kind":"ttf"}"#).unwrap();
        assert_eq!(args.kind, Some(FileKind::Ttf));
        assert_eq!(
            serde_json::to_string(&args).unwrap(),
            r#"{"path":"font.dat","kind":"ttf"}"#
        );
        let bytes = postcard::to_allocvec(&args).unwrap();
        let back: FileArgs = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.kind, Some(FileKind::Ttf));
        assert_eq!(back.path, "font.dat");
    }
}
