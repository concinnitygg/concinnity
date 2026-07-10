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
