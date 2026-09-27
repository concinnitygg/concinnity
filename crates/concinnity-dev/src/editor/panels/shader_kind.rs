//! The kind a create form in the Shaders panel makes: a Shader, or a new
//! volume with a distance field of its own, surface or volumetric. Each kind is
//! one type's form, so choosing a kind of another type turns the form into
//! that type's.

use super::form_extras::{ExtraControl, ExtraRow};

// The kind row's id; the Shader and SdfVolume forms both leave it free.
pub(crate) const KIND_ROW: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShaderKind {
    Surface,
    SdfField,
    VolumetricField,
}

impl ShaderKind {
    pub(crate) const ALL: [ShaderKind; 3] = [
        ShaderKind::Surface,
        ShaderKind::SdfField,
        ShaderKind::VolumetricField,
    ];

    pub(crate) fn caption(self) -> &'static str {
        match self {
            ShaderKind::Surface => "Surface Shader",
            ShaderKind::SdfField => "SDF field",
            ShaderKind::VolumetricField => "Volumetric SDF field",
        }
    }

    // The type an entry of this kind is.
    pub(crate) fn ty(self) -> &'static str {
        match self {
            ShaderKind::Surface => "Shader",
            ShaderKind::SdfField | ShaderKind::VolumetricField => "SdfVolume",
        }
    }

    pub(crate) fn volumetric(self) -> bool {
        self == ShaderKind::VolumetricField
    }

    // The kind a press on the row moves to.
    pub(crate) fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&k| k == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    // The type the form should become when it shows this kind while being a
    // form for `ty`.
    pub(crate) fn switch_from(self, ty: &str) -> Option<&'static str> {
        (self.ty() != ty).then_some(self.ty())
    }

    // The row choosing among the kinds, showing this one.
    pub(crate) fn row(self) -> ExtraRow {
        ExtraRow {
            id: KIND_ROW,
            caption: "Kind".to_string(),
            indent: false,
            control: ExtraControl::Choice {
                options: Self::ALL.iter().map(|k| k.caption().to_string()).collect(),
                selected: Self::ALL.iter().position(|&k| k == self).unwrap_or(0),
            },
            detail: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The row cycles every kind, and only a kind of another type switches the
    // form: from a Shader to a surface field, from a volumetric field back.
    #[test]
    fn the_kinds_cycle_and_switch_types_between_them() {
        let mut kind = ShaderKind::Surface;
        let mut seen = Vec::new();
        for _ in 0..ShaderKind::ALL.len() {
            seen.push(kind);
            kind = kind.next();
        }
        assert_eq!(seen, ShaderKind::ALL);
        assert_eq!(kind, ShaderKind::Surface);
        assert_eq!(
            ShaderKind::Surface.next().switch_from("Shader"),
            Some("SdfVolume")
        );
        assert_eq!(ShaderKind::SdfField.next().switch_from("SdfVolume"), None);
        assert_eq!(
            ShaderKind::VolumetricField.next().switch_from("SdfVolume"),
            Some("Shader")
        );
        assert!(ShaderKind::VolumetricField.volumetric() && !ShaderKind::SdfField.volumetric());
    }

    #[test]
    fn the_row_offers_every_kind_with_its_own_selected() {
        let row = ShaderKind::VolumetricField.row();
        assert_eq!(row.id, KIND_ROW);
        assert!(row.pressable());
        assert_eq!(
            row.control,
            ExtraControl::Choice {
                options: vec![
                    "Surface Shader".to_string(),
                    "SDF field".to_string(),
                    "Volumetric SDF field".to_string()
                ],
                selected: 2,
            }
        );
    }
}
