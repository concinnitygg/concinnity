//! Viewport view-mode and show-flag state: backend-agnostic per-frame render
//! selection. The mode picks what the final image shows (the lit scene, a flat
//! shading, or one G-buffer channel); the flags switch individual feature
//! passes off for the frame without touching their resources. Consumed by the
//! render graph (masking pass gates) and each backend's composite.

/// What the viewport's final image shows. `Lit` is the shipping path; the
/// other modes visualize one stage of the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ViewMode {
    #[default]
    /// The fully lit shipping image.
    Lit = 0,
    /// Surface base color with no lighting.
    Unlit = 1,
    /// Triangle edges only.
    Wireframe = 2,
    /// View-space normals from the geometry prepass.
    Normals = 3,
    /// Perceptual roughness from the geometry prepass.
    Roughness = 4,
    /// Screen-space ambient occlusion.
    Occlusion = 5,
    /// Linear view depth.
    Depth = 6,
}

impl ViewMode {
    /// Every mode, in the cycling/UI order.
    pub const ALL: [ViewMode; 7] = [
        ViewMode::Lit,
        ViewMode::Unlit,
        ViewMode::Wireframe,
        ViewMode::Normals,
        ViewMode::Roughness,
        ViewMode::Occlusion,
        ViewMode::Depth,
    ];

    /// True for the flat-shaded modes (Unlit, Wireframe), which drop the
    /// surface-effect passes for a clean readable image.
    pub fn is_flat(self) -> bool {
        matches!(self, ViewMode::Unlit | ViewMode::Wireframe)
    }

    /// True for the modes whose image is a geometry-prepass channel sampled
    /// by the composite.
    pub fn is_gbuffer_channel(self) -> bool {
        matches!(
            self,
            ViewMode::Normals | ViewMode::Roughness | ViewMode::Occlusion | ViewMode::Depth
        )
    }

    /// Short display label for menus.
    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Lit => "Lit",
            ViewMode::Unlit => "Unlit",
            ViewMode::Wireframe => "Wireframe",
            ViewMode::Normals => "Normals",
            ViewMode::Roughness => "Roughness",
            ViewMode::Occlusion => "Occlusion",
            ViewMode::Depth => "Depth",
        }
    }
}

/// Per-frame feature-pass toggles. A cleared bit skips the matching pass for
/// the frame (the cheap runtime counterpart to the init-time trims); set bits
/// leave the pass to its normal gates. Billboard icons are editor overlay
/// sprites rather than a render pass, so they have no bit here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShowFlags(pub u32);

impl ShowFlags {
    /// Directional and local shadow passes.
    pub const SHADOWS: ShowFlags = ShowFlags(1 << 0);
    /// Volumetric fog.
    pub const FOG: ShowFlags = ShowFlags(1 << 1);
    /// Bloom.
    pub const BLOOM: ShowFlags = ShowFlags(1 << 2);
    /// Screen-space global illumination.
    pub const SSGI: ShowFlags = ShowFlags(1 << 3);
    /// Screen-space reflections.
    pub const SSR: ShowFlags = ShowFlags(1 << 4);
    /// The debug line pass.
    pub const LINES: ShowFlags = ShowFlags(1 << 5);

    /// Every flag, paired with its display label, in UI order.
    pub const LABELED: [(ShowFlags, &'static str); 6] = [
        (ShowFlags::SHADOWS, "Shadows"),
        (ShowFlags::FOG, "Fog"),
        (ShowFlags::BLOOM, "Bloom"),
        (ShowFlags::SSGI, "SSGI"),
        (ShowFlags::SSR, "Reflections"),
        (ShowFlags::LINES, "Lines"),
    ];

    /// Every flag set.
    pub const fn all() -> ShowFlags {
        ShowFlags(
            ShowFlags::SHADOWS.0
                | ShowFlags::FOG.0
                | ShowFlags::BLOOM.0
                | ShowFlags::SSGI.0
                | ShowFlags::SSR.0
                | ShowFlags::LINES.0,
        )
    }

    /// Whether every bit in `other` is set here.
    pub const fn contains(self, other: ShowFlags) -> bool {
        self.0 & other.0 == other.0
    }

    #[must_use]
    /// This set with `other`'s bits flipped.
    pub const fn toggled(self, other: ShowFlags) -> ShowFlags {
        ShowFlags(self.0 ^ other.0)
    }
}

impl Default for ShowFlags {
    fn default() -> Self {
        ShowFlags::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_shows_everything() {
        let all = ShowFlags::default();
        for (flag, _) in ShowFlags::LABELED {
            assert!(all.contains(flag));
        }
        assert_eq!(all, ShowFlags::all());
    }

    #[test]
    fn toggling_clears_and_restores_one_bit() {
        let some = ShowFlags::all().toggled(ShowFlags::FOG);
        assert!(!some.contains(ShowFlags::FOG));
        assert!(some.contains(ShowFlags::SHADOWS));
        assert_eq!(some.toggled(ShowFlags::FOG), ShowFlags::all());
    }

    #[test]
    fn mode_classes_partition_as_expected() {
        assert!(!ViewMode::Lit.is_flat() && !ViewMode::Lit.is_gbuffer_channel());
        assert!(ViewMode::Unlit.is_flat() && ViewMode::Wireframe.is_flat());
        for m in [
            ViewMode::Normals,
            ViewMode::Roughness,
            ViewMode::Occlusion,
            ViewMode::Depth,
        ] {
            assert!(m.is_gbuffer_channel() && !m.is_flat());
        }
        assert_eq!(ViewMode::ALL.len(), 7);
    }
}
