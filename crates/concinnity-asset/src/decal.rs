// Projected-decal schema.

use crate::{AssetId, TextureHandle, de_opt_texture_handle};

/// A projected texture stamped onto whatever scene geometry sits inside the
/// decal's oriented box.
///
/// The decal is a box volume positioned by `position`/`rotation_deg`/`size` in
/// world space. The texture is projected down the box's local +Y axis onto the
/// local X-Z plane and stamped onto the surfaces inside the box; anything
/// outside the box is unaffected. Surfaces near the box's top and bottom faces
/// fade out so the stamp doesn't show a hard edge on a curved surface.
///
/// The defaults orient the decal as a ground stamp: a flat 1×1 m square laid on
/// the world X-Z plane, projecting down from +Y. To stamp a wall, rotate so
/// local +Y points into the surface (e.g. `rotation_deg:[0,0,90]` for a +X
/// wall).
///
/// Decals blend over the lit image without affecting depth, so they layer on
/// top of the surfaces they stamp.
///
/// ```jsonl
/// // ground stamp (1.5 m square, projects down)
/// {"name":"footprint_a","type":"Decal","args":{"texture":"tex_footprint","position":[2.0,0.01,-1.5],"size":[1.5,0.5,1.5]}}
///
/// // wall stamp (rotated so local +Y faces +X, into the wall)
/// {"name":"bullet_hole_a","type":"Decal","args":{"texture":"tex_bullet","position":[3.0,1.6,-2.0],"rotation_deg":[0,0,90],"size":[0.4,0.2,0.4]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Decal {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Texture](#texture) asset projected onto the scene.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// World-space position of the decal box's centre.
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], YXZ order, same as
    /// [Prop](#prop).
    pub rotation_deg: [f32; 3],
    /// Local-space box extents. Local +Y is the projection axis; the texture
    /// is sampled on the local X-Z plane. A non-positive component disables
    /// the decal.
    pub size: [f32; 3],
    /// Linear-space RGBA tint multiplied with the sampled texture. The alpha
    /// channel scales the final blend, so `[1,1,1,0]` hides the decal.
    pub tint: [f32; 4],
    /// When false the decal is skipped each frame.
    pub visible: bool,
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
            visible: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_decal_is_a_visible_untinted_ground_stamp() {
        let d = Decal::default();
        assert_eq!(d.rotation_deg, [0.0, 0.0, 0.0]);
        assert_eq!(d.size, [1.0, 1.0, 1.0]);
        // An identity tint leaves the sampled texture alone; alpha 1 keeps it
        // fully blended.
        assert_eq!(d.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(d.visible);
        assert!(d.texture.is_none());
    }

    #[test]
    fn a_wall_stamp_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let d: Decal = serde_json::from_str(
            r#"{"texture":"tex_bullet","position":[3,1.6,-2],"rotation_deg":[0,0,90],
                "size":[0.4,0.2,0.4],"tint":[1,1,1,0.5],"visible":false}"#,
        )
        .unwrap();
        assert_eq!(d.texture, Some(TextureHandle(10)));
        assert_eq!(d.rotation_deg, [0.0, 0.0, 90.0]);
        assert!(!d.visible);

        let bytes = postcard::to_allocvec(&d).unwrap();
        let back: Decal = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.texture, Some(TextureHandle(10)));
        assert_eq!(back.position, [3.0, 1.6, -2.0]);
        assert_eq!(back.size, [0.4, 0.2, 0.4]);
        assert_eq!(back.tint, [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
