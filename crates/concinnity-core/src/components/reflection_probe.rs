// Reflection-probe schema.

/// A localized reflection probe. The renderer captures the surrounding scene
/// into a cubemap from `position` and uses it for the specular reflection on
/// glossy surfaces within the influence box (`position` plus or minus
/// `half_extents`). The box is also the parallax-correction volume, so a
/// reflection stays anchored to the surrounding geometry as the camera moves.
///
/// Place several across a level so reflections stay accurate as a first-person
/// camera moves between areas (a room, a courtyard, a corridor): each surface
/// uses the probe whose box it sits deepest inside, and cross-fades into the
/// neighboring box near a shared boundary so reflections don't pop as the camera
/// crosses between them. When a world declares no `ReflectionProbe`, the renderer
/// auto-seeds a small grid of probes from the scene bounds, so existing scenes
/// still get local reflections without authoring.
///
/// Reflections are most accurate near `position`; a tighter box around a
/// distinct space (a room) parallax-corrects better than one large box. Boxes may
/// overlap freely: a surface inside several boxes blends all of them, so reflections
/// cross-fade smoothly as the camera moves between probes.
///
/// ```rust
/// # use concinnity_core::components::ReflectionProbe;
/// ReflectionProbe {
///     position: [0.0, 1.7, 0.0],
///     half_extents: [8.0, 4.0, 8.0],
///     ..Default::default()
/// };
/// ```
///
/// By default a probe captures everything it can see, however far away.
/// Setting `capture_distance` limits that, which suits a probe inside a room:
/// it skips the geometry beyond the walls, which it could not reflect anyway,
/// and its capture costs less.
///
/// ```rust
/// # use concinnity_core::components::ReflectionProbe;
/// // A 6 m room whose probe captures nothing past 10 m on any axis.
/// ReflectionProbe {
///     position: [0.0, 1.5, 0.0],
///     half_extents: [3.0, 1.5, 3.0],
///     capture_distance: Some(10.0),
///     ..Default::default()
/// };
/// ```
///
/// A probe's influence fades out across its box surface rather than stopping
/// at it, reaching `blend_distance` past the box on every axis. By default
/// that is a fifth of the box's smallest half-extent, which keeps a room's
/// reflection from leaking far into the next room through a shared wall.
/// Raise it where neighboring boxes meet in open space (two halves of a
/// courtyard, a hall opening into a corridor) so the hand-off between them is
/// a gradual cross-fade instead of a visible step.
///
/// ```rust
/// # use concinnity_core::components::ReflectionProbe;
/// // A long hall that cross-fades with its neighbors over 2 m.
/// ReflectionProbe {
///     position: [0.0, 1.7, 0.0],
///     half_extents: [12.0, 3.0, 4.0],
///     blend_distance: Some(2.0),
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ReflectionProbe {
    /// World-space capture point the cubemap is rendered from. Put it at roughly
    /// eye height in open space (not inside geometry) for the area it serves.
    #[asset(default = [0.0, 1.7, 0.0])]
    pub position: [f32; 3],
    /// Half-size of the influence box around `position`, per axis. A surface
    /// inside `position` plus or minus `half_extents` may select this probe, and
    /// the box is the parallax-correction volume. Make it span the local space
    /// the probe represents (e.g. a room's walls).
    #[asset(default = [10.0, 5.0, 10.0])]
    pub half_extents: [f32; 3],
    /// How far the capture reaches, in world units: an object lying wholly
    /// farther than this from `position` along the axis a cube face looks down
    /// is left out of that face, so the capture covers a cube of this half-size
    /// around `position`. An object that straddles the distance is captured
    /// whole, never cut, and the sky shows wherever an object was left out.
    /// `null` (the default) captures without limit.
    pub capture_distance: Option<f32>,
    /// How far past the influence box the probe's influence reaches, in
    /// meters, the same on every axis. A surface fades from this probe's
    /// reflection into its neighbors' over that distance either side of the box
    /// surface. `null` (the default) is a fifth of the box's smallest
    /// half-extent.
    pub blend_distance: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capture_distance_is_unlimited_unless_set() {
        let bare: ReflectionProbe = crate::test_support::from_json("{}");
        assert_eq!(bare.capture_distance, None);
        let set: ReflectionProbe = crate::test_support::from_json(r#"{"capture_distance":12.5}"#);
        assert_eq!(set.capture_distance, Some(12.5));
        assert_eq!(
            crate::components::validate::reflection_probe(set).capture_distance,
            Some(12.5)
        );
    }

    #[test]
    fn the_blend_distance_is_derived_from_the_box_unless_set() {
        let bare: ReflectionProbe = crate::test_support::from_json("{}");
        assert_eq!(bare.blend_distance, None);
        let set: ReflectionProbe = crate::test_support::from_json(r#"{"blend_distance":1.5}"#);
        assert_eq!(set.blend_distance, Some(1.5));
    }

    #[test]
    fn a_negative_blend_distance_does_not_blend() {
        let probe = ReflectionProbe {
            blend_distance: Some(-2.0),
            ..Default::default()
        };
        let probe = crate::components::validate::reflection_probe(probe);
        assert_eq!(probe.blend_distance, Some(0.0));
    }

    #[test]
    fn a_negative_capture_distance_captures_nothing() {
        let probe = ReflectionProbe {
            capture_distance: Some(-4.0),
            ..Default::default()
        };
        let probe = crate::components::validate::reflection_probe(probe);
        assert_eq!(probe.capture_distance, Some(0.0));
    }
}
