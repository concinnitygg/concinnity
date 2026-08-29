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
/// neighbouring box near a shared boundary so reflections don't pop as the camera
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ReflectionProbe {
    /// World-space capture point the cubemap is rendered from. Put it at roughly
    /// eye height in open space (not inside geometry) for the area it serves.
    pub position: [f32; 3],
    /// Half-size of the influence box around `position`, per axis. A surface
    /// inside `position` plus or minus `half_extents` may select this probe, and
    /// the box is the parallax-correction volume. Make it span the local space
    /// the probe represents (e.g. a room's walls).
    pub half_extents: [f32; 3],
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            position: [0.0, 1.7, 0.0],
            half_extents: [10.0, 5.0, 10.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_probe_captures_from_eye_height_over_a_room_sized_box() {
        let p = ReflectionProbe::default();
        assert_eq!(p.position, [0.0, 1.7, 0.0]);
        // The parallax box is wider than it is tall, matching a room rather than
        // a cube, so floor reflections land where the geometry actually is.
        assert_eq!(p.half_extents, [10.0, 5.0, 10.0]);
    }

    #[test]
    fn an_authored_probe_parses_and_round_trips_through_postcard() {
        let p: ReflectionProbe =
            serde_json::from_str(r#"{"position":[4,2,-6],"half_extents":[6,3,8]}"#).unwrap();
        assert_eq!(p.position, [4.0, 2.0, -6.0]);

        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: ReflectionProbe = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.half_extents, [6.0, 3.0, 8.0]);
    }
}
