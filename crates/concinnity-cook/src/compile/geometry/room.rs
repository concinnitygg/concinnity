// JSON arg parsing for the room generator; the geometry itself is
// `concinnity_engine::geometry::build_room_geometry`.

use super::json_args::f32_arg;

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub(super) fn build_room(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let half_width = f32_arg(args, "half_width", 8.0);
    let half_depth = f32_arg(args, "half_depth", 10.0);
    let ceiling_height = f32_arg(args, "ceiling_height", 3.5);
    Ok(concinnity_engine::geometry::build_room_geometry(
        half_width,
        half_depth,
        0.0,
        ceiling_height,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(verts: &Verts) -> ([f32; 3], [f32; 3]) {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (pos, ..) in verts {
            for k in 0..3 {
                mn[k] = mn[k].min(pos[k]);
                mx[k] = mx[k].max(pos[k]);
            }
        }
        (mn, mx)
    }

    #[test]
    fn room_args_override_the_default_extents() {
        let args = serde_json::json!({
            "half_width": 2.0, "half_depth": 5.0, "ceiling_height": 4.0,
        });
        let (verts, _) = build_room(&args).unwrap();
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-2.0, 0.0, -5.0]);
        assert_eq!(mx, [2.0, 4.0, 5.0]);
    }

    #[test]
    fn room_args_default_to_a_sixteen_by_twenty_room() {
        let (verts, _) = build_room(&serde_json::json!({})).unwrap();
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-8.0, 0.0, -10.0]);
        assert_eq!(mx, [8.0, 3.5, 10.0]);
    }
}
