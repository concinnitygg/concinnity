// JSON arg parsing for the room generator; the geometry itself is
// `concinnity_engine::geometry::build_room_geometry`.

use super::Verts;
use super::json_args::f32_arg;

pub(super) fn build_room(args: &serde_json::Value) -> (Verts, Vec<u16>) {
    let half_width = f32_arg(args, "half_width", 8.0);
    let half_depth = f32_arg(args, "half_depth", 10.0);
    let ceiling_height = f32_arg(args, "ceiling_height", 3.5);
    concinnity_engine::geometry::build_room_geometry(half_width, half_depth, 0.0, ceiling_height)
}

#[cfg(test)]
mod tests {
    use super::super::tests::bounds;
    use super::*;

    #[test]
    fn room_args_override_the_default_extents() {
        let args = serde_json::json!({
            "half_width": 2.0, "half_depth": 5.0, "ceiling_height": 4.0,
        });
        let (verts, _) = build_room(&args);
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-2.0, 0.0, -5.0]);
        assert_eq!(mx, [2.0, 4.0, 5.0]);
    }

    #[test]
    fn room_args_default_to_a_sixteen_by_twenty_room() {
        let (verts, _) = build_room(&serde_json::json!({}));
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-8.0, 0.0, -10.0]);
        assert_eq!(mx, [8.0, 3.5, 10.0]);
    }
}
