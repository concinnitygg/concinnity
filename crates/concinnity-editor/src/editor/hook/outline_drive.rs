// src/editor/hook/outline_drive.rs
//
// EditorHook: the extent-outline drive. Each frame it appends the wireframes
// of the selection's spatial extents (and any category the Display menu keeps
// always-on) to the frame's world-space line buffer: trigger-volume colliders,
// light ranges and cones, camera frusta, probe bounds, prop colliders. Shapes
// read the live components (and the seeded Transform, so a gizmo drag moves
// the outline), so form edits and drags update immediately. The lines ride
// the renderer's line pass, which handles depth: occluded runs draw faint.

use super::*;
use crate::assets::{
    Collider, PointLight, RectAreaLight, ReflectionProbe, SPOT_MAX_ANGLE_DEG, SpotLight,
    SpotLightGeometry, Transform, TriggerVolume,
};
use concinnity_core::gfx::lines::Line;
use outlines::shapes;

impl EditorHook {
    // Append this frame's extent outlines. Nothing while the HUD is hidden or
    // the sim is playing; otherwise the selection always outlines and the
    // Display menu's category toggles add their always-on layer.
    pub(super) fn push_extent_lines(&self, world: &World, vp: [f32; 2], out: &mut Vec<Line>) {
        if !self.hud_visible || self.sim.playing() {
            return;
        }
        if self.selection.iter().next().is_none() && !self.extent_show.any() {
            return;
        }
        let mut budget = outlines::MAX_OUTLINED;
        // Selected entries first, so the cap can never starve the selection.
        self.push_outline_pass(world, vp, out, true, &mut budget);
        self.push_outline_pass(world, vp, out, false, &mut budget);
    }

    fn push_outline_pass(
        &self,
        world: &World,
        vp: [f32; 2],
        out: &mut Vec<Line>,
        selected_pass: bool,
        budget: &mut usize,
    ) {
        let colliders_on = self.extent_show.contains(outlines::Category::Colliders);
        for e in &self.entries {
            if *budget == 0 {
                return;
            }
            let (Some(name), Some(ty)) = (entry_name(e), entry_type(e)) else {
                continue;
            };
            if self.name_hidden(name) || self.selection.contains(name) != selected_pass {
                continue;
            }
            let category = outlines::Category::of_type(ty);
            let wants_extent =
                category.is_some_and(|c| selected_pass || self.extent_show.contains(c));
            if !wants_extent && !colliders_on {
                continue;
            }
            let Some(entity) = billboard_drive::entity_by_name(world, name) else {
                continue;
            };
            let before = out.len();
            if wants_extent {
                push_entity_extent(world, entity, e, ty, selected_pass, vp, out);
            }
            if colliders_on {
                push_collider_outline(world, entity, out);
            }
            if out.len() > before {
                *budget -= 1;
            }
        }
    }
}

fn push_entity_extent(
    world: &World,
    entity: crate::ecs::Entity,
    entry: &serde_json::Value,
    ty: &str,
    selected: bool,
    vp: [f32; 2],
    out: &mut Vec<Line>,
) {
    let s = outlines::stroke(ty, selected);
    match ty {
        "TriggerVolume" => {
            let Some(volume) = world.get::<TriggerVolume>(entity) else {
                return;
            };
            let transform = anchored_transform(world, entity, volume.position, volume.rotation_deg);
            let c = &volume.collider;
            match c.shape.as_str() {
                "ball" => shapes::push_sphere(out, transform.position, c.radius, s),
                "capsule" => {
                    shapes::push_capsule(
                        out,
                        &transform.model_matrix(),
                        c.radius,
                        c.half_height,
                        s,
                    );
                }
                _ => shapes::push_box(out, &transform.model_matrix(), c.half_extents, s),
            }
        }
        "PointLight" => {
            let Some(light) = world.get::<PointLight>(entity) else {
                return;
            };
            let center = anchored_position(world, entity, light.position);
            shapes::push_sphere(out, center, light.range, s);
        }
        "SpotLight" => {
            let Some(light) = world.get::<SpotLight>(entity) else {
                return;
            };
            let apex = anchored_position(world, entity, light.position);
            let dir = light.unit_direction();
            let outer = light
                .outer_angle
                .clamp(0.0, SPOT_MAX_ANGLE_DEG)
                .to_radians();
            shapes::push_cone(out, apex, dir, outer, light.range, s);
            // The full-bright inner cone reads as one dimmer circle on the
            // same range sphere.
            let inner = light.inner_angle.to_radians().clamp(0.0, outer);
            if inner > 0.0 && light.range > 0.0 {
                let center = [
                    apex[0] + dir[0] * light.range * inner.cos(),
                    apex[1] + dir[1] * light.range * inner.cos(),
                    apex[2] + dir[2] * light.range * inner.cos(),
                ];
                shapes::push_circle(
                    out,
                    center,
                    dir,
                    light.range * inner.sin(),
                    outlines::inner_stroke(s),
                );
            }
        }
        "RectAreaLight" => {
            let Some(light) = world.get::<RectAreaLight>(entity) else {
                return;
            };
            let centre = anchored_position(world, entity, light.centre);
            shapes::push_rect(
                out,
                centre,
                unit_or_down(light.normal),
                light.half_size,
                light.range,
                s,
            );
        }
        "ReflectionProbe" => {
            let Some(probe) = world.get::<ReflectionProbe>(entity) else {
                return;
            };
            let position = anchored_position(world, entity, probe.position);
            let model = Transform {
                position,
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            }
            .model_matrix();
            shapes::push_box(out, &model, probe.half_extents, s);
        }
        "Camera3D" => {
            // The live Camera3D is the editor's own viewpoint (fly moves
            // it), so the frustum builds from the authored args, anchored
            // on the seeded Transform so a gizmo drag carries it.
            let args = form::working_args(ty, entry.get("args").and_then(|v| v.as_object()));
            let num = |key: &str, default: f32| {
                args.get(key)
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(default)
            };
            let authored = billboards::position_of(&args).unwrap_or([0.0; 3]);
            let origin = anchored_position(world, entity, authored);
            let view = concinnity_core::gfx::camera::view_matrix(
                origin,
                num("yaw", 0.0),
                num("pitch", 0.0),
            );
            let right = [view[0][0], view[1][0], view[2][0]];
            let up = [view[0][1], view[1][1], view[2][1]];
            let forward = [-view[0][2], -view[1][2], -view[2][2]];
            let aspect = if vp[1] > 0.0 {
                vp[0] / vp[1]
            } else {
                16.0 / 9.0
            };
            shapes::push_frustum(
                out,
                &shapes::Frustum {
                    origin,
                    right,
                    up,
                    forward,
                    fov_y_rad: num("fov_y_degrees", 75.0).to_radians(),
                    aspect,
                    near: num("near", 0.05),
                    far: num("far", 200.0),
                },
                s,
            );
        }
        _ => {}
    }
}

// The entity's collider wireframe (the Colliders category), through its live
// Transform so scale and a gizmo drag both apply.
fn push_collider_outline(world: &World, entity: crate::ecs::Entity, out: &mut Vec<Line>) {
    let (Some(collider), Some(transform)) = (
        world.get::<Collider>(entity),
        world.get::<Transform>(entity),
    ) else {
        return;
    };
    let s = outlines::stroke("Collider", false);
    let c = &collider.0;
    match c.shape.as_str() {
        "ball" => {
            let scale = transform.scale.iter().fold(0.0f32, |a, &b| a.max(b));
            shapes::push_sphere(out, transform.position, c.radius * scale, s);
        }
        "capsule" => {
            shapes::push_capsule(out, &transform.model_matrix(), c.radius, c.half_height, s)
        }
        _ => shapes::push_box(out, &transform.model_matrix(), c.half_extents, s),
    }
}

// The entity's position, preferring the seeded / gizmo-dragged Transform over
// the component's authored field.
fn anchored_position(world: &World, entity: crate::ecs::Entity, authored: [f32; 3]) -> [f32; 3] {
    world
        .get::<Transform>(entity)
        .map(|t| t.position)
        .unwrap_or(authored)
}

// The entity's full pose, preferring the live Transform.
fn anchored_transform(
    world: &World,
    entity: crate::ecs::Entity,
    position: [f32; 3],
    rotation_deg: [f32; 3],
) -> Transform {
    world
        .get::<Transform>(entity)
        .cloned()
        .unwrap_or(Transform {
            position,
            rotation_deg,
            scale: [1.0; 3],
        })
}

fn unit_or_down(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 {
        [0.0, -1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VP: [f32; 2] = [1280.0, 720.0];

    fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    fn entry(name: &str, ty: &str) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": {}})
    }

    // A world holding one named entity carrying `component`, resolvable
    // through the same EntityByName path the gizmo and billboards use.
    fn world_with<C: crate::ecs::ComponentSlot>(name: &str, component: C) -> World {
        crate::ecs::asset_id::reset_interner();
        let id = crate::ecs::asset_id::intern(name);
        let mut world = World::new_empty();
        let entity = world.push(component);
        let mut by_name = std::collections::BTreeMap::new();
        by_name.insert(id, entity);
        world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
        world
    }

    fn lines_of(h: &EditorHook, world: &World) -> Vec<Line> {
        let mut out = Vec::new();
        h.push_extent_lines(world, VP, &mut out);
        out
    }

    #[test]
    fn a_selected_point_light_outlines_its_range_sphere() {
        let world = world_with(
            "lamp",
            PointLight {
                position: [1.0, 2.0, 3.0],
                range: 6.0,
                ..Default::default()
            },
        );
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        assert!(
            lines_of(&h, &world).is_empty(),
            "nothing selected, nothing drawn"
        );

        h.selection.replace("lamp".to_string());
        let lines = lines_of(&h, &world);
        assert_eq!(lines.len(), 3 * shapes::CIRCLE_SEGMENTS);
        // Every point sits on the range sphere around the light.
        for l in &lines {
            let d = [l.start[0] - 1.0, l.start[1] - 2.0, l.start[2] - 3.0];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((r - 6.0).abs() < 1e-3, "{r}");
        }
    }

    #[test]
    fn a_category_toggle_outlines_without_selection_and_dimmer() {
        let world = world_with(
            "lamp",
            PointLight {
                range: 6.0,
                ..Default::default()
            },
        );
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        h.extent_show = h.extent_show.toggled(outlines::Category::Lights);
        let layer = lines_of(&h, &world);
        assert_eq!(layer.len(), 3 * shapes::CIRCLE_SEGMENTS);

        h.selection.replace("lamp".to_string());
        let selected = lines_of(&h, &world);
        assert_eq!(selected.len(), layer.len());
        assert!(selected[0].start_color[3] > layer[0].start_color[3]);
        assert!(selected[0].width_px > layer[0].width_px);
    }

    #[test]
    fn a_spot_light_draws_its_cone_and_inner_circle() {
        let world = world_with(
            "spot",
            SpotLight {
                position: [0.0, 5.0, 0.0],
                direction: [0.0, -1.0, 0.0],
                range: 10.0,
                inner_angle: 18.0,
                outer_angle: 30.0,
                ..Default::default()
            },
        );
        let mut h = hook(vec![entry("spot", "SpotLight")]);
        h.selection.replace("spot".to_string());
        let lines = lines_of(&h, &world);
        assert_eq!(
            lines.len(),
            shapes::CONE_SIDES + 2 * shapes::CIRCLE_SEGMENTS,
            "outer cone + inner circle"
        );
        // The inner circle carries the dimmer stroke.
        let faintest = lines
            .iter()
            .map(|l| l.start_color[3])
            .fold(f32::MAX, f32::min);
        assert!(faintest < lines[0].start_color[3]);
    }

    #[test]
    fn trigger_volume_shapes_follow_the_live_transform() {
        let mut world = world_with(
            "zone",
            TriggerVolume {
                position: [0.0; 3],
                collider: crate::assets::PropCollider {
                    shape: "ball".to_string(),
                    radius: 2.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let mut h = hook(vec![entry("zone", "TriggerVolume")]);
        h.selection.replace("zone".to_string());

        // A gizmo-dragged Transform moves the outline off the authored spot.
        let e = world
            .resource::<concinnity_core::ecs::EntityByName>()
            .unwrap()
            .0
            .values()
            .next()
            .copied()
            .unwrap();
        world.insert(
            e,
            Transform {
                position: [10.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
        let lines = lines_of(&h, &world);
        assert_eq!(lines.len(), 3 * shapes::CIRCLE_SEGMENTS);
        for l in &lines {
            let d = [l.start[0] - 10.0, l.start[1], l.start[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((r - 2.0).abs() < 1e-3, "outline follows the Transform");
        }
    }

    #[test]
    fn colliders_show_only_through_their_toggle() {
        let mut world = world_with(
            "crate",
            Collider(crate::assets::PropCollider {
                shape: "cuboid".to_string(),
                half_extents: [1.0, 1.0, 1.0],
                ..Default::default()
            }),
        );
        let e = world
            .resource::<concinnity_core::ecs::EntityByName>()
            .unwrap()
            .0
            .values()
            .next()
            .copied()
            .unwrap();
        world.insert(e, Transform::default());
        let mut h = hook(vec![entry("crate", "Prop")]);

        // Selection alone never draws a collider (props carry the highlight
        // rect instead).
        h.selection.replace("crate".to_string());
        assert!(lines_of(&h, &world).is_empty());

        h.extent_show = h.extent_show.toggled(outlines::Category::Colliders);
        assert_eq!(lines_of(&h, &world).len(), shapes::BOX_EDGES);
    }

    #[test]
    fn hidden_entries_and_a_hidden_hud_draw_nothing() {
        let world = world_with(
            "lamp",
            PointLight {
                range: 6.0,
                ..Default::default()
            },
        );
        let mut h = hook(vec![entry("lamp", "PointLight")]);
        h.selection.replace("lamp".to_string());
        assert!(!lines_of(&h, &world).is_empty());

        h.hidden_assets.insert("lamp".to_string());
        assert!(
            lines_of(&h, &world).is_empty(),
            "editor-hidden entries skip"
        );
        h.hidden_assets.clear();

        h.hud_visible = false;
        assert!(
            lines_of(&h, &world).is_empty(),
            "F1 clears the outlines too"
        );
    }
}
