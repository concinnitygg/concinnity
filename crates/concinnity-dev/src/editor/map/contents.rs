// What a place holds: the entries whose membership arg names it, counted by
// type. A card's second line is what makes a world of one scene worth looking
// at, so it says what is in that scene rather than that the scene exists.

use concinnity_cook::authoring::flow::Place;
use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::WorldJsonlAsset;

// How many kinds a card names before the rest is left off. The line is one
// clipped row of a card wide, so a third kind would not be read.
const KINDS: usize = 2;

/// What the entries naming `place` amount to. A `SceneImport` is one of them:
/// a scene built from an imported file holds the import rather than the props
/// it expands into, and that is the whole of what such a scene contains.
pub(super) fn held_by(assets: &[WorldJsonlAsset], place: &str) -> String {
    phrase(&counts(assets, |asset| names(asset) == Some(place)))
}

/// What the world holds outside its places: everything naming none of them and
/// not being one. A world declaring nowhere to be holds all of it.
pub(super) fn unplaced(assets: &[WorldJsonlAsset], places: &[Place]) -> String {
    phrase(&counts(assets, |asset| {
        names(asset).is_none() && !places.iter().any(|place| place.id == asset.id)
    }))
}

// The place an entry sits in, the way `build_only::membership` spells it: a
// `scene` on scene-scoped content, a `screen` on an overlay element. An empty
// one names nothing, which is how an author writes "bound to no place".
fn names(asset: &WorldJsonlAsset) -> Option<&str> {
    ["scene", "screen"].into_iter().find_map(|key| {
        asset
            .args
            .get(key)?
            .as_str()
            .filter(|name| !name.is_empty())
    })
}

// How many of each type are kept, most of them first and ties by name so one
// world always reads the same way.
fn counts(
    assets: &[WorldJsonlAsset],
    keep: impl Fn(&WorldJsonlAsset) -> bool,
) -> Vec<(RegisteredType, usize)> {
    let mut counts: Vec<(RegisteredType, usize)> = Vec::new();
    for asset in assets.iter().filter(|asset| keep(asset)) {
        match counts.iter_mut().find(|(ty, _)| *ty == asset.asset_type) {
            Some((_, held)) => *held += 1,
            None => counts.push((asset.asset_type, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
    counts
}

fn phrase(counts: &[(RegisteredType, usize)]) -> String {
    counts
        .iter()
        .take(KINDS)
        .map(|(ty, held)| format!("{held} {}", ty.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use concinnity_cook::authoring::flow::places;
    use serde_json::json;

    use super::*;

    fn asset(ty: RegisteredType, id: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type: ty,
            args,
        }
    }

    fn bistro() -> Vec<WorldJsonlAsset> {
        vec![
            asset(RegisteredType::Scene, "bistro", json!({})),
            asset(
                RegisteredType::GraphicsConfig,
                "GraphicsConfig#0",
                json!({}),
            ),
            asset(
                RegisteredType::SceneImport,
                "SceneImport#0",
                json!({"source": "b.fbx", "scene": "bistro"}),
            ),
        ]
    }

    // The world the counts exist for: one scene whose whole content is the file
    // it imports. A card saying only "scene" would be worth nothing.
    #[test]
    fn an_imported_scene_holds_the_import() {
        assert_eq!(held_by(&bistro(), "bistro"), "1 SceneImport");
    }

    #[test]
    fn an_overlay_element_is_held_by_the_screen_it_names() {
        let world = [
            asset(RegisteredType::Screen, "pause", json!({})),
            asset(RegisteredType::HitRegion, "a", json!({"screen": "pause"})),
            asset(RegisteredType::HitRegion, "b", json!({"screen": "pause"})),
            asset(RegisteredType::TextLabel, "t", json!({"screen": "pause"})),
            asset(RegisteredType::TextLabel, "u", json!({"screen": "other"})),
        ];
        assert_eq!(held_by(&world, "pause"), "2 HitRegion, 1 TextLabel");
    }

    #[test]
    fn a_place_holding_nothing_says_nothing() {
        assert_eq!(held_by(&bistro(), "GraphicsConfig#0"), "");
    }

    // Membership is the arg, never the name: an entry bound to no place is not
    // held by one whose name happens to sit in its args.
    #[test]
    fn an_empty_membership_arg_names_no_place() {
        let world = [
            asset(RegisteredType::Prop, "a", json!({"scene": ""})),
            asset(RegisteredType::Prop, "b", json!({"mesh": "bistro"})),
        ];
        assert_eq!(held_by(&world, "bistro"), "");
        assert_eq!(held_by(&world, ""), "");
    }

    // More of a kind reads before less of one, however the world declared them.
    #[test]
    fn the_largest_kinds_read_first_and_the_rest_are_left_off() {
        let mut world = vec![asset(RegisteredType::Scene, "s", json!({}))];
        for (ty, n) in [
            (RegisteredType::Decal, 1),
            (RegisteredType::Prop, 3),
            (RegisteredType::PointLight, 2),
        ] {
            for i in 0..n {
                world.push(asset(
                    ty,
                    &format!("{}{i}", ty.as_str()),
                    json!({"scene": "s"}),
                ));
            }
        }
        assert_eq!(held_by(&world, "s"), "3 Prop, 2 PointLight");
    }

    // A world declaring nowhere to be still declares what it is made of, which
    // is the whole of what its map has to show.
    #[test]
    fn a_world_with_no_place_holds_everything_it_declares() {
        let world = [
            asset(RegisteredType::Material, "m0", json!({})),
            asset(RegisteredType::Material, "m1", json!({})),
            asset(RegisteredType::Prop, "p0", json!({})),
            asset(RegisteredType::Camera3D, "c", json!({})),
        ];
        assert_eq!(unplaced(&world, &places(&world)), "2 Material, 1 Camera3D");
    }

    // A place is a card of its own, so counting it among what the world holds
    // loose would draw it twice.
    #[test]
    fn what_the_world_holds_loose_leaves_out_the_places_themselves() {
        let world = bistro();
        assert_eq!(unplaced(&world, &places(&world)), "1 GraphicsConfig");
    }
}
