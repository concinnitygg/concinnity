//! Rebuilding the part of a built panel whose shape follows the editor's
//! model, such as a form's fields: what a layout holds is despawned and built
//! again, only when the shape changes.

use std::collections::HashSet;

use concinnity_engine::components::{Layout, LayoutChild};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::{BuiltPanel, UiBuilder};

// Every element and layout `host` holds, however deep.
fn held(world: &World, host: AssetId) -> Vec<AssetId> {
    let mut out = Vec::new();
    let mut walk: Vec<AssetId> = Vec::new();
    let children = |id: AssetId| {
        world.get_by_id::<Layout>(id).map_or_default(|l| {
            l.children
                .iter()
                .filter_map(|c| c.item)
                .map(concinnity_engine::ecs::Ref::id)
                .collect::<Vec<_>>()
        })
    };
    walk.extend(children(host));
    while let Some(id) = walk.pop() {
        out.push(id);
        walk.extend(children(id));
    }
    out
}

// Replace what the layout `host` of `panel` holds with the children `build`
// makes: the elements it held are despawned, and the new ones join the panel.
pub(crate) fn rebuild(
    world: &mut World,
    panel: &mut BuiltPanel,
    host: AssetId,
    build: impl FnOnce(&mut UiBuilder<'_>) -> Option<Vec<LayoutChild>>,
) {
    let held = held(world, host);
    if let Some(layout) = world.get_mut_by_id::<Layout>(host) {
        layout.children.clear();
    }
    for &id in &held {
        if let Some(entity) = world.entity_of(id) {
            world.despawn(entity);
        }
    }
    let old: HashSet<AssetId> = held.into_iter().collect();
    panel.elements.retain(|id| !old.contains(id));
    let Some(screen) = panel.screen else {
        return;
    };
    let mut ui = UiBuilder::new(world, screen, panel.font);
    let children = build(&mut ui).unwrap_or_default();
    let pushed = ui.take_pushed();
    panel.elements.extend(pushed);
    if let Some(layout) = world.get_mut_by_id::<Layout>(host) {
        layout.children = children;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::TextLabel;

    #[test]
    fn a_rebuild_replaces_what_the_layout_held() {
        let mut world = World::new();
        let mut ui = UiBuilder::new(&mut world, AssetId(9), None);
        let first = ui.label("first", [1.0; 3]).unwrap();
        let old = first.item.unwrap().id();
        let host = ui
            .stack(
                concinnity_engine::components::LayoutDirection::Column,
                vec![first],
                0.0,
                0.0,
            )
            .unwrap();
        let mut panel = BuiltPanel {
            screen: Some(AssetId(9)),
            elements: ui.take_pushed(),
            ..Default::default()
        };
        rebuild(&mut world, &mut panel, host, |ui| {
            Some(vec![ui.label("a", [1.0; 3])?, ui.label("b", [1.0; 3])?])
        });
        assert!(world.entity_of(old).is_none(), "the old label is gone");
        let layout = world.get_by_id::<Layout>(host).unwrap();
        assert_eq!(layout.children.len(), 2);
        let ids: Vec<AssetId> = layout
            .children
            .iter()
            .filter_map(|c| c.item)
            .map(concinnity_engine::ecs::Ref::id)
            .collect();
        assert!(ids.iter().all(|id| panel.elements.contains(id)));
        assert!(!panel.elements.contains(&old));
        assert_eq!(world.get_by_id::<TextLabel>(ids[1]).unwrap().content, "b");
    }
}
