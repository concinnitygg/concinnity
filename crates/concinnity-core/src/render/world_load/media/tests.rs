use super::*;
use crate::components::{StoryImage, StoryNode, StoryPage, StoryStage};
use crate::ecs::World;
use alloc::vec;

#[test]
fn a_world_without_the_media_loads_without_it() {
    let mut world = World::new();
    let mut ctx = world.context();
    assert_eq!(environment_map_payload(&mut ctx), Some(None));
    assert_eq!(color_lut_payload(&mut ctx), Some(None));
    assert_eq!(environment_background(&ctx), Some(true));
}

// A baked payload is read from the entry itself; only the first declaration
// is used.
#[test]
fn the_first_baked_payload_is_the_one_read() {
    let mut world = World::new();
    world.insert_resource(ColorLutTable(vec![
        ResourceEntry::baked(vec![1, 2, 3]),
        ResourceEntry::baked(vec![9]),
    ]));
    assert_eq!(
        color_lut_payload(&mut world.context()),
        Some(Some(vec![1, 2, 3]))
    );
}

// A declared entry with no payload at all has nothing to read.
#[test]
fn an_entry_without_a_payload_reads_none() {
    let mut world = World::new();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    assert_eq!(environment_map_payload(&mut world.context()), Some(None));
}

// Sprites and story stages share the pool: each texture once, in handle order.
#[test]
fn sprite_textures_gather_once_in_handle_order() {
    let image = |index| {
        Some(StoryImage {
            texture: TextureHandle::new(index),
            ..Default::default()
        })
    };
    let mut world = World::new();
    world.add_component(Sprite {
        texture: Some(TextureHandle::new(4)),
        ..Default::default()
    });
    world.add_component(Sprite::default());
    world.add_component(Story {
        nodes: vec![StoryNode {
            pages: vec![StoryPage {
                stage: StoryStage {
                    bg: image(2),
                    left: image(4),
                    ..Default::default()
                },
                ..Default::default()
            }],
            choice_stage: StoryStage {
                right: image(1),
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    });
    let handles = sprite_texture_handles(&world.context());
    assert_eq!(
        handles,
        [1, 2, 4].map(TextureHandle::new),
        "sorted and deduplicated"
    );
}
