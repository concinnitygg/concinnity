use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use super::*;
use crate::components::{
    Camera3D, DebugHud, EngineDefaults, LoadingOverlay, PhysicsConfig, ProceduralMesh, Prop,
    PropBody, Scene, Screen, Sprite, StatHud, StreamingConfig, TextLabel,
};
use crate::ecs::World;
use crate::resource::{EnvironmentMapTable, FontTable, MaterialTable, MeshTable, ResourceEntry};

// Run the pass over a world, as `World::start` does.
fn complete(world: &mut World) -> Result<(), CnResult> {
    let mut ctx = world.context();
    run(&mut ctx)
}

// A world with the render marker, so the drawn defaults apply.
fn rendering() -> World {
    let mut world = World::new();
    world.add_component(GraphicsConfig::default());
    world
}

fn labels(world: &World) -> Vec<TextLabel> {
    world.query::<TextLabel>().cloned().collect()
}

#[test]
fn a_rendering_world_gets_the_debug_hud_its_chips_and_a_font() {
    let mut world = rendering();
    complete(&mut world).unwrap();

    let hud = world.query::<DebugHud>().next().expect("a debug HUD");
    let named = [
        hud.passes_label,
        hud.mouse_label,
        hud.camera_label,
        hud.sys_label,
    ];
    assert!(named.iter().all(Option::is_some), "every chip is named");

    let chips = labels(&world);
    assert_eq!(chips.len(), 4);
    // Every chip is one of the HUD's, drawn with the one baked face.
    let font = world.resource::<FontTable>().expect("a font table");
    assert_eq!(font.len(), 1);
    assert!(
        font.0[0].baked_bytes().is_some(),
        "the face is baked, not compiled"
    );
    for chip in &chips {
        assert_eq!(chip.font, Some(crate::ecs::FontHandle(0)));
        assert!(named.contains(&Some(chip.asset_id)));
        assert!(chip.asset_id.is_minted());
    }
}

#[test]
fn a_world_that_does_not_render_gets_no_hud() {
    let mut world = World::new();
    complete(&mut world).unwrap();
    assert!(world.query::<DebugHud>().next().is_none());
    assert!(labels(&world).is_empty());
    assert!(world.resource::<FontTable>().is_none());
}

#[test]
fn an_authored_debug_hud_keeps_the_labels_it_names() {
    let mut world = rendering();
    world.add_component(DebugHud {
        passes_label: Some(AssetId(7)),
        ..Default::default()
    });
    complete(&mut world).unwrap();

    assert_eq!(world.query::<DebugHud>().count(), 1);
    let hud = world.query::<DebugHud>().next().unwrap();
    assert_eq!(hud.passes_label, Some(AssetId(7)));
    // Only the three unset slots minted a chip.
    assert_eq!(labels(&world).len(), 3);
}

// The stats strip is driven by menu toggles the runtime cannot see, so it is
// completed where it is declared and never synthesized.
#[test]
fn a_stat_hud_is_completed_but_never_synthesized() {
    let mut world = rendering();
    complete(&mut world).unwrap();
    assert!(world.query::<StatHud>().next().is_none());

    let mut world = rendering();
    world.add_component(StatHud::default());
    complete(&mut world).unwrap();
    let hud = world.query::<StatHud>().next().unwrap();
    assert!(hud.fps_label.is_some() && hud.edr_label.is_some());
    // Five stat chips plus the debug HUD's four, all on one font.
    assert_eq!(labels(&world).len(), 9);
    assert_eq!(world.resource::<FontTable>().expect("fonts").len(), 1);
}

#[test]
fn the_hud_toggles_opt_out() {
    let mut world = rendering();
    world.add_component(StatHud::default());
    world.add_component(EngineDefaults {
        hud: false,
        debug_hud: false,
        ..Default::default()
    });
    complete(&mut world).unwrap();

    assert!(world.query::<DebugHud>().next().is_none());
    assert!(labels(&world).is_empty());
    // The directive is a load-time instruction, not something a system reads.
    assert!(world.query::<EngineDefaults>().next().is_none());
}

#[test]
fn two_engine_defaults_are_an_error() {
    let mut world = rendering();
    world.add_component(EngineDefaults::default());
    world.add_component(EngineDefaults::default());
    assert_eq!(complete(&mut world), Err(CnResult::InvalidState));
}

// A world lit by an environment map gets the geometry that displays it: a
// skybox mesh baked at start, its material, and the prop that places it.
#[test]
fn an_environment_map_gets_the_sky() {
    let mut world = rendering();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    complete(&mut world).unwrap();

    let mesh = world
        .query::<ProceduralMesh>()
        .next()
        .expect("the skybox mesh");
    assert_eq!(mesh.generator, "skybox");
    assert!(mesh.locator.is_none(), "baked at start, not compiled");
    let payloads = world
        .resource::<crate::resource::RuntimeMeshPayloads>()
        .expect("a baked payload");
    assert!(payloads.get(mesh.asset_id).is_some());

    let prop = world.query::<Prop>().next().expect("the sky prop");
    assert_eq!(prop.mesh, Some(crate::ecs::MeshHandle(0)));
    assert_eq!(prop.material, Some(crate::ecs::MaterialHandle(0)));
    assert_eq!(
        world.resource::<MaterialTable>().expect("materials").len(),
        1
    );
}

// The baked mesh takes the first handle past everything the build assigned, so
// a handle already baked into a Prop still reaches its own geometry.
#[test]
fn the_baked_sky_mesh_trails_every_build_assigned_handle() {
    let mut world = rendering();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    // Two compiled Mesh resources and one compiled ProceduralMesh: handles 0..3.
    world.insert_resource(MeshTable(vec![
        ResourceEntry::default(),
        ResourceEntry::default(),
    ]));
    world.add_component(ProceduralMesh {
        asset_id: AssetId(1),
        generator: "box".to_string(),
        locator: Some(crate::ecs::PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 1,
        }),
        ..Default::default()
    });
    complete(&mut world).unwrap();

    let prop = world.query::<Prop>().next().expect("the sky prop");
    assert_eq!(prop.mesh, Some(crate::ecs::MeshHandle(3)));
}

#[test]
fn the_sky_mesh_tracks_the_camera_far_plane_and_caps() {
    let size_for = |far: f32| {
        let mut world = rendering();
        world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
        world.add_component(Camera3D::bake(crate::components::cook::Camera3D {
            far,
            ..Default::default()
        }));
        complete(&mut world).unwrap();
        world.query::<ProceduralMesh>().next().unwrap().size
    };
    assert_eq!(size_for(100.0), Some(90.0));
    assert_eq!(size_for(900.0), Some(400.0));
}

#[test]
fn a_world_with_its_own_skybox_geometry_gets_no_sky() {
    let mut world = rendering();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    world.add_component(ProceduralMesh {
        asset_id: AssetId(1),
        generator: "skybox".to_string(),
        ..Default::default()
    });
    complete(&mut world).unwrap();
    assert!(world.query::<Prop>().next().is_none());
    assert_eq!(world.query::<ProceduralMesh>().count(), 1);
}

#[test]
fn no_environment_map_means_no_sky() {
    let mut world = rendering();
    complete(&mut world).unwrap();
    assert!(world.query::<Prop>().next().is_none());
    assert!(world.query::<ProceduralMesh>().next().is_none());
}

// The cube example turns the sky off so its spin behavior, scoped to `Prop`,
// still resolves to the one prop the world declares.
#[test]
fn opting_out_of_the_sky_leaves_the_world_its_own_props() {
    let mut world = rendering();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    world.add_component(Prop::default());
    world.add_component(EngineDefaults {
        sky: false,
        ..Default::default()
    });
    complete(&mut world).unwrap();

    assert_eq!(world.query::<Prop>().count(), 1);
    assert!(world.query::<ProceduralMesh>().next().is_none());
    assert!(world.resource::<MaterialTable>().is_none());
}

#[test]
fn physics_content_gets_the_config_its_simulation_runs_on() {
    let mut world = World::new();
    world.add_component(PropBody::default());
    complete(&mut world).unwrap();

    let config = world.query::<PhysicsConfig>().next().expect("a config");
    assert_eq!(
        config.spawn_headroom,
        PhysicsConfig::default().spawn_headroom
    );
    // Physics is not a rendering concern, so the headless tier gets it too.
    assert!(world.query::<DebugHud>().next().is_none());
}

#[test]
fn an_authored_physics_config_is_left_alone() {
    let mut world = World::new();
    world.add_component(PropBody::default());
    world.add_component(PhysicsConfig {
        spawn_headroom: 8,
        ..Default::default()
    });
    complete(&mut world).unwrap();
    assert_eq!(world.query::<PhysicsConfig>().count(), 1);
    assert_eq!(
        world
            .query::<PhysicsConfig>()
            .next()
            .unwrap()
            .spawn_headroom,
        8
    );
}

#[test]
fn a_world_without_physics_content_gets_no_config() {
    let mut world = rendering();
    complete(&mut world).unwrap();
    assert!(world.query::<PhysicsConfig>().next().is_none());
}

#[test]
fn the_physics_toggle_opts_out() {
    let mut world = World::new();
    world.add_component(PropBody::default());
    world.add_component(EngineDefaults {
        physics_config: false,
        ..Default::default()
    });
    complete(&mut world).unwrap();
    assert!(world.query::<PhysicsConfig>().next().is_none());
}

fn streamed() -> World {
    let mut world = rendering();
    world.add_component(Scene::default());
    world.add_component(StreamingConfig::default());
    world
}

#[test]
fn a_streamed_world_gets_the_loading_overlay_and_its_pieces() {
    let mut world = streamed();
    complete(&mut world).unwrap();

    let overlay = world
        .query::<LoadingOverlay>()
        .next()
        .cloned()
        .expect("an overlay");
    let screen = overlay.screen.expect("a screen");
    assert_eq!(world.query::<Screen>().count(), 1);
    assert_eq!(world.query::<Screen>().next().unwrap().asset_id, screen);

    // Backdrop, track, and fill, each on the overlay's screen.
    let sprites: Vec<Sprite> = world.query::<Sprite>().cloned().collect();
    assert_eq!(sprites.len(), 3);
    assert!(sprites.iter().all(|s| s.screen == Some(screen)));
    let named = [overlay.backdrop, overlay.track, overlay.fill];
    assert!(sprites.iter().all(|s| named.contains(&Some(s.asset_id))));

    let label = overlay.label.expect("a label");
    let text = labels(&world)
        .into_iter()
        .find(|l| l.asset_id == label)
        .expect("the label was injected");
    assert_eq!(text.content, "Loading");
    assert_eq!(text.screen, Some(screen));
}

// Declaring an overlay is the opt-in: it is completed even where none would be
// synthesized, and every field it names is kept.
#[test]
fn an_authored_overlay_is_completed_and_keeps_what_it_names() {
    let mut world = rendering();
    world.add_component(LoadingOverlay {
        backdrop: Some(AssetId(4)),
        ..Default::default()
    });
    complete(&mut world).unwrap();

    let overlay = world.query::<LoadingOverlay>().next().unwrap();
    assert_eq!(overlay.backdrop, Some(AssetId(4)));
    assert!(overlay.screen.is_some() && overlay.track.is_some() && overlay.fill.is_some());
    // Only track and fill were minted; the authored backdrop stands.
    assert_eq!(world.query::<Sprite>().count(), 2);
}

#[test]
fn a_world_without_scenes_or_streaming_gets_no_overlay() {
    for extra in 0..2 {
        let mut world = rendering();
        if extra == 0 {
            world.add_component(Scene::default());
        } else {
            world.add_component(StreamingConfig::default());
        }
        complete(&mut world).unwrap();
        assert!(world.query::<LoadingOverlay>().next().is_none());
        assert!(world.query::<Screen>().next().is_none());
    }
}

#[test]
fn the_loading_overlay_toggle_opts_out() {
    let mut world = streamed();
    world.add_component(EngineDefaults {
        loading_overlay: false,
        ..Default::default()
    });
    complete(&mut world).unwrap();
    assert!(world.query::<LoadingOverlay>().next().is_none());
}

// Every injected name is minted from the reserved range and unique, so a
// cross-reference the pass writes cannot land on a declared asset.
#[test]
fn minted_names_are_unique_and_out_of_the_declared_range() {
    let mut world = streamed();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    world.add_component(StatHud::default());
    complete(&mut world).unwrap();

    let mut ids: Vec<AssetId> = world.query::<TextLabel>().map(|l| l.asset_id).collect();
    ids.extend(world.query::<Sprite>().map(|s| s.asset_id));
    ids.extend(world.query::<Screen>().map(|s| s.asset_id));
    ids.extend(world.query::<Prop>().map(|p| p.asset_id));
    ids.extend(world.query::<ProceduralMesh>().map(|m| m.asset_id));
    assert!(ids.iter().all(|id| id.is_minted()), "{ids:?}");
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "minted names collide: {ids:?}");
}

// Every default yields to what is already there, so a second run is a no-op.
#[test]
fn completing_an_already_completed_world_adds_nothing() {
    let mut world = streamed();
    world.insert_resource(EnvironmentMapTable(vec![ResourceEntry::default()]));
    world.add_component(PropBody::default());
    complete(&mut world).unwrap();
    let census = world.component_census();
    let fonts = world.resource::<FontTable>().expect("fonts").len();

    complete(&mut world).unwrap();
    assert_eq!(world.component_census(), census);
    assert_eq!(world.resource::<FontTable>().expect("fonts").len(), fonts);
}
