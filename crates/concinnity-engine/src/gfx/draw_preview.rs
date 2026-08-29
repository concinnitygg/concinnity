// src/gfx/draw_preview.rs
//
// Live reassignment of a placement's draw slots. A Prop's `material` and
// `cull_distance` are read once at init: the draw list bakes the material into
// the GPU draw object and the entity keeps only the handle, so an editor
// changing either has nothing in the ECS the renderer re-reads. This is that
// seam. Each function records the backend call into the frame's op queue, where
// submission replays it before the next draw, and keeps the entity's renderer
// component in step so the running world and its draws agree on what is bound.
//
// What a rebuild would show is the standard. A material the world never loaded,
// a placement drawn from a Model (whose sub-meshes carry their own materials),
// and a backend that bakes per-object material state at build time are all
// refused here rather than reported as applied.

use crate::components::MeshRenderer;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ActiveRenderQueues, Entity, MaterialHandle, World};
use crate::gfx::material_entry::{self, MaterialEntry};
use concinnity_core::render::ops::RenderOps;

/// One draw slot's material as the backend holds it: the GPU uniforms plus the
/// texture-pool slots they sample, and the handle the entity records.
#[derive(Clone, Copy)]
pub struct DrawMaterial {
    // `None` for a placement drawing on its legacy texture (or on nothing),
    // which is what the entity's renderer then records.
    handle: Option<MaterialHandle>,
    entry: MaterialEntry,
}

impl DrawMaterial {
    /// Whether swapping this material for `other` is only a per-draw rewrite.
    /// The shader bucket picks the pipeline a draw renders under, and the
    /// transparency flags decide at init which pass draws it; both are baked
    /// into structures no per-draw call rebuilds, so a swap that moves either
    /// needs the world rebuilt.
    pub fn swappable_with(&self, other: &Self) -> bool {
        self.entry.shader_bucket == other.entry.shader_bucket
            && self.entry.uniforms.transparent == other.entry.uniforms.transparent
            && self.entry.uniforms.see_through == other.entry.uniforms.see_through
    }
}

/// Whether the running world can take a draw change: it has a graphics context
/// to record into, and its backend rewrites built draw slots in place.
pub fn is_available(world: &World) -> bool {
    world
        .resource::<ActiveRenderQueues>()
        .is_some_and(|slot| slot.0.is_some())
        && world
            .resource::<crate::ecs::ActiveDeviceCaps>()
            .is_some_and(|caps| caps.0.rewrites_draws)
}

/// The material the `Material` asset interned as `name` resolves to in the
/// running world. `None` when the world loaded no material under that name
/// (only a dev session records their identities, see
/// [`crate::resource::MaterialNames`]) or its record does not decode.
pub fn material(world: &World, name: AssetId) -> Option<DrawMaterial> {
    let handle = world
        .resource::<crate::resource::MaterialNames>()?
        .0
        .iter()
        .position(|&id| id == name.0)?;
    by_handle(world, MaterialHandle(handle as u32))
}

/// The material `entity`'s draw slots currently render with. `None` for an
/// entity the renderer draws from a Model: its materials come from the model's
/// sub-meshes, which the placement's own `material` never reaches.
pub fn drawn_material(world: &World, entity: Entity) -> Option<DrawMaterial> {
    let renderer = world.get::<MeshRenderer>(entity)?;
    match renderer.material {
        Some(handle) => by_handle(world, handle),
        None => Some(DrawMaterial {
            handle: None,
            entry: material_entry::from_texture(renderer.texture, texture_count(world)),
        }),
    }
}

/// Bind `material` to every draw slot `entity` owns, and record it on the
/// entity so the world it was drawn from agrees. `false` when the world has
/// nothing to record into or the entity owns no draws.
pub fn apply_material(world: &mut World, entity: Entity, material: DrawMaterial) -> bool {
    let draws = draws_of(world, entity);
    // Asked before the entity is touched: a refused change leaves the world
    // exactly as it was, so the caller's rebuild is what applies it.
    if draws.is_empty() || !is_available(world) {
        return false;
    }
    if let Some(renderer) = world.get_mut::<MeshRenderer>(entity) {
        renderer.material = material.handle;
    }
    let entry = material.entry;
    with_ops(world, |ops| {
        for draw in draws {
            ops.record(move |backend| {
                backend.set_draw_material(
                    draw as usize,
                    entry.uniforms,
                    entry.albedo_slot,
                    entry.normal_map_slot,
                );
            });
        }
    })
    .is_some()
}

/// Set the view-distance cutoff of every draw slot `entity` owns. Unlike a
/// material, this reaches a model-backed placement too: the cutoff is the
/// placement's own, applied to each of its sub-mesh draws.
pub fn apply_cull_distance(world: &mut World, entity: Entity, cull_distance: f32) -> bool {
    let draws = draws_of(world, entity);
    if draws.is_empty() || !is_available(world) {
        return false;
    }
    if let Some(renderer) = world.get_mut::<MeshRenderer>(entity) {
        renderer.cull_distance = cull_distance;
    } else if let Some(renderer) = world.get_mut::<crate::components::ModelRenderer>(entity) {
        renderer.cull_distance = cull_distance;
    }
    with_ops(world, |ops| {
        for draw in draws {
            ops.record(move |backend| backend.set_draw_cull_distance(draw as usize, cull_distance));
        }
    })
    .is_some()
}

// The material at `handle`, decoded from the running world's table by the same
// translation the draw list ran at init.
fn by_handle(world: &World, handle: MaterialHandle) -> Option<DrawMaterial> {
    let bytes = world
        .resource::<crate::resource::MaterialTable>()?
        .data_bytes(handle.index())?;
    let mat: crate::components::Material = postcard::from_bytes(bytes).ok()?;
    Some(DrawMaterial {
        handle: Some(handle),
        entry: material_entry::of(&mat, texture_count(world)).ok()?,
    })
}

// The shared texture pool's size, which the material's references index into.
fn texture_count(world: &World) -> usize {
    world
        .resource::<crate::resource::TextureTable>()
        .map_or(0, |t| t.len())
}

// The backend draw slots the entity owns; empty for an entity the renderer
// never gave one (nothing drawable, or a world with no graphics).
fn draws_of(world: &World, entity: Entity) -> Vec<u32> {
    world
        .get::<crate::components::RenderHandle>(entity)
        .map(|h| h.draws.to_vec())
        .unwrap_or_default()
}

// Run `f` against the frame's op queue, taking it for the call and parking it
// again after (the handoff every recording system uses). `None` when the world
// cannot take a draw change, which leaves the slot exactly as it was.
fn with_ops<R>(world: &mut World, f: impl FnOnce(&mut RenderOps) -> R) -> Option<R> {
    if !is_available(world) {
        return None;
    }
    let mut queues = world.resource_mut::<ActiveRenderQueues>()?.0.take()?;
    let out = f(&mut queues.ops);
    if let Some(slot) = world.resource_mut::<ActiveRenderQueues>() {
        slot.0 = Some(queues);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Material, RenderHandle};
    use crate::ecs::{RenderQueues, TextureHandle};
    use crate::gfx::backend::DeviceCapabilities;
    use crate::gfx::mock_backend::{Call, MockBackend, MockState, recording_backend};
    use crate::resource::{MaterialNames, MaterialTable, ResourceEntry, TextureTable};
    use concinnity_core::ecs::ShaderHandle;
    use std::sync::{Arc, Mutex};

    struct Fixture {
        world: World,
        backend: MockBackend,
        calls: Arc<Mutex<MockState>>,
    }

    // Two materials: "steel" at handle 0 and "glass" at handle 1, the latter
    // transparent so a swap between them is not a per-draw rewrite.
    fn materials() -> (Vec<ResourceEntry>, Vec<u32>) {
        let steel = Material {
            roughness: 0.5,
            albedo: Some(TextureHandle(1)),
            ..Default::default()
        };
        let glass = Material {
            transparent: true,
            ..Default::default()
        };
        let entry = |m: &Material| ResourceEntry {
            payload: None,
            data_bytes: postcard::to_allocvec(m).expect("serialises"),
        };
        (vec![entry(&steel), entry(&glass)], vec![10, 20])
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_caps(DeviceCapabilities::ALL)
        }

        fn with_caps(caps: DeviceCapabilities) -> Self {
            let mut world = World::new();
            world.insert_resource(ActiveRenderQueues(Some(RenderQueues {
                ops: RenderOps::default(),
                slots: crate::gfx::render_slots::RenderSlots::new(0, true, &[]),
            })));
            world.insert_resource(crate::ecs::ActiveDeviceCaps(caps));
            let (entries, names) = materials();
            world.insert_resource(MaterialTable(entries));
            world.insert_resource(MaterialNames(names));
            world.insert_resource(TextureTable(vec![
                ResourceEntry::default(),
                ResourceEntry::default(),
            ]));
            let (calls, backend) = recording_backend();
            Self {
                world,
                backend,
                calls,
            }
        }

        // A mesh-backed placement holding two draw slots.
        fn prop(&mut self, material: Option<MaterialHandle>) -> Entity {
            let entity = self.world.push(MeshRenderer {
                material,
                cull_distance: 10.0,
                ..Default::default()
            });
            self.world.insert(
                entity,
                RenderHandle {
                    draws: [3u32, 4].into_iter().collect(),
                },
            );
            entity
        }

        // A model-backed placement, whose sub-meshes are its two draw slots.
        fn model_prop(&mut self) -> Entity {
            let entity = self.world.push(crate::components::ModelRenderer {
                cull_distance: 10.0,
                ..Default::default()
            });
            self.world.insert(
                entity,
                RenderHandle {
                    draws: [5u32, 6].into_iter().collect(),
                },
            );
            entity
        }

        fn replay(&mut self) -> Vec<Call> {
            let mut queues = self
                .world
                .resource_mut::<ActiveRenderQueues>()
                .and_then(|slot| slot.0.take())
                .expect("the queue is parked again");
            queues.ops.replay(&mut self.backend);
            if let Some(slot) = self.world.resource_mut::<ActiveRenderQueues>() {
                slot.0 = Some(queues);
            }
            self.calls.lock().unwrap().calls.clone()
        }
    }

    // A world with no renderer takes nothing, and says so rather than
    // pretending: that is the signal the caller rebuilds on.
    #[test]
    fn a_world_without_a_renderer_takes_nothing() {
        let mut world = World::new();
        let entity = world.push(MeshRenderer::default());
        assert!(!is_available(&world));
        assert!(material(&world, AssetId(10)).is_none());
        assert!(!apply_cull_distance(&mut world, entity, 5.0));
    }

    // A backend that bakes per-object material state at build time would keep
    // drawing the old material, so the seam is closed there.
    #[test]
    fn a_backend_that_bakes_its_draws_takes_nothing() {
        let mut f = Fixture::with_caps(DeviceCapabilities {
            rewrites_draws: false,
            ..DeviceCapabilities::ALL
        });
        let entity = f.prop(None);
        assert!(!is_available(&f.world));
        let steel = material(&f.world, AssetId(10)).expect("the table still decodes");
        assert!(!apply_material(&mut f.world, entity, steel));
        assert!(f.replay().is_empty());
        assert!(
            f.world
                .get::<MeshRenderer>(entity)
                .and_then(|r| r.material)
                .is_none(),
            "a refused change leaves the entity as it was"
        );
    }

    // A named material decodes to the entry its args describe; a name the
    // world never loaded resolves to nothing.
    #[test]
    fn a_named_material_resolves_through_the_running_world() {
        let f = Fixture::new();
        let steel = material(&f.world, AssetId(10)).expect("steel is loaded");
        assert_eq!(steel.entry.uniforms.roughness, 0.5);
        assert_eq!(steel.entry.albedo_slot, 1);
        assert_eq!(steel.handle, Some(MaterialHandle(0)));
        assert!(material(&f.world, AssetId(99)).is_none());
    }

    // The swap reaches every draw slot the placement owns, and the entity
    // records the new handle so the next edit compares against it.
    #[test]
    fn a_material_swap_reaches_every_draw_slot() {
        let mut f = Fixture::new();
        let entity = f.prop(None);
        let steel = material(&f.world, AssetId(10)).expect("steel is loaded");
        assert!(apply_material(&mut f.world, entity, steel));
        assert_eq!(
            f.replay(),
            vec![
                Call::SetDrawMaterial {
                    draw_idx: 3,
                    texture_slot: 1,
                    normal_map_slot: crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                },
                Call::SetDrawMaterial {
                    draw_idx: 4,
                    texture_slot: 1,
                    normal_map_slot: crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                },
            ]
        );
        assert_eq!(
            f.world.get::<MeshRenderer>(entity).and_then(|r| r.material),
            Some(MaterialHandle(0))
        );
    }

    #[test]
    fn a_cull_distance_change_reaches_every_draw_slot() {
        let mut f = Fixture::new();
        let entity = f.prop(None);
        assert!(apply_cull_distance(&mut f.world, entity, 40.0));
        assert_eq!(
            f.replay(),
            vec![
                Call::SetDrawCullDistance(3, 40.0),
                Call::SetDrawCullDistance(4, 40.0),
            ]
        );
        assert_eq!(
            f.world.get::<MeshRenderer>(entity).map(|r| r.cull_distance),
            Some(40.0)
        );
    }

    // The cutoff is the placement's own, so a model-backed one carries it to
    // every sub-mesh draw even though its materials are out of reach.
    #[test]
    fn a_model_backed_placement_takes_the_cull_distance() {
        let mut f = Fixture::new();
        let entity = f.model_prop();
        assert!(apply_cull_distance(&mut f.world, entity, 25.0));
        assert_eq!(
            f.replay(),
            vec![
                Call::SetDrawCullDistance(5, 25.0),
                Call::SetDrawCullDistance(6, 25.0),
            ]
        );
        assert_eq!(
            f.world
                .get::<crate::components::ModelRenderer>(entity)
                .map(|r| r.cull_distance),
            Some(25.0)
        );
    }

    // An entity the renderer gave no draw slots has nothing to bind to.
    #[test]
    fn an_entity_with_no_draws_takes_nothing() {
        let mut f = Fixture::new();
        let entity = f.world.push(MeshRenderer::default());
        let steel = material(&f.world, AssetId(10)).expect("steel is loaded");
        assert!(!apply_material(&mut f.world, entity, steel));
        assert!(f.replay().is_empty());
    }

    // What the placement draws with now: its own material, or the default over
    // its legacy texture when it names none. A model-backed placement has no
    // per-placement material at all.
    #[test]
    fn the_drawn_material_follows_the_renderer() {
        let mut f = Fixture::new();
        let bare = f.prop(None);
        assert!(
            drawn_material(&f.world, bare)
                .expect("a mesh placement always draws with something")
                .handle
                .is_none()
        );
        let glassy = f.prop(Some(MaterialHandle(1)));
        assert_eq!(
            drawn_material(&f.world, glassy).expect("glass").handle,
            Some(MaterialHandle(1))
        );
        let model_backed = f.world.push(crate::components::ModelRenderer::default());
        assert!(drawn_material(&f.world, model_backed).is_none());
    }

    // A swap that moves the shader bucket or the transparency flags changes
    // structures built at init, which no per-draw call rebuilds.
    #[test]
    fn a_swap_across_pass_or_pipeline_is_not_expressible() {
        let f = Fixture::new();
        let steel = material(&f.world, AssetId(10)).expect("steel");
        let glass = material(&f.world, AssetId(20)).expect("glass");
        assert!(steel.swappable_with(&steel));
        assert!(
            !steel.swappable_with(&glass),
            "opaque to transparent moves the pass"
        );

        let mut shaded = steel;
        shaded.entry.shader_bucket = 2;
        assert!(!steel.swappable_with(&shaded), "the pipeline moves");
    }

    // A material whose texture reference points past the pool is a corrupt
    // build; nothing is bound rather than a garbage slot.
    #[test]
    fn a_material_referencing_a_missing_texture_declines() {
        let mut f = Fixture::new();
        let broken = Material {
            albedo: Some(TextureHandle(9)),
            ..Default::default()
        };
        f.world.insert_resource(MaterialTable(vec![ResourceEntry {
            payload: None,
            data_bytes: postcard::to_allocvec(&broken).expect("serialises"),
        }]));
        f.world.insert_resource(MaterialNames(vec![10]));
        assert!(material(&f.world, AssetId(10)).is_none());
    }

    // The shader bucket travels with the material, so the swappability check
    // reads what the world compiled rather than a default.
    #[test]
    fn the_shader_reference_travels_with_the_material() {
        let mut f = Fixture::new();
        let shaded = Material {
            shader: Some(ShaderHandle(3)),
            ..Default::default()
        };
        f.world.insert_resource(MaterialTable(vec![ResourceEntry {
            payload: None,
            data_bytes: postcard::to_allocvec(&shaded).expect("serialises"),
        }]));
        f.world.insert_resource(MaterialNames(vec![10]));
        assert_eq!(
            material(&f.world, AssetId(10))
                .expect("decodes")
                .entry
                .shader_bucket,
            3
        );
    }
}
