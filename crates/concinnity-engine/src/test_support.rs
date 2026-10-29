// Shared fixtures for the crate's unit tests. The timing harness its
// in-module benchmarks report through is `concinnity_core::bench`.
//
// The draw records are wide enough that constructing one inline dominates a
// test; they are built here once so a test states only the fields it is about.
//
// The name interner here stands in for the production one, which keeps a
// per-thread table and so lives in the std-linked crate above this one. Asset
// tests that deserialize a named reference need names to resolve to dense
// declaration-ordered ids; this reproduces exactly that, and nothing else the
// production interner does.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use core::cell::RefCell;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawObject, FarFieldTint, MaterialUniforms, SkinnedDrawObject};
use crate::transform::IDENTITY;
pub(crate) use concinnity_core::bench::{Pace, Rng, bench};

// `#[global_allocator]` is a per-program item, so the test binary is the one
// build of this crate that installs it, for the allocation guard and benchmarks.
concinnity_core::install_global_allocator!();

// Per-thread, like the production interner: the harness runs tests in parallel
// and each one resets before interning, so a shared table would let them
// clobber each other.
std::thread_local! {
    static NAMES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn resolve(name: &str) -> u32 {
    NAMES.with(|names| {
        let mut names = names.borrow_mut();
        if let Some(id) = names.iter().position(|n| n == name) {
            u32::saturating_from(id)
        } else {
            names.push(name.to_string());
            u32::saturating_from(names.len() - 1)
        }
    })
}

// Install this thread's interner behind the resolver seam, keeping what it
// already holds.
pub(crate) fn install_interner() {
    crate::ecs::resolver::set_name_resolver(resolve);
}

// The id `name` interns to on this thread, with the interner installed.
pub(crate) fn intern(name: &str) -> crate::ecs::asset_id::AssetId {
    install_interner();
    crate::ecs::asset_id::AssetId(resolve(name))
}

// Empty this thread's interner and install it behind the resolver seam. Call first in any test that interns or deserializes a named
// reference.
pub(crate) fn reset_interner() {
    crate::ecs::resolver::set_name_resolver(resolve);
    NAMES.with(|names| names.borrow_mut().clear());
}

// A populated static draw record with distinct values in every slot, so a test
// that reads one back can tell which field it got.
pub(crate) fn draw_object() -> DrawObject {
    DrawObject {
        vertex_offset: 0,
        vertex_count: 8,
        index_offset: 12,
        index_count: 36,
        planet_tile: None,
        base_vertex: 4,
        geometry_generation: 0,
        shader_bucket: 0,
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [5.0, 6.0, 7.0, 1.0],
        ],
        texture_slot: 9,
        normal_map_slot: 2,
        material: MaterialUniforms {
            roughness: 0.3,
            metallic: 0.7,
            alpha_cutoff: 0.25,
            opacity: 1.0,
            tint: [0.1, 0.2, 0.3],
            _pad0: 0.0,
            emissive: [0.4, 0.5, 0.6],
            _pad1: 0.0,
            emissive_map_index: 0,
            orm_map_index: 0,
            transparent: 0,
            see_through: 0,
            params_index: 5,
            far_field: FarFieldTint::NONE,
        },
        visible: true,
        resident: true,
        bb_min: [-1.0, -2.0, -3.0],
        bb_max: [1.0, 2.0, 3.0],
        local_bounds: None,
        cull_distance: 42.0,
        lod_alternates: Vec::new(),
    }
}

// A vertex at the origin with every attribute zeroed.
pub(crate) fn vertex() -> Vertex {
    Vertex {
        pos: [0.0; 3],
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
    }
}

// A skinned draw record at the origin with a unit bind-pose box.
pub(crate) fn skinned_draw_object() -> SkinnedDrawObject {
    SkinnedDrawObject {
        vertex_base: 0,
        vertex_count: 10,
        index_offset: 0,
        index_count: 30,
        model: IDENTITY,
        texture_slot: 5,
        normal_map_slot: 2,
        material: MaterialUniforms::DEFAULT,
        visible: true,
        joint_count: 4,
        local_bb_min: [-1.0, -1.0, -1.0],
        local_bb_max: [1.0, 1.0, 1.0],
        lod_alternates: Vec::new(),
    }
}

// A name resolves to its byte length: a deterministic stand-in for the build's
// declaration-ordered resource tables. Names prefixed `unknown_` resolve to
// nothing, standing in for a reference no resource of that kind declares.
pub(crate) fn len_handle_resolver(_: crate::ecs::handle::HandleKind, name: &str) -> Option<u32> {
    if name.starts_with("unknown_") {
        None
    } else {
        Some(u32::saturating_from(name.len()))
    }
}

// A name resolves to its byte length, standing in for the build-time interner
// the handle resolvers fall back to.
pub(crate) fn len_name_resolver(name: &str) -> u32 {
    u32::saturating_from(name.len())
}

// Installs every seam with the stand-ins above.
pub(crate) fn install_resolvers() {
    crate::ecs::resolver::set_name_resolver(len_name_resolver);
    for kind in crate::ecs::handle::HandleKind::ALL {
        crate::ecs::resolver::set_handle_resolver(*kind, len_handle_resolver);
    }
}

/// Deserialize `json` with the test resolvers installed.
///
/// Every component test that parses a name-string field goes through here
/// rather than opening with [`install_resolvers`] and then deserializing, so
/// the setup cannot be forgotten: without it a name field deserializes to an
/// error that reads like a data bug.
pub(crate) fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> T {
    install_resolvers();
    serde_json::from_str(json).expect("the test JSON deserializes")
}

// Start `world` on the headless table with `system` registered in `phase`, so
// a test runs the system beside the simulation it shares a world with.
pub(crate) fn start_with<S: crate::ecs::System>(
    world: &mut crate::ecs::World,
    phase: crate::ecs::Phase,
    name: &'static str,
    system: S,
) {
    world.add_system(phase, name, system);
    world
        .start(crate::ecs::HEADLESS_SYSTEMS)
        .expect("the test world starts");
}

// One resident payload section every locator reads from by its byte range,
// filled through `payload` the way cook appends a payload to a blob.
#[derive(Default)]
pub(crate) struct OneSection(pub(crate) Vec<u8>);

impl OneSection {
    // Append `bytes` and return the locator that reads them back.
    pub(crate) fn payload(&mut self, bytes: &[u8]) -> crate::ecs::PayloadLocator {
        let offset = self.0.len() as u64;
        self.0.extend_from_slice(bytes);
        crate::ecs::PayloadLocator {
            blob_index: 0,
            offset,
            len: bytes.len() as u64,
        }
    }
}

impl crate::ecs::PayloadStore for OneSection {
    fn read(
        &mut self,
        locator: &crate::ecs::PayloadLocator,
    ) -> Result<&[u8], crate::error::PayloadError> {
        let start = usize::saturating_from(locator.offset);
        let end = start + usize::saturating_from(locator.len);
        self.0
            .get(start..end)
            .filter(|_| locator.blob_index == 0)
            .ok_or(crate::error::PayloadError::OutOfBounds {
                index: locator.blob_index,
                offset: locator.offset,
                len: locator.len,
                section_len: self.0.len(),
            })
    }

    fn release(&mut self, _blob_index: u32) {}

    fn disk_backed(&self) -> bool {
        false
    }
}
