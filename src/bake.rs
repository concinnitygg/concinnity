//! Bake payloads from typed values, entirely in memory.
//!
//! Everything here is pure computation over the value it is handed -- a
//! generator's geometry, an image-based-lighting convolution, the built-in
//! face's glyph atlas -- so it needs no files, no importers, and no operating
//! system: this module is available on every build of the crate,
//! `--no-default-features` included.
//!
//! The baked bytes go into a [`World`](crate::World) through its data-entry
//! methods ([`add_mesh`](crate::World::add_mesh),
//! [`add_environment_map`](crate::World::add_environment_map)), which hand
//! back the handle a component references the result by:
//!
//! ```no_run
//! use concinnity::components::{DirectionalLight, ProceduralMesh, Prop};
//! use concinnity::{App, World, bake};
//!
//! fn main() {
//!     let mut world = World::new();
//!     world.add_component(DirectionalLight::default());
//!
//!     let mesh = ProceduralMesh {
//!         generator: "box".into(),
//!         ..Default::default()
//!     };
//!     let payload = bake::procedural_mesh(&mesh).expect("the box bakes");
//!     let mesh = world.add_mesh(mesh, payload);
//!     let stone = world.add_material(bake::Material::default());
//!
//!     world.add_component(Prop {
//!         mesh: Some(mesh),
//!         material: Some(stone),
//!         ..Default::default()
//!     });
//!
//!     App::from_world(world).run().expect("the app runs");
//! }
//! ```
//!
//! # Baking vs cooking
//!
//! Reach for this module when everything a world needs can be computed: the
//! built-in generators, the built-in font, image-based lighting from a
//! generator. Reach for the `cook` module when an asset has to be *read* -- a
//! model or texture from disk, a shader to compile, a prefab to expand --
//! which is what its importers are for. A value this module cannot bake (a
//! `source` naming a file, a generator that decodes an image) is refused with
//! an error naming the cook module.

use alloc::string::String;
use alloc::vec::Vec;

/// The bakeable types that [`components`](crate::components) does not carry.
///
/// `EnvironmentMap`, `Font` and `Material` are resources: a compiled world
/// reaches each by handle, so a value that uses one holds the handle its
/// `World::add_*` method returned rather than the value itself. `Camera3D` is
/// the authored form of the component of the same name, which [`camera`]
/// bakes. The `cook` module carries these same types under its own namespace,
/// along with the ones that need an importer.
pub use concinnity_core::components::cook::{Camera3D, EnvironmentMap, Font, Material};

use concinnity_core::components::{self, ProceduralMesh};

/// Bake a [`ProceduralMesh`]'s generator into its geometry payload, for
/// [`World::add_mesh`](crate::World::add_mesh).
pub fn procedural_mesh(mesh: &ProceduralMesh) -> Result<Vec<u8>, String> {
    concinnity_core::bake::payload::procedural_mesh(mesh)
}

/// Convolve an [`EnvironmentMap`]'s generator into its image-based-lighting
/// payload, for [`World::add_environment_map`](crate::World::add_environment_map).
///
/// This is the expensive bake -- hundreds of millions of float operations at
/// default sizes. On the std tier the convolutions are spread over the
/// engine's job pool; without it they run on the calling thread.
#[cfg(feature = "std")]
pub fn environment_map(map: &EnvironmentMap) -> Result<Vec<u8>, String> {
    concinnity_core::bake::payload::environment_map(map, &concinnity_engine::jobs::PoolRows)
}

/// Convolve an [`EnvironmentMap`]'s generator into its image-based-lighting
/// payload, for [`World::add_environment_map`](crate::World::add_environment_map),
/// on the calling thread.
#[cfg(not(feature = "std"))]
pub fn environment_map(map: &EnvironmentMap) -> Result<Vec<u8>, String> {
    concinnity_core::bake::payload::environment_map(
        map,
        &concinnity_core::bake::environment_map::Serial,
    )
}

/// Rasterise a [`Font`] into its glyph-atlas payload. Only the built-in face
/// bakes; a `path` naming a TTF file needs the cook module's importer.
pub fn font(font: &Font) -> Result<Vec<u8>, String> {
    concinnity_core::bake::payload::font(font)
}

/// Bake an authored [`Camera3D`] into the runtime component: the view matrix
/// is computed from its position, yaw, and pitch. A camera with a controller
/// recomputes its view every frame; one without renders from what is baked
/// here.
pub fn camera(args: Camera3D) -> components::Camera3D {
    components::Camera3D::bake(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The doc example's whole path, minus the window: bake, hand over, run.
    #[test]
    fn baked_data_reaches_a_world_that_starts() {
        let mesh = ProceduralMesh {
            generator: "box".into(),
            half_extents: Some([0.7, 0.7, 0.7]),
            ..Default::default()
        };
        let payload = procedural_mesh(&mesh).expect("the box bakes");

        let mut world = crate::World::new();
        let mesh = world.add_mesh(mesh, payload);
        let stone = world.add_material(Material {
            roughness: 0.2,
            ..Default::default()
        });
        world.add_component(components::Prop {
            mesh: Some(mesh),
            material: Some(stone),
            ..Default::default()
        });

        assert_eq!((mesh.index(), stone.index()), (0, 0));
        let mut app = crate::App::from_world(world);
        crate::test_support::assert_starts_headless(&mut app);
    }

    // What cannot be computed is refused with directions, not a wrong payload.
    #[test]
    fn a_file_backed_value_is_refused_toward_the_cook() {
        let err = procedural_mesh(&ProceduralMesh {
            generator: "heightfield".into(),
            ..Default::default()
        })
        .expect_err("an image-decoding generator");
        assert!(err.contains("cook"), "{err}");

        let err = font(&Font {
            path: "face.ttf".into(),
            ..Default::default()
        })
        .expect_err("a file-backed face");
        assert!(err.contains("cook"), "{err}");
    }

    #[test]
    fn a_baked_camera_carries_its_view() {
        let baked = camera(Camera3D {
            position: [0.0, 2.0, 8.0],
            pitch: -0.245,
            ..Default::default()
        });
        assert_ne!(baked.view_matrix, [[0.0; 4]; 4]);
    }

    #[test]
    fn the_sky_bakes_a_readable_environment_payload() {
        let payload = environment_map(&EnvironmentMap {
            generator: "sky".into(),
            prefilter_face_size: 16,
            irradiance_face_size: 8,
            prefilter_samples: 4,
            ..Default::default()
        })
        .expect("the sky bakes");
        let view = concinnity_core::bake::environment_map::deserialise(&payload)
            .expect("the payload reads back");
        assert_eq!(view.prefilter_face, 16);

        let mut world = crate::World::new();
        let handle = world.add_environment_map(payload);
        assert_eq!(handle.index(), 0);
    }
}
