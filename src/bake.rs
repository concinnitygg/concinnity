//! Bake payloads from typed values, entirely in memory.
//!
//! Everything here is pure computation over the value it is handed -- a
//! generator's geometry, an image-based-lighting convolution, the built-in
//! face's glyph atlas -- so it needs no files, no importers, and no operating
//! system: this module is available on every build of the crate,
//! `--no-default-features` included.
//!
//! The baked payloads go into a [`World`](crate::World) through its data-entry
//! methods ([`add_mesh`](crate::World::add_mesh),
//! [`add_environment_map`](crate::World::add_environment_map),
//! [`add_font`](crate::World::add_font)), which hand back the handle a
//! component references the result by. Each bake returns its own payload type,
//! so a payload only fits the method for its kind:
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
//!     let payload = bake::procedural_mesh(mesh).expect("the box bakes");
//!     let mesh = world.add_mesh(payload).expect("the world has ids to mint");
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

use alloc::vec::Vec;

/// The bakeable types that [`components`](crate::components) does not carry.
///
/// `EnvironmentMap`, `Font`, `Material` and `Mesh` are resources: a compiled
/// world reaches each by handle, so a value that uses one holds the handle its
/// `World::add_*` method returned rather than the value itself. `Camera3D` and
/// `CameraTrack` are the authored forms of the components of the same names,
/// which [`camera`] and [`camera_track`] bake. The `cook` module carries these
/// same types under its own namespace, along with the ones that need an importer.
pub use concinnity_core::components::cook::{
    Camera3D, CameraTrack, EnvironmentMap, Font, Material, Mesh, VertexData,
};

/// Baked geometry, for [`World::add_mesh`](crate::World::add_mesh).
pub use concinnity_core::bake::payload::MeshPayload;

use concinnity_core::components::{self, ProceduralMesh};

// A payload only a bake in this module constructs, so what a data-entry method
// takes is known to be the kind it installs.
macro_rules! payload {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq)]
        pub struct $name(Vec<u8>);

        impl $name {
            /// The baked bytes.
            pub fn as_bytes(&self) -> &[u8] {
                &self.0
            }

            pub(crate) fn into_bytes(self) -> Vec<u8> {
                self.0
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("len", &self.0.len())
                    .finish()
            }
        }
    };
}

payload! {
    /// Baked image-based lighting, for
    /// [`World::add_environment_map`](crate::World::add_environment_map).
    EnvironmentMapPayload
}

payload! {
    /// A baked glyph atlas, for [`World::add_font`](crate::World::add_font).
    FontPayload
}

/// Bake a [`ProceduralMesh`]'s generator into its geometry payload, for
/// [`World::add_mesh`](crate::World::add_mesh).
pub fn procedural_mesh(mesh: ProceduralMesh) -> Result<MeshPayload, crate::Error> {
    concinnity_core::bake::payload::procedural_mesh(mesh).map_err(crate::Error::Bake)
}

/// Bake a raw [`Mesh`]'s vertices and indices into its geometry payload, for
/// [`World::add_mesh`](crate::World::add_mesh). Normals and tangents are
/// derived from the triangles; a `source` naming a model file needs the cook
/// module's importer.
pub fn mesh(mesh: &Mesh) -> Result<MeshPayload, crate::Error> {
    concinnity_core::bake::payload::mesh(mesh).map_err(crate::Error::Bake)
}

/// Convolve an [`EnvironmentMap`]'s generator into its image-based-lighting
/// payload, for [`World::add_environment_map`](crate::World::add_environment_map).
///
/// This is the expensive bake -- hundreds of millions of float operations at
/// default sizes. On the std tier the convolutions are spread over the
/// engine's job pool; without it they run on the calling thread.
#[cfg(feature = "std")]
pub fn environment_map(map: &EnvironmentMap) -> Result<EnvironmentMapPayload, crate::Error> {
    use concinnity_host::thread::jobs;
    concinnity_core::bake::payload::environment_map(map, &jobs::PoolRows(jobs::pool()))
        .map(EnvironmentMapPayload)
        .map_err(crate::Error::Bake)
}

/// Convolve an [`EnvironmentMap`]'s generator into its image-based-lighting
/// payload, for [`World::add_environment_map`](crate::World::add_environment_map),
/// on the calling thread.
#[cfg(not(feature = "std"))]
pub fn environment_map(map: &EnvironmentMap) -> Result<EnvironmentMapPayload, crate::Error> {
    concinnity_core::bake::payload::environment_map(
        map,
        &concinnity_core::bake::environment_map::Serial,
    )
    .map(EnvironmentMapPayload)
    .map_err(crate::Error::Bake)
}

/// Rasterize a [`Font`] into its glyph-atlas payload, for
/// [`World::add_font`](crate::World::add_font). Only the built-in face bakes;
/// a `path` naming a TTF file needs the cook module's importer.
pub fn font(font: &Font) -> Result<FontPayload, crate::Error> {
    concinnity_core::bake::payload::font(font)
        .map(FontPayload)
        .map_err(crate::Error::Bake)
}

/// Bake an authored [`Camera3D`] into the runtime component: the view matrix
/// is computed from its position, yaw, and pitch. A camera with a controller
/// recomputes its view every frame; one without renders from what is baked
/// here.
pub fn camera(args: Camera3D) -> components::Camera3D {
    components::Camera3D::bake(args)
}

/// Bake an authored [`CameraTrack`] into the runtime component: each travel
/// leg's duration and cumulative offset are resolved, and the segment names it
/// reports under are gathered. The turn legs are timed when the world starts,
/// from the heading the camera was authored at.
pub fn camera_track(args: CameraTrack) -> components::CameraTrack {
    components::CameraTrack::bake(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    // The doc example's whole path, minus the window: bake, hand over, run.
    #[test]
    fn baked_data_reaches_a_world_that_starts() {
        let mesh = ProceduralMesh {
            generator: "box".into(),
            half_extents: Some([0.7, 0.7, 0.7]),
            ..Default::default()
        };
        let payload = procedural_mesh(mesh).expect("the box bakes");

        let mut world = crate::World::new();
        let mesh = world.add_mesh(payload).expect("the first mint");
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
        crate::test_support::assert_starts_headless(crate::App::from_world(world));
    }

    // The generator a mesh was baked from stays in the world beside its payload.
    #[test]
    fn a_baked_sphere_leaves_its_procedural_mesh_in_the_world() {
        let sphere = ProceduralMesh {
            generator: "sphere".into(),
            radius: Some(0.5),
            ..Default::default()
        };
        let payload = procedural_mesh(sphere.clone()).expect("the sphere bakes");

        let mut world = crate::World::new();
        world.add_mesh(payload).expect("the first mint");

        let meshes: alloc::vec::Vec<_> = world.inner().query::<ProceduralMesh>().collect();
        assert_eq!(meshes.len(), 1);
        assert_eq!(
            meshes[0],
            &ProceduralMesh {
                asset_id: meshes[0].asset_id,
                ..sphere
            }
        );
    }

    // Raw geometry takes the same path as a generator's: bake, hand over, run.
    #[test]
    fn raw_geometry_reaches_a_world_that_starts() {
        let vertex = |pos: [f32; 3]| VertexData {
            pos,
            color: [1.0; 3],
            uv: [0.0; 2],
        };
        let triangle = Mesh {
            vertices: alloc::vec![
                vertex([0.0, 0.0, 0.0]),
                vertex([1.0, 0.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: alloc::vec![0, 1, 2],
            ..Default::default()
        };
        let payload = mesh(&triangle).expect("the triangle bakes");

        let mut world = crate::World::new();
        let handle = world.add_mesh(payload).expect("the first mint");
        world.add_component(components::Prop {
            mesh: Some(handle),
            ..Default::default()
        });

        assert_eq!(handle.index(), 0);
        crate::test_support::assert_starts_headless(crate::App::from_world(world));
    }

    // The built-in face takes the same path: bake, hand over, reference, run.
    #[test]
    fn a_baked_font_reaches_a_text_label_that_starts() {
        let payload = font(&Font::default()).expect("the built-in face bakes");

        let mut world = crate::World::new();
        let handle = world.add_font(payload);
        world.add_component(components::TextLabel {
            content: "Hello, world!".into(),
            font: Some(handle),
            ..Default::default()
        });

        let table = world
            .inner()
            .resource::<concinnity_core::resource::FontTable>()
            .expect("the font table");
        assert!(
            table
                .0
                .get(handle.index())
                .and_then(|entry| entry.baked_bytes())
                .is_some_and(|bytes| !bytes.is_empty()),
            "the atlas is installed at its handle"
        );
        crate::test_support::assert_starts_headless(crate::App::from_world(world));
    }

    // What cannot be computed is refused with directions, not a wrong payload.
    #[test]
    fn a_file_backed_value_is_refused_toward_the_cook() {
        let err = procedural_mesh(ProceduralMesh {
            generator: "heightfield".into(),
            ..Default::default()
        })
        .expect_err("an image-decoding generator");
        assert!(matches!(err, crate::Error::Bake(_)), "{err:?}");
        assert!(err.to_string().contains("cook"), "{err}");

        let err = font(&Font {
            path: "face.ttf".into(),
            ..Default::default()
        })
        .expect_err("a file-backed face");
        assert!(matches!(err, crate::Error::Bake(_)), "{err:?}");
        assert!(err.to_string().contains("cook"), "{err}");

        let err = mesh(&Mesh {
            source: "chair.glb".into(),
            ..Default::default()
        })
        .expect_err("a file-backed mesh");
        assert!(matches!(err, crate::Error::Bake(_)), "{err:?}");
        assert!(err.to_string().contains("cook"), "{err}");
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
        let view = concinnity_core::bake::environment_map::deserialize(payload.as_bytes())
            .expect("the payload reads back");
        assert_eq!(view.prefilter_face, 16);

        let mut world = crate::World::new();
        let handle = world.add_environment_map(payload);
        assert_eq!(handle.index(), 0);
    }
}
