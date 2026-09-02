//! Baking an asset's payload from its typed value.
//!
//! Every bake here is pure computation over the value's fields: a generator's
//! geometry, an IBL convolution, the built-in face's glyph atlas, a material's
//! clamped parameters. An asset whose payload needs a file read, an image
//! decode, or a shader compiler is refused with an error naming the cook
//! module, which is where the importers live.
//!
//! Where a generator argument is optional, the fallback is the one the
//! authored path applies to the same absent argument, so a world declared
//! either way bakes the same bytes.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::bake::environment_map::RowScheduler;
use crate::bake::environment_map::source::{HdrImage, bake_payload, generate_sky_equirect};
use crate::bake::environment_map::stars::generate_stars_equirect;
use crate::bake::mesh::{finish_mesh_payload, vertices_from_data};
use crate::components::{EnvironmentMap, Font, Material, Mesh, ProceduralMesh, validate};
use crate::geometry::{
    Vert, build_box, build_cylinder, build_extrude, build_plane, build_room_geometry, build_skybox,
    build_sphere, build_terrain, water_grid,
};

// Fallbacks for the generator arguments a `ProceduralMesh` leaves unset. Each
// matches what the authored path uses for the same absent argument.
const BOX_HALF_EXTENTS: [f32; 3] = [0.5, 0.5, 0.5];
const CYLINDER_RADIUS: f32 = 0.5;
const CYLINDER_HEIGHT: f32 = 1.0;
const CYLINDER_SEGMENTS: u32 = 16;
const SPHERE_RADIUS: f32 = 1.0;
const SPHERE_RINGS: u32 = 12;
const SPHERE_SEGMENTS: u32 = 16;
const TERRAIN_SUBDIVISIONS: u32 = 64;
const TERRAIN_AMPLITUDE: f32 = 4.0;
const SKYBOX_SIZE: f32 = 490.0;
const EXTRUDE_HEIGHT: f32 = 1.0;
const EXTRUDE_CORNER_RADIUS: f32 = 0.0;
const EXTRUDE_CORNER_SEGMENTS: u32 = 8;
const WATER_SUBDIVISIONS: u32 = 64;

/// Bake a `ProceduralMesh`'s geometry into its blob payload.
pub fn procedural_mesh(mesh: &ProceduralMesh) -> Result<Vec<u8>, String> {
    let (vertices, indices): (Vec<Vert>, Vec<u16>) = match mesh.generator.as_str() {
        "room" => build_room_geometry(mesh.half_width, mesh.half_depth, 0.0, mesh.ceiling_height),
        "box" => build_box(mesh.half_extents.unwrap_or(BOX_HALF_EXTENTS)),
        "cylinder" => build_cylinder(
            mesh.radius.unwrap_or(CYLINDER_RADIUS),
            mesh.height.unwrap_or(CYLINDER_HEIGHT),
            mesh.segments.unwrap_or(CYLINDER_SEGMENTS),
        ),
        "plane" => build_plane(mesh.half_width, mesh.half_depth),
        "sphere" => build_sphere(
            mesh.radius.unwrap_or(SPHERE_RADIUS),
            mesh.rings.unwrap_or(SPHERE_RINGS),
            mesh.segments.unwrap_or(SPHERE_SEGMENTS),
        )?,
        "terrain" => build_terrain(
            mesh.half_width,
            mesh.half_depth,
            mesh.subdivisions.unwrap_or(TERRAIN_SUBDIVISIONS),
            mesh.amplitude.unwrap_or(TERRAIN_AMPLITUDE),
        )?,
        "skybox" => build_skybox(mesh.size.unwrap_or(SKYBOX_SIZE)),
        "extrude" => {
            let profile = mesh
                .profile
                .as_deref()
                .ok_or("the `extrude` generator needs a `profile` of [x, z] points")?;
            build_extrude(
                profile,
                mesh.height.unwrap_or(EXTRUDE_HEIGHT),
                mesh.corner_radius.unwrap_or(EXTRUDE_CORNER_RADIUS),
                mesh.corner_segments.unwrap_or(EXTRUDE_CORNER_SEGMENTS),
            )?
        }
        "water_grid" => water_grid::build_water_grid(
            mesh.half_width,
            mesh.half_depth,
            mesh.subdivisions.unwrap_or(WATER_SUBDIVISIONS),
        )?,
        // The heightfield generator reads a greyscale image, which is an
        // importer's job.
        "heightfield" => {
            return Err(
                "the `heightfield` generator reads a source image; compile it with the \
                 cook module"
                    .to_string(),
            );
        }
        "" => return Err("a ProceduralMesh needs a `generator`".to_string()),
        other => return Err(alloc::format!("unknown mesh generator '{other}'")),
    };
    finish_mesh_payload(vertices, indices, mesh.lod_levels, &mesh.lod_distances)
}

/// Bake a raw `Mesh`'s vertices and indices into its blob payload. Normals
/// and tangents are derived here; a `source` naming a file needs an importer.
pub fn mesh(mesh: &Mesh) -> Result<Vec<u8>, String> {
    if !mesh.source.is_empty() {
        return Err(
            "a Mesh with a `source` reads a model file; compile it with the cook module"
                .to_string(),
        );
    }
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return Err("a Mesh needs `vertices` and `indices`".to_string());
    }
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(alloc::format!(
            "a Mesh's indices come in triangles; {} is not a multiple of 3",
            mesh.indices.len()
        ));
    }
    let vertices = vertices_from_data(&mesh.vertices, &mesh.indices)?;
    finish_mesh_payload(
        vertices,
        mesh.indices.clone(),
        mesh.lod_levels,
        &mesh.lod_distances,
    )
}

/// Bake an `EnvironmentMap`'s IBL cubemaps into its blob payload, spreading
/// each convolution's rows over `rows`.
pub fn environment_map<S: RowScheduler>(map: &EnvironmentMap, rows: &S) -> Result<Vec<u8>, String> {
    if !map.source.is_empty() {
        return Err(
            "an EnvironmentMap with a `source` reads a panorama file; compile it with the \
             cook module"
                .to_string(),
        );
    }
    let generate: fn() -> HdrImage = match map.generator.as_str() {
        "sky" => generate_sky_equirect,
        "stars" => generate_stars_equirect,
        "" => return Err("an EnvironmentMap needs a `source` or a `generator`".to_string()),
        other => return Err(alloc::format!("unknown EnvironmentMap generator '{other}'")),
    };
    crate::bake::environment_map::check_sizes(map)?;
    Ok(bake_payload(
        &generate(),
        map.prefilter_face_size,
        map.irradiance_face_size,
        map.prefilter_samples,
        map.prefilter_clamp,
        rows,
    ))
}

/// Rasterise a `Font` into its glyph-atlas payload.
pub fn font(font: &Font) -> Result<Vec<u8>, String> {
    if !font.path.is_empty() {
        return Err(
            "a Font with a `path` reads a TTF file; compile it with the cook module".to_string(),
        );
    }
    crate::bake::font::compile(
        crate::bake::font::BUILTIN_FONT_BYTES,
        font.size_px,
        "<built-in>",
    )
}

/// Bake a `Material` into the runtime bytes its resource record carries. A
/// material has no blob payload: the clamped parameters are the whole of it.
pub fn material(material: Material) -> Result<Vec<u8>, String> {
    postcard::to_allocvec(&validate::material(material))
        .map_err(|e| alloc::format!("Material serialise: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::environment_map::Serial;
    use crate::gfx::mesh_payload;

    fn mesh(generator: &str) -> ProceduralMesh {
        ProceduralMesh {
            generator: generator.into(),
            ..Default::default()
        }
    }

    // Every generator the builder reaches produces a payload the runtime's own
    // reader accepts; the geometry itself is tested where it is generated.
    #[test]
    fn every_reachable_generator_bakes_a_readable_payload() {
        let mut extrude = mesh("extrude");
        extrude.profile = Some(alloc::vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
        for m in [
            mesh("room"),
            mesh("box"),
            mesh("cylinder"),
            mesh("plane"),
            mesh("sphere"),
            mesh("terrain"),
            mesh("skybox"),
            mesh("water_grid"),
            extrude,
        ] {
            let payload = procedural_mesh(&m).unwrap_or_else(|e| panic!("{}: {e}", m.generator));
            let read = mesh_payload::deserialise(&payload)
                .unwrap_or_else(|e| panic!("{}: {e}", m.generator));
            assert!(!read.0.is_empty(), "{} has vertices", m.generator);
        }
    }

    // The unset optional arguments fall back to the same values the authored
    // path uses, so a box with no half-extents is the same unit cube either
    // way.
    #[test]
    fn an_unset_optional_argument_falls_back_to_the_authored_default() {
        let payload = procedural_mesh(&mesh("box")).expect("a box bakes");
        let (vertices, indices) = build_box(BOX_HALF_EXTENTS);
        let expected = finish_mesh_payload(vertices, indices, 1, &[]).expect("the same box packs");
        assert_eq!(payload, expected);
    }

    #[test]
    fn a_generator_that_needs_an_importer_says_so() {
        for (m, needle) in [
            (mesh("heightfield"), "source image"),
            (mesh(""), "needs a `generator`"),
            (mesh("nonesuch"), "unknown mesh generator"),
            (mesh("extrude"), "needs a `profile`"),
        ] {
            let err = procedural_mesh(&m).expect_err("not bakeable");
            assert!(err.contains(needle), "{err}");
        }
    }

    fn triangle() -> Mesh {
        let vd = |pos: [f32; 3], uv: [f32; 2]| crate::components::VertexData {
            pos,
            color: [1.0; 3],
            uv,
        };
        Mesh {
            vertices: alloc::vec![
                vd([0.0, 0.0, 0.0], [0.0, 0.0]),
                vd([1.0, 0.0, 0.0], [1.0, 0.0]),
                vd([0.0, 1.0, 0.0], [0.0, 1.0]),
            ],
            indices: alloc::vec![0, 1, 2],
            ..Default::default()
        }
    }

    #[test]
    fn raw_geometry_bakes_with_derived_normals() {
        let payload = super::mesh(&triangle()).expect("a triangle bakes");
        let (verts, indices) = mesh_payload::deserialise(&payload).expect("the payload reads back");
        assert_eq!(indices, alloc::vec![0, 1, 2]);
        assert!(verts.iter().all(|v| (v.normal[2] - 1.0).abs() < 1e-5));
    }

    #[test]
    fn raw_geometry_reports_what_it_cannot_bake() {
        let sourced = Mesh {
            source: "chair.glb".into(),
            ..Default::default()
        };
        let err = super::mesh(&sourced).expect_err("a file source");
        assert!(err.contains("cook module"), "{err}");

        let err = super::mesh(&Mesh::default()).expect_err("nothing to bake");
        assert!(err.contains("needs `vertices` and `indices`"), "{err}");

        let mut ragged = triangle();
        ragged.indices.push(0);
        let err = super::mesh(&ragged).expect_err("a partial triangle");
        assert!(err.contains("multiple of 3"), "{err}");

        let mut past = triangle();
        past.indices[2] = 7;
        let err = super::mesh(&past).expect_err("an index past the list");
        assert!(err.contains("indexes past"), "{err}");
    }

    #[test]
    fn the_sky_generator_bakes_an_environment_payload() {
        let map = EnvironmentMap {
            generator: "sky".into(),
            prefilter_face_size: 16,
            irradiance_face_size: 8,
            prefilter_samples: 4,
            ..Default::default()
        };
        let payload = environment_map(&map, &Serial).expect("the sky bakes");
        let view =
            crate::bake::environment_map::deserialise(&payload).expect("the payload reads back");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
    }

    #[test]
    fn an_environment_map_reports_what_it_cannot_bake() {
        let with_source = EnvironmentMap {
            source: "studio.hdr".into(),
            ..Default::default()
        };
        let err = environment_map(&with_source, &Serial).expect_err("a file source");
        assert!(err.contains("cook module"), "{err}");

        let blank = EnvironmentMap::default();
        let err = environment_map(&blank, &Serial).expect_err("nothing to bake");
        assert!(err.contains("needs a `source` or a `generator`"), "{err}");

        let unknown = EnvironmentMap {
            generator: "swamp".into(),
            ..Default::default()
        };
        let err = environment_map(&unknown, &Serial).expect_err("unknown generator");
        assert!(err.contains("unknown EnvironmentMap generator"), "{err}");

        let oversized = EnvironmentMap {
            generator: "sky".into(),
            prefilter_face_size: 3,
            ..Default::default()
        };
        let err = environment_map(&oversized, &Serial).expect_err("out of range");
        assert!(err.contains("prefilter_face_size"), "{err}");
    }

    #[test]
    fn the_builtin_face_rasterises_and_a_file_one_does_not() {
        let payload = font(&Font {
            size_px: 12,
            ..Default::default()
        })
        .expect("the built-in face bakes");
        let (_, _, _, size_px, _, metrics) =
            crate::bake::font::deserialise(&payload).expect("the atlas reads back");
        assert_eq!(size_px, 12);
        assert!(!metrics.is_empty());

        let err = font(&Font {
            path: "assets/face.ttf".into(),
            ..Default::default()
        })
        .expect_err("a file face");
        assert!(err.contains("cook module"), "{err}");
    }

    // A material's bytes are its clamped parameters, the same clamps the
    // authored path applies on its way into the blob.
    #[test]
    fn a_material_bakes_its_clamped_parameters() {
        let bytes = material(Material {
            roughness: 4.0,
            see_through: true,
            ..Default::default()
        })
        .expect("a material bakes");
        let read: Material = postcard::from_bytes(&bytes).expect("the bytes read back");
        assert_eq!(read.roughness, 1.0);
        assert!(read.transparent, "see-through implies transparent");
    }
}
