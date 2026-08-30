// The sky mesh that displays an environment map: an inside-out skybox cube,
// the material it draws with, and the prop that places it. Injected only when
// the world has an environment map and no skybox geometry of its own.

use alloc::string::ToString;

use crate::components::{Camera3D, Material, ProceduralMesh, Prop};
use crate::ecs::PipelineContext;
use crate::resource::{EnvironmentMapTable, append_material, append_mesh};
use crate::result::CnResult;
use crate::{bake, geometry};

use super::Minter;

// Sky depth is pinned to the far plane, so the mesh only has to enclose the
// camera while staying inside it.
const SKY_SIZE_MAX: f32 = 400.0;
const SKY_FAR_FRACTION: f32 = 0.9;
const CAMERA_FAR_DEFAULT: f32 = 200.0;

pub(super) fn inject(ctx: &mut PipelineContext, minter: &mut Minter) -> Result<(), CnResult> {
    let lit = ctx
        .resource::<EnvironmentMapTable>()
        .is_some_and(|t| !t.is_empty());
    if !lit
        || ctx
            .query::<ProceduralMesh>()
            .any(|m| m.generator == "skybox")
    {
        return Ok(());
    }

    let far = ctx
        .query::<Camera3D>()
        .next()
        .map_or(CAMERA_FAR_DEFAULT, |c| c.far);
    let size = (far * SKY_FAR_FRACTION).min(SKY_SIZE_MAX);

    let mesh_id = minter.id();
    let (vertices, indices) = geometry::build_skybox(size);
    let payload = bake::mesh::finish_mesh_payload(vertices, indices, 1, &[])
        .map_err(|_| CnResult::InvalidArgument)?;
    let mesh = append_mesh(ctx, mesh_id, payload);
    ctx.push(ProceduralMesh {
        asset_id: mesh_id,
        generator: "skybox".to_string(),
        size: Some(size),
        ..Default::default()
    });

    let material = append_material(
        ctx,
        Material {
            roughness: 1.0,
            metallic: 0.0,
            tint: [1.0, 1.0, 1.0],
            ..Default::default()
        },
    );

    ctx.push(Prop {
        asset_id: minter.id(),
        mesh: Some(mesh),
        material: Some(material),
        position: [0.0, 0.0, 0.0],
        ..Default::default()
    });
    Ok(())
}
