//! The world's methods for adding a resource built in code: a baked mesh,
//! material, environment map or glyph atlas, each appended to its table.

use crate::bake::payload::{EnvironmentMapPayload, FontPayload, MeshPayload};
use crate::ecs::{EnvironmentMapHandle, FontHandle, MaterialHandle, MeshHandle, World};
use crate::error::WorldError;

impl World {
    /// Add a baked mesh `payload` and return the handle a
    /// [`Prop`](crate::components::Prop) references it by.
    ///
    /// The world names the mesh itself (from the minted range) and holds the
    /// payload directly, so no compiled blob is involved; handles count up in
    /// call order, after any the build assigned. A payload baked from a
    /// [`ProceduralMesh`](crate::components::ProceduralMesh) also adds that
    /// mesh, carrying the minted id, as the record of what was generated.
    ///
    /// # Errors
    ///
    /// Returns an error once the world has minted [`AssetId::MINTED_CAPACITY`](crate::ecs::asset_id::AssetId::MINTED_CAPACITY)
    /// names.
    pub fn add_mesh(&mut self, payload: MeshPayload) -> Result<MeshHandle, WorldError> {
        let (bytes, procedural) = payload.into_parts();
        let mut ctx = self.context();
        let id = ctx.mint_id()?;
        let handle = super::append_mesh(&mut ctx, id, bytes);
        if let Some(mesh) = procedural {
            ctx.push_identified(id, mesh);
        }
        Ok(handle)
    }

    /// Add a material and return the handle a
    /// [`Prop`](crate::components::Prop) references it by. The value's fields
    /// are clamped into their valid ranges on the way in, the same way the
    /// cook clamps an authored material.
    pub fn add_material(&mut self, material: crate::components::Material) -> MaterialHandle {
        super::append_material(&mut self.context(), material)
    }

    /// Add a baked image-based-lighting payload, from
    /// [`bake::payload::environment_map`](crate::bake::payload::environment_map),
    /// and return its handle. The renderer lights with the map at handle 0
    /// and, unless the map set `background` to `false`, draws it behind all
    /// geometry.
    pub fn add_environment_map(&mut self, payload: EnvironmentMapPayload) -> EnvironmentMapHandle {
        let (bytes, record) = payload.into_parts();
        super::append_environment_map(&mut self.context(), bytes, record)
    }

    /// Add a baked glyph atlas, from
    /// [`bake::payload::font`](crate::bake::payload::font), and return the
    /// handle a text component references it by.
    pub fn add_font(&mut self, payload: FontPayload) -> FontHandle {
        super::append_font(&mut self.context(), payload.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::ecs::World;

    #[test]
    fn a_procedural_payload_adds_its_mesh_under_the_minted_id() {
        let mesh = crate::components::ProceduralMesh {
            generator: "box".into(),
            ..Default::default()
        };
        let payload = crate::bake::payload::procedural_mesh(mesh.clone()).expect("a box bakes");
        let bytes = payload.as_bytes().to_vec();

        let mut world = World::new();
        let handle = world.add_mesh(payload).expect("the first mint");

        let payloads = world
            .resource::<crate::resource::RuntimeMeshPayloads>()
            .expect("the runtime payload store");
        let (id, stored) = payloads.iter().nth(handle.index()).expect("the payload");
        assert_eq!(stored, &bytes[..]);
        let added: Vec<_> = world.query::<crate::components::ProceduralMesh>().collect();
        assert_eq!(added, [&mesh]);
        assert_eq!(
            world.get_by_id::<crate::components::ProceduralMesh>(id),
            Some(&mesh)
        );
    }

    #[test]
    fn a_raw_mesh_payload_adds_no_procedural_mesh() {
        let vertex = |pos: [f32; 3]| crate::components::VertexData {
            pos,
            color: [1.0; 3],
            uv: [0.0; 2],
        };
        let triangle = crate::components::Mesh {
            vertices: alloc::vec![
                vertex([0.0, 0.0, 0.0]),
                vertex([1.0, 0.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: alloc::vec![0, 1, 2],
            ..Default::default()
        };
        let payload = crate::bake::payload::mesh(&triangle).expect("a triangle bakes");

        let mut world = World::new();
        let handle = world.add_mesh(payload).expect("the first mint");

        assert_eq!(handle.index(), 0);
        assert_eq!(
            world.query::<crate::components::ProceduralMesh>().count(),
            0
        );
    }
}
