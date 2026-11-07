// A Material is a pure data resource: its baked bytes ride inline in its
// record rather than in a payload section.

use concinnity_engine::components::Material;
use serde::Deserialize;

use crate::asset::{BuildAsset, BuildCtx};

impl BuildAsset for Material {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        compile_material_data(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// Bake a Material into its resource `data_bytes`: resolve its texture references
// to handles (the texture-handle resolver is installed before this runs), apply
// the runtime validation clamps -- which no longer run via a generated `from_args`
// now that Material is a resource -- and serialize. The runtime deserializes these
// bytes straight back into a `Material` to build its material map.
fn compile_material_data(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mat: Material =
        Deserialize::deserialize(args).map_err(|e| format!("Material args: {e}"))?;
    let mat = crate::authoring::validate::material(mat);
    postcard::to_allocvec(&mat).map_err(|e| format!("Material serialize: {e}"))
}
