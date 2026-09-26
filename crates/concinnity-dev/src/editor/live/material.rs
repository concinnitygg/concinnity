//! A Material's Shader parameters. Every other Material field is baked into the
//! draws that use it at load, but `params` live in the renderer's parameter
//! table, one row per material, so an edit that changes only them rewrites that
//! row through the engine's `gfx::material_preview` seam.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::gfx::render_types::MATERIAL_PARAM_COUNT;
use concinnity_engine::gfx::{draw_preview, material_preview};
use concinnity_host::thread::asset_id;
use serde_json::{Map, Value};

use super::Apply;

const KEY: &str = "params";

/// One planned rewrite of a material's parameter row.
pub(crate) struct ParamsChange {
    name: AssetId,
    params: [f32; MATERIAL_PARAM_COUNT],
}

/// Plan the parameter rewrite for the Material `name`, or `None` when the edit
/// touches anything but its `params`, the running world has no renderer, or it
/// loaded no material under that name.
pub(super) fn plan(
    world: &World,
    ct: RegisteredType,
    name: &str,
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if !claims(ct, keys) || !material_preview::is_available(world) {
        return None;
    }
    let name = asset_id::lookup(name)?;
    draw_preview::material(world, name)?;
    let params = params_of(args.get(KEY))?;
    Some(Apply::MaterialParams(ParamsChange { name, params }))
}

// Whether an edit moving `keys` of a `ct` is this path's: a Material's
// `params`, and nothing else.
fn claims(ct: RegisteredType, keys: &[String]) -> bool {
    ct == RegisteredType::Material && !keys.is_empty() && keys.iter().all(|k| k == KEY)
}

/// Perform a planned parameter rewrite.
pub(super) fn commit(world: &mut World, change: ParamsChange) {
    material_preview::apply_params(world, change.name, change.params);
}

// The eight parameters `value` spells, all zero when it is absent (the
// default the key uncovers); `None` for anything the cook would reject.
fn params_of(value: Option<&Value>) -> Option<[f32; MATERIAL_PARAM_COUNT]> {
    let Some(value) = value else {
        return Some([0.0; MATERIAL_PARAM_COUNT]);
    };
    let values = value.as_array()?;
    if values.len() != MATERIAL_PARAM_COUNT {
        return None;
    }
    let mut out = [0.0; MATERIAL_PARAM_COUNT];
    for (slot, v) in out.iter_mut().zip(values) {
        *slot = v.as_f64()? as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|k| k.to_string()).collect()
    }

    #[test]
    fn params_parse_from_eight_numbers() {
        let v = json!([1, 2.5, 0, 0, 0, 0, 0, -3]);
        assert_eq!(
            params_of(Some(&v)),
            Some([1.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0])
        );
        assert_eq!(params_of(None), Some([0.0; MATERIAL_PARAM_COUNT]));
    }

    #[test]
    fn a_malformed_params_array_is_left_to_the_build() {
        assert_eq!(params_of(Some(&json!([1, 2, 3]))), None);
        assert_eq!(params_of(Some(&json!([1, 2, 3, 4, 5, 6, 7, "x"]))), None);
        assert_eq!(params_of(Some(&json!(4))), None);
    }

    // Only an edit that moves nothing but a Material's `params` is claimed; any
    // other Material field, and another type's `params`, rebuilds.
    #[test]
    fn only_a_material_params_edit_is_claimed() {
        let ty = |name| RegisteredType::parse(name).expect("a registered type");
        assert!(claims(ty("Material"), &keys(&["params"])));
        assert!(!claims(ty("Material"), &keys(&["roughness"])));
        assert!(!claims(ty("Material"), &keys(&["params", "roughness"])));
        assert!(!claims(ty("SdfVolume"), &keys(&["params"])));
    }

    // A world with no renderer to rewrite leaves the edit to the build.
    #[test]
    fn a_world_without_a_renderer_declines() {
        let world = World::new();
        let params = json!({ "params": [0, 0, 0, 0, 0, 0, 0, 1] });
        let ct = RegisteredType::parse("Material").expect("a registered type");
        assert!(plan(&world, ct, "steel", &args(params), &keys(&["params"])).is_none());
    }
}
