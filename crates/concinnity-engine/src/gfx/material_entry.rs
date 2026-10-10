//! One decoded Material as the draw list consumes it: the GPU uniforms plus the
//! shared-pool slots each texture reference resolves to. The translation from a
//! compiled `Material` to this is the renderer's own reading of the asset, so it
//! lives here rather than inside the init pass that first needed it: init bakes
//! the whole table at load, and the editor's live draw seam bakes one material
//! again when an edit reassigns it.

use concinnity_core::components::Material;
use concinnity_core::ecs::{MaterialHandle, TextureHandle};
use concinnity_core::gfx::render_types::{
    FarFieldTint, MaterialUniforms, NO_ALBEDO_SLOT, NO_NORMAL_MAP_SLOT,
};
use concinnity_core::render::material_params;

// One decoded material as build_draw_list consumes it: resolved texture pool
// slots, the GPU uniforms, and the shader bucket its draws render under.
#[derive(Clone, Copy)]
pub(crate) struct MaterialEntry {
    pub(crate) albedo_slot: usize,
    pub(crate) normal_map_slot: usize,
    pub(crate) uniforms: MaterialUniforms,
    // Dense ShaderHandle value of the material's `shader` reference; 0 (the
    // world default) when the material names none.
    pub(crate) shader_bucket: u32,
}

impl MaterialEntry {
    /// The entry a draw with no material of its own binds: the default
    /// uniforms over the reserved albedo and normal-map fallbacks.
    pub(crate) const UNTEXTURED: MaterialEntry = MaterialEntry {
        albedo_slot: NO_ALBEDO_SLOT,
        normal_map_slot: NO_NORMAL_MAP_SLOT,
        uniforms: MaterialUniforms::DEFAULT,
        shader_bucket: 0,
    };
}

/// The entry the `Material` at `handle` bakes to against a texture pool of
/// `texture_count` entries. `Err` names the reference that points past the
/// pool, which cook validated and so marks a corrupt build.
pub(crate) fn of(
    handle: MaterialHandle,
    mat: &Material,
    texture_count: usize,
) -> Result<MaterialEntry, &'static str> {
    // Unset fallbacks differ per field. Albedo and the normal map select a
    // reserved fallback entry through a sentinel no real handle can collide
    // with. Slot 0 stays the sentinel the shader gates on for the emissive and
    // ORM maps, which keeps their scalar value.
    let slot_of = |field: &'static str, handle: Option<TextureHandle>, unset: usize| {
        let Some(handle) = handle else {
            return Ok(unset);
        };
        let slot = handle.index();
        if slot >= texture_count {
            return Err(field);
        }
        Ok(slot)
    };
    let albedo_slot = slot_of("albedo", mat.albedo, NO_ALBEDO_SLOT)?;
    let normal_map_slot = slot_of("normal_map", mat.normal_map, NO_NORMAL_MAP_SLOT)?;
    let emissive_map_slot = slot_of("emissive_map", mat.emissive_map, 0)?;
    let orm_map_slot = slot_of("orm_map", mat.orm_map, 0)?;
    Ok(MaterialEntry {
        albedo_slot,
        normal_map_slot,
        uniforms: MaterialUniforms {
            roughness: mat.roughness,
            metallic: mat.metallic,
            alpha_cutoff: mat.alpha_cutoff,
            opacity: mat.opacity,
            tint: mat.tint,
            _pad0: 0.0,
            emissive: mat.emissive_factor,
            _pad1: 0.0,
            emissive_map_index: emissive_map_slot as u32,
            orm_map_index: orm_map_slot as u32,
            transparent: u32::from(mat.transparent),
            see_through: u32::from(mat.see_through),
            params_index: material_params::row_of(Some(handle)),
            far_field: FarFieldTint::NONE,
        },
        shader_bucket: mat.shader.map_or(0, |h| h.0),
    })
}

// Resolve the material a draw object binds. A material handle must resolve in
// `material_map`; an unresolved one comes back as `Err(handle)` so the caller
// can log its own context. A draw naming no material binds `UNTEXTURED`.
pub(crate) fn resolve_material_slots(
    material: Option<MaterialHandle>,
    material_map: &std::collections::HashMap<MaterialHandle, MaterialEntry>,
) -> Result<MaterialEntry, MaterialHandle> {
    match material {
        Some(mat_id) => material_map.get(&mat_id).copied().ok_or(mat_id),
        None => Ok(MaterialEntry::UNTEXTURED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material() -> Material {
        Material {
            roughness: 0.25,
            metallic: 0.75,
            ..Default::default()
        }
    }

    // An unset reference takes its field's own fallback: the two sentinels for
    // albedo and the normal map, and slot 0 for the maps the shader gates on.
    #[test]
    fn unset_references_take_their_fallbacks() {
        let entry = of(MaterialHandle::new(0), &material(), 4).expect("bakes");
        assert_eq!(entry.albedo_slot, NO_ALBEDO_SLOT);
        assert_eq!(entry.normal_map_slot, NO_NORMAL_MAP_SLOT);
        assert_eq!(entry.uniforms.emissive_map_index, 0);
        assert_eq!(entry.uniforms.orm_map_index, 0);
        assert_eq!(entry.uniforms.roughness, 0.25);
        assert_eq!(entry.uniforms.metallic, 0.75);
    }

    #[test]
    fn a_set_reference_resolves_to_its_pool_slot() {
        let mut mat = material();
        mat.albedo = Some(TextureHandle::new(2));
        mat.normal_map = Some(TextureHandle::new(3));
        let entry = of(MaterialHandle::new(0), &mat, 4).expect("bakes");
        assert_eq!(entry.albedo_slot, 2);
        assert_eq!(entry.normal_map_slot, 3);
    }

    // A reference past the pool is a corrupt build; the field is named so the
    // caller can log which one.
    #[test]
    fn a_reference_past_the_pool_names_its_field() {
        let mut mat = material();
        mat.orm_map = Some(TextureHandle::new(9));
        assert_eq!(of(MaterialHandle::new(0), &mat, 4).err(), Some("orm_map"));
    }

    #[test]
    fn the_shader_reference_becomes_the_draw_bucket() {
        let mut mat = material();
        assert_eq!(
            of(MaterialHandle::new(0), &mat, 0)
                .expect("bakes")
                .shader_bucket,
            0
        );
        mat.shader = Some(concinnity_core::ecs::ShaderHandle::new(3));
        assert_eq!(
            of(MaterialHandle::new(0), &mat, 0)
                .expect("bakes")
                .shader_bucket,
            3
        );
    }

    // Every draw of one material reads the parameter row after its handle, so
    // draws sharing a material share the row; a draw without one reads row 0.
    #[test]
    fn a_material_draws_with_the_parameter_row_after_its_handle() {
        let mat = material();
        let entry = of(MaterialHandle::new(3), &mat, 0).expect("bakes");
        assert_eq!(entry.uniforms.params_index, 4);
        let map = std::collections::HashMap::from([(MaterialHandle::new(3), entry)]);
        let draws = [
            Some(MaterialHandle::new(3)),
            Some(MaterialHandle::new(3)),
            None,
        ];
        let rows: Vec<u32> = draws
            .iter()
            .map(|&m| {
                resolve_material_slots(m, &map)
                    .expect("resolves")
                    .uniforms
                    .params_index
            })
            .collect();
        assert_eq!(rows, [4, 4, material_params::NO_MATERIAL_ROW]);
    }

    // A draw naming no material binds the reserved fallbacks under the default
    // uniforms and the world default shader bucket.
    #[test]
    fn a_draw_without_a_material_is_untextured() {
        let entry = MaterialEntry::UNTEXTURED;
        assert_eq!(entry.albedo_slot, NO_ALBEDO_SLOT);
        assert_eq!(entry.normal_map_slot, NO_NORMAL_MAP_SLOT);
        assert_eq!(entry.shader_bucket, 0);

        let map = std::collections::HashMap::new();
        assert_eq!(
            resolve_material_slots(None, &map)
                .expect("resolves")
                .albedo_slot,
            NO_ALBEDO_SLOT
        );
    }

    // A material handle must resolve in the table.
    #[test]
    fn a_material_handle_must_resolve() {
        let entry = of(MaterialHandle::new(0), &material(), 4).expect("bakes");
        let map = std::collections::HashMap::from([(MaterialHandle::new(2), entry)]);
        let got = resolve_material_slots(Some(MaterialHandle::new(2)), &map).expect("resolves");
        assert_eq!(got.uniforms.roughness, 0.25);
        assert_eq!(got.albedo_slot, NO_ALBEDO_SLOT, "the material's own albedo");
        assert_eq!(
            resolve_material_slots(Some(MaterialHandle::new(5)), &map).err(),
            Some(MaterialHandle::new(5))
        );
    }
}
