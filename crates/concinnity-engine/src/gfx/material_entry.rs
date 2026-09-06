// src/gfx/material_entry.rs
//
// One decoded Material as the draw list consumes it: the GPU uniforms plus the
// shared-pool slots each texture reference resolves to. The translation from a
// compiled `Material` to this is the renderer's own reading of the asset, so it
// lives here rather than inside the init pass that first needed it: init bakes
// the whole table at load, and the editor's live draw seam bakes one material
// again when an edit reassigns it.

use crate::components::Material;
use crate::ecs::{MaterialHandle, TextureHandle};
use crate::gfx::render_types::{MaterialUniforms, NO_ALBEDO_SLOT, NO_NORMAL_MAP_SLOT};

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

/// The entry a decoded `Material` bakes to against a texture pool of
/// `texture_count` entries. `Err` names the reference that points past the
/// pool, which cook validated and so marks a corrupt build.
pub(crate) fn of(mat: &Material, texture_count: usize) -> Result<MaterialEntry, &'static str> {
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
        },
        shader_bucket: mat.shader.map_or(0, |h| h.0),
    })
}

/// The entry a draw binds with no material of its own: the legacy texture
/// reference contributes its albedo slot (falling back to the white entry when
/// past `texture_count`) over the default material.
pub(crate) fn from_texture(texture: Option<TextureHandle>, texture_count: usize) -> MaterialEntry {
    let albedo_slot = match texture {
        Some(tex_id) if tex_id.index() < texture_count => tex_id.index(),
        _ => NO_ALBEDO_SLOT,
    };
    MaterialEntry {
        albedo_slot,
        normal_map_slot: NO_NORMAL_MAP_SLOT,
        uniforms: MaterialUniforms::DEFAULT,
        shader_bucket: 0,
    }
}

// Resolve the (albedo_slot, normal_map_slot, material) a draw object binds. A
// material handle wins and must resolve in `material_map`; an unresolved one
// comes back as `Err(handle)` so the caller can log its own context. With no
// material, the legacy texture path above applies.
pub(crate) fn resolve_material_slots(
    material: Option<MaterialHandle>,
    texture: Option<TextureHandle>,
    material_map: &std::collections::HashMap<MaterialHandle, MaterialEntry>,
    texture_count: usize,
) -> Result<MaterialEntry, MaterialHandle> {
    if let Some(mat_id) = material {
        return material_map.get(&mat_id).copied().ok_or(mat_id);
    }
    Ok(from_texture(texture, texture_count))
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
        let entry = of(&material(), 4).expect("bakes");
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
        mat.albedo = Some(TextureHandle(2));
        mat.normal_map = Some(TextureHandle(3));
        let entry = of(&mat, 4).expect("bakes");
        assert_eq!(entry.albedo_slot, 2);
        assert_eq!(entry.normal_map_slot, 3);
    }

    // A reference past the pool is a corrupt build; the field is named so the
    // caller can log which one.
    #[test]
    fn a_reference_past_the_pool_names_its_field() {
        let mut mat = material();
        mat.orm_map = Some(TextureHandle(9));
        assert_eq!(of(&mat, 4).err(), Some("orm_map"));
    }

    #[test]
    fn the_shader_reference_becomes_the_draw_bucket() {
        let mut mat = material();
        assert_eq!(of(&mat, 0).expect("bakes").shader_bucket, 0);
        mat.shader = Some(concinnity_core::ecs::ShaderHandle(3));
        assert_eq!(of(&mat, 0).expect("bakes").shader_bucket, 3);
    }

    // The legacy texture path: an in-range handle is the albedo slot. Slot 0
    // would be a real texture, so an unusable handle takes the white fallback
    // instead, under the default material and bucket.
    #[test]
    fn a_texture_only_draw_takes_the_default_material() {
        let entry = from_texture(Some(TextureHandle(1)), 4);
        assert_eq!(entry.albedo_slot, 1);
        assert_eq!(entry.normal_map_slot, NO_NORMAL_MAP_SLOT);
        assert_eq!(entry.shader_bucket, 0);
        assert_eq!(
            from_texture(Some(TextureHandle(9)), 4).albedo_slot,
            NO_ALBEDO_SLOT
        );
        assert_eq!(from_texture(None, 4).albedo_slot, NO_ALBEDO_SLOT);
    }

    // A material handle outranks the legacy texture and must resolve.
    #[test]
    fn a_material_handle_wins_over_the_texture() {
        let entry = of(&material(), 4).expect("bakes");
        let map = std::collections::HashMap::from([(MaterialHandle(2), entry)]);
        let got = resolve_material_slots(Some(MaterialHandle(2)), Some(TextureHandle(1)), &map, 4)
            .expect("resolves");
        assert_eq!(got.uniforms.roughness, 0.25);
        assert_eq!(got.albedo_slot, NO_ALBEDO_SLOT, "the material's own albedo");
        assert_eq!(
            resolve_material_slots(Some(MaterialHandle(5)), None, &map, 4).err(),
            Some(MaterialHandle(5))
        );
    }
}
