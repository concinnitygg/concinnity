//! The texture and material tables a world load reads: each texture's payload
//! locator and whether only the cook reads it, and each material's draw entry
//! and parameter row.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::components::Material;
use crate::ecs::{MaterialHandle, PayloadLocator};
use crate::gfx::render_types::GpuMaterialParams;
use crate::render::material_entry::{self, MaterialEntry};
use crate::render::material_params;
use crate::resource::{MaterialTable, TextureTable};

/// The texture pool a world load reads from its TextureTable.
pub struct TexturePool {
    /// Each texture's payload locator, by pool slot.
    pub locators: Vec<PayloadLocator>,
    /// Slots only the cook reads, which stay placeholders on the GPU.
    pub cook_only: BTreeSet<usize>,
}

/// The materials a world load decodes from its MaterialTable.
pub struct DecodedMaterials {
    /// Each material's draw entry, by handle.
    pub map: HashMap<MaterialHandle, MaterialEntry>,
    /// The material parameter table's rows.
    pub params: Vec<GpuMaterialParams>,
}

/// Why a table could not be read: each marks a corrupt build.
#[derive(Debug, PartialEq)]
pub enum TableError {
    /// A texture with no compiled payload.
    MissingTexturePayload,
    /// A texture record that does not decode.
    MalformedTexture(String),
    /// A material that does not decode.
    MalformedMaterial {
        /// The material's handle.
        handle: usize,
        /// The decode error.
        error: String,
    },
    /// A material naming a texture past the pool.
    MaterialTextureOutOfRange {
        /// The material's handle.
        handle: usize,
        /// The reference that points past the pool.
        field: &'static str,
        /// How many textures the pool holds.
        textures: usize,
    },
}

impl core::fmt::Display for TableError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingTexturePayload => {
                write!(
                    f,
                    "a Texture has no compiled payload; did the build succeed?"
                )
            }
            Self::MalformedTexture(e) => write!(f, "malformed texture record: {e}"),
            Self::MalformedMaterial { handle, error } => {
                write!(f, "Material handle {handle} failed to decode: {error}")
            }
            Self::MaterialTextureOutOfRange {
                handle,
                field,
                textures,
            } => write!(
                f,
                "Material {handle} references an out-of-range {field} texture handle (only {textures} textures)"
            ),
        }
    }
}

/// The texture pool `table` describes.
pub fn texture_pool(table: &TextureTable) -> Result<TexturePool, TableError> {
    let locators = table
        .0
        .iter()
        .map(|entry| entry.payload.clone())
        .collect::<Option<Vec<_>>>()
        .ok_or(TableError::MissingTexturePayload)?;
    let cook_only = table
        .cook_only_slots()
        .map_err(|e| TableError::MalformedTexture(e.to_string()))?;
    Ok(TexturePool {
        locators,
        cook_only,
    })
}

/// Every material in `table` (dense by `MaterialHandle`) against a texture
/// pool of `texture_count` entries. Materials have no payload; all their data
/// lives in the baked `data_bytes`.
pub fn decode_materials(
    table: &MaterialTable,
    texture_count: usize,
) -> Result<DecodedMaterials, TableError> {
    let mut map = HashMap::with_capacity(table.len());
    let mut params = Vec::with_capacity(table.len());
    for (index, entry) in table.0.iter().enumerate() {
        let mat: Material =
            postcard::from_bytes(&entry.data_bytes).map_err(|e| TableError::MalformedMaterial {
                handle: index,
                error: e.to_string(),
            })?;
        let handle = MaterialHandle::new(index as u32);
        params.push(mat.params);
        let drawn = material_entry::of(handle, &mat, texture_count).map_err(|field| {
            TableError::MaterialTextureOutOfRange {
                handle: index,
                field,
                textures: texture_count,
            }
        })?;
        map.insert(handle, drawn);
    }
    Ok(DecodedMaterials {
        map,
        params: material_params::rows(params),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;
    use alloc::vec;

    fn locator(offset: u64) -> PayloadLocator {
        PayloadLocator {
            blob_index: 0,
            offset,
            len: 4,
        }
    }

    fn texture(payload: Option<PayloadLocator>) -> ResourceEntry {
        ResourceEntry {
            payload,
            ..Default::default()
        }
    }

    #[test]
    fn a_texture_pool_lists_each_locator_by_slot() {
        let table = TextureTable(vec![texture(Some(locator(0))), texture(Some(locator(4)))]);
        let pool = texture_pool(&table).expect("every texture has a payload");
        assert_eq!(pool.locators, vec![locator(0), locator(4)]);
        assert!(pool.cook_only.is_empty());
    }

    #[test]
    fn a_texture_without_a_payload_fails_the_pool() {
        let table = TextureTable(vec![texture(Some(locator(0))), texture(None)]);
        assert_eq!(
            texture_pool(&table).err(),
            Some(TableError::MissingTexturePayload)
        );
    }

    #[test]
    fn materials_decode_by_handle_with_a_row_each() {
        let bytes = postcard::to_allocvec(&Material::default()).expect("encodes");
        let table = MaterialTable(vec![ResourceEntry {
            data_bytes: bytes,
            ..Default::default()
        }]);
        let decoded = decode_materials(&table, 1).expect("decodes");
        assert!(decoded.map.contains_key(&MaterialHandle::new(0)));
        // Row 0 is the no-material row.
        assert_eq!(decoded.params.len(), 2);
    }

    #[test]
    fn an_undecodable_material_names_its_handle() {
        let table = MaterialTable(vec![ResourceEntry {
            data_bytes: vec![0xFF; 3],
            ..Default::default()
        }]);
        assert!(matches!(
            decode_materials(&table, 1),
            Err(TableError::MalformedMaterial { handle: 0, .. })
        ));
    }
}
