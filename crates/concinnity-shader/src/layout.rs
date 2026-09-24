//! The byte layout of every buffer-backed struct a SPIR-V module declares.
//!
//! A `#[repr(C)]` struct the CPU uploads has to match what the shader reads,
//! and the only reading of the shader's side that cannot drift is the compiled
//! module's own. This reports it: member offsets and sizes as the decorations
//! state them, so a `#[repr(C)]` mirror can be compared field by field.
//!
//! The layout is a property of the module, not of the language, so which
//! layout rules dxc applied is the caller's choice
//! ([`HlslTarget::SpirvWithVulkanLayout`] or [`HlslTarget::SpirvWithDxLayout`])
//! -- they are not the same for every backend.
//!
//! [`HlslTarget::SpirvWithVulkanLayout`]: crate::HlslTarget::SpirvWithVulkanLayout
//! [`HlslTarget::SpirvWithDxLayout`]: crate::HlslTarget::SpirvWithDxLayout

use std::collections::BTreeMap;

use spirv_cross2::reflect::{ResourceType, TypeInner};
use spirv_cross2::targets::None as NoTarget;
use spirv_cross2::{Compiler, Module};

/// One member of a shader struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// The member's declared name.
    pub name: String,
    /// Bytes from the start of the struct.
    pub offset: usize,
    /// Declared size of the member.
    pub size: usize,
}

/// One shader struct's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout {
    /// Members, in declaration order.
    pub fields: Vec<Field>,
    /// The size the struct occupies, as the module declares it.
    pub size: usize,
}

impl StructLayout {
    /// The byte past the last declared member: what the struct occupies before
    /// a target rounds the block it sits in.
    pub fn extent(&self) -> usize {
        self.fields
            .iter()
            .map(|f| f.offset + f.size)
            .max()
            .unwrap_or(0)
    }
}

// Every kind whose element type is a struct the CPU can upload into.
const BUFFERS: &[ResourceType] = &[
    ResourceType::UniformBuffer,
    ResourceType::StorageBuffer,
    ResourceType::PushConstant,
];

/// Every struct `spirv` declares behind a buffer, keyed by its shader-side
/// name, whether or not its entry point uses the buffer. Nested structs are reported under their own names as well, so a block
/// that embeds a record covers both, and so does a structured buffer, whose
/// record sits inside the wrapper dxc generates for it.
pub fn struct_layouts(spirv: &[u8]) -> Result<BTreeMap<String, StructLayout>, String> {
    let words = crate::words(spirv)?;
    let compiler =
        Compiler::<NoTarget>::new(Module::from_words(&words)).map_err(|e| format!("hlsl: {e}"))?;
    let resources = compiler
        .shader_resources()
        .map_err(|e| format!("hlsl: {e}"))?;
    let mut found = BTreeMap::new();
    for kind in BUFFERS {
        for resource in resources
            .resources_for_type(*kind)
            .map_err(|e| format!("hlsl: {e}"))?
        {
            collect(&compiler, resource.base_type_id, &mut found)?;
        }
    }
    Ok(found)
}

fn collect(
    compiler: &Compiler<NoTarget>,
    type_id: spirv_cross2::handle::Handle<spirv_cross2::handle::TypeId>,
    found: &mut BTreeMap<String, StructLayout>,
) -> Result<(), String> {
    let described = compiler
        .type_description(type_id)
        .map_err(|e| format!("hlsl: {e}"))?;
    let structure = match &described.inner {
        TypeInner::Struct(structure) => structure,
        // A structured buffer's record and an array member both reach their
        // element type through one of these.
        TypeInner::Array { base, .. } | TypeInner::Pointer { base, .. } => {
            return collect(compiler, *base, found);
        }
        _ => return Ok(()),
    };
    let name = shader_name(described.name.as_ref().map(ToString::to_string));
    let fields = structure
        .members
        .iter()
        .map(|member| Field {
            name: member
                .name
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            offset: member.offset as usize,
            size: member.size,
        })
        .collect();
    // A struct reached twice is the same declaration, so the first reading
    // stands and recursion stops rather than looping on a self-referential type.
    if found
        .insert(
            name,
            StructLayout {
                fields,
                size: structure.size,
            },
        )
        .is_none()
    {
        for member in &structure.members {
            collect(compiler, member.id, found)?;
        }
    }
    Ok(())
}

// dxc names a SPIR-V struct after the HLSL type it wraps: a
// `ConstantBuffer<DecalView>` becomes `type.ConstantBuffer.DecalView` and a
// `StructuredBuffer<Particle>` a `type.StructuredBuffer.Particle` whose one
// member is the record. A caller comparing a `#[repr(C)]` mirror names the
// struct the shader declares, so the wrapper spelling is peeled off.
fn shader_name(name: Option<String>) -> String {
    let name = name.unwrap_or_default();
    match name.strip_prefix("type.") {
        Some(wrapped) => wrapped.rsplit('.').next().unwrap_or(wrapped).to_string(),
        None => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A mirror names the struct the shader declares, not the buffer wrapper dxc
    // generated around it.
    #[test]
    fn a_wrapper_spelling_reduces_to_the_declared_struct() {
        assert_eq!(
            shader_name(Some("type.ConstantBuffer.DecalView".into())),
            "DecalView"
        );
        assert_eq!(
            shader_name(Some("type.StructuredBuffer.Particle".into())),
            "Particle"
        );
        assert_eq!(
            shader_name(Some("type.PushConstant.Params".into())),
            "Params"
        );
        assert_eq!(shader_name(Some("ParticleParams".into())), "ParticleParams");
        assert_eq!(shader_name(None), "");
    }

    // The extent ends at the member that reaches furthest, which is not always
    // the last one declared, and ignores the rounding the block size carries.
    #[test]
    fn the_extent_ends_at_the_furthest_member() {
        let field = |name: &str, offset, size| Field {
            name: name.to_string(),
            offset,
            size,
        };
        let layout = StructLayout {
            fields: vec![field("wide", 0, 16), field("tail", 8, 4)],
            size: 32,
        };
        assert_eq!(layout.extent(), 16);
        let empty = StructLayout {
            fields: Vec::new(),
            size: 0,
        };
        assert_eq!(empty.extent(), 0);
    }
}
