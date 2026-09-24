//! SPIR-V to MSL, through spirv-cross as a linked library rather than its CLI.
//!
//! The CLI cannot assign Metal indices per resource at any version: it exposes
//! `--msl-decoration-binding`, which forces the Metal index to equal the SPIR-V
//! binding and ignores the descriptor set, and nothing finer. Only
//! `spvc_compiler_msl_add_resource_binding_2` can, which is the call this
//! module is here to make -- and the table it is handed is generated, never
//! written (see `metal_bindings`).

use spirv_cross2::compile::CompilableTarget;
use spirv_cross2::compile::msl::{ArgumentBuffersTier, BindTarget, MslVersion, ResourceBinding};
use spirv_cross2::handle::{Handle, VariableId};
use spirv_cross2::reflect::{DecorationValue, ResourceType};
use spirv_cross2::targets::Msl;
use spirv_cross2::{Compiler, Module, spirv};

use crate::declarations::{Slot, resource_declarations};
use crate::metal_bindings::{
    ArgumentBufferSet, MetalClass, Reflected, argument_buffer_sets, metal_bindings,
};
use crate::stage::Stage;

// The MSL version the Metal backend's pipelines are built against, matching the
// `-std=metal3.0` the metallib step compiles with.
const MSL_VERSION: MslVersion = MslVersion::new(3, 0, 0);

// Which SPIR-V resource kinds land in which Metal namespace. Storage images and
// subpass inputs are textures; acceleration structures and every buffer kind
// are buffers.
const KINDS: &[(ResourceType, MetalClass)] = &[
    (ResourceType::UniformBuffer, MetalClass::Buffer),
    (ResourceType::StorageBuffer, MetalClass::Buffer),
    (ResourceType::PushConstant, MetalClass::Buffer),
    (ResourceType::AccelerationStructure, MetalClass::Buffer),
    (ResourceType::SeparateImage, MetalClass::Texture),
    (ResourceType::StorageImage, MetalClass::Texture),
    (ResourceType::SubpassInput, MetalClass::Texture),
    (ResourceType::SeparateSamplers, MetalClass::Sampler),
];

/// Translate a SPIR-V module to MSL, binding every resource at the index
/// `source`'s `register()` annotations name.
///
/// `source` is the preprocessed HLSL the module was compiled from: a
/// declaration behind an inactive `#if` is not a declaration.
pub(crate) fn translate(spirv: &[u8], source: &str, stage: Stage) -> Result<String, String> {
    let words = crate::words(spirv)?;
    let mut compiler =
        Compiler::<Msl>::new(Module::from_words(&words)).map_err(|e| format!("hlsl: {e}"))?;

    // dxc kept every declared resource in this module and in the entry's
    // interface (see `dxc::Emit`), so the reflection below sees each one.
    // spirv-cross narrows what it emits to the resources the entry uses, and
    // `force_active_argument_buffer_resources` exempts the members of an
    // argument buffer: Metal lays the buffer out by the members the function
    // declares, and the host encodes all of them. The binding table follows
    // the same split.
    let declarations = resource_declarations(source);
    let used = compiler
        .active_interface_variables()
        .map_err(|e| format!("hlsl: {e}"))?
        .to_handles();
    let declared = declared_resources(&compiler, &used)?;
    let argument_buffers = live_argument_buffers(argument_buffer_sets(&declarations)?, &declared);

    let asked: Vec<Reflected<'_>> = declared
        .iter()
        .filter(|r| r.used || in_argument_buffer(r.slot, &argument_buffers))
        .map(|r| Reflected {
            class: r.class,
            slot: r.slot,
            name: &r.name,
        })
        .collect();
    let table = metal_bindings(&asked, &declarations)?;

    let model = execution_model(stage);
    for row in &table {
        let binding = match row.slot {
            Slot::PushConstant => ResourceBinding::PushConstantBuffer,
            Slot::Qualified { set, binding } => ResourceBinding::from_qualified(set, binding),
        };
        let target = BindTarget {
            buffer: row.buffer,
            texture: row.texture,
            sampler: row.sampler,
            count: row.count.and_then(|c| c.try_into().ok()),
        };
        compiler
            .add_resource_binding(model, binding, &target)
            .map_err(|e| format!("hlsl: {e}"))?;
    }

    // A set declared `[[cn::metal_argument_buffer(n)]]` is bound as one buffer
    // at `n`; every other set stays discrete, which is what the Metal encoders
    // write. spirv-cross has no per-set opt-IN, only a global switch plus a
    // per-set opt-OUT, so a module with one argument buffer turns the switch on
    // and names every other set discrete -- including a declared argument
    // buffer this entry uses nothing from, which would otherwise be forced
    // active into a parameter its encoder never binds.
    for set in &argument_buffers {
        compiler
            .add_resource_binding(
                model,
                ResourceBinding::ArgumentBuffer(set.set),
                &BindTarget {
                    buffer: set.buffer,
                    texture: 0,
                    sampler: 0,
                    count: None,
                },
            )
            .map_err(|e| format!("hlsl: {e}"))?;
        // An unsized member is indexed past its declared length, which Metal
        // allows only through a `device` reference to the buffer.
        if set.runtime_sized {
            compiler
                .set_argument_buffer_device_address_space(set.set, true)
                .map_err(|e| format!("hlsl: {e}"))?;
        }
    }
    for set in discrete_sets(declared.iter().map(|r| r.slot), &argument_buffers) {
        compiler
            .add_discrete_descriptor_set(set)
            .map_err(|e| format!("hlsl: {e}"))?;
    }

    let mut options = Msl::options();
    options.version = MSL_VERSION;
    // Off unless the source asked for one: the Metal encoders write buffer,
    // texture and sampler slots directly, and an argument buffer nothing
    // declared would be a binding-shape change rather than a translation.
    options.argument_buffers = !argument_buffers.is_empty();
    options.force_active_argument_buffer_resources = true;
    // An unsized array of resources needs tier 2, where the host writes each
    // member as a plain resource id rather than through an argument encoder.
    if argument_buffers.iter().any(|set| set.runtime_sized) {
        options.argument_buffers_tier = ArgumentBuffersTier::Tier2;
    }
    compiler
        .compile(&options)
        .map(|artifact| artifact.to_string())
        .map_err(|e| format!("hlsl: MSL translation failed: {e}"))
}

// One resource the module declares, and whether the entry uses it.
struct Declared {
    class: MetalClass,
    slot: Slot,
    name: String,
    used: bool,
}

fn declared_resources(
    compiler: &Compiler<Msl>,
    used: &[Handle<VariableId>],
) -> Result<Vec<Declared>, String> {
    let resources = compiler
        .shader_resources()
        .map_err(|e| format!("hlsl: {e}"))?;
    let mut declared = Vec::new();
    for (kind, class) in KINDS {
        for resource in resources
            .resources_for_type(*kind)
            .map_err(|e| format!("hlsl: {e}"))?
        {
            declared.push(Declared {
                class: *class,
                slot: slot_of(compiler, &resource, *kind)?,
                name: resource.name.to_string(),
                used: used.contains(&resource.id),
            });
        }
    }
    Ok(declared)
}

// The declared argument buffers this entry uses at least one member of. One
// it uses nothing from is not part of its interface at all.
fn live_argument_buffers(
    sets: Vec<ArgumentBufferSet>,
    declared: &[Declared],
) -> Vec<ArgumentBufferSet> {
    sets.into_iter()
        .filter(|set| {
            declared
                .iter()
                .any(|r| r.used && in_argument_buffer(r.slot, std::slice::from_ref(set)))
        })
        .collect()
}

/// Every descriptor set among `slots` that is not an argument buffer.
///
/// Only consulted when the module has an argument buffer: with the global
/// switch off, naming a set discrete means nothing either way.
fn discrete_sets(
    slots: impl Iterator<Item = Slot>,
    argument_buffers: &[ArgumentBufferSet],
) -> Vec<u32> {
    let mut sets: Vec<u32> = slots
        .filter_map(|slot| match slot {
            Slot::Qualified { set, .. } => Some(set),
            Slot::PushConstant => None,
        })
        .filter(|set| !argument_buffers.iter().any(|a| a.set == *set))
        .collect();
    sets.sort_unstable();
    sets.dedup();
    sets
}

fn in_argument_buffer(slot: Slot, argument_buffers: &[ArgumentBufferSet]) -> bool {
    match slot {
        Slot::Qualified { set, .. } => argument_buffers.iter().any(|a| a.set == set),
        Slot::PushConstant => false,
    }
}

fn slot_of(
    compiler: &Compiler<Msl>,
    resource: &spirv_cross2::reflect::Resource<'_>,
    kind: ResourceType,
) -> Result<Slot, String> {
    if kind == ResourceType::PushConstant {
        return Ok(Slot::PushConstant);
    }
    let literal = |decoration| match compiler.decoration(resource.id, decoration) {
        Ok(Some(DecorationValue::Literal(value))) => Ok(value),
        _ => Err(format!(
            "hlsl: `{}` carries no {decoration:?} decoration",
            resource.name
        )),
    };
    Ok(Slot::Qualified {
        set: literal(spirv::Decoration::DescriptorSet)?,
        binding: literal(spirv::Decoration::Binding)?,
    })
}

fn execution_model(stage: Stage) -> spirv::ExecutionModel {
    match stage {
        Stage::Vertex => spirv::ExecutionModel::Vertex,
        Stage::Pixel => spirv::ExecutionModel::Fragment,
        Stage::Compute => spirv::ExecutionModel::GLCompute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(set: u32, binding: u32) -> Slot {
        Slot::Qualified { set, binding }
    }

    fn declared(slot: Slot, used: bool) -> Declared {
        Declared {
            class: MetalClass::Texture,
            slot,
            name: String::new(),
            used,
        }
    }

    // spirv-cross only takes a per-set opt-OUT, so every set the module binds
    // and did not declare has to be named. A set left off would be swept into
    // an argument buffer no encoder writes.
    #[test]
    fn every_set_but_the_declared_ones_is_named_discrete() {
        let slots = [Slot::PushConstant, at(0, 0), at(0, 3), at(1, 7), at(2, 0)];
        let declared = [ArgumentBufferSet {
            set: 2,
            buffer: 11,
            runtime_sized: false,
        }];
        assert_eq!(discrete_sets(slots.into_iter(), &declared), [0, 1]);
    }

    // A module declaring none keeps the switch off entirely, so the discrete
    // list is never consulted -- but it still has to be the whole set list.
    #[test]
    fn with_no_argument_buffer_every_set_is_discrete() {
        assert_eq!(discrete_sets([at(0, 0), at(1, 0)].into_iter(), &[]), [0, 1]);
    }

    // A vertex stage that reads none of the fragment's textures must not gain
    // the texture argument buffer as a parameter: nothing binds it there.
    #[test]
    fn an_argument_buffer_the_entry_uses_nothing_from_is_not_live() {
        let sets = vec![
            ArgumentBufferSet {
                set: 1,
                buffer: 7,
                runtime_sized: false,
            },
            ArgumentBufferSet {
                set: 2,
                buffer: 10,
                runtime_sized: false,
            },
        ];
        let resources = [
            declared(at(0, 0), true),
            declared(at(1, 0), false),
            declared(at(1, 1), false),
            declared(at(2, 0), false),
            declared(at(2, 4), true),
        ];
        assert_eq!(
            live_argument_buffers(sets, &resources),
            [ArgumentBufferSet {
                set: 2,
                buffer: 10,
                runtime_sized: false
            }]
        );
    }

    // An argument-buffer member is bound whether the entry reads it or not;
    // a discrete resource only when it does.
    #[test]
    fn only_a_declared_set_counts_as_an_argument_buffer() {
        let declared = [ArgumentBufferSet {
            set: 2,
            buffer: 7,
            runtime_sized: false,
        }];
        assert!(in_argument_buffer(
            Slot::Qualified { set: 2, binding: 5 },
            &declared
        ));
        assert!(!in_argument_buffer(
            Slot::Qualified { set: 1, binding: 5 },
            &declared
        ));
        assert!(!in_argument_buffer(Slot::PushConstant, &declared));
    }

    // Every kind the engine's shaders can declare has a Metal namespace. A kind
    // missing here would reflect as nothing and take whatever slot the emitter
    // picked, which is the silent misbind the table exists to prevent.
    #[test]
    fn every_reflected_kind_the_engine_declares_has_a_metal_class() {
        for kind in [
            ResourceType::UniformBuffer,
            ResourceType::StorageBuffer,
            ResourceType::PushConstant,
            ResourceType::AccelerationStructure,
            ResourceType::SeparateImage,
            ResourceType::StorageImage,
            ResourceType::SeparateSamplers,
        ] {
            assert!(KINDS.iter().any(|(k, _)| *k == kind), "{kind:?}");
        }
    }
}
