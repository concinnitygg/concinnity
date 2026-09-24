//! The Metal binding table, derived from a module's SPIR-V resources and the
//! `register()` annotations its source carries.
//!
//! A hand-maintained table would let the three targets drift apart as a silent
//! misbind, so the table is derived, and the policy is one sentence: **a resource's Metal index is the number on its `register()`, in
//! the Metal class its SPIR-V kind implies.** The three Metal namespaces are
//! disjoint, so a structured buffer at `t0` and a texture at `t0` are
//! `buffer(0)` and `texture(0)` and do not collide -- which is what lets one
//! annotation serve D3D and Metal at once wherever the two agree, and an
//! existing `#ifdef CN_BACKEND_DIRECTX` block move it where they do not.
//!
//! A resource is paired with its declaration by that Vulkan slot, so every
//! engine `.hlsl` resource carries a `vk::` annotation as well as a
//! `register()` -- including one on a block only Metal binds, where the
//! annotation is simply the slot dxc would assign from the register anyway.
//! Two declarations may not share a slot, because then the pairing has no
//! answer and would take whichever came first.

use crate::declarations::{Declaration, Slot};

/// The Metal namespace a reflected SPIR-V resource binds into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalClass {
    /// Constant buffers, structured buffers, push constants, acceleration
    /// structures: everything MSL takes as a `[[buffer(n)]]`.
    Buffer,
    /// A separate texture, storage image or subpass input.
    Texture,
    /// A separate sampler.
    Sampler,
}

/// One resource as SPIR-V reflection reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reflected<'a> {
    /// Metal namespace implied by the SPIR-V kind.
    pub class: MetalClass,
    /// Where the module binds it.
    pub slot: Slot,
    /// The declared name, for the error when no annotation matches.
    pub name: &'a str,
}

/// One row of the table handed to the MSL emitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MetalBinding {
    /// The SPIR-V slot this row remaps.
    pub slot: Slot,
    /// `[[buffer(n)]]`, when the class uses one.
    pub buffer: u32,
    /// `[[texture(n)]]`, when the class uses one.
    pub texture: u32,
    /// `[[sampler(n)]]`, when the class uses one.
    pub sampler: u32,
    /// Elements, for a resource array.
    pub count: Option<u32>,
}

/// One descriptor set Metal binds as a single argument buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArgumentBufferSet {
    /// The descriptor set, as its members' `vk::binding` names it.
    pub set: u32,
    /// The Metal buffer index the whole set lands on.
    pub buffer: u32,
    /// A member is an unsized array. Metal can only index one past its declared
    /// length from a `device` argument buffer, which argument-buffer tier 2
    /// writes as plain resource ids.
    pub runtime_sized: bool,
}

/// Every descriptor set `source` declares as a Metal argument buffer.
///
/// Errs when two declarations in one set name different buffer indices, or when
/// the annotation sits on a push constant, which is not a set.
pub(crate) fn argument_buffer_sets(
    declarations: &[Declaration],
) -> Result<Vec<ArgumentBufferSet>, String> {
    let mut found: Vec<ArgumentBufferSet> = Vec::new();
    for d in declarations {
        let Some(buffer) = d.metal_argument_buffer else {
            continue;
        };
        let Some(Slot::Qualified { set, .. }) = d.slot else {
            return Err(format!(
                "`{}` declares metal_argument_buffer({buffer}) but binds no descriptor set: an \
                 argument buffer is a whole set, so the declaration needs a \
                 [[vk::binding(binding, set)]]",
                d.name
            ));
        };
        match found.iter().find(|a| a.set == set) {
            Some(a) if a.buffer != buffer => {
                return Err(format!(
                    "set {set} is declared as a Metal argument buffer at both buffer({}) and \
                     buffer({buffer}); one set lands on one index",
                    a.buffer
                ));
            }
            Some(_) => {}
            None => found.push(ArgumentBufferSet {
                set,
                buffer,
                runtime_sized: false,
            }),
        }
    }
    for set in &mut found {
        set.runtime_sized = declarations.iter().any(|d| {
            d.runtime_sized
                && matches!(d.slot, Some(Slot::Qualified { set: s, .. }) if s == set.set)
        });
    }
    Ok(found)
}

/// The Metal index for every resource the module binds.
///
/// Errs naming the resource when its source declares no matching annotation:
/// an index the policy cannot derive must fail the compile rather than reach a
/// pipeline as whatever slot the emitter picked.
pub(crate) fn metal_bindings(
    resources: &[Reflected<'_>],
    declarations: &[Declaration],
) -> Result<Vec<MetalBinding>, String> {
    one_declaration_per_slot(declarations)?;
    resources
        .iter()
        .map(|resource| binding(resource, declarations))
        .collect()
}

// Two declarations at one slot leave the pairing with no answer: the resource
// takes whichever was declared first, which is a silent misbind rather than a
// failure.
fn one_declaration_per_slot(declarations: &[Declaration]) -> Result<(), String> {
    for (i, d) in declarations.iter().enumerate() {
        if let Some(slot) = d.slot
            && let Some(other) = declarations[..i].iter().find(|o| o.slot == Some(slot))
        {
            return Err(format!(
                "`{}` and `{}` both bind at {slot:?}: one slot names one resource, and a \
                 Metal index cannot be derived for either",
                other.name, d.name,
            ));
        }
    }
    Ok(())
}

fn binding(resource: &Reflected<'_>, declarations: &[Declaration]) -> Result<MetalBinding, String> {
    let at: Vec<&Declaration> = declarations
        .iter()
        .filter(|d| d.slot == Some(resource.slot))
        .collect();
    let index = |classes: &str| {
        at.iter()
            .find(|d| classes.contains(d.register.class))
            .ok_or_else(|| missing(resource, classes))
    };
    let (buffer, texture, sampler, count) = match resource.class {
        MetalClass::Buffer => {
            let d = index("btu")?;
            (d.register.index, 0, 0, d.count)
        }
        MetalClass::Texture => {
            let d = index("tu")?;
            (0, d.register.index, 0, d.count)
        }
        MetalClass::Sampler => {
            let d = index("s")?;
            (0, 0, d.register.index, d.count)
        }
    };
    Ok(MetalBinding {
        slot: resource.slot,
        buffer,
        texture,
        sampler,
        count,
    })
}

fn missing(resource: &Reflected<'_>, classes: &str) -> String {
    let wanted: Vec<String> = classes.chars().map(|c| format!("register({c}N)")).collect();
    format!(
        "`{}` binds at {:?} but its source declares no {} for it: the Metal \
         index of an engine shader resource is the number on its register \
         annotation, so every one has to carry one",
        resource.name,
        resource.slot,
        wanted.join(" or "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declarations::{Register, resource_declarations};

    fn declaration(name: &str, slot: Slot, class: char, index: u32) -> Declaration {
        Declaration {
            name: name.to_string(),
            slot: Some(slot),
            register: Register {
                class,
                index,
                space: 0,
            },
            count: None,
            runtime_sized: false,
            metal_argument_buffer: None,
        }
    }

    const PUSH: Slot = Slot::PushConstant;
    const SET0: Slot = Slot::Qualified { set: 0, binding: 0 };
    const SET0_1: Slot = Slot::Qualified { set: 0, binding: 1 };
    const SET1: Slot = Slot::Qualified { set: 1, binding: 0 };

    // The whole policy on one shader: a push constant, a texture and its
    // sampler, and a structured buffer whose `t` register is a Metal buffer
    // index.
    #[test]
    fn the_register_number_is_the_metal_index_in_the_class_the_kind_implies() {
        let declarations = [
            declaration("params", PUSH, 'b', 2),
            declaration("albedo", SET0, 't', 1),
            declaration("albedo_sampler", SET0_1, 's', 3),
            declaration("pool", SET1, 't', 0),
        ];
        let resources = [
            Reflected {
                class: MetalClass::Buffer,
                slot: PUSH,
                name: "params",
            },
            Reflected {
                class: MetalClass::Texture,
                slot: SET0,
                name: "albedo",
            },
            Reflected {
                class: MetalClass::Sampler,
                slot: SET0_1,
                name: "albedo_sampler",
            },
            Reflected {
                class: MetalClass::Buffer,
                slot: SET1,
                name: "pool",
            },
        ];
        let table = metal_bindings(&resources, &declarations).unwrap();
        assert_eq!(table[0].buffer, 2);
        assert_eq!(table[1].texture, 1);
        assert_eq!(table[2].sampler, 3);
        assert_eq!(table[3].buffer, 0);
    }

    // Two `t` declarations at one slot is the shape that resolved by accident:
    // a texture with no `vk::` annotation reflected at the slot a structured
    // buffer held, and took that buffer's register number as its texture index.
    #[test]
    fn two_declarations_at_one_slot_are_refused() {
        let declarations = [
            declaration("objects", SET0, 't', 0),
            declaration("hiz_tex", SET0, 't', 4),
        ];
        let err = metal_bindings(&[], &declarations).unwrap_err();
        assert!(err.contains("objects") && err.contains("hiz_tex"), "{err}");
    }

    // A texture and a sampler at one slot is still two resources at one slot.
    #[test]
    fn a_texture_and_a_sampler_may_not_share_a_slot() {
        let declarations = [
            declaration("albedo", SET0, 't', 0),
            declaration("albedo_sampler", SET0, 's', 0),
        ];
        let err = metal_bindings(&[], &declarations).unwrap_err();
        assert!(
            err.contains("albedo") && err.contains("albedo_sampler"),
            "{err}"
        );
    }

    // The three namespaces are disjoint, which is the load-bearing half of the
    // policy: without it `t0` could not mean a buffer here and a texture there.
    #[test]
    fn a_buffer_and_a_texture_may_share_a_register_number() {
        let declarations = [
            declaration("pool", SET1, 't', 0),
            declaration("albedo", SET0, 't', 0),
        ];
        let resources = [
            Reflected {
                class: MetalClass::Buffer,
                slot: SET1,
                name: "pool",
            },
            Reflected {
                class: MetalClass::Texture,
                slot: SET0,
                name: "albedo",
            },
        ];
        let table = metal_bindings(&resources, &declarations).unwrap();
        assert_eq!(table[0].buffer, 0);
        assert_eq!(table[1].texture, 0);
    }

    #[test]
    fn a_resource_with_no_annotation_at_its_slot_errs_naming_it() {
        let resources = [Reflected {
            class: MetalClass::Buffer,
            slot: SET0,
            name: "view",
        }];
        let err = metal_bindings(&resources, &[]).unwrap_err();
        assert!(err.contains("view"), "{err}");
    }

    // A sampler declared in the wrong class is not quietly taken for a texture.
    #[test]
    fn a_register_of_the_wrong_class_does_not_satisfy_a_resource() {
        let declarations = [declaration("shadow_sampler", SET0, 't', 1)];
        let resources = [Reflected {
            class: MetalClass::Sampler,
            slot: SET0,
            name: "shadow_sampler",
        }];
        assert!(metal_bindings(&resources, &declarations).is_err());
    }

    #[test]
    fn an_array_carries_its_element_count_through() {
        let mut cubes = declaration("probe_cubes", SET0, 't', 4);
        cubes.count = Some(8);
        let resources = [Reflected {
            class: MetalClass::Texture,
            slot: SET0,
            name: "probe_cubes",
        }];
        let table = metal_bindings(&resources, &[cubes]).unwrap();
        assert_eq!((table[0].texture, table[0].count), (4, Some(8)));
    }

    // The set an argument buffer covers is named by the declarations inside it,
    // and the buffer index is the one thing no `register()` can carry.
    #[test]
    fn an_argument_buffer_set_is_read_off_its_members() {
        let found = resource_declarations(
            "[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(11)]] \
             TextureCube<float4> cubes[8] : register(t4, space2);",
        );
        assert_eq!(found[0].metal_argument_buffer, Some(11));
        assert_eq!(
            argument_buffer_sets(&found).unwrap(),
            [ArgumentBufferSet {
                set: 2,
                buffer: 11,
                runtime_sized: false
            }]
        );
    }

    // An unsized member is what marks a set for Metal's device address space;
    // a sized array and a single resource in the same set do not.
    #[test]
    fn an_unsized_member_marks_its_argument_buffer() {
        let found = resource_declarations(
            "[[vk::binding(0, 1)]] [[cn::metal_argument_buffer(7)]] \
             Texture2DArray<float> shadow : register(t0, space1);\
             [[vk::binding(1, 1)]] Texture2D<float4> pool[] : register(t1, space1);\
             [[vk::binding(0, 2)]] [[cn::metal_argument_buffer(10)]] \
             Texture2D<float4> fixed[4] : register(t0, space2);",
        );
        assert_eq!(
            argument_buffer_sets(&found).unwrap(),
            [
                ArgumentBufferSet {
                    set: 1,
                    buffer: 7,
                    runtime_sized: true
                },
                ArgumentBufferSet {
                    set: 2,
                    buffer: 10,
                    runtime_sized: false
                },
            ]
        );
    }

    // Every other declaration reads as no argument buffer at all, so a shader
    // that declares none keeps every set discrete.
    #[test]
    fn a_plain_declaration_names_no_argument_buffer() {
        let found = resource_declarations(
            "[[vk::binding(0, 0)]] Texture2D<float4> src : register(t0);\n\
             [[vk::push_constant]] ConstantBuffer<P> post : register(b0);",
        );
        assert!(argument_buffer_sets(&found).unwrap().is_empty());
    }

    // Two members of one set agreeing is the normal case; disagreeing is a
    // misbind that has to fail the compile rather than pick a winner.
    #[test]
    fn one_set_lands_on_one_buffer_index() {
        let agree = resource_declarations(
            "[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(11)]] \
             TextureCube<float4> cubes[8] : register(t4, space2);\n\
             [[vk::binding(1, 2)]] [[cn::metal_argument_buffer(11)]] \
             Texture2D<float4> extra : register(t5, space2);",
        );
        assert_eq!(argument_buffer_sets(&agree).unwrap().len(), 1);

        let disagree = resource_declarations(
            "[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(11)]] \
             TextureCube<float4> cubes[8] : register(t4, space2);\n\
             [[vk::binding(1, 2)]] [[cn::metal_argument_buffer(12)]] \
             Texture2D<float4> extra : register(t5, space2);",
        );
        let err = argument_buffer_sets(&disagree).unwrap_err();
        assert!(
            err.contains("buffer(11)") && err.contains("buffer(12)"),
            "{err}"
        );
    }

    // A push constant is not a descriptor set, so it cannot name one.
    #[test]
    fn a_push_constant_cannot_declare_an_argument_buffer() {
        let found = resource_declarations(
            "[[vk::push_constant]] [[cn::metal_argument_buffer(3)]] \
             ConstantBuffer<P> params : register(b0);",
        );
        let err = argument_buffer_sets(&found).unwrap_err();
        assert!(err.contains("binds no descriptor set"), "{err}");
    }
}
