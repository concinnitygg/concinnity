// Metal pipeline-reflection engine for build-time shader layout validation.
// This is the only file that touches the Metal reflection API; the engine
// binding contract and the comparison live in `shader_layout.rs` (Metal-free,
// unit-tested without a GPU). It reflects a user-authored `.metal` stage's
// engine-provided buffer bindings into the backend-neutral `ReflectedStruct`
// form and compares them against the engine's `#[repr(C)]` layouts.
//
// The cook build pipeline drives this through the thin `ShaderBuildValidator`
// bridge in `concinnity-dev` (the one crate that depends on both cook and this
// backend), so a layout mismatch fails `cn build` with a clear message instead
// of faulting the GPU at run time.
//
// Reflection needs a live pipeline. A vertex/shadow stage reflects through a
// vertex-only pipeline (`rasterizationEnabled = false`, no fragment function); a
// fragment stage is paired with the engine's built-in `vertex_main` so its
// `[[stage_in]]` links. Anything that prevents reflection (no device, a pipeline
// that won't create for an unrelated reason) is reported as an infrastructure
// issue and fails open: only a layout mismatch we actually observed fails the
// build.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBufferBinding, MTLCompileOptions, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLFunction, MTLLibrary, MTLPipelineOption, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineReflection, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use crate::metal::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::shader_layout::{
    EngineStage, ReflectedField, ReflectedStruct, ReflectedStructs, validate_stage,
};

// A no-input fragment used only to make vertex/shadow reflection pipelines
// link. Declaring no `[[stage_in]]` means it imposes no constraint on the
// vertex stage's outputs, so any real vertex/shadow entry pairs with it.
const STUB_FRAGMENT_SRC: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
    fragment float4 __reflect_stub_fragment() { return float4(0.0); }\n";

/// The outcome of reflecting a shader. A `Mismatch` is a real layout error that
/// fails the build; an `Infra` issue is a reflection problem the caller fails
/// open on.
#[derive(Debug)]
pub enum ShaderLayoutIssue {
    /// The shader's layout disagrees with the engine struct.
    Mismatch(String),
    /// Reflection could not run at all.
    Infra(String),
}

/// True when a Metal device is available to reflect against. Lets a caller (and
/// its tests) skip reflection on headless CI without a device rather than fail.
pub fn metal_device_available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// Whether a user `.metal` source defines `entry`, by compiling it and reading
/// the library's function names. Lets the build reject a shader that is missing
/// an entry point its role requires, instead of failing when the pipeline is
/// built (which, for a scene-owned shader, is mid-session).
pub fn metal_source_defines(source: &str, entry: &str) -> Result<bool, ShaderLayoutIssue> {
    objc2::rc::autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| ShaderLayoutIssue::Infra("no Metal device".into()))?;
        let lib = compile_library(&device, source).map_err(|e| {
            ShaderLayoutIssue::Infra(format!("source did not compile for reflection: {e}"))
        })?;
        Ok(function_names(&lib).iter().any(|n| n == entry))
    })
}

/// Reflect a compiled user `.metal` source and validate every engine-provided
/// buffer struct it binds. `kind` is the compile kind (`"vertex"` | `"fragment"`);
/// a `"vertex"` source may carry a main vertex shader, a shadow caster, or both,
/// disambiguated by entry-point name.
pub fn validate_metal_shader_layout(source: &str, kind: &str) -> Result<(), ShaderLayoutIssue> {
    objc2::rc::autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| ShaderLayoutIssue::Infra("no Metal device".into()))?;
        let user_lib = compile_library(&device, source).map_err(|e| {
            ShaderLayoutIssue::Infra(format!("source did not compile for reflection: {e}"))
        })?;
        let names = function_names(&user_lib);

        // Each (stage, entry point) we recognise gets reflected and checked. A
        // source that exposes no engine entry point is skipped (fail open).
        let mut targets: Vec<(EngineStage, &str)> = Vec::new();
        if kind == "fragment" {
            if names.iter().any(|n| n == "fragment_main") {
                targets.push((EngineStage::Fragment, "fragment_main"));
            }
        } else {
            if names.iter().any(|n| n == "vertex_main") {
                targets.push((EngineStage::Vertex, "vertex_main"));
            } else if names.iter().any(|n| n == "vertex_main_instanced") {
                targets.push((EngineStage::Vertex, "vertex_main_instanced"));
            }
            if names.iter().any(|n| n == "shadow_vertex_main") {
                targets.push((EngineStage::Shadow, "shadow_vertex_main"));
            }
        }
        if targets.is_empty() {
            return Err(ShaderLayoutIssue::Infra(format!(
                "no recognised engine entry point for kind '{kind}'"
            )));
        }

        for (stage, entry) in targets {
            let reflected = reflect_stage(&device, &user_lib, entry, stage).map_err(|e| {
                ShaderLayoutIssue::Infra(format!("reflection of '{entry}' failed: {e}"))
            })?;
            validate_stage(stage, &reflected).map_err(ShaderLayoutIssue::Mismatch)?;
        }
        Ok(())
    })
}

// Reflect one stage's engine buffer bindings into `index -> ReflectedStruct`.
fn reflect_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    user_lib: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    stage: EngineStage,
) -> Result<ReflectedStructs, String> {
    let entry_fn = function(user_lib, entry)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&standard_vertex_descriptor()));

    let is_fragment = matches!(stage, EngineStage::Fragment);
    if is_fragment {
        // A fragment pipeline needs a vertex function for its `[[stage_in]]` to
        // link; pair with the engine's built-in `vertex_main`, exactly what the
        // fragment runs against at draw time.
        let builtin_lib = super::pipeline::shader_library(device, false, "main.metal")?;
        let vert_fn = function(&builtin_lib, "vertex_main")?;
        desc.setVertexFunction(Some(&vert_fn));
        desc.setFragmentFunction(Some(&entry_fn));
    } else {
        // A vertex/shadow pipeline needs a fragment to link, but we only care
        // about the vertex bindings. Pair with a trivial stub fragment that has
        // no `[[stage_in]]`: it imposes no constraint on the vertex's outputs,
        // so any real vertex/shadow entry links regardless of what it returns.
        let stub_lib = compile_library(device, STUB_FRAGMENT_SRC)?;
        let stub_fn = function(&stub_lib, "__reflect_stub_fragment")?;
        desc.setVertexFunction(Some(&entry_fn));
        desc.setFragmentFunction(Some(&stub_fn));
    }
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }

    let reflection = create_reflection(device, &desc)?;
    let bindings = if is_fragment {
        reflection.fragmentBindings()
    } else {
        reflection.vertexBindings()
    };
    Ok(bindings_to_map(&bindings))
}

// Create a pipeline with binding reflection enabled and return the reflection.
fn create_reflection(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &MTLRenderPipelineDescriptor,
) -> Result<Retained<MTLRenderPipelineReflection>, String> {
    let mut reflection: Option<Retained<MTLRenderPipelineReflection>> = None;
    device
        .newRenderPipelineStateWithDescriptor_options_reflection_error(
            desc,
            MTLPipelineOption::BindingInfo,
            Some(&mut reflection),
        )
        .map_err(|e| format!("pipeline creation failed: {e:?}"))?;
    reflection.ok_or_else(|| "pipeline returned no reflection".to_string())
}

// Collect the buffer bindings into `index -> ReflectedStruct`, keeping only
// bindings backed by a struct (a `constant X&` struct or a `constant X*`
// pointer to one). Other buffer/texture/sampler bindings are ignored: they are
// either user-owned or outside the engine contract.
fn bindings_to_map(bindings: &NSArray<ProtocolObject<dyn MTLBinding>>) -> ReflectedStructs {
    let mut map = ReflectedStructs::new();
    for binding in bindings.iter() {
        let binding: &ProtocolObject<dyn MTLBinding> = &binding;
        if binding.r#type() != MTLBindingType::Buffer {
            continue;
        }
        // The binding conforms to MTLBufferBinding once it is the Buffer type.
        // ProtocolObject is a transparent wrapper over the same object, so this
        // cast just re-views it through the buffer sub-protocol.
        // SAFETY: the type check above proved this binding is the Buffer variant, so it conforms to
        // MTLBufferBinding; `ProtocolObject` is a transparent wrapper over the same object, so the
        // cast only re-views it through the sub-protocol.
        let buf: &ProtocolObject<dyn MTLBufferBinding> = unsafe {
            &*(binding as *const ProtocolObject<dyn MTLBinding>
                as *const ProtocolObject<dyn MTLBufferBinding>)
        };

        // A struct binding carries its layout directly; a pointer/array binding
        // (e.g. `constant GpuObjectData*`) carries it on the pointee, whose
        // dataSize is the per-element stride.
        let (struct_ty, size) = if let Some(st) = buf.bufferStructType() {
            (Some(st), buf.bufferDataSize())
        } else if let Some(ptr) = buf.bufferPointerType() {
            (ptr.elementStructType(), ptr.dataSize())
        } else {
            (None, 0)
        };
        let Some(st) = struct_ty else {
            continue;
        };

        let fields = st
            .members()
            .iter()
            .map(|m| ReflectedField {
                name: m.name().to_string(),
                offset: m.offset(),
            })
            .collect();
        map.insert(
            binding.index() as u32,
            ReflectedStruct {
                name: binding.name().to_string(),
                size,
                fields,
            },
        );
    }
    map
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let options = MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| format!("{e:?}"))
}

fn function(
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
    lib.newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("entry point '{name}' not found"))
}

fn function_names(lib: &ProtocolObject<dyn MTLLibrary>) -> Vec<String> {
    lib.functionNames().iter().map(|n| n.to_string()).collect()
}

// The engine's standard five-attribute mesh vertex descriptor (the `Vertex`
// layout), with the vertex stream at buffer index 1 so it does not collide with
// the engine's `ViewUniforms` at buffer 0. Required for the `[[stage_in]]` of
// the vertex/shadow stages to link during reflection.
fn standard_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    const STREAM: usize = 1;
    vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: STREAM,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: STREAM,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 24,
                buffer_index: STREAM,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: STREAM,
            },
            VertexAttr {
                index: 4,
                format: MTLVertexFormat::Float2,
                offset: 48,
                buffer_index: STREAM,
            },
        ],
        &[VertexLayout {
            buffer_index: STREAM,
            stride: std::mem::size_of::<crate::gfx::mesh_payload::Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // A correct user vertex shader: declares ViewUniforms exactly as the engine
    // does (packed_float3 cam_pos) and binds it at buffer(0).
    const GOOD_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            packed_float3 cam_pos;
            float prefilter_mip_count;
            float shade_mode;
            float _end_pad;
            float4 sky_rot[3];
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + float3(view.cam_pos) * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // The same shader but with `float3 cam_pos` (16-byte aligned, size 16)
    // instead of packed_float3: exactly the float3-vs-[f32;3] class of bug. It
    // grows the struct's stride past the engine's 208 bytes, so the size check
    // catches it (the `RtGeomEntry` failure mode).
    const BAD_SIZE_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            float3 cam_pos;
            float prefilter_mip_count;
            float shade_mode;
            float _end_pad;
            float4 sky_rot[3];
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + view.cam_pos * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // `vp` and `view` swapped: the total size is unchanged (two float4x4 + the
    // tail), but every named field lands at the wrong offset: exercises the
    // field-offset check rather than the size check.
    const BAD_OFFSET_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 view;
            float4x4 vp;
            float elapsed;
            float _pad;
            packed_float3 cam_pos;
            float prefilter_mip_count;
            float shade_mode;
            float _end_pad;
            float4 sky_rot[3];
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + float3(view.cam_pos) * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    #[test]
    fn faithful_view_uniforms_validate() {
        if !metal_device_available() {
            return;
        }
        assert!(
            matches!(validate_metal_shader_layout(GOOD_VERTEX, "vertex"), Ok(())),
            "a faithful ViewUniforms copy must validate"
        );
    }

    #[test]
    fn wrong_struct_size_is_rejected() {
        if !metal_device_available() {
            return;
        }
        // The float3-vs-packed bug grows the struct stride; caught by the size
        // check (MSL `float3` is 16 bytes, pushing ViewUniforms past 208).
        match validate_metal_shader_layout(BAD_SIZE_VERTEX, "vertex") {
            Err(ShaderLayoutIssue::Mismatch(msg)) => {
                assert!(
                    msg.contains("ViewUniforms"),
                    "names the engine struct: {msg}"
                );
                assert!(
                    msg.contains("bytes") && msg.contains("stride"),
                    "reports the size: {msg}"
                );
            }
            other => panic!("expected a layout mismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_field_offset_is_rejected() {
        if !metal_device_available() {
            return;
        }
        // Swapped fields keep the size but move every offset.
        match validate_metal_shader_layout(BAD_OFFSET_VERTEX, "vertex") {
            Err(ShaderLayoutIssue::Mismatch(msg)) => {
                assert!(msg.contains("offset"), "reports the offset: {msg}");
                assert!(
                    msg.contains("vp") || msg.contains("view"),
                    "names a shifted field: {msg}"
                );
            }
            other => panic!("expected a layout mismatch, got {other:?}"),
        }
    }
}
