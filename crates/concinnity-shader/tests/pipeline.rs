//! The toolchain against a real dxc: one shader of the shape every engine
//! `.hlsl` has -- a push constant, a texture and its sampler, and a
//! vertex/fragment pair -- compiled through each leg.
//!
//! The source is built here rather than read from the shader tree: these check
//! the toolchain, not a shader.

use concinnity_shader::{HlslJob, HlslTarget};

// A texture and a sampler at registers that differ from their bindings, so a
// table that fell back to the SPIR-V binding would land somewhere visible. The
// second varying is one the fragment never reads.
const SOURCE: &str = r#"
struct Uniforms { float width; float height; float _pad0; float _pad1; };
[[vk::push_constant]] ConstantBuffer<Uniforms> uni : register(b1);

[[vk::binding(0, 0)]] Texture2D<float4> atlas : register(t2);
[[vk::binding(1, 0)]] SamplerState atlas_sampler : register(s3);

struct VertexOut
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float3 unread : TEXCOORD1;
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOut probe_vertex(uint vid : SV_VertexID)
{
    VertexOut o;
    o.uv = float2(float(vid), uni.width);
    o.unread = float3(o.uv, 1.0);
    o.position = float4(o.uv, 0.0, uni.height);
    return o;
}

[shader("pixel")]
float4 probe_fragment(VertexOut i) : SV_Target
{
    return atlas.Sample(atlas_sampler, i.uv);
}
"#;

// The same shape plus a resource array on a set of its own, declared as a
// Metal argument buffer, a set no `register()` can name.
const ARGUMENT_BUFFER_SOURCE: &str = r#"
struct Uniforms { float width; float height; float _pad0; float _pad1; };
[[vk::push_constant]] ConstantBuffer<Uniforms> uni : register(b1);

[[vk::binding(0, 0)]] Texture2D<float4> atlas : register(t2);
[[vk::binding(2, 0)]] SamplerState atlas_sampler : register(s3);

[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(11)]]
TextureCube<float4> cubes[4] : register(t4, space2);
[[vk::binding(1, 0)]] SamplerState cube_sampler : register(s4);

[shader("pixel")]
float4 pool_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    float4 c = atlas.Sample(atlas_sampler, uv);
    for (uint i = 0u; i < 4u; i++)
    {
        c += cubes[i].SampleLevel(cube_sampler, float3(uv, uni.width), 0.0);
    }
    return c;
}
"#;

// A three-member argument buffer whose fragment reads only the last member,
// beside a texture, a sampler and a constant buffer nothing reads. The
// registers differ from the bindings, so a member placed by anything but its
// register lands on a visibly wrong id.
#[cfg(feature = "spirv-cross")]
const LATE_MEMBER_SOURCE: &str = r#"
struct Tint { float4 color; };
[[vk::binding(0, 0)]] Texture2D<float4> atlas : register(t2);
[[vk::binding(3, 0)]] SamplerState atlas_sampler : register(s3);
[[vk::binding(1, 0)]] ConstantBuffer<Tint> tint : register(b0);
[[vk::binding(2, 0)]] SamplerState linear_sampler : register(s4);

[[vk::binding(0, 1)]] [[cn::metal_argument_buffer(7)]]
Texture2D<float4> first : register(t4, space1);
[[vk::binding(1, 1)]] Texture2D<float4> middle : register(t5, space1);
[[vk::binding(2, 1)]] Texture2D<float4> last : register(t9, space1);

[shader("vertex")]
float4 late_vertex(float3 p : POSITION) : SV_Position
{
    return float4(p, 1.0);
}

[shader("pixel")]
float4 late_fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target
{
    return last.Sample(linear_sampler, uv);
}
"#;

fn job<'a>(entry: &'a str, target: HlslTarget) -> HlslJob<'a> {
    job_from(SOURCE, entry, target)
}

fn job_from<'a>(source: &'a str, entry: &'a str, target: HlslTarget) -> HlslJob<'a> {
    HlslJob {
        source,
        file_name: "probe.hlsl",
        entry,
        target,
    }
}

fn compile(entry: &str, target: HlslTarget) -> Vec<u8> {
    compile_from(SOURCE, entry, target)
}

fn compile_from(source: &str, entry: &str, target: HlslTarget) -> Vec<u8> {
    let work = concinnity_testing::TempTree::new();
    concinnity_shader::compile(&job_from(source, entry, target), work.path()).expect("compile")
}

// First word of a SPIR-V module, little-endian on every target we build for.
const SPIRV_MAGIC: u32 = 0x0723_0203;

#[test]
fn the_vulkan_leg_emits_a_spirv_module_for_each_stage() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    for entry in ["probe_vertex", "probe_fragment"] {
        let bytes = compile(entry, HlslTarget::Spirv);
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, SPIRV_MAGIC, "{entry}");
    }
}

// The stage comes off the entry point's own attribute, so a table that names
// only an entry point compiles.
#[test]
fn a_profile_is_not_needed_to_compile_either_stage() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    assert_eq!(
        concinnity_shader::stage_of(SOURCE, "probe_vertex"),
        Ok(concinnity_shader::Stage::Vertex)
    );
    assert_eq!(
        concinnity_shader::stage_of(SOURCE, "probe_fragment"),
        Ok(concinnity_shader::Stage::Pixel)
    );
}

#[test]
fn an_entry_point_the_source_does_not_declare_fails_the_compile() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let work = concinnity_testing::TempTree::new();
    let err = concinnity_shader::compile(&job("no_such_entry", HlslTarget::Spirv), work.path())
        .unwrap_err();
    assert!(err.contains("no_such_entry"), "{err}");
}

// The source behind a misspelled `vk::` attribute on a declaration nothing
// reads, which dxc knows as an unknown attribute and drops.
fn misspelled_vk_source() -> String {
    format!("[[vk::bindng(9, 0)]] static const uint unused = 0u;\n{SOURCE}")
}

// A misspelled `vk::` attribute is an unknown attribute on every leg, the DXIL
// one included, so a strict compile fails naming it.
#[test]
fn a_misspelled_vk_attribute_fails_a_strict_compile_on_every_leg() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let source = misspelled_vk_source();
    let work = concinnity_testing::TempTree::new();
    let targets = [
        HlslTarget::Spirv,
        HlslTarget::Dxil {
            shader_model_6_5: false,
        },
    ];
    for target in targets {
        let err =
            concinnity_shader::compile(&job_from(&source, "probe_fragment", target), work.path())
                .unwrap_err();
        assert!(err.contains("'bindng'"), "{target:?}: {err}");
    }
}

// Authored text reports the same warning beside its artifact instead, and a
// clean compile reports none.
#[test]
fn a_reported_warning_rides_beside_the_artifact() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let work = concinnity_testing::TempTree::new();
    let source = misspelled_vk_source();
    let warned = concinnity_shader::compile_with_warnings(
        &job_from(&source, "probe_fragment", HlslTarget::Spirv),
        work.path(),
    )
    .expect("a warning alone does not fail a reported compile");
    let warnings = warned.warnings.expect("the misspelling is reported");
    assert!(warnings.contains("'bindng'"), "{warnings}");

    let clean = concinnity_shader::compile_with_warnings(
        &job("probe_fragment", HlslTarget::Spirv),
        work.path(),
    )
    .expect("clean compile");
    assert_eq!(clean.warnings, None);
    assert_eq!(clean.artifact, compile("probe_fragment", HlslTarget::Spirv));
    assert_eq!(warned.artifact, clean.artifact, "the attribute is dropped");
}

// The engine's own attributes never reach dxc, so a source carrying one
// compiles strictly, and one it does not know fails before dxc runs.
#[test]
fn engine_attributes_compile_strictly_and_an_unknown_one_is_refused() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let spirv = compile_from(ARGUMENT_BUFFER_SOURCE, "pool_fragment", HlslTarget::Spirv);
    assert!(!spirv.is_empty());
    let unknown =
        ARGUMENT_BUFFER_SOURCE.replace("cn::metal_argument_buffer", "cn::argument_buffer");
    let work = concinnity_testing::TempTree::new();
    let err = concinnity_shader::compile(
        &job_from(&unknown, "pool_fragment", HlslTarget::Spirv),
        work.path(),
    )
    .unwrap_err();
    assert!(err.contains("`cn::argument_buffer`"), "{err}");
}

// dxc keeps nothing it was asked to preserve in the entry interface of a
// source that declares a combined image sampler, so none compiles.
#[test]
fn a_combined_image_sampler_is_refused_before_dxc_runs() {
    let combined = SOURCE.replace(
        "[[vk::binding(1, 0)]] SamplerState",
        "[[vk::binding(0, 0)]] [[vk::combinedImageSampler]] SamplerState",
    );
    let work = concinnity_testing::TempTree::new();
    let err = concinnity_shader::compile(
        &job_from(&combined, "probe_fragment", HlslTarget::Spirv),
        work.path(),
    )
    .unwrap_err();
    assert!(err.contains("combinedImageSampler"), "{err}");
}

// The varying the fragment never reads stays in its Vulkan interface, so the
// vertex output at that location has a consumer. The DXIL leg keeps it in the
// signature on its own; the MSL leg links by attribute and may drop it.
#[test]
fn a_vulkan_fragment_keeps_the_varyings_it_never_reads() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let names = |target| {
        let bytes = compile("probe_fragment", target);
        let has = |name: &[u8]| bytes.windows(name.len()).any(|w| w == name);
        (has(b"in.var.TEXCOORD0"), has(b"in.var.TEXCOORD1"))
    };
    assert_eq!(names(HlslTarget::Spirv), (true, true));
    assert_eq!(names(HlslTarget::SpirvWithVulkanLayout), (true, false));
}

#[cfg(feature = "spirv-cross")]
mod msl {
    use super::{HlslTarget, compile, compile_from};

    fn text(entry: &str) -> String {
        String::from_utf8(compile(entry, HlslTarget::Msl)).expect("utf8 MSL")
    }

    fn argument_buffer_text() -> String {
        String::from_utf8(compile_from(
            super::ARGUMENT_BUFFER_SOURCE,
            "pool_fragment",
            HlslTarget::Msl,
        ))
        .expect("utf8 MSL")
    }

    // The whole point of the C-API wrapper: the Metal index is the number on
    // the `register()`, not the SPIR-V binding and not the emitter's own count.
    #[test]
    fn every_resource_lands_on_the_index_its_register_names() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let vertex = text("probe_vertex");
        assert!(vertex.contains("uni [[buffer(1)]]"), "{vertex}");
        let fragment = text("probe_fragment");
        assert!(fragment.contains("atlas [[texture(2)]]"), "{fragment}");
        assert!(
            fragment.contains("atlas_sampler [[sampler(3)]]"),
            "{fragment}"
        );
    }

    // The Metal host looks a function up by name, so the entry point must keep
    // the one the program table asks for.
    #[test]
    fn the_emitted_entry_point_keeps_its_own_name() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        assert!(text("probe_fragment").contains("fragment probe_fragment_out probe_fragment("));
        assert!(text("probe_vertex").contains("vertex probe_vertex_out probe_vertex("));
    }

    // The two stages link by attribute, so both halves have to come out of the
    // same emitter with the same naming.
    #[test]
    fn the_two_stages_agree_on_the_varying_attribute() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        assert!(text("probe_vertex").contains("out_var_TEXCOORD0 [[user(locn0)]]"));
        assert!(text("probe_fragment").contains("in_var_TEXCOORD0 [[user(locn0)]]"));
    }

    // `[[cn::metal_argument_buffer(n)]]` is the only way the source can say
    // that Metal binds a whole set as one buffer, and at which index. dxc
    // ignores it; this is the step that reads it.
    #[test]
    fn a_declared_set_rides_one_argument_buffer_at_the_index_it_names() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let msl = argument_buffer_text();
        assert!(msl.contains("spvDescriptorSet2 [[buffer(11)]]"), "{msl}");
        assert!(
            msl.contains("array<texturecube<float>, 4>"),
            "the array rides the buffer whole: {msl}"
        );
    }

    // The point of naming the other sets discrete: turning argument buffers on
    // is a global switch, and a set left off would be swept in behind the
    // encoders' backs.
    #[test]
    fn the_sets_that_declared_nothing_stay_discrete() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let msl = argument_buffer_text();
        assert!(msl.contains("atlas [[texture(2)]]"), "{msl}");
        assert!(msl.contains("atlas_sampler [[sampler(3)]]"), "{msl}");
        assert!(msl.contains("cube_sampler [[sampler(4)]]"), "{msl}");
        assert!(
            !msl.contains("spvDescriptorSet0"),
            "set 0 was swept in: {msl}"
        );
    }

    fn late_member_text(entry: &str) -> String {
        String::from_utf8(compile_from(
            super::LATE_MEMBER_SOURCE,
            entry,
            HlslTarget::Msl,
        ))
        .expect("utf8 MSL")
    }

    // Metal lays an argument buffer out by the members the function declares,
    // so one that declared only `last` would read the host's buffer at the
    // wrong offset.
    #[test]
    fn an_argument_buffer_declares_the_members_its_entry_never_reads() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let msl = late_member_text("late_fragment");
        for member in ["first [[id(4)]]", "middle [[id(5)]]", "last [[id(9)]]"] {
            assert!(msl.contains(member), "no `{member}`: {msl}");
        }
    }

    // A discrete resource the entry never reads is a parameter no encoder
    // would fill.
    #[test]
    fn an_unread_discrete_resource_stays_off_the_signature() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let msl = late_member_text("late_fragment");
        for unread in ["tint", "atlas"] {
            assert!(!msl.contains(unread), "{unread}: {msl}");
        }
    }

    // An argument buffer the entry reads nothing from is not its interface at
    // all, however many members the fragment beside it declares.
    #[test]
    fn an_unread_argument_buffer_stays_off_the_signature() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let msl = late_member_text("late_vertex");
        assert!(!msl.contains("spvDescriptorSet1"), "{msl}");
    }

    #[test]
    fn the_metal_leg_links_a_library() {
        if !concinnity_shader::dxc_available() || !concinnity_shader::metallib::toolchain_present()
        {
            return;
        }
        let bytes = compile("probe_fragment", HlslTarget::Metallib);
        assert_eq!(&bytes[..4], b"MTLB", "not a metallib");
    }
}
