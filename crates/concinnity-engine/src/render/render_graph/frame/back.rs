// The back of the frame: the main pass and everything that decorates,
// reflects and resolves the scene after it.

use super::descs::{
    cloud_target_desc, draw_args_desc, froxel_volume_desc, planar_mirrors_desc, scene_color_desc,
};
use super::inputs::FrameGraphInputs;
use super::{FrontEnd, GBufferHandles, HdrTargets, SceneInputs, SceneVersions};
use crate::render::render_graph::types::BufferHandle;
use crate::render::render_graph::{GraphBuilder, PassId, PassKind, ReadStages, TextureHandle};

// Main pass: reads optional shadow_map / spot_shadow_map / far_shadow_map /
// draw_args / ao_output / cluster lists; writes the three HDR targets.
// Captures hdr_resolve_v1 (head of the hdr_resolve RMW chain, the version
// AutoExposure reads when two-pass is off) and hdr_depth_v1 (the depth
// HizEarly reduces under two-pass).
pub(super) fn add_main(b: &mut GraphBuilder, hdr: HdrTargets, front: &FrontEnd) -> SceneVersions {
    let mut main = b.add_pass(PassId::Main, PassKind::Render);
    if let Some(h) = front.scene_inputs.shadow {
        main.read_texture(h);
    }
    if let Some(h) = front.scene_inputs.prop_probe {
        main.read_texture(h);
    }
    if let Some(h) = front.scene_inputs.spot_shadow {
        main.read_texture(h);
    }
    if let Some(h) = front.scene_inputs.far_shadow {
        main.read_texture(h);
    }
    if let Some(h) = front.cull.draw_args {
        main.read_buffer(h);
    }
    if let Some(h) = front.scene_inputs.cluster_lists {
        main.read_buffer(h);
    }
    if let Some(h) = front.scene_inputs.atmosphere_luts {
        main.read_texture(h);
    }
    if let Some(h) = front.prepass.ao_output {
        main.read_texture(h);
    }
    if let Some(h) = front.prepass.ssgi_indirect {
        main.read_texture(h);
    }
    // Main does not sample the mask; the read lands the mark ahead of Main
    // with the other pre-pass consumers, so the compute passes after Main stay
    // ordered against every render pass.
    if let Some(h) = front.prepass.disocclusion {
        main.read_texture(h);
    }
    if let Some(g) = front.cull.grass {
        g.read_into(&mut main);
    }
    if let Some(h) = hdr.color {
        let _ = main.write_texture(h);
    }
    let depth_v1 = main.write_texture(hdr.depth);
    let resolve_v1 = main.write_texture(hdr.resolve);
    SceneVersions {
        resolve: resolve_v1,
        depth: depth_v1,
    }
}

// The two-pass occlusion late pass over Main's output, returning the rebuilt
// pyramid and MainLate's scene versions.
pub(super) fn add_late_occlusion(
    b: &mut GraphBuilder,
    hiz: TextureHandle,
    cull_status: Option<BufferHandle>,
    hdr_color: Option<TextureHandle>,
    main_out: SceneVersions,
) -> (TextureHandle, SceneVersions) {
    // HizEarly (compute): read early depth, write the Hi-Z pyramid.
    // The depth RAW edge pins it after Main; the pyramid write is a WAR
    // against Cull's read of the previous frame's contents.
    let mut hizb = b.add_pass(PassId::HizEarly, PassKind::Compute);
    hizb.read_texture(main_out.depth);
    let hiz_v1 = hizb.write_texture(hiz);

    // CullLate (compute): read the rebuilt pyramid + the early pass status
    // buffer, write a second indirect-draw-args buffer MainLate consumes.
    let draw_args_late = b.import_buffer("draw_args_late", draw_args_desc());
    let mut cull_late = b.add_pass(PassId::CullLate, PassKind::Compute);
    cull_late.read_texture(hiz_v1);
    if let Some(cs) = cull_status {
        cull_late.read_buffer(cs);
    }
    let draw_args_late_v1 = cull_late.write_buffer(draw_args_late);

    // MainLate (render): read the late draw args; RMW hdr_color /
    // hdr_depth / hdr_resolve. The draw_args_late RAW edge pins it after
    // CullLate; the hdr_depth write (WAR vs HizEarly's read) pins it after
    // HizEarly; the hdr_color / hdr_resolve WAW edges pin it after Main.
    let mut main_late = b.add_pass(PassId::MainLate, PassKind::Render);
    main_late.read_buffer(draw_args_late_v1);
    let depth = main_late.write_texture(main_out.depth);
    if let Some(h) = hdr_color {
        let _ = main_late.write_texture(h);
    }
    let resolve = main_late.write_texture(main_out.resolve);
    (hiz_v1, SceneVersions { resolve, depth })
}

// hdr_resolve post-Main RMW chain: Raymarch → Decals → Fog →
// ParticlesDraw, each blend- or opaque-writing on top of the
// previous version. The handle walks forward through `h` so each
// write picks up the latest version, giving the compile pass clean
// WAW edges to derive the chain order. Raymarch slots first so its
// depth+color write is visible to every later post-decoration
// pass; AutoExposure's WAR-read on hdr_resolve_head pins it before
// Raymarch (so SDF brightness doesn't skew exposure for the same
// frame), matching the doc's chosen one-frame-lag trade-off.
//
// Main depth rides the same pattern: Raymarch sphere-traces against it and
// writes the hit depth back, so it bumps the depth version, and every later
// decoration samples that version rather than the one Main left.
pub(super) fn add_decorations(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    head: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> SceneVersions {
    let SceneVersions {
        resolve: mut h,
        depth: mut depth_cur,
    } = head;
    let ssr_gbuffer_v1 = front.prepass.ssr_gbuffer();
    let gbuffer_v1 = front.prepass.gbuffer;
    if inputs.raymarch_enabled {
        let mut rm = b.add_pass(PassId::Raymarch, PassKind::Render);
        rm.read_texture(h);
        h = rm.write_texture(h);
        depth_cur = rm.write_texture(depth_cur);
    }
    // SSGI reads the lit scene (its bounce-radiance source) into its own
    // accumulation, which the next frame's resolve hands Main. Slots right
    // after Raymarch so it can bounce raymarched surfaces too, and before
    // Decals / Fog / Particles, which are not light a surface reflects.
    if inputs.ssgi_enabled {
        let mut ssgi = b.add_pass(PassId::Ssgi, PassKind::Render);
        ssgi.read_texture(h);
        // The trace is against the pre-pass view normal + linear depth, weighs
        // each hit by its reflectance, and the accumulation reprojects through
        // the motion vectors; with no G-buffer there is nothing to trace
        // against and the encoder skips.
        if let Some(g) = ssr_gbuffer_v1 {
            ssgi.read_texture(g);
        }
        if let Some(g) = gbuffer_v1 {
            ssgi.read_texture(g.roughness);
            ssgi.read_texture(g.velocity);
        }
    }
    if inputs.decals_enabled {
        // Projected decals reconstruct each pixel's world position from the
        // scene depth, so the pass samples depth while blend-writing color.
        let mut decals = b.add_pass(PassId::Decals, PassKind::Render);
        decals.read_texture(depth_cur);
        h = decals.write_texture(h);
    }
    if let Some(luts) = front.scene_inputs.atmosphere_luts {
        // The air in front of each surface, under the fog: a volumetric fog is
        // the nearer layer, so it blends over the haze rather than under it.
        let mut aerial = b.add_pass(PassId::AerialPerspective, PassKind::Render);
        aerial.read_texture(luts);
        aerial.read_texture(depth_cur);
        h = aerial.write_texture(h);
    }
    if inputs.fog_enabled {
        // FogFroxel (compute) populates the 3D scatter/transmittance
        // volume the Fog fragment shader samples. The post-write handle
        // (`froxel_v1`) is what Fog reads: that gives the compile pass
        // a clean RAW edge so FogFroxel runs before Fog in the toposort.
        // All three backends implement the froxel path; the Fog render
        // pass trilinear-samples the volume by (screen_uv, view_z).
        let froxel_v0 = b.import_texture("fog_froxel_volume", froxel_volume_desc());
        let mut froxel = b.add_pass(PassId::FogFroxel, PassKind::Compute);
        // Each slab does a cascade tap, so the kernel is a second reader of the
        // shadow map alongside Main. Declaring it puts the compute stage into the
        // read run's union, which is what makes one transition serve both.
        if let Some(h) = front.scene_inputs.shadow {
            froxel.read_texture(h);
        }
        if let Some(h) = front.scene_inputs.far_shadow {
            froxel.read_texture(h);
        }
        let froxel_v1 = froxel.write_texture(froxel_v0);
        let mut fog_pass = b.add_pass(PassId::Fog, PassKind::Render);
        fog_pass.read_texture(froxel_v1);
        // Scene depth bounds the ray march / froxel lookup per pixel.
        fog_pass.read_texture(depth_cur);
        h = fog_pass.write_texture(h);
    }
    if let Some(pools) = front.scene_inputs.particle_pool {
        // The billboard vertex shader pulls each particle out of the pool the
        // sim just wrote, so the read is in the VERTEX stage rather than the
        // FRAGMENT one a render pass otherwise implies. Naming it is what makes
        // the derived transition a vertex-visible one on both explicit backends.
        let mut particles = b.add_pass(PassId::ParticlesDraw, PassKind::Render);
        particles.read_buffer_in_stage(pools, ReadStages::VERTEX);
        // The fragment tests scene depth itself, so opaque geometry hides a
        // sprite behind it.
        particles.read_texture(depth_cur);
        h = particles.write_texture(h);
        *reactive_mask = reactive_mask.map(|m| particles.write_texture(m));
    }
    if inputs.lines_enabled {
        // Last of the hdr_resolve decorations: line geometry draws over the
        // lit + decorated scene, and SSR / TAA then treat it like any other
        // scene content. Samples depth rather than testing against it, so a
        // line behind geometry fades instead of disappearing.
        let mut lines = b.add_pass(PassId::Lines, PassKind::Render);
        lines.read_texture(depth_cur);
        h = lines.write_texture(h);
    }
    SceneVersions {
        resolve: h,
        depth: depth_cur,
    }
}

// The clouds over the pre-TAA scene the see-through surfaces left, so they
// stand in front of the sea as well as the ground: the march reads the
// atmosphere's tables and the depth it stops at, and its target is
// backend-owned, so the import only orders it ahead of the composite. Returns
// the pre-TAA scene's next version.
pub(super) fn add_clouds(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene: SceneVersions,
    front: &FrontEnd,
) -> TextureHandle {
    let Some(luts) = front
        .scene_inputs
        .atmosphere_luts
        .filter(|_| inputs.clouds_enabled)
    else {
        return scene.resolve;
    };
    let target = b.import_texture("cloud_target", cloud_target_desc(inputs));
    let mut march = b.add_pass(PassId::Clouds, PassKind::Compute);
    march.read_texture(luts);
    march.read_texture(scene.depth);
    let target = march.write_texture(target);
    let mut composite = b.add_pass(PassId::CloudComposite, PassKind::Render);
    composite.read_texture(target);
    composite.read_texture(scene.depth);
    composite.write_texture(scene.resolve)
}

// The emitters that draw over the see-through surfaces: from the pools
// `ParticlesDraw` read, onto the pre-TAA scene Transparent left, testing the
// same depth. Returns the pre-TAA scene's next version.
pub(super) fn add_late_particles(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> TextureHandle {
    let Some(pools) = front
        .scene_inputs
        .particle_pool
        .filter(|_| inputs.late_particles_enabled)
    else {
        return scene.resolve;
    };
    let mut late = b.add_pass(PassId::ParticlesLateDraw, PassKind::Render);
    late.read_buffer_in_stage(pools, ReadStages::VERTEX);
    late.read_texture(scene.depth);
    let resolve = late.write_texture(scene.resolve);
    *reactive_mask = reactive_mask.map(|m| late.write_texture(m));
    resolve
}

// Planar mirrors, the reflection resolve, and Transparent, returning the
// pre-TAA scene.
pub(super) fn add_reflections_and_transparent(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    decorated: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> TextureHandle {
    let SceneVersions {
        resolve: hdr_resolve_cur,
        depth: depth_cur,
    } = decorated;
    let SceneInputs {
        shadow: shadow_v1,
        spot_shadow: spot_shadow_v1,
        far_shadow: far_shadow_v1,
        atmosphere_luts,
        cluster_lists,
        water_cascades,
        ..
    } = front.scene_inputs;
    let gbuffer_v1 = front.prepass.gbuffer;
    let ssr_gbuffer_v1 = front.prepass.ssr_gbuffer();

    // Planar mirrors: the scene re-rendered across each reflector plane, lit like
    // Main (so after the shadow maps) from the frame's culled geometry (so after
    // Cull). The mirror targets are encoder-owned, so the import only orders the
    // mirrors ahead of their one reader, Transparent.
    let planar_mirrors = if inputs.planar_reflection_enabled && inputs.transparent_enabled {
        let mirrors = b.import_texture("planar_mirrors", planar_mirrors_desc(inputs));
        let mut planar = b.add_pass(PassId::PlanarReflection, PassKind::Render);
        if let Some(h) = shadow_v1 {
            planar.read_texture(h);
        }
        if let Some(h) = spot_shadow_v1 {
            planar.read_texture(h);
        }
        if let Some(h) = far_shadow_v1 {
            planar.read_texture(h);
        }
        if let Some(h) = front.cull.draw_args {
            planar.read_buffer(h);
        }
        if let Some(h) = atmosphere_luts {
            planar.read_texture(h);
        }
        Some(planar.write_texture(mirrors))
    } else {
        None
    };

    // `scene_pre_taa` is a distinct texture only when a pass writes it:
    // SsrResolve / RtReflections produce it, and Transparent read-modify-writes
    // it. With neither resolve the engine binds the pre-TAA scene name straight
    // to `hdr_resolve`, so declaring it here would put two graph resources on
    // one GPU object -- each with its own barrier timeline over the same memory.
    // Threading the upstream handle expresses that binding with one resource.
    if inputs.rt_reflections_enabled || inputs.ssr_enabled {
        let scene_pre_taa = b.import_texture("scene_pre_taa", scene_color_desc(inputs));
        // SsrResolve and RtReflections occupy the same slot: both read the
        // post-decoration hdr_resolve and write scene_pre_taa. Hardware RT
        // *takes precedence* over SSR: a world can enable both (RT on the
        // backend / GPU that supports it, SSR as the cross-backend fallback),
        // and where RT is live the builder picks it and omits SsrResolve. Only
        // one of the two is ever inserted.
        // Both resolves trace against the pre-pass view normal + linear depth
        // and pick their blur radius from its roughness. Declaring those reads
        // is what keeps the G-buffer's modeled lifetime as long as its real
        // one: roughness has no other consumer, so without this it would look
        // dead the moment the pre-pass finished.
        let mut current = if inputs.rt_reflections_enabled {
            // The classify and trace kernels read the scene and the G-buffer in
            // compute, the composite after them in its fragment.
            let screen = ReadStages::COMPUTE | ReadStages::FRAGMENT;
            let mut rt = b.add_pass(PassId::RtReflections, PassKind::Render);
            rt.read_texture_in_stage(hdr_resolve_cur, screen);
            if let Some(g) = ssr_gbuffer_v1 {
                rt.read_texture_in_stage(g, screen);
            }
            if let Some(g) = gbuffer_v1 {
                rt.read_texture_in_stage(g.roughness, screen);
            }
            // A missed ray's probe fallback reads its cluster's probe mask.
            if let Some(h) = cluster_lists {
                rt.read_buffer_in_stage(h, ReadStages::COMPUTE);
            }
            // A ray that meets the sea reads its waves from the cascades the
            // simulation wrote this frame.
            if let Some(h) = water_cascades {
                rt.read_texture_in_stage(h, ReadStages::COMPUTE);
            }
            rt.write_texture(scene_pre_taa)
        } else {
            let mut ssr = b.add_pass(PassId::SsrResolve, PassKind::Render);
            ssr.read_texture(hdr_resolve_cur);
            if let Some(g) = ssr_gbuffer_v1 {
                ssr.read_texture(g);
            }
            if let Some(g) = gbuffer_v1 {
                ssr.read_texture(g.roughness);
            }
            // A missed ray's probe fallback reads its cluster's probe mask.
            if let Some(h) = cluster_lists {
                ssr.read_buffer(h);
            }
            ssr.write_texture(scene_pre_taa)
        };
        if inputs.transparent_enabled {
            let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
            // Pin Transparent after the whole post-decoration hdr_resolve chain
            // (Main → Decals → Fog → ParticlesDraw), which the scene_pre_taa
            // edge below does not imply.
            trans.read_texture(hdr_resolve_cur);
            // Depth is read at its latest version, not the imported v0:
            // reading v0 would be a WAR against Main's write and pin
            // Transparent *before* Main, closing a cycle.
            trans.read_texture(depth_cur);
            if let Some(h) = cluster_lists {
                trans.read_buffer(h);
            }
            if let Some(h) = planar_mirrors {
                trans.read_texture(h);
            }
            // The water body is lit through the sun's cascaded and far shadows.
            if let Some(h) = shadow_v1 {
                trans.read_texture(h);
            }
            if let Some(h) = far_shadow_v1 {
                trans.read_texture(h);
            }
            if let Some(h) = water_cascades {
                trans.read_texture_in_stage(h, ReadStages::VERTEX | ReadStages::FRAGMENT);
            }
            // RMW the resolve output. The read declares the sample dependency
            // (translucents sample the resolved scene for refraction); the
            // write produces the blended version downstream passes consume.
            trans.read_texture(current);
            current = trans.write_texture(current);
            *reactive_mask = reactive_mask.map(|m| trans.write_texture(m));
        }
        current
    } else if inputs.transparent_enabled {
        // With no reflection resolve the pre-TAA scene *is* `hdr_resolve` and
        // glass blends straight into it, so Transparent extends the hdr_resolve
        // chain by one version instead of branching a second resource onto the
        // same object.
        let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
        trans.read_texture(depth_cur);
        if let Some(h) = cluster_lists {
            trans.read_buffer(h);
        }
        if let Some(h) = planar_mirrors {
            trans.read_texture(h);
        }
        if let Some(h) = shadow_v1 {
            trans.read_texture(h);
        }
        if let Some(h) = far_shadow_v1 {
            trans.read_texture(h);
        }
        if let Some(h) = water_cascades {
            trans.read_texture_in_stage(h, ReadStages::VERTEX | ReadStages::FRAGMENT);
        }
        trans.read_texture(hdr_resolve_cur);
        *reactive_mask = reactive_mask.map(|m| trans.write_texture(m));
        trans.write_texture(hdr_resolve_cur)
    } else {
        hdr_resolve_cur
    }
}

// `scene_color` is the engine-owned output the post-TAA composite stack
// consumes, and -- with TAA on -- the history slot next frame samples, which
// is why it stays imported rather than becoming a transient. Declared only
// when TaaResolve or Upscale writes it, for the same one-object-one-resource
// reason as above: with neither, the engine binds the name to the latest
// pre-TAA scene texture. The two writers are mutually exclusive -- the
// upscaler does its own temporal accumulation, so layering TaaResolve on top
// would double-temporal the scene.
pub(super) fn add_temporal(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene_pre_taa_cur: TextureHandle,
    gbuffer_v1: Option<GBufferHandles>,
    reactive_mask: Option<TextureHandle>,
) -> TextureHandle {
    let velocity_v1 = gbuffer_v1.map(|g| g.velocity);
    if inputs.upscale_enabled {
        let scene_color = b.import_texture("scene_color", scene_color_desc(inputs));
        // Compute, not render: the temporal upscaler is a dispatch, so its reads
        // want the non-pixel shader-resource state. Declaring it as a render
        // pass put the fragment stage in the read-stage union and left the
        // backend flipping the scene and the motion buffer by hand.
        let mut up = b.add_pass(PassId::Upscale, PassKind::Compute);
        up.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // Upscale. The scaler consumes motion vectors directly.
        if let Some(v) = velocity_v1 {
            up.read_texture(v);
        }
        // The scaler also samples the pre-pass depth (single-sample at render
        // resolution), which is the only consumer that target has.
        if let Some(g) = gbuffer_v1 {
            up.read_texture(g.depth);
        }
        if let Some(m) = reactive_mask {
            up.read_texture(m);
        }
        up.write_texture(scene_color)
    } else if inputs.taa_enabled {
        let scene_color = b.import_texture("scene_color", scene_color_desc(inputs));
        let mut taa = b.add_pass(PassId::TaaResolve, PassKind::Render);
        taa.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // TaaResolve. Without this the order rests on declaration order
        // alone.
        if let Some(v) = velocity_v1 {
            taa.read_texture(v);
        }
        if let Some(m) = reactive_mask {
            taa.read_texture(m);
        }
        taa.write_texture(scene_color)
    } else {
        scene_pre_taa_cur
    }
}
