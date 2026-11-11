// The front of the frame: the cull and grass, the pre-passes, and the
// inputs the scene's main pass reads.

use super::descs::{
    ao_output_desc, atmosphere_luts_desc, cluster_lists_desc, cull_status_desc, draw_args_desc,
    far_shadow_map_desc, gbuffer_depth_desc, gbuffer_normal_depth_desc, gbuffer_roughness_desc,
    grass_bend_desc, grass_blades_desc, particle_pool_desc, prop_probe_desc, shadow_map_desc,
    spot_shadow_map_desc, ssgi_indirect_desc, velocity_desc, water_cascades_desc,
};
use super::inputs::FrameGraphInputs;
use super::{CullOutputs, GBufferHandles, GrassHandles, PrepassOutputs, SceneInputs};
use crate::render::post::ssgi;
use crate::render::render_graph::{GraphBuilder, PassId, PassKind, TextureHandle};

// Cull (compute) writes the indirect-draw args buffer Main consumes
// through executeCommandsInBuffer. Under two-pass occlusion it also
// writes a per-object status buffer (drawn / hi-z-candidate / culled)
// that `CullLate` reads to decide which early-occluded objects to
// re-test against the rebuilt pyramid.
pub(super) fn add_cull_and_grass(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    two_pass: bool,
    hiz_pyramid: Option<TextureHandle>,
) -> CullOutputs {
    let (draw_args_v1, cull_status_v1) = if inputs.bindless_cull_enabled {
        // Import both buffers up front: a live `PassBuilder` holds `&mut b`,
        // so the resource declarations have to happen before `add_pass`.
        let draw_args = b.import_buffer("draw_args", draw_args_desc());
        let cull_status = if two_pass {
            Some(b.import_buffer("cull_status", cull_status_desc()))
        } else {
            None
        };
        let mut cull = b.add_pass(PassId::Cull, PassKind::Compute);
        // The previous frame's pyramid is this cull's occlusion test. Declaring
        // the read is what gives the terminal rebuild a WAR edge to wait on.
        if let Some(h) = hiz_pyramid {
            cull.read_texture(h);
        }
        let da = cull.write_buffer(draw_args);
        let cs = cull_status.map(|h| cull.write_buffer(h));
        (Some(da), cs)
    } else {
        (None, None)
    };

    // Grass (compute): places, culls and appends this frame's visible blades,
    // and fills the indirect arguments the pre-pass and Main draw them with.
    // Its tiles are occlusion-tested against the pyramid the previous frame
    // left, a read that, like Cull's, gives the pyramid's rebuilds a WAR edge to
    // wait on. It reads nothing else, so the schedule may overlap it with the
    // raster front. Both buffers are persistent and backend-owned, hence
    // imported.
    //
    // GrassBend (compute) runs first: it relaxes the camera-centered bend field
    // and stamps this frame's characters and bodies into it, reading the half
    // the last frame wrote and writing the other, so the one persistent buffer
    // is both its read and its write. Both kernels that place blades read it.
    let grass_bend_v1 = inputs.grass_enabled.then(|| {
        let field = b.import_buffer("grass_bend", grass_bend_desc());
        let mut bend = b.add_pass(PassId::GrassBend, PassKind::Compute);
        bend.read_buffer(field);
        bend.write_buffer(field)
    });
    let grass_v1 = if let Some(bend) = grass_bend_v1 {
        let blades = b.import_buffer("grass_blades", grass_blades_desc());
        let args = b.import_buffer("grass_args", draw_args_desc());
        let mut grass = b.add_pass(PassId::Grass, PassKind::Compute);
        if let Some(h) = hiz_pyramid {
            grass.read_texture(h);
        }
        grass.read_buffer(bend);
        Some(GrassHandles {
            blades: grass.write_buffer(blades),
            args: grass.write_buffer(args),
        })
    } else {
        None
    };

    // GrassShadow (compute): the same kernel placing the blades the nearest
    // cascade sees, into their own buffers, which the Shadow pass draws.
    let grass_shadow_v1 = match grass_bend_v1 {
        Some(bend) if inputs.shadow_enabled && inputs.grass_shadow_enabled => {
            let blades = b.import_buffer("grass_shadow_blades", grass_blades_desc());
            let args = b.import_buffer("grass_shadow_args", draw_args_desc());
            let mut shadow = b.add_pass(PassId::GrassShadow, PassKind::Compute);
            shadow.read_buffer(bend);
            Some(GrassHandles {
                blades: shadow.write_buffer(blades),
                args: shadow.write_buffer(args),
            })
        }
        _ => None,
    };

    CullOutputs {
        draw_args: draw_args_v1,
        cull_status: cull_status_v1,
        grass: grass_v1,
        grass_shadow: grass_shadow_v1,
    }
}

// The G-buffer pre-pass and the screen-space passes that read only it: SSAO and
// SSGI's resolve.
pub(super) fn add_prepasses(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    cull: &CullOutputs,
    reactive_mask: &mut Option<TextureHandle>,
) -> PrepassOutputs {
    // Geometry pre-pass: one node writes the view-space normal+depth /
    // roughness / velocity / depth that SSR, SSAO, SSGI, RT, TAA, and the
    // upscaler read. Runs when the backend built the targets and any of those
    // consumers is on.
    let gbuffer_v1 = if inputs.gbuffer_live() {
        let normal_depth =
            b.create_texture("gbuffer_normal_depth", gbuffer_normal_depth_desc(inputs));
        let roughness = b.create_texture("gbuffer_roughness", gbuffer_roughness_desc(inputs));
        let velocity = b.create_texture("gbuffer_velocity", velocity_desc(inputs));
        let depth = b.create_texture("gbuffer_depth", gbuffer_depth_desc(inputs));
        let mut gb = b.add_pass(PassId::GBufferPrepass, PassKind::Render);
        // When the GPU-driven cull path is active the pre-pass reuses the main
        // pass's per-frame indirect command buffer (camera frustum, same cull
        // output), so it must run after Cull. Reading the cull-produced draw_args
        // buffer pins that ordering in the toposort (a no-op when bindless cull is
        // off, where draw_args_v1 is None). Mirrors the Main pass's edge.
        if let Some(h) = cull.draw_args {
            gb.read_buffer(h);
        }
        if let Some(g) = cull.grass {
            g.read_into(&mut gb);
        }
        // One draw writes all four attachments; they are separate resources
        // because their shapes and their consumers differ.
        Some(GBufferHandles {
            normal_depth: gb.write_texture(normal_depth),
            roughness: gb.write_texture(roughness),
            velocity: gb.write_texture(velocity),
            depth: gb.write_texture(depth),
        })
    } else {
        None
    };

    let ssr_gbuffer_v1 = gbuffer_v1.map(|g| g.normal_depth);

    // Disocclusion: the reactive mask's first writer, marking where this
    // frame's G-buffer uncovers a surface last frame's did not show.
    let disocclusion_v1 = match (gbuffer_v1, *reactive_mask) {
        (Some(g), Some(mask)) if inputs.disocclusion_live() => {
            let mut disocclusion = b.add_pass(PassId::Disocclusion, PassKind::Render);
            disocclusion.read_texture(g.normal_depth);
            disocclusion.read_texture(g.velocity);
            Some(disocclusion.write_texture(mask))
        }
        _ => None,
    };
    if disocclusion_v1.is_some() {
        *reactive_mask = disocclusion_v1;
    }

    // SSAO bundle writes ao_output. PassId::SsaoBlur is the single
    // graph node for the entire encode_ssao bundle; SsaoDepth +
    // SsaoKernel keep their per-pass timing slots via inline
    // `pass_timing.attach_render` calls inside encode_ssao but they're
    // not graph nodes (the executor rejects them if mis-added).
    let ao_output_v1 = if inputs.ssao_enabled {
        let ao_output = b.create_texture("ao_output", ao_output_desc(inputs));
        let mut ssao = b.add_pass(PassId::SsaoBlur, PassKind::Render);
        if let Some(h) = ssr_gbuffer_v1 {
            ssao.read_texture(h);
        }
        Some(ssao.write_texture(ao_output))
    } else {
        None
    };

    // SSGI's resolve: last frame's accumulation reprojected onto this frame's
    // G-buffer, folded with the occlusion into the target Main samples for its
    // indirect light. Backend-owned and carried across frames, like the
    // accumulation it reads.
    let ssgi_indirect_v1 = inputs.ssgi_resolve_enabled.then(|| {
        let indirect = b.import_texture(ssgi::INDIRECT_LABEL, ssgi_indirect_desc(inputs));
        let mut resolve = b.add_pass(PassId::SsgiResolve, PassKind::Render);
        if let Some(g) = gbuffer_v1 {
            resolve.read_texture(g.normal_depth);
            resolve.read_texture(g.velocity);
        }
        if let Some(h) = ao_output_v1 {
            resolve.read_texture(h);
        }
        resolve.write_texture(indirect)
    });

    PrepassOutputs {
        gbuffer: gbuffer_v1,
        ao_output: ao_output_v1,
        ssgi_indirect: ssgi_indirect_v1,
        disocclusion: disocclusion_v1,
    }
}

// The shadow maps and the compute passes ahead of Main: the atmosphere tables,
// clustered binning, and the particle and ocean simulations.
pub(super) fn add_scene_inputs(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    cull: &CullOutputs,
) -> SceneInputs {
    // Shadow optionally precedes Main and produces the shadow_map
    // handle Main samples. When off, Main does not declare a shadow_map
    // read, mirroring the encoder's `enable_shadows` shader path.
    let shadow_v1 = if inputs.shadow_enabled {
        let shadow_map = b.import_texture("shadow_map", shadow_map_desc(inputs.shadow_map_size));
        let mut shadow = b.add_pass(PassId::Shadow, PassKind::Render);
        // The GPU-driven cascade path draws from a per-cascade indirect command
        // buffer the cull pass fills from the same cull output, so it must run
        // after Cull. Reading the cull-produced draw_args buffer pins that
        // ordering in the toposort, the way GBufferPrepass and Main do, and on a
        // backend that submits the two queues separately it is what puts a
        // cross-queue wait between an async cull and this pass. A no-op when
        // bindless cull is off, where the cascades draw per-object.
        if let Some(h) = cull.draw_args {
            shadow.read_buffer(h);
        }
        if let Some(g) = cull.grass_shadow {
            g.read_into(&mut shadow);
        }
        Some(shadow.write_texture(shadow_map))
    } else {
        None
    };

    // Clustered binning (compute): bins the scene's local lights and reflection
    // probes into per-cluster light lists and probe masks. Writes the imported
    // cluster buffer; Main's read below pins LightCull before Main in the
    // toposort. Backend-owned buffer, so the import is a dependency-tracking
    // stub.
    // The atmosphere's tables (compute). Backend-owned and carried across
    // frames, so the import only orders the pass ahead of its readers: Main and
    // the mirrors (the sky and the ambient light) and the haze pass.
    let atmosphere_luts = inputs.atmosphere_enabled.then(|| {
        let luts = b.import_texture("atmosphere_luts", atmosphere_luts_desc());
        b.add_pass(PassId::Atmosphere, PassKind::Compute)
            .write_texture(luts)
    });

    let cluster_lists = if inputs.clustering_enabled {
        let lists = b.import_buffer("cluster_lists", cluster_lists_desc());
        Some(
            b.add_pass(PassId::LightCull, PassKind::Compute)
                .write_buffer(lists),
        )
    } else {
        None
    };

    // Particle simulation (compute): ages, integrates and respawns every live
    // emitter's pool in place. One label covers the whole per-emitter set; each
    // backend's registry resolves it to the live emitters' pool buffers, the way
    // `draw_args` resolves to this frame's slot of a ring.
    //
    // The kernel reads no depth and no scene texture -- its inputs are the pools
    // and the frame's params -- so its only
    // successor is ParticlesDraw and it is concurrent with the whole raster
    // front. Declared here rather than beside the draw so the compiled order puts
    // it ahead of the passes it can overlap: the async queue runs its own passes
    // in order, and a position after a compute pass that waits on Shadow would
    // inherit that wait.
    //
    // The pools are persistent (one per emitter, not one per frame in flight) and
    // each frame integrates them in place, reading what the last frame wrote,
    // which is why they are imported rather than created.
    let particle_pool_v1 = if inputs.particles_enabled {
        let pools = b.import_buffer("particle_pool", particle_pool_desc());
        let mut sim = b.add_pass(PassId::ParticlesSim, PassKind::Compute);
        sim.read_buffer(pools);
        Some(sim.write_buffer(pools))
    } else {
        None
    };

    // The simulated ocean (compute): evolves and transforms every simulated
    // sea into its cascade arrays, which only the water draw reads. Like the
    // particle simulation it depends on nothing else in the frame, so it is
    // declared ahead of the raster front it can overlap. The arrays are
    // backend-owned and rewritten whole each frame.
    let water_cascades = (inputs.water_sim_enabled && inputs.transparent_enabled).then(|| {
        let cascades = b.import_texture("water_cascades", water_cascades_desc());
        b.add_pass(PassId::WaterSim, PassKind::Compute)
            .write_texture(cascades)
    });

    // Spot shadows: one depth-only render per shadowed spot into its slice of
    // the spot shadow array. Like the cascade pass it precedes Main, which
    // samples the array; backend-owned, so the import tracks dependencies only.
    // Its slices draw from the same cull output the cascades do, so it reads
    // draw_args for the same edge after Cull.
    let spot_shadow_v1 = if inputs.shadowed_spot_count > 0 {
        let spot_map = b.import_texture(
            "spot_shadow_map",
            spot_shadow_map_desc(inputs.spot_shadow_slice_size, inputs.shadowed_spot_count),
        );
        let mut spot = b.add_pass(PassId::SpotShadow, PassKind::Render);
        if let Some(h) = cull.draw_args {
            spot.read_buffer(h);
        }
        Some(spot.write_texture(spot_map))
    } else {
        None
    };

    // The far shadow: one depth-only render of the large casters around the
    // camera into its own map, which the shaded passes sample past the
    // cascades. Backend-owned and kept across frames, so the import tracks
    // dependencies only; it draws from the same cull output as the cascades.
    let far_shadow_v1 = (inputs.far_shadow_map_size > 0).then(|| {
        let far_map = b.import_texture(
            "far_shadow_map",
            far_shadow_map_desc(inputs.far_shadow_map_size),
        );
        let mut far = b.add_pass(PassId::FarShadow, PassKind::Render);
        if let Some(h) = cull.draw_args {
            far.read_buffer(h);
        }
        far.write_texture(far_map)
    });

    // The prop reflection probe: one cube face of the scene seen from the
    // prop, lit like Main (so after the shadow maps) from the frame's records
    // (so after Cull), and on the cycle's last face the convolution into the
    // probe's cube. The cube is encoder-owned, so the import only orders the
    // pass ahead of Main, the first pass reading the probe set.
    let prop_probe = inputs.prop_probe_enabled.then(|| {
        let cube = b.import_texture("prop_probe", prop_probe_desc());
        let mut capture = b.add_pass(PassId::PropProbe, PassKind::Render);
        if let Some(h) = shadow_v1 {
            capture.read_texture(h);
        }
        if let Some(h) = spot_shadow_v1 {
            capture.read_texture(h);
        }
        if let Some(h) = far_shadow_v1 {
            capture.read_texture(h);
        }
        if let Some(h) = cull.draw_args {
            capture.read_buffer(h);
        }
        if let Some(h) = atmosphere_luts {
            capture.read_texture(h);
        }
        capture.write_texture(cube)
    });

    SceneInputs {
        shadow: shadow_v1,
        spot_shadow: spot_shadow_v1,
        far_shadow: far_shadow_v1,
        prop_probe,
        atmosphere_luts,
        cluster_lists,
        particle_pool: particle_pool_v1,
        water_cascades,
    }
}
