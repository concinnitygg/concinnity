// The front of the frame: the cull and grass, the pre-passes, and the
// inputs the scene's main pass reads.

use super::descs::{
    FAR_SHADOW_MAP_RESTS, ao_output_desc, atmosphere_luts_desc, cluster_lists_desc,
    cull_status_desc, draw_args_desc, far_shadow_map_desc, gbuffer_depth_desc,
    gbuffer_normal_depth_desc, gbuffer_roughness_desc, grass_bend_desc, grass_blades_desc,
    particle_pool_desc, prop_probe_desc, shadow_map_desc, spot_shadow_map_desc, ssgi_indirect_desc,
    velocity_desc, water_cascades_desc,
};
use super::inputs::FrameGraphInputs;
use super::{CullOutputs, GBufferHandles, GrassHandles, PrepassOutputs, SceneInputs};
use crate::render::post::ssgi;
use crate::render::render_graph::{GraphBuilder, PassId, PassKind, TextureHandle};

// Cull writes the indirect args Main executes and, under two-pass occlusion,
// the per-object status `CullLate` re-tests.
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

    // Grass places, culls and appends this frame's visible blades and fills the
    // draw arguments, testing tiles against last frame's pyramid (a read that
    // gives the pyramid rebuilds a WAR edge). GrassBend runs first, relaxing the
    // persistent bend field from the half last frame wrote into the other.
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
        // The pre-pass reuses the main cull's indirect buffer, so reading
        // `draw_args` orders it after Cull.
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

    // SsaoBlur is the single graph node for the whole SSAO bundle; SsaoDepth
    // and SsaoKernel only keep timing slots.
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
        // The cascades draw from the cull's output, so reading `draw_args` orders
        // them after Cull, across queues too.
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

    // The atmosphere's tables, carried across frames; the import orders the pass
    // ahead of Main, the mirrors and the haze.
    let atmosphere_luts = inputs.atmosphere_enabled.then(|| {
        let luts = b.import_texture("atmosphere_luts", atmosphere_luts_desc());
        b.add_pass(PassId::Atmosphere, PassKind::Compute)
            .write_texture(luts)
    });

    // Clustered binning of local lights and probes; Main's read orders it first.
    let cluster_lists = if inputs.clustering_enabled {
        let lists = b.import_buffer("cluster_lists", cluster_lists_desc());
        Some(
            b.add_pass(PassId::LightCull, PassKind::Compute)
                .write_buffer(lists),
        )
    } else {
        None
    };

    // Particle simulation: integrate every live emitter's persistent pool in
    // place. It reads no depth or scene, so its only successor is ParticlesDraw;
    // it is declared early so the async queue does not inherit a later pass's
    // wait.
    let particle_pool_v1 = if inputs.particles_enabled {
        let pools = b.import_buffer("particle_pool", particle_pool_desc());
        let mut sim = b.add_pass(PassId::ParticlesSim, PassKind::Compute);
        sim.read_buffer(pools);
        Some(sim.write_buffer(pools))
    } else {
        None
    };

    // The ocean simulation, read only by the water draw, declared early like
    // the particle simulation.
    let water_cascades = (inputs.water_sim_enabled && inputs.transparent_enabled).then(|| {
        let cascades = b.import_texture("water_cascades", water_cascades_desc());
        b.add_pass(PassId::WaterSim, PassKind::Compute)
            .write_texture(cascades)
    });

    // Spot shadows: one depth-only render per shadowed spot, before Main and
    // after Cull.
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

    // The far shadow: the large casters around the camera into their own map.
    // It rests readable, so a frame that keeps it needs no pass.
    let far_shadow_v1 = (inputs.far_shadow_map_size > 0).then(|| {
        let far_map = b.import_readable_texture(
            "far_shadow_map",
            far_shadow_map_desc(inputs.far_shadow_map_size),
            FAR_SHADOW_MAP_RESTS,
        );
        if !inputs.far_shadow_drawn {
            return far_map;
        }
        let mut far = b.add_pass(PassId::FarShadow, PassKind::Render);
        if let Some(h) = cull.draw_args {
            far.read_buffer(h);
        }
        far.write_texture(far_map)
    });

    // The prop probe: one cube face lit like Main (after the shadows) from the
    // frame's records (after Cull), and on the last face the convolution.
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
