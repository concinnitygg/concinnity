// The space of selection inputs the tests walk: one toggle per input the
// selection reads, applied to a base with everything off or everything on.

use super::PassInputs;
use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::render::render_graph::gates::FrameFacts;

pub(in crate::render::render_graph) type Toggle = fn(&mut PassInputs);

// Every input the selection reads, each flipped by its toggle.
pub(in crate::render::render_graph) const TOGGLES: &[(&str, Toggle)] = &[
    ("taa", |i| i.gates.taa ^= true),
    ("ssao", |i| i.gates.ssao ^= true),
    ("ssr", |i| i.gates.ssr ^= true),
    ("rt_pass", |i| i.gates.rt_pass ^= true),
    ("ssgi", |i| i.gates.ssgi ^= true),
    ("ssgi_resolve", |i| i.gates.ssgi_resolve ^= true),
    ("auto_exposure", |i| i.gates.auto_exposure ^= true),
    ("gbuffer", |i| i.gates.gbuffer ^= true),
    ("reactive_target", |i| i.gates.traits.reactive_mask ^= true),
    ("composite_scene_fixed", |i| {
        i.gates.traits.composite_scene_fixed ^= true;
    }),
    ("shadow", |i| i.facts.shadow ^= true),
    ("far_shadow_map", |i| {
        i.facts.far_shadow_map_size = 1024 - i.facts.far_shadow_map_size;
    }),
    ("far_shadow_drawn", |i| i.facts.far_shadow_drawn ^= true),
    ("spots", |i| {
        i.facts.shadowed_spot_count = 2 - i.facts.shadowed_spot_count;
    }),
    ("world_hidden", |i| i.facts.world_hidden ^= true),
    ("cull_ready", |i| i.facts.cull_ready ^= true),
    ("cull_records", |i| {
        i.facts.cull_records = 10 - i.facts.cull_records;
    }),
    ("two_pass_ready", |i| i.facts.two_pass_ready ^= true),
    ("hiz", |i| i.facts.hiz ^= true),
    ("shadow_receivers", |i| i.facts.shadow_receivers ^= true),
    ("bloom_chain", |i| i.facts.bloom_chain ^= true),
    ("bloom_intensity", |i| {
        i.facts.bloom_intensity = 0.5 - i.facts.bloom_intensity;
    }),
    ("upscaler", |i| i.facts.upscaler ^= true),
    ("upscaler_reads_reactive", |i| {
        i.facts.upscaler_reads_reactive ^= true;
    }),
    ("disocclusion", |i| i.facts.disocclusion ^= true),
    ("bvh_live", |i| i.facts.bvh_live ^= true),
    ("particles", |i| i.facts.particles ^= true),
    ("late_particles", |i| i.facts.late_particles ^= true),
    ("fog", |i| i.facts.fog ^= true),
    ("decals", |i| i.facts.decals ^= true),
    ("visible_surfaces", |i| i.facts.visible_surfaces ^= true),
    ("visible_see_through", |i| {
        i.facts.visible_see_through ^= true;
    }),
    ("rt_transparent_ready", |i| {
        i.facts.rt_transparent_ready ^= true;
    }),
    ("planar_targets", |i| i.facts.planar_targets ^= true),
    ("water_planar_slot", |i| i.facts.water_planar_slot ^= true),
    ("lines", |i| i.facts.lines ^= true),
    ("visible_volumes", |i| i.facts.visible_volumes ^= true),
    ("atmosphere", |i| i.facts.atmosphere ^= true),
    ("clouds", |i| i.facts.clouds ^= true),
    ("water_sim", |i| i.facts.water_sim ^= true),
    ("grass", |i| i.facts.grass ^= true),
    ("grass_casts", |i| i.facts.grass_casts ^= true),
    ("clustered", |i| i.facts.clustered ^= true),
    ("prop_probe", |i| i.facts.prop_probe ^= true),
];

// A sample flips each toggle by one bit of a draw.
const _: () = assert!(TOGGLES.len() <= u64::BITS as usize);

// Nothing on and nothing held, in the lit view with every show flag set.
pub(in crate::render::render_graph) fn nothing() -> PassInputs {
    PassInputs {
        facts: FrameFacts {
            hdr_width: 1920,
            hdr_height: 1080,
            hdr_sample_count: 1,
            shadow_map_size: 2048,
            spot_shadow_slice_size: 512,
            ..FrameFacts::default()
        },
        mode: ViewMode::Lit,
        show: ShowFlags::all(),
        ..PassInputs::default()
    }
}

// Every toggle flipped on from `nothing`.
pub(in crate::render::render_graph) fn everything() -> PassInputs {
    let mut inputs = nothing();
    for (_, toggle) in TOGGLES {
        toggle(&mut inputs);
    }
    inputs.facts.world_hidden = false;
    inputs
}

// The show flag sets a sweep visits: all, none, and each flag alone cleared
// or alone set. Each flag masks its own passes, so these reach every mask.
pub(in crate::render::render_graph) fn show_sets() -> impl Iterator<Item = ShowFlags> {
    let named = ShowFlags::NAMED.into_iter().map(|(flag, _)| flag);
    [ShowFlags::all(), ShowFlags(0)]
        .into_iter()
        .chain(named.clone().map(|flag| ShowFlags::all().toggled(flag)))
        .chain(named)
}

// A small deterministic generator for sampling the whole input space.
pub(in crate::render::render_graph) struct Rng(u64);

impl Rng {
    pub(in crate::render::render_graph) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(in crate::render::render_graph) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    // A sample with each toggle flipped from `nothing` by a coin, in a random
    // view mode under random show flags.
    pub(in crate::render::render_graph) fn sample(&mut self) -> PassInputs {
        let mut inputs = nothing();
        let bits = self.next();
        for (i, (_, toggle)) in TOGGLES.iter().enumerate() {
            if bits >> i & 1 == 1 {
                toggle(&mut inputs);
            }
        }
        let pick = self.next();
        let modes = ViewMode::ALL.len() as u64;
        inputs.mode = ViewMode::ALL[usize::try_from(pick % modes).unwrap_or(0)];
        let shows = 1u32 << ShowFlags::NAMED.len();
        inputs.show = ShowFlags(u32::try_from((pick >> 8) % u64::from(shows)).unwrap_or(0));
        inputs
    }
}
