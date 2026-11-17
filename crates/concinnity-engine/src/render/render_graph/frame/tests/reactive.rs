// The reactive mask, from its writers to the temporal pass that reads it.

use super::*;

// The passes that read or write `label`, in compiled order, with whether
// each writes it.
fn touches(g: &CompiledGraph, label: &str) -> Vec<(PassId, bool)> {
    let Some(index) = g.resources.iter().position(|r| r.label == label) else {
        return Vec::new();
    };
    g.passes
        .iter()
        .filter_map(|p| {
            let writes = p.writes.iter().any(|w| w.resource_index() == index);
            let reads = p.reads.iter().any(|r| r.resource_index() == index);
            (writes || reads).then_some((p.id, writes))
        })
        .collect()
}

fn reactive_writers_and_taa() -> FrameGraphInputs {
    let mut i = all_off();
    i.reactive_mask_enabled = true;
    i.particles_enabled = true;
    i.transparent_enabled = true;
    i.taa_enabled = true;
    i.velocity_enabled = true;
    i
}

#[test]
fn the_reactive_mask_runs_from_its_writers_to_taa() {
    let g = build_frame_graph(&reactive_writers_and_taa()).expect("compiles");
    assert_eq!(
        touches(&g, "reactive_mask"),
        [
            (PassId::ParticlesDraw, true),
            (PassId::Transparent, true),
            (PassId::TaaResolve, false),
        ]
    );
}

#[test]
fn the_upscaler_reads_the_reactive_mask_in_place_of_taa() {
    let mut i = reactive_writers_and_taa();
    i.particles_enabled = false;
    i.upscale_enabled = true;
    i.upscale_reads_reactive = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert_eq!(
        touches(&g, "reactive_mask"),
        [(PassId::Transparent, true), (PassId::Upscale, false)]
    );
}

// An upscaler with no reactive input leaves the mask out of the frame.
#[test]
fn an_upscaler_that_takes_no_mask_declares_none() {
    let mut i = reactive_writers_and_taa();
    i.upscale_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(touches(&g, "reactive_mask").is_empty());
}

// A frame with no reader, or no writer, declares no mask at all, so it
// costs nothing there.
#[test]
fn the_reactive_mask_needs_a_writer_and_a_reader() {
    let mut no_reader = reactive_writers_and_taa();
    no_reader.taa_enabled = false;
    let mut no_writer = reactive_writers_and_taa();
    no_writer.particles_enabled = false;
    no_writer.transparent_enabled = false;
    let mut not_built = reactive_writers_and_taa();
    not_built.reactive_mask_enabled = false;
    for i in [no_reader, no_writer, not_built] {
        assert!(!i.reactive_mask_live());
        let g = build_frame_graph(&i).expect("compiles");
        assert!(touches(&g, "reactive_mask").is_empty());
    }
    assert!(reactive_writers_and_taa().reactive_mask_live());
}

// The reactive view reads the mask itself, so it needs no temporal pass.
#[test]
fn the_reactive_view_reads_the_mask_without_a_temporal_pass() {
    let mut i = reactive_writers_and_taa();
    i.taa_enabled = false;
    i.velocity_enabled = false;
    i.composite_reads_reactive = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert_eq!(
        touches(&g, "reactive_mask"),
        [
            (PassId::ParticlesDraw, true),
            (PassId::Transparent, true),
            (PassId::Composite, false),
        ]
    );
}
