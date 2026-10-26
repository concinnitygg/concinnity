// Measures the world cook and blob load path: the front half (parse + expand
// + validate), the full in-memory compile, and the blob encode / parse pair
// the shipped load path pays. The fixture is a procedural prop world in the
// shape of the CPU stress worlds: one mesh, one texture, one material, N
// props. No payload cache is anchored, so the mesh and texture payloads
// compile every iteration beside the per-prop work that dominates the cook.
//
//     cargo test -p concinnity-cook --release -- --ignored --nocapture \
//         --test-threads=1 bench_cook

use concinnity_core::bench::{Pace, bench};
use concinnity_engine::blob::{BlobMeta, WorldManifest, encode_cnb, parse_cnb};
use concinnity_engine::platform::Platform;
use std::fmt;

use crate::build_only::prepare_world;
use crate::pipeline::build_pipeline_from_str;

// The allocation column needs live counters, and `#[global_allocator]` is a
// per-program item, so this crate's test binary is the one build that installs it.
concinnity_core::install_global_allocator!();

const SIZES: [(usize, &str); 2] = [(1_000, "1k"), (10_000, "10k")];

fn world_jsonl(props: usize) -> String {
    fmt::from_fn(|out| write_world_jsonl(out, props)).to_string()
}

fn write_world_jsonl(out: &mut impl fmt::Write, props: usize) -> fmt::Result {
    out.write_str(concat!(
        "[\"Camera3D\",{\"$id\":\"cam\",\"position\":[0,2,12]}]\n",
        "[\"GraphicsConfig\",{\"$id\":\"gfx\",",
        "\"frame_pacing\":{\"vsync\":false},\"shadows\":{\"map_size\":0}}]\n",
        "[\"ProceduralMesh\",",
        "{\"$id\":\"bench_mesh\",\"generator\":\"box\",\"half_extents\":[0.4,0.4,0.4]}]\n",
        "[\"Texture\",{\"$id\":\"bench_tex\",\"generator\":\"checker\",\"resolution\":64}]\n",
        "[\"Material\",{\"$id\":\"bench_mat\",\"albedo\":\"bench_tex\",\"roughness\":0.6}]\n",
    ))?;
    for i in 0..props {
        let x = (i % 100) as f32 * 1.2;
        let z = (i / 100) as f32 * 1.2;
        writeln!(
            out,
            "[\"Prop\",{{\"mesh\":\"bench_mesh\",\"material\":\"bench_mat\",\
             \"position\":[{x},0.45,{z}]}}]"
        )?;
    }
    Ok(())
}

fn run(pace: Pace) {
    // A single-run pass only proves the fixtures, so it skips the large size.
    let sizes = match pace {
        Pace::Timed => &SIZES[..],
        Pace::Once => &SIZES[..1],
    };
    for &(n, label) in sizes {
        let content = world_jsonl(n);

        bench(
            pace,
            &format!("cook/prepare_world/{label}"),
            n as u64,
            || {
                let loaded = prepare_world(&content, None).expect("bench world validates");
                loaded.assets.len()
            },
        );

        bench(pace, &format!("cook/build/{label}"), n as u64, || {
            let result = build_pipeline_from_str(&content, None, Platform::Metal.into())
                .expect("bench world compiles");
            result.defs.len()
        });

        // The blob image the compile above would ship: full metadata plus the
        // primary payload section, assembled the way the cook's writer does.
        let result = build_pipeline_from_str(&content, None, Platform::Metal.into())
            .expect("bench world compiles");
        let meta = BlobMeta {
            manifest: WorldManifest::from_records(&result.defs, &result.resources),
            defs: result.defs,
            resources: result.resources,
            scene_groups: result.scene_groups,
            mesh_bounds: result.mesh_bounds,
            physics_budget: result.physics_budget,
        };
        let payload = result
            .payloads
            .first()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let image =
            encode_cnb(concinnity_engine::SCHEMA_VERSION, &meta, payload).expect("blob encodes");

        bench(pace, &format!("cook/blob_encode/{label}"), n as u64, || {
            encode_cnb(concinnity_engine::SCHEMA_VERSION, &meta, payload)
                .expect("blob encodes")
                .len()
        });

        bench(pace, &format!("cook/blob_parse/{label}"), n as u64, || {
            let (meta, payload_start) =
                parse_cnb::<BlobMeta>(concinnity_engine::SCHEMA_VERSION, &image)
                    .expect("blob parses");
            (meta.defs.len(), payload_start)
        });
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_cook() {
    run(Pace::Timed);
}

#[test]
fn cook_fixtures_build_and_run() {
    run(Pace::Once);
}
