//! Benchmarks over the world cook and blob load path: the front half
//! (parse + expand + validate), the full in-memory compile, and the blob
//! encode / parse pair the shipped load path pays. The fixture is a
//! procedural prop world in the shape of the CPU stress worlds: one mesh,
//! one texture, one material, N props.
//!
//! The compile consults the on-disk payload cache for its two compiled
//! payloads (the mesh and the texture), so calibration warms it and the
//! measured passes see the steady state a rebuild sees. The per-prop work
//! that dominates the cook (def creation, arg reserialization, packing) is
//! never cached and runs whole every iteration. The cache lives under a
//! state root in target/ so bench runs never touch a real project's state.
//!
//! Run with `cargo bench -p concinnity-bench --bench cook`.

use concinnity_cook::{build_pipeline_from_str, prepare_world};
use concinnity_core::blob::{BlobMeta, WorldManifest, encode_cnb, parse_cnb};

use concinnity_bench::Bench;

const SIZES: [(usize, &str); 2] = [(1_000, "1k"), (10_000, "10k")];

fn world_jsonl(props: usize) -> String {
    let mut out = String::with_capacity(props * 128 + 512);
    out.push_str(concat!(
        "{\"name\":\"cam\",\"type\":\"Camera3D\",\"args\":{\"position\":[0,2,12]}}\n",
        "{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":",
        "{\"vsync\":false,\"shadow_map_size\":0}}\n",
        "{\"name\":\"bench_mesh\",\"type\":\"ProceduralMesh\",\"args\":",
        "{\"generator\":\"box\",\"half_extents\":[0.4,0.4,0.4]}}\n",
        "{\"name\":\"bench_tex\",\"type\":\"Texture\",\"args\":",
        "{\"generator\":\"checker\",\"resolution\":64}}\n",
        "{\"name\":\"bench_mat\",\"type\":\"Material\",\"args\":",
        "{\"albedo\":\"bench_tex\",\"roughness\":0.6}}\n",
    ));
    for i in 0..props {
        let x = (i % 100) as f32 * 1.2;
        let z = (i / 100) as f32 * 1.2;
        out.push_str(&format!(
            "{{\"name\":\"p{i}\",\"type\":\"Prop\",\"args\":{{\"mesh\":\"bench_mesh\",\
             \"material\":\"bench_mat\",\"position\":[{x},0.45,{z}]}}}}\n"
        ));
    }
    out
}

fn main() {
    // Anchor .concinnity/ (the payload cache) inside the workspace target/
    // dir so bench runs never read or write a real project's build state.
    // Absolute, because cargo runs bench binaries from the package dir.
    concinnity_cook::paths::set_root(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../target/bench-cook-state"
    ));

    let mut bench = Bench::from_env();

    for (n, label) in SIZES {
        let content = world_jsonl(n);

        bench.run(&format!("cook/prepare_world/{label}"), n as u64, || {
            let loaded = prepare_world(&content).expect("bench world validates");
            loaded.assets.len()
        });

        bench.run(&format!("cook/build/{label}"), n as u64, || {
            let result = build_pipeline_from_str(&content, None).expect("bench world compiles");
            result.defs.len()
        });

        // The blob image the compile above would ship: full metadata plus the
        // primary payload section, assembled the way the cook's writer does.
        let result = build_pipeline_from_str(&content, None).expect("bench world compiles");
        let meta = BlobMeta {
            manifest: WorldManifest::from_records(&result.defs, &result.resources),
            defs: result.defs,
            resources: result.resources,
            scene_groups: result.scene_groups,
            mesh_bounds: result.mesh_bounds,
            physics_budget: result.physics_budget,
        };
        let payload = result.payloads.first().map(Vec::as_slice).unwrap_or(&[]);
        let image = encode_cnb(concinnity_core::SCHEMA_HASH, &meta, payload).expect("blob encodes");

        bench.run(&format!("cook/blob_encode/{label}"), n as u64, || {
            encode_cnb(concinnity_core::SCHEMA_HASH, &meta, payload)
                .expect("blob encodes")
                .len()
        });

        bench.run(&format!("cook/blob_parse/{label}"), n as u64, || {
            let (meta, payload_start) =
                parse_cnb(concinnity_core::SCHEMA_HASH, &image).expect("blob parses");
            (meta.defs.len(), payload_start)
        });
    }

    bench.finish();
}
