//! Packing the static draws' cull records on the job pool, shared by the three
//! backends: in one chunk on the render thread unless measuring both ways says
//! a chunk per worker pays.

use std::cell::Cell;

use concinnity_core::gfx::render_types::{DrawObject, GpuObjectData};
use concinnity_core::profile::{FanOutJudge, FanOutTiming};
use concinnity_core::render::record_pack::{
    MAX_PACK_CHUNKS, PackChunk, PackView, StaticPack, chunk_count,
};
use concinnity_host::thread::jobs::pool;

// Below this much inline packing a split is never tried. Measured on a 12-core
// Mac with the workers asleep between packs: at ~0.3 ms (4k records) a split is
// even to slightly ahead, from ~0.5 ms (8k) it cuts the wait by a quarter.
const MIN_FANOUT_WORK_NS: u64 = 300_000;

/// The render thread's record packer. Holds the fan-out judge's measurements
/// across frames; scheduling never changes the bytes written.
#[derive(Debug, Default)]
pub(crate) struct RecordPacker {
    judge: Cell<FanOutJudge<MIN_FANOUT_WORK_NS>>,
}

impl RecordPacker {
    /// Pack `pack` for `view`, each draw's object record from `record`, and
    /// report how long the calling thread was held against the work.
    pub(crate) fn pack<F>(&self, pack: StaticPack<'_>, view: &PackView, record: &F) -> FanOutTiming
    where
        F: Fn(&DrawObject) -> GpuObjectData + Sync,
    {
        let records = pack.len();
        let jobs = pool();
        let workers = jobs.thread_count();
        let mut judge = self.judge.get();
        let count = match judge.fans_out(records, workers) {
            true => chunk_count(records, workers),
            false => 1,
        };
        let mut chunks: [Option<PackChunk<'_>>; MAX_PACK_CHUNKS] = Default::default();
        let used = pack.split(count, &mut chunks);
        let timing = jobs.parallel_for_timed(&mut chunks[..used], |chunk| {
            if let Some(chunk) = chunk.take() {
                chunk.pack(view, record);
            }
        });
        judge.observe(records, used > 1, timing);
        self.judge.set(judge);
        timing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::render_types::{FarFieldTint, GpuDrawArgs, MaterialUniforms};
    use concinnity_core::render::model_history::{HistoryMode, ModelHistory};
    use concinnity_core::render::record_pack::{MIN_CHUNK_RECORDS, pool_object_record};

    fn object(i: usize) -> DrawObject {
        let x = i as f32;
        DrawObject {
            vertex_offset: 0,
            vertex_count: 3,
            index_offset: i * 3,
            index_count: 3,
            base_vertex: 0,
            geometry_generation: 0,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [x, 0.0, 0.0, 1.0],
            ],
            texture_slot: i % 5,
            normal_map_slot: i % 3,
            material: MaterialUniforms {
                roughness: 0.5,
                metallic: 0.0,
                alpha_cutoff: 0.0,
                opacity: 1.0,
                tint: [1.0; 3],
                _pad0: 0.0,
                emissive: [0.0; 3],
                _pad1: 0.0,
                emissive_map_index: 0,
                orm_map_index: 0,
                transparent: 0,
                see_through: 0,
                params_index: 0,
                far_field: FarFieldTint::NONE,
            },
            shader_bucket: (i % 2) as u32,
            visible: !i.is_multiple_of(9),
            resident: true,
            bb_min: [x - 0.5, -0.5, -0.5],
            bb_max: [x + 0.5, 0.5, 0.5],
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        }
    }

    fn pack_with(packer: &RecordPacker, objects: &[DrawObject]) -> (Vec<u8>, Vec<u8>) {
        let mut records = vec![bytemuck::Zeroable::zeroed(); objects.len()];
        let mut args: Vec<GpuDrawArgs> = vec![bytemuck::Zeroable::zeroed(); objects.len()];
        let mut history = ModelHistory::new();
        history.begin(HistoryMode::Track, objects.len());
        let view = PackView {
            cam_pos: [0.0; 3],
            mesh_glass_active: false,
        };
        packer.pack(
            StaticPack {
                objects,
                records: &mut records,
                args: &mut args,
                history: history.window(),
            },
            &view,
            &|obj| pool_object_record(obj, 8),
        );
        (
            bytemuck::cast_slice(&records).to_vec(),
            bytemuck::cast_slice(&args).to_vec(),
        )
    }

    // Whether a pack runs inline or across the pool, the bytes are the same.
    #[test]
    fn inline_and_fanned_out_packs_write_the_same_bytes() {
        let objects: Vec<DrawObject> = (0..8 * MIN_CHUNK_RECORDS).map(object).collect();
        let inline = RecordPacker::default();
        let fanned = RecordPacker::default();
        let mut judge = FanOutJudge::default();
        let inline_ms = FanOutTiming {
            wall_us: 1_000_000,
            job_sum_us: 1_000_000,
            ..FanOutTiming::default()
        };
        judge.observe(1, false, inline_ms);
        fanned.judge.set(judge);
        assert!(fanned.judge.get().fans_out(objects.len(), 8));
        assert_eq!(pack_with(&inline, &objects), pack_with(&fanned, &objects));
    }

    // The first pack is unmeasured, so it runs inline and then informs the judge.
    #[test]
    fn a_pack_feeds_the_next_packs_judgment() {
        let objects: Vec<DrawObject> = (0..64).map(object).collect();
        let packer = RecordPacker::default();
        assert_eq!(packer.judge.get(), FanOutJudge::default());
        pack_with(&packer, &objects);
        assert_ne!(packer.judge.get(), FanOutJudge::default());
    }
}
