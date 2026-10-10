//! The static draws' records, packed a chunk at a time.

use crate::gfx::lod;
use crate::gfx::render_types::{
    DrawObject, GpuDrawArgs, GpuObjectData, albedo_pool_index, draw_args_bucket_bits,
    draw_args_flags, normal_pool_index, pack_object_record,
};
use crate::render::model_history::HistoryWindow;

/// What every chunk of one frame's pack reads besides its own draws.
#[derive(Clone, Copy, Debug)]
pub struct PackView {
    /// The eye the active LOD is picked from.
    pub cam_pos: [f32; 3],
    /// Whether see-through glass draws in the transparent pass this frame
    /// instead of the opaque one, which disables its opaque record.
    pub mesh_glass_active: bool,
}

/// One pack of the static draws: draw `i` fills record `i` of both outputs,
/// from record 0, and asks `history` for record `i`'s flags.
pub struct StaticPack<'a> {
    /// The draws to pack.
    pub objects: &'a [DrawObject],
    /// Their object records, at least as long as `objects`.
    pub records: &'a mut [GpuObjectData],
    /// Their draw arguments, at least as long as `objects`.
    pub args: &'a mut [GpuDrawArgs],
    /// The model-history tracker's window from record 0.
    pub history: HistoryWindow<'a>,
}

impl<'a> StaticPack<'a> {
    /// How many records the pack writes: every draw that both outputs have
    /// room for.
    pub fn len(&self) -> usize {
        self.objects
            .len()
            .min(self.records.len())
            .min(self.args.len())
    }

    /// Whether the pack writes nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Divide into `count` chunks (see [`super::chunk_range`]), stored in
    /// order at the front of `chunks`. Returns how many were stored: `count`
    /// clamped to `1..=chunks.len()` and to the record count, or 0 for an empty
    /// pack.
    pub fn split(self, count: usize, chunks: &mut [Option<PackChunk<'a>>]) -> usize {
        let len = self.len();
        if len == 0 || chunks.is_empty() {
            return 0;
        }
        let count = count.clamp(1, chunks.len()).min(len);
        let mut objects = &self.objects[..len];
        let mut records = &mut self.records[..len];
        let mut args = &mut self.args[..len];
        let mut history = self.history;
        for (index, slot) in chunks.iter_mut().take(count).enumerate() {
            let n = super::chunk_range(len, count, index).len();
            let (chunk_objects, rest_objects) = objects.split_at(n);
            let (chunk_records, rest_records) = core::mem::take(&mut records).split_at_mut(n);
            let (chunk_args, rest_args) = core::mem::take(&mut args).split_at_mut(n);
            let (chunk_history, rest_history) = history.split_at(n);
            *slot = Some(PackChunk {
                objects: chunk_objects,
                records: chunk_records,
                args: chunk_args,
                history: chunk_history,
            });
            (objects, records, args, history) =
                (rest_objects, rest_records, rest_args, rest_history);
        }
        count
    }
}

/// A contiguous run of a [`StaticPack`]'s draws and the records only it
/// writes. Chunks of one pack share nothing, so each can run on its own
/// thread.
pub struct PackChunk<'a> {
    objects: &'a [DrawObject],
    records: &'a mut [GpuObjectData],
    args: &'a mut [GpuDrawArgs],
    history: HistoryWindow<'a>,
}

impl PackChunk<'_> {
    /// Write the chunk's records: each draw's object record from `record`,
    /// and its draw arguments for this frame's view.
    pub fn pack(self, view: &PackView, record: &impl Fn(&DrawObject) -> GpuObjectData) {
        let PackChunk {
            objects,
            records,
            args,
            mut history,
        } = self;
        let first = history.first();
        for (k, ((obj, rec), arg)) in objects.iter().zip(records).zip(args).enumerate() {
            *rec = record(obj);
            *arg = draw_args(obj, first + k, view, &mut history);
        }
    }
}

// A static draw's arguments: this frame's LOD slice and its cull-decision
// bits. Record `index` is draw `index`.
fn draw_args(
    obj: &DrawObject,
    index: usize,
    view: &PackView,
    history: &mut HistoryWindow<'_>,
) -> GpuDrawArgs {
    let (index_offset, index_count) = obj.active_lod(lod::camera_distance(obj, view.cam_pos));
    // See-through glass leaves the opaque pass while it draws as transparent,
    // keeping its record so every per-record index stays put.
    let opaque_visible = obj.visible && !(view.mesh_glass_active && obj.material.see_through != 0);
    GpuDrawArgs {
        index_count: index_count as u32,
        index_offset: index_offset as u32,
        base_vertex: obj.base_vertex as u32,
        // The shader bucket rides the upper flag bits so the cull kernel can
        // route the record's command into that bucket's region.
        flags: draw_args_flags(opaque_visible, obj.resident, obj.cullable())
            | draw_args_bucket_bits(obj.shader_bucket)
            | history.draw_flags(index, index),
    }
}

/// A draw's object record addressing the handle-indexed texture pool of
/// `texture_count` real textures: albedo at its own slot (or the white
/// fallback), normal at its own slot (or the flat-normal fallback).
pub fn pool_object_record(obj: &DrawObject, texture_count: u32) -> GpuObjectData {
    pack_object_record(
        obj,
        albedo_pool_index(obj.texture_slot, texture_count),
        normal_pool_index(obj.normal_map_slot, texture_count),
    )
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::gfx::render_types::DrawIndex;
    use crate::gfx::render_types::{LodSlice, draw_args_no_history};
    use crate::render::model_history::{HistoryMode, ModelHistory};
    use crate::render::record_pack::MAX_PACK_CHUNKS;
    use crate::test_support;

    const TEXTURES: u32 = 64;

    // Draws that differ in every input the pack reads: position (LOD pick),
    // visibility, residency, glass, bucket, alternates and unbounded boxes.
    // `texture_slot` is the draw's index, so a record names its draw.
    fn objects(n: usize) -> Vec<DrawObject> {
        (0..n)
            .map(|i| {
                let mut obj = test_support::draw_object();
                let x = i as f32 * 3.0;
                obj.model[3] = [x, 0.0, 0.0, 1.0];
                obj.bb_min = [x - 1.0, -1.0, -1.0];
                obj.bb_max = [x + 1.0, 1.0, 1.0];
                obj.texture_slot = i;
                obj.visible = !i.is_multiple_of(7);
                obj.resident = !i.is_multiple_of(11);
                obj.material.see_through = u32::from(i.is_multiple_of(5));
                obj.shader_bucket = (i % 3) as u32;
                if i.is_multiple_of(13) {
                    obj.bb_min = [f32::NAN; 3];
                }
                if i.is_multiple_of(2) {
                    obj.lod_alternates = vec![
                        LodSlice {
                            index_offset: 100,
                            index_count: 10,
                            switch_distance: 30.0,
                        },
                        LodSlice {
                            index_offset: 200,
                            index_count: 5,
                            switch_distance: 90.0,
                        },
                    ];
                }
                obj
            })
            .collect()
    }

    fn view() -> PackView {
        PackView {
            cam_pos: [0.0, 0.0, 0.0],
            mesh_glass_active: true,
        }
    }

    fn zeroed_records(n: usize) -> Vec<GpuObjectData> {
        vec![bytemuck::Zeroable::zeroed(); n]
    }

    fn zeroed_args(n: usize) -> Vec<GpuDrawArgs> {
        vec![bytemuck::Zeroable::zeroed(); n]
    }

    // Pack `objects` in `count` chunks, one after another on this thread.
    fn pack(
        objects: &[DrawObject],
        count: usize,
        history: &mut ModelHistory,
    ) -> (Vec<GpuObjectData>, Vec<GpuDrawArgs>) {
        let mut records = zeroed_records(objects.len());
        let mut args = zeroed_args(objects.len());
        let pack = StaticPack {
            objects,
            records: &mut records,
            args: &mut args,
            history: history.window(),
        };
        let mut chunks: [Option<PackChunk<'_>>; MAX_PACK_CHUNKS] = Default::default();
        let used = pack.split(count, &mut chunks);
        for chunk in chunks.iter_mut().take(used) {
            let chunk = chunk.take().expect("split stores every chunk it counts");
            chunk.pack(&view(), &|obj| pool_object_record(obj, TEXTURES));
        }
        (records, args)
    }

    #[test]
    fn any_split_writes_the_same_bytes_as_a_serial_pack() {
        let objects = objects(203);
        let mut serial_history = ModelHistory::new();
        serial_history.begin(HistoryMode::Track, objects.len());
        let (records, args) = pack(&objects, 1, &mut serial_history);
        for count in 2..=MAX_PACK_CHUNKS {
            let mut history = ModelHistory::new();
            history.begin(HistoryMode::Track, objects.len());
            let (split_records, split_args) = pack(&objects, count, &mut history);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&split_records),
                bytemuck::cast_slice::<_, u8>(&records),
                "{count} chunks"
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&split_args),
                bytemuck::cast_slice::<_, u8>(&args),
                "{count} chunks"
            );
        }
    }

    #[test]
    fn every_record_is_written_exactly_once() {
        let objects = objects(97);
        for count in 1..=MAX_PACK_CHUNKS {
            let writes: Vec<AtomicU32> = (0..objects.len()).map(|_| AtomicU32::new(0)).collect();
            let mut records = zeroed_records(objects.len());
            let mut args = zeroed_args(objects.len());
            let mut history = ModelHistory::new();
            history.begin(HistoryMode::Track, objects.len());
            let pack = StaticPack {
                objects: &objects,
                records: &mut records,
                args: &mut args,
                history: history.window(),
            };
            let mut chunks: [Option<PackChunk<'_>>; MAX_PACK_CHUNKS] = Default::default();
            let used = pack.split(count, &mut chunks);
            assert_eq!(used, count);
            for chunk in chunks.iter_mut().filter_map(Option::take) {
                chunk.pack(&view(), &|obj| {
                    writes[obj.texture_slot].fetch_add(1, Ordering::Relaxed);
                    pool_object_record(obj, TEXTURES)
                });
            }
            assert!(
                writes.iter().all(|w| w.load(Ordering::Relaxed) == 1),
                "{count} chunks"
            );
        }
    }

    // The history flags are part of the bytes: a reoccupied slot and the
    // first observation must flag the same records however the pack splits.
    #[test]
    fn history_flags_match_across_frames_for_any_split() {
        let objects = objects(64);
        let mut serial = ModelHistory::new();
        let mut split = ModelHistory::new();
        for frame in 0..3 {
            for history in [&mut serial, &mut split] {
                history.begin(HistoryMode::Track, objects.len());
                if frame == 2 {
                    history.reoccupy_draw(DrawIndex(40));
                }
            }
            let (_, serial_args) = pack(&objects, 1, &mut serial);
            let (_, split_args) = pack(&objects, 5, &mut split);
            let flags = |args: &[GpuDrawArgs]| -> Vec<u32> {
                args.iter()
                    .map(|a| a.flags & draw_args_no_history())
                    .collect()
            };
            assert_eq!(flags(&split_args), flags(&serial_args), "frame {frame}");
            let stale = flags(&serial_args).iter().filter(|f| **f != 0).count();
            let expected = match frame {
                0 => objects.len(),
                1 => 0,
                _ => 1,
            };
            assert_eq!(stale, expected, "frame {frame}");
        }
    }

    #[test]
    fn a_pack_writes_only_what_both_outputs_hold() {
        let objects = objects(10);
        let mut records = zeroed_records(6);
        let mut args = zeroed_args(8);
        let mut history = ModelHistory::new();
        let pack = StaticPack {
            objects: &objects,
            records: &mut records,
            args: &mut args,
            history: history.window(),
        };
        assert_eq!(pack.len(), 6);
        let mut chunks: [Option<PackChunk<'_>>; 4] = Default::default();
        assert_eq!(pack.split(9, &mut chunks), 4);
    }

    #[test]
    fn an_empty_pack_has_no_chunks() {
        let mut history = ModelHistory::new();
        let pack = StaticPack {
            objects: &[],
            records: &mut [],
            args: &mut [],
            history: history.window(),
        };
        assert!(pack.is_empty());
        let mut chunks: [Option<PackChunk<'_>>; 4] = Default::default();
        assert_eq!(pack.split(4, &mut chunks), 0);
    }

    // The record for one draw, field by field, so the shared kernel is pinned
    // to what each backend wrote before it existed.
    #[test]
    fn a_draws_arguments_carry_its_lod_bucket_and_flags() {
        let objects = objects(40);
        let mut history = ModelHistory::new();
        history.begin(HistoryMode::Stale, objects.len());
        let (records, args) = pack(&objects, 3, &mut history);
        // Draw 34: x = 102 so LOD2, visible, resident, opaque, bucket 1.
        let a = args[34];
        assert_eq!((a.index_offset, a.index_count, a.base_vertex), (200, 5, 4));
        assert_eq!(
            a.flags,
            draw_args_flags(true, true, true) | draw_args_bucket_bits(1) | draw_args_no_history()
        );
        // Draw 10: see-through glass with the glass path live, so disabled.
        assert_eq!(args[10].flags & draw_args_flags(true, true, false), 0);
        assert_eq!(
            bytemuck::bytes_of(&records[34]),
            bytemuck::bytes_of(&pool_object_record(&objects[34], TEXTURES))
        );
    }
}
