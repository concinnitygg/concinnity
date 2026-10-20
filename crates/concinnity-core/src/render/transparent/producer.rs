//! The static transparent producers (glass panes and water surfaces) as the
//! VK and DX backends hold them: each producer's pipeline set over its records,
//! and the whole-pass pipeline choice the encoder makes from them. `P` is the
//! backend's pipeline handle and `R` its per-record GPU state.

use alloc::vec::Vec;

use crate::planet::Rebase;
use crate::render::transparent::Producer;
use crate::render::uniforms::WaterParams;
use crate::render::water::frame::WaterMotion;

/// A backend record that accepts a water surface's per-frame params into its
/// frame slot's copy.
pub trait WaterParamsSink {
    /// Overwrite frame slot `frame_idx`'s params copy with `params`.
    fn write_water_params(&self, frame_idx: usize, params: &WaterParams);
}

/// One drawable record of a static producer: the backend's geometry and params
/// state plus what the pass decides from.
pub struct TransparentRecord<R> {
    /// The backend's buffers and descriptors for this record.
    pub gpu: R,
    /// Whether the record draws.
    pub visible: bool,
    /// World-space center, the back-to-front sort key.
    pub center: [f32; 3],
    /// The planar reflection slot the record samples, or `None` when it keeps
    /// the probe cube.
    pub planar_slot: Option<usize>,
}

/// The base pipeline and the two ray-traced variants one shape draws with. The
/// RT pair is `Some` only when the device can trace and the compile succeeded.
pub struct ShapePipelines<P> {
    /// The probe / planar pipeline.
    pub base: P,
    /// The flat-tint per-pixel trace.
    pub flat_rt: Option<P>,
    /// The trace that shades hits from the bindless texture pool.
    pub textured_rt: Option<P>,
}

impl<P> ShapePipelines<P> {
    /// The pipeline a frame draws with: the textured trace when `rt_live` and
    /// `textured`, the flat trace when `rt_live` alone, else the base.
    ///
    /// The variants are built against different layouts and the encoder binds
    /// one layout for the whole pass, so falling back across them would draw
    /// under an incompatible layout. The pass is gated on
    /// [`TransparentProducers::traces`] for every live producer first, so a
    /// shape is never asked for a variant it lacks.
    ///
    /// # Panics
    ///
    /// Panics if `rt_live` asks for a variant this producer was not built with.
    pub fn pick(&self, rt_live: bool, textured: bool) -> &P {
        match (rt_live, textured) {
            (true, true) => self
                .textured_rt
                .as_ref()
                .expect("rt_textured_ready gated the frame on every producer's textured pipeline"),
            (true, false) => self
                .flat_rt
                .as_ref()
                .expect("rt_pipelines_ready gated the frame on every producer's flat RT pipeline"),
            _ => &self.base,
        }
    }

    /// Whether this shape built the trace variant `textured` names.
    pub fn traces(&self, textured: bool) -> bool {
        if textured {
            self.textured_rt.is_some()
        } else {
            self.flat_rt.is_some()
        }
    }
}

/// One static producer's pipelines plus its records. `water` is aligned with
/// `records` for the water producer and empty for glass, whose params never
/// change.
pub struct TransparentProducer<P, R> {
    /// The producer's own shape pipelines.
    pub shape: ShapePipelines<P>,
    /// The glass reflection pre-pass pipeline (flat), built beside the RT pair
    /// for panes; `None` for water, which traces in place.
    pub reflection_flat: Option<P>,
    /// The glass reflection pre-pass pipeline (textured).
    pub reflection_textured: Option<P>,
    /// A planet sea's own shape pipelines, for the water producer's sea
    /// records; `None` when no record is one.
    pub sea: Option<ShapePipelines<P>>,
    /// The records, in declaration order.
    pub records: Vec<TransparentRecord<R>>,
    /// Each water record's motion, parallel to `records`.
    pub water: Vec<WaterMotion>,
}

impl<P, R> TransparentProducer<P, R> {
    /// The pipeline record `i` draws with: a planet sea's own, or the
    /// producer's.
    pub fn record_pipeline(&self, i: usize, rt_live: bool, textured: bool) -> &P {
        match &self.sea {
            Some(sea) if self.water.get(i).is_some_and(|w| w.body().sea.is_some()) => {
                sea.pick(rt_live, textured)
            }
            _ => self.shape.pick(rt_live, textured),
        }
    }

    /// Whether every record can trace: the producer's and its sea's variant
    /// `textured` names.
    pub fn traces(&self, textured: bool) -> bool {
        self.shape.traces(textured) && self.sea.as_ref().is_none_or(|s| s.traces(textured))
    }

    /// This producer's reflection pre-pass pipeline, or `None` when it traces in
    /// place.
    pub fn reflection_pipeline(&self, textured: bool) -> Option<&P> {
        if textured {
            self.reflection_textured.as_ref()
        } else {
            self.reflection_flat.as_ref()
        }
    }

    /// Whether any record draws.
    pub fn any_visible(&self) -> bool {
        self.records.iter().any(|r| r.visible)
    }

    /// Whether a visible record samples a simulated ocean.
    pub fn samples_ocean(&self) -> bool {
        self.records
            .iter()
            .zip(&self.water)
            .any(|(r, w)| r.visible && w.body().ocean.is_some())
    }

    /// Whether a visible record holds a planar slot.
    fn planar_slot_live(&self) -> bool {
        self.records
            .iter()
            .any(|r| r.visible && r.planar_slot.is_some())
    }

    /// Carry every water surface into the frame the world moved to.
    pub fn rebase_water(&mut self, rebase: &Rebase) {
        for motion in &mut self.water {
            motion.rebase(rebase);
        }
    }
}

impl<P, R: WaterParamsSink> TransparentProducer<P, R> {
    /// Write each visible water surface's params, advanced to `time` seconds
    /// and seen from `camera`, into its `frame_idx` copy.
    pub fn advance_water(&self, frame_idx: usize, time: f64, camera: [f32; 3]) {
        for (record, water) in self.records.iter().zip(&self.water) {
            if record.visible {
                record
                    .gpu
                    .write_water_params(frame_idx, &water.params_at(time, camera));
            }
        }
    }
}

/// The two static producers, each `None` when the world declared none of it.
pub struct TransparentProducers<P, R> {
    /// The `GlassPanel` producer.
    pub glass: Option<TransparentProducer<P, R>>,
    /// The `WaterSurface` producer.
    pub water: Option<TransparentProducer<P, R>>,
}

impl<P, R> TransparentProducers<P, R> {
    /// Whether every live producer built the trace variant `textured` names.
    /// All-or-nothing, so the RT choice is per frame rather than per producer.
    pub fn traces(&self, textured: bool) -> bool {
        self.glass.as_ref().is_none_or(|p| p.traces(textured))
            && self.water.as_ref().is_none_or(|p| p.traces(textured))
    }

    /// Whether any record of either producer draws.
    pub fn any_visible(&self) -> bool {
        self.glass
            .as_ref()
            .is_some_and(TransparentProducer::any_visible)
            || self
                .water
                .as_ref()
                .is_some_and(TransparentProducer::any_visible)
    }

    /// Whether a visible water surface holds a planar slot, so the mirror
    /// re-render has a consumer even while the trace is live. Glass is not
    /// counted: it takes the trace over the mirror.
    pub fn water_planar_slot_live(&self) -> bool {
        self.water
            .as_ref()
            .is_some_and(TransparentProducer::planar_slot_live)
    }

    /// The record-holding producer `kind` names, or `None` for the mesh
    /// producer or one the world did not declare.
    pub fn producer(&self, kind: Producer) -> Option<&TransparentProducer<P, R>> {
        match kind {
            Producer::Glass => self.glass.as_ref(),
            Producer::Water => self.water.as_ref(),
            Producer::GlassMesh => None,
        }
    }

    /// Record `index` of producer `kind`, if both exist.
    pub fn record(&self, kind: Producer, index: usize) -> Option<&TransparentRecord<R>> {
        self.producer(kind)?.records.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::WaterSurface;
    use crate::math::SaturatingFrom;
    use crate::render::water::WaterBody;
    use crate::render::water::sea::{SeaBody, SeaSphere};
    use crate::render::wind::WindField;
    use core::cell::Cell;

    fn shape(base: u8, flat: Option<u8>, tex: Option<u8>) -> ShapePipelines<u8> {
        ShapePipelines {
            base,
            flat_rt: flat,
            textured_rt: tex,
        }
    }

    fn record(visible: bool, planar_slot: Option<usize>) -> TransparentRecord<Cell<u32>> {
        TransparentRecord {
            gpu: Cell::new(0),
            visible,
            center: [0.0; 3],
            planar_slot,
        }
    }

    fn producer(
        shape: ShapePipelines<u8>,
        records: Vec<TransparentRecord<Cell<u32>>>,
    ) -> TransparentProducer<u8, Cell<u32>> {
        TransparentProducer {
            shape,
            reflection_flat: None,
            reflection_textured: None,
            sea: None,
            records,
            water: Vec::new(),
        }
    }

    fn body() -> WaterBody {
        WaterBody::resolve(&WaterSurface::default(), &WindField::new(None))
    }

    impl WaterParamsSink for Cell<u32> {
        fn write_water_params(&self, frame_idx: usize, _: &WaterParams) {
            self.set(self.get() + 1 + u32::saturating_from(frame_idx) * 100);
        }
    }

    #[test]
    fn pick_takes_the_variant_the_frame_asks_for() {
        let s = shape(1, Some(2), Some(3));
        assert_eq!(*s.pick(false, false), 1);
        assert_eq!(*s.pick(false, true), 1);
        assert_eq!(*s.pick(true, false), 2);
        assert_eq!(*s.pick(true, true), 3);
        assert!(s.traces(false) && s.traces(true));
        assert!(!shape(1, None, Some(3)).traces(false));
    }

    #[test]
    fn a_sea_record_draws_with_the_sea_pipelines() {
        let mut p = producer(shape(1, Some(2), None), vec_of(2));
        let mut sea = body();
        let planet = crate::components::Planet::default().shape();
        sea.sea = Some(SeaBody::new(SeaSphere::new(&planet, 0.0), 1.0));
        p.water = alloc::vec![
            WaterMotion::new(&body(), false),
            WaterMotion::new(&sea, false)
        ];
        p.sea = Some(shape(10, Some(20), None));
        assert_eq!(*p.record_pipeline(0, true, false), 2);
        assert_eq!(*p.record_pipeline(1, true, false), 20);
        assert_eq!(*p.record_pipeline(1, false, false), 10);
        // The sea's textured variant is missing, so the producer cannot trace
        // textured even though its own shape would not either.
        assert!(p.traces(false));
        assert!(!p.traces(true));
    }

    fn vec_of(n: usize) -> Vec<TransparentRecord<Cell<u32>>> {
        (0..n).map(|_| record(true, None)).collect()
    }

    #[test]
    fn producers_trace_only_when_every_live_one_does() {
        let mut all = TransparentProducers {
            glass: Some(producer(shape(1, Some(2), None), vec_of(1))),
            water: None,
        };
        assert!(all.traces(false));
        assert!(!all.traces(true));
        all.water = Some(producer(shape(1, None, None), vec_of(1)));
        assert!(!all.traces(false));
        let none: TransparentProducers<u8, Cell<u32>> = TransparentProducers {
            glass: None,
            water: None,
        };
        assert!(none.traces(true), "nothing live has nothing to lack");
        assert!(!none.any_visible());
    }

    #[test]
    fn visibility_and_planar_consumers_count_visible_records_only() {
        let all = TransparentProducers {
            glass: Some(producer(
                shape(0, None, None),
                alloc::vec![record(true, Some(0))],
            )),
            water: Some(producer(
                shape(0, None, None),
                alloc::vec![record(false, Some(1)), record(true, None)],
            )),
        };
        assert!(all.any_visible());
        assert!(!all.water_planar_slot_live(), "glass's slot is not counted");
        assert!(all.record(Producer::Water, 1).is_some());
        assert!(all.record(Producer::GlassMesh, 0).is_none());
    }

    #[test]
    fn advance_water_writes_only_visible_surfaces() {
        let mut p = producer(
            shape(0, None, None),
            alloc::vec![record(true, None), record(false, None)],
        );
        p.water = alloc::vec![
            WaterMotion::new(&body(), false),
            WaterMotion::new(&body(), false)
        ];
        p.advance_water(1, 0.5, [0.0; 3]);
        assert_eq!(p.records[0].gpu.get(), 101);
        assert_eq!(p.records[1].gpu.get(), 0);
    }

    #[test]
    fn reflection_pipeline_names_the_requested_variant() {
        let mut p = producer(shape(0, None, None), Vec::new());
        assert_eq!(p.reflection_pipeline(false), None);
        p.reflection_flat = Some(4);
        p.reflection_textured = Some(5);
        assert_eq!(p.reflection_pipeline(false), Some(&4));
        assert_eq!(p.reflection_pipeline(true), Some(&5));
    }
}
