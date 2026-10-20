//! Which reflection stages run this frame, decided once for every backend.
//!
//! Two resolves can feed the reflection composite, a screen-space one (SSR)
//! and a ray-traced one, and they share one slot of the frame: at most one
//! runs. Ray tracing takes the slot while its BVH is live; without one, an
//! authored SSR resolve covers.

/// The reflection stages a frame runs, from what the world authored and what
/// is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReflectionPath {
    /// The ray-traced resolve traces a live BVH.
    pub rt_trace: bool,
    /// The ray-traced resolve's node runs: tracing a live BVH, or, with no
    /// authored SSR to cover, writing an empty reflection that leaves the scene
    /// as it was. A renderer that reads the composite's output as the scene
    /// every frame needs the composite written every frame, and this is the
    /// node that does it while the BVH is missing.
    pub rt_node: bool,
    /// The SSR resolve runs: SSR is authored and ray tracing did not take its
    /// slot.
    pub ssr_resolve: bool,
    /// The composite exists: some resolve can feed it, so a BVH coming or
    /// going never rebuilds it.
    pub composite: bool,
}

impl ReflectionPath {
    /// The stages for a world that authored SSR (`ssr_authored`), runs the
    /// ray-traced reflection pass (`rt_pass`), and has a live BVH to trace
    /// (`bvh_live`).
    pub fn new(ssr_authored: bool, rt_pass: bool, bvh_live: bool) -> Self {
        let rt_trace = rt_pass && bvh_live;
        Self {
            rt_trace,
            rt_node: rt_trace || (rt_pass && !ssr_authored),
            ssr_resolve: ssr_authored && !rt_trace,
            composite: rt_pass || ssr_authored,
        }
    }

    /// Whether a resolve composites a reflection over the scene this frame,
    /// which is when the forward pass hands it the glossy dielectric specular.
    pub fn resolves(&self) -> bool {
        self.rt_trace || self.ssr_resolve
    }
}

#[cfg(test)]
mod tests {
    use super::ReflectionPath;

    #[test]
    fn a_live_trace_takes_the_resolve_slot_from_authored_ssr() {
        let path = ReflectionPath::new(true, true, true);
        assert!(path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn authored_ssr_covers_while_the_rt_pass_has_no_bvh() {
        let path = ReflectionPath::new(true, true, false);
        assert!(!path.rt_trace && !path.rt_node);
        assert!(path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn authored_ssr_resolves_without_rt() {
        for bvh_live in [false, true] {
            let path = ReflectionPath::new(true, false, bvh_live);
            assert!(!path.rt_trace && !path.rt_node);
            assert!(path.ssr_resolve);
            assert!(path.composite && path.resolves());
        }
    }

    #[test]
    fn rt_alone_traces_its_live_bvh() {
        let path = ReflectionPath::new(false, true, true);
        assert!(path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn rt_alone_without_a_bvh_feeds_the_composite_nothing() {
        // The RT node still runs, so the composite stays fed, but it traces
        // nothing and the forward pass keeps its own specular.
        let path = ReflectionPath::new(false, true, false);
        assert!(!path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && !path.resolves());
    }

    #[test]
    fn no_resolve_leaves_no_composite() {
        // RT off or unsupported without authored SSR, or a SSGI-only world.
        for bvh_live in [false, true] {
            let path = ReflectionPath::new(false, false, bvh_live);
            assert!(!path.rt_trace && !path.rt_node && !path.ssr_resolve);
            assert!(!path.composite && !path.resolves());
        }
    }

    #[test]
    fn at_most_one_resolve_node_runs() {
        for bits in 0..8u8 {
            let path = ReflectionPath::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            assert!(!(path.rt_node && path.ssr_resolve), "{bits:03b}: {path:?}");
            assert!(!(path.rt_trace && path.ssr_resolve), "{bits:03b}: {path:?}");
            assert_eq!(
                path.composite,
                path.rt_node || path.ssr_resolve,
                "{bits:03b}"
            );
            assert_eq!(path.resolves(), path.rt_trace || path.ssr_resolve);
        }
    }

    // The full table, in (ssr_authored, rt_pass, bvh_live) order.
    #[test]
    fn the_table() {
        let rows = [
            ((false, false, false), (false, false, false, false)),
            ((false, false, true), (false, false, false, false)),
            ((false, true, false), (false, true, false, true)),
            ((false, true, true), (true, true, false, true)),
            ((true, false, false), (false, false, true, true)),
            ((true, false, true), (false, false, true, true)),
            ((true, true, false), (false, false, true, true)),
            ((true, true, true), (true, true, false, true)),
        ];
        for ((ssr, rt, bvh), (rt_trace, rt_node, ssr_resolve, composite)) in rows {
            let want = ReflectionPath {
                rt_trace,
                rt_node,
                ssr_resolve,
                composite,
            };
            assert_eq!(ReflectionPath::new(ssr, rt, bvh), want, "{ssr} {rt} {bvh}");
        }
    }
}
