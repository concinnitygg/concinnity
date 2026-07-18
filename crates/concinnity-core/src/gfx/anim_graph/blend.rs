// src/gfx/anim_graph/blend.rs
//
// What a graph state plays: a single clip or a blendspace over one or two
// parameters. Blendspace weights are pure functions of the parameter values,
// so the same math serves the cursor's clock (effective duration), the pose
// sampler, and the `anim-state` debug report.

// One playable member: a clip index in the target's clip list plus its
// duration, copied at compile time for wrap / phase math (refreshed on clip
// hot-reload).
#[derive(Debug, Clone)]
pub struct ClipPlay {
    pub clip: usize,
    pub duration_secs: f32,
}

// A 1D blendspace: members at ascending `thresholds` positions along one
// parameter. The parameter picks the bracketing pair and lerps their
// weights; outside the range the nearest end member plays alone.
#[derive(Debug, Clone)]
pub struct Blend1D {
    pub param: usize,
    pub thresholds: Vec<f32>,
    pub plays: Vec<ClipPlay>,
    pub sync: bool,
}

// A 2D blendspace: members on a regular grid over two parameters, weighted
// bilinearly across the four members surrounding the parameter point.
// `plays` is row-major: `plays[iy * x_values.len() + ix]` sits at
// `(x_values[ix], y_values[iy])`.
#[derive(Debug, Clone)]
pub struct Blend2D {
    pub param_x: usize,
    pub param_y: usize,
    pub x_values: Vec<f32>,
    pub y_values: Vec<f32>,
    pub plays: Vec<ClipPlay>,
    pub sync: bool,
}

// What a compiled state plays.
#[derive(Debug, Clone)]
pub enum StatePlay {
    Clip(ClipPlay),
    Blend1D(Blend1D),
    Blend2D(Blend2D),
}

impl StatePlay {
    // Whether members share one normalized phase clock (foot-phase
    // alignment). Single clips have nothing to sync against.
    pub fn sync(&self) -> bool {
        match self {
            StatePlay::Clip(_) => false,
            StatePlay::Blend1D(b) => b.sync,
            StatePlay::Blend2D(b) => b.sync,
        }
    }

    // The playable members, in weight order.
    pub fn members(&self) -> &[ClipPlay] {
        match self {
            StatePlay::Clip(play) => core::slice::from_ref(play),
            StatePlay::Blend1D(b) => &b.plays,
            StatePlay::Blend2D(b) => &b.plays,
        }
    }

    pub fn members_mut(&mut self) -> &mut [ClipPlay] {
        match self {
            StatePlay::Clip(play) => core::slice::from_mut(play),
            StatePlay::Blend1D(b) => &mut b.plays,
            StatePlay::Blend2D(b) => &mut b.plays,
        }
    }

    // One weight per member at the given parameter values. A single clip is
    // always `[1.0]`.
    pub fn weights(&self, params: &[f32]) -> Vec<f32> {
        let at = |i: usize| params.get(i).copied().unwrap_or(0.0);
        match self {
            StatePlay::Clip(_) => vec![1.0],
            StatePlay::Blend1D(b) => blend1d_weights(&b.thresholds, at(b.param)),
            StatePlay::Blend2D(b) => {
                blend2d_weights(&b.x_values, &b.y_values, at(b.param_x), at(b.param_y))
            }
        }
    }

    // Weighted-average member duration: the length of one pass of the state
    // at the given weights. Members with zero weight contribute nothing, so
    // a pure-walk pose is one walk cycle long.
    pub fn effective_duration(&self, weights: &[f32]) -> f32 {
        let members = self.members();
        let mut total_w = 0.0f32;
        let mut acc = 0.0f32;
        for (member, &w) in members.iter().zip(weights) {
            let w = w.max(0.0);
            acc += w * member.duration_secs;
            total_w += w;
        }
        if total_w <= 1e-6 {
            members.first().map(|m| m.duration_secs).unwrap_or(0.0)
        } else {
            acc / total_w
        }
    }
}

// 1D blendspace weights: `x` against ascending `thresholds`. Outside the
// range the nearest end gets full weight; inside, the bracketing pair lerps.
// At most two weights are nonzero.
pub fn blend1d_weights(thresholds: &[f32], x: f32) -> Vec<f32> {
    let mut weights = vec![0.0; thresholds.len()];
    let Some((&first, &last)) = thresholds.first().zip(thresholds.last()) else {
        return weights;
    };
    if x <= first {
        weights[0] = 1.0;
        return weights;
    }
    if x >= last {
        *weights.last_mut().expect("non-empty") = 1.0;
        return weights;
    }
    for i in 0..thresholds.len() - 1 {
        let (a, b) = (thresholds[i], thresholds[i + 1]);
        if x >= a && x <= b {
            let f = (x - a) / (b - a).max(1e-6);
            weights[i] = 1.0 - f;
            weights[i + 1] = f;
            break;
        }
    }
    weights
}

// 2D blendspace weights: bilinear over the grid cell containing `(x, y)`,
// clamped to the grid edges. Row-major to match `Blend2D::plays`; at most
// four weights are nonzero.
pub fn blend2d_weights(x_values: &[f32], y_values: &[f32], x: f32, y: f32) -> Vec<f32> {
    let mut weights = vec![0.0; x_values.len() * y_values.len()];
    if x_values.is_empty() || y_values.is_empty() {
        return weights;
    }
    let (ix, fx) = axis_segment(x_values, x);
    let (iy, fy) = axis_segment(y_values, y);
    let nx = x_values.len();
    let ix1 = (ix + 1).min(nx - 1);
    let iy1 = (iy + 1).min(y_values.len() - 1);
    weights[iy * nx + ix] += (1.0 - fx) * (1.0 - fy);
    weights[iy * nx + ix1] += fx * (1.0 - fy);
    weights[iy1 * nx + ix] += (1.0 - fx) * fy;
    weights[iy1 * nx + ix1] += fx * fy;
    weights
}

// The segment of an ascending axis containing `v`: the lower sample index
// plus the fraction toward the next. Clamps outside the range.
fn axis_segment(values: &[f32], v: f32) -> (usize, f32) {
    if v <= values[0] || values.len() == 1 {
        return (0, 0.0);
    }
    if v >= values[values.len() - 1] {
        return (values.len() - 1, 0.0);
    }
    for i in 0..values.len() - 1 {
        let (a, b) = (values[i], values[i + 1]);
        if v >= a && v <= b {
            return (i, (v - a) / (b - a).max(1e-6));
        }
    }
    (values.len() - 1, 0.0)
}
