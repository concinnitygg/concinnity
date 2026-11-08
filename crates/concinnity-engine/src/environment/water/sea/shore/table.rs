//! Each shore train's phase across the shallows, tabled once per sea: the
//! integral of its wavenumber over the distance offshore, which is what slows
//! a wave and bunches its crests up as the water shallows. The nodes crowd
//! toward the waterline, where the wavenumber changes fastest, and the shader
//! reads them back with the same cubic Hermite step.

use super::dispersion::wavenumber;
use super::{SHORE_NODES, SHORE_TRAINS};
use concinnity_core::math::SaturatingFrom;

/// The phase, in radians, and the wavenumber, in radians per meter, of every
/// train at each node; node `j` lies `reach * (j / (SHORE_NODES - 1))^2`
/// meters offshore.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTable {
    reach: f64,
    phases: [[f64; SHORE_TRAINS]; SHORE_NODES],
    wavenumbers: [[f64; SHORE_TRAINS]; SHORE_NODES],
}

// Simpson steps per node interval.
const STEPS: usize = 8;

impl PhaseTable {
    /// The table out to `reach` meters offshore for trains of angular
    /// frequencies `omegas` over water `depth(offshore)` deep, never read
    /// shallower than `shallowest`.
    pub fn new(
        reach: f64,
        omegas: [f64; SHORE_TRAINS],
        shallowest: f64,
        depth: impl Fn(f64) -> f64,
    ) -> Self {
        let k = |i: usize, u: f64| wavenumber(omegas[i], depth(u).max(shallowest));
        let mut phases = [[0.0; SHORE_TRAINS]; SHORE_NODES];
        let mut wavenumbers = [[0.0; SHORE_TRAINS]; SHORE_NODES];
        for i in 0..SHORE_TRAINS {
            wavenumbers[0][i] = k(i, 0.0);
            for j in 1..SHORE_NODES {
                let (a, b) = (node(reach, j - 1), node(reach, j));
                let h = (b - a) / STEPS as f64;
                let sum: f64 = (0..=STEPS)
                    .map(|s| {
                        let w = match s {
                            0 | STEPS => 1.0,
                            s if s % 2 == 1 => 4.0,
                            _ => 2.0,
                        };
                        w * k(i, a + s as f64 * h)
                    })
                    .sum();
                phases[j][i] = phases[j - 1][i] + sum * h / 3.0;
                wavenumbers[j][i] = k(i, b);
            }
        }
        Self {
            reach,
            phases,
            wavenumbers,
        }
    }

    /// How far offshore the table reaches, in meters.
    pub fn reach(&self) -> f64 {
        self.reach
    }

    /// Node `j`'s phases, one per train.
    pub fn phases(&self, j: usize) -> [f64; SHORE_TRAINS] {
        self.phases[j]
    }

    /// Node `j`'s wavenumbers, one per train.
    pub fn wavenumbers(&self, j: usize) -> [f64; SHORE_TRAINS] {
        self.wavenumbers[j]
    }

    /// Every train's phase and wavenumber `offshore` meters out: Hermite
    /// between the nodes, straight on at the last node's wavenumber past the
    /// table, and the waterline's at and behind it.
    pub fn at(&self, offshore: f64) -> ([f64; SHORE_TRAINS], [f64; SHORE_TRAINS]) {
        let u = offshore.max(0.0);
        let last = SHORE_NODES - 1;
        if u >= self.reach {
            let k = self.wavenumbers[last];
            let p = core::array::from_fn(|i| self.phases[last][i] + k[i] * (u - self.reach));
            return (p, k);
        }
        let x = last as f64 * libm::sqrt(u / self.reach);
        let j = usize::saturating_from(libm::floor(x)).min(last - 1);
        let (a, b) = (node(self.reach, j), node(self.reach, j + 1));
        let span = b - a;
        let s = (u - a) / span;
        let (h00, h10, h01, h11) = hermite(s);
        let p = core::array::from_fn(|i| {
            h00 * self.phases[j][i]
                + h10 * span * self.wavenumbers[j][i]
                + h01 * self.phases[j + 1][i]
                + h11 * span * self.wavenumbers[j + 1][i]
        });
        let k = core::array::from_fn(|i| {
            self.wavenumbers[j][i] + s * (self.wavenumbers[j + 1][i] - self.wavenumbers[j][i])
        });
        (p, k)
    }
}

/// How far offshore node `j` of a table reaching `reach` meters lies.
pub fn node(reach: f64, j: usize) -> f64 {
    let f = j as f64 / (SHORE_NODES - 1) as f64;
    reach * f * f
}

fn hermite(s: f64) -> (f64, f64, f64, f64) {
    let s2 = s * s;
    let s3 = s2 * s;
    (
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    fn shelf(u: f64) -> f64 {
        60.0 * (1.0 - libm::exp(-u / 750.0))
    }

    fn table() -> PhaseTable {
        let omegas = [1.2, 1.1, 1.3];
        PhaseTable::new(320.0, omegas, 0.1, shelf)
    }

    // Read back between the nodes, the phase grows at the wavenumber the
    // dispersion relation gives at that depth, and it matches a fine
    // integral of it.
    #[test]
    fn the_phase_grows_at_the_local_wavenumber() {
        let t = table();
        for u in [0.7, 3.3, 17.0, 64.0, 150.5, 299.0] {
            let (p0, k) = t.at(u - 0.01);
            let (p1, _) = t.at(u + 0.01);
            for i in 0..SHORE_TRAINS {
                let slope = (p1[i] - p0[i]) / 0.02;
                let want = wavenumber([1.2, 1.1, 1.3][i], shelf(u).max(0.1));
                assert!(
                    (slope - want).abs() / want < 0.01,
                    "{u} {i}: {slope} {want}"
                );
                assert!(
                    (k[i] - want).abs() / want < 0.02,
                    "{u} {i}: {} {want}",
                    k[i]
                );
            }
        }
        let fine: f64 = (0..20_000)
            .map(|s| (f64::from(s) + 0.5) * 0.01)
            .map(|u| wavenumber(1.2, shelf(u).max(0.1)) * 0.01)
            .sum();
        let (p, _) = t.at(200.0);
        assert!((p[0] - fine).abs() < 1e-3 * fine, "{} {fine}", p[0]);
    }

    // The table starts at the waterline with no phase, is continuous across
    // its nodes and runs straight on past its reach.
    #[test]
    fn the_table_is_continuous_and_extends() {
        let t = table();
        assert_eq!(t.at(0.0).0, [0.0; SHORE_TRAINS]);
        assert_eq!(t.at(-5.0).0, [0.0; SHORE_TRAINS]);
        for j in 1..SHORE_NODES - 1 {
            let u = node(320.0, j);
            let (a, _) = t.at(u - 1e-7);
            let (b, _) = t.at(u + 1e-7);
            assert!((a[0] - b[0]).abs() < 1e-5, "{j}");
        }
        let (edge, k) = t.at(320.0);
        let (far, _) = t.at(420.0);
        assert!((far[1] - edge[1] - 100.0 * k[1]).abs() < 1e-9);
        let deep = 1.1 * 1.1 / 9.81;
        assert!(k[1] > deep && k[1] < 1.2 * deep, "{} {deep}", k[1]);
        assert!(TAU / k[1] > 30.0);
    }
}
