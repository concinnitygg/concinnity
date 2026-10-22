//! A CPU mirror of `water_evolve` in `shaders/water_fft.hlsl`: the spectra of
//! the fields one texel contributes in a frame. It pins the shader's sign
//! conventions (which way the waves travel, which way the choppiness pushes)
//! against a single wave summed by hand.

use core::f32::consts::TAU;

type Complex = [f32; 2];

fn mul(a: Complex, b: Complex) -> Complex {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn scale(a: Complex, s: f32) -> Complex {
    [a[0] * s, a[1] * s]
}

// `i z`.
fn times_i(z: Complex) -> Complex {
    [-z[1], z[0]]
}

/// One texel's field spectra, as the shader packs them before the inverse
/// FFT.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TexelFields {
    height: Complex,
    /// Horizontal displacement per unit choppiness, along X and Z.
    dx: Complex,
    dz: Complex,
    /// The displacement's derivatives per unit choppiness: d dx / dx,
    /// d dz / dz and d dx / dz.
    pub(crate) jxx: Complex,
    pub(crate) jzz: Complex,
    jxz: Complex,
}

/// The fields of the texel at wavenumber `k` holding `texel`, whose mirror
/// texel at `-k` holds `mirror`, `time_fraction` into the repeat.
pub(crate) fn evolve(
    texel: [f32; 4],
    mirror: [f32; 4],
    time_fraction: f32,
    k: [f32; 2],
) -> TexelFields {
    let x = texel[2] * time_fraction;
    let phase = TAU * (x - libm::floorf(x));
    let e = [libm::cosf(phase), libm::sinf(phase)];
    let a = mul([texel[0], texel[1]], [e[0], -e[1]]);
    let b = mul([mirror[0], -mirror[1]], e);
    let h = [a[0] + b[0], a[1] + b[1]];
    let kl = libm::hypotf(k[0], k[1]);
    let kn = if kl > 0.0 {
        [k[0] / kl, k[1] / kl]
    } else {
        [0.0; 2]
    };
    TexelFields {
        height: h,
        dx: scale(times_i(h), kn[0]),
        dz: scale(times_i(h), kn[1]),
        jxx: scale(h, -kn[0] * k[0]),
        jzz: scale(h, -kn[1] * k[1]),
        jxz: scale(h, -kn[0] * k[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The real field the inverse FFT makes of a spectrum, at `p`.
    fn at(terms: &[(Complex, [f32; 2])], p: [f32; 2]) -> f32 {
        terms
            .iter()
            .map(|(f, k)| {
                let t = k[0] * p[0] + k[1] * p[1];
                f[0] * libm::cosf(t) - f[1] * libm::sinf(t)
            })
            .sum()
    }

    // One wave, its amplitude at k and none at -k (both texels turn at the
    // same rate, as every texel does with its mirror): the horizontal shift
    // points toward the crest from both sides, and the Jacobian of the
    // horizontal map is least at the crest, so crests are what sharpen and
    // fold.
    #[test]
    fn choppiness_gathers_points_into_the_crest() {
        let amplitude = 0.5;
        let k = [0.12_f32, -0.09];
        let kl = libm::hypotf(k[0], k[1]);
        let kn = [k[0] / kl, k[1] / kl];
        let steps = 7.0;
        let fraction = 0.31;
        let texel = [amplitude / 2.0, 0.0, steps, 0.0];
        let empty = [0.0, 0.0, steps, 0.0];
        let plus = evolve(texel, empty, fraction, k);
        let minus = evolve(empty, texel, fraction, [-k[0], -k[1]]);
        let field = |f: fn(&TexelFields) -> Complex, s: f32| {
            let p = [kn[0] * s, kn[1] * s];
            at(&[(f(&plus), k), (f(&minus), [-k[0], -k[1]])], p)
        };

        // The crest: the height there is the full amplitude.
        let x = steps * fraction;
        let crest = TAU * (x - libm::floorf(x)) / kl;
        assert!((field(|f| f.height, crest) - amplitude).abs() < 1e-4);

        let chop = 0.85;
        let along = |s: f32| kn[0] * field(|f| f.dx, s) + kn[1] * field(|f| f.dz, s);
        let jacobian = |s: f32| {
            let jxx = 1.0 + chop * field(|f| f.jxx, s);
            let jzz = 1.0 + chop * field(|f| f.jzz, s);
            let jxz = chop * field(|f| f.jxz, s);
            jxx * jzz - jxz * jxz
        };
        let wavelength = TAU / kl;
        for frac in [0.05_f32, 0.15, 0.3, 0.45] {
            let d = frac * wavelength;
            assert!(along(crest - d) > 0.0, "behind the crest by {d}");
            assert!(along(crest + d) < 0.0, "ahead of the crest by {d}");
        }
        let lowest = (0..400)
            .map(|i| crest + (i as f32 / 400.0 - 0.5) * wavelength)
            .min_by(|a, b| jacobian(*a).total_cmp(&jacobian(*b)))
            .unwrap();
        assert!(
            (lowest - crest).abs() < wavelength / 100.0,
            "{lowest} vs {crest}"
        );
        assert!(jacobian(crest) < 1.0 && jacobian(crest + wavelength / 2.0) > 1.0);
    }
}
