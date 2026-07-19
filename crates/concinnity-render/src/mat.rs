// src/mat.rs
//
// Small column-major matrix and vector helpers shared by the shadow projection
// builders (`csm.rs` for the directional cascades, `spot_shadow.rs` for the spot
// slices). All projections are right-handed with depth mapped to [0, 1], which
// is what Metal, Vulkan, and DirectX shadow sampling all expect, so the matrices
// built here are valid for every backend.

pub const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn look_at(eye: [f32; 3], centre: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(centre, eye));
    let r = normalize3(cross3(f, up));
    let u = cross3(r, f);
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot3(r, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

// Right-handed orthographic projection with depth mapped to [0, 1].
pub fn ortho_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    [
        [2.0 / rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tmb, 0.0, 0.0],
        [0.0, 0.0, -1.0 / fmn, 0.0],
        [
            -(right + left) / rml,
            -(top + bottom) / tmb,
            -near / fmn,
            1.0,
        ],
    ]
}

// Right-handed perspective projection with depth mapped to [0, 1]. `fov_y_rad`
// is the full vertical field of view.
pub fn perspective_rh(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let t = (fov_y_rad * 0.5).tan().max(1e-6);
    let fmn = far - near;
    [
        [1.0 / (aspect * t), 0.0, 0.0, 0.0],
        [0.0, 1.0 / t, 0.0, 0.0],
        [0.0, 0.0, -far / fmn, -1.0],
        [0.0, 0.0, -(far * near) / fmn, 0.0],
    ]
}

pub fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

// An axis not parallel to `dir`, for building a look-at basis. Cone axes are
// commonly straight up or down, where the usual +Y up vector is degenerate.
pub fn up_for(dir: [f32; 3]) -> [f32; 3] {
    if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0_f32; 4];
        for row in 0..4 {
            out[row] = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
        }
        out
    }

    // A point on the near plane maps to depth 0 and one on the far plane to
    // depth 1 after the perspective divide -- the [0, 1] convention every
    // backend's shadow compare assumes.
    #[test]
    fn perspective_maps_near_and_far_to_zero_and_one() {
        let p = perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 50.0);
        let near = transform(p, [0.0, 0.0, -0.1]);
        let far = transform(p, [0.0, 0.0, -50.0]);
        assert!((near[2] / near[3]).abs() < 1e-4);
        assert!(((far[2] / far[3]) - 1.0).abs() < 1e-4);
    }

    // At 90 degrees vertical FOV and square aspect, the frustum edge sits at
    // x = -z, so an edge point lands exactly on the NDC boundary.
    #[test]
    fn perspective_edge_lands_on_the_ndc_boundary() {
        let p = perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 50.0);
        let edge = transform(p, [10.0, 0.0, -10.0]);
        assert!(((edge[0] / edge[3]) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn look_at_puts_the_eye_at_the_origin_looking_down_negative_z() {
        let v = look_at([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let eye = transform(v, [0.0, 5.0, 0.0]);
        assert!(eye[0].abs() < 1e-5 && eye[1].abs() < 1e-5 && eye[2].abs() < 1e-5);
        // The target sits 5 units ahead, i.e. at -5 on the view Z axis.
        let target = transform(v, [0.0, 0.0, 0.0]);
        assert!((target[2] + 5.0).abs() < 1e-5);
    }

    // A straight-up or straight-down axis must not pick a parallel up vector,
    // or the look-at basis collapses.
    #[test]
    fn up_for_avoids_a_degenerate_basis() {
        assert_eq!(up_for([0.0, -1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(up_for([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(up_for([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        // The chosen up is never parallel to the axis.
        for dir in [[0.0, -1.0, 0.0], [0.3, -0.9, 0.2], [1.0, 0.0, 0.0]] {
            let d = normalize3(dir);
            assert!(dot3(d, up_for(d)).abs() < 0.999);
        }
    }
}
