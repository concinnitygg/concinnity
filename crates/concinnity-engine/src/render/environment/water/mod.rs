//! Water as the backends draw it: the per-frame motion and parameters of
//! each resolved [`WaterBody`], its grid, and the shader mirrors that pin the
//! GPU's water to the host's.

pub mod frame;
pub mod motion;
pub mod sea;

#[cfg(test)]
pub(crate) mod fields;
#[cfg(test)]
mod shader_tests;

use crate::gfx::mesh_payload::Vertex;
use crate::water::{WaterBody, cell_size};
use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::narrow_f32;

impl WaterBody {
    /// Where the body sorts among the transparent draws, farthest first: a
    /// rectangle's center, and a planet sea's under everything else, at the
    /// planet's center.
    pub fn sort_center(&self) -> [f32; 3] {
        match &self.sea {
            Some(sea) => sea.sphere.center.map(narrow_f32),
            None => self.surface.center,
        }
    }

    /// `[inner cell, inner half-width, widening, _]` of the grid the body is
    /// drawn on: an ocean's widening grid, or the surface's uniform one.
    pub fn grid_lane(&self) -> [f32; 4] {
        if let Some(sea) = &self.sea {
            return sea.grid.gpu_lane();
        }
        match &self.ocean {
            Some(ocean) => ocean.grid.gpu_lane(),
            None => [cell_size(&self.surface), 0.0, 0.0, 0.0],
        }
    }

    /// The grid the body is drawn on, relative to its anchor: the surface's
    /// own tessellation for a summed body, the camera-centered grid for a
    /// simulated one.
    ///
    /// # Errors
    ///
    /// Returns an error if the body's grid cannot be built.
    pub fn mesh(&self) -> Result<(Vec<Vertex>, Vec<u16>), String> {
        if let Some(sea) = &self.sea {
            return Ok(sea.grid.mesh());
        }
        if let Some(ocean) = &self.ocean {
            return Ok(ocean.grid.mesh());
        }
        let s = &self.surface;
        let (verts, indices) = crate::geometry::water_grid::build_water_grid(
            s.extent[0],
            s.extent[1],
            s.subdivisions,
        )?;
        let vertices = verts
            .into_iter()
            .map(|(pos, normal, color, uv)| Vertex {
                pos,
                normal,
                tangent: [1.0, 0.0, 0.0],
                color,
                uv,
            })
            .collect();
        Ok((vertices, indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, WaterWave, Wind};
    use crate::water::ocean::OceanQuality;
    use crate::wind::WindField;
    use alloc::vec;

    fn fresh_breeze() -> WindField {
        WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }))
    }

    fn open_sea(center: [f32; 3]) -> WaterSurface {
        WaterSurface {
            center,
            extent: [4000.0, 3000.0],
            fetch_meters: 1.0e5,
            waves: vec![WaterWave {
                amplitude: 0.4,
                wavelength: 80.0,
                ..WaterWave::default()
            }],
            ..Default::default()
        }
    }

    // A pond's mesh is its own tessellation; an ocean's is the
    // camera-centered grid.
    #[test]
    fn each_path_draws_its_own_grid() {
        let surfaces = [open_sea([0.0; 3]), WaterSurface::default()];
        let (bodies, _) =
            WaterBody::resolve_all(&surfaces, &fresh_breeze(), OceanQuality::FULL, None);
        let (sea_v, sea_i) = bodies[0].mesh().unwrap();
        let ocean = bodies[0].ocean.as_ref().unwrap();
        assert_eq!((sea_v.len(), sea_i.len()), {
            let (v, i) = ocean.grid.mesh();
            (v.len(), i.len())
        });
        let (pond_v, _) = bodies[1].mesh().unwrap();
        assert_eq!(pond_v.len(), 65 * 65);
        let x = pond_v.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max);
        assert_eq!(x, 10.0);
    }
}
