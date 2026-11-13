//! The level surfaces reflection rays meet analytically instead of in the
//! acceleration structure or on screen: a planet's sea, a shell about the
//! planet's center, and a flat world's water rectangles. Only the see-through
//! pass draws them, so no traced or screen-space ray could otherwise see
//! them. A ray that reaches one stops on it, and whatever lies beneath it
//! stays hidden.

use crate::environment::planet::PlanetFrame;
use crate::environment::water::{DistantLook, WaterBody};
use crate::gfx::render_types::{LevelLayerGpu, MAX_LEVEL_LAYERS};
use alloc::vec::Vec;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::{narrow_f32, sqrt, vec3};

/// The GPU records of no layer at all.
pub const NO_LAYERS: [LevelLayerGpu; MAX_LEVEL_LAYERS] = [LevelLayerGpu {
    shape: [0.0; 4],
    extent: [0.0; 4],
    body: [0.0; 4],
    surface: [0.0; 4],
}; MAX_LEVEL_LAYERS];

/// Where a level surface lies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerShape {
    /// The shell of a sphere: a planet's sea, in the planet's authored frame.
    Shell {
        /// The center, in meters.
        center: DVec3,
        /// The radius, in meters.
        radius: f64,
    },
    /// A horizontal rectangle in a world without a planet.
    Rectangle {
        /// The center.
        center: [f32; 3],
        /// Half its extent along x and z.
        half_extent: [f32; 2],
    },
}

/// A level surface as reflections see it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionLayer {
    /// Where it lies.
    pub shape: LayerShape,
    /// How it looks from afar.
    pub look: DistantLook,
}

/// A world's level surfaces, resolved once when it loads: its planet's sea,
/// or, in a world without a planet, its largest water rectangles, at most
/// [`MAX_LEVEL_LAYERS`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReflectionLayers {
    layers: Vec<ReflectionLayer>,
}

impl ReflectionLayers {
    /// The layers of the visible `bodies`, each looking like its entry in
    /// `looks` ([`WaterBody::distant_look`]).
    pub fn resolve(bodies: &[WaterBody], looks: &[DistantLook]) -> Self {
        let visible = || {
            bodies
                .iter()
                .zip(looks)
                .filter(|(body, _)| body.surface.visible)
        };
        let sea = visible().find_map(|(body, look)| {
            let sphere = body.sea.as_ref()?.sphere;
            Some(ReflectionLayer {
                shape: LayerShape::Shell {
                    center: sphere.center,
                    radius: sphere.radius,
                },
                look: *look,
            })
        });
        let planet = bodies.iter().any(|body| body.sea.is_some());
        let mut layers: Vec<ReflectionLayer> = match sea {
            Some(sea) => alloc::vec![sea],
            None if planet => Vec::new(),
            None => visible()
                .map(|(body, look)| ReflectionLayer {
                    shape: LayerShape::Rectangle {
                        center: body.surface.center,
                        half_extent: body.surface.extent,
                    },
                    look: *look,
                })
                .collect(),
        };
        if layers.len() > MAX_LEVEL_LAYERS {
            layers.sort_by(|a, b| area(b).total_cmp(&area(a)));
            tracing::warn!(
                "reflections meet only the {MAX_LEVEL_LAYERS} largest of {} water surfaces",
                layers.len()
            );
            layers.truncate(MAX_LEVEL_LAYERS);
        }
        Self { layers }
    }

    /// The layers' GPU records for a camera at simulated `camera`, each moved
    /// by `offset` into the space the frame draws in. A shell needs `planet`,
    /// whose frame places it; without one it is left out.
    pub fn records(
        &self,
        planet: Option<&PlanetFrame>,
        camera: [f32; 3],
        offset: [f32; 3],
    ) -> [LevelLayerGpu; MAX_LEVEL_LAYERS] {
        let mut out = NO_LAYERS;
        let placed = self
            .layers
            .iter()
            .filter_map(|l| l.record(planet, camera, offset));
        for (slot, record) in out.iter_mut().zip(placed) {
            *slot = record;
        }
        out
    }
}

impl ReflectionLayer {
    // This layer's record for a camera at simulated `camera`, moved by
    // `offset` into the drawn space, with the camera's height over it formed
    // in double precision.
    fn record(
        &self,
        planet: Option<&PlanetFrame>,
        camera: [f32; 3],
        offset: [f32; 3],
    ) -> Option<LevelLayerGpu> {
        let (shape, extent) = match self.shape {
            LayerShape::Shell { center, radius } => {
                let frame = &planet?.frame;
                let [x, y, z] = vec3::add(frame.to_local(center), offset);
                let reach = dvec3::length(dvec3::sub(frame.to_world(camera), center));
                let height = narrow_f32(reach - radius);
                ([x, y, z, narrow_f32(radius)], [0.0, 0.0, height, 1.0])
            }
            LayerShape::Rectangle {
                center,
                half_extent: [hx, hz],
            } => {
                let [x, y, z] = vec3::add(center, offset);
                ([x, y, z, 0.0], [hx, hz, camera[1] - center[1], 1.0])
            }
        };
        let look = &self.look;
        let [r, g, b] = look.body;
        Some(LevelLayerGpu {
            shape,
            extent,
            body: [r, g, b, look.fresnel_power],
            surface: [sqrt(look.alpha), look.alpha, 0.0, 0.0],
        })
    }
}

// How much of the world a layer covers, the larger kept first.
fn area(layer: &ReflectionLayer) -> f32 {
    match layer.shape {
        LayerShape::Shell { .. } => f32::INFINITY,
        LayerShape::Rectangle {
            half_extent: [x, z],
            ..
        } => x * z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::WaterSurface;
    use crate::environment::planet::LocalFrame;
    use crate::environment::water::PlanetWater;
    use crate::environment::water::ocean::OceanQuality;
    use crate::environment::wind::WindField;

    fn look() -> DistantLook {
        DistantLook {
            body: [0.1, 0.2, 0.3],
            fresnel_power: 5.0,
            alpha: 0.25,
        }
    }

    fn pool(center: [f32; 3], extent: [f32; 2]) -> WaterBody {
        let surface = WaterSurface {
            center,
            extent,
            ..WaterSurface::default()
        };
        WaterBody::resolve(&surface, &WindField::STILL)
    }

    #[test]
    fn a_planet_sea_is_one_shell_placed_by_the_frame() {
        let planet = crate::components::Planet::default().shape();
        let surface = WaterSurface::default();
        let sea = PlanetWater {
            planet: planet.clone(),
            level: 2.0,
            surface: &surface,
        };
        let (bodies, plan) = WaterBody::resolve_all(
            core::iter::empty(),
            &WindField::STILL,
            OceanQuality::default(),
            Some(sea),
        );
        let looks: Vec<DistantLook> = bodies
            .iter()
            .map(|b| b.distant_look(&plan.simulations))
            .collect();
        let layers = ReflectionLayers::resolve(&bodies, &looks);
        assert_eq!(layers.layers.len(), 1);
        assert_eq!(layers.layers[0].look, looks[0]);

        let frame = PlanetFrame {
            shape: planet.clone(),
            frame: LocalFrame::AUTHORED,
        };
        let top = narrow_f32(planet.center[1] + planet.radius);
        let camera = [0.0, top + 12.0, 0.0];
        let records = layers.records(Some(&frame), camera, [0.0, 1.0, 0.0]);
        let shell = records[0];
        assert_eq!(shell.shape[3], narrow_f32(planet.radius + 2.0));
        assert_eq!(shell.shape[1], narrow_f32(planet.center[1]) + 1.0);
        assert!((shell.extent[2] - 10.0).abs() < 0.01, "{}", shell.extent[2]);
        assert_eq!(shell.extent[3], 1.0);
        assert_eq!(records[1..], NO_LAYERS[1..]);
        // A shell has nowhere to be without its planet.
        assert_eq!(layers.records(None, camera, [0.0; 3]), NO_LAYERS);
    }

    #[test]
    fn a_flat_world_keeps_its_largest_rectangles() {
        let bodies: Vec<WaterBody> = (1..=6)
            .map(|i| pool([i as f32, 0.5, 0.0], [i as f32, 1.0]))
            .collect();
        let looks = alloc::vec![look(); bodies.len()];
        let layers = ReflectionLayers::resolve(&bodies, &looks);
        assert_eq!(layers.layers.len(), MAX_LEVEL_LAYERS);
        let widths: Vec<f32> = layers
            .layers
            .iter()
            .map(|l| match l.shape {
                LayerShape::Rectangle { half_extent, .. } => half_extent[0],
                LayerShape::Shell { .. } => 0.0,
            })
            .collect();
        assert_eq!(widths, [6.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn hidden_water_is_no_layer() {
        let mut hidden = pool([0.0; 3], [9.0, 9.0]);
        hidden.surface.visible = false;
        let bodies = [hidden, pool([0.0; 3], [1.0, 1.0])];
        let layers = ReflectionLayers::resolve(&bodies, &[look(), look()]);
        assert_eq!(layers.layers.len(), 1);
        assert!(matches!(
            layers.layers[0].shape,
            LayerShape::Rectangle {
                half_extent: [1.0, 1.0],
                ..
            }
        ));
    }

    #[test]
    fn a_rectangle_record_carries_its_bounds_and_the_camera_height() {
        let layer = ReflectionLayer {
            shape: LayerShape::Rectangle {
                center: [1.0, 2.0, 3.0],
                half_extent: [4.0, 5.0],
            },
            look: look(),
        };
        let record = layer
            .record(None, [0.0, 7.0, 0.0], [1.0, 1.0, 1.0])
            .expect("a rectangle needs no planet");
        assert_eq!(record.shape, [2.0, 3.0, 4.0, 0.0]);
        assert_eq!(record.extent, [4.0, 5.0, 5.0, 1.0]);
        assert_eq!(record.body, [0.1, 0.2, 0.3, 5.0]);
        assert_eq!(record.surface, [0.5, 0.25, 0.0, 0.0]);
    }
}
