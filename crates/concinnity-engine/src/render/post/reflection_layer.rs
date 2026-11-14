//! The level surfaces reflection rays meet analytically instead of in the
//! acceleration structure or on screen: a planet's sea, a shell about the
//! planet's center, and a flat world's water rectangles. Only the see-through
//! pass draws them, so no traced or screen-space ray could otherwise see
//! them. A ray that reaches one stops on it, and whatever lies beneath it
//! stays hidden. A shell whose waves are simulated carries them too, laid out
//! each frame the way the water draws them.

use crate::environment::planet::PlanetFrame;
use crate::environment::water::{DistantLook, WaterBody};
use crate::gfx::render_types::{
    LEVEL_WAVE_FACES, LevelLayerGpu, LevelSurfacesGpu, LevelWavesGpu, MAX_LEVEL_LAYERS,
};
use crate::render::uniforms::WaterOceanGpu;
use alloc::vec::Vec;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::{narrow_f32, sqrt, vec3};

/// The GPU record of no layer and no waves at all.
pub const NO_LEVEL_SURFACES: LevelSurfacesGpu = LevelSurfacesGpu {
    layers: [LevelLayerGpu {
        shape: [0.0; 4],
        extent: [0.0; 4],
        body: [0.0; 4],
        surface: [0.0; 4],
    }; MAX_LEVEL_LAYERS],
    waves: LevelWavesGpu {
        anchor: [0.0; 4],
        shape: [0.0; 4],
        layers: [0; 4],
        inv_lengths: [0.0; 4],
        face_rows: [[0.0; 4]; LEVEL_WAVE_FACES * 3],
        face_offsets: [[0.0; 4]; LEVEL_WAVE_FACES * 2],
    },
};

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
#[derive(Debug, Clone, Default)]
pub struct ReflectionLayers {
    layers: Vec<ReflectionLayer>,
    // The visible planet sea whose waves a simulated ocean raises.
    waves: Option<WaterBody>,
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
        let waves = visible()
            .map(|(body, _)| body)
            .find(|body| body.sea.is_some() && body.ocean.is_some())
            .cloned();
        if layers.len() > MAX_LEVEL_LAYERS {
            layers.sort_by(|a, b| area(b).total_cmp(&area(a)));
            tracing::warn!(
                "reflections meet only the {MAX_LEVEL_LAYERS} largest of {} water surfaces",
                layers.len()
            );
            layers.truncate(MAX_LEVEL_LAYERS);
        }
        Self { layers, waves }
    }

    /// The layers' GPU records for a camera at simulated `camera`, each moved
    /// by `offset` into the space the frame draws in, and the shell's waves
    /// `time` seconds into the world. A shell needs `planet`, whose frame
    /// places it; without one it is left out.
    pub fn frame(
        &self,
        planet: Option<&PlanetFrame>,
        camera: [f32; 3],
        offset: [f32; 3],
        time: f64,
    ) -> LevelSurfacesGpu {
        let mut out = NO_LEVEL_SURFACES;
        let placed = self
            .layers
            .iter()
            .filter_map(|l| l.record(planet, camera, offset));
        for (slot, record) in out.layers.iter_mut().zip(placed) {
            *slot = record;
        }
        if let Some((body, planet)) = self.waves.as_ref().zip(planet)
            && let Some(waves) = waves_record(body, planet, camera, offset, time)
        {
            out.waves = waves;
        }
        out
    }
}

// The waves of the planet sea `body` for a camera at simulated `camera`,
// `time` seconds into the world, laid out from the anchor the water draws
// from and moved by `offset` into the drawn space; `None` for a body without
// a simulated ocean.
fn waves_record(
    body: &WaterBody,
    planet: &PlanetFrame,
    camera: [f32; 3],
    offset: [f32; 3],
    time: f64,
) -> Option<LevelWavesGpu> {
    let ocean = body.ocean.as_ref()?;
    let sphere = body.sea.as_ref()?.sphere;
    let motion = body.sea_motion_at(time, &planet.frame, camera)?;
    let mut waves = NO_LEVEL_SURFACES.waves;
    let [x, y, z] = vec3::add(motion.anchor, offset);
    waves.anchor = [x, y, z, 1.0];
    waves.shape = [
        narrow_f32(1.0 / sphere.radius),
        narrow_f32(sphere.blend_rate()),
        motion.reach,
        body.fine_slope_variance() + body.summed_slope_variance(),
    ];
    let cascades = &ocean.cascades;
    waves.layers = [
        u32::try_from(cascades.count()).unwrap_or(0),
        ocean.first_layer,
        cascades.size(),
        0,
    ];
    for (slot, length) in waves.inv_lengths.iter_mut().zip(cascades.lengths()) {
        *slot = 1.0 / length;
    }
    for (f, face) in motion.faces.iter().enumerate() {
        waves.face_rows[f * 3..f * 3 + 3].copy_from_slice(&face.rows);
        let offsets = WaterOceanGpu::offsets_lane(&face.motion.ocean_offsets);
        waves.face_offsets[f * 2..f * 2 + 2].copy_from_slice(&offsets);
    }
    Some(waves)
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
        let (shape, extent, relief) = match self.shape {
            LayerShape::Shell { center, radius } => {
                let planet = planet?;
                let frame = &planet.frame;
                let [x, y, z] = vec3::add(frame.to_local(center), offset);
                let reach = dvec3::length(dvec3::sub(frame.to_world(camera), center));
                let height = narrow_f32(reach - radius);
                let top = planet.shape.radius + planet.shape.highest_height();
                let relief = narrow_f32(top - radius).max(0.0);
                (
                    [x, y, z, narrow_f32(radius)],
                    [0.0, 0.0, height, 1.0],
                    relief,
                )
            }
            LayerShape::Rectangle {
                center,
                half_extent: [hx, hz],
            } => {
                let [x, y, z] = vec3::add(center, offset);
                ([x, y, z, 0.0], [hx, hz, camera[1] - center[1], 1.0], 0.0)
            }
        };
        let look = &self.look;
        let [r, g, b] = look.body;
        Some(LevelLayerGpu {
            shape,
            extent,
            body: [r, g, b, look.fresnel_power],
            surface: [sqrt(look.alpha), look.alpha, relief, 0.0],
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
    use crate::render::environment::water::frame::WaterMotion;

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
        let records = layers
            .frame(Some(&frame), camera, [0.0, 1.0, 0.0], 0.0)
            .layers;
        let shell = records[0];
        assert_eq!(shell.shape[3], narrow_f32(planet.radius + 2.0));
        assert_eq!(shell.shape[1], narrow_f32(planet.center[1]) + 1.0);
        assert!((shell.extent[2] - 10.0).abs() < 0.01, "{}", shell.extent[2]);
        assert_eq!(shell.extent[3], 1.0);
        // The relief reaches the hilltops from the sea's shell.
        let hills = planet.radius + planet.highest_height();
        assert!((f64::from(shell.surface[2]) - (hills - (planet.radius + 2.0))).abs() < 0.01);
        assert_eq!(records[1..], NO_LEVEL_SURFACES.layers[1..]);
        // A shell has nowhere to be without its planet.
        assert_eq!(layers.frame(None, camera, [0.0; 3], 0.0), NO_LEVEL_SURFACES);
    }

    // A windy planet sea, simulated, and its planet.
    fn windy_sea() -> (crate::environment::planet::PlanetShape, Vec<WaterBody>) {
        let planet = crate::components::Planet::default().shape();
        let surface = WaterSurface {
            fetch_meters: 1.0e5,
            ..WaterSurface::default()
        };
        let wind = WindField::new(Some(&crate::components::Wind {
            strength: 9.0,
            ..crate::components::Wind::default()
        }));
        let sea = PlanetWater {
            planet: planet.clone(),
            level: 0.0,
            surface: &surface,
        };
        let (bodies, _) = WaterBody::resolve_all(
            core::iter::empty(),
            &wind,
            OceanQuality::default(),
            Some(sea),
        );
        (planet, bodies)
    }

    #[test]
    fn a_simulated_sea_carries_its_waves_as_the_water_lays_them() {
        assert_eq!(
            LEVEL_WAVE_FACES,
            crate::render::environment::water::sea::SEA_FACES
        );
        let (planet, bodies) = windy_sea();
        let looks = alloc::vec![look(); bodies.len()];
        let layers = ReflectionLayers::resolve(&bodies, &looks);
        let frame = PlanetFrame {
            shape: planet.clone(),
            frame: LocalFrame::AUTHORED,
        };
        let top = narrow_f32(planet.center[1] + planet.radius);
        let camera = [3.0, top + 2.0, -4.0];
        let waves = layers.frame(Some(&frame), camera, [0.0; 3], 12.5).waves;
        assert_eq!(waves.anchor[3], 1.0);
        let body = &bodies[0];
        let ocean = body.ocean.as_ref().expect("a windy open sea is simulated");
        assert_eq!(waves.layers[0] as usize, ocean.cascades.count());
        assert_eq!(waves.layers[1], ocean.first_layer);
        // A sea sums no swell, so its finest slopes are all the record adds to
        // the cascades.
        assert!(body.waves.is_empty());
        assert_eq!(
            waves.shape[3],
            body.fine_slope_variance() + body.summed_slope_variance()
        );
    }

    #[test]
    fn the_record_follows_the_water_across_a_rebase() {
        let (planet, bodies) = windy_sea();
        let looks = alloc::vec![look(); bodies.len()];
        let layers = ReflectionLayers::resolve(&bodies, &looks);
        let top = narrow_f32(planet.center[1] + planet.radius);
        let at = [700.0, top, -250.0];
        let up = planet.up_at(LocalFrame::AUTHORED.to_world(at));
        let (frame, rebase) = LocalFrame::AUTHORED.recentered(at, up);
        let mut water = WaterMotion::new(&bodies[0], false);
        water.rebase(&rebase);

        let camera = [1.5, 2.5, -0.5];
        let time = 31.25;
        let params = water.params_at(time, camera);
        let planet = PlanetFrame {
            shape: planet,
            frame,
        };
        let waves = layers.frame(Some(&planet), camera, [0.0; 3], time).waves;
        // The water carries its frame through each move in single precision
        // and the planet's frame is kept in double, so the two agree to well
        // under a millimeter rather than bit for bit.
        let near = |a: &[f32], b: &[f32], tolerance: f32| {
            a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
        };
        assert!(near(&waves.anchor[..3], &params.anchor[..3], 1e-4));
        assert_eq!(waves.shape[0], params.sea.shape[0]);
        assert_eq!(waves.shape[1], params.sea.shape[1]);
        assert!(near(&waves.shape[2..3], &params.sea.tangent_v[3..4], 1e-3));
        for (f, face) in params.sea.faces.iter().enumerate() {
            for (ours, theirs) in waves.face_rows[f * 3..f * 3 + 3].iter().zip(&face.rows) {
                assert!(near(ours, theirs, 1e-5), "{ours:?} vs {theirs:?}");
            }
            for (ours, theirs) in waves.face_offsets[f * 2..f * 2 + 2]
                .iter()
                .zip(&face.ocean_offsets)
            {
                assert!(near(ours, theirs, 1e-5), "{ours:?} vs {theirs:?}");
            }
        }
        assert_eq!(waves.inv_lengths, params.ocean.inv_lengths);
        assert_eq!(waves.layers[..3], params.ocean.layers[..3]);
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
