//! A row of vents throwing lit plumes: the station that loads the particle
//! path.
//!
//! Stepping the particles is a compute node of its own and costs little; what
//! the plumes cost is drawing them, since a sprite covers many pixels and the
//! plumes stand in front of one another.

use concinnity::components::{ParticleEmitter, PointLight, ProceduralMesh, Prop};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

// The vents the plumes rise from.
const VENTS: usize = 6;
const VENT_SPACING: f32 = 4.6;
const VENT_RADIUS: f32 = 1.1;
const VENT_HEIGHT: f32 = 1.0;

// One plume. The lifetime and the rate together fix how many particles are
// alive at once, and the cap is set above that so the sim is never clipped by
// its own budget.
const SPAWN_RATE: f32 = 2_600.0;
const LIFETIME: [f32; 2] = [3.0, 5.0];
const MAX_PARTICLES: u32 = 16_000;

/// Declare the vents, their plumes, and the light inside each one.
pub(crate) fn declare(world: &mut WorldBuilder, center: [f32; 3]) {
    world.add(
        "particles_vent_mesh",
        ProceduralMesh {
            generator: "cylinder".to_string(),
            radius: Some(VENT_RADIUS),
            height: Some(VENT_HEIGHT),
            segments: Some(24),
            ..Default::default()
        },
    );

    for index in 0..VENTS {
        let along = spread(index, VENTS, VENT_SPACING);
        let at = [center[0] + along * 0.35, 0.0, center[2] + along];
        world
            .add(
                format!("particles_vent_{index}"),
                Prop {
                    position: [at[0], VENT_HEIGHT * 0.5, at[2]],
                    ..Default::default()
                },
            )
            .reference("mesh", "particles_vent_mesh")
            .reference("material", palette::METAL);

        world
            .add(
                format!("particles_plume_{index}"),
                ParticleEmitter {
                    position: [at[0], VENT_HEIGHT, at[2]],
                    direction: [0.0, 1.0, 0.0],
                    spread_deg: 26.0,
                    speed_min: 2.4,
                    speed_max: 5.2,
                    lifetime_min: LIFETIME[0],
                    lifetime_max: LIFETIME[1],
                    gravity: [0.25, -0.35, 0.0],
                    spawn_rate: SPAWN_RATE,
                    max_particles: MAX_PARTICLES,
                    size_start: 0.20,
                    size_end: 1.30,
                    color_start: [1.0, 0.72, 0.34, 0.85],
                    color_end: [0.16, 0.17, 0.20, 0.0],
                    visible: true,
                    ..Default::default()
                },
            )
            .reference("texture", palette::SOOT);

        // A light in the throat of each vent, so the plume is lit from inside
        // rather than reading as flat sprites.
        world.add(
            format!("particles_ember_{index}"),
            PointLight {
                position: [at[0], VENT_HEIGHT + 0.6, at[2]],
                color: [1.0, 0.55, 0.22],
                intensity: 22.0,
                range: 9.0,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // How many particles the station keeps alive once every plume is running.
    fn live_particle_estimate() -> u32 {
        let mean_lifetime = (LIFETIME[0] + LIFETIME[1]) * 0.5;
        (SPAWN_RATE * mean_lifetime) as u32 * VENTS as u32
    }

    // The cap has to sit above what the rate and the lifetime imply, or the
    // emitter silently stops spawning partway up the plume and the station
    // measures a smaller simulation than it declares.
    #[test]
    fn every_plume_has_room_for_the_particles_it_spawns() {
        let per_plume = SPAWN_RATE * LIFETIME[1];
        assert!(
            per_plume < MAX_PARTICLES as f32,
            "{per_plume} particles into a cap of {MAX_PARTICLES}",
        );
    }

    #[test]
    fn the_station_runs_a_simulation_worth_measuring() {
        assert!(live_particle_estimate() > 50_000);
    }
}
