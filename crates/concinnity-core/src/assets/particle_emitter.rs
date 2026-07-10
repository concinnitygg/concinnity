// src/assets/particle_emitter.rs

use crate::assets::ParticleEmitter;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for ParticleEmitter {
    const NAME: &'static str = "ParticleEmitter";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("texture", "Texture")]
    }

    fn from_args(mut args: Self) -> Self {
        // Asset-side floor: keep every authored field in a self-consistent
        // range. The gfx-side `build_particle_records` adds its own clamps
        // for fields that affect GPU buffer sizing.
        args.spread_deg = args.spread_deg.clamp(0.0, 180.0);
        args.speed_min = args.speed_min.max(0.0);
        if !args.speed_max.is_finite() || args.speed_max < args.speed_min {
            args.speed_max = args.speed_min;
        }
        if !args.lifetime_min.is_finite() || args.lifetime_min <= 0.0 {
            args.lifetime_min = 0.001;
        }
        if !args.lifetime_max.is_finite() || args.lifetime_max < args.lifetime_min {
            args.lifetime_max = args.lifetime_min;
        }
        args.spawn_rate = args.spawn_rate.max(0.0);
        args.max_particles = args.max_particles.clamp(1, 65_536);
        args.size_start = args.size_start.max(0.0);
        args.size_end = args.size_end.max(0.0);
        for c in args.color_start.iter_mut().chain(args.color_end.iter_mut()) {
            if !c.is_finite() {
                *c = 0.0;
            }
        }
        args
    }

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for ParticleEmitter {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();
        let tex = arg("texture");
        if !tex.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex.to_string(),
                error: format!(
                    "ParticleEmitter '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex
                ),
            });
        }
        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let p: ParticleEmitter = serde_json::from_str("{}").unwrap();
        assert_eq!(p.position, [0.0, 0.0, 0.0]);
        assert_eq!(p.direction, [0.0, 1.0, 0.0]);
        assert_eq!(p.max_particles, 256);
        assert!(p.visible);
        assert!(p.texture.is_none());
    }

    #[test]
    fn deserialises_with_all_fields() {
        crate::ecs::asset_id::reset_interner();
        let json = r#"{
            "texture":"tex_spark","position":[1,2,3],"direction":[0,1,0],
            "spread_deg":30,"speed_min":1.5,"speed_max":4.0,
            "lifetime_min":0.5,"lifetime_max":1.0,"gravity":[0,-1,0],
            "spawn_rate":60,"max_particles":128,"size_start":0.1,"size_end":0.02,
            "color_start":[1,0.5,0,1],"color_end":[1,0,0,0],"visible":false
        }"#;
        let p: ParticleEmitter = serde_json::from_str(json).unwrap();
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        assert_eq!(p.max_particles, 128);
        assert_eq!(p.color_start, [1.0, 0.5, 0.0, 1.0]);
        assert!(!p.visible);
        assert!(p.texture.is_some());
    }

    #[test]
    fn from_args_clamps_invalid_inputs() {
        let a = ParticleEmitter {
            spread_deg: 300.0,
            speed_min: -1.0,
            speed_max: -5.0,
            lifetime_min: -0.4,
            lifetime_max: -2.0,
            spawn_rate: -10.0,
            max_particles: 0,
            size_start: -0.5,
            size_end: -0.1,
            ..Default::default()
        };
        let n = ParticleEmitter::from_args(a);
        assert_eq!(n.spread_deg, 180.0);
        assert_eq!(n.speed_min, 0.0);
        assert_eq!(n.speed_max, 0.0);
        assert!(n.lifetime_min > 0.0);
        assert!(n.lifetime_max >= n.lifetime_min);
        assert_eq!(n.spawn_rate, 0.0);
        assert_eq!(n.max_particles, 1);
        assert_eq!(n.size_start, 0.0);
        assert_eq!(n.size_end, 0.0);
    }

    #[test]
    fn from_args_lifts_speed_max_to_speed_min() {
        let a = ParticleEmitter {
            speed_min: 5.0,
            speed_max: 2.0,
            ..Default::default()
        };
        let n = ParticleEmitter::from_args(a);
        assert!((n.speed_max - n.speed_min).abs() < 1e-6);
    }

    #[test]
    fn from_args_clamps_max_particles_upper() {
        let a = ParticleEmitter {
            max_particles: 200_000,
            ..Default::default()
        };
        let n = ParticleEmitter::from_args(a);
        assert_eq!(n.max_particles, 65_536);
    }
}
