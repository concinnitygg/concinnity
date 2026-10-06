//! The verbs applied against the render backend: runtime decals and particle
//! emitters, a screenshot of the last presented frame, and the GPU cull's
//! status readback. Each runs once a backend is parked, so one sent before the
//! render loop starts waits for it.
//!
//! `decal-add` and `emitter-add` answer with the stable slot index the matching
//! remove takes. Their fields mirror the `Decal` and `ParticleEmitter` assets.

use concinnity_core::components::ParticleEmitter;
use concinnity_core::render::decal::{self, DecalRecord};
use concinnity_core::render::particles::ParticleEmitterRecord;
use concinnity_engine::live_edit::parked::TextureNameSlots;
use concinnity_host::thread::asset_id;
use serde_json::{Value, json};

use crate::debug::call::{Call, READBACK_TIMEOUT, REPLY_TIMEOUT};
use crate::debug::verb::{Access, Args, Kind, Reply, Verb, optional, required};

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "decal-add",
        description: "Place a decal in the running world and return the slot id that removes it.",
        access: Access::Mutating,
        params: &[
            optional(
                "texture",
                Kind::TextOrNull,
                "Texture asset name; omit for the built-in decal texture.",
            ),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "rotation_deg",
                Kind::Vec3,
                "Euler rotation in degrees. Defaults to no rotation.",
            ),
            optional(
                "size",
                Kind::Vec3,
                "Projection box extents. Defaults to one unit on each axis.",
            ),
            optional(
                "tint",
                Kind::Vec4,
                "Linear RGBA tint. Defaults to opaque white.",
            ),
        ],
        run: decal_add,
    },
    Verb {
        name: "decal-remove",
        description: "Remove a decal previously placed by decal-add.",
        access: Access::Mutating,
        params: &[required(
            "id",
            Kind::Count,
            "Slot id returned by decal-add.",
        )],
        run: decal_remove,
    },
    Verb {
        name: "emitter-add",
        description: "Place a particle emitter in the running world and return the slot id that removes it.",
        access: Access::Mutating,
        params: &[
            optional(
                "texture",
                Kind::TextOrNull,
                "Particle texture asset name; omit for the built-in texture.",
            ),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "direction",
                Kind::Vec3,
                "Emission direction. Defaults to the emitter's own default.",
            ),
            optional(
                "spread_deg",
                Kind::Number,
                "Cone half-angle around the direction, in degrees.",
            ),
            optional(
                "speed_min",
                Kind::Number,
                "Lower bound of the initial particle speed.",
            ),
            optional(
                "speed_max",
                Kind::Number,
                "Upper bound of the initial particle speed.",
            ),
            optional(
                "lifetime_min",
                Kind::Number,
                "Lower bound of the particle lifetime, in seconds.",
            ),
            optional(
                "lifetime_max",
                Kind::Number,
                "Upper bound of the particle lifetime, in seconds.",
            ),
            optional(
                "gravity",
                Kind::Vec3,
                "Constant acceleration applied to every particle.",
            ),
            optional("spawn_rate", Kind::Number, "Particles emitted per second."),
            optional(
                "max_particles",
                Kind::Count,
                "Ceiling on live particles for this emitter.",
            ),
            optional("size_start", Kind::Number, "Particle size at birth."),
            optional("size_end", Kind::Number, "Particle size at death."),
            optional("color_start", Kind::Vec4, "Linear RGBA color at birth."),
            optional("color_end", Kind::Vec4, "Linear RGBA color at death."),
        ],
        run: emitter_add,
    },
    Verb {
        name: "emitter-remove",
        description: "Remove a particle emitter previously placed by emitter-add.",
        access: Access::Mutating,
        params: &[required(
            "id",
            Kind::Count,
            "Slot id returned by emitter-add.",
        )],
        run: emitter_remove,
    },
    Verb {
        name: "screenshot",
        description: "Capture the last presented frame to a PNG file.",
        access: Access::Mutating,
        params: &[required("path", Kind::Name, "Destination PNG path (.png).")],
        run: screenshot,
    },
    Verb {
        name: "cull-status",
        description: "Read the GPU cull's per-object status buffer back and report how many objects were drawn, frustum-culled, Hi-Z-rejected, or redrawn by the disocclusion pass.",
        access: Access::ReadOnly,
        params: &[],
        run: cull_status,
    },
];

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
struct DecalAdd {
    texture: Option<String>,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    size: [f32; 3],
    tint: [f32; 4],
}

impl Default for DecalAdd {
    fn default() -> Self {
        Self {
            texture: None,
            position: [0.0; 3],
            rotation_deg: [0.0; 3],
            size: [1.0; 3],
            tint: [1.0; 4],
        }
    }
}

// The emitter's tunables are the `ParticleEmitter` asset's own fields, defaults
// included; only the texture is named rather than cooked.
#[derive(serde::Deserialize)]
struct EmitterAdd {
    #[serde(default)]
    texture: Option<String>,
    #[serde(flatten)]
    emitter: ParticleEmitter,
}

#[derive(serde::Deserialize)]
struct Slot {
    id: usize,
}

#[derive(serde::Deserialize)]
struct Screenshot {
    path: String,
}

fn decal_add(call: &Call, args: Args) -> Reply {
    let decal: DecalAdd = args.parse()?;
    let id = call.on_backend(REPLY_TIMEOUT, move |backend, slots| {
        let slot = texture_slot(decal.texture.as_deref(), slots)?;
        backend
            .add_decal(decal_record(&decal, slot)?)
            .map_err(|e| e.to_string())
    })?;
    Ok(json!({ "id": id }))
}

fn decal_remove(call: &Call, args: Args) -> Reply {
    let Slot { id } = args.parse()?;
    call.on_backend(REPLY_TIMEOUT, move |backend, _| {
        backend.remove_decal(id).map_err(|e| e.to_string())
    })?;
    Ok(json!({ "removed": true }))
}

fn emitter_add(call: &Call, args: Args) -> Reply {
    let EmitterAdd { texture, emitter } = args.parse()?;
    let id = call.on_backend(REPLY_TIMEOUT, move |backend, slots| {
        let slot = texture_slot(texture.as_deref(), slots)?;
        backend
            .add_emitter(ParticleEmitterRecord::new(&emitter, slot))
            .map_err(|e| e.to_string())
    })?;
    Ok(json!({ "id": id }))
}

fn emitter_remove(call: &Call, args: Args) -> Reply {
    let Slot { id } = args.parse()?;
    call.on_backend(REPLY_TIMEOUT, move |backend, _| {
        backend.remove_emitter(id).map_err(|e| e.to_string())
    })?;
    Ok(json!({ "removed": true }))
}

// The capture idles the GPU, copies the swapchain image back, and encodes the
// PNG on the render thread, so it gets the readback wait. The path is checked
// and written as the same trimmed value.
fn screenshot(call: &Call, args: Args) -> Reply {
    let path = args.parse::<Screenshot>()?.path.trim().to_string();
    let is_png = std::path::Path::new(&path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if !is_png {
        return Err("screenshot: 'path' must end in .png".to_string());
    }
    let saved = call.on_backend(READBACK_TIMEOUT, move |backend, _| {
        backend.screenshot(&path).map_err(|e| e.to_string())
    })?;
    Ok(json!({ "path": saved }))
}

fn cull_status(call: &Call, _: Args) -> Reply {
    let raw = call.on_backend(READBACK_TIMEOUT, |backend, _| {
        backend.read_cull_status().map_err(|e| e.to_string())
    })?;
    Ok(cull_status_reply(&raw))
}

// The outcome histogram of one cull-status readback, plus the two derived
// numbers a Hi-Z A/B compares: everything the cull let through, and everything
// the Hi-Z test rejected across both phases.
fn cull_status_reply(raw: &[u32]) -> Value {
    let c = concinnity_core::gfx::cull_status::tally(raw);
    json!({
        "objects": c.total(),
        "drawn": c.drawn,
        "frustum_culled": c.frustum_culled,
        "hiz_candidate": c.hiz_candidate,
        "hiz_culled": c.hiz_culled,
        "redrawn": c.redrawn,
        "unknown": c.unknown,
        "visible": c.visible(),
        "hiz_rejected": c.hiz_rejected(),
    })
}

fn decal_record(decal: &DecalAdd, texture_slot: usize) -> Result<DecalRecord, String> {
    let model = decal::decal_model_matrix(decal.position, decal.rotation_deg, decal.size);
    let inv_model =
        decal::invert_decal_model(model).ok_or_else(|| "decal-add: degenerate size".to_string())?;
    Ok(DecalRecord {
        model,
        inv_model,
        texture_slot,
        tint: decal.tint,
    })
}

// A texture asset name's slot in the live pool. No name samples the renderer's
// white fallback at slot 0, so the tint or color gradient still shows; a name
// that does not resolve is an error rather than a silent fallback. Names
// resolve through the table graphics init parks under `cn debug`.
fn texture_slot(texture: Option<&str>, slots: Option<&TextureNameSlots>) -> Result<usize, String> {
    let Some(name) = texture else {
        return Ok(0);
    };
    let id =
        asset_id::lookup(name).ok_or_else(|| format!("texture '{name}' not found in interner"))?;
    let slots = slots.ok_or_else(|| {
        "texture-name resolution requires cn debug (texture-name slots not captured)".to_string()
    })?;
    slots
        .0
        .get(&id)
        .copied()
        .ok_or_else(|| format!("texture '{name}' is not in the live texture pool"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use crate::test_support;
    use concinnity_core::ecs::World;
    use concinnity_core::ecs::asset_id::AssetId;

    #[test]
    fn decal_record_inverts_the_projection_box() {
        let decal = DecalAdd {
            position: [1.0, 2.0, 3.0],
            tint: [1.0, 0.0, 0.0, 1.0],
            ..DecalAdd::default()
        };
        let record = decal_record(&decal, 5).expect("a unit box inverts");
        assert_eq!(record.texture_slot, 5);
        assert_eq!(record.tint, [1.0, 0.0, 0.0, 1.0]);
        assert_ne!(record.model, record.inv_model);
    }

    #[test]
    fn a_zero_size_decal_is_refused_before_the_backend() {
        let decal = DecalAdd {
            size: [0.0; 3],
            ..DecalAdd::default()
        };
        assert_eq!(
            decal_record(&decal, 0).err().as_deref(),
            Some("decal-add: degenerate size")
        );
    }

    // A backend without the feature reports it rather than handing out an id.
    #[test]
    fn each_backend_verb_surfaces_the_backends_refusal() {
        let mut engine = Engine::new(World::new());
        let cases = [
            ("decal-add", json!({}), "add_decal"),
            ("decal-remove", json!({ "id": 3 }), "remove_decal"),
            ("emitter-add", json!({}), "add_emitter"),
            ("emitter-remove", json!({ "id": 5 }), "remove_emitter"),
            (
                "screenshot",
                json!({ "path": "shot.png" }),
                "screenshot: not supported",
            ),
            ("cull-status", json!({}), "read_cull_status: not supported"),
        ];
        for (verb, args, needle) in cases {
            let error = engine.call(verb, args).expect_err(verb);
            assert!(error.contains(needle), "{verb}: {error}");
        }
    }

    // The emitter's defaults are the asset's, so a bare call spawns the same
    // emitter an authored one with no fields would.
    #[test]
    fn a_bare_emitter_add_takes_the_asset_defaults() {
        let Value::Object(fields) = json!({ "texture": "sparks", "speed_max": 4 }) else {
            unreachable!()
        };
        let verb = crate::debug::catalog::find("emitter-add").expect("emitter-add");
        let EmitterAdd { texture, emitter } = verb.arguments(fields).unwrap().parse().unwrap();
        let defaults = ParticleEmitter::default();
        assert_eq!(texture.as_deref(), Some("sparks"));
        assert_eq!(emitter.speed_max, 4.0);
        assert_eq!(emitter.direction, defaults.direction);
        assert_eq!(emitter.max_particles, defaults.max_particles);
        assert!(emitter.visible && emitter.texture.is_none());
    }

    #[test]
    fn screenshot_requires_a_png_path() {
        let mut engine = Engine::new(World::new());
        for path in ["/tmp/out.rs", "/tmp/out", " shot.jpg "] {
            assert_eq!(
                engine.call("screenshot", json!({ "path": path })),
                Err("screenshot: 'path' must end in .png".to_string()),
                "{path}"
            );
        }
    }

    // An empty readback is a clean all-zero histogram, so a probe can tell "no
    // objects" from "the command failed".
    #[test]
    fn cull_status_reply_reports_every_outcome() {
        use concinnity_core::gfx::cull_status::CullStatus;
        let raw = [
            CullStatus::DRAWN,
            CullStatus::DRAWN,
            CullStatus::CULLED,
            CullStatus::HIZ_CANDIDATE,
            CullStatus::REDRAW,
            CullStatus::HIZ_CULLED,
        ];
        assert_eq!(
            cull_status_reply(&raw),
            json!({
                "objects": 6,
                "drawn": 2,
                "frustum_culled": 1,
                "hiz_candidate": 1,
                "hiz_culled": 1,
                "redrawn": 1,
                "unknown": 0,
                "visible": 3,
                "hiz_rejected": 2,
            })
        );
        let empty = cull_status_reply(&[]);
        assert_eq!(
            (empty["objects"].clone(), empty["visible"].clone()),
            (json!(0), json!(0))
        );
    }

    #[test]
    fn texture_slot_covers_every_case() {
        let _guard = test_support::lock();
        asset_id::reset_interner();

        assert_eq!(texture_slot(None, None), Ok(0));

        let err = texture_slot(Some("ghost"), None).unwrap_err();
        assert!(err.contains("not found in interner"), "got: {err}");

        asset_id::intern_all(&["grid"]);
        let err = texture_slot(Some("grid"), None).unwrap_err();
        assert!(err.contains("slots not captured"), "got: {err}");

        let empty = TextureNameSlots::default();
        let err = texture_slot(Some("grid"), Some(&empty)).unwrap_err();
        assert!(err.contains("not in the live texture pool"), "got: {err}");

        let slots = TextureNameSlots([(AssetId(0), 5usize)].into());
        assert_eq!(texture_slot(Some("grid"), Some(&slots)), Ok(5));
    }
}
