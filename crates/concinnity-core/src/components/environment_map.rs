// Baked image-based lighting environment schema.

use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
use alloc::string::String;

/// A baked lighting environment built from an equirectangular source (or a
/// built-in generator). It provides the scene's ambient image-based lighting
/// (soft diffuse fill plus glossy reflections that follow surface roughness)
/// and the on-screen sky.
///
/// **Source formats:** a Radiance `.hdr`, or a panorama-sphere `.glb` /
/// `.gltf` -- the packaging where an environment image is painted on the
/// emissive channel of a sphere you stand inside. `cn add` recognises the
/// latter and produces an EnvironmentMap instead of scene geometry.
///
/// **Dynamic range:** a `.hdr` carries real radiance, so its sun can be
/// thousands of times brighter than the sky and bakes into a bright key light
/// with a hot specular highlight. A panorama inside a `.glb` is a display
/// image whose brightest value is white; it is read literally, with the sRGB
/// curve inverted and white landing at 1.0 radiance. That makes it an exact
/// backdrop and a soft, low-contrast fill light, never a key light. Raise
/// [PostProcessConfig](#postprocessconfig)'s `ambient_intensity` to lift the
/// level rather than expecting the bake to invent range the file lacks.
///
/// **`prefilter_face_size` note:** this controls both the reflection detail and
/// the on-screen sky sharpness. 512 is the default balance: 256 visibly
/// pixelates a 4K-source sky, 1024 sharpens it further at 4× the size.
///
/// **Built-in generators:** `sky` produces a procedural blue sky with a soft
/// sun, useful when no source file is available.
///
/// The sky mesh that displays the map (a skybox
/// [ProceduralMesh](#proceduralmesh) plus its [Material](#material) and
/// [Prop](#prop)) is injected at world start when the world declares no skybox
/// mesh of its own. Declare an [EngineDefaults](#enginedefaults) with
/// `"sky": false` to use the map for image-based lighting only, with the
/// background left to `clear_color` or your own geometry.
///
/// ```rust
/// # use concinnity_core::components::EnvironmentMap;
/// EnvironmentMap {
///     source: "assets/hdri/studio.hdr".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnvironmentMap {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source equirectangular panorama -- a Radiance `.hdr`, or a
    /// panorama-sphere `.glb` / `.gltf` -- relative to the project root.
    /// Mutually exclusive with `generator`.
    pub source: String,
    /// Built-in source name (e.g. "sky"). Mutually exclusive with `source`.
    pub generator: String,
    /// Face size of the reflection/sky cubemap, in pixels. Higher is sharper
    /// but larger.
    pub prefilter_face_size: u32,
    /// Face size of the diffuse ambient cubemap, in pixels.
    pub irradiance_face_size: u32,
    /// Number of samples used to filter each reflection texel. Higher reduces
    /// noise at the cost of build time.
    pub prefilter_samples: u32,
    /// Upper bound on how bright a single source texel may count while building
    /// the glossy reflection mips. A clear-sky HDR holds a few sun or sky
    /// texels thousands of times brighter than their surroundings; left
    /// unbounded they survive into the small (coarse) reflection mips as lone
    /// hot texels and smear across glossy floors as hard bright squares. This
    /// caps each sampled texel so that energy spreads smoothly across the
    /// reflection instead. It affects reflections only, never the on-screen
    /// sky. Set to `0` to disable (no cap); lower values clamp harder.
    pub prefilter_clamp: f32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

// The face-size / sample-count defaults below are the single source of truth:
// the build pipeline deserialises args through this struct, so a field absent
// from a JSONL entry inherits these values rather than a constant duplicated in
// the build crate. They are chosen for ~32 MB payloads and a few seconds of
// build cost on the dev box. `prefilter_face_size` does double duty: mips 1..N
// feed the GGX specular IBL lookup (fine at low resolution) while mip 0 is
// sampled directly by the skybox sentinel branch in the fragment shaders, so it
// has to be large enough that the displayed sky doesn't look blocky. 512 is the
// balance point; 256 visibly pixelates a 4K HDR sky, 1024 quadruples the payload
// for sharpness only the skybox (not the IBL math) actually uses.
//
// `prefilter_clamp` defaults to a moderate cap rather than off: an unbounded
// clear-sky HDR aliases its sun and bright sky into hard squares on glossy
// floors (the coarse reflection mips hold only a handful of texels, so one hot
// texel paints a whole region). The cap spreads that energy without touching
// the on-screen sky, and a uniform sky below the cap is unchanged.
impl Default for EnvironmentMap {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            generator: String::new(),
            prefilter_face_size: 512,
            irradiance_face_size: 8,
            prefilter_samples: 1024,
            prefilter_clamp: 12.0,
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_size_the_prefilter_for_a_sharp_skybox() {
        // Mip 0 is sampled directly by the skybox branch, so the prefilter face
        // has to be far larger than the irradiance one, which only ever feeds
        // the diffuse convolution.
        let e = EnvironmentMap::default();
        assert_eq!(e.prefilter_face_size, 512);
        assert_eq!(e.irradiance_face_size, 8);
        assert!(e.prefilter_face_size > e.irradiance_face_size);
        assert_eq!(e.prefilter_samples, 1024);
        // The clamp defaults on: an unbounded HDR sun aliases into hard squares
        // in the coarse reflection mips.
        assert_eq!(e.prefilter_clamp, 12.0);
        assert!(e.source.is_empty());
        assert!(e.generator.is_empty());
        assert!(e.locator.is_none());
    }

    #[test]
    fn an_authored_bake_parses_and_round_trips_through_postcard() {
        let e: EnvironmentMap = serde_json::from_str(
            r#"{"source":"sky.hdr","prefilter_face_size":1024,"irradiance_face_size":16,
                "prefilter_samples":512,"prefilter_clamp":0}"#,
        )
        .unwrap();
        assert_eq!(e.source, "sky.hdr");
        assert_eq!(e.prefilter_face_size, 1024);
        // A zero clamp turns the cap off rather than blacking out the sky.
        assert_eq!(e.prefilter_clamp, 0.0);

        let bytes = postcard::to_allocvec(&e).unwrap();
        let back: EnvironmentMap = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.irradiance_face_size, 16);
        assert_eq!(back.prefilter_samples, 512);
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }

    #[test]
    fn a_generated_environment_names_its_generator_instead_of_a_source() {
        let e: EnvironmentMap = serde_json::from_str(r#"{"generator":"gradient_sky"}"#).unwrap();
        assert_eq!(e.generator, "gradient_sky");
        assert!(e.source.is_empty());
    }
}
