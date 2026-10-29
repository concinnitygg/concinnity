//! The `EnvironmentMap` component, and what the renderer reads of an
//! environment map beside its baked lighting.

use alloc::vec::Vec;

use super::EnvironmentMapTable;
use crate::blob::{FrameError, decode_exact};
use crate::ecs::PayloadLocator;
use alloc::string::String;

/// The runtime fields an environment map's resource record carries beside its
/// baked payload. Only what the renderer reads: the authored source and bake
/// settings stay with the cook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentMapRecord {
    /// Draw the map as the background (see [`EnvironmentMap::background`]).
    pub background: bool,
}

impl Default for EnvironmentMapRecord {
    fn default() -> Self {
        Self::from(&EnvironmentMap::default())
    }
}

impl From<&EnvironmentMap> for EnvironmentMapRecord {
    fn from(map: &EnvironmentMap) -> Self {
        Self {
            background: map.background,
        }
    }
}

impl EnvironmentMapRecord {
    /// The record's bytes, as a resource record's `data_bytes` carries them.
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap_or_default()
    }

    /// Decode a resource record's `data_bytes`. A record with no runtime
    /// fields holds the defaults.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, FrameError> {
        match bytes {
            [] => Ok(Self::default()),
            _ => decode_exact(bytes),
        }
    }
}

impl EnvironmentMapTable {
    /// The runtime fields of the map at `handle`; the defaults when there is
    /// no such map.
    ///
    /// # Errors
    ///
    /// Returns an error if the record's bytes do not decode.
    pub fn record(&self, handle: usize) -> Result<EnvironmentMapRecord, FrameError> {
        self.0.get(handle).map_or_else(
            || Ok(EnvironmentMapRecord::default()),
            |e| EnvironmentMapRecord::from_bytes(&e.data_bytes),
        )
    }
}

/// A baked lighting environment built from an equirectangular source (or a
/// built-in generator). It provides the scene's ambient image-based lighting
/// (soft diffuse fill plus glossy reflections that follow surface roughness)
/// and the on-screen sky.
///
/// **Source formats:** a Radiance `.hdr`, or a panorama-sphere `.glb` /
/// `.gltf` -- the packaging where an environment image is painted on the
/// emissive channel of a sphere you stand inside. `cn add` recognizes the
/// latter and produces an `EnvironmentMap` instead of scene geometry.
///
/// **Dynamic range:** a `.hdr` carries real radiance, so its sun can be
/// thousands of times brighter than the sky and bakes into a bright key light
/// with a hot specular highlight. A panorama inside a `.glb` is a display
/// image whose brightest value is white; it is read literally, with the sRGB
/// curve inverted and white landing at 1.0 radiance. That makes it an exact
/// backdrop and a soft, low-contrast fill light, never a key light. Raise
/// [`IndirectLighting`](#indirectlighting)'s `ambient_intensity` to lift the
/// level rather than expecting the bake to invent range the file lacks.
///
/// **`prefilter_face_size` note:** this controls both the reflection detail and
/// the on-screen sky sharpness. 512 is the default balance: 256 visibly
/// pixelates a 4K-source sky, 1024 sharpens it further at 4× the size.
///
/// **Built-in generators:** `sky` produces a procedural blue sky with a soft
/// sun, and `stars` a near-black night sky of scattered points, the brightest
/// of them above 1.0 radiance so bloom catches them, darker below the
/// horizon. Both are useful when no source file is available. The
/// sky is drawn at the resolution `prefilter_face_size` sets, so a starfield
/// wants the largest of them, 1024; below that each point is magnified into a
/// blob.
///
/// The map is also the world's background: every pixel no surface covers
/// shows it along that pixel's view direction, at infinite distance, so it
/// never clips against the camera's far plane and never moves as the camera
/// travels. Set `background` to `false` to use the map for image-based
/// lighting only; the background is then
/// [Background](#background)'s `color`, as it is in a world
/// with no `EnvironmentMap`. A world uses one map; extra declarations are
/// ignored.
///
/// ```rust
/// # use concinnity_engine::components::EnvironmentMap;
/// EnvironmentMap {
///     source: "assets/hdri/studio.hdr".into(),
///     ..Default::default()
/// };
/// // Light with the map but keep the clear color behind the scene.
/// EnvironmentMap {
///     generator: "sky".into(),
///     background: false,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct EnvironmentMap {
    /// Path to the source equirectangular panorama -- a Radiance `.hdr`, or a
    /// panorama-sphere `.glb` / `.gltf` -- relative to the project root.
    /// Mutually exclusive with `generator`.
    pub source: String,
    /// Built-in source name, `sky` or `stars`. Mutually exclusive with
    /// `source`.
    pub generator: String,
    /// Face size of the reflection/sky cubemap, in pixels. Higher is sharper
    /// but larger.
    #[asset(default = 512)]
    pub prefilter_face_size: u32,
    /// Face size of the diffuse ambient cubemap, in pixels.
    #[asset(default = 8)]
    pub irradiance_face_size: u32,
    /// Number of samples used to filter each reflection texel. Higher reduces
    /// noise at the cost of build time.
    #[asset(default = 1024)]
    pub prefilter_samples: u32,
    /// Upper bound on how bright a single source texel may count while building
    /// the glossy reflection mips. A clear-sky HDR holds a few sun or sky
    /// texels thousands of times brighter than their surroundings; left
    /// unbounded they survive into the small (coarse) reflection mips as lone
    /// hot texels and smear across glossy floors as hard bright squares. This
    /// caps each sampled texel so that energy spreads smoothly across the
    /// reflection instead. It affects reflections only, never the on-screen
    /// sky. Set to `0` to disable (no cap); lower values clamp harder.
    #[asset(default = 12.0)]
    pub prefilter_clamp: f32,
    /// Draw the map as the background behind all geometry. `false` lights
    /// the world with the map but leaves the background to `clear_color`.
    #[asset(default = true)]
    pub background: bool,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;
    use alloc::vec;

    fn table(data_bytes: Vec<u8>) -> EnvironmentMapTable {
        EnvironmentMapTable(vec![ResourceEntry {
            data_bytes,
            ..Default::default()
        }])
    }

    #[test]
    fn a_record_round_trips_through_its_bytes() {
        for background in [true, false] {
            let record = EnvironmentMapRecord { background };
            assert_eq!(
                EnvironmentMapRecord::from_bytes(&record.to_bytes()),
                Ok(record)
            );
        }
    }

    #[test]
    fn the_default_record_draws_the_background_as_the_asset_does() {
        assert!(EnvironmentMapRecord::default().background);
        assert!(EnvironmentMap::default().background);
    }

    #[test]
    fn a_record_without_runtime_fields_holds_the_defaults() {
        assert_eq!(
            table(Vec::new()).record(0),
            Ok(EnvironmentMapRecord::default())
        );
        assert_eq!(
            EnvironmentMapTable::default().record(0),
            Ok(EnvironmentMapRecord::default())
        );
    }

    // Bytes that are not a record are a broken build, not a default.
    #[test]
    fn undecodable_bytes_are_an_error() {
        assert!(table(vec![7]).record(0).is_err());
        let mut long = EnvironmentMapRecord::default().to_bytes();
        long.push(0);
        assert_eq!(table(long).record(0), Err(FrameError::Trailing(1)));
    }
}
