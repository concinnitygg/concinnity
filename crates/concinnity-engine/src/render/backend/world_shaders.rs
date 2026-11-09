//! The pipelines a world's own shader programs are built into: the
//! per-bucket world Shader pipelines a pinned scene installs, and the
//! raymarched `SdfVolume` pipelines. Both can be rebuilt from fresh programs
//! while the world runs.

use crate::components::ShaderPrograms;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::backend::{PipelineBuilder, PreparedPipelines};
use crate::render::sdf::programs::SdfPrograms;
use alloc::sync::Arc;

/// What a pipeline rebuild from fresh programs did
/// ([`WorldShaders::update_world_shader`],
/// [`SdfVolumeShaders::replace_sdf_volume_pipelines`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSwap {
    /// The live pipelines were rebuilt and swapped in.
    Swapped,
    /// Nothing the programs feed has a pipeline installed, so nothing was built.
    NotResident,
}

/// The world Shader buckets: installed when a scene that owns one pins,
/// released when it unpins, and rebuilt when the Shader's files change.
///
/// Install and evict default to success, since a backend without per-bucket
/// pipelines has nothing to install. The rebuild defaults to
/// [`RenderError::Unsupported`], so a reload is never reported as applied.
pub trait WorldShaders {
    /// Make one shader bucket's draws renderable by installing its main-pass
    /// pipeline, built from `programs`. Called by the streaming pump when a
    /// scene that exclusively owns the bucket's `Shader` pins; init skipped the
    /// build. `prepared` is the pipeline a [`Self::pipeline_builder`] already
    /// built from `programs` on a worker: the backend installs it as is, or
    /// builds on the calling thread when there is none or it was built against
    /// targets the backend no longer draws into. Replaces whatever the bucket
    /// holds. Bucket 0 is the world default program and is never installed this
    /// way.
    ///
    /// Default no-op-with-Ok: a backend that renders every draw with the world
    /// default program has no per-bucket pipeline to build, and the bucket is
    /// resident as far as scene loading is concerned.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline fails to build.
    fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        let _ = (bucket, programs, prepared);
        Ok(())
    }

    /// Release one shader bucket's render pipeline, undoing
    /// [`Self::install_world_shader`]. Called when the owning scene unpins;
    /// draws carrying the bucket stop rendering until it is installed again.
    /// Default no-op, for the same reason as above.
    fn evict_world_shader(&mut self, bucket: u32) {
        let _ = bucket;
    }

    /// Rebuild one world Shader's pipeline from freshly compiled programs.
    /// Driven by asset hot-reload (`cn run -f` and `cn edit`) when one of the
    /// Shader's files is saved or a `reload-assets` debug tool call fires.
    /// Bucket 0 is the world default program; any other bucket is a
    /// Material-named Shader. The backend builds the replacement first and only
    /// swaps when the build succeeds, so a compile error never overwrites a live
    /// pipeline. A bucket whose pipeline is not installed (its scene is not
    /// loaded) builds nothing and reports [`PipelineSwap::NotResident`].
    /// `prepared` is the pipeline a [`Self::pipeline_builder`] already built
    /// from `programs`; the backend swaps it in rather than building, unless it
    /// was built against targets the backend no longer draws into.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without shader reload,
    /// or an error if the replacement pipeline fails to build.
    fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        let _ = (bucket, programs, prepared);
        Err(RenderError::Unsupported {
            op: "update_world_shader",
        })
    }

    /// A builder that creates world Shader and `SdfVolume` pipelines against this
    /// backend's device on another thread, so a scene's bucket install and a
    /// hot reload's swap receive a finished pipeline instead of building one on
    /// the frame thread. Default `None`: the install builds for itself.
    fn pipeline_builder(&self) -> Option<Arc<dyn PipelineBuilder>> {
        None
    }
}

/// The raymarched `SdfVolume` pipelines, rebuilt when a volume's field changes.
pub trait SdfVolumeShaders {
    /// Rebuild one raymarched `SdfVolume`'s pipelines from a freshly compiled
    /// field. Driven by asset hot-reload (`cn run -f` and `cn edit`) when the
    /// volume's field file is saved. `volume` is the volume's position in
    /// [`WorldFx::sdf_volumes`](crate::render::backend::init::WorldFx::sdf_volumes),
    /// the order the backend built them in. Every pipeline the volume draws
    /// with is rebuilt from `programs`: its surface or volumetric pipeline, and
    /// its shadow caster when it casts one, keeping the flags it was built
    /// with. Every replacement is built before any is swapped in, so a failed
    /// build leaves the live pipelines drawing. A volume the backend holds no
    /// pipelines for builds nothing and reports [`PipelineSwap::NotResident`].
    /// `prepared` is what a
    /// [`WorldShaders::pipeline_builder`] already built from `programs`, taken
    /// on the same terms as in [`WorldShaders::update_world_shader`] and also
    /// only when it was built for the volume's own flags.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without SDF reload, or
    /// an error if a replacement pipeline fails to build.
    fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        let _ = (volume, programs, prepared);
        Err(RenderError::Unsupported {
            op: "replace_sdf_volume_pipelines",
        })
    }
}
