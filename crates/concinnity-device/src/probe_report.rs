//! How the reflection-probe bake's progress and failures are reported.

use concinnity_engine::render::probe::bake::BakeReport;

/// Log placements dropped because the cube array could not hold them, a probe
/// that installed, and a failure that abandoned the rest of the bake.
pub(crate) fn report_probe_bake(report: BakeReport) {
    if let Some(e) = report.dropped {
        tracing::warn!("reflection probes: {e}; keeping the sky");
    }
    if let Some(progress) = report.installed {
        tracing::info!("reflection probes: {progress}");
    }
    if let Some(failure) = report.failed {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {}",
            failure.kept,
            failure.error
        );
    }
}
