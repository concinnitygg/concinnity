//! SdfVolume field hot reload. A save marks every volume whose field is the
//! file; the marked volumes are grouped by field and flags, since the flags
//! decide which entries compile, and each group's field is read on the frame
//! thread and compiled once on a worker (see `compile`), which also builds each
//! volume's pipelines when the backend offers a builder. The finished pipelines
//! are swapped into every volume of the group on the frame thread through
//! `replace_sdf_volume_pipelines`. Volumes are built once, at init, so one the
//! backend does not hold has nothing to swap until the world is next built.

mod compile;

#[cfg(test)]
mod tests;

use concinnity_core::components::ShaderSource;
use concinnity_core::render::backend::{
    LiveEdit, PipelineBuilder, PipelineSwap, PreparedPipelines,
};
use concinnity_engine::live_edit::sdf_field_sources::{SdfFieldEntry, SdfFieldMap};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use super::compile_queue::CompileQueue;
use super::files::FileIndex;
use super::report::{ReloadFailure, ReloadOutcome, ReloadReport, ReloadSubject};
use super::signals::PendingSdfVolumes;
use compile::{FieldKey, FieldResult, RebuildResult, Rebuilt, read_field};

// Compiles one field for a group of volumes, named by the first; the cook's
// compile outside tests.
type Compiler = Arc<dyn Fn(&str, &FieldKey, &ShaderSource) -> FieldResult + Send + Sync>;

// Every field file with the volume reading it.
pub(super) fn file_index(catalog: &SdfFieldMap) -> FileIndex<String> {
    FileIndex::new(catalog.files().map(|(path, name)| (path, name.to_string())))
}

// The field catalog and the compiles in flight.
pub(crate) struct SdfReload {
    pub(super) catalog: SdfFieldMap,
    compiles: CompileQueue<FieldKey, RebuildResult>,
    compiler: Compiler,
}

impl SdfReload {
    pub(crate) fn new(catalog: SdfFieldMap) -> Self {
        Self::with_compiler(catalog, Arc::new(compile::compile))
    }

    fn with_compiler(catalog: SdfFieldMap, compiler: Compiler) -> Self {
        Self {
            catalog,
            compiles: CompileQueue::new(),
            compiler,
        }
    }

    // Every volume in the catalog, as the report board names it.
    pub(super) fn subjects(&self) -> impl Iterator<Item = ReloadSubject> + '_ {
        self.catalog
            .entries
            .iter()
            .map(|e| ReloadSubject::sdf_volume(&e.name))
    }

    // Read the field of every group of volumes `pending` names and start its
    // compile, which also builds each volume's pipelines through `builder` when
    // there is one. A group whose field cannot be read is reported failed right
    // away, once per volume.
    pub(crate) fn request(
        &mut self,
        pending: &PendingSdfVolumes,
        builder: Option<Arc<dyn PipelineBuilder>>,
    ) -> Vec<ReloadReport> {
        let mut failed = Vec::new();
        for (key, volumes) in groups(&self.catalog, pending) {
            let names: Vec<&str> = volumes.iter().map(|v| v.name.as_str()).collect();
            let started = read_field(&key.path).and_then(|field| {
                let compiler = Arc::clone(&self.compiler);
                let builder = builder.clone();
                let name = names[0].to_string();
                let job_key = key.clone();
                let targets: Vec<(usize, String)> =
                    volumes.iter().map(|v| (v.volume, v.name.clone())).collect();
                self.compiles.submit(key.clone(), move || {
                    compiler(&name, &job_key, &field)
                        .and_then(|c| compile::prepare(c, builder.as_deref(), &job_key, &targets))
                })
            });
            match started {
                Ok(()) => tracing::info!(
                    "SdfVolume hot-reload: recompiling '{}' for {}",
                    key.path,
                    quoted(&names)
                ),
                Err(e) => failed.extend(volumes.iter().map(|v| ReloadReport {
                    subject: ReloadSubject::sdf_volume(&v.name),
                    outcome: ReloadOutcome::Failed(ReloadFailure::Unstarted(e.clone())),
                })),
            }
        }
        failed
    }

    // Swap every compile that finished since the last call into each volume
    // it was compiled for.
    pub(crate) fn poll(&mut self, backend: &mut dyn LiveEdit) -> Vec<ReloadReport> {
        let mut reports = Vec::new();
        for (key, result) in self.compiles.drain() {
            let volumes = self
                .catalog
                .entries
                .iter()
                .filter(|e| FieldKey::of(e) == key);
            match result {
                Ok(mut rebuilt) => {
                    for entry in volumes {
                        let prepared = rebuilt.prepared.remove(&entry.volume);
                        reports.push(ReloadReport {
                            subject: ReloadSubject::sdf_volume(&entry.name),
                            outcome: apply(entry, &rebuilt, prepared, backend),
                        });
                    }
                }
                Err(e) => reports.extend(volumes.map(|entry| ReloadReport {
                    subject: ReloadSubject::sdf_volume(&entry.name),
                    outcome: ReloadOutcome::Failed(e.clone()),
                })),
            }
        }
        reports
    }
}

// The compiles a request asks for: every volume `pending` names, grouped by
// field file and flags.
fn groups<'a>(
    catalog: &'a SdfFieldMap,
    pending: &PendingSdfVolumes,
) -> BTreeMap<FieldKey, Vec<&'a SdfFieldEntry>> {
    let mut groups: BTreeMap<FieldKey, Vec<&SdfFieldEntry>> = BTreeMap::new();
    for entry in catalog.entries.iter().filter(|e| pending.wants(&e.name)) {
        groups.entry(FieldKey::of(entry)).or_default().push(entry);
    }
    groups
}

// Hand a freshly compiled field, and the pipelines built from it for this
// volume, to the backend.
fn apply(
    entry: &SdfFieldEntry,
    rebuilt: &Rebuilt,
    prepared: Option<PreparedPipelines>,
    backend: &mut dyn LiveEdit,
) -> ReloadOutcome {
    let warnings = &rebuilt.compiled.warnings;
    let started = Instant::now();
    match backend.replace_sdf_volume_pipelines(entry.volume, &rebuilt.compiled.programs, prepared) {
        Ok(PipelineSwap::Swapped) => ReloadOutcome::Swapped {
            frame_time: started.elapsed(),
            warnings: warnings.clone(),
        },
        Ok(PipelineSwap::NotResident) => ReloadOutcome::AppliesOnLoad {
            warnings: warnings.clone(),
        },
        Err(e) => ReloadOutcome::Failed(ReloadFailure::Rejected(e.to_string())),
    }
}

// "'a', 'b'".
fn quoted(names: &[&str]) -> String {
    names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ")
}
