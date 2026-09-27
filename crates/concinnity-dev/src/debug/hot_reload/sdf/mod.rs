//! SdfVolume field hot reload. A save marks every volume whose field is the
//! file; the marked volumes are grouped by field and flags, since the flags
//! decide which entries compile, and each group's field is read on the frame
//! thread and compiled once on a worker (see `compile`). A finished compile is
//! swapped into every volume of its group on the frame thread through
//! `replace_sdf_volume_pipelines`. Volumes are built once, at init, so one the
//! backend does not hold has nothing to swap until the world is next built.

mod compile;

#[cfg(test)]
mod tests;

use concinnity_cook::compile::program::Diagnostic;
use concinnity_core::components::ShaderSource;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{LiveEdit, PipelineSwap};
use concinnity_engine::gfx::system::sdf_field_sources::{SdfFieldEntry, SdfFieldMap};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use super::compile_queue::CompileQueue;
use super::files::FileIndex;
use super::pending::PendingSdfVolumes;
use super::report::{ReloadFailure, ReloadOutcome, ReloadReport, ReloadSubject};
use compile::{FieldKey, FieldResult, read_field};

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
    compiles: CompileQueue<FieldKey, FieldResult>,
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
    // compile. A group whose field cannot be read is reported failed right
    // away, once per volume.
    pub(crate) fn request(&mut self, pending: &PendingSdfVolumes) -> Vec<ReloadReport> {
        let mut failed = Vec::new();
        for (key, volumes) in groups(&self.catalog, pending) {
            let names: Vec<&str> = volumes.iter().map(|v| v.name.as_str()).collect();
            let started = read_field(&key.path).and_then(|field| {
                let compiler = Arc::clone(&self.compiler);
                let name = names[0].to_string();
                let job_key = key.clone();
                self.compiles
                    .submit(key.clone(), move || compiler(&name, &job_key, &field))
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
            for entry in volumes {
                let outcome = match &result {
                    Ok(compiled) => apply(entry, &compiled.programs, &compiled.warnings, backend),
                    Err(e) => ReloadOutcome::Failed(e.clone()),
                };
                reports.push(ReloadReport {
                    subject: ReloadSubject::sdf_volume(&entry.name),
                    outcome,
                });
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

// Hand a freshly compiled field to the backend for one volume.
fn apply(
    entry: &SdfFieldEntry,
    programs: &SdfPrograms,
    warnings: &[Diagnostic],
    backend: &mut dyn LiveEdit,
) -> ReloadOutcome {
    let started = Instant::now();
    match backend.replace_sdf_volume_pipelines(entry.volume, programs) {
        Ok(PipelineSwap::Swapped) => ReloadOutcome::Swapped {
            frame_time: started.elapsed(),
            warnings: warnings.to_vec(),
        },
        Ok(PipelineSwap::NotResident) => ReloadOutcome::AppliesOnLoad {
            warnings: warnings.to_vec(),
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
