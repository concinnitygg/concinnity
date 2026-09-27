//! World Shader hot reload. A save marks the Shaders reading the file; each
//! marked Shader's files are read on the frame thread and compiled on a worker
//! (see `compile`); a finished compile is swapped in on the frame thread
//! through `update_world_shader`. A Shader whose scene is not loaded has no
//! pipeline to swap, so its programs wait in the shared `ShaderOverrides` and
//! install when the scene loads.

mod compile;

#[cfg(test)]
mod tests;

use concinnity_cook::compile::program::Diagnostic;
use concinnity_core::components::ShaderPrograms;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::backend::{LiveEdit, PipelineSwap};
use concinnity_engine::gfx::system::parked::ShaderOverrides;
use concinnity_engine::gfx::system::shader_sources::{ShaderSourceEntry, ShaderSourceMap};
use std::sync::Arc;
use std::time::Instant;

use super::compile_queue::CompileQueue;
use super::files::FileIndex;
use super::pending::PendingShaders;
use super::report::{ReloadFailure, ReloadOutcome, ReloadReport, ReloadSubject};
use compile::{CompileResult, ShaderTexts};

// Compiles a Shader's texts; the cook's compile outside tests.
type Compiler = Arc<dyn Fn(&str, &ShaderTexts) -> CompileResult + Send + Sync>;

// Every Shader file with the Shader reading it.
pub(super) fn file_index(catalog: &ShaderSourceMap) -> FileIndex<AssetId> {
    FileIndex::new(catalog.files())
}

// The Shader catalog, the overrides shared with the streaming pump, and the
// compiles in flight.
pub(crate) struct ShaderReload {
    pub(super) catalog: ShaderSourceMap,
    overrides: ShaderOverrides,
    compiles: CompileQueue<AssetId, CompileResult>,
    compiler: Compiler,
}

impl ShaderReload {
    pub(crate) fn new(catalog: ShaderSourceMap, overrides: ShaderOverrides) -> Self {
        Self::with_compiler(catalog, overrides, Arc::new(compile::compile))
    }

    fn with_compiler(
        catalog: ShaderSourceMap,
        overrides: ShaderOverrides,
        compiler: Compiler,
    ) -> Self {
        Self {
            catalog,
            overrides,
            compiles: CompileQueue::new(),
            compiler,
        }
    }

    // Every Shader in the catalog, as the report board names it.
    pub(super) fn subjects(&self) -> impl Iterator<Item = ReloadSubject> + '_ {
        self.catalog
            .entries
            .iter()
            .map(|e| ReloadSubject::shader(&e.name))
    }

    // Read the files of every Shader `pending` names and start its compile.
    // A Shader whose files cannot be read is reported failed right away.
    pub(crate) fn request(&mut self, pending: &PendingShaders) -> Vec<ReloadReport> {
        let mut failed = Vec::new();
        for entry in self.catalog.entries.iter().filter(|e| pending.wants(&e.id)) {
            let started = ShaderTexts::read(entry).and_then(|texts| {
                let compiler = Arc::clone(&self.compiler);
                let name = entry.name.clone();
                self.compiles
                    .submit(entry.id, move || compiler(&name, &texts))
            });
            match started {
                Ok(()) => tracing::info!("Shader hot-reload: recompiling '{}'", entry.name),
                Err(e) => failed.push(ReloadReport {
                    subject: ReloadSubject::shader(&entry.name),
                    outcome: ReloadOutcome::Failed(ReloadFailure::Unstarted(e)),
                }),
            }
        }
        failed
    }

    // Swap in every compile that finished since the last call.
    pub(crate) fn poll(&mut self, backend: &mut dyn LiveEdit) -> Vec<ReloadReport> {
        self.compiles
            .drain()
            .into_iter()
            .filter_map(|(id, result)| {
                let entry = self.catalog.get(id)?;
                let outcome = match result {
                    Ok(compiled) => apply(
                        entry,
                        compiled.programs,
                        compiled.warnings,
                        backend,
                        &self.overrides,
                    ),
                    Err(e) => ReloadOutcome::Failed(e),
                };
                Some(ReloadReport {
                    subject: ReloadSubject::shader(&entry.name),
                    outcome,
                })
            })
            .collect()
    }
}

// Hand fresh programs to the backend. A Material-named Shader's edit is also
// kept as its override, so a later install (its scene loading, or loading
// again after an unload) builds the edit rather than the cooked programs.
fn apply(
    entry: &ShaderSourceEntry,
    programs: ShaderPrograms,
    warnings: Vec<Diagnostic>,
    backend: &mut dyn LiveEdit,
    overrides: &ShaderOverrides,
) -> ReloadOutcome {
    let programs = Arc::new(programs);
    let started = Instant::now();
    let outcome = match backend.update_world_shader(entry.bucket, &programs) {
        Ok(PipelineSwap::Swapped) => ReloadOutcome::Swapped {
            frame_time: started.elapsed(),
            warnings,
        },
        Ok(PipelineSwap::NotResident) => ReloadOutcome::AppliesOnLoad { warnings },
        Err(e) => {
            return ReloadOutcome::Failed(ReloadFailure::Rejected(e.to_string()));
        }
    };
    if entry.bucket != 0 {
        overrides.set(entry.bucket, programs);
    }
    outcome
}
