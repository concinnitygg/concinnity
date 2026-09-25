//! World Shader hot reload. A save marks the Shaders reading the file (see
//! `files`); each marked Shader's files are read on the frame thread and
//! compiled on a worker (see `compile`); a finished compile is swapped in on
//! the frame thread through `update_world_shader`. A Shader whose scene is not
//! loaded has no pipeline to swap, so its programs wait in the shared
//! `ShaderOverrides` and install when the scene loads.

mod compile;
mod files;

#[cfg(test)]
mod tests;

use concinnity_core::components::ShaderPrograms;
use concinnity_core::render::backend::{LiveEdit, WorldShaderSwap};
use concinnity_engine::gfx::system::parked::ShaderOverrides;
use concinnity_engine::gfx::system::shader_sources::{ShaderSourceEntry, ShaderSourceMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::pending::PendingShaders;
use compile::{CompileQueue, CompileResult, ShaderTexts};
pub(super) use files::ShaderFileIndex;

// What became of one Shader's reload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShaderReloadOutcome {
    // The live pipeline was rebuilt from the edit, taking this long on the
    // frame thread.
    Swapped(Duration),
    // The Shader's scene is not loaded; the edit installs when it loads.
    AppliesOnLoad,
    // The live pipeline keeps its previous source.
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShaderReloadReport {
    pub name: String,
    pub outcome: ShaderReloadOutcome,
}

// Compiles a Shader's texts; the cook's compile outside tests.
type Compiler = Arc<dyn Fn(&str, &ShaderTexts) -> CompileResult + Send + Sync>;

// The Shader catalog, the overrides shared with the streaming pump, and the
// compiles in flight.
pub(crate) struct ShaderReload {
    pub(super) catalog: ShaderSourceMap,
    overrides: ShaderOverrides,
    compiles: CompileQueue,
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

    // Read the files of every Shader `pending` names and start its compile.
    // A Shader whose files cannot be read is reported failed right away.
    pub(crate) fn request(&mut self, pending: &PendingShaders) -> Vec<ShaderReloadReport> {
        let mut failed = Vec::new();
        for entry in self.catalog.entries.iter().filter(|e| pending.wants(e.id)) {
            let started = ShaderTexts::read(entry).and_then(|texts| {
                let compiler = Arc::clone(&self.compiler);
                let name = entry.name.clone();
                self.compiles
                    .submit(entry.id, move || compiler(&name, &texts))
            });
            match started {
                Ok(()) => tracing::info!("Shader hot-reload: recompiling '{}'", entry.name),
                Err(e) => failed.push(ShaderReloadReport {
                    name: entry.name.clone(),
                    outcome: ShaderReloadOutcome::Failed(e),
                }),
            }
        }
        failed
    }

    // Swap in every compile that finished since the last call.
    pub(crate) fn poll(&mut self, backend: &mut dyn LiveEdit) -> Vec<ShaderReloadReport> {
        self.compiles
            .drain()
            .into_iter()
            .filter_map(|(id, result)| {
                let entry = self.catalog.get(id)?;
                let outcome = match result {
                    Ok(programs) => apply(entry, programs, backend, &self.overrides),
                    Err(e) => ShaderReloadOutcome::Failed(e),
                };
                Some(ShaderReloadReport {
                    name: entry.name.clone(),
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
    backend: &mut dyn LiveEdit,
    overrides: &ShaderOverrides,
) -> ShaderReloadOutcome {
    let programs = Arc::new(programs);
    let started = Instant::now();
    let outcome = match backend.update_world_shader(entry.bucket, &programs) {
        Ok(WorldShaderSwap::Swapped) => ShaderReloadOutcome::Swapped(started.elapsed()),
        Ok(WorldShaderSwap::NotResident) => ShaderReloadOutcome::AppliesOnLoad,
        Err(e) => {
            return ShaderReloadOutcome::Failed(format!("pipeline rebuild rejected: {e}"));
        }
    };
    if entry.bucket != 0 {
        overrides.set(entry.bucket, programs);
    }
    outcome
}

// Log each report and, in an editor session, toast it.
pub(crate) fn report(
    reports: &[ShaderReloadReport],
    notify: Option<&crate::editor::notify::Notifier>,
) {
    use crate::editor::notify::Action;
    for ShaderReloadReport { name, outcome } in reports {
        match outcome {
            ShaderReloadOutcome::Swapped(frame_time) => {
                tracing::info!(
                    "Shader hot-reload: '{name}' recompiled, pipeline swapped ({:.1} ms on the \
                     frame thread)",
                    frame_time.as_secs_f64() * 1000.0
                );
                if let Some(n) = notify {
                    n.success(&format!("Shader '{name}' reloaded"));
                }
            }
            ShaderReloadOutcome::AppliesOnLoad => {
                tracing::info!(
                    "Shader hot-reload: '{name}' recompiled; its scene is not loaded, so the \
                     edit installs when it loads"
                );
                if let Some(n) = notify {
                    n.success(&format!(
                        "Shader '{name}' reloaded (applies when its scene loads)"
                    ));
                }
            }
            ShaderReloadOutcome::Failed(e) => {
                tracing::error!(
                    "Shader hot-reload: '{name}' failed: {e} (live pipeline kept its previous source)"
                );
                if let Some(n) = notify {
                    n.error_with(
                        &format!("Shader '{name}' reload failed"),
                        Action::OpenConsole,
                    );
                }
            }
        }
    }
}
