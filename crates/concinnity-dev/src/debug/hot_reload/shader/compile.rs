//! Off-thread Shader recompiles. Each request runs on its own worker so the
//! frame loop keeps drawing while dxc works, and results are tagged with the
//! request's generation: only a Shader's newest request is applied, so an older
//! compile that finishes late never overwrites a newer save.

use concinnity_cook::compile::shader::CompiledShader;
use concinnity_core::components::{ShaderSource, ShaderStage};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::shader_programs::surface::Sources;
use concinnity_engine::gfx::system::shader_sources::ShaderSourceEntry;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};

use super::ShaderReloadFailure;

pub(in crate::debug::hot_reload) type CompileResult = Result<CompiledShader, ShaderReloadFailure>;

// A Shader's files as read at the moment of the save, each under the resolved
// path it was read from, which is the path its diagnostics then name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::debug::hot_reload) struct ShaderTexts {
    pub vertex: Option<ShaderSource>,
    pub fragment: ShaderSource,
}

impl ShaderTexts {
    // Read `entry`'s files. The error names the file that could not be read.
    pub(in crate::debug::hot_reload) fn read(entry: &ShaderSourceEntry) -> Result<Self, String> {
        let read = |stage| {
            entry
                .path(stage)
                .map(|path| {
                    concinnity_cook::compile::shader::read_shader_source(path)
                        .map(|text| ShaderSource {
                            path: path.to_string(),
                            text,
                        })
                        .map_err(|e| e.to_string())
                })
                .transpose()
        };
        Ok(Self {
            vertex: read(ShaderStage::Vertex)?,
            fragment: read(ShaderStage::Fragment)?
                .ok_or_else(|| "the Shader declares no fragment file".to_string())?,
        })
    }

    pub(in crate::debug::hot_reload) fn sources(&self) -> Sources<'_> {
        Sources {
            vertex: self.vertex.as_ref().map(ShaderSource::as_file),
            fragment: self.fragment.as_file(),
        }
    }
}

// Compile `texts` exactly as `cn build` would for this host's backend, then do
// the device-free part of the pipeline build so the swap on the frame thread
// is short.
pub(in crate::debug::hot_reload) fn compile(name: &str, texts: &ShaderTexts) -> CompileResult {
    // Inside the bounded job pool, so the compile's rayon fan-out over the
    // programs does not claim every core the frame loop also needs.
    let compiled = concinnity_host::thread::jobs::pool().install(|| {
        concinnity_cook::compile::shader::compile_world_shader(
            name,
            &texts.sources(),
            crate::cook_platform(),
        )
    })?;
    // A Shader catalog is only captured in a dev-loop session, whose backend is
    // always built with hot reload on.
    if let Err(e) = concinnity_engine::warm_world_shader(&compiled.programs, true) {
        tracing::warn!(
            "Shader hot-reload: '{name}' could not be prepared off the frame thread ({e}); \
             the swap will build it"
        );
    }
    Ok(compiled)
}

// One worker's result, tagged with the request it answers.
#[derive(Debug)]
pub(in crate::debug::hot_reload) struct Finished {
    pub id: AssetId,
    pub generation: u64,
    pub result: CompileResult,
}

// The generation each Shader's newest request was given. A result answers the
// newest request only when its generation matches.
#[derive(Debug, Default)]
pub(in crate::debug::hot_reload) struct Generations {
    latest: HashMap<AssetId, u64>,
    next: u64,
}

impl Generations {
    // Start a request for `id`, superseding any still in flight.
    pub(in crate::debug::hot_reload) fn begin(&mut self, id: AssetId) -> u64 {
        self.next += 1;
        self.latest.insert(id, self.next);
        self.next
    }

    // The result to apply, or `None` when a newer request for the Shader has
    // started since this one.
    pub(in crate::debug::hot_reload) fn accept(
        &self,
        finished: Finished,
    ) -> Option<(AssetId, CompileResult)> {
        (self.latest.get(&finished.id) == Some(&finished.generation))
            .then_some((finished.id, finished.result))
    }
}

// The in-flight recompiles and the channel their results come back on.
pub(in crate::debug::hot_reload) struct CompileQueue {
    generations: Generations,
    tx: Sender<Finished>,
    rx: Receiver<Finished>,
}

impl CompileQueue {
    pub(in crate::debug::hot_reload) fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            generations: Generations::default(),
            tx,
            rx,
        }
    }

    // Run `job` on a worker as the newest request for `id`.
    pub(in crate::debug::hot_reload) fn submit(
        &mut self,
        id: AssetId,
        job: impl FnOnce() -> CompileResult + Send + 'static,
    ) -> Result<(), String> {
        let generation = self.generations.begin(id);
        let tx = self.tx.clone();
        std::thread::Builder::new()
            .name("cn-shader-reload".into())
            .spawn(move || {
                // A dropped receiver means the reload state was rebuilt for
                // another world; the result is no longer wanted.
                let _ = tx.send(Finished {
                    id,
                    generation,
                    result: job(),
                });
            })
            .map(drop)
            .map_err(|e| format!("could not spawn the compile worker: {e}"))
    }

    // Every result that has arrived and still answers its Shader's newest
    // request. Never blocks.
    pub(in crate::debug::hot_reload) fn drain(&mut self) -> Vec<(AssetId, CompileResult)> {
        let generations = &self.generations;
        self.rx
            .try_iter()
            .filter_map(|finished| {
                let generation = finished.generation;
                let id = finished.id;
                let accepted = generations.accept(finished);
                if accepted.is_none() {
                    tracing::debug!(
                        "Shader hot-reload: dropped a superseded compile ({id:?}, request {generation})"
                    );
                }
                accepted
            })
            .collect()
    }
}
