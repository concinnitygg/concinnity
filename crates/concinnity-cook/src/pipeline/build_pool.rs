//! A worker pool for the compile pass whose threads share the calling thread's
//! build tables.
//!
//! Each thread resolves names and resource handles through the tables it was
//! given, so concurrent builds cannot see each other's ids. A compile that
//! resolves a reference (a Material's texture, a `SkinnedMesh`'s own name) on
//! a pool worker reads the caller's own tables, and a name it interns there is
//! the caller's too.

use std::sync::Arc;

use concinnity_host::thread::asset_id::{self, BuildScope};

pub(in crate::pipeline) fn build_pool() -> std::io::Result<rayon::ThreadPool> {
    let tables = asset_id::current_tables();
    rayon::ThreadPoolBuilder::new()
        .spawn_handler(move |thread| {
            let tables = Arc::clone(&tables);
            let mut builder =
                std::thread::Builder::new().name(format!("cook-compile-{}", thread.index()));
            if let Some(size) = thread.stack_size() {
                builder = builder.stack_size(size);
            }
            builder.spawn(move || {
                let _build = BuildScope::enter(tables);
                thread.run();
            })?;
            Ok(())
        })
        .build()
        .map_err(std::io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;
    use concinnity_engine::components::Material;
    use concinnity_engine::ecs::ResourceKind;
    use concinnity_engine::ecs::TextureHandle;
    use concinnity_engine::resource::ResourceHandles;

    // Every pool thread resolves names and handles exactly as the thread that
    // built the pool does, however the work is spread across it.
    #[test]
    fn pool_threads_resolve_through_the_callers_tables() {
        asset_id::begin_build();
        asset_id::intern_all(&["floor", "brick", "moss"]);
        asset_id::set_resource_handles(ResourceHandles::from_assets([
            (asset_id::intern("brick"), ResourceKind::Texture),
            (asset_id::intern("moss"), ResourceKind::Texture),
        ]));
        let resolve = || {
            let bytes = RegisteredType::Material
                .compile_payload(&serde_json::json!({"albedo": "moss"}), None)
                .unwrap();
            let mat: Material = postcard::from_bytes(&bytes).unwrap();
            (asset_id::lookup("moss"), mat.albedo)
        };
        let expected = resolve();
        assert_eq!(
            expected,
            (Some(asset_id::intern("moss")), Some(TextureHandle::new(1)))
        );

        let pool = build_pool().expect("pool");
        let seen: Vec<_> = pool.install(|| {
            use rayon::prelude::*;
            (0..64).into_par_iter().map(|_| resolve()).collect()
        });
        assert!(seen.iter().all(|r| *r == expected), "{seen:?}");
    }

    // A name a pool thread interns joins the build's own table, so the
    // caller sees the same id the worker assigned.
    #[test]
    fn a_name_a_pool_thread_interns_is_the_callers() {
        asset_id::begin_build();
        asset_id::intern_all(&["floor"]);
        let pool = build_pool().expect("pool");
        let assigned = pool.install(|| asset_id::intern("lantern"));
        assert_eq!(asset_id::lookup("lantern"), Some(assigned));
    }
}
