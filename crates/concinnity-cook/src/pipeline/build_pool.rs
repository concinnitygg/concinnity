//! A worker pool for the compile pass whose threads share the calling thread's
//! build tables.
//!
//! The name interner and the resource-handle map are per-thread, so concurrent
//! builds cannot see each other's ids. A compile that resolves a reference (a
//! Material's texture, a SkinnedMesh's own name) on a shared pool worker would
//! read that worker's stale or empty tables instead of this build's. Every
//! thread of this pool starts with a copy of the caller's.

use concinnity_host::thread::asset_id;

pub(in crate::pipeline) fn build_pool() -> std::io::Result<rayon::ThreadPool> {
    let interner = asset_id::snapshot();
    let handles = crate::resource_handles::current_resource_handles();
    rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("cook-compile-{i}"))
        .start_handler(move |_| {
            asset_id::install_snapshot(&interner);
            crate::resource_handles::install_resource_handles(handles.clone());
        })
        .build()
        .map_err(std::io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;
    use concinnity_core::blob::ResourceKind;
    use concinnity_core::components::Material;
    use concinnity_core::ecs::TextureHandle;
    use concinnity_core::resource::ResourceHandles;

    // Every pool thread resolves names and handles exactly as the thread that
    // built the pool does, however the work is spread across it.
    #[test]
    fn pool_threads_resolve_through_the_callers_tables() {
        asset_id::reset_interner();
        asset_id::intern_all(&["floor", "brick", "moss"]);
        crate::resource_handles::reset_resource_handles();
        crate::resource_handles::install_resource_handles(ResourceHandles::from_assets([
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
}
