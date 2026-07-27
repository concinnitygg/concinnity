// src/metal/metallib.rs
//
// Precompiled engine shader libraries. The build script compiles every static
// `.metal` under src/metal/shaders/ to a metallib and generates the
// `embedded_metallib` lookup included here; `shader_library` in pipeline.rs
// prefers these bytes over startup source compilation. When the build host
// lacked the Metal toolchain the generated lookup returns `None` for every
// name and the source path takes over.

include!(concat!(env!("OUT_DIR"), "/engine_metallibs.rs"));

#[cfg(test)]
mod tests {
    use super::embedded_metallib;

    #[test]
    fn precompiled_coverage_is_all_or_nothing() {
        // The build script either compiles every eligible shader or emits the
        // stub lookup; per-shader gaps would mean a silent slow path. Skip when
        // the build host had no Metal toolchain (stub lookup).
        if embedded_metallib("post.metal").is_none() {
            return;
        }
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/shaders");
        for entry in std::fs::read_dir(dir).expect("read shaders dir") {
            let file_name = entry.expect("dir entry").file_name();
            let name = file_name.to_str().expect("utf8 shader filename");
            if !name.ends_with(".metal") || name.starts_with("raymarch_") {
                continue;
            }
            let bytes = embedded_metallib(name)
                .unwrap_or_else(|| panic!("{name}: no precompiled metallib embedded"));
            assert!(!bytes.is_empty(), "{name}: embedded metallib is empty");
        }
    }

    #[test]
    fn unregistered_names_return_none() {
        assert!(embedded_metallib("nope.metal").is_none());
    }
}
