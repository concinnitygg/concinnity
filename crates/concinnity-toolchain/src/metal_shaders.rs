// Build-time precompilation of the engine's static Metal shaders.
//
// The device crate's build script hands this module the shader directory, the
// names that must stay source-only (runtime-assembled templates), and the
// shared fragments spliced into shaders at a marker. Every other `.metal` file
// is assembled into OUT_DIR and compiled to a `.metallib` with the same
// `xcrun metal` + `xcrun metallib` pair the cook's shader toolchain uses, and a
// lookup function mapping shader name to embedded bytes is generated there too.
//
// When the Metal compiler is not installed (Command Line Tools without a full
// Xcode) the generated lookup returns `None` for every name and the renderer
// falls back to compiling its embedded source at startup, exactly as before.
// A cargo warning flags the slower path so it is never a silent regression.

use std::path::{Path, PathBuf};
use std::process::Command;

// Precompile every eligible `.metal` under `shaders_dir` into OUT_DIR and
// generate `engine_metallibs.rs` there. `fragments` pairs a source marker with
// the file under `shaders_dir` that replaces it, matching the substitution the
// renderer applies when it compiles the same shader from source. Panics if the
// Metal toolchain is present but a shader fails to compile: a broken shader
// must fail the build, not surface at renderer init.
pub fn precompile_metal_shaders(
    shaders_dir: &Path,
    source_only: &[&str],
    fragments: &[(&str, &str)],
) {
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let generated = out_dir.join("engine_metallibs.rs");

    let shaders = eligible_shaders(shaders_dir, source_only);
    if !metal_toolchain_present() {
        println!(
            "cargo:warning=Metal compiler not found (full Xcode required); engine shaders \
             will compile from source at startup"
        );
        std::fs::write(&generated, stub_lookup_source()).expect("write engine_metallibs.rs");
        return;
    }

    let fragments: Vec<(&str, String)> = fragments
        .iter()
        .map(|(marker, file)| {
            let path = shaders_dir.join(file);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read fragment {}: {e}", path.display()));
            (*marker, text)
        })
        .collect();

    let lib_dir = out_dir.join("engine_shaders");
    std::fs::create_dir_all(&lib_dir).expect("create engine_shaders dir");
    let mut entries = Vec::with_capacity(shaders.len());
    for path in &shaders {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf8 shader filename")
            .to_string();
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let assembled = lib_dir.join(&name);
        std::fs::write(&assembled, assemble(&source, &fragments))
            .unwrap_or_else(|e| panic!("write {}: {e}", assembled.display()));
        let lib_path = lib_dir.join(&name).with_extension("metallib");
        compile_metallib(&assembled, &lib_path);
        entries.push((name, lib_path));
    }
    std::fs::write(&generated, metallib_lookup_source(&entries))
        .expect("write engine_metallibs.rs");
}

// Replace every fragment marker in `source`. Kept pure for unit testing.
fn assemble(source: &str, fragments: &[(&str, String)]) -> String {
    let mut out = source.to_string();
    for (marker, text) in fragments {
        out = out.replace(marker, text);
    }
    out
}

// Every `.metal` in the directory except the runtime-assembled template
// fragments, sorted for a deterministic generated file.
fn eligible_shaders(shaders_dir: &Path, source_only: &[&str]) -> Vec<PathBuf> {
    let mut shaders: Vec<PathBuf> = std::fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", shaders_dir.display()))
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "metal")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| !source_only.contains(&name))
        })
        .collect();
    shaders.sort();
    shaders
}

fn metal_toolchain_present() -> bool {
    Command::new("xcrun")
        .args(["--sdk", "macosx", "-f", "metal"])
        .output()
        .is_ok_and(|out| out.status.success())
}

// Same two-step pipeline the cook's Metal toolchain runs: source to AIR, AIR
// linked into a single-file metallib.
fn compile_metallib(source: &Path, lib_path: &Path) {
    let air_path = lib_path.with_extension("air");
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metal", "-c"])
            .arg(source)
            .arg("-o")
            .arg(&air_path),
        source,
        "xcrun metal",
    );
    run_step(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "metallib"])
            .arg(&air_path)
            .arg("-o")
            .arg(lib_path),
        source,
        "xcrun metallib",
    );
}

fn run_step(cmd: &mut Command, source: &Path, what: &str) {
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("{what} failed to launch for {}: {e}", source.display()));
    if !output.status.success() {
        panic!(
            "{what} failed for {}:\n{}\n{}",
            source.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

// Generated lookup mapping a registered shader name to its precompiled
// metallib bytes. Kept pure (paths in, source out) for unit testing.
fn metallib_lookup_source(entries: &[(String, PathBuf)]) -> String {
    let mut src = String::from(
        "// @generated by concinnity-toolchain::precompile_metal_shaders\n\
         pub(crate) fn embedded_metallib(name: &str) -> Option<&'static [u8]> {\n\
         \x20   match name {\n",
    );
    for (name, lib_path) in entries {
        src.push_str(&format!(
            "        {name:?} => Some(include_bytes!({:?})),\n",
            lib_path.display().to_string()
        ));
    }
    src.push_str("        _ => None,\n    }\n}\n");
    src
}

// Fallback when the Metal toolchain is unavailable at build time.
fn stub_lookup_source() -> String {
    "// @generated by concinnity-toolchain::precompile_metal_shaders (no Metal toolchain)\n\
     pub(crate) fn embedded_metallib(_name: &str) -> Option<&'static [u8]> {\n\
     \x20   None\n\
     }\n"
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_source_maps_each_entry_and_falls_through() {
        let entries = vec![
            (
                "post.metal".to_string(),
                PathBuf::from("/out/post.metallib"),
            ),
            ("taa.metal".to_string(), PathBuf::from("/out/taa.metallib")),
        ];
        let src = metallib_lookup_source(&entries);
        assert!(src.contains("\"post.metal\" => Some(include_bytes!(\"/out/post.metallib\"))"));
        assert!(src.contains("\"taa.metal\" => Some(include_bytes!(\"/out/taa.metallib\"))"));
        assert!(src.contains("_ => None"));
        assert!(src.contains("fn embedded_metallib"));
    }

    #[test]
    fn stub_lookup_source_returns_none_for_everything() {
        let src = stub_lookup_source();
        assert!(src.contains("fn embedded_metallib"));
        assert!(src.contains("None"));
        assert!(!src.contains("include_bytes!"));
    }

    #[test]
    fn assemble_substitutes_every_marker() {
        let fragments = vec![
            (
                "{OBJECT_DATA}",
                "struct GpuObjectData { float4x4 model; };".to_string(),
            ),
            ("{OTHER}", "// other".to_string()),
        ];
        let out = assemble("head\n{OBJECT_DATA}\nmid\n{OTHER}\ntail\n", &fragments);
        assert!(out.contains("struct GpuObjectData"));
        assert!(out.contains("// other"));
        assert!(!out.contains("{OBJECT_DATA}"));
        assert!(!out.contains("{OTHER}"));
        assert!(out.starts_with("head\n") && out.ends_with("tail\n"));
    }

    #[test]
    fn assemble_replaces_every_occurrence_and_leaves_markerless_source_alone() {
        let fragments = vec![("{OBJECT_DATA}", "RECORD".to_string())];
        assert_eq!(
            assemble("{OBJECT_DATA} a {OBJECT_DATA}", &fragments),
            "RECORD a RECORD"
        );
        assert_eq!(assemble("no markers", &fragments), "no markers");
    }

    #[test]
    fn eligible_shaders_excludes_source_only_and_non_metal() {
        let dir =
            std::env::temp_dir().join(format!("cn_metal_shaders_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["a.metal", "b.metal", "template.metal", "notes.txt"] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        let shaders = eligible_shaders(&dir, &["template.metal"]);
        let names: Vec<_> = shaders
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["a.metal", "b.metal"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
