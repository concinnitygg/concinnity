// src/app/commands.rs
//
// Commands issued over the WebSocket channel from the server.
//
// add / rm / load operate on an in-memory pending scene (see pending.rs).
// No disk write or build is triggered until `save` is called.
//
// save writes the pending scene to world.jsonl. The running app is NOT
// automatically restarted; the user re-runs `cn debug --websocket` to render
// the saved scene.
//
// write_file writes arbitrary content to a path relative to cwd (used by the
// server to push generated shader files and other assets alongside world.jsonl).
//
// validate_shader compiles a Metal source string with xcrun and returns the
// compiler output. An empty output string means no errors or warnings.
//
// fetch_assets downloads binary library assets from the server's HTTP endpoint
// and writes them to cwd so Texture assets can reference them by filename.

// Driven only by the binary's debug WebSocket channel; unreferenced in the
// FFI lib build.
#![allow(dead_code)]

use crate::app::pending;
use concinnity_cook::build_pipeline_from_str;

// Side-effect of a processed command beyond the ack string.
#[derive(Debug, PartialEq)]
pub enum CommandEffect {
    None,
    // The pending scene was saved; the caller should rebuild the world.
    Rebuild,
}

#[derive(Debug)]
pub enum AppCommand {
    // Stage an asset in the pending scene (no build).
    Add {
        asset_type: String,
        name: Option<String>,
        args: Option<serde_json::Value>,
    },
    // Remove a staged asset by name (no build).
    Rm {
        name: String,
    },
    // Replace the entire pending scene with the supplied asset list (no build).
    Load {
        assets: Vec<serde_json::Value>,
    },
    // Write the pending scene to world.jsonl.
    Save,
    // Write a file to the working directory (used for server-pushed assets).
    WriteFile {
        path: String,
        content: String,
    },
    // Compile a Metal shader and return compiler output.
    ValidateShader {
        source: String,
        name: String,
    },
    // Download named library assets from the server and write them to cwd.
    FetchAssets {
        names: Vec<String>,
        base_url: String,
        account_id: String,
    },
    // Validate world JSONL content through the full build pipeline without writing anything.
    TestWorld {
        content: String,
    },
}

// Execute a command. Returns Ok((output, effect)) where output is non-empty
// only for ValidateShader (carries compiler diagnostics), and effect signals
// whether the caller should take additional action (e.g. rebuild the world).
pub fn process_command(cmd: AppCommand) -> Result<(String, CommandEffect), String> {
    match cmd {
        AppCommand::Add {
            asset_type,
            name,
            args,
        } => pending::add(asset_type, name, args).map(|_| (String::new(), CommandEffect::None)),

        AppCommand::Rm { name } => pending::rm(&name).map(|_| (String::new(), CommandEffect::None)),

        AppCommand::Load { assets } => {
            pending::load(assets);
            Ok((String::new(), CommandEffect::None))
        }

        AppCommand::Save => pending::save()
            .map(|_| (String::new(), CommandEffect::Rebuild))
            .map_err(|e| e.to_string()),

        AppCommand::WriteFile { path, content } => {
            write_file(&path, &content).map(|out| (out, CommandEffect::None))
        }

        AppCommand::ValidateShader { source, name } => {
            validate_shader(&source, &name).map(|out| (out, CommandEffect::None))
        }

        AppCommand::FetchAssets {
            names,
            base_url,
            account_id,
        } => fetch_assets(&names, &base_url, &account_id).map(|out| (out, CommandEffect::None)),

        AppCommand::TestWorld { content } => match build_pipeline_from_str(&content, None) {
            Ok(result) => Ok((
                format!("ok: {} assets", result.defs.len()),
                CommandEffect::None,
            )),
            Err(e) => Err(e.to_string()),
        },
    }
}

fn write_file(path: &str, content: &str) -> Result<String, String> {
    // Reject path traversal and absolute paths.
    if path.contains("..") {
        return Err(format!("invalid path '{path}': contains '..'"));
    }
    if path.starts_with('/') || std::path::Path::new(path).is_absolute() {
        return Err(format!(
            "invalid path '{path}': absolute paths are not allowed"
        ));
    }

    // A bare filename with no directory component belongs in .concinnity/assets/.
    // The server sends paths like "default.metal" or "my_texture.png" which
    // should land in the fetched assets directory, not in cwd.
    let owned;
    let target = {
        let p = std::path::Path::new(path);
        if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
            owned = concinnity_core::paths::assets_dir().join(p);
            owned.as_path()
        } else {
            p
        }
    };

    // Only allow writing flat filenames or paths within cwd (no absolute paths).
    if target.is_absolute() {
        return Err(format!("invalid path '{path}': must be relative"));
    }

    if let Some(parent) = target.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create directories for '{path}': {e}"))?;
    }

    std::fs::write(target, content.as_bytes())
        .map(|_| String::new())
        .map_err(|e| format!("could not write '{path}': {e}"))
}

fn validate_shader(source: &str, name: &str) -> Result<String, String> {
    // Write source to a temp file under the system temp directory.
    let tmp_path = std::env::temp_dir().join(format!(
        "cn_validate_{}.metal",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));

    std::fs::write(&tmp_path, source.as_bytes())
        .map_err(|e| format!("could not write temp file: {e}"))?;

    let result = std::process::Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            tmp_path.to_str().unwrap_or("/tmp/shader.metal"),
            "-o",
            "/dev/null",
        ])
        .output()
        .map_err(|e| format!("could not run xcrun (is Xcode installed?): {e}"));

    // Clean up temp file regardless of outcome.
    let _ = std::fs::remove_file(&tmp_path);

    let output = result?;

    // Metal compiler emits all diagnostics to stderr.
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    // Replace the opaque temp path with the user-supplied name in diagnostics.
    let diagnostics = stderr.replace(tmp_path.to_str().unwrap_or(""), name);

    if output.status.success() {
        Ok(diagnostics)
    } else {
        Err(diagnostics)
    }
}

// Download each named asset from GET /v1/library/{name}?account_id={id} and
// write the bytes to .concinnity/assets/<name>. The name must be a plain
// filename with no path separators; anything else is rejected to avoid
// writing outside of the working directory.
fn fetch_assets(names: &[String], base_url: &str, _account_id: &str) -> Result<String, String> {
    use std::io::Read;

    let assets_dir = concinnity_core::paths::assets_dir();

    let mut fetched = Vec::new();
    let mut errors = Vec::new();

    for name in names {
        // Reject names that look like paths.
        if name.contains('/') || name.contains('\\') || name.contains("..") {
            errors.push(format!(
                "'{name}': name must be a plain filename, not a path"
            ));
            continue;
        }

        let url = format!("{}/static/assets/{}", base_url.trim_end_matches('/'), name);

        let response = match ureq::get(&url).call() {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("'{name}': request failed: {e}"));
                continue;
            }
        };

        let status = response.status();
        if status != 200 {
            errors.push(format!("'{name}': server returned HTTP {status}"));
            continue;
        }

        let mut bytes = Vec::new();
        if let Err(e) = response.into_body().into_reader().read_to_end(&mut bytes) {
            errors.push(format!("'{name}': read failed: {e}"));
            continue;
        }

        if let Err(e) = std::fs::create_dir_all(&assets_dir) {
            errors.push(format!("'{name}': could not create assets dir: {e}"));
            continue;
        }

        let dest = assets_dir.join(name);
        if let Err(e) = std::fs::write(&dest, &bytes) {
            errors.push(format!("'{name}': write failed: {e}"));
            continue;
        }

        fetched.push(name.clone());
    }

    if !errors.is_empty() {
        return Err(errors.join("; "));
    }

    Ok(format!("fetched: {}", fetched.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::pending::test_support;

    #[test]
    fn write_file_rejects_traversal() {
        let result = write_file("../outside.txt", "bad");
        assert!(result.is_err());
    }

    #[test]
    fn write_file_rejects_absolute() {
        let result = write_file("/etc/passwd", "bad");
        assert!(result.is_err());
    }

    #[test]
    fn write_file_rejects_embedded_traversal() {
        let err = write_file("shaders/../../outside.txt", "bad").unwrap_err();
        assert!(err.contains(".."), "unexpected error: {err}");
    }

    #[test]
    fn load_and_add_commands_report_no_effect() {
        let _guard = test_support::lock();
        let (out, effect) = process_command(AppCommand::Load { assets: Vec::new() }).unwrap();
        assert!(out.is_empty());
        assert_eq!(effect, CommandEffect::None);

        let (out, effect) = process_command(AppCommand::Add {
            asset_type: "Prop".to_string(),
            name: Some("staged".to_string()),
            args: None,
        })
        .unwrap();
        assert!(out.is_empty());
        assert_eq!(effect, CommandEffect::None);
    }

    #[test]
    fn add_command_surfaces_duplicate_errors() {
        let _guard = test_support::lock();
        process_command(AppCommand::Load { assets: Vec::new() }).unwrap();
        process_command(AppCommand::Add {
            asset_type: "Prop".to_string(),
            name: Some("twin".to_string()),
            args: None,
        })
        .unwrap();
        let err = process_command(AppCommand::Add {
            asset_type: "Prop".to_string(),
            name: Some("twin".to_string()),
            args: None,
        })
        .unwrap_err();
        assert!(err.contains("already exists"), "unexpected error: {err}");
    }

    #[test]
    fn rm_command_surfaces_missing_name_errors() {
        let _guard = test_support::lock();
        process_command(AppCommand::Load { assets: Vec::new() }).unwrap();
        let err = process_command(AppCommand::Rm {
            name: "ghost".to_string(),
        })
        .unwrap_err();
        assert!(err.contains("no asset named"), "unexpected error: {err}");
    }

    #[test]
    fn test_world_rejects_invalid_content() {
        let err = process_command(AppCommand::TestWorld {
            content: "{ not json".to_string(),
        })
        .unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_world_rejects_an_unknown_asset_type() {
        let err = process_command(AppCommand::TestWorld {
            content: r#"{"name":"odd","type":"NotARealAssetType","args":{}}"#.to_string(),
        })
        .unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn fetch_assets_rejects_path_like_names() {
        // Every name is invalid, so the helper errors before any request is
        // attempted (no network in tests).
        let names = vec![
            "../evil.png".to_string(),
            "a/b.png".to_string(),
            "c\\d.png".to_string(),
        ];
        let err = fetch_assets(&names, "http://127.0.0.1:0", "acct").unwrap_err();
        assert!(err.contains("plain filename"), "unexpected error: {err}");
        assert_eq!(err.matches("plain filename").count(), 3);
    }

    #[test]
    fn fetch_assets_with_no_names_is_an_empty_success() {
        let out = fetch_assets(&[], "http://127.0.0.1:0", "acct").unwrap();
        assert_eq!(out, "fetched: ");
    }
}
