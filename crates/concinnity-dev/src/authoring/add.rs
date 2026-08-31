// src/add.rs
// Add an asset to a world JSONL and rebuild.
//
// The CLI and FFI both funnel through `add_to_path`, which:
//   - bootstraps a missing world from a `.glb` / `.txt` / `.md` target (a
//     text target becomes a `TextLabel` whose content is the file body); the
//     renderer stack itself is injected at build time from the entries'
//     companions, so no scaffold lines are written,
//   - appends a named content template's entries (`--template minimal-3d-world`)
//     when one is requested for a `.glb` landing in a renderer-less world,
//   - resolves `target` as a file path, a known asset type name, or inline
//     JSON, building one or more asset entries,
//   - patches the world JSONL atomically (via a tmp file) and reruns the
//     build pipeline so blobs and the lock file stay in sync. Only the
//     requested entries are written; injected companions and engine defaults
//     stay build-time only (see world-lock.json).

use crate::world::{WORLD_JSONL, patch_world_jsonl_to};
use concinnity_cook::asset_api::{AssetRequest, create_asset_def};
use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::build_from_path;

/// Add an asset to `world_path` and rebuild. See module docs.
///
/// `template` selects a named scaffold preset when scaffolding fires
/// (target is `.glb`, world has no renderer trigger). `None` uses the
/// default scaffold; `Some("minimal-3d-world")` layers that template's
/// entries. Unknown names error out before touching the world file.
pub fn add_to_path(
    world_path: &str,
    name: Option<&str>,
    target: &str,
    template: Option<&str>,
) -> std::io::Result<()> {
    let scaffold = scaffold_to_inject(world_path, target, template)?;
    ensure_world_file_exists(world_path)?;

    let mut entries = resolve_add_target(target)?;

    if let Some(n) = name {
        apply_name_override(&mut entries, n);
    }

    let entry_names: Vec<String> = entries
        .iter()
        .map(|e| {
            e.get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "resolved asset entry has no `name` field",
                    )
                })
                .map(str::to_string)
        })
        .collect::<Result<_, _>>()?;

    let tmp_path = format!("{}.tmp", world_path);

    patch_world_jsonl_to(world_path, &tmp_path, |assets| {
        // Re-adding a text file (or any other single-entry TextLabel target)
        // refreshes the existing same-name TextLabel's `content` in place
        // instead of erroring. Drag-dropping the same `.txt` twice (or
        // editing its body and re-adding) should "just work". Other args
        // (font, x/y, color, scale, centered, ...) are left alone so any
        // hand edits to the TextLabel survive the refresh.
        if let Some(refreshed) = try_refresh_text_label(assets, &entries) {
            tracing::info!("refreshed TextLabel '{}' content", refreshed);
            return Ok(());
        }

        // Likewise for an `.hdr`: a world binds one lighting environment, so a
        // second EnvironmentMap would never render. Point the existing one at
        // the new source instead of appending.
        if let Some((retargeted, source)) = try_retarget_environment_map(assets, &entries) {
            // Printed, not just traced: this edits an asset the user already
            // authored, so it must not look like a plain append.
            println!("Pointed existing EnvironmentMap '{retargeted}' at {source}");
            return Ok(());
        }

        for entry_name in &entry_names {
            if let Some(existing) = assets
                .iter()
                .find(|a| a.get("name").and_then(|v| v.as_str()) == Some(entry_name.as_str()))
            {
                let existing_type = existing.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "an asset named '{}' (type: {}) already exists in {}; \
                         remove it first with `concinnity rm {}`",
                        entry_name, existing_type, WORLD_JSONL, entry_name
                    ),
                ));
            }
        }
        // Append template entries first so the new asset's own systems (e.g.
        // a glTF's Camera3D) run alongside the template's setup. Any template
        // entry whose name already exists in the world is skipped to avoid
        // clobbering user-authored assets that happen to share the name.
        for entry in scaffold {
            let n = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if !assets
                .iter()
                .any(|a| a.get("name").and_then(|v| v.as_str()) == Some(n))
            {
                assets.push(entry);
            }
        }
        assets.extend(entries);
        Ok(())
    })?;

    match build_from_path(
        &crate::project::require()?,
        &tmp_path,
        crate::cook_platform(),
    ) {
        Ok(()) => std::fs::rename(&tmp_path, world_path).inspect_err(|_e| {
            let _ = std::fs::remove_file(&tmp_path);
        }),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

// Apply a caller-supplied name to freshly resolved entries: a single entry is
// renamed outright; a multi-entry target (e.g. a .metal file with both vertex
// and fragment stages) uses the supplied name as a prefix, keeping each
// entry's existing `_vert` / `_frag` style suffix.
pub(crate) fn apply_name_override(entries: &mut [serde_json::Value], name: &str) {
    if let [only] = entries {
        only["name"] = serde_json::Value::String(name.to_string());
        return;
    }
    for entry in entries {
        let existing = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let suffix = existing
            .rsplit_once('_')
            .map(|(_, s)| format!("_{s}"))
            .unwrap_or_default();
        entry["name"] = serde_json::Value::String(format!("{name}{suffix}"));
    }
}

// If `entries` is a single TextLabel whose name matches an existing
// TextLabel in `assets`, overwrite only the existing entry's `args.content`
// from the new entry and return the name. Returns `None` otherwise: the
// caller falls back to the normal "append, error on duplicate" flow.
//
// Scope is intentionally narrow:
//   - Single-entry only, so a `.glb` re-add (which fans into many entries)
//     doesn't accidentally clobber one of the existing materials/meshes.
//   - Same name AND same type, so a TextLabel never overwrites an
//     unrelated asset that happens to share a name.
//   - Only `content` is copied across; the user's edits to font, x/y,
//     color, scale, centered, background, padding, visible, view all stay.
fn try_refresh_text_label(
    assets: &mut [serde_json::Value],
    entries: &[serde_json::Value],
) -> Option<String> {
    if entries.len() != 1 {
        return None;
    }
    let new = &entries[0];
    if new.get("type").and_then(|v| v.as_str()) != Some("TextLabel") {
        return None;
    }
    let new_name = new.get("name").and_then(|v| v.as_str())?.to_string();
    let new_content = new.get("args").and_then(|a| a.get("content")).cloned()?;

    let existing = assets.iter_mut().find(|a| {
        a.get("name").and_then(|v| v.as_str()) == Some(new_name.as_str())
            && a.get("type").and_then(|v| v.as_str()) == Some("TextLabel")
    })?;
    let args = existing.get_mut("args")?.as_object_mut()?;
    args.insert("content".to_string(), new_content);
    Some(new_name)
}

// If `entries` is a single EnvironmentMap and `assets` already declares one,
// point that existing entry at the new source and return its name plus the
// source. Returns `None` otherwise: the caller falls back to the normal
// "append, error on duplicate" flow.
//
// The runtime binds the first EnvironmentMap it finds and ignores the rest, so
// appending a second would leave the just-added `.hdr` dark. Only `source` (and
// the `generator` it displaces, since the two are mutually exclusive) is
// written; the existing prefilter / irradiance / clamp tuning survives.
pub(crate) fn try_retarget_environment_map(
    assets: &mut [serde_json::Value],
    entries: &[serde_json::Value],
) -> Option<(String, String)> {
    if entries.len() != 1 {
        return None;
    }
    let new = &entries[0];
    if new.get("type").and_then(|v| v.as_str()) != Some("EnvironmentMap") {
        return None;
    }
    let new_source = new.get("args")?.get("source")?.as_str()?.to_string();

    let existing = assets
        .iter_mut()
        .find(|a| a.get("type").and_then(|v| v.as_str()) == Some("EnvironmentMap"))?;
    let name = existing.get("name").and_then(|v| v.as_str())?.to_string();
    let args = existing.get_mut("args")?.as_object_mut()?;
    args.insert(
        "source".to_string(),
        serde_json::Value::String(new_source.clone()),
    );
    args.insert(
        "generator".to_string(),
        serde_json::Value::String(String::new()),
    );
    Some((name, new_source))
}

// Decide which scaffold entries the patch closure should inject. Returns
// empty when no scaffolding is needed: the target doesn't bootstrap a world,
// the world already has a renderer-trigger asset (GraphicsSystem /
// GraphicsConfig / TextLabel / Window), or the target type provides its own
// renderer trigger (e.g. a text target's TextLabel). Errors when the world
// file is missing and the target can't bootstrap one: only `.glb`, `.txt`,
// and `.md` are allowed to create a world from nothing.
//
// `template` picks which scaffold flavour to inject. Unknown names error
// out before any file I/O so the user sees the typo immediately. An unknown
// template is treated as an error even when scaffolding wouldn't fire
// (e.g. existing world with a renderer trigger), since silently ignoring
// `--template foo` would mask the typo. `--template` is GLB-only: passing
// it with a text target is also rejected so the typo doesn't survive.
fn scaffold_to_inject(
    world_path: &str,
    target: &str,
    template: Option<&str>,
) -> std::io::Result<Vec<serde_json::Value>> {
    let template_entries = resolve_template(template)?;

    let world_exists = std::path::Path::new(world_path).exists();
    let bootstrap = target_bootstrap_kind(target);

    if !world_exists && matches!(bootstrap, BootstrapKind::None) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no world found at '{}': create one with `cn fetch-world` or `cn new`",
                world_path
            ),
        ));
    }

    if template_entries.is_some() && !matches!(bootstrap, BootstrapKind::Scene) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--template only applies to 3D scene targets (.glb)",
        ));
    }

    match bootstrap {
        BootstrapKind::None | BootstrapKind::Text => Ok(Vec::new()),
        BootstrapKind::Scene => {
            if world_exists && has_renderer_trigger(world_path)? {
                return Ok(Vec::new());
            }
            // No default scaffold: the renderer stack is injected at build
            // time from the scene's own assets. Only an explicitly requested
            // template writes extra entries.
            Ok(template_entries.unwrap_or_default())
        }
    }
}

// Map a `--template <name>` value to its entries, looked up in the engine-owned
// `concinnity_cook::authoring::template` registry. Returns:
//   - `Ok(None)`              when no template was requested (use default scaffold)
//   - `Ok(Some(entries))`     when the named template is known
//   - `Err(InvalidInput)`     when the name is unrecognised (typo → fail fast)
fn resolve_template(template: Option<&str>) -> std::io::Result<Option<Vec<serde_json::Value>>> {
    let Some(name) = template else {
        return Ok(None);
    };
    match concinnity_cook::authoring::template::by_name(name) {
        Some(t) => Ok(Some(
            crate::authoring::template_spec::world_template_entries(t),
        )),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "unknown template '{name}'; available: {}",
                available_templates()
            ),
        )),
    }
}

// Comma-separated list of known template names, for the "unknown template" error.
fn available_templates() -> String {
    concinnity_cook::authoring::template::TEMPLATES
        .iter()
        .map(|t| t.name)
        .collect::<Vec<_>>()
        .join(", ")
}

// Whether the JSONL file at `world_path` already renders on its own: it
// declares GraphicsConfig or a type whose presence implies it at build time
// (the registry's `renders` flag, which drives cook's GraphicsConfig companion
// injection). When true the world will start the GraphicsSystem without any
// scaffold. Malformed lines are skipped silently: the regular load path will
// surface any parse problems with full diagnostics.
fn has_renderer_trigger(world_path: &str) -> std::io::Result<bool> {
    let content = std::fs::read_to_string(world_path)?;
    Ok(jsonl_has_renderer_trigger(&content))
}

// Pure-string variant of `has_renderer_trigger`, exposed for unit tests.
fn jsonl_has_renderer_trigger(content: &str) -> bool {
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(t) = value.get("type").and_then(|v| v.as_str())
            && concinnity_cook::authoring::registry::type_renders(t)
        {
            return true;
        }
    }
    false
}

// Make sure `world_path` (and its parent directory) exists so
// `patch_world_jsonl_to` can read from it. Creates an empty file when missing;
// no-op when the file already exists.
fn ensure_world_file_exists(world_path: &str) -> std::io::Result<()> {
    if std::path::Path::new(world_path).exists() {
        return Ok(());
    }
    if let Some(parent) = std::path::Path::new(world_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(world_path, "")?;
    tracing::info!("created empty world file at {}", world_path);
    Ok(())
}

// How a target relates to renderer scaffolding. Targets that can bootstrap a
// world from nothing fall into `Scene` (needs the GLB scaffold injected
// alongside) or `Text` (the TextLabel that gets emitted is itself a renderer
// trigger and its companions inject the rest of the stack). Everything else
// is `None`: adding a shader or font into a missing world is rejected.
pub(crate) enum BootstrapKind {
    None,
    Scene,
    Text,
}

pub(crate) fn target_bootstrap_kind(target: &str) -> BootstrapKind {
    let ext = std::path::Path::new(target)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("glb") | Some("gltf") | Some("fbx") => BootstrapKind::Scene,
        Some("txt") | Some("md") => BootstrapKind::Text,
        _ => BootstrapKind::None,
    }
}

pub(crate) fn is_path_like(s: &str) -> bool {
    if s.contains('/') || s.contains('\\') {
        return true;
    }
    if s.starts_with('.') || s.starts_with('~') {
        return true;
    }
    // has a dot but the full string isn't a known type name
    if s.contains('.') {
        return RegisteredType::parse(s).is_none();
    }
    false
}

fn validated_entry(
    name: &str,
    asset_type: &str,
    args: serde_json::Value,
) -> std::io::Result<serde_json::Value> {
    // A resource asset does not build a component def. Resolve its args against
    // its registration (supplied over defaults) instead of `create_asset_def`.
    if let Some(rt) = RegisteredType::parse(asset_type).filter(|t| t.is_resource()) {
        let mut resolved = rt
            .registration()
            .default_args
            .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
        if let (serde_json::Value::Object(base), serde_json::Value::Object(supplied)) =
            (&mut resolved, &args)
        {
            for (k, v) in supplied {
                base.insert(k.clone(), v.clone());
            }
        }
        return Ok(serde_json::json!({
            "name": name,
            "type": asset_type,
            "args": resolved,
        }));
    }

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args.clone()),
    };
    create_asset_def(&req, crate::cook_platform())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let resolved_args = normalized_args_value(asset_type, &args);

    Ok(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": resolved_args,
    }))
}

// The args JSON written into world.jsonl for a validated add: the supplied
// args merged over the type's defaults (as `create_asset_def` resolves them)
// and normalized through the typed schema. The def's baked bytes are postcard
// and cannot round-trip to JSON.
fn normalized_args_value(asset_type: &str, args: &serde_json::Value) -> serde_json::Value {
    let empty = || serde_json::Value::Object(Default::default());
    let Some(ct) = RegisteredType::parse(asset_type) else {
        return empty();
    };
    let mut merged = ct.registration().default_args.unwrap_or_else(empty);
    if let (serde_json::Value::Object(base), serde_json::Value::Object(supplied)) =
        (&mut merged, args)
    {
        for (k, v) in supplied {
            base.insert(k.clone(), v.clone());
        }
    }
    ct.normalized_args(&merged, crate::cook_platform())
        .unwrap_or_else(|_| empty())
}

// Build an entry for a build-time (BuildOnly) import asset that expands from
// a source file (SceneImport, StoryImport). Such an asset can't go through
// `validated_entry` / `create_asset_def` (which only build External
// components); materialize its default args from the registration and set the
// source path.
fn import_entry(asset_type: &str, name: &str, source: &str) -> std::io::Result<serde_json::Value> {
    let reg = RegisteredType::parse(asset_type)
        .ok_or_else(|| std::io::Error::other(format!("{} asset type is unavailable", asset_type)))?
        .registration();
    let mut args = reg
        .default_args
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    if let serde_json::Value::Object(map) = &mut args {
        map.insert(
            "source".to_string(),
            serde_json::Value::String(source.to_string()),
        );
    }
    Ok(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": args,
    }))
}

// Every file extension the dispatch below resolves, grouped for a file
// picker's category filters (the editor's Import panel builds its dialog from
// this). Kept beside the dispatch so a newly handled extension is offered by
// the picker the same day the CLI learns it;
// `import_extension_groups_track_the_dispatch` fails if one drifts out.
pub(crate) const IMPORT_EXTENSION_GROUPS: &[(&str, &[&str])] = &[
    ("Scenes", &["glb", "gltf", "fbx"]),
    ("Stories & text", &["md", "txt"]),
    ("Images", &["png", "jpg", "jpeg", "bmp", "tga", "gif"]),
    ("Textures", &["ktx2"]),
    ("Environment maps", &["hdr"]),
    ("Audio", &["ogg", "wav", "mp3", "flac"]),
    ("Fonts", &["ttf", "otf"]),
    ("Shaders", &["vert", "frag", "glsl", "metal", "wgsl"]),
    ("Models & data", &["obj", "mtl", "json", "gguf"]),
];

// Resolve a file path into its asset entries by extension. Shared by the CLI
// add flow above and the editor's Import panel, so both produce identical
// entries for the same file.
pub(crate) fn entry_from_path(path_str: &str) -> std::io::Result<Vec<serde_json::Value>> {
    let path = std::path::Path::new(path_str);

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    // stem without extension, dots replaced with underscores (shared with
    // companion injection so a generated default asset is named identically)
    let stem = concinnity_cook::authoring::world::asset_name_from_path(path_str);

    // full filename with dots replaced with underscores (used for most types)
    let base_name = if ext == "json" {
        stem.clone()
    } else {
        path.file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_else(|| path_str.to_string())
    };

    if ext == "json" {
        return entry_from_json_file(path, &base_name).map(|e| vec![e]);
    }

    match ext.as_str() {
        // Dedicated-extension GLSL shaders: a Shader needs both stages, so
        // the file is paired with its sibling stage file.
        "vert" => {
            let frag = sibling_path(path, "frag")?;
            shader_pair_entry(&stem, path_str, &frag)
        }
        "frag" => {
            let vert = sibling_path(path, "vert")?;
            shader_pair_entry(&stem, &vert, path_str)
        }

        // GLSL: infer the stage from the filename stem and pair with the
        // counterpart file (foo_vert.glsl <-> foo_frag.glsl).
        "glsl" => {
            let (vert, frag, pair_stem) = glsl_pair(path, path_str, &stem)?;
            shader_pair_entry(&pair_stem, &vert, &frag)
        }

        // Metal: parse source to detect which stages are present
        "metal" => {
            let source = read_source_file(path_str)?;
            shader_entries_from_stages(path_str, &stem, "metal", &detect_metal_stages(&source))
        }

        // WGSL: parse source to detect which stages are present
        "wgsl" => {
            let source = read_source_file(path_str)?;
            shader_entries_from_stages(path_str, &stem, "wgsl", &detect_wgsl_stages(&source))
        }

        // Fonts: stem only, no extension suffix needed, font names won't conflict with shaders
        "ttf" | "otf" => Ok(vec![validated_entry(
            &stem,
            "Font",
            serde_json::json!({ "path": path_str }),
        )?]),

        // LLM weights
        "gguf" => Ok(vec![validated_entry(
            &base_name,
            "LLM",
            serde_json::json!({ "lib_path": "", "model_path": path_str }),
        )?]),

        // Audio files: an AudioClip whose payload is compiled (and decode-
        // validated) at build. Played through an AudioEmitter (positional) or
        // an AudioCue (view-triggered). Stem only, like fonts, so emitter and
        // cue declarations read naturally.
        "ogg" | "wav" | "mp3" | "flac" => Ok(vec![validated_entry(
            &stem,
            "AudioClip",
            serde_json::json!({ "source": path_str }),
        )?]),

        // Radiance HDR: an EnvironmentMap, whose build convolves the
        // equirectangular source into the irradiance + prefiltered radiance
        // cubemaps that light the scene. Its presence also injects the skybox
        // mesh that displays it (see cook's `inject_sky`). Stem only, like
        // fonts and audio.
        "hdr" => Ok(vec![environment_map_entry(&stem, path_str)?]),

        // KTX2: a GPU-ready compressed texture. Becomes a Texture asset whose
        // source the build compiles into a block-compressed payload (BCn / Basis
        // transcode), unlike the raw File assets below.
        "ktx2" => Ok(vec![validated_entry(
            &base_name,
            "Texture",
            serde_json::json!({ "source": path_str }),
        )?]),

        // File-backed assets: path is stored as-is; build compiles the blob
        "obj" | "mtl" | "png" | "jpg" | "jpeg" | "bmp" | "tga" | "gif" => {
            Ok(vec![validated_entry(
                &base_name,
                "File",
                serde_json::json!({ "path": path_str, "kind": ext.as_str() }),
            )?])
        }

        // Text files become a TextLabel carrying the file contents. The label
        // is its own renderer trigger (the registry's `renders` flag) and
        // companion injection adds GraphicsSystem, so a fresh `cn add notes.txt`
        // lands a renderable world without a scaffold. Naming no Font draws it
        // with the built-in face.
        //
        // `centered: true` keeps the contents off the HUD chips in the top-left
        // corner, and matches what `cn init` writes.
        "txt" => Ok(vec![text_label_entry(&stem, path_str)?]),

        // Markdown: a file opening with a frontmatter fence is a story and
        // becomes one StoryImport line the build expands into its UI assets;
        // plain Markdown is treated as text, same as `.txt`.
        "md" => {
            if md_has_frontmatter(path_str)? {
                Ok(vec![import_entry("StoryImport", &stem, path_str)?])
            } else {
                Ok(vec![text_label_entry(&stem, path_str)?])
            }
        }

        // 3D scene files: one SceneImport line. The build expands it into
        // Textures / Materials / Meshes / Models / Props at compile time, so
        // world.jsonl stays compact (see concinnity_core::bake::import).
        //
        // A `.glb` is checked for the panorama-sphere packaging first: those
        // files carry an environment image, not geometry, and importing one as
        // a mesh puts a ball in the scene where a sky belongs.
        "glb" | "gltf" => Ok(vec![scene_or_panorama_entry(&stem, path_str)?]),
        "fbx" => Ok(vec![import_entry("SceneImport", &stem, path_str)?]),

        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "unknown extension '.{}'; pass a type name instead \
                 (e.g. `concinnity add Logger`) or edit {} directly",
                other, WORLD_JSONL
            ),
        )),
    }
}

// A `.glb` / `.gltf` becomes an EnvironmentMap when it is a panorama sphere
// (see `concinnity_cook::import::panorama`) and scene geometry otherwise. The choice
// is logged because the same extension lands two different asset types.
fn scene_or_panorama_entry(stem: &str, path_str: &str) -> std::io::Result<serde_json::Value> {
    if concinnity_cook::import::panorama::file_is_panorama_sphere(path_str) {
        tracing::info!(
            "'{}' is a panorama sphere: importing it as an EnvironmentMap (sky \
             and image-based lighting) rather than scene geometry",
            path_str
        );
        return environment_map_entry(stem, path_str);
    }
    import_entry("SceneImport", stem, path_str)
}

fn environment_map_entry(stem: &str, path_str: &str) -> std::io::Result<serde_json::Value> {
    validated_entry(
        stem,
        "EnvironmentMap",
        serde_json::json!({ "source": path_str }),
    )
}

// Build one Shader entry from a source file that must carry both stages.
// Named "{stem}_shader", with both stages reading the same source file.
fn shader_entries_from_stages(
    path_str: &str,
    stem: &str,
    ext: &str,
    stages: &[&str],
) -> std::io::Result<Vec<serde_json::Value>> {
    if !(stages.contains(&"vertex") && stages.contains(&"fragment")) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "a Shader needs both a vertex and a fragment stage, but the .{ext} \
                 source declares only {stages:?}. Add the missing stage function, or \
                 declare a Shader entry in world.jsonl with per-stage sources"
            ),
        ));
    }
    shader_pair_entry(stem, path_str, path_str)
}

// One Shader entry named "{stem}_shader" from a vertex + fragment source pair
// (the two paths are the same file for multi-stage sources).
fn shader_pair_entry(
    stem: &str,
    vert_path: &str,
    frag_path: &str,
) -> std::io::Result<Vec<serde_json::Value>> {
    Ok(vec![validated_entry(
        &format!("{stem}_shader"),
        "Shader",
        serde_json::json!({
            "vertex": { "source": vert_path },
            "fragment": { "source": frag_path },
        }),
    )?])
}

// The sibling stage file next to a dedicated-extension GLSL shader
// (x.vert <-> x.frag). Errors when the counterpart is missing: half a shader
// program cannot render.
fn sibling_path(path: &std::path::Path, sibling_ext: &str) -> std::io::Result<String> {
    let sibling = path.with_extension(sibling_ext);
    if !sibling.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "a Shader needs both stages: expected the {} stage next to {} \
                 (looked for {})",
                if sibling_ext == "vert" {
                    "vertex"
                } else {
                    "fragment"
                },
                path.display(),
                sibling.display(),
            ),
        ));
    }
    Ok(sibling.to_string_lossy().into_owned())
}

// Pair a .glsl file with its counterpart stage by swapping the stage marker in
// the filename stem (foo_vert.glsl <-> foo_frag.glsl). Returns (vertex,
// fragment) source paths plus the marker-stripped stem, so adding either file
// of the pair produces the same Shader entry name.
fn glsl_pair(
    path: &std::path::Path,
    path_str: &str,
    stem: &str,
) -> std::io::Result<(String, String, String)> {
    let (this_marker, other_marker) = if stem.contains("fragment") {
        ("fragment", "vertex")
    } else if stem.contains("frag") {
        ("frag", "vert")
    } else if stem.contains("vertex") {
        ("vertex", "fragment")
    } else if stem.contains("vert") {
        ("vert", "frag")
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "cannot infer the shader stage of {path_str}: name the files with \
                 vert/frag markers (foo_vert.glsl + foo_frag.glsl), or declare a \
                 Shader entry in world.jsonl with per-stage sources"
            ),
        ));
    };
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path_str);
    let counterpart = path.with_file_name(file_name.replace(this_marker, other_marker));
    if !counterpart.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "a Shader needs both stages: expected the {other_marker} counterpart \
                 of {path_str} (looked for {})",
                counterpart.display()
            ),
        ));
    }
    let counterpart = counterpart.to_string_lossy().into_owned();
    let this = path_str.to_string();
    let pair_stem = stem
        .replace(this_marker, "")
        .trim_matches('_')
        .replace("__", "_");
    let (vert, frag) = if this_marker.starts_with('v') {
        (this, counterpart)
    } else {
        (counterpart, this)
    };
    Ok((vert, frag, pair_stem))
}

// Detect Metal pipeline stages from source text.
// Metal uses `vertex` / `fragment` as function-qualifier keywords at the start of declarations.
// Returns a non-empty list; defaults to ["vertex"] when no qualifiers are found.
pub(crate) fn detect_metal_stages(source: &str) -> Vec<&'static str> {
    let has_vertex = source.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("vertex ") || t.starts_with("vertex\t")
    });
    let has_fragment = source.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("fragment ") || t.starts_with("fragment\t")
    });
    stages_from_flags(has_vertex, has_fragment)
}

// Detect WGSL pipeline stages from source text.
// WGSL uses `@vertex` / `@fragment` attribute decorators.
// Returns a non-empty list; defaults to ["vertex"] when no attributes are found.
pub(crate) fn detect_wgsl_stages(source: &str) -> Vec<&'static str> {
    stages_from_flags(source.contains("@vertex"), source.contains("@fragment"))
}

fn stages_from_flags(has_vertex: bool, has_fragment: bool) -> Vec<&'static str> {
    let mut stages = Vec::new();
    if has_vertex {
        stages.push("vertex");
    }
    if has_fragment {
        stages.push("fragment");
    }
    if stages.is_empty() {
        stages.push("vertex");
    }
    stages
}

fn read_source_file(path_str: &str) -> std::io::Result<String> {
    std::fs::read_to_string(path_str)
        .map_err(|e| std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e)))
}

// Cap text-label contents at a size that makes sense for a HUD overlay. A
// TextLabel isn't a document viewer; multi-MB drops would silently bloat
// world.jsonl and the per-frame layout pass.
const TEXT_LABEL_MAX_BYTES: usize = 64 * 1024;

// Read a `.txt` / `.md` file for use as TextLabel `content`. Strips a single
// trailing `\n` (and a preceding `\r`, in case the file is CRLF) so labels
// don't render with a stray blank line at the bottom, but otherwise preserves
// internal whitespace and newlines verbatim. Files larger than
// `TEXT_LABEL_MAX_BYTES` are rejected up front rather than truncated, so the
// user knows the contents weren't silently clipped.
fn read_text_content(path_str: &str) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path_str).map_err(|e| {
        std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e))
    })?;
    if metadata.len() as usize > TEXT_LABEL_MAX_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "'{}' is {} bytes; TextLabel content is capped at {} bytes: \
                 trim the file or add it as a `File` asset via inline JSON instead",
                path_str,
                metadata.len(),
                TEXT_LABEL_MAX_BYTES
            ),
        ));
    }
    let mut content = std::fs::read_to_string(path_str).map_err(|e| {
        std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e))
    })?;
    if content.ends_with('\n') {
        content.pop();
        if content.ends_with('\r') {
            content.pop();
        }
    }
    Ok(content)
}

fn text_label_entry(name: &str, path_str: &str) -> std::io::Result<serde_json::Value> {
    validated_entry(
        name,
        "TextLabel",
        serde_json::json!({
            "content": read_text_content(path_str)?,
            "centered": true,
        }),
    )
}

// A Markdown story opens with a `---` frontmatter fence on its first line.
// Detection reads only that line, so the TextLabel size cap never applies to
// a story file.
fn md_has_frontmatter(path_str: &str) -> std::io::Result<bool> {
    use std::io::BufRead;
    let file = std::fs::File::open(path_str).map_err(|e| {
        std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e))
    })?;
    let mut first = String::new();
    std::io::BufReader::new(file).read_line(&mut first)?;
    Ok(first.trim_start_matches('\u{feff}').trim_end() == "---")
}

fn entry_from_json_file(
    path: &std::path::Path,
    stem_name: &str,
) -> std::io::Result<serde_json::Value> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("could not read '{}': {}", path.display(), e),
        )
    })?;
    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("could not parse '{}': {}", path.display(), e),
        )
    })?;

    let asset_type = json.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("'{}' has no `type` field", path.display()),
        )
    })?;

    if asset_type.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let args = json
        .get("args")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

    let name = json
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(stem_name)
        .to_string();

    validated_entry(&name, asset_type, args)
}

fn entry_from_inline_json(raw: &str) -> std::io::Result<serde_json::Value> {
    let json: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("could not parse inline JSON: {}", e),
        )
    })?;

    if !json.is_object() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "inline JSON must be an object (e.g. '{\"type\": \"Window\"}')",
        ));
    }

    let asset_type = json.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "inline JSON must contain a `type` field",
        )
    })?;

    if asset_type.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let name = json
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| asset_type.to_lowercase());

    let args = json
        .get("args")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args.clone()),
    };
    create_asset_def(&req, crate::cook_platform())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let resolved_args = normalized_args_value(asset_type, &args);

    Ok(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": resolved_args,
    }))
}

// Resolve an add target -- a file path, a known asset type name, or inline
// JSON -- into its world entries. Shared with the editor console's /add, so a
// console add and a CLI add accept the same targets.
pub(crate) fn resolve_add_target(target: &str) -> std::io::Result<Vec<serde_json::Value>> {
    if is_path_like(target) {
        return entry_from_path(target);
    }

    if RegisteredType::parse(target).is_some() {
        return entry_from_type_name(target).map(|e| vec![e]);
    }

    let as_path = std::path::Path::new(target);
    if as_path.exists() && as_path.is_file() {
        let name = as_path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_else(|| target.to_string());
        return entry_from_json_file(as_path, &name).map(|e| vec![e]);
    }

    if target.trim_start().starts_with('{') {
        return entry_from_inline_json(target).map(|e| vec![e]);
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "could not resolve '{}' as any of:\n  \
             - a file path (no such file found)\n  \
             - a known asset type (use `concinnity list` to see available types)\n  \
             - an inline JSON object (must start with '{{' and contain a `type` field)",
            target
        ),
    ))
}

fn entry_from_type_name(type_str: &str) -> std::io::Result<serde_json::Value> {
    if type_str.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let req = AssetRequest {
        asset_type: type_str.to_string(),
        args: None,
    };
    create_asset_def(&req, crate::cook_platform())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let args = normalized_args_value(type_str, &serde_json::Value::Object(Default::default()));

    let name = type_str.to_lowercase();

    Ok(serde_json::json!({
        "name": name,
        "type": type_str,
        "args": args,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // detect_metal_stages

    #[test]
    fn metal_both_stages() {
        let src = "vertex VertexOut vert_main() {}\nfragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex", "fragment"]);
    }

    #[test]
    fn metal_vertex_only() {
        let src = "vertex VertexOut vert_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_fragment_only() {
        let src = "fragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["fragment"]);
    }

    #[test]
    fn metal_no_qualifiers_defaults_to_vertex() {
        let src = "// helper only\nfloat4 helper() { return float4(1.0); }";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_tab_separated_qualifier() {
        let src = "vertex\tVertexOut vert_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_indented_qualifier_still_detected() {
        // Metal qualifiers can appear with leading whitespace (e.g. inside a namespace-like block)
        let src = "  vertex VertexOut vert_main() {}\n  fragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex", "fragment"]);
    }

    // detect_wgsl_stages

    #[test]
    fn wgsl_both_stages() {
        let src =
            "@vertex\nfn vs() -> VertexOutput {}\n@fragment\nfn fs() -> @location(0) vec4<f32> {}";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex", "fragment"]);
    }

    #[test]
    fn wgsl_vertex_only() {
        let src = "@vertex fn vs() {}";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex"]);
    }

    #[test]
    fn wgsl_no_attributes_defaults_to_vertex() {
        let src = "fn helper() -> f32 { return 1.0; }";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex"]);
    }

    #[test]
    fn a_two_stage_source_becomes_one_shader_entry() {
        let entries =
            shader_entries_from_stages("s.metal", "s", "metal", &["vertex", "fragment"]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "s_shader");
        assert_eq!(entries[0]["type"], "Shader");
        assert_eq!(entries[0]["args"]["vertex"]["source"], "s.metal");
        assert_eq!(entries[0]["args"]["fragment"]["source"], "s.metal");
    }

    #[test]
    fn a_single_stage_source_is_rejected_with_guidance() {
        let err = shader_entries_from_stages("s.metal", "s", "metal", &["vertex"]).unwrap_err();
        assert!(err.to_string().contains("both a vertex and a fragment"));
    }

    #[test]
    fn glsl_pair_swaps_the_stage_marker_and_shares_a_stem() {
        let dir = tempfile::tempdir().unwrap();
        let vert = dir.path().join("foo_vert.glsl");
        let frag = dir.path().join("foo_frag.glsl");
        std::fs::write(&vert, "").unwrap();
        std::fs::write(&frag, "").unwrap();

        // Adding either file of the pair resolves the same (vert, frag, stem).
        let from_vert = glsl_pair(&vert, vert.to_str().unwrap(), "foo_vert").unwrap();
        let from_frag = glsl_pair(&frag, frag.to_str().unwrap(), "foo_frag").unwrap();
        assert_eq!(from_vert.0, vert.to_str().unwrap());
        assert_eq!(from_vert.1, frag.to_str().unwrap());
        assert_eq!(from_vert.0, from_frag.0);
        assert_eq!(from_vert.1, from_frag.1);
        assert_eq!(from_vert.2, "foo");
        assert_eq!(from_frag.2, "foo");

        // A missing counterpart is an error, not half a Shader.
        std::fs::remove_file(&frag).unwrap();
        assert!(glsl_pair(&vert, vert.to_str().unwrap(), "foo_vert").is_err());
    }

    // font stem naming

    #[test]
    fn font_name_uses_stem_only() {
        let path = std::path::Path::new("fonts/JetBrainsMono-Regular.ttf");
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_default();
        // stem should not include the .ttf extension
        assert_eq!(stem, "JetBrainsMono-Regular");
        assert!(!stem.contains("ttf"));
    }

    // unknown extension

    #[test]
    fn unknown_extension_errors() {
        // Use a path that won't exist on disk so it goes through entry_from_path
        let result = entry_from_path("assets/shader.xyz");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains(".xyz"));
    }

    // stages_from_flags edge cases

    #[test]
    fn stages_from_flags_neither() {
        assert_eq!(stages_from_flags(false, false), vec!["vertex"]);
    }

    #[test]
    fn stages_from_flags_both() {
        assert_eq!(stages_from_flags(true, true), vec!["vertex", "fragment"]);
    }

    #[test]
    fn stages_from_flags_fragment_only() {
        assert_eq!(stages_from_flags(false, true), vec!["fragment"]);
    }

    // target_bootstrap_kind

    #[test]
    fn target_bootstrap_kind_glb_is_scene() {
        assert!(matches!(
            target_bootstrap_kind("models/scene.glb"),
            BootstrapKind::Scene
        ));
        assert!(matches!(
            target_bootstrap_kind("scene.GLB"),
            BootstrapKind::Scene
        ));
    }

    #[test]
    fn target_bootstrap_kind_text_is_text() {
        assert!(matches!(
            target_bootstrap_kind("notes.txt"),
            BootstrapKind::Text
        ));
        assert!(matches!(
            target_bootstrap_kind("README.MD"),
            BootstrapKind::Text
        ));
    }

    #[test]
    fn target_bootstrap_kind_other_is_none() {
        assert!(matches!(
            target_bootstrap_kind("Logger"),
            BootstrapKind::None
        ));
        assert!(matches!(
            target_bootstrap_kind("shader.vert"),
            BootstrapKind::None
        ));
        assert!(matches!(
            target_bootstrap_kind("font.ttf"),
            BootstrapKind::None
        ));
    }

    // jsonl_has_renderer_trigger

    #[test]
    fn renderer_trigger_matches_graphics_config() {
        let jsonl = r#"{"name":"gc","type":"GraphicsConfig","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_matches_text_label() {
        let jsonl = r#"{"name":"lbl","type":"TextLabel","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    // A Prop implies the world renders: cook injects the GraphicsConfig marker
    // for it, so no scaffold is needed.
    #[test]
    fn renderer_trigger_matches_prop() {
        let jsonl = r#"{"name":"crate","type":"Prop","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    // A bare Window does NOT start the renderer (cook injects no GraphicsConfig
    // for it), so it is not a trigger.
    #[test]
    fn renderer_trigger_ignores_window() {
        let jsonl = r#"{"name":"win","type":"Window","args":{}}"#;
        assert!(!jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_absent_for_render_data_only_world() {
        // Exactly the shape that produced the no-window bug: textures /
        // materials / meshes / models / camera but no renderer-trigger.
        let jsonl = concat!(
            r#"{"name":"tex","type":"Texture","args":{}}"#,
            "\n",
            r#"{"name":"mat","type":"Material","args":{}}"#,
            "\n",
            r#"{"name":"mesh","type":"Mesh","args":{}}"#,
            "\n",
            r#"{"name":"model","type":"Model","args":{}}"#,
            "\n",
            r#"{"name":"cam","type":"Camera3D","args":{}}"#,
            "\n",
        );
        assert!(!jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_skips_malformed_lines() {
        let jsonl = concat!(
            "garbage line\n",
            r#"{"name":"tex","type":"Texture","args":{}}"#,
            "\n",
        );
        assert!(!jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_handles_empty_jsonl() {
        assert!(!jsonl_has_renderer_trigger(""));
    }

    // scaffold_to_inject

    #[test]
    fn scaffold_to_inject_writes_nothing_without_a_template() {
        // Existing world with renderer-less assets + a .glb target: the
        // renderer stack is injected at build time, so no lines are written.
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");
        std::fs::write(
            &world,
            concat!(
                r#"{"name":"tex","type":"Texture","args":{}}"#,
                "\n",
                r#"{"name":"cam","type":"Camera3D","args":{}}"#,
                "\n",
            ),
        )
        .unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());
    }

    #[test]
    fn scaffold_to_inject_returns_empty_when_world_has_graphics_config() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");
        std::fs::write(
            &world,
            r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
        )
        .unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());
    }

    #[test]
    fn scaffold_to_inject_allows_missing_world_with_glb() {
        // No file present, target is `.glb`: the caller is expected to create
        // the file; the renderer stack comes from build-time injection, so no
        // template entries are needed.
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());
    }

    #[test]
    fn scaffold_to_inject_errors_for_missing_world_with_non_scene_target() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let err = scaffold_to_inject(world.to_str().unwrap(), "Logger", None)
            .expect_err("missing world + non-scene target must error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn scaffold_to_inject_returns_empty_for_non_scene_target_into_existing_world() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");
        std::fs::write(&world, "").unwrap();

        // Non-scene target shouldn't trigger scaffolding even when the world
        // has no renderer: we don't try to guess intent for shaders/fonts.
        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "Logger", None).unwrap();
        assert!(scaffold.is_empty());
    }

    // template dispatch

    #[test]
    fn scaffold_to_inject_uses_named_template() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let name = concinnity_cook::authoring::template::TEMPLATES[0].name;
        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some(name))
            .expect("a named template should apply for a renderer-less glb add");
        let expected = concinnity_cook::authoring::template::by_name(name)
            .unwrap()
            .assets()
            .len();
        assert_eq!(scaffold.len(), expected, "expected the template's entries");
        assert!(!scaffold.is_empty());
    }

    // Every entry of every engine-owned template validates as a real, buildable
    // asset with its declared args. This is the typed round-trip the templates
    // crate (pure data) cannot do itself: it guards against a template naming a
    // type that doesn't exist or shipping args the asset rejects, and against the
    // spec builders drifting from the real asset schemas.
    #[test]
    fn every_template_entry_validates_as_a_real_asset() {
        let _guard = crate::test_support::lock();
        for t in concinnity_cook::authoring::template::TEMPLATES {
            let entries = crate::authoring::template_spec::world_template_entries(t);
            assert!(!entries.is_empty(), "template '{}' is empty", t.name);
            for entry in entries {
                let ty = entry["type"].as_str().expect("entry has a type");
                let name = entry["name"].as_str().expect("entry has a name");
                let args = entry
                    .get("args")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                concinnity_cook::validate_asset(ty, name, &args, crate::cook_platform())
                    .unwrap_or_else(|e| {
                        panic!(
                            "template '{}' entry '{name}' ({ty}) failed to validate: {e}",
                            t.name
                        )
                    });
            }
        }
    }

    #[test]
    fn scaffold_to_inject_errors_on_unknown_template() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let err = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some("nope"))
            .expect_err("unknown template name must surface as InvalidInput");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("nope"),
            "error should name the bad template: {err}"
        );
    }

    #[test]
    fn scaffold_to_inject_rejects_unknown_template_even_when_no_scaffold_would_fire() {
        // An unknown template name is a typo, full stop. We don't want the
        // user to silently get nothing when they intended to ask for a
        // template.
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");
        // Existing world with a renderer trigger: scaffolding wouldn't fire.
        std::fs::write(
            &world,
            r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
        )
        .unwrap();

        let err = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some("nope"))
            .expect_err("unknown template should fail fast regardless of scaffold path");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    // text-file targets

    #[test]
    fn text_file_becomes_text_label_with_content() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("greeting.txt");
        std::fs::write(&path, "Hello, world!").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["type"], "TextLabel");
        assert_eq!(entry["name"], "greeting");
        assert_eq!(entry["args"]["content"], "Hello, world!");
        // Mirrors `cn init`: short labels render centered by default.
        assert_eq!(entry["args"]["centered"], serde_json::json!(true));
    }

    #[test]
    fn text_file_strips_single_trailing_newline() {
        let dir = concinnity_testing::TempTree::new();
        let lf = dir.join("lf.txt");
        std::fs::write(&lf, "line one\nline two\n").unwrap();
        let crlf = dir.join("crlf.md");
        std::fs::write(&crlf, "line one\r\nline two\r\n").unwrap();

        let lf_entry = &entry_from_path(lf.to_str().unwrap()).unwrap()[0];
        assert_eq!(lf_entry["args"]["content"], "line one\nline two");
        let crlf_entry = &entry_from_path(crlf.to_str().unwrap()).unwrap()[0];
        assert_eq!(crlf_entry["args"]["content"], "line one\r\nline two");
    }

    #[test]
    fn audio_file_becomes_audio_clip() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("door_creak.wav");
        std::fs::write(&path, b"not-really-audio").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["type"], "AudioClip");
        assert_eq!(entry["name"], "door_creak");
        assert_eq!(entry["args"]["source"], path.to_str().unwrap());
    }

    #[test]
    fn hdr_file_becomes_an_environment_map() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("san_giuseppe_4k.hdr");
        std::fs::write(&path, b"not-really-radiance").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["type"], "EnvironmentMap");
        assert_eq!(entry["name"], "san_giuseppe_4k");
        assert_eq!(entry["args"]["source"], path.to_str().unwrap());
        // `source` and `generator` are mutually exclusive: the file source wins,
        // so the schema default leaves `generator` blank.
        assert_eq!(entry["args"]["generator"], "");
        // Registration defaults are materialized alongside the source.
        assert_eq!(entry["args"]["prefilter_face_size"], serde_json::json!(512));
    }

    // The entry a `.hdr` add produces is one the cook accepts.
    #[test]
    fn an_hdr_entry_validates_against_the_environment_map_schema() {
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("studio.hdr");
        std::fs::write(&path, b"radiance").unwrap();

        let entry = entry_from_path(path.to_str().unwrap()).unwrap().remove(0);
        concinnity_cook::validate_asset(
            "EnvironmentMap",
            "studio",
            &entry["args"],
            crate::cook_platform(),
        )
        .expect("a `.hdr` add must produce a cookable EnvironmentMap");
    }

    // try_retarget_environment_map

    // A world binds one lighting environment, so a second `.hdr` retargets the
    // existing map rather than appending one the runtime would ignore. Only the
    // source changes; the tuning the user set survives.
    #[test]
    fn try_retarget_environment_map_repoints_the_existing_map() {
        let mut assets = vec![serde_json::json!({
            "name": "env_sky",
            "type": "EnvironmentMap",
            "args": {
                "source": "assets/hdri/old.hdr",
                "generator": "",
                "prefilter_face_size": 1024,
                "prefilter_clamp": 4.0,
            }
        })];
        let entries = vec![serde_json::json!({
            "name": "studio",
            "type": "EnvironmentMap",
            "args": {"source": "assets/hdri/studio.hdr", "generator": ""}
        })];

        let out = try_retarget_environment_map(&mut assets, &entries);
        assert_eq!(
            out,
            Some(("env_sky".to_string(), "assets/hdri/studio.hdr".to_string()))
        );
        assert_eq!(assets.len(), 1, "no second map appended");
        let args = assets[0]["args"].as_object().unwrap();
        assert_eq!(args["source"], "assets/hdri/studio.hdr");
        // Hand-tuned knobs survive the retarget.
        assert_eq!(args["prefilter_face_size"], 1024);
        assert_eq!(args["prefilter_clamp"], 4.0);
        // The name is untouched, so anything referring to it still resolves.
        assert_eq!(assets[0]["name"], "env_sky");
    }

    // Retargeting a procedural map clears its generator: the two are mutually
    // exclusive and the cook rejects an entry carrying both.
    #[test]
    fn try_retarget_environment_map_clears_a_generator() {
        let mut assets = vec![serde_json::json!({
            "name": "env",
            "type": "EnvironmentMap",
            "args": {"source": "", "generator": "sky"}
        })];
        let entries = vec![serde_json::json!({
            "name": "dusk",
            "type": "EnvironmentMap",
            "args": {"source": "dusk.hdr", "generator": ""}
        })];

        assert!(try_retarget_environment_map(&mut assets, &entries).is_some());
        let args = assets[0]["args"].as_object().unwrap();
        assert_eq!(args["source"], "dusk.hdr");
        assert_eq!(args["generator"], "");
    }

    // The first map is the one the runtime binds, so it is the one retargeted.
    #[test]
    fn try_retarget_environment_map_takes_the_first_of_several() {
        let mut assets = vec![
            serde_json::json!({"name": "a", "type": "EnvironmentMap", "args": {"source": "a.hdr"}}),
            serde_json::json!({"name": "b", "type": "EnvironmentMap", "args": {"source": "b.hdr"}}),
        ];
        let entries = vec![
            serde_json::json!({"name": "c", "type": "EnvironmentMap", "args": {"source": "c.hdr"}}),
        ];

        let (name, _) = try_retarget_environment_map(&mut assets, &entries).unwrap();
        assert_eq!(name, "a");
        assert_eq!(assets[0]["args"]["source"], "c.hdr");
        assert_eq!(
            assets[1]["args"]["source"], "b.hdr",
            "the rest are untouched"
        );
    }

    // No existing map: the caller falls through to the normal append.
    #[test]
    fn try_retarget_environment_map_skips_a_world_without_one() {
        let mut assets = vec![serde_json::json!({
            "name": "lamp", "type": "PointLight", "args": {}
        })];
        let entries = vec![
            serde_json::json!({"name": "env", "type": "EnvironmentMap", "args": {"source": "e.hdr"}}),
        ];
        assert!(try_retarget_environment_map(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_retarget_environment_map_skips_other_types() {
        let mut assets = vec![serde_json::json!({
            "name": "env", "type": "EnvironmentMap", "args": {"source": "e.hdr"}
        })];
        let entries =
            vec![serde_json::json!({"name": "face", "type": "Font", "args": {"path": "face.ttf"}})];
        assert!(try_retarget_environment_map(&mut assets, &entries).is_none());
        assert_eq!(assets[0]["args"]["source"], "e.hdr");
    }

    // A fan-out target (a scene, a multi-stage shader) must never retarget.
    #[test]
    fn try_retarget_environment_map_skips_multi_entry() {
        let mut assets = vec![serde_json::json!({
            "name": "env", "type": "EnvironmentMap", "args": {"source": "e.hdr"}
        })];
        let entries = vec![
            serde_json::json!({"name": "a", "type": "EnvironmentMap", "args": {"source": "a.hdr"}}),
            serde_json::json!({"name": "b", "type": "EnvironmentMap", "args": {"source": "b.hdr"}}),
        ];
        assert!(try_retarget_environment_map(&mut assets, &entries).is_none());
    }

    #[test]
    fn markdown_with_frontmatter_becomes_story_import() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("crossroads.md");
        std::fs::write(&path, "---\ntitle: T\n---\n\n# a\n\nhi\n").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["type"], "StoryImport");
        assert_eq!(entry["name"], "crossroads");
        assert_eq!(entry["args"]["source"], path.to_str().unwrap());
        // Registration defaults are materialized alongside the source.
        assert_eq!(entry["args"]["title_screen"], serde_json::json!(true));
    }

    #[test]
    fn markdown_without_frontmatter_stays_a_text_label() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("notes.md");
        std::fs::write(&path, "# Notes\n\nplain markdown\n").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["type"], "TextLabel");
    }

    #[test]
    fn text_file_over_size_cap_errors() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("huge.txt");
        let blob = "a".repeat(TEXT_LABEL_MAX_BYTES + 1);
        std::fs::write(&path, blob).unwrap();

        let err = entry_from_path(path.to_str().unwrap())
            .expect_err("oversized text file must fail rather than silently truncate");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("capped"));
    }

    #[test]
    fn scaffold_to_inject_empty_for_missing_world_with_text_target() {
        // Text targets bootstrap a missing world via the TextLabel itself:
        // no separate scaffold needed, and the missing file must not error.
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "notes.txt", None).unwrap();
        assert!(scaffold.is_empty(), "text target should emit no scaffold");
    }

    #[test]
    fn scaffold_to_inject_empty_for_renderer_less_world_with_text_target() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");
        std::fs::write(&world, r#"{"name":"tex","type":"Texture","args":{}}"#).unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "notes.md", None).unwrap();
        assert!(
            scaffold.is_empty(),
            "text target never injects GLB scaffold"
        );
    }

    // try_refresh_text_label

    #[test]
    fn try_refresh_text_label_updates_content_in_place() {
        // Existing TextLabel was hand-edited: y, color, scale, centered all
        // diverged from defaults. A refresh should touch only `content`.
        let mut assets = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {
                "content": "Old text",
                "font": "Questrial-Regular",
                "x": 10.0,
                "y": 200.0,
                "color": [0.2, 0.8, 0.4],
                "scale": 1.5,
                "centered": false,
                "background": [0.0, 0.0, 0.0, 0.0],
                "padding": 0.0,
                "visible": true,
                "screen": null,
            }
        })];
        let entries = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {
                "content": "New text",
                "centered": true,
            }
        })];

        let refreshed = try_refresh_text_label(&mut assets, &entries);
        assert_eq!(refreshed.as_deref(), Some("greeting"));
        let args = assets[0]["args"].as_object().unwrap();
        assert_eq!(args["content"], "New text");
        // Hand edits survive: only `content` was copied across.
        assert_eq!(args["y"], 200.0);
        assert_eq!(args["color"], serde_json::json!([0.2, 0.8, 0.4]));
        assert_eq!(args["scale"], 1.5);
        assert_eq!(args["centered"], false);
    }

    #[test]
    fn try_refresh_text_label_skips_non_textlabel_new_entry() {
        let mut assets = vec![serde_json::json!({
            "name": "thing",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![serde_json::json!({
            "name": "thing",
            "type": "Font",
            "args": {"path": "x.ttf"}
        })];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_when_existing_is_different_type() {
        let mut assets = vec![serde_json::json!({
            "name": "thing",
            "type": "Font",
            "args": {"path": "x.ttf"}
        })];
        let entries = vec![serde_json::json!({
            "name": "thing",
            "type": "TextLabel",
            "args": {"content": "hi"}
        })];
        // Same name, different existing type: refresh shouldn't fire; caller
        // falls through to the duplicate-name error.
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_when_name_misses() {
        let mut assets = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![serde_json::json!({
            "name": "caption",
            "type": "TextLabel",
            "args": {"content": "new"}
        })];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_multi_entry() {
        // A fan-out target (e.g. a metal shader producing vert+frag) must
        // never trigger the in-place refresh path.
        let mut assets = vec![serde_json::json!({
            "name": "label",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![
            serde_json::json!({"name": "label", "type": "TextLabel", "args": {"content": "new"}}),
            serde_json::json!({"name": "other", "type": "TextLabel", "args": {"content": "other"}}),
        ];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn scaffold_to_inject_rejects_template_with_text_target() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("world.jsonl");

        let name = concinnity_cook::authoring::template::TEMPLATES[0].name;
        let err = scaffold_to_inject(world.to_str().unwrap(), "notes.txt", Some(name))
            .expect_err("--template should only apply to scene targets");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    // entry_from_inline_json

    #[test]
    fn inline_json_must_be_an_object() {
        let err = entry_from_inline_json("[1, 2]").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("must be an object"), "got: {err}");
    }

    #[test]
    fn inline_json_requires_a_type_field() {
        let err = entry_from_inline_json(r#"{"name":"thing"}"#).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("`type` field"), "got: {err}");
    }

    #[test]
    fn inline_json_rejects_malformed_text() {
        let err = entry_from_inline_json("{ nope").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn inline_json_rejects_build_config() {
        let err = entry_from_inline_json(r#"{"type":"BuildConfig"}"#).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // The underscore spelling is caught by the same normalisation.
        let err = entry_from_inline_json(r#"{"type":"build_config"}"#).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn inline_json_defaults_the_name_to_the_lowercased_type() {
        let entry = entry_from_inline_json(r#"{"type":"Window"}"#).unwrap();
        assert_eq!(entry["name"], "window");
        assert_eq!(entry["type"], "Window");
        // Default args are materialized as an object.
        assert!(entry["args"].is_object());
    }

    #[test]
    fn inline_json_keeps_an_explicit_name() {
        let entry = entry_from_inline_json(r#"{"type":"Window","name":"main"}"#).unwrap();
        assert_eq!(entry["name"], "main");
    }

    // entry_from_type_name

    #[test]
    fn type_name_builds_a_default_entry() {
        let entry = entry_from_type_name("Window").unwrap();
        assert_eq!(entry["name"], "window");
        assert_eq!(entry["type"], "Window");
        assert!(entry["args"].is_object());
    }

    #[test]
    fn type_name_rejects_build_config() {
        let err = entry_from_type_name("BuildConfig").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn type_name_rejects_an_unknown_type() {
        assert!(entry_from_type_name("NotARealAssetType").is_err());
    }

    // entry_from_json_file

    #[test]
    fn json_file_requires_a_type_field() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("thing.json");
        std::fs::write(&path, r#"{"name":"thing"}"#).unwrap();

        let err = entry_from_json_file(&path, "thing").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("no `type` field"), "got: {err}");
    }

    #[test]
    fn json_file_rejects_build_config() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("cfg.json");
        std::fs::write(&path, r#"{"type":"BuildConfig"}"#).unwrap();

        let err = entry_from_json_file(&path, "cfg").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn json_file_uses_the_stem_when_unnamed() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("main_window.json");
        std::fs::write(&path, r#"{"type":"Window","args":{}}"#).unwrap();

        let entry = entry_from_json_file(&path, "main_window").unwrap();
        assert_eq!(entry["name"], "main_window");
        assert_eq!(entry["type"], "Window");
    }

    #[test]
    fn json_file_read_and_parse_failures_are_reported() {
        let dir = concinnity_testing::TempTree::new();
        // Missing file.
        let missing = dir.join("missing.json");
        assert!(entry_from_json_file(&missing, "missing").is_err());
        // Unparseable content.
        let junk = dir.join("junk.json");
        std::fs::write(&junk, "{ nope").unwrap();
        let err = entry_from_json_file(&junk, "junk").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // resolve_add_target

    #[test]
    fn resolve_add_target_dispatches_type_names_and_inline_json() {
        let entries = resolve_add_target("Window").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["type"], "Window");

        let entries = resolve_add_target(r#"{"type":"Window","name":"main"}"#).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["name"], "main");
    }

    #[test]
    fn resolve_add_target_reports_an_unresolvable_target() {
        let err = resolve_add_target("DefinitelyNotAnAssetOrFile").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("could not resolve"), "got: {err}");
    }

    // Every extension the picker advertises is one `entry_from_path` actually
    // dispatches: each resolves, or fails for a content reason -- never with
    // "unknown extension". Guards the two lists against drifting apart.
    #[test]
    fn import_extension_groups_track_the_dispatch() {
        let dir = tempfile::tempdir().unwrap();
        for (_, exts) in IMPORT_EXTENSION_GROUPS {
            for ext in *exts {
                let path = dir.path().join(format!("probe.{ext}"));
                // Valid-enough content for the arms that read their file.
                std::fs::write(&path, b"{}").unwrap();
                let path_str = path.to_string_lossy();
                if let Err(e) = entry_from_path(&path_str) {
                    assert!(
                        !e.to_string().contains("unknown extension"),
                        ".{ext} is advertised by the picker but not dispatched: {e}"
                    );
                }
            }
        }
    }

    // A picker group is only useful if the dialog can build a filter from it.
    #[test]
    fn import_extension_groups_are_non_empty_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for (name, exts) in IMPORT_EXTENSION_GROUPS {
            assert!(!exts.is_empty(), "{name} has no extensions");
            for ext in *exts {
                assert!(seen.insert(*ext), "'{ext}' listed in two groups");
                assert!(
                    !ext.starts_with('.'),
                    "'{ext}' should be bare (rfd adds the dot)"
                );
            }
        }
    }

    // import_entry

    #[test]
    fn import_entry_sets_the_source_on_registration_defaults() {
        let entry = import_entry("SceneImport", "bistro", "scenes/bistro.fbx").unwrap();
        assert_eq!(entry["name"], "bistro");
        assert_eq!(entry["type"], "SceneImport");
        assert_eq!(entry["args"]["source"], "scenes/bistro.fbx");
    }

    #[test]
    fn import_entry_rejects_an_unknown_type() {
        assert!(import_entry("NotARealAssetType", "x", "x.glb").is_err());
    }

    // scene / panorama split
    //
    // Which side of the split a file lands on is
    // `concinnity_cook::import::panorama`'s call and is covered against real panorama
    // and multi-mesh fixtures there; these pin what each answer produces here.

    #[test]
    fn a_glb_that_is_not_a_panorama_imports_as_geometry() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("scene.glb");
        std::fs::write(&path, b"not a panorama").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["type"], "SceneImport");
        assert_eq!(entries[0]["args"]["source"], path.to_str().unwrap());
    }

    #[test]
    fn a_panorama_becomes_an_environment_map_naming_its_source() {
        let entry = environment_map_entry("galaxy", "assets/hdri/galaxy.glb").unwrap();
        assert_eq!(entry["name"], "galaxy");
        assert_eq!(entry["type"], "EnvironmentMap");
        assert_eq!(entry["args"]["source"], "assets/hdri/galaxy.glb");
    }

    #[test]
    fn an_hdr_still_becomes_an_environment_map() {
        let dir = concinnity_testing::TempTree::new();
        let path = dir.join("studio.hdr");
        std::fs::write(&path, b"#?RADIANCE\n").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries[0]["type"], "EnvironmentMap");
    }

    #[test]
    fn an_fbx_never_takes_the_panorama_branch() {
        let entries = entry_from_path("scenes/bistro.fbx").unwrap();
        assert_eq!(entries[0]["type"], "SceneImport");
    }

    // ensure_world_file_exists

    #[test]
    fn ensure_world_file_exists_creates_parents_and_is_idempotent() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.join("nested").join("deeper").join("world.jsonl");

        ensure_world_file_exists(world.to_str().unwrap()).unwrap();
        assert!(world.exists());
        assert_eq!(std::fs::read_to_string(&world).unwrap(), "");

        // A second call leaves the existing file alone.
        std::fs::write(&world, "content").unwrap();
        ensure_world_file_exists(world.to_str().unwrap()).unwrap();
        assert_eq!(std::fs::read_to_string(&world).unwrap(), "content");
    }

    // is_path_like

    #[test]
    fn is_path_like_recognises_separators_prefixes_and_dotted_files() {
        assert!(is_path_like("models/scene.glb"));
        assert!(is_path_like("models\\scene.glb"));
        assert!(is_path_like("./scene.glb"));
        assert!(is_path_like("~/scene.glb"));
        assert!(is_path_like("scene.glb"));
        // A bare known type name is not a path, even without a dot.
        assert!(!is_path_like("Window"));
    }
}
