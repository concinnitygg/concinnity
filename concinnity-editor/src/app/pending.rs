// src/app/pending.rs
//
// In-memory staging area for the active scene.
//
// While the app is connected to the server via `cn debug --websocket`, the
// agentic loop issues add / rm / load commands that modify this pending list.
// No disk write or build is triggered until `save` is called.
//
// On first access the list is seeded from the current world.jsonl so that
// the LLM can extend or modify an existing scene incrementally.

// Driven only by the binary's debug WebSocket channel; unreferenced in the
// FFI lib build.
#![allow(dead_code)]

use crate::world::{find_world_jsonl, parse_world_jsonl, write_world_jsonl};
use serde_json::Value;
use std::sync::Mutex;

static PENDING: Mutex<Option<Vec<Value>>> = Mutex::new(None);

fn ensure_initialized(guard: &mut std::sync::MutexGuard<Option<Vec<Value>>>) {
    if guard.is_none() {
        let assets = find_world_jsonl(None)
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|content| parse_world_jsonl(&content).ok())
            .unwrap_or_default();
        **guard = Some(assets);
    }
}

pub fn add(asset_type: String, name: Option<String>, args: Option<Value>) -> Result<(), String> {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized(&mut guard);
    let list = guard.as_mut().unwrap();

    let name = name.unwrap_or_else(|| asset_type.to_lowercase());

    if list
        .iter()
        .any(|a| a.get("name").and_then(|v| v.as_str()) == Some(&name))
    {
        return Err(format!(
            "asset '{}' already exists in pending scene; rm it first",
            name
        ));
    }

    let args = args.unwrap_or_else(|| Value::Object(Default::default()));
    list.push(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": args,
    }));
    tracing::debug!("pending add: {} ({})", name, asset_type);
    Ok(())
}

pub fn rm(name: &str) -> Result<(), String> {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized(&mut guard);
    let list = guard.as_mut().unwrap();

    let before = list.len();
    list.retain(|a| a.get("name").and_then(|v| v.as_str()) != Some(name));

    if list.len() == before {
        Err(format!("no asset named '{}' in pending scene", name))
    } else {
        tracing::debug!("pending rm: {}", name);
        Ok(())
    }
}

pub fn load(assets: Vec<Value>) {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(assets);
    tracing::debug!("pending load: {} assets", guard.as_ref().unwrap().len());
}

pub fn save() -> std::io::Result<()> {
    let guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let list = match guard.as_ref() {
        Some(l) => l,
        None => return Ok(()),
    };

    // Use existing world.jsonl path or fall back to cwd
    let world_path = find_world_jsonl(None).unwrap_or_else(|_| "world.jsonl".to_string());

    let content = write_world_jsonl(list).map_err(|e| std::io::Error::other(e.to_string()))?;

    std::fs::write(&world_path, &content)?;
    tracing::info!(
        "saved pending scene ({} assets) to {}",
        list.len(),
        world_path
    );
    Ok(())
}

// Serialises every test that touches the process-global PENDING scene, both
// here and in app::commands. Cargo runs tests in parallel threads within one
// binary, so unsynchronised access would interleave staged scenes.
#[cfg(test)]
pub(crate) mod test_support {
    pub(crate) static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) fn lock() -> std::sync::MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every test resets the staged scene through `load` first, which both
    // gives a known baseline and keeps `ensure_initialized` from reading
    // whatever world.jsonl the test runner's cwd happens to contain.

    #[test]
    fn add_rejects_a_duplicate_name() {
        let _guard = test_support::lock();
        load(vec![serde_json::json!({
            "name": "crate", "type": "Prop", "args": {}
        })]);
        let err = add("Prop".to_string(), Some("crate".to_string()), None).unwrap_err();
        assert!(err.contains("already exists"), "unexpected error: {err}");
    }

    #[test]
    fn add_defaults_the_name_to_the_lowercased_type() {
        let _guard = test_support::lock();
        load(Vec::new());
        add("Prop".to_string(), None, None).unwrap();
        // A second default-named add collides, proving the first landed as
        // "prop".
        let err = add("Prop".to_string(), Some("prop".to_string()), None).unwrap_err();
        assert!(err.contains("'prop'"), "unexpected error: {err}");
    }

    #[test]
    fn add_stores_supplied_args() {
        let _guard = test_support::lock();
        load(Vec::new());
        add(
            "Prop".to_string(),
            Some("table".to_string()),
            Some(serde_json::json!({"position": [1, 2, 3]})),
        )
        .unwrap();
        // Present: removing it succeeds.
        rm("table").unwrap();
    }

    #[test]
    fn rm_removes_a_staged_asset_once() {
        let _guard = test_support::lock();
        load(vec![serde_json::json!({
            "name": "lamp", "type": "Prop", "args": {}
        })]);
        rm("lamp").unwrap();
        let err = rm("lamp").unwrap_err();
        assert!(err.contains("no asset named"), "unexpected error: {err}");
    }

    #[test]
    fn rm_of_an_unknown_name_errors() {
        let _guard = test_support::lock();
        load(Vec::new());
        assert!(rm("ghost").is_err());
    }

    #[test]
    fn load_replaces_the_whole_scene() {
        let _guard = test_support::lock();
        load(vec![serde_json::json!({
            "name": "old", "type": "Prop", "args": {}
        })]);
        load(vec![serde_json::json!({
            "name": "new", "type": "Prop", "args": {}
        })]);
        assert!(rm("old").is_err());
        rm("new").unwrap();
    }
}
