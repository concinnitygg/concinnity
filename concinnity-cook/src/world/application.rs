// src/world/application.rs
// The Application asset names the world for distribution. At most one may be
// declared. When a Window carries no title of its own, the application name
// fills it, so a running game shows its own name in the title bar.

use super::expand::{ExpandReport, asset_name, type_norm};

// Enforce the single-Application rule and feed its name into any untitled
// Window. Runs after the first companion round, so a rendering world's
// companion-injected Window is present to receive the title. The Application
// asset itself is External and stays in the world (the runtime and the export
// both read it); only the Window title is adjusted here.
pub(crate) fn apply_application(
    assets: &mut [serde_json::Value],
    report: &mut ExpandReport,
) -> Result<(), String> {
    let mut app_name: Option<String> = None;
    let mut first: Option<String> = None;
    for v in assets.iter() {
        if type_norm(v) != "application" {
            continue;
        }
        let name = asset_name(v);
        if let Some(prev) = &first {
            return Err(format!(
                "Application '{}': the world already declares Application '{}'; \
                 declare at most one",
                name, prev
            ));
        }
        first = Some(name);
        app_name = v
            .get("args")
            .and_then(|a| a.get("name"))
            .and_then(|n| n.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
    }

    // Nothing to feed when there is no Application or its name is empty.
    let Some(app_name) = app_name else {
        return Ok(());
    };

    // Fill the title of every Window that authored none. A companion-injected
    // Window arrives with empty args, so a rendering world's window picks up the
    // application name; an authored `title` is left untouched.
    for v in assets.iter_mut() {
        if type_norm(v) != "window" {
            continue;
        }
        let name = asset_name(v);
        {
            let args = v
                .get_mut("args")
                .and_then(|a| a.as_object_mut())
                .ok_or_else(|| format!("Window '{}': args must be an object", name))?;
            if args.contains_key("title") {
                continue;
            }
            args.insert(
                "title".to_string(),
                serde_json::Value::String(app_name.clone()),
            );
        }
        // Keep the lock's copy of a companion-injected Window in sync, so
        // world-lock.json shows the title the blob carries.
        if let Some(entry) = report
            .injected
            .iter_mut()
            .find(|i| i.name == name && i.asset_type == "Window")
        {
            entry.args = v
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(name: &str) -> serde_json::Value {
        serde_json::json!({"name":"app","type":"Application","args":{"name": name}})
    }

    fn window(args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name":"w","type":"Window","args": args})
    }

    fn title_of(assets: &[serde_json::Value]) -> Option<String> {
        assets
            .iter()
            .find(|v| type_norm(v) == "window")
            .and_then(|v| v.get("args"))
            .and_then(|a| a.get("title"))
            .and_then(|t| t.as_str())
            .map(str::to_string)
    }

    #[test]
    fn name_fills_an_untitled_window() {
        let mut assets = vec![app("My Game"), window(serde_json::json!({}))];
        let mut report = ExpandReport::default();
        apply_application(&mut assets, &mut report).unwrap();
        assert_eq!(title_of(&assets).as_deref(), Some("My Game"));
    }

    #[test]
    fn authored_title_is_not_overridden() {
        let mut assets = vec![app("My Game"), window(serde_json::json!({"title": "Kept"}))];
        let mut report = ExpandReport::default();
        apply_application(&mut assets, &mut report).unwrap();
        assert_eq!(title_of(&assets).as_deref(), Some("Kept"));
    }

    #[test]
    fn no_application_leaves_the_window_untitled() {
        let mut assets = vec![window(serde_json::json!({}))];
        let mut report = ExpandReport::default();
        apply_application(&mut assets, &mut report).unwrap();
        assert_eq!(title_of(&assets), None);
    }

    #[test]
    fn empty_application_name_fills_nothing() {
        let mut assets = vec![app(""), window(serde_json::json!({}))];
        let mut report = ExpandReport::default();
        apply_application(&mut assets, &mut report).unwrap();
        assert_eq!(title_of(&assets), None);
    }

    #[test]
    fn duplicate_application_is_an_error() {
        let mut assets = vec![
            serde_json::json!({"name":"a","type":"Application","args":{"name":"A"}}),
            serde_json::json!({"name":"b","type":"Application","args":{"name":"B"}}),
        ];
        let mut report = ExpandReport::default();
        let err = apply_application(&mut assets, &mut report).unwrap_err();
        assert!(err.contains("at most one"));
    }

    #[test]
    fn injected_window_args_are_synced_in_the_report() {
        let mut assets = vec![app("My Game"), window(serde_json::json!({}))];
        let mut report = ExpandReport::default();
        report.record("w", "Window", serde_json::json!({}), "companion");
        apply_application(&mut assets, &mut report).unwrap();
        let entry = report
            .injected
            .iter()
            .find(|i| i.name == "w" && i.asset_type == "Window")
            .unwrap();
        assert_eq!(entry.args["title"], serde_json::json!("My Game"));
    }
}
