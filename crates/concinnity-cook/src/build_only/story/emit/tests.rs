use super::*;
use crate::build_only::story::parse::parse_story;

fn dims(_path: &str) -> Result<(u32, u32), String> {
    Ok((456, 700))
}

fn collect_names<'a>(value: &'a serde_json::Value, out: &mut Vec<&'a str>) {
    match value {
        serde_json::Value::String(name) => out.push(name),
        serde_json::Value::Array(items) => items.iter().for_each(|v| collect_names(v, out)),
        serde_json::Value::Object(map) => map.values().for_each(|v| collect_names(v, out)),
        _ => {}
    }
}

#[test]
fn every_scaffold_name_resolves_to_an_emitted_entry() {
    let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n- [x](#a)\n- [y](#a)\n";
    let story = parse_story(src).unwrap();
    for title_screen in [true, false] {
        let entries = emit_story("s", &story, title_screen, 45.0, &dims).unwrap();
        let scaffold = &entries
            .iter()
            .find(|e| e["type"] == "Story")
            .expect("the Story entry")["args"]["scaffold"];
        let mut names = Vec::new();
        collect_names(scaffold, &mut names);
        assert!(names.len() > 20, "{names:?}");
        for name in names {
            assert!(
                entries.iter().any(|e| e["name"] == name),
                "scaffold names '{name}', which no entry declares"
            );
        }
    }
}

// Every emitted overlay element names the screen it plays on: the title screen,
// the stage (including the overlay and choice furniture), or the ending.
#[test]
fn every_overlay_element_names_its_screen() {
    let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n- [x](#a)\n- [y](#a)\n";
    let story = parse_story(src).unwrap();
    let entries = emit_story("s", &story, true, 45.0, &dims).unwrap();

    let screens: Vec<&str> = entries
        .iter()
        .filter(|e| e["type"] == "Screen")
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert_eq!(screens.len(), 3, "title, stage and ending: {screens:?}");

    for entry in &entries {
        let overlay = matches!(
            entry["type"].as_str(),
            Some("Sprite" | "TextLabel" | "TextInput" | "HitRegion" | "ScrollPanel")
        );
        if !overlay {
            continue;
        }
        let screen = entry["args"]["screen"].as_str().unwrap_or_default();
        assert!(
            screens.contains(&screen),
            "{} names screen {screen:?}, which the story does not emit",
            entry["name"]
        );
    }
}
