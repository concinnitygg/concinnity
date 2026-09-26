use concinnity_core::components::{ColorRun, color_run_error};

// The two fields the check reads; the rest of the label parses when it bakes.
#[derive(serde::Deserialize)]
struct Runs {
    #[serde(default)]
    content: String,
    #[serde(default)]
    color_runs: Vec<ColorRun>,
}

/// Check a `TextLabel`'s authored args: its color runs lie inside its content,
/// in order, and do not overlap.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let Ok(runs) = serde_json::from_value::<Runs>(args.clone()) else {
        return Ok(());
    };
    match color_run_error(&runs.content, &runs.color_runs) {
        Some(e) => Err(format!("Asset '{name}': TextLabel {e}")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(start: u32, length: u32) -> serde_json::Value {
        json!({"start": start, "length": length, "color": [1, 0, 0]})
    }

    #[test]
    fn a_label_without_runs_passes() {
        assert!(check("t", &json!({"content": "hi"})).is_ok());
        assert!(check("t", &json!({})).is_ok());
    }

    #[test]
    fn runs_inside_the_content_pass() {
        let args = json!({"content": "Mara: hi", "color_runs": [run(0, 4), run(6, 2)]});
        assert!(check("t", &args).is_ok());
    }

    #[test]
    fn a_run_past_the_content_names_the_asset_and_the_run() {
        let args = json!({"content": "hi", "color_runs": [run(0, 1), run(1, 5)]});
        let err = check("title", &args).unwrap_err();
        assert!(err.contains("'title'"), "{err}");
        assert!(err.contains("color_runs[1]"), "{err}");
    }

    #[test]
    fn overlapping_runs_fail() {
        let args = json!({"content": "hello", "color_runs": [run(0, 3), run(2, 1)]});
        let err = check("t", &args).unwrap_err();
        assert!(
            err.contains("color_runs[1]") && err.contains("overlap"),
            "{err}"
        );
    }
}
