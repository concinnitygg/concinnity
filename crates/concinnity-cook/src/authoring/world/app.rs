use super::line::parse_entry;
use crate::authoring::registry::RegisteredType;
use concinnity_engine::components::cook::AppConfig;

/// The file `concinnity init`, and a first `concinnity add` in a directory
/// with no application, start an application in.
pub const APP_JSONL: &str = "app.jsonl";

/// The name of the application `text` declares: the `name` of its own
/// `AppConfig` line, or the default name when the line leaves it unset. `None`
/// when no line of `text` declares an `AppConfig`.
///
/// Only the file's own lines count, not what it includes, so a file that
/// includes another application's file is not an application itself. Lines
/// that do not parse are skipped; validating them is `check`'s job.
pub fn declared_app_name(text: &str) -> Option<String> {
    text.lines()
        .filter_map(|line| parse_entry(line.trim()).ok())
        .find(|entry| {
            entry.get("type").and_then(serde_json::Value::as_str)
                == Some(RegisteredType::AppConfig.as_str())
        })
        .map(|entry| {
            let args: AppConfig = entry
                .get("args")
                .and_then(|args| serde_json::from_value(args.clone()).ok())
                .unwrap_or_default();
            if args.name.is_empty() {
                AppConfig::default().name
            } else {
                args.name
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_with_an_app_config_line_is_named_by_it() {
        let text = "// Arena\n[\"Window\",{}]\n[\"AppConfig\",{\"name\":\"Arena\"}]\n";
        assert_eq!(declared_app_name(text).as_deref(), Some("Arena"));
    }

    #[test]
    fn an_unnamed_app_config_takes_the_default_name() {
        let default = AppConfig::default().name;
        assert_eq!(
            declared_app_name("[\"AppConfig\"]\n"),
            Some(default.clone())
        );
        assert_eq!(
            declared_app_name("[\"AppConfig\",{\"name\":\"\"}]\n"),
            Some(default)
        );
    }

    #[test]
    fn a_file_without_an_app_config_line_is_not_an_application() {
        assert_eq!(declared_app_name(""), None);
        assert_eq!(
            declared_app_name("[\"Include\",{\"path\":\"app.jsonl\"}]\n"),
            None
        );
    }

    // A malformed line elsewhere does not hide the application, and a malformed
    // AppConfig line does not declare one.
    #[test]
    fn lines_that_do_not_parse_are_skipped() {
        let text = "{ not json\n[\"AppConfig\",{\"name\":\"Kept\"}]\n";
        assert_eq!(declared_app_name(text).as_deref(), Some("Kept"));
        assert_eq!(declared_app_name("[\"AppConfig\",{\"name\":\n"), None);
    }
}
