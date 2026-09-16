use super::helpers::{split_top_level, unquote};
use super::model::{BlockCharacter, Character, Story};

// The frontmatter is a deliberately strict YAML subset: a `title` line and a
// `characters:` block whose entries take three forms:
//
//   keeper: Innkeeper                                  (name only)
//   ayame: { name: Ayame, color: [1.0, 0.85, 0.8] }    (flow map)
//   ayame:                                             (block map)
//     name: Ayame
//     color: [1.0, 0.85, 0.8]
//
// Names may be quoted or plain; `color` values are JSON arrays.
pub(super) fn parse_frontmatter(text: &str, story: &mut Story) -> Result<(), String> {
    let mut in_characters = false;
    // Indent of character-id lines, fixed by the first one seen.
    let mut id_indent: Option<usize> = None;
    // A block-form character collects its indented fields until a line at or
    // below the id indent closes it.
    let mut block: Option<BlockCharacter> = None;

    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let err = |msg: String| Err(format!("frontmatter line {}: {}", i + 1, msg));
        let indent = raw.len() - raw.trim_start().len();

        if indent == 0 {
            finish_block_character(&mut block, story)?;
            in_characters = false;
            id_indent = None;
            let Some((key, value)) = line.split_once(':') else {
                return err(format!("expected `key: value`, got '{}'", line));
            };
            match key.trim() {
                "title" => story.title = unquote(value.trim()).to_string(),
                "background" => story.background = Some(unquote(value.trim()).to_string()),
                "characters" => {
                    if !value.trim().is_empty() {
                        return err("`characters` takes an indented block".to_string());
                    }
                    in_characters = true;
                }
                other => {
                    return err(format!(
                        "unknown key '{}'; supported keys are `title`, `background`, and \
                         `characters`",
                        other
                    ));
                }
            }
            continue;
        }

        if !in_characters {
            return err(format!("unexpected indented line '{}'", line.trim()));
        }

        // Deeper than the id indent: a field of the open block character.
        if indent > *id_indent.get_or_insert(indent) {
            let Some(b) = block.as_mut() else {
                return err(format!(
                    "unexpected indented line '{}'; character fields need an `id:` line \
                     above them",
                    line.trim()
                ));
            };
            let Some((key, val)) = line.trim().split_once(':') else {
                return err(format!("expected `key: value`, got '{}'", line.trim()));
            };
            let val = val.trim();
            match key.trim() {
                "name" => b.name = Some(parse_name_value(val).map_err(&err_str(i))?),
                "color" => b.color = parse_color_value(val).map_err(&err_str(i))?,
                other => return err(format!("unknown character key '{}'", other)),
            }
            continue;
        }

        finish_block_character(&mut block, story)?;

        let Some((id, value)) = line.trim().split_once(':') else {
            return err(format!("expected `id: ...`, got '{}'", line.trim()));
        };
        let id = unquote(id.trim()).to_string();
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return err(format!(
                "character id '{}' must be alphanumeric ('_' and '-' allowed)",
                id
            ));
        }
        let value = value.trim();
        if value.is_empty() {
            block = Some(BlockCharacter {
                id,
                line: i + 1,
                name: None,
                color: [1.0, 1.0, 1.0],
            });
        } else {
            let character = parse_character(value)
                .map_err(|e| format!("frontmatter line {}: character '{}': {}", i + 1, id, e))?;
            story.characters.insert(id, character);
        }
    }
    finish_block_character(&mut block, story)?;
    Ok(())
}

fn finish_block_character(
    block: &mut Option<BlockCharacter>,
    story: &mut Story,
) -> Result<(), String> {
    let Some(b) = block.take() else {
        return Ok(());
    };
    let Some(name) = b.name else {
        return Err(format!(
            "frontmatter line {}: character '{}': missing `name`",
            b.line, b.id
        ));
    };
    story.characters.insert(
        b.id,
        Character {
            name,
            color: b.color,
        },
    );
    Ok(())
}

fn err_str(i: usize) -> impl Fn(String) -> String {
    move |e| format!("frontmatter line {}: {}", i + 1, e)
}

fn parse_character(value: &str) -> Result<Character, String> {
    if !value.starts_with('{') {
        let name = unquote(value).to_string();
        if name.is_empty() {
            return Err("expected a display name or `{ name: ..., color: [...] }`".to_string());
        }
        return Ok(Character {
            name,
            color: [1.0, 1.0, 1.0],
        });
    }

    let inner = value
        .strip_prefix('{')
        .and_then(|v| v.strip_suffix('}'))
        .ok_or_else(|| "unterminated `{ ... }`".to_string())?;

    let mut name = None;
    let mut color = [1.0, 1.0, 1.0];
    for field in split_top_level(inner) {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let Some((key, val)) = field.split_once(':') else {
            return Err(format!("expected `key: value`, got '{}'", field));
        };
        let val = val.trim();
        match unquote(key.trim()) {
            "name" => name = Some(parse_name_value(val)?),
            "color" => color = parse_color_value(val)?,
            other => return Err(format!("unknown key '{}'", other)),
        }
    }
    let name = name.ok_or_else(|| "missing `name`".to_string())?;
    Ok(Character { name, color })
}

// A display name: quoted (JSON string) or plain text.
fn parse_name_value(val: &str) -> Result<String, String> {
    let name = if val.starts_with('"') {
        serde_json::from_str::<String>(val)
            .map_err(|_| format!("`name` has unbalanced quotes: {}", val))?
    } else {
        val.to_string()
    };
    if name.is_empty() {
        return Err("`name` must not be empty".to_string());
    }
    Ok(name)
}

fn parse_color_value(val: &str) -> Result<[f32; 3], String> {
    let parsed: Vec<f32> = serde_json::from_str(val)
        .map_err(|_| format!("`color` must be `[r, g, b]`, got '{}'", val))?;
    let [r, g, b] = parsed[..] else {
        return Err(format!(
            "`color` must have 3 components, got {}",
            parsed.len()
        ));
    };
    Ok([r, g, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_only::story::parse::parse_story;

    // Wrap a frontmatter block in a minimal valid body. Frontmatter errors
    // surface before the body is validated, so a mis-set-up test still fails
    // loudly rather than passing on the wrong error.
    fn with_body(frontmatter: &str) -> String {
        format!("---\n{frontmatter}\n---\n\n# a\n\nhi\n")
    }

    // Frontmatter scalar-value parsers reached directly.

    #[test]
    fn name_value_parses_quoted_and_plain() {
        assert_eq!(parse_name_value("\"Ayame Doe\"").unwrap(), "Ayame Doe");
        assert_eq!(parse_name_value("Plain Name").unwrap(), "Plain Name");
    }

    #[test]
    fn name_value_rejects_unbalanced_quotes_and_empty() {
        assert!(
            parse_name_value("\"unterminated")
                .unwrap_err()
                .contains("unbalanced quotes")
        );
        assert!(
            parse_name_value("\"\"")
                .unwrap_err()
                .contains("must not be empty")
        );
    }

    #[test]
    fn color_value_parses_a_triple() {
        assert_eq!(
            parse_color_value("[1.0, 0.5, 0.25]").unwrap(),
            [1.0, 0.5, 0.25]
        );
    }

    #[test]
    fn color_value_rejects_non_arrays_and_wrong_lengths() {
        // A non-array JSON value trips the serde branch; a well-formed array
        // of the wrong length trips the component-count check.
        assert!(parse_color_value("nope").unwrap_err().contains("[r, g, b]"));
        assert!(
            parse_color_value("[1, 2]")
                .unwrap_err()
                .contains("3 components")
        );
    }

    #[test]
    fn inline_character_parses_and_reports_its_own_errors() {
        let ok = parse_character("{ name: \"Bo\", color: [1, 1, 1] }").unwrap();
        assert_eq!(ok.name, "Bo");
        let plain = parse_character("Bob").unwrap();
        assert_eq!((plain.name.as_str(), plain.color), ("Bob", [1.0, 1.0, 1.0]));

        assert!(
            parse_character("{ name: X")
                .unwrap_err()
                .contains("unterminated")
        );
        assert!(
            parse_character("{ name: X, bad }")
                .unwrap_err()
                .contains("key: value")
        );
        assert!(
            parse_character("{ name: X, voice: low }")
                .unwrap_err()
                .contains("unknown key")
        );
        assert!(
            parse_character("{ color: [1, 1, 1] }")
                .unwrap_err()
                .contains("missing `name`")
        );
        assert!(
            parse_character("\"\"")
                .unwrap_err()
                .contains("display name")
        );
    }

    // Frontmatter branches via full parses.

    #[test]
    fn frontmatter_unknown_key_is_an_error() {
        let err = parse_story(&with_body("title: T\nfoo: bar")).unwrap_err();
        assert!(err.contains("unknown key 'foo'"), "{err}");
    }

    #[test]
    fn characters_key_with_inline_value_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters: stuff")).unwrap_err();
        assert!(err.contains("indented block"), "{err}");
    }

    #[test]
    fn top_level_line_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\nnocolon")).unwrap_err();
        assert!(err.contains("key: value"), "{err}");
    }

    #[test]
    fn stray_indented_line_outside_characters_is_an_error() {
        let err = parse_story(&with_body("title: T\n  indented")).unwrap_err();
        assert!(err.contains("unexpected indented line"), "{err}");
    }

    #[test]
    fn character_field_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  a:\n    nocolon")).unwrap_err();
        assert!(err.contains("key: value"), "{err}");
    }

    #[test]
    fn character_id_line_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  nocolon")).unwrap_err();
        assert!(err.contains("id: ..."), "{err}");
    }

    #[test]
    fn invalid_character_id_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  bad id: Name")).unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn inline_character_missing_name_reports_its_id() {
        let err = parse_story(&with_body(
            "title: T\ncharacters:\n  a: { color: [1, 1, 1] }",
        ))
        .unwrap_err();
        assert!(err.contains("missing `name`"), "{err}");
        assert!(err.contains("character 'a'"), "{err}");
    }
}
