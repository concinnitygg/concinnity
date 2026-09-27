//! The data half of an SdfVolume form's extras. Creating a volume chooses its
//! kind and a starter for the new field file the volume reads, which the kind
//! decides the `volumetric` flag of; editing one shows its field and the other
//! volumes reading the same file. Every other flag is a plain field of the
//! volume's form.

use super::form_extras::{ExtraControl, ExtraRow};
use super::sdf_field_list::declared_fields;
use super::sdf_templates;
use super::shader_edit::check_name;
use super::shader_kind::{KIND_ROW, ShaderKind};

const STARTER: usize = 6;

// What the form edits: a volume to create, or the declared one it opened on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdfMode {
    New {
        kind: ShaderKind,
        starter: usize,
    },
    Edit {
        declared: String,
        others: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdfForm {
    pub(crate) mode: SdfMode,
}

impl SdfForm {
    // The form over `entries`, editing the volume at `editing` or a new one;
    // `resolve` finds a field's file on disk, so two spellings of one file
    // count as one.
    pub(crate) fn open(
        entries: &[serde_json::Value],
        editing: Option<usize>,
        resolve: impl FnMut(&str) -> String,
    ) -> Self {
        let Some(i) = editing else {
            return Self {
                mode: SdfMode::New {
                    kind: ShaderKind::SdfField,
                    starter: 0,
                },
            };
        };
        let declared = entries[i]
            .get("args")
            .and_then(|a| a.get("fragment_shader"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let name = concinnity_cook::authoring::world::entry_handles(entries).swap_remove(i);
        let others = declared_fields(entries, resolve)
            .into_iter()
            .find(|f| f.volumes.iter().any(|v| Some(&v.name) == name.as_ref()))
            .map(|f| {
                f.volumes
                    .into_iter()
                    .map(|v| v.name)
                    .filter(|n| Some(n) != name.as_ref())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            mode: SdfMode::Edit { declared, others },
        }
    }

    pub(crate) fn is_new(&self) -> bool {
        matches!(self.mode, SdfMode::New { .. })
    }

    pub(crate) fn volumetric(&self) -> bool {
        matches!(self.mode, SdfMode::New { kind, .. } if kind.volumetric())
    }

    // The new field file's text.
    pub(crate) fn starter_text(&self) -> Option<&'static str> {
        match self.mode {
            SdfMode::New { kind, starter } => Some(sdf_templates::text(kind.volumetric(), starter)),
            SdfMode::Edit { .. } => None,
        }
    }

    pub(crate) fn press(&mut self, id: usize) {
        let SdfMode::New { kind, starter } = &mut self.mode else {
            return;
        };
        match id {
            KIND_ROW => {
                *kind = kind.next();
                *starter = 0;
            }
            STARTER => *starter = (*starter + 1) % sdf_templates::of(kind.volumetric()).len(),
            _ => {}
        }
    }

    // The type the form should become, after a press chose a kind of another.
    pub(crate) fn switch_type(&self) -> Option<&'static str> {
        match self.mode {
            SdfMode::New { kind, .. } => kind.switch_from(super::sdf_field_list::SDF_VOLUME),
            SdfMode::Edit { .. } => None,
        }
    }

    // The schema fields the extras decide: a new volume's file and flag.
    pub(crate) fn hidden_fields(&self) -> &'static [&'static str] {
        match self.mode {
            SdfMode::New { .. } => &["fragment_shader", "volumetric"],
            SdfMode::Edit { .. } => &[],
        }
    }

    // The rows, with `file` the path a new volume's field would be written to.
    pub(crate) fn rows(&self, file: &str) -> Vec<ExtraRow> {
        match &self.mode {
            SdfMode::New { kind, starter } => vec![
                kind.row(),
                ExtraRow {
                    id: STARTER,
                    caption: "Starter".to_string(),
                    indent: false,
                    control: ExtraControl::Choice {
                        options: sdf_templates::of(kind.volumetric())
                            .iter()
                            .map(|t| t.name.to_string())
                            .collect(),
                        selected: *starter,
                    },
                    detail: None,
                },
                ExtraRow::label("Field file", Some(file.to_string())),
            ],
            SdfMode::Edit { declared, others } => {
                let mut out = vec![ExtraRow::label("Field file", Some(declared.clone()))];
                let none = others.is_empty().then(|| "no other volume".to_string());
                out.push(ExtraRow::label("Also read by", none));
                out.extend(
                    others
                        .iter()
                        .map(|n| ExtraRow::label(n.clone(), None).indented()),
                );
                out
            }
        }
    }

    // Why a new volume cannot be created under `name`: its field file takes
    // the name, so it must be one, and one no other entry declares (`taken`).
    pub(crate) fn blocked(&self, name: &str, taken: bool) -> Option<String> {
        if !self.is_new() {
            return None;
        }
        let name = match check_name(name, "volume") {
            Ok(name) => name,
            Err(reason) => return Some(reason),
        };
        taken.then(|| format!("'{name}' is already taken; choose another name."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn volume(name: &str, file: &str) -> serde_json::Value {
        json!({"type": "SdfVolume", "args": {"$id": name, "fragment_shader": file}})
    }

    fn captions(rows: &[ExtraRow]) -> Vec<&str> {
        rows.iter().map(|r| r.caption.as_str()).collect()
    }

    // A new volume starts as a surface field on its first starter, hides the
    // file and flag it decides, and names the file it would write.
    #[test]
    fn a_new_volume_chooses_its_kind_and_starter() {
        let mut form = SdfForm::open(&[], None, str::to_string);
        assert!(form.is_new() && !form.volumetric());
        assert_eq!(form.hidden_fields(), ["fragment_shader", "volumetric"]);
        let rows = form.rows("shaders/blob.hlsl");
        assert_eq!(captions(&rows), ["Kind", "Starter", "Field file"]);
        assert_eq!(rows[2].detail.as_deref(), Some("shaders/blob.hlsl"));
        assert_eq!(form.starter_text(), Some(sdf_templates::SURFACE[0].text));

        form.press(STARTER);
        assert_eq!(form.starter_text(), Some(sdf_templates::SURFACE[1].text));
        form.press(KIND_ROW);
        assert!(form.volumetric());
        assert_eq!(
            form.starter_text(),
            Some(sdf_templates::VOLUMETRIC[0].text),
            "a new kind starts from its first starter"
        );
        assert_eq!(form.switch_type(), None);
        form.press(KIND_ROW);
        assert_eq!(form.switch_type(), Some("Shader"));
    }

    // Editing shows the file and the other volumes reading it, and leaves
    // every field of the volume in the form.
    #[test]
    fn an_edited_volume_lists_who_else_reads_its_field() {
        let entries = [
            volume("a", "shaders/blob.hlsl"),
            volume("b", "./shaders/blob.hlsl"),
            volume("c", "shaders/cloud.hlsl"),
        ];
        let resolve = |d: &str| d.trim_start_matches("./").to_string();
        let form = SdfForm::open(&entries, Some(1), resolve);
        assert!(form.hidden_fields().is_empty());
        let rows = form.rows("");
        assert_eq!(captions(&rows), ["Field file", "Also read by", "a"]);
        assert_eq!(rows[0].detail.as_deref(), Some("./shaders/blob.hlsl"));
        assert!(rows[2].indent);
        let alone = SdfForm::open(&entries, Some(2), resolve).rows("");
        assert_eq!(alone[1].detail.as_deref(), Some("no other volume"));
        assert_eq!(form.switch_type(), None);
        assert_eq!(
            form.blocked("", false),
            None,
            "the plain form checks an edit"
        );
    }

    #[test]
    fn a_new_volume_needs_a_free_name() {
        let form = SdfForm::open(&[], None, str::to_string);
        assert_eq!(
            form.blocked(" ", false).as_deref(),
            Some("Enter a name for the volume.")
        );
        assert!(
            form.blocked("blob", true)
                .unwrap()
                .contains("already taken")
        );
        assert_eq!(form.blocked(" blob ", false), None);
    }
}
