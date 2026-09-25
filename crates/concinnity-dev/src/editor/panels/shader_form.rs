//! The data half of the Shader form's extras: its stages, starter files,
//! world-default choice and the Materials that name it, as form rows; why the
//! form cannot confirm; and the edits a commit makes to other entries.

use concinnity_cook::authoring::world::{entry_handles, find_entry};
use concinnity_core::components::ShaderStage;

use super::form_extras::{ExtraControl, ExtraRow};
use super::shader_edit::check_name;
use super::shader_list::{can_add_shader, limit_reason};
use super::shader_source::stage_name;
use super::shader_templates;
use crate::editor::entry_list::EntryList;

const SHADER: &str = "Shader";

// Row ids.
const VERTEX: usize = 1;
const FRAGMENT_STARTER: usize = 2;
const VERTEX_STARTER: usize = 3;
const DEFAULT: usize = 4;
const MATERIAL: usize = 16;

// What the form edits: a Shader to create, or the declared one it opened on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Mode {
    New,
    Edit { had_vertex: bool, was_default: bool },
}

// One Material the form can assign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UsedBy {
    pub(crate) name: String,
    pub(crate) on: bool,
    // The Shader it names now, when that is another one.
    pub(crate) other: Option<String>,
    // Read from an included file, so not the editor's to change.
    pub(crate) read_only: bool,
}

// The form's own state beyond the schema fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShaderForm {
    pub(crate) mode: Mode,
    pub(crate) vertex: bool,
    pub(crate) fragment_starter: usize,
    pub(crate) vertex_starter: usize,
    pub(crate) default: bool,
    pub(crate) used_by: Vec<UsedBy>,
}

// The files a row shows: the declared path, or the path a commit would write.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Paths {
    pub(crate) fragment: String,
    pub(crate) vertex: Option<String>,
}

impl ShaderForm {
    // The form over `entries`, editing the Shader at `editing` or a new one.
    pub(crate) fn open(entries: &EntryList, editing: Option<usize>) -> Self {
        let handles = entry_handles(entries);
        let name = editing.and_then(|i| handles.get(i).cloned().flatten());
        let first = entries.iter().position(is_shader);
        let mode = match editing {
            Some(i) => Mode::Edit {
                had_vertex: vertex_of(&entries[i]).is_some(),
                was_default: first == Some(i),
            },
            None => Mode::New,
        };
        let used_by = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| entry_type(e) == Some("Material"))
            .filter_map(|(i, e)| {
                let material = handles[i].clone()?;
                let names = shader_of(e);
                let on = name.is_some() && names == name.as_deref();
                Some(UsedBy {
                    name: material,
                    on,
                    other: names.filter(|_| !on).map(str::to_string),
                    read_only: entries.is_read_only(i),
                })
            })
            .collect();
        Self {
            vertex: matches!(
                mode,
                Mode::Edit {
                    had_vertex: true,
                    ..
                }
            ),
            default: match mode {
                Mode::Edit { was_default, .. } => was_default,
                Mode::New => first.is_none(),
            },
            mode,
            fragment_starter: 0,
            vertex_starter: 0,
            used_by,
        }
    }

    pub(crate) fn is_new(&self) -> bool {
        self.mode == Mode::New
    }

    fn was_default(&self) -> bool {
        matches!(
            self.mode,
            Mode::Edit {
                was_default: true,
                ..
            }
        )
    }

    // Whether the default box can change in a world of `shaders` Shaders: the
    // only Shader is the default whatever the box says.
    fn default_open(&self, shaders: usize) -> bool {
        match self.mode {
            Mode::New => shaders > 0,
            Mode::Edit { .. } => shaders > 1,
        }
    }

    pub(crate) fn press(&mut self, id: usize) {
        match id {
            VERTEX => self.vertex = !self.vertex,
            FRAGMENT_STARTER => {
                self.fragment_starter =
                    (self.fragment_starter + 1) % shader_templates::FRAGMENT.len()
            }
            VERTEX_STARTER => {
                self.vertex_starter = (self.vertex_starter + 1) % shader_templates::VERTEX.len()
            }
            DEFAULT => self.default = !self.default,
            _ => {
                if let Some(m) = id
                    .checked_sub(MATERIAL)
                    .and_then(|i| self.used_by.get_mut(i))
                    .filter(|m| !m.read_only)
                {
                    m.on = !m.on;
                }
            }
        }
    }

    // The rows, in a world declaring `shaders` in order, showing `paths`.
    pub(crate) fn rows(&self, shaders: &[String], paths: &Paths) -> Vec<ExtraRow> {
        let mut out = vec![ExtraRow::label("Stages", None)];
        out.push(check(
            0,
            stage_name(ShaderStage::Fragment),
            true,
            false,
            Some(paths.fragment.clone()),
        ));
        out.push(check(
            VERTEX,
            stage_name(ShaderStage::Vertex),
            self.vertex,
            true,
            paths.vertex.clone().filter(|_| self.vertex),
        ));
        if self.is_new() {
            out.push(ExtraRow::label("Starters", None));
            out.push(choice(
                FRAGMENT_STARTER,
                ShaderStage::Fragment,
                self.fragment_starter,
            ));
            if self.vertex {
                out.push(choice(
                    VERTEX_STARTER,
                    ShaderStage::Vertex,
                    self.vertex_starter,
                ));
            }
        }
        out.push(ExtraRow {
            id: DEFAULT,
            caption: "World default".to_string(),
            indent: false,
            control: ExtraControl::Check {
                on: self.default,
                enabled: self.default_open(shaders.len()),
            },
            detail: None,
        });
        if let Some(note) = self.default_note(shaders) {
            out.push(ExtraRow::label(note, None).indented());
        }
        let none = self.used_by.is_empty().then(|| "no Materials".to_string());
        out.push(ExtraRow::label("Used by", none));
        for (i, m) in self.used_by.iter().enumerate() {
            let detail = match (&m.other, m.read_only) {
                (_, true) => Some("included".to_string()),
                (Some(other), false) => Some(format!("names '{other}'")),
                (None, false) => None,
            };
            out.push(check(MATERIAL + i, &m.name, m.on, !m.read_only, detail));
        }
        out
    }

    // What the default box does, when it does anything: the Materials naming
    // no Shader move to this one, or to the next Shader in `shaders`.
    fn default_note(&self, shaders: &[String]) -> Option<String> {
        match (self.default, self.was_default()) {
            (true, true) => Some("Materials naming no Shader use it".to_string()),
            (true, false) if !shaders.is_empty() => {
                Some("Materials naming no Shader switch to it".to_string())
            }
            (false, true) => shaders
                .get(1)
                .map(|next| format!("'{next}' becomes the world default")),
            _ => None,
        }
    }

    // Why the form cannot confirm under `name`: a bad name, one another entry
    // declares (`taken`), or no room for another of the world's `shaders`.
    pub(crate) fn blocked(&self, name: &str, taken: bool, shaders: usize) -> Option<String> {
        let name = match check_name(name) {
            Ok(name) => name,
            Err(reason) => return Some(reason),
        };
        if taken {
            return Some(format!("'{name}' is already taken; choose another name."));
        }
        if self.is_new() && !can_add_shader(shaders) {
            return Some(format!("No room: {}.", limit_reason()));
        }
        None
    }

    // Assign and unassign the Materials to Shader `shader`, as the boxes say.
    pub(crate) fn assign_materials(&self, entries: &mut [serde_json::Value], shader: &str) {
        for m in self.used_by.iter().filter(|m| !m.read_only) {
            let Some(args) = find_entry(entries, &m.name)
                .and_then(|i| entries[i].get_mut("args"))
                .and_then(|a| a.as_object_mut())
            else {
                continue;
            };
            let names = args.get("shader").and_then(|v| v.as_str()) == Some(shader);
            match (m.on, names) {
                (true, false) => {
                    args.insert("shader".to_string(), shader.into());
                }
                (false, true) => {
                    args.remove("shader");
                }
                _ => {}
            }
        }
    }

    // Put the Shader at `idx` first among the Shaders, or behind the next one,
    // as the default box says; the others keep their order.
    pub(crate) fn place_default(&self, entries: &mut EntryList, idx: usize) {
        match (self.default, first_shader(entries) == Some(idx)) {
            (true, false) => make_default(entries, idx),
            (false, true) => give_up_default(entries, idx),
            _ => {}
        }
    }
}

fn check(id: usize, caption: &str, on: bool, enabled: bool, detail: Option<String>) -> ExtraRow {
    ExtraRow {
        id,
        caption: caption.to_string(),
        indent: true,
        control: ExtraControl::Check { on, enabled },
        detail,
    }
}

fn choice(id: usize, stage: ShaderStage, selected: usize) -> ExtraRow {
    ExtraRow {
        id,
        caption: stage_name(stage).to_string(),
        indent: true,
        control: ExtraControl::Choice {
            options: shader_templates::of(stage)
                .iter()
                .map(|t| t.name.to_string())
                .collect(),
            selected,
        },
        detail: None,
    }
}

fn entry_type(e: &serde_json::Value) -> Option<&str> {
    e.get("type").and_then(|v| v.as_str())
}

fn is_shader(e: &serde_json::Value) -> bool {
    entry_type(e) == Some(SHADER)
}

fn vertex_of(e: &serde_json::Value) -> Option<&str> {
    e.get("args")?
        .get(stage_name(ShaderStage::Vertex))?
        .as_str()
}

fn shader_of(e: &serde_json::Value) -> Option<&str> {
    e.get("args")?.get("shader")?.as_str()
}

fn first_shader(entries: &[serde_json::Value]) -> Option<usize> {
    entries.iter().position(is_shader)
}

// Move the Shader at `idx` before the first Shader, and before the `Include`
// line that brings that one in, when it is included.
fn make_default(entries: &mut EntryList, idx: usize) {
    let Some(mut to) = first_shader(entries).filter(|&f| f < idx) else {
        return;
    };
    while to > 0 && entries.included_from(to).is_some() {
        to -= 1;
    }
    entries.move_entry(idx, to);
}

// Move the Shader at `idx` behind the next Shader, and past the rest of the
// included file that one comes from.
fn give_up_default(entries: &mut EntryList, idx: usize) {
    let Some(next) = (idx + 1..entries.len()).find(|&i| is_shader(&entries[i])) else {
        return;
    };
    let mut after = next + 1;
    while after < entries.len() && entries.included_from(after).is_some() {
        after += 1;
    }
    entries.move_entry(idx, after - 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::build_only::include::SourcedEntry;
    use serde_json::json;

    fn shader(name: &str) -> serde_json::Value {
        json!({"type": "Shader", "args": {"$id": name, "fragment": format!("{name}.hlsl")}})
    }

    fn material(name: &str, shader: Option<&str>) -> serde_json::Value {
        match shader {
            Some(s) => json!({"type": "Material", "args": {"$id": name, "shader": s}}),
            None => json!({"type": "Material", "args": {"$id": name}}),
        }
    }

    fn prop(name: &str) -> serde_json::Value {
        json!({"type": "Prop", "args": {"$id": name}})
    }

    fn ids(entries: &[serde_json::Value]) -> Vec<&str> {
        entries
            .iter()
            .map(|e| e["args"]["$id"].as_str().unwrap_or("?"))
            .collect()
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn row<'a>(rows: &'a [ExtraRow], caption: &str) -> &'a ExtraRow {
        rows.iter().find(|r| r.caption == caption).expect(caption)
    }

    #[test]
    fn a_form_on_a_shader_starts_from_what_it_declares() {
        let entries = EntryList::new(vec![
            shader("lit"),
            json!({"type": "Shader", "args": {"$id": "reeds", "fragment": "r.hlsl", "vertex": "s.hlsl"}}),
            material("reed_mat", Some("reeds")),
            material("rock", Some("lit")),
            material("plain", None),
        ]);
        let form = ShaderForm::open(&entries, Some(1));
        assert_eq!(
            form.mode,
            Mode::Edit {
                had_vertex: true,
                was_default: false
            }
        );
        assert!(form.vertex && !form.default);
        let on: Vec<(&str, bool, Option<&str>)> = form
            .used_by
            .iter()
            .map(|m| (m.name.as_str(), m.on, m.other.as_deref()))
            .collect();
        assert_eq!(
            on,
            [
                ("reed_mat", true, None),
                ("rock", false, Some("lit")),
                ("plain", false, None)
            ]
        );
    }

    #[test]
    fn a_new_shader_is_the_default_only_in_a_world_without_one() {
        let empty = ShaderForm::open(&EntryList::new(vec![material("m", None)]), None);
        assert!(empty.default && empty.is_new() && !empty.vertex);
        let rows = empty.rows(&[], &Paths::default());
        assert_eq!(
            row(&rows, "World default").control,
            ExtraControl::Check {
                on: true,
                enabled: false
            }
        );
        let other = ShaderForm::open(&EntryList::new(vec![shader("lit")]), None);
        assert!(!other.default);
    }

    // Stages, then starters (create only, vertex only while it is on), the
    // default box, and one box per Material.
    #[test]
    fn the_rows_follow_the_form() {
        let entries = EntryList::new(vec![shader("lit"), material("m", Some("lit"))]);
        let mut form = ShaderForm::open(&entries, None);
        let paths = Paths {
            fragment: "assets/shaders/water.hlsl".into(),
            vertex: Some("assets/shaders/water_vertex.hlsl".into()),
        };
        let shaders = names(&["lit"]);
        let captions = |f: &ShaderForm| -> Vec<String> {
            f.rows(&shaders, &paths)
                .iter()
                .map(|r| r.caption.clone())
                .collect()
        };
        assert_eq!(
            captions(&form),
            [
                "Stages",
                "fragment",
                "vertex",
                "Starters",
                "fragment",
                "World default",
                "Used by",
                "m"
            ]
        );
        let rows = form.rows(&shaders, &paths);
        assert_eq!(rows[1].detail.as_deref(), Some("assets/shaders/water.hlsl"));
        assert_eq!(rows[2].detail, None, "no vertex file while it is off");
        assert_eq!(row(&rows, "m").detail.as_deref(), Some("names 'lit'"));

        form.press(VERTEX);
        form.press(DEFAULT);
        let rows = form.rows(&shaders, &paths);
        assert_eq!(
            rows[2].detail.as_deref(),
            Some("assets/shaders/water_vertex.hlsl")
        );
        assert_eq!(
            captions(&form)[3..7],
            ["Starters", "fragment", "vertex", "World default"]
        );
        assert!(
            rows.iter()
                .any(|r| r.caption == "Materials naming no Shader switch to it")
        );

        let edit = ShaderForm::open(&entries, Some(0));
        let rows = edit.rows(&shaders, &paths);
        assert!(!rows.iter().any(|r| r.caption == "Starters"));
        assert_eq!(
            row(&rows, "World default").control,
            ExtraControl::Check {
                on: true,
                enabled: false
            },
            "the only Shader stays the default"
        );
    }

    #[test]
    fn starters_cycle_and_a_read_only_material_stays_put() {
        let entries = EntryList::with_includes(vec![
            SourcedEntry {
                entry: material("inc", None),
                file: Some("lib.jsonl".into()),
            },
            SourcedEntry {
                entry: material("own", None),
                file: None,
            },
        ]);
        let mut form = ShaderForm::open(&entries, None);
        for _ in 0..shader_templates::FRAGMENT.len() + 1 {
            form.press(FRAGMENT_STARTER);
        }
        assert_eq!(form.fragment_starter, 1);
        form.press(MATERIAL);
        form.press(MATERIAL + 1);
        assert!(!form.used_by[0].on && form.used_by[1].on);
        let rows = form.rows(&[], &Paths::default());
        assert_eq!(row(&rows, "inc").detail.as_deref(), Some("included"));
    }

    #[test]
    fn a_bad_taken_or_surplus_name_blocks_the_form() {
        let form = ShaderForm::open(&EntryList::new(Vec::new()), None);
        assert!(
            form.blocked("  ", false, 0)
                .unwrap()
                .contains("Enter a name")
        );
        assert!(
            form.blocked("Shader#2", false, 0)
                .unwrap()
                .contains("reserved")
        );
        assert!(
            form.blocked("lit", true, 0)
                .unwrap()
                .contains("already taken")
        );
        let full = render_types_max();
        assert!(
            form.blocked("lit", false, full)
                .unwrap()
                .contains("No room")
        );
        assert_eq!(form.blocked(" lit ", false, full - 1), None);
        let edit = ShaderForm::open(&EntryList::new(vec![shader("lit")]), Some(0));
        assert_eq!(edit.blocked("lit", false, full), None, "an edit adds none");
    }

    fn render_types_max() -> usize {
        concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS
    }

    #[test]
    fn materials_are_assigned_and_unassigned_as_the_boxes_say() {
        let mut entries = EntryList::new(vec![
            shader("lit"),
            shader("water"),
            material("a", Some("water")),
            material("b", Some("lit")),
            material("c", None),
        ]);
        let mut form = ShaderForm::open(&entries, Some(1));
        form.press(MATERIAL);
        form.press(MATERIAL + 1);
        form.assign_materials(&mut entries, "water");
        assert!(entries[2]["args"].get("shader").is_none(), "unassigned");
        assert_eq!(entries[3]["args"]["shader"], "water", "moved from lit");
        assert!(entries[4]["args"].get("shader").is_none(), "left alone");
    }

    #[test]
    fn the_default_moves_first_and_back_keeping_the_rest_in_order() {
        let mut entries = EntryList::new(vec![
            prop("p"),
            shader("a"),
            shader("b"),
            prop("q"),
            shader("c"),
            shader("d"),
        ]);
        let c = entries.key_at(4).unwrap();
        let mut form = ShaderForm::open(&entries, Some(4));
        form.press(DEFAULT);
        form.place_default(&mut entries, 4);
        assert_eq!(ids(&entries), ["p", "c", "a", "b", "q", "d"]);
        assert_eq!(entries.index_of(c), Some(1));

        let mut form = ShaderForm::open(&entries, Some(1));
        form.press(DEFAULT);
        form.place_default(&mut entries, 1);
        assert_eq!(ids(&entries), ["p", "a", "c", "b", "q", "d"]);
    }

    #[test]
    fn the_default_moves_around_an_included_block_whole() {
        let own = |e| SourcedEntry {
            entry: e,
            file: None,
        };
        let inc = |e| SourcedEntry {
            entry: e,
            file: Some("lib.jsonl".into()),
        };
        let include = json!({"type": "Include", "args": {"$id": "lib", "path": "lib.jsonl"}});
        let mut entries = EntryList::with_includes(vec![
            own(include),
            inc(shader("a")),
            inc(shader("b")),
            own(shader("c")),
        ]);
        let mut form = ShaderForm::open(&entries, Some(3));
        form.press(DEFAULT);
        form.place_default(&mut entries, 3);
        assert_eq!(ids(&entries), ["c", "lib", "a", "b"]);

        let mut form = ShaderForm::open(&entries, Some(0));
        form.press(DEFAULT);
        form.place_default(&mut entries, 0);
        assert_eq!(ids(&entries), ["lib", "a", "b", "c"]);
    }
}
