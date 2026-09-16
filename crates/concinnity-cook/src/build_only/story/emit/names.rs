use super::overlay::VISIBLE_SLOTS;

// The title menu buttons, top to bottom.
pub(super) const TITLE_BUTTON_KEYS: [&str; 5] = ["start", "continue", "load", "settings", "quit"];
// The quick-row controls along the dialog box, left to right.
pub(super) const QUICK_KEYS: [&str; 4] = ["qlog", "qauto", "qskip", "qsave"];

// A clickable row: its label and the hit region that fires it.
pub(super) struct ButtonNames {
    pub(super) label: String,
    pub(super) region: String,
}

impl ButtonNames {
    fn new(base: &str) -> Self {
        Self {
            label: format!("{}_lbl", base),
            region: format!("{}_btn", base),
        }
    }
}

// A choice option or save slot: a rounded box behind a button.
pub(super) struct RowNames {
    pub(super) row_box: String,
    pub(super) button: ButtonNames,
}

impl RowNames {
    fn new(base: &str) -> Self {
        Self {
            row_box: format!("{}_box", base),
            button: ButtonNames::new(base),
        }
    }
}

pub(super) struct TitleNames {
    pub(super) screen: String,
    pub(super) bg: String,
    pub(super) heading: String,
    // In `TITLE_BUTTON_KEYS` order.
    pub(super) buttons: [ButtonNames; 5],
}

impl TitleNames {
    fn button(&self, key: &str) -> &ButtonNames {
        let i = TITLE_BUTTON_KEYS
            .iter()
            .position(|k| *k == key)
            .expect("a title button key");
        &self.buttons[i]
    }
}

pub(super) struct StageNames {
    pub(super) screen: String,
    pub(super) bg: String,
    pub(super) left: String,
    pub(super) center: String,
    pub(super) right: String,
    pub(super) dialog_box: String,
    pub(super) name_label: String,
    pub(super) text_label: String,
    pub(super) advance: String,
    // The Space and Enter key bindings.
    pub(super) advance_keys: [String; 2],
    pub(super) marker: String,
    // In `QUICK_KEYS` order.
    pub(super) quick: [ButtonNames; 4],
    pub(super) options: Vec<RowNames>,
    pub(super) dim: String,
    pub(super) history: String,
    pub(super) slot_title: String,
    pub(super) slots: Vec<RowNames>,
}

pub(super) struct EndingNames {
    pub(super) screen: String,
    pub(super) bg: String,
    pub(super) fin: String,
    pub(super) back: ButtonNames,
}

// Every generated asset name of one story, built once so each emitter and the
// runtime scaffold read the same strings.
pub(super) struct StoryNames {
    pub(super) prefix: String,
    pub(super) font_title: String,
    pub(super) font_menu: String,
    pub(super) font_dialog: String,
    pub(super) title: Option<TitleNames>,
    pub(super) stage: StageNames,
    pub(super) ending: EndingNames,
}

impl StoryNames {
    pub(super) fn new(prefix: &str, title_screen: bool, max_choices: usize) -> Self {
        let title = title_screen.then(|| {
            let screen = format!("{}_title", prefix);
            TitleNames {
                bg: format!("{}_bg", screen),
                heading: format!("{}_heading", screen),
                buttons: TITLE_BUTTON_KEYS
                    .map(|key| ButtonNames::new(&format!("{}_{}", screen, key))),
                screen,
            }
        });
        let stage = format!("{}_stage", prefix);
        let member = |suffix: &str| format!("{}_{}", stage, suffix);
        let stage = StageNames {
            bg: member("bg"),
            left: member("left"),
            center: member("center"),
            right: member("right"),
            dialog_box: member("box"),
            name_label: member("name"),
            text_label: member("text"),
            advance: member("advance"),
            advance_keys: [
                format!("{}_advance_key", prefix),
                format!("{}_advance_key_enter", prefix),
            ],
            marker: member("marker"),
            quick: QUICK_KEYS.map(|key| ButtonNames::new(&member(key))),
            options: (0..max_choices)
                .map(|i| RowNames::new(&member(&format!("opt{}", i))))
                .collect(),
            dim: member("dim"),
            history: member("history"),
            slot_title: member("slot_title"),
            slots: (0..VISIBLE_SLOTS)
                .map(|i| RowNames::new(&member(&format!("slot{}", i))))
                .collect(),
            screen: stage,
        };
        let ending = format!("{}_ending", prefix);
        let ending = EndingNames {
            bg: format!("{}_bg", ending),
            fin: format!("{}_fin", ending),
            back: ButtonNames::new(&format!("{}_back", ending)),
            screen: ending,
        };
        Self {
            prefix: prefix.to_string(),
            font_title: format!("{}_font_title", prefix),
            font_menu: format!("{}_font_menu", prefix),
            font_dialog: format!("{}_font_dialog", prefix),
            title,
            stage,
            ending,
        }
    }

    // The Story asset's `scaffold` block: the stage assets the story system
    // drives, by name.
    pub(super) fn scaffold(&self) -> serde_json::Value {
        let stage = &self.stage;
        let title_label = |key: &str| self.title.as_ref().map(|t| &t.button(key).label);
        let quick_label = |i: usize| &stage.quick[i].label;
        let (option_boxes, option_labels) = row_names(&stage.options);
        let (slot_boxes, slot_labels) = row_names(&stage.slots);
        serde_json::json!({
            "screen": stage.screen,
            "ending": self.ending.screen,
            "bg": stage.bg,
            "left": stage.left,
            "center": stage.center,
            "right": stage.right,
            "dialog_box": stage.dialog_box,
            "name_label": stage.name_label,
            "text_label": stage.text_label,
            "option_boxes": option_boxes,
            "options": option_labels,
            "start_label": title_label("start"),
            "quit_label": title_label("quit"),
            "continue_label": title_label("continue"),
            "title": self.title.as_ref().map(|t| &t.screen),
            "load_label": title_label("load"),
            "settings_label": title_label("settings"),
            "advance_marker": stage.marker,
            "log_label": quick_label(0),
            "auto_label": quick_label(1),
            "skip_label": quick_label(2),
            "save_label": quick_label(3),
            "overlay_dim": stage.dim,
            "backlog_label": stage.history,
            "slot_title": stage.slot_title,
            "slot_boxes": slot_boxes,
            "slot_labels": slot_labels,
        })
    }

    fn screen_names(&self) -> Vec<&str> {
        let mut names = vec![self.stage.screen.as_str(), self.ending.screen.as_str()];
        if let Some(title) = &self.title {
            names.push(&title.screen);
        }
        names
    }
}

// The box names and label names of a set of rows.
fn row_names(rows: &[RowNames]) -> (Vec<&str>, Vec<&str>) {
    rows.iter()
        .map(|r| (r.row_box.as_str(), r.button.label.as_str()))
        .unzip()
}

// UI assets attach to a Screen by name prefix, so one generated screen name
// must never be a `_`-extension of another or the members of the longer
// screen would be ambiguous.
pub(super) fn check_screen_names(names: &StoryNames) -> Result<(), String> {
    let mut screen_names = names.screen_names();
    screen_names.sort();
    for pair in screen_names.windows(2) {
        if pair[1].starts_with(&format!("{}_", pair[0])) {
            return Err(format!(
                "generated screen '{}' is a name-prefix of '{}'",
                pair[0], pair[1]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_title_screen_adds_the_title_names() {
        let names = StoryNames::new("s", true, 2);
        let title = names.title.as_ref().expect("title names");
        assert_eq!(title.screen, "s_title");
        assert_eq!(title.button("continue").region, "s_title_continue_btn");
        assert_eq!(names.stage.screen, "s_stage");
        assert_eq!(names.ending.screen, "s_ending");
        assert_eq!(names.screen_names(), ["s_stage", "s_ending", "s_title"]);
        assert_eq!(names.stage.options.len(), 2);
        assert_eq!(names.stage.slots.len(), VISIBLE_SLOTS);
    }

    #[test]
    fn without_a_title_screen_only_stage_and_ending_remain() {
        let names = StoryNames::new("s", false, 0);
        assert!(names.title.is_none());
        assert_eq!(names.screen_names(), ["s_stage", "s_ending"]);
        assert!(names.stage.options.is_empty());
        let scaffold = names.scaffold();
        for key in [
            "title",
            "start_label",
            "quit_label",
            "continue_label",
            "load_label",
            "settings_label",
        ] {
            assert!(scaffold[key].is_null(), "{key}");
        }
    }

    #[test]
    fn scaffold_names_match_the_fields() {
        let names = StoryNames::new("s", true, 3);
        let scaffold = names.scaffold();
        let stage = &names.stage;
        let title = names.title.as_ref().unwrap();
        for (key, field) in [
            ("screen", &stage.screen),
            ("ending", &names.ending.screen),
            ("bg", &stage.bg),
            ("left", &stage.left),
            ("center", &stage.center),
            ("right", &stage.right),
            ("dialog_box", &stage.dialog_box),
            ("name_label", &stage.name_label),
            ("text_label", &stage.text_label),
            ("title", &title.screen),
            ("start_label", &title.button("start").label),
            ("quit_label", &title.button("quit").label),
            ("continue_label", &title.button("continue").label),
            ("load_label", &title.button("load").label),
            ("settings_label", &title.button("settings").label),
            ("advance_marker", &stage.marker),
            ("log_label", &stage.quick[0].label),
            ("auto_label", &stage.quick[1].label),
            ("skip_label", &stage.quick[2].label),
            ("save_label", &stage.quick[3].label),
            ("overlay_dim", &stage.dim),
            ("backlog_label", &stage.history),
            ("slot_title", &stage.slot_title),
        ] {
            assert_eq!(scaffold[key], *field, "{key}");
        }
        assert_eq!(scaffold["log_label"], "s_stage_qlog_lbl");
        for (i, row) in stage.options.iter().enumerate() {
            assert_eq!(scaffold["option_boxes"][i], row.row_box);
            assert_eq!(scaffold["options"][i], row.button.label);
        }
        for (i, row) in stage.slots.iter().enumerate() {
            assert_eq!(scaffold["slot_boxes"][i], row.row_box);
            assert_eq!(scaffold["slot_labels"][i], row.button.label);
        }
        assert_eq!(scaffold["options"][2], "s_stage_opt2_lbl");
    }

    #[test]
    fn a_screen_name_extending_another_is_rejected() {
        let mut names = StoryNames::new("s", true, 0);
        assert_eq!(check_screen_names(&names), Ok(()));
        names.stage.screen = "s_ending_stage".to_string();
        let err = check_screen_names(&names).unwrap_err();
        assert_eq!(
            err,
            "generated screen 's_ending' is a name-prefix of 's_ending_stage'"
        );
    }
}
