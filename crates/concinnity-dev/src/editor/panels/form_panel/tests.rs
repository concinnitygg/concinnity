use super::*;

fn field(key: &str, kind: FieldKind) -> FormField {
    FormField {
        key: key.into(),
        kind,
        initial: String::new(),
        boolval: false,
        variants: Vec::new(),
        variant_idx: 0,
        choices: Vec::new(),
        sparse: false,
        unset: false,
    }
}

fn choice(key: &str, n: usize) -> FormField {
    FormField {
        variants: (0..n).map(|i| format!("v{i}")).collect(),
        ..field(key, FieldKind::Enum)
    }
}

fn event(name: &str, value: Option<UiValue>) -> UiEvent {
    UiEvent {
        name: name.into(),
        element: None,
        value,
    }
}

fn at(i: i32) -> UiValue {
    UiValue::Int(i)
}

fn built(world: &mut World) -> BuiltPanel {
    let mut ui = UiBuilder::new(world, AssetId(9), None);
    build(&mut ui).unwrap()
}

#[test]
fn a_choice_cycles_a_short_set_and_opens_a_long_one() {
    let fields = [choice("a", CYCLE_MAX), choice("b", CYCLE_MAX + 1)];
    let act = |e: &UiEvent| action(e, &fields, None);
    assert_eq!(
        act(&event(CHOICE, Some(at(0)))),
        Some(FormAction::CycleField(0))
    );
    assert_eq!(
        act(&event(CHOICE, Some(at(1)))),
        Some(FormAction::OpenFieldDropdown(1))
    );
    assert_eq!(act(&event(CHOICE, Some(at(2)))), None, "no such field");
    assert_eq!(
        act(&event("editor.form.check.3", None)),
        Some(FormAction::ToggleField(3))
    );
    assert_eq!(
        act(&event("editor.form.extra_check.7", None)),
        Some(FormAction::PressExtra(7))
    );
    assert_eq!(act(&event(APPLY, None)), Some(FormAction::Confirm));
    assert_eq!(
        act(&event(ADD, Some(at(2)))),
        Some(FormAction::AddArrayElement(2))
    );
}

#[test]
fn a_menu_pick_acts_on_the_menu_that_is_open() {
    let pick = |menu| action(&event(MENU_PICK, Some(at(1))), &[], menu);
    assert_eq!(
        pick(Some(FormMenu::Field(0))),
        Some(FormAction::PickFieldOption(1))
    );
    assert_eq!(
        pick(Some(FormMenu::Override(0))),
        Some(FormAction::PickOverrideOption(1))
    );
    assert_eq!(
        pick(Some(FormMenu::Entity)),
        Some(FormAction::PickEntityOption(1))
    );
    assert_eq!(pick(None), None);
}

#[test]
fn swatch_rgb_falls_back_and_clamps() {
    assert_eq!(swatch_rgb("1"), NO_SWATCH);
    assert_eq!(swatch_rgb("nope"), NO_SWATCH);
    assert_eq!(swatch_rgb("2, -1, 0.5"), [1.0, 0.0, 0.5, 1.0]);
}

#[test]
fn captions_name_axes_counts_and_unset_leaves() {
    let fields = [
        field(
            "position",
            FieldKind::Vec {
                len: 3,
                color: false,
            },
        ),
        field("position.0", FieldKind::Float),
        FormField {
            variant_idx: 2,
            ..field("waves", FieldKind::Array)
        },
        field("waves.0.height", FieldKind::Float),
        field("waves.1.height", FieldKind::Float),
    ];
    assert!(vec_expanded(&fields, 0));
    assert_eq!(field_caption(&fields, &fields[1]), "x");
    let unset = FormField {
        unset: true,
        ..field("bloom.strength", FieldKind::Float)
    };
    assert_eq!(
        field_caption(&[], &unset),
        format!("strength {}", form::AUTO_LABEL)
    );
    assert_eq!(axis_letter(5, 8), "[5]");
}

// The rows follow the fields' shape, and the model shows in them: a field's
// text, a bool's state, a choice's caption, and a color's swatch.
#[test]
fn the_rows_show_each_fields_value() {
    let mut world = World::new();
    let mut panel = built(&mut world);
    let fields = [
        field("label", FieldKind::Str),
        FormField {
            boolval: true,
            ..field("on", FieldKind::Bool)
        },
        FormField {
            variant_idx: 1,
            ..choice("mode", 3)
        },
        field(
            "tint",
            FieldKind::Vec {
                len: 3,
                color: true,
            },
        ),
    ];
    let shown = rebuild_rows(&mut world, &mut panel, &fields, &[], (4, 1), None);
    assert_eq!(shown.fields.len(), 4);
    let texts = [
        "hello".to_string(),
        String::new(),
        String::new(),
        "1, 0, 0".to_string(),
    ];
    let view = FormView {
        title: "Edit Thing",
        editing: true,
        fields: &fields,
        texts: &texts,
        name: "thing",
        focus: Some(FormFocus::Field(0)),
        error: None,
        blocked: None,
        marks: None,
        extras: &[],
        menu: None,
        reveal: None,
    };
    sync(&view, &shown, &mut world, &panel);
    let text = world
        .get_by_id::<TextField>(shown.fields[0].text.unwrap())
        .unwrap();
    assert_eq!(text.content, "hello");
    assert!(text.focused);
    let s = world.resource::<UiState>().unwrap();
    assert_eq!(s.get("editor.form.check.1"), Some(&UiValue::Bool(true)));
    assert_eq!(
        s.get("editor.form.choice.2"),
        Some(&UiValue::Text("v1".into()))
    );
    assert_eq!(s.get(APPLY_TEXT), Some(&UiValue::Text("Apply".into())));
    let swatch = world
        .get_by_id::<Sprite>(shown.fields[3].swatch.unwrap())
        .unwrap();
    assert_eq!(swatch.tint, [1.0, 0.0, 0.0, 1.0]);

    // Typing reads back into the hook's texts.
    if let Some(f) = world.get_mut_by_id::<TextField>(shown.fields[0].text.unwrap()) {
        f.content = "typed".into();
    }
    let mut name = "thing".to_string();
    let mut read = texts.clone();
    let (changed, focus) = observe(&world, &panel, &shown, &mut name, &mut read);
    assert!(changed);
    assert_eq!(read[0], "typed");
    assert_eq!(focus, Some(FormFocus::Field(0)));

    // A rebuild for another shape despawns the old rows.
    let old = shown.fields[0].text.unwrap();
    let shown = rebuild_rows(&mut world, &mut panel, &fields[..1], &[], (5, 1), None);
    assert_eq!(shown.fields.len(), 1);
    assert!(world.entity_of(old).is_none());
}

// A blocked form says why and disables its confirm; an override summary shows
// only for a template-derived asset.
#[test]
fn a_blocked_form_says_why_and_overrides_show_their_count() {
    let mut world = World::new();
    let mut panel = built(&mut world);
    let fields = [field("label", FieldKind::Str)];
    let shown = rebuild_rows(&mut world, &mut panel, &fields, &[], (1, 1), None);
    let marks = [FieldOrigin::Overridden];
    let view = FormView {
        title: "New Thing",
        editing: false,
        fields: &fields,
        texts: &[String::new()],
        name: "",
        focus: None,
        error: None,
        blocked: Some("needs a name"),
        marks: Some((&marks, 1)),
        extras: &[],
        menu: None,
        reveal: None,
    };
    sync(&view, &shown, &mut world, &panel);
    let apply = world.get_by_id::<Button>(panel.slots[APPLY_SLOT]).unwrap();
    assert!(apply.disabled);
    let s = world.resource::<UiState>().unwrap();
    assert_eq!(s.get(STATUS), Some(&UiValue::Text("needs a name".into())));
    assert_eq!(s.get(CHIP), Some(&UiValue::Text("1 overridden".into())));
    let bar = world
        .get_by_id::<Sprite>(shown.fields[0].bar.unwrap())
        .unwrap();
    assert!(bar.visible);
    assert!(
        world
            .get_by_id::<Layout>(panel.slots[OVERRIDES_SLOT])
            .unwrap()
            .visible
    );
}

fn extra(id: usize, control: ExtraControl, detail: Option<&str>) -> ExtraRow {
    ExtraRow {
        id,
        caption: "row".into(),
        indent: false,
        control,
        detail: detail.map(str::to_string),
    }
}

// The extra rows keep their shape while only their values and text change,
// and lose it when a row comes, goes, or changes kind or gains a detail.
#[test]
fn extra_rows_match_by_id_kind_and_detail() {
    let mut world = World::new();
    let mut panel = built(&mut world);
    let check = |on| ExtraControl::Check { on, enabled: true };
    let rows = [
        extra(1, check(false), Some("a")),
        extra(2, ExtraControl::Label, None),
    ];
    let shown = rebuild_rows(&mut world, &mut panel, &[], &rows, (1, 1), None);
    let toggled = [
        extra(1, check(true), Some("b")),
        extra(2, ExtraControl::Label, None),
    ];
    assert!(
        shown.extras_match(&toggled),
        "a value and a text are not a shape"
    );
    assert!(!shown.extras_match(&rows[..1]));
    let no_note = [
        extra(1, check(false), None),
        extra(2, ExtraControl::Label, None),
    ];
    assert!(
        shown.extras_match(&no_note),
        "a check's note is its own text"
    );
    let detailed = [
        extra(1, check(false), Some("a")),
        extra(2, ExtraControl::Label, Some("d")),
    ];
    assert!(
        !shown.extras_match(&detailed),
        "a label's detail is a label of its own"
    );
    let relabeled = [
        extra(1, ExtraControl::Label, Some("a")),
        extra(2, ExtraControl::Label, None),
    ];
    assert!(!shown.extras_match(&relabeled));

    // A changed caption and note show without a rebuild.
    let mut renamed = toggled;
    renamed[0].caption = "renamed".into();
    let view = FormView {
        title: "",
        editing: false,
        fields: &[],
        texts: &[],
        name: "",
        focus: None,
        error: None,
        blocked: None,
        marks: None,
        extras: &renamed,
        menu: None,
        reveal: None,
    };
    sync(&view, &shown, &mut world, &panel);
    let (_, id) = shown.extras[0].check.clone().unwrap();
    let check = world.get_by_id::<Checkbox>(id).unwrap();
    assert_eq!(
        (check.label.as_str(), check.note.as_str()),
        ("renamed", "b")
    );
}

// Rows rebuilt for the same form keep its scroll; another form starts at the
// top. A reveal scrolls a row into place.
#[test]
fn rows_keep_their_scroll_until_another_form_opens() {
    let mut world = World::new();
    let mut panel = built(&mut world);
    let fields = [field("a", FieldKind::Str), field("b", FieldKind::Str)];
    let host = panel.slots[ROWS_SLOT];
    let scrolled = |world: &World| world.get_by_id::<Layout>(host).unwrap().scrolled;
    let first = rebuild_rows(&mut world, &mut panel, &fields, &[], (1, 1), None);
    let view = FormView {
        title: "",
        editing: false,
        fields: &fields,
        texts: &[String::new(), String::new()],
        name: "",
        focus: None,
        error: None,
        blocked: None,
        marks: None,
        extras: &[],
        menu: None,
        reveal: Some(1),
    };
    sync(&view, &first, &mut world, &panel);
    assert_eq!(scrolled(&world), FIELD_H + ROW_GAP);
    let again = rebuild_rows(&mut world, &mut panel, &fields, &[], (2, 1), Some(&first));
    assert_eq!(
        scrolled(&world),
        FIELD_H + ROW_GAP,
        "the same form keeps its place"
    );
    rebuild_rows(&mut world, &mut panel, &fields, &[], (3, 2), Some(&again));
    assert_eq!(scrolled(&world), 0.0, "another form starts at the top");
}

// The name field read back, and the focus it holds.
#[test]
fn observe_reads_the_name_and_its_focus() {
    let mut world = World::new();
    let mut panel = built(&mut world);
    let shown = rebuild_rows(&mut world, &mut panel, &[], &[], (1, 1), None);
    if let Some(f) = world.get_mut_by_id::<TextField>(panel.slots[NAME_SLOT]) {
        f.content = "lamp".into();
        f.focused = true;
    }
    let mut name = String::new();
    let (changed, focus) = observe(&world, &panel, &shown, &mut name, &mut []);
    assert!(changed);
    assert_eq!(name, "lamp");
    assert_eq!(focus, Some(FormFocus::Name));
    let (changed, _) = observe(&world, &panel, &shown, &mut name, &mut []);
    assert!(!changed, "nothing new typed");
}
