use super::*;
use concinnity_engine::components::UiState;

fn built(world: &mut World) -> (BuiltPanel, LightingIds) {
    let mut ui = UiBuilder::new(world, AssetId(9), None);
    let mut panel = build(&mut ui).unwrap();
    let ids = build_rows(world, &mut panel);
    (panel, ids)
}

// The bindings in order, each with its section's default field.
fn defaults() -> Vec<Option<FormField>> {
    SECTIONS
        .iter()
        .flat_map(|s| lighting::section_fields(s, None))
        .map(Some)
        .collect()
}

fn event(name: &str, value: Option<UiValue>) -> UiEvent {
    UiEvent {
        name: name.into(),
        element: None,
        value,
    }
}

#[test]
fn events_name_their_binding_or_section() {
    assert_eq!(
        action(&event("editor.lighting.toggle.5", None)),
        Some(LightingAction::Toggle(5))
    );
    assert_eq!(
        action(&event(ADD, Some(UiValue::Int(2)))),
        Some(LightingAction::Add(2))
    );
    assert_eq!(action(&event(APPLY, None)), Some(LightingAction::Apply));
    assert_eq!(action(&event(ADD, None)), None);
}

// A section with its asset shows its bindings; one without shows its add
// button. The texts, checks, swatches, and title follow the hook's model.
#[test]
fn a_section_shows_its_bindings_or_its_add_button() {
    let mut world = World::new();
    let (_, ids) = built(&mut world);
    assert_eq!(ids.bindings.len(), lighting::binding_count());
    let mut present = vec![true; SECTIONS.len()];
    present[1] = false;
    let fields = defaults();
    let mut texts = vec![String::new(); fields.len()];
    let color = (0..fields.len())
        .find(|&b| ids.bindings[b].swatch.is_some())
        .unwrap();
    texts[color] = "1, 0, 0".into();
    let view = LightingView {
        present: &present,
        fields: &fields,
        texts: &texts,
        focus: Some(color),
        status: Some("bad"),
        dirty: true,
    };
    sync(&view, &ids, &mut world);
    let shown = |id: Option<AssetId>| world.get_by_id::<Layout>(id.unwrap()).unwrap().visible;
    assert!(shown(ids.bindings[0].row), "the Sun's rows show");
    assert!(!shown(ids.adds[0]), "and not its add button");
    let fog = lighting::section_base(1);
    assert!(!shown(ids.bindings[fog].row), "the Fog's rows hide");
    assert!(shown(ids.adds[1]), "behind its add button");
    let text = world
        .get_by_id::<TextField>(ids.bindings[color].text.unwrap())
        .unwrap();
    assert_eq!(text.content, "1, 0, 0");
    assert!(text.focused);
    let swatch = world
        .get_by_id::<Sprite>(ids.bindings[color].swatch.unwrap())
        .unwrap();
    assert_eq!(swatch.tint, [1.0, 0.0, 0.0, 1.0]);
    let s = world.resource::<UiState>().unwrap();
    assert_eq!(s.get(TITLE), Some(&UiValue::Text("Lighting *".into())));
    assert_eq!(s.get(STATUS), Some(&UiValue::Text("bad".into())));

    // Typing reads back.
    if let Some(f) = world.get_mut_by_id::<TextField>(ids.bindings[color].text.unwrap()) {
        f.content = "0, 1, 0".into();
    }
    let focus = observe(&world, &ids, &mut texts);
    assert_eq!(texts[color], "0, 1, 0");
    assert_eq!(focus, Some(color));
}

// Rows missing from the ids, such as none built yet, are skipped.
#[test]
fn sync_skips_rows_it_has_no_ids_for() {
    let mut world = World::new();
    let present = vec![true; SECTIONS.len()];
    let fields = defaults();
    let texts = vec![String::new(); fields.len()];
    let view = LightingView {
        present: &present,
        fields: &fields,
        texts: &texts,
        focus: None,
        status: None,
        dirty: false,
    };
    sync(&view, &LightingIds::default(), &mut world);
}
