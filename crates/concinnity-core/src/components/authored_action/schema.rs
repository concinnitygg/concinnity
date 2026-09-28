// The action's authored form as the schema describes it, for authoring tools
// and the generated reference.

use crate::components::{AuthoredAction, UiAction};
use crate::ecs::schema::{Body, Described, FieldType, TypeSchema, VariantSchema};

const fn bare(name: &'static str, doc: &'static str) -> VariantSchema {
    VariantSchema {
        name,
        doc,
        payload: None,
    }
}

const fn keyed(name: &'static str, doc: &'static str, payload: fn() -> FieldType) -> VariantSchema {
    VariantSchema {
        name,
        doc,
        payload: Some(payload),
    }
}

pub(super) static ACTION: TypeSchema = TypeSchema {
    name: "Action",
    doc: "What a [HitRegion](#hitregion) click or a [KeyBinding](#keybinding) press does.\n\
          \n\
          An action that names nothing is written as its name; one that names\n\
          something is an object with the action as its only key, e.g.\n\
          `{\"show\": \"pause\"}`.",
    body: Body::Variants(&[
        bare("quit", "Stop the application."),
        keyed(
            "scene",
            "Change to the named [Scene](#scene), dismissing every open screen.",
            || FieldType::Reference(&["Scene"]),
        ),
        keyed(
            "show",
            "Show the named [Screen](#screen), replacing the top of the stack.",
            || FieldType::Reference(&["Screen"]),
        ),
        keyed(
            "push",
            "Open the named [Screen](#screen) on top of what is showing.",
            || FieldType::Reference(&["Screen"]),
        ),
        keyed(
            "toggle",
            "Close the named [Screen](#screen) if it is on top, open it otherwise.",
            || FieldType::Reference(&["Screen"]),
        ),
        bare("hide", "Close the top [Screen](#screen)."),
        keyed("story", "Drive the story system.", || {
            FieldType::Enum(&STORY_COMMAND)
        }),
        keyed(
            "group_toggle",
            "Expand or collapse a settings-screen group by index. Emitted by generated settings menus.",
            || FieldType::Integer,
        ),
        keyed(
            "setting",
            "Operate a settings row: `{\"key\": \"<setting>\", \"verb\": \"<verb>\"}`. Emitted by generated settings menus.",
            || FieldType::Object,
        ),
    ]),
    default: None,
};

pub(super) static STORY_COMMAND: TypeSchema = TypeSchema {
    name: "StoryCommand",
    doc: "What an [Action](#action) does to the story. An index-carrying\n\
          command is an object with its name as the only key, e.g. `{\"choose\": 1}`.",
    body: Body::Variants(&[
        bare("start", "Reset to the first node and show the stage."),
        bare(
            "continue",
            "Resume from the saved position, or start fresh when no save exists.",
        ),
        bare(
            "advance",
            "Advance the current page: complete a mid-reveal, else move on.",
        ),
        keyed(
            "choose",
            "Pick the current choice menu's option by index.",
            || FieldType::Integer,
        ),
        keyed("slot", "Pick a slot in the open slot overlay.", || {
            FieldType::Integer
        }),
        bare("auto", "Toggle auto-advance."),
        bare("skip", "Toggle fast-forward, which stops at menus."),
        bare("log", "Toggle the dialogue-history overlay."),
        bare("save", "Open the slot overlay in save mode."),
        bare("load", "Open the slot overlay in load mode."),
        bare("pause", "Toggle the pause menu over the stage."),
        bare(
            "settings",
            "Open the settings screen, remembering the menu that opened it.",
        ),
        bare(
            "settings_back",
            "Close the settings screen, returning to the menu that opened it.",
        ),
    ]),
    default: None,
};

impl<C, S> Described for AuthoredAction<C, S> {
    const TYPE: FieldType = FieldType::Enum(&ACTION);
}

impl Described for UiAction {
    const TYPE: FieldType = FieldType::Enum(&ACTION);
}
