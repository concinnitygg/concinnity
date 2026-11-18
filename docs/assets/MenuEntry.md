<!-- Auto-generated - do not edit. -->

# MenuEntry

One entry of a [Menu](Menu.md).

## Parameters

- `label`: A string. The entry's text. Defaults to `""`.
- `shortcut`: A string. Dimmer text at the entry's right end, such as its key shortcut. Defaults to `""`.
- `action`: A string or single-key object (see [Action](Action.md)). What picking the entry fires. Optional.
- `disabled`: A boolean. Whether the entry draws dimmed and cannot be picked. Defaults to `false`.
- `separator`: A boolean. Whether a divider line draws above the entry. Defaults to `false`.
- `mark`: A string or single-key object (see [EntryMark](EntryMark.md)). The mark the entry shows beside it while a `setting:`, `state:`, or `var:` boolean is true. Optional.
- `stay_open`: A boolean. Whether picking the entry leaves the menu open, so several toggles land in one visit. Defaults to `false`.
