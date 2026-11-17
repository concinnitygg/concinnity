<!-- Auto-generated - do not edit. -->

# ScrollGroup

A collapsible group of rows inside a [`ScrollPanel`](ScrollPanel.md).

## Parameters

- `collapsed`: A boolean. Whether the group starts collapsed (its body rows hidden). A group with a `collapsed_binding` follows its state value instead, which starts at the value its `UiStateValues` declares. Defaults to `false`.
- `header`: A string. The header [`TextLabel`](TextLabel.md) whose text gets a `+`/`-` prefix to reflect the collapsed state. `None` leaves the header text unchanged.
- `title`: A string. The header's base title (e.g. `"Advanced"`); the UI shows `"+ Advanced"` when collapsed and `"- Advanced"` when expanded. Defaults to `""`.
- `collapsed_binding`: A string. A `state:` binding the group's collapse follows, so a header [`HitRegion`](HitRegion.md) firing `{"flip": ...}` on it opens and shuts the group. Unset, the group stays as `collapsed` says.
