<!-- Auto-generated - do not edit. -->

# UiStateValues

The values the app's UI state holds when the world loads, and the lists it
shows.

Every name a `state:` binding uses is declared here, and its starting value
fixes its type, so the build catches a misspelled name or an action that
does not fit the value (flipping a number, adding to text). Every list a
[List](List.md) or [Menu](Menu.md) shows is declared in `lists`, with the rows
it starts with; the app's systems fill it at run time. A world may declare
several, one per screen, say; a name is declared once across them.

## Parameters

- `values`: An array of [UiStateValue](UiStateValue.md) objects. The declared values. Defaults to `[]`.
- `lists`: An array of [UiStateList](UiStateList.md) objects. The declared lists. Defaults to `[]`.
