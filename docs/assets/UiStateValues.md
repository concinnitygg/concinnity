<!-- Auto-generated - do not edit. -->

# UiStateValues

The values the app's UI state holds when the world loads.

Every name a `state:` binding uses is declared here, and its starting value
fixes its type, so the build catches a misspelled name or an action that
does not fit the value (flipping a number, adding to text). A world may
declare several, one per screen, say; a name is declared once across them.

## Parameters

- `values`: An array of [UiStateValue](UiStateValue.md) objects. The declared values. Defaults to `[]`.
