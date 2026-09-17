// Resolves the `setting` string a settings-row asset (OptionSelect, Slider)
// authors into the engine's typed key, once, at expansion time.

use concinnity_core::settings::{SettingKey, SettingKind};

// The setting row asset `name` (an `asset_type`) binds. A blank key, a key that
// names no setting, or a setting of another kind than `kind` is a build error
// naming the asset.
pub(crate) fn row_setting(
    asset_type: &str,
    name: &str,
    setting: &str,
    kind: SettingKind,
) -> Result<SettingKey, String> {
    if setting.is_empty() {
        return Err(format!("{asset_type} '{name}': missing `setting`"));
    }
    let key = SettingKey::parse(setting)
        .ok_or_else(|| format!("{asset_type} '{name}': unknown setting '{setting}'"))?;
    if key.kind() != kind {
        return Err(format!(
            "{asset_type} '{name}': setting '{setting}' is a {:?} setting, not {kind:?}",
            key.kind()
        ));
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_setting_of_the_expected_kind_resolves() {
        assert_eq!(
            row_setting("OptionSelect", "opt", "vsync", SettingKind::Cycle),
            Ok(SettingKey::Vsync)
        );
        assert_eq!(
            row_setting("Slider", "sld", "exposure", SettingKind::Slider),
            Ok(SettingKey::Exposure)
        );
    }

    #[test]
    fn a_blank_setting_names_the_asset() {
        let err = row_setting("OptionSelect", "opt", "", SettingKind::Cycle).unwrap_err();
        assert_eq!(err, "OptionSelect 'opt': missing `setting`");
    }

    #[test]
    fn an_unknown_setting_names_the_asset_and_key() {
        let err = row_setting("Slider", "sld", "taa", SettingKind::Slider).unwrap_err();
        assert_eq!(err, "Slider 'sld': unknown setting 'taa'");
    }

    #[test]
    fn a_setting_of_another_kind_is_rejected() {
        let err = row_setting("OptionSelect", "opt", "exposure", SettingKind::Cycle).unwrap_err();
        assert!(err.contains("OptionSelect 'opt'"), "{err}");
        assert!(err.contains("'exposure' is a Slider setting"), "{err}");
        assert!(row_setting("Slider", "sld", "key_jump", SettingKind::Slider).is_err());
    }
}
