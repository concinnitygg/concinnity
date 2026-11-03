// The graphics settings page: every setting a player can change at runtime that
// shapes the rendered image, with its options, what it needs, the asset field
// it overrides (whose documented meaning, default and apply timing it takes),
// and what each quality preset allows.
//
// The settings come from `SettingKey` and the settings registry, the prose from
// the rustdoc of the engine types the page describes, the field descriptions
// from the asset reference pages, and the preset table from the
// `QualityCeiling` tiers, so the page follows the engine without a hand-kept
// list.

use concinnity_engine::ecs::schema::Documented;
use concinnity_engine::platform::Platform;
use concinnity_engine::render::backend::{GpuProfile, GpuTier};
use concinnity_engine::settings::quality::{AUTO_UPSCALE_PIXELS, preset_label};
use concinnity_engine::settings::{
    QualityCeiling, QualityPreset, SettingGroup, SettingKey, SettingKind, levels, options, registry,
};

use super::page::AUTOGEN_MARKER;
use super::prose::first_paragraph;
use super::reference::AssetDoc;

// Where the page lands, relative to the directory given on the command line.
pub(super) const PAGE: &str = "docs/settings/graphics.md";

// Where the asset pages sit relative to this one.
const ASSETS: &str = "../assets/";

// The named presets, lowest first, with the ceilings they impose.
const TIERS: [(QualityPreset, QualityCeiling); 4] = [
    (QualityPreset::Low, QualityCeiling::LOW),
    (QualityPreset::Medium, QualityCeiling::MEDIUM),
    (QualityPreset::High, QualityCeiling::HIGH),
    (QualityPreset::Ultra, QualityCeiling::ULTRA),
];

// The GPU classes in the order of the tier `Auto` picks for them, lowest first.
const GPU_CLASSES: [GpuTier; 5] = [
    GpuTier::Integrated,
    GpuTier::Unknown,
    GpuTier::EntryDiscrete,
    GpuTier::MidDiscrete,
    GpuTier::HighDiscrete,
];

/// The graphics settings page.
pub(super) fn render(docs: &[AssetDoc]) -> String {
    let mut out = format!(
        "{AUTOGEN_MARKER}\n\n# Graphics settings\n\nThe settings below change how the scene \
         is rendered or shown, each under the key a settings row's action names.\n\n{}\n\n",
        SettingKey::DOC
    );
    out.push_str("## Quality presets\n\n");
    out.push_str(QualityPreset::DOC);
    out.push_str("\n\n### `Auto`\n\n");
    out.push_str(&first_paragraph(QualityPreset::Auto.doc()));
    out.push_str("\n\n");
    out.push_str(&auto_table());
    out.push_str("\n### Ceilings\n\n");
    out.push_str(&ceiling_table());
    for group in SettingGroup::ALL.into_iter().filter(|g| g.is_graphics()) {
        out.push_str("\n## ");
        out.push_str(group.name());
        out.push_str("\n\n");
        out.push_str(&first_paragraph(group.doc()));
        out.push('\n');
        for key in SettingKey::ALL.into_iter().filter(|k| k.group() == group) {
            out.push('\n');
            out.push_str(&setting_section(key, docs));
        }
    }
    out
}

// A markdown table row.
fn table_row(cells: &[String]) -> String {
    format!("| {} |\n", cells.join(" | "))
}

fn auto_table() -> String {
    let mut out = String::from("| GPU | Auto picks |\n| --- | --- |\n");
    for tier in GPU_CLASSES {
        let profile = GpuProfile {
            tier,
            ..GpuProfile::UNKNOWN
        };
        let label = preset_label(QualityPreset::Auto, &profile);
        let picked = label
            .strip_prefix("Auto (")
            .and_then(|l| l.strip_suffix(')'))
            .unwrap_or(&label);
        let gpu = first_paragraph(tier.doc());
        let gpu = gpu.strip_suffix('.').unwrap_or(&gpu);
        out.push_str(&table_row(&[gpu.to_string(), picked.to_string()]));
    }
    out
}

fn ceiling_table() -> String {
    let names: Vec<&str> = TIERS.iter().map(|(preset, _)| preset.name()).collect();
    let mut out = format!("| Setting | {} |\n", names.join(" | "));
    out.push_str(&table_row(&vec!["---".to_string(); 1 + TIERS.len()]));
    for key in SettingKey::ALL {
        if cap(key, &QualityCeiling::ULTRA).is_some() {
            let cells: Vec<String> = core::iter::once(format!("`{}`", key.as_str()))
                .chain(TIERS.iter().map(|(_, c)| cap(key, c).unwrap_or_default()))
                .collect();
            out.push_str(&table_row(&cells));
        }
    }
    out
}

// What a ceiling allows a setting the preset governs, or `None` for a setting
// no preset touches.
fn cap(key: SettingKey, c: &QualityCeiling) -> Option<String> {
    use SettingKey as K;
    let label =
        |labels: &[&str], index: usize| labels.get(index).copied().unwrap_or("").to_string();
    let allowed = |on: bool| if on { "On" } else { "Off" }.to_string();
    let pass = |r| {
        label(
            levels::PASS_RESOLUTION.labels(),
            levels::PASS_RESOLUTION.index_of(r),
        )
    };
    let uncapped = |v: u32, unit: &str| {
        if v == u32::MAX {
            "No cap".to_string()
        } else {
            format!("{v}{unit}")
        }
    };
    Some(match key {
        K::AaMode => label(
            levels::AA_MODE.labels(),
            levels::AA_MODE.index_of(c.aa_mode),
        ),
        K::Ssao => allowed(c.ssao),
        K::Ssr => allowed(c.ssr),
        K::RayTracedReflections => allowed(c.ray_traced_reflections),
        K::RtReflectionShadows => allowed(c.rt_reflection_shadows),
        K::Ssgi => allowed(c.ssgi),
        K::AutoExposure => allowed(c.auto_exposure),
        K::SsgiResolution => pass(c.ssgi_resolution),
        K::RtReflectionResolution => pass(c.rt_reflection_resolution),
        K::ReflectionBlurResolution => pass(c.reflection_blur_resolution),
        K::SsgiRays => c.ssgi_rays.to_string(),
        K::RenderScale => label(
            levels::RENDER_SCALE.labels(),
            levels::RENDER_SCALE.index_of(c.min_upscale),
        ),
        K::TemporalUpscaling if *c == QualityCeiling::ULTRA => "Native".to_string(),
        K::TemporalUpscaling => "On above 2560x1440".to_string(),
        K::ShadowMapSize => uncapped(c.shadow_map_size, ""),
        K::ShadowUpdate if c.allow_every_frame_shadows => "Every Frame".to_string(),
        K::ShadowUpdate => "Hybrid".to_string(),
        K::ShadowDistance => uncapped(c.shadow_distance, " m"),
        K::ShadowCascades => c.shadow_cascades.to_string(),
        K::Anisotropy => format!("{}x", c.anisotropy),
        _ => return None,
    })
}

fn setting_section(key: SettingKey, docs: &[AssetDoc]) -> String {
    let mut lines = vec![
        format!("### `{}`", key.as_str()),
        String::new(),
        first_paragraph(key.doc()),
        String::new(),
    ];
    if let Some(choices) = choices(key) {
        let label = if key.kind() == SettingKind::Slider {
            "Range"
        } else {
            "Options"
        };
        lines.push(format!("- **{label}:** {choices}"));
    }
    if let Some(field) = key.field() {
        let target = format!(
            "[`{}.{}`]({ASSETS}{}.md)",
            field.asset, field.key, field.asset
        );
        lines.push(match field_text(docs, field.asset, field.key) {
            Some(text) => format!("- **Overrides:** {target}. {text}"),
            None => format!("- **Overrides:** {target}."),
        });
    }
    if let Some(needs) = registry::availability(key) {
        lines.push(format!("- **Available with:** {needs}"));
    }
    if cap(key, &QualityCeiling::ULTRA).is_some() {
        let caps: Vec<String> = TIERS
            .iter()
            .map(|(preset, c)| format!("{} {}", preset.name(), cap(key, c).unwrap_or_default()))
            .collect();
        lines.push(format!("- **Preset caps:** {}", caps.join(", ")));
    }
    lines.push(String::new());
    lines.join("\n")
}

// The values a setting takes, as the menu shows them.
fn choices(key: SettingKey) -> Option<String> {
    match key.kind() {
        SettingKind::Slider => registry::slider(key)
            .map(|s| format!("{} to {}", (s.format)(s.range.0), (s.format)(s.range.1))),
        SettingKind::Cycle => match key {
            SettingKey::GraphicsApi => Some(
                Platform::ALL
                    .iter()
                    .map(|p| p.label())
                    .collect::<Vec<_>>()
                    .join(", ")
                    + ", where the build compiled them",
            ),
            SettingKey::Resolution => Some("The display's own modes".to_string()),
            _ => options(key).map(|o| o.join(", ")),
        },
        SettingKind::KeyRebind | SettingKind::PadRebind => None,
    }
}

// A field's description from its asset page, its links pointed at the asset
// pages.
fn field_text(docs: &[AssetDoc], asset: &str, key: &str) -> Option<String> {
    let doc = docs.iter().find(|d| d.type_name == asset)?;
    let (_, text) = doc.fields.iter().find(|(k, _)| k == key)?;
    Some(link_to_assets(text))
}

// Point every relative `.md` link at the asset pages beside this one.
fn link_to_assets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("](") {
        let (head, tail) = rest.split_at(at + 2);
        out.push_str(head);
        let target_end = tail.find(')').unwrap_or(tail.len());
        let target = &tail[..target_end];
        let is_page = std::path::Path::new(target)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("md"));
        if is_page && !target.contains('/') && !target.contains(':') {
            out.push_str(ASSETS);
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

// The upscaling row's text names the threshold `auto_upscaling` applies.
const _: () = assert!(AUTO_UPSCALE_PIXELS == 2560 * 1440);

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::render::backend::{
        GpuClassInput, GpuVendor, apple_family_from_device_name, apple_variant_from_device_name,
        classify_tier,
    };
    use concinnity_engine::settings::SettingField;

    const GB: u64 = 1 << 30;

    fn page() -> String {
        render(&super::super::reference::build(
            &super::super::reference::registry_assets(),
        ))
    }

    #[test]
    fn every_graphics_setting_has_a_section_and_no_other_does() {
        let page = page();
        for key in SettingKey::ALL {
            let heading = format!("### `{}`\n", key.as_str());
            assert_eq!(
                page.contains(&heading),
                key.group().is_graphics(),
                "{key:?}"
            );
        }
    }

    #[test]
    fn every_bound_field_has_a_description_on_its_asset_page() {
        let docs = super::super::reference::build(&super::super::reference::registry_assets());
        for key in SettingKey::ALL {
            if let Some(SettingField { asset, key: field }) = key.field() {
                let text = field_text(&docs, asset, field);
                assert!(
                    text.is_some_and(|t| !t.is_empty()),
                    "{key:?} -> {asset}.{field}"
                );
            }
        }
    }

    #[test]
    fn the_ceiling_table_reads_each_tier() {
        assert_eq!(
            cap(SettingKey::RtReflectionResolution, &QualityCeiling::ULTRA).as_deref(),
            Some("Full")
        );
        assert_eq!(
            cap(SettingKey::RayTracedReflections, &QualityCeiling::HIGH).as_deref(),
            Some("Off")
        );
        assert_eq!(
            cap(SettingKey::ShadowMapSize, &QualityCeiling::ULTRA).as_deref(),
            Some("No cap")
        );
        assert_eq!(
            cap(SettingKey::ShadowDistance, &QualityCeiling::LOW).as_deref(),
            Some("40 m")
        );
        assert_eq!(cap(SettingKey::Vsync, &QualityCeiling::ULTRA), None);
        let page = page();
        assert!(page.contains("| `rt_reflection_resolution` | Quarter | Half | Half | Full |"));
    }

    // Each GPU description names hardware the probe really places in that
    // class.
    #[test]
    fn the_gpu_descriptions_match_the_probe() {
        let discrete = |vram| GpuClassInput {
            vendor: GpuVendor::Nvidia,
            memory_budget_bytes: vram,
            discrete: true,
            apple_family: 0,
            apple_variant: None,
        };
        let examples = [
            GpuClassInput {
                discrete: false,
                ..discrete(0)
            },
            discrete(0),
            discrete(4 * GB),
            discrete(8 * GB),
            discrete(32 * GB),
        ];
        for (tier, input) in GPU_CLASSES.into_iter().zip(examples) {
            assert_eq!(classify_tier(&input), tier, "{}", tier.doc());
        }
        // The Apple chips the Mid and High rows name.
        let apple = |name: &str| GpuClassInput {
            vendor: GpuVendor::Apple,
            discrete: false,
            apple_family: apple_family_from_device_name(name),
            apple_variant: apple_variant_from_device_name(name),
            ..discrete(32 * GB)
        };
        for (name, tier) in [
            ("Apple M1 Max", GpuTier::MidDiscrete),
            ("Apple M2 Max", GpuTier::MidDiscrete),
            ("Apple M1 Ultra", GpuTier::MidDiscrete),
            ("Apple M3 Max", GpuTier::HighDiscrete),
            ("Apple M2 Ultra", GpuTier::HighDiscrete),
        ] {
            assert_eq!(classify_tier(&apple(name)), tier, "{name}");
        }
    }

    #[test]
    fn the_gpu_classes_rise_with_the_tier_auto_picks() {
        let rank = |tier| {
            let profile = GpuProfile {
                tier,
                ..GpuProfile::UNKNOWN
            };
            let label = preset_label(QualityPreset::Auto, &profile);
            TIERS
                .iter()
                .position(|(preset, _)| label == format!("Auto ({})", preset.name()))
                .expect("Auto picks a named tier")
        };
        let ranks: Vec<usize> = GPU_CLASSES.into_iter().map(rank).collect();
        assert!(ranks.is_sorted(), "{ranks:?}");
    }

    // Every tier has a row. Adding a GpuTier stops the match compiling until it
    // gets an arm here.
    #[test]
    fn every_gpu_class_has_a_row() {
        let all = [
            GpuTier::Unknown,
            GpuTier::Integrated,
            GpuTier::EntryDiscrete,
            GpuTier::MidDiscrete,
            GpuTier::HighDiscrete,
        ];
        for tier in all {
            match tier {
                GpuTier::Unknown
                | GpuTier::Integrated
                | GpuTier::EntryDiscrete
                | GpuTier::MidDiscrete
                | GpuTier::HighDiscrete => assert!(GPU_CLASSES.contains(&tier), "{tier:?}"),
            }
        }
    }

    #[test]
    fn relative_links_point_at_the_asset_pages() {
        assert_eq!(
            link_to_assets("see [PassResolution](PassResolution.md) or [x](https://a.b/c.md)"),
            "see [PassResolution](../assets/PassResolution.md) or [x](https://a.b/c.md)"
        );
        assert_eq!(link_to_assets("no links"), "no links");
    }
}
