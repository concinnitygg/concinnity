// src/vulkan/instance_exts.rs
//
// Which optional instance extensions to enable, resolved against the names the
// Vulkan loader advertises. Separated from the instance creation in `init.rs` so
// the decision is a pure function over an extension-name list and can be tested
// without a loader.

use std::ffi::CStr;

use ash::vk;

// The optional instance extensions an instance should enable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OptionalInstanceExts {
    // VK_EXT_swapchain_colorspace, which widens the surface-format query to the
    // extended-range colour spaces the HDR output path looks for. Only ever set
    // when the world asked for HDR.
    pub swapchain_colorspace: bool,
    // VK_KHR_portability_enumeration. The loader hides portability drivers
    // (MoltenVK on macOS) from `enumerate_physical_devices` unless this is
    // enabled and `ENUMERATE_PORTABILITY_KHR` is set on the create info, so
    // without it a macOS Vulkan build finds no physical device at all.
    pub portability_enumeration: bool,
}

impl OptionalInstanceExts {
    // The names to append to `VkInstanceCreateInfo::ppEnabledExtensionNames`.
    pub fn names(self) -> Vec<&'static CStr> {
        let mut names = Vec::new();
        if self.swapchain_colorspace {
            names.push(ash::ext::swapchain_colorspace::NAME);
        }
        if self.portability_enumeration {
            names.push(ash::khr::portability_enumeration::NAME);
        }
        names
    }

    // The create flags these extensions require.
    pub fn flags(self) -> vk::InstanceCreateFlags {
        if self.portability_enumeration {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        }
    }
}

// Resolve the optional extensions against `available`, the loader's advertised
// instance-extension names. A missing extension degrades (SDR output, no
// portability drivers) rather than failing instance creation.
pub(super) fn select(available: &[&CStr], hdr_display: bool) -> OptionalInstanceExts {
    let has = |name: &CStr| available.contains(&name);
    OptionalInstanceExts {
        swapchain_colorspace: hdr_display && has(ash::ext::swapchain_colorspace::NAME),
        portability_enumeration: has(ash::khr::portability_enumeration::NAME),
    }
}

// The loader's advertised instance-extension names, as borrowed C strings over
// `props`. Split from `select` so the caller owns the properties buffer the
// names point into.
pub(super) fn names_of(props: &[vk::ExtensionProperties]) -> Vec<&CStr> {
    props
        .iter()
        .map(|p| unsafe { CStr::from_ptr(p.extension_name.as_ptr()) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLORSPACE: &CStr = ash::ext::swapchain_colorspace::NAME;
    const PORTABILITY: &CStr = ash::khr::portability_enumeration::NAME;
    const SURFACE: &CStr = ash::khr::surface::NAME;

    #[test]
    fn colorspace_needs_both_the_request_and_the_extension() {
        assert!(select(&[COLORSPACE, SURFACE], true).swapchain_colorspace);
        // Requested but unavailable: degrade to SDR.
        assert!(!select(&[SURFACE], true).swapchain_colorspace);
        // Available but not requested: an SDR world never pays for it.
        assert!(!select(&[COLORSPACE], false).swapchain_colorspace);
    }

    #[test]
    fn portability_follows_the_loader_alone() {
        // Unlike the colour space, nothing requests portability: enable it
        // wherever the loader has it so MoltenVK is enumerable.
        assert!(select(&[PORTABILITY], false).portability_enumeration);
        assert!(select(&[PORTABILITY], true).portability_enumeration);
        assert!(!select(&[SURFACE], false).portability_enumeration);
    }

    #[test]
    fn names_and_flags_follow_the_selection() {
        let none = OptionalInstanceExts::default();
        assert!(none.names().is_empty());
        assert_eq!(none.flags(), vk::InstanceCreateFlags::empty());

        let both = select(&[COLORSPACE, PORTABILITY], true);
        assert_eq!(both.names(), vec![COLORSPACE, PORTABILITY]);
        assert_eq!(
            both.flags(),
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        );

        // The colour space carries no create flag of its own.
        let colorspace_only = select(&[COLORSPACE], true);
        assert_eq!(colorspace_only.names(), vec![COLORSPACE]);
        assert_eq!(colorspace_only.flags(), vk::InstanceCreateFlags::empty());
    }

    #[test]
    fn names_read_back_from_extension_properties() {
        let mut props = vk::ExtensionProperties::default();
        let bytes = PORTABILITY.to_bytes_with_nul();
        for (dst, src) in props.extension_name.iter_mut().zip(bytes) {
            *dst = *src as std::os::raw::c_char;
        }
        let props = [props];
        assert_eq!(names_of(&props), vec![PORTABILITY]);
    }
}
