//! The runtime SDKs an app loads when it runs: the libraries the upscalers and
//! the newer D3D12 runtime come in. All of them are Windows-only.

use super::component::{Component, License, RuntimeFile, Source};
use super::host::Host;

/// Every runtime SDK, whose libraries the runtime directory stages.
pub(crate) const SDKS: [&Component; 4] = [&AGILITY, &FIDELITYFX, &XESS, &STREAMLINE];

const WINDOWS: &[Host] = &[Host::WindowsX64];

fn download(url: String) -> Source {
    Source::Download { url, keep: None }
}

/// Microsoft's D3D12 Agility SDK, a package from Microsoft's own feed: a zip
/// under another name, which its flat container serves without an API key. Its
/// runtime goes in a `D3D12/` directory of its own, which the renderer opens as
/// the process's D3D12 runtime.
pub(crate) const AGILITY: Component = Component {
    name: "agility",
    release: "1.619.3",
    summary: "Microsoft's D3D12 Agility SDK, which FSR on DirectX needs",
    hosts: WINDOWS,
    source: |_| {
        let release = AGILITY.release;
        download(format!(
            "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/{release}/microsoft.direct3d.d3d12.{release}.nupkg"
        ))
    },
    payload: |_| "build/native/bin/x64/D3D12Core.dll",
    sha256: Some(&[(
        Host::WindowsX64,
        "43a7d5a3973812eb4b42623fae5275c790a005b8e48b8d7f5bb43cef39e073c5",
    )]),
    tool: false,
    runtime: &[
        RuntimeFile {
            from: "build/native/bin/x64/D3D12Core.dll",
            to: "D3D12/D3D12Core.dll",
        },
        RuntimeFile {
            from: "build/native/bin/x64/d3d12SDKLayers.dll",
            to: "D3D12/d3d12SDKLayers.dll",
        },
    ],
    notices: &[
        RuntimeFile {
            from: "LICENSE.txt",
            to: "licenses/agility/LICENSE.txt",
        },
        RuntimeFile {
            from: "distributable files.txt",
            to: "licenses/agility/distributable files.txt",
        },
    ],
    // The package itself asks for no acceptance before it is installed.
    licenses: &[License {
        name: "Microsoft DirectX SDK license (Agility SDK)",
        url: "https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.619.3/License",
        agreement: false,
    }],
};

/// AMD's FidelityFX SDK, for FSR on DirectX and Vulkan.
///
/// The SDK publishes no release asset, only the tagged source archive, which
/// carries AMD's signed runtimes. It is cut to the loader and upscaler library
/// of each API and the license. The loader is what the renderer opens; it finds
/// the upscaler library beside itself.
pub(crate) const FIDELITYFX: Component = Component {
    name: "fidelityfx",
    release: "2.4.0",
    summary: "AMD FidelityFX, for FSR on DirectX and Vulkan",
    hosts: WINDOWS,
    source: |_| {
        let release = FIDELITYFX.release;
        Source::Download {
            url: format!(
                "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/archive/refs/tags/v{release}.zip"
            ),
            keep: Some(&[
                "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll",
                "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll",
                "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll",
                "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
                "Kits/FidelityFX/docs/license.md",
                "3rdpartynotice.md",
            ]),
        }
    },
    payload: |_| "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll",
    sha256: Some(&[(
        Host::WindowsX64,
        "27120e3bbe0d0b30213825ad40f19077563f04f04572e252109a392d5fb522cb",
    )]),
    tool: false,
    runtime: &[
        RuntimeFile {
            from: "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll",
            to: "amd_fidelityfx_loader_dx12.dll",
        },
        RuntimeFile {
            from: "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll",
            to: "amd_fidelityfx_upscaler_dx12.dll",
        },
        RuntimeFile {
            from: "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll",
            to: "amd_fidelityfx_loader_vk.dll",
        },
        RuntimeFile {
            from: "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
            to: "amd_fidelityfx_upscaler_vk.dll",
        },
    ],
    notices: &[
        RuntimeFile {
            from: "Kits/FidelityFX/docs/license.md",
            to: "licenses/fidelityfx/license.md",
        },
        RuntimeFile {
            from: "3rdpartynotice.md",
            to: "licenses/fidelityfx/3rdpartynotice.md",
        },
    ],
    licenses: &[License {
        name: "MIT (FidelityFX SDK)",
        url: "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v2.4.0/Kits/FidelityFX/docs/license.md",
        agreement: false,
    }],
};

/// Intel's XeSS SDK.
pub(crate) const XESS: Component = Component {
    name: "xess",
    release: "3.0.1",
    summary: "Intel XeSS",
    hosts: WINDOWS,
    source: |_| {
        let release = XESS.release;
        download(format!(
            "https://github.com/intel/xess/releases/download/v{release}/XeSS_SDK_{release}.zip"
        ))
    },
    payload: |_| "bin/libxess.dll",
    sha256: Some(&[(
        Host::WindowsX64,
        "b1a833677884b644a605276194c64706591dc2c41d66a2ff46808fedfab02df9",
    )]),
    tool: false,
    runtime: &[RuntimeFile {
        from: "bin/libxess.dll",
        to: "libxess.dll",
    }],
    notices: &[
        RuntimeFile {
            from: "LICENSE.txt",
            to: "licenses/xess/LICENSE.txt",
        },
        RuntimeFile {
            from: "third-party-programs.txt",
            to: "licenses/xess/third-party-programs.txt",
        },
    ],
    licenses: &[License {
        name: "Intel Simplified Software License",
        url: "https://github.com/intel/xess/blob/v3.0.1/LICENSE.txt",
        agreement: false,
    }],
};

/// NVIDIA Streamline, which DLSS runs through: its core, the common plugin and
/// the low-latency library that plugin opens on Vulkan, and the DLSS plugin with
/// its feature library.
pub(crate) const STREAMLINE: Component = Component {
    name: "streamline",
    release: "2.14.16",
    summary: "NVIDIA Streamline, for DLSS",
    hosts: WINDOWS,
    source: |_| {
        let release = STREAMLINE.release;
        download(format!(
            "https://github.com/NVIDIA-RTX/Streamline/releases/download/v{release}/streamline-sdk-v{release}.zip"
        ))
    },
    payload: |_| "bin/x64/sl.interposer.dll",
    sha256: Some(&[(
        Host::WindowsX64,
        "32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b",
    )]),
    tool: false,
    runtime: &[
        RuntimeFile {
            from: "bin/x64/sl.interposer.dll",
            to: "sl.interposer.dll",
        },
        RuntimeFile {
            from: "bin/x64/sl.common.dll",
            to: "sl.common.dll",
        },
        RuntimeFile {
            from: "bin/x64/NvLowLatencyVk.dll",
            to: "NvLowLatencyVk.dll",
        },
        RuntimeFile {
            from: "bin/x64/sl.dlss.dll",
            to: "sl.dlss.dll",
        },
        RuntimeFile {
            from: "bin/x64/nvngx_dlss.dll",
            to: "nvngx_dlss.dll",
        },
    ],
    notices: &[
        RuntimeFile {
            from: "license.txt",
            to: "licenses/streamline/license.txt",
        },
        RuntimeFile {
            from: "bin/x64/nvngx_dlss.license.txt",
            to: "licenses/streamline/nvngx_dlss.license.txt",
        },
        RuntimeFile {
            from: "bin/x64/reflex.license.txt",
            to: "licenses/streamline/reflex.license.txt",
        },
    ],
    licenses: &[
        License {
            name: "MIT (Streamline)",
            url: "https://github.com/NVIDIA-RTX/Streamline/blob/v2.14.16/license.txt",
            agreement: false,
        },
        License {
            name: "NVIDIA RTX SDKs License (DLSS)",
            url: "https://github.com/NVIDIA-RTX/Streamline/blob/v2.14.16/external/ngx-sdk/license.txt",
            agreement: true,
        },
        License {
            name: "NVIDIA Reflex SDK License (NvLowLatencyVk)",
            url: "https://github.com/NVIDIA-RTX/Streamline/blob/v2.14.16/external/reflex-sdk-vk/reflex.license.txt",
            agreement: true,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_agility_release_is_the_runtime_the_renderer_opens() {
        assert_eq!(
            AGILITY.release.split('.').nth(1),
            Some(concinnity_device::AGILITY_SDK_VERSION.to_string().as_str()),
            "{}",
            AGILITY.release
        );
    }

    #[test]
    fn every_download_is_pinned() {
        for sdk in SDKS {
            for &host in sdk.hosts {
                assert!(
                    sdk.pinned_digest(host).is_some(),
                    "{}/{} downloads with no pinned sha256",
                    sdk.name,
                    host.slug()
                );
            }
        }
    }

    #[test]
    fn every_sdk_is_downloaded_and_ships_libraries_under_its_terms() {
        for sdk in SDKS {
            for &host in sdk.hosts {
                assert!(!sdk.is_built(host), "{}", sdk.name);
            }
            assert!(!sdk.runtime.is_empty(), "{}", sdk.name);
            assert!(!sdk.licenses.is_empty(), "{}", sdk.name);
            for file in sdk.runtime {
                assert!(
                    std::path::Path::new(file.to)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("dll")),
                    "{file:?}"
                );
                assert!(!file.to.starts_with('/'), "{file:?}");
            }
        }
    }

    // The terms that are agreements gate the download; the notices do not.
    #[test]
    fn only_streamline_needs_its_terms_accepted() {
        assert!(STREAMLINE.needs_acceptance());
        assert!(!AGILITY.needs_acceptance());
        assert!(!FIDELITYFX.needs_acceptance());
        assert!(!XESS.needs_acceptance());
    }

    // A license link names the release it was read for, so a bumped pin that
    // forgets its links fails here.
    #[test]
    fn every_license_link_names_its_release() {
        for sdk in SDKS {
            for license in sdk.licenses {
                assert!(
                    license.url.contains(sdk.release),
                    "{}: {}",
                    sdk.name,
                    license.url
                );
            }
        }
    }

    #[test]
    fn notices_land_under_the_sdks_own_license_directory() {
        for sdk in SDKS {
            assert!(!sdk.notices.is_empty(), "{}", sdk.name);
            for notice in sdk.notices {
                assert!(
                    notice.to.starts_with(&format!("licenses/{}/", sdk.name)),
                    "{notice:?}"
                );
            }
        }
    }

    #[test]
    fn the_fidelityfx_archive_is_cut_to_the_runtimes() {
        let Source::Download { keep, .. } = (FIDELITYFX.source)(Host::WindowsX64) else {
            panic!("fidelityfx is downloaded");
        };
        let root = "FidelityFX-SDK-2.4.0/";
        let members = [
            (
                "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll",
                true,
            ),
            (
                "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
                true,
            ),
            ("Kits/FidelityFX/docs/license.md", true),
            (
                "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.lib",
                false,
            ),
            (
                "Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_vk.dll",
                false,
            ),
        ];
        for (member, want) in members {
            assert_eq!(
                crate::vendor::unpack::wanted(keep, &format!("{root}{member}")),
                want,
                "{member}"
            );
        }
        for file in FIDELITYFX.runtime.iter().chain(FIDELITYFX.notices) {
            assert!(crate::vendor::unpack::wanted(
                keep,
                &format!("{root}{}", file.from)
            ));
        }
    }
}
