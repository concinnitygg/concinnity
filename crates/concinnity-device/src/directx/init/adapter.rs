// src/directx/init/adapter.rs
//
// Which adapter the renderer runs on, and whether it can run there at all.
//
// Selection and the capability gate are one decision rather than two: an
// adapter is only a candidate once a device made on it reports a resource
// binding tier the texture pool can bind, so rejecting one has to leave the
// next one free to be tried. Hardware is always preferred; the DXGI software
// adapter is the last candidate, so a machine whose GPU is below the
// renderer's floor gets a picture instead of an exit.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

/// The adapter the renderer will run on, together with the device made on it.
/// The device is kept rather than remade because creating it is what proved
/// the adapter usable.
pub(super) struct Selection {
    pub adapter: IDXGIAdapter1,
    pub device: ID3D12Device,
}

/// Pick an adapter the renderer can run on: hardware first, in DXGI's own
/// order, then the software adapter.
///
/// Falling back is automatic because the alternative is refusing to start, and
/// an app that opens is worth more than one that explains itself and exits. It
/// is loud rather than silent: the warning names the software adapter and why
/// every hardware one was passed over, since a user whose GPU should have
/// worked needs to learn that from the app rather than from its frame rate.
///
/// Every rejection is carried rather than counted, since the reason a
/// particular machine cannot run the renderer is the whole content of both the
/// warning and, when even the software adapter fails, the error.
pub(super) fn select(factory: &IDXGIFactory4) -> Result<Selection, String> {
    let mut rejections: Vec<String> = Vec::new();

    let mut i = 0u32;
    // SAFETY: a query on a live COM object; the out-parameter it fills is a live local that
    // outlives the call.
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        i += 1;
        // SAFETY: a property query on a live COM object; it only reads.
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            continue;
        }
        match consider(&adapter) {
            Ok(selection) => return Ok(selection),
            Err(reason) => rejections.push(reason),
        }
    }

    // SAFETY: a query on a live COM object; the new COM object lands in a binding that owns it.
    match unsafe { factory.EnumWarpAdapter::<IDXGIAdapter1>() } {
        Ok(warp) => match consider(&warp) {
            Ok(selection) => {
                tracing::warn!(
                    "d3d12 adapter: falling back to the software adapter, which renders \
                     every pass on the CPU and is far slower than any GPU. No hardware \
                     adapter could run the renderer: {}",
                    rejections.join("; ")
                );
                return Ok(selection);
            }
            Err(reason) => rejections.push(reason),
        },
        Err(e) => rejections.push(format!("the software adapter is unavailable: {e}")),
    }

    Err(no_usable_adapter_message(&rejections))
}

// Make a device on one adapter and gate it on the binding tier. An adapter that
// cannot make a device and one whose device cannot bind the texture pool are
// the same answer here: not this one.
fn consider(adapter: &IDXGIAdapter1) -> Result<Selection, String> {
    let mut device_opt: Option<ID3D12Device> = None;
    // SAFETY: the adapter is live for the call, and the new COM object lands in a binding that
    // owns it.
    unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &mut device_opt) }
        .map_err(|e| format!("{}: D3D12CreateDevice: {e}", adapter_name(adapter)))?;
    let device = device_opt
        .ok_or_else(|| format!("{}: D3D12CreateDevice returned None", adapter_name(adapter)))?;
    if let Some(refusal) =
        binding_tier_refusal(resource_binding_tier(&device), &adapter_name(adapter))
    {
        return Err(refusal);
    }
    Ok(Selection {
        adapter: adapter.clone(),
        device,
    })
}

/// The resource binding tier a device reports, or tier 1 when the query fails
/// (the tier the runtime guarantees, and the one that refuses).
pub(super) fn resource_binding_tier(device: &ID3D12Device) -> D3D12_RESOURCE_BINDING_TIER {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a query on a live COM object; the descriptor it fills is a live local that outlives
    // the call.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            &mut options as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };
    if ok.is_ok() {
        options.ResourceBindingTier
    } else {
        D3D12_RESOURCE_BINDING_TIER_1
    }
}

// Why this adapter cannot run the renderer, if it cannot.
//
// The texture pool is one unbounded descriptor range, which the runtime accepts
// only on binding tier 3. Below it every root signature holding that range is
// rejected at creation, so the renderer dies on its first pipeline with a bare
// `E_INVALIDARG` naming nothing. Hardware has reported tier 3 since 2016; the
// adapters that do not are virtualised or software ones.
pub(super) fn binding_tier_refusal(
    tier: D3D12_RESOURCE_BINDING_TIER,
    adapter: &str,
) -> Option<String> {
    if tier.0 >= D3D12_RESOURCE_BINDING_TIER_3.0 {
        return None;
    }
    Some(format!(
        "{adapter} reports D3D12 resource binding tier {}; the renderer needs tier 3, \
         because it binds its textures through an unbounded descriptor range",
        tier.0
    ))
}

// Not even the software adapter was usable, which leaves nothing to run on.
// Every rejection is quoted, because a machine that cannot run the renderer is
// told why by exactly this string, and the reasons differ per adapter on a
// laptop with two of them.
//
// One reason is worth adding to. On a binary that bundles the Agility SDK,
// "D3D12CreateDevice" failing on every adapter almost never means the GPU:
// `d3d12.dll` read the `D3D12SDKPath` export at process start (see
// `directx::agility`), could not load `D3D12Core.dll` from beside the
// executable, and left the D3D12 runtime dead. That takes the software adapter
// down with it, which is how this path is reached on a machine that has one.
fn no_usable_adapter_message(rejections: &[String]) -> String {
    compose_refusal(rejections, cfg!(agility_sdk_configured), || {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|d| d.join("D3D12").display().to_string()))
    })
}

// The message composition, separated from the queries that feed it so it can be
// read back without a device.
fn compose_refusal(
    rejections: &[String],
    agility_bundled: bool,
    agility_dir: impl Fn() -> Option<String>,
) -> String {
    let mut message = if rejections.is_empty() {
        "no D3D12 adapter found".to_string()
    } else {
        format!("no usable D3D12 adapter: {}", rejections.join("; "))
    };
    if agility_bundled {
        message.push_str(&format!(
            ". This binary was built with CN_ENABLE_AGILITY_SDK=1, so it bundles \
             Microsoft's Agility SDK and needs D3D12Core.dll in {}; without it \
             D3D12 fails to start and every adapter reports unsupported. Copy \
             that directory next to the executable, or rebuild without the \
             opt-in to use the OS D3D12 runtime",
            agility_dir()
                .unwrap_or_else(|| "a `D3D12` directory beside the executable".to_string())
        ));
    }
    message
}

/// The adapter's description string, or a stand-in when the query fails.
pub(super) fn adapter_name(adapter: &IDXGIAdapter1) -> String {
    // SAFETY: a property query on a live COM object; it only fills the descriptor it is given.
    let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
        return "this adapter".to_string();
    };
    let end = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_three_is_what_the_texture_pool_needs() {
        assert_eq!(
            binding_tier_refusal(D3D12_RESOURCE_BINDING_TIER_3, "GPU"),
            None
        );
    }

    #[test]
    fn a_lower_tier_is_refused_by_name_and_number() {
        for tier in [D3D12_RESOURCE_BINDING_TIER_1, D3D12_RESOURCE_BINDING_TIER_2] {
            let refusal = binding_tier_refusal(tier, "Parallels Display Adapter")
                .expect("a tier below 3 cannot bind the texture pool");
            assert!(refusal.contains("Parallels Display Adapter"), "{refusal}");
            assert!(refusal.contains(&tier.0.to_string()), "{refusal}");
        }
    }

    // Every adapter's reason reaches the message, including the software
    // adapter's: reaching this at all means that one failed too, and which of
    // them failed how is the whole content of the error.
    #[test]
    fn every_rejection_reaches_the_message() {
        let message = compose_refusal(
            &[
                "one is tier 1".to_string(),
                "the software adapter is unavailable".to_string(),
            ],
            false,
            || None,
        );
        assert!(message.contains("one is tier 1"), "{message}");
        assert!(
            message.contains("the software adapter is unavailable"),
            "{message}"
        );
    }

    #[test]
    fn an_enumeration_that_found_nothing_says_so() {
        let message = compose_refusal(&[], false, || None);
        assert!(message.contains("no D3D12 adapter found"), "{message}");
    }

    // A bundled Agility SDK that cannot load leaves every adapter looking
    // unsupported, so the message names the directory rather than the hardware.
    #[test]
    fn a_bundled_agility_sdk_names_the_directory_it_needs() {
        let message = compose_refusal(&["D3D12CreateDevice failed".to_string()], true, || {
            Some("C:\\app\\D3D12".to_string())
        });
        assert!(message.contains("C:\\app\\D3D12"), "{message}");
        assert!(message.contains("D3D12Core.dll"), "{message}");
    }

    #[test]
    fn an_unlocatable_agility_directory_still_names_what_is_missing() {
        let message = compose_refusal(&[], true, || None);
        assert!(message.contains("D3D12Core.dll"), "{message}");
        assert!(message.contains("beside the executable"), "{message}");
    }
}
