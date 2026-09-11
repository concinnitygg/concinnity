<!-- Auto-generated - do not edit. -->

# UpscalerBackend

Upscaler backend selector for `PostProcessConfig.temporal_upscaling`.
`Auto` resolves at runtime to the best available (DLSS, then XeSS, then
FSR3); the explicit variants request a specific backend and fall back when
it is unavailable. DirectX and Vulkan both resolve all three, each from the
vendor SDK staged beside the binary; Metal uses MetalFX and treats any value
as its native path.

## Values

- `auto`: Pick the best backend the device offers.
- `fsr3`: AMD FidelityFX Super Resolution 3.
- `dlss`: NVIDIA DLSS, through NGX.
- `xess`: Intel XeSS.
