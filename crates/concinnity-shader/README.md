# concinnity-shader

Build-time shader compilers for the
[Concinnity](https://crates.io/crates/concinnity) engine backends.

The platform compilers that turn a Shader asset's authored source into the
bytecode one backend loads: Metal (`xcrun metal` + `xcrun metallib`),
DirectX (the Direct3D `Fxc` compiler), Vulkan (`shaderc`). Exactly one
compiles per build, resolved by the build script into a single `backend_*`
cfg.

This is the build-side twin of `concinnity-device`: the device crate owns
runtime GPU submission, this one owns build-time shader production, and
neither is on the other's path. Keeping them apart lets `concinnity-cook`,
which must run on build hosts with no GPU and no platform compiler, stay
free of backend cfgs and native dependencies: the cook declares a
`ShaderToolchain` seam and calls whatever is registered; a binary calls
`install` once at startup to fill it in.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
