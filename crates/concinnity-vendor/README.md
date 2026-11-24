# concinnity-vendor

Vendoring for the [Concinnity](https://crates.io/crates/concinnity)
workspace: each third-party binary pinned to one release, downloaded,
checked against its pinned sha256 where it has one, and unpacked into
`<vendor>/<name>-<version>-<os>-<arch>/`.

`cargo xtask vendor` uses it for the shader compiler the build needs, and
`concinnity vendor` for the runtime SDKs an app loads (Agility, FidelityFX,
XeSS and Streamline), which it also stages into one runtime directory that
runs load from and exports bundle.
