# concinnity-ffi

The C ABI a host application links to embed the Concinnity engine.

Builds as a static archive, a shared object and an rlib at once: an iOS app
links the archive, an Android one loads the shared object, a desktop host takes
either, and the rlib is what gives the surface unit tests. `include/concinnity.h`
is generated from `src/ffi.rs` by cbindgen and is the public API.

The surface is one world's lifecycle inside a view the host owns: `cn_init`,
`cn_world_open`, `cn_world_step`, `cn_world_close`. A host whose OS owns the run
loop calls `cn_world_step` once per display refresh. Authoring lives in the dev
tooling and is deliberately not here.

`private/scripts/release.py build ios-aarch64` packages the device and
simulator slices as an `.xcframework`. See `private/docs/mobile-port.md` for
what each platform can drive today.
