#!/usr/bin/env python3
"""Vendor the third-party binaries this checkout builds against.

A component is pinned to one release and unpacks into
`vendor/<name>-<version>-<os>-<arch>/`, which the build prefers over whatever
the machine happens to have installed, so what a checkout builds with is a
property of the revision rather than of the host.

    vendor.py fetch [component ...]     download the pinned release
    vendor.py build [component ...]     build what no upstream publishes
    vendor.py status [component ...]    report what the build would pick
    vendor.py selftest                  check the digest gate, downloading nothing

With no component named, every one applies, and a component the verb does not
apply to is skipped: most are downloaded, `fidelityfx-vk` is built.

A downloaded archive is checked against a pinned sha256 before it is unpacked,
so a release re-tagged in place fails the fetch rather than reaching a build.

Nothing here runs during `cargo build`: a build script that reached the network
could not build offline, could not build on docs.rs, and would pull a dependency
past Cargo.lock, cargo-vendor and cargo-deny. Vendoring is a setup step, run
once, and the build only ever reads what it left behind.
"""

import argparse
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"


def is_windows():
    return platform.system() == "Windows"


class Slang:
    """The Slang shader compiler, which every backend's shaders compile through.

    Raising the pin is a deliberate change: a different compiler emits different
    bytes for the same shader source, so the bump belongs in a commit alongside
    whatever it was verified against.
    """

    name = "slang"
    release = "2026.16.1"
    reports_version = True
    build_payload = None
    summary = "the Slang shader compiler (slangc)"
    # The build scripts that resolve this component out of `vendor/`.
    watchers = ("crates/concinnity-slang/build.rs",)
    slugs = {
        ("Darwin", "arm64"): "macos-aarch64",
        ("Darwin", "x86_64"): "macos-x86_64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Linux", "x86_64"): "linux-x86_64",
        ("Windows", "ARM64"): "windows-aarch64",
        ("Windows", "AMD64"): "windows-x86_64",
    }
    # sha256 of each published archive, taken from the release's own asset
    # digests. Raising `release` means replacing every one of these: a pin that
    # outlives its version accepts nothing, which is the intended failure.
    sha256 = {
        "macos-aarch64": "31bb295d0ead64f5906ae140fb42067029412ca02330c11ff8ea63986560216a",
        "macos-x86_64": "ced9cd7f3773cdf1cb083bab96a806942d16d139614be104de47492253a38621",
        "linux-aarch64": "ce0f5a1a0dc1824aa4d89b17df22c8c06911dc37b31a75cadb50850f97bd4bea",
        "linux-x86_64": "6c271f69309af124cf948a9f442b813fec190feb46ff7a883e11001d29df005f",
        "windows-aarch64": "315a18a2ee56803bf558778d91481b47cefb51df14207342afdc9a4d9166c588",
        "windows-x86_64": "0fd3e6a9a5d05ed4cdd000d467f1ffb5d9701b827e83bfb428902a45c37ef8a5",
    }

    @staticmethod
    def payload():
        """The file inside the release that must be there for it to be usable."""
        return Path("bin") / ("slangc.exe" if is_windows() else "slangc")

    @classmethod
    def url(cls, slug):
        archive = "zip" if is_windows() else "tar.gz"
        return (
            "https://github.com/shader-slang/slang/releases/download"
            f"/v{cls.release}/slang-{cls.release}-{slug}.{archive}"
        )

    @staticmethod
    def version(payload):
        """What the vendored binary reports, or None if it cannot be run."""
        return first_line(run_quietly([str(payload), "-version"]))

    @staticmethod
    def alternatives():
        """Where else the build would look, for `status` to report."""
        found = []
        on_path = shutil.which("slangc")
        if on_path:
            found.append(("PATH", Path(on_path)))
        sdk = os.environ.get("VULKAN_SDK")
        if sdk:
            found.append(("VULKAN_SDK", Path(sdk) / Slang.payload()))
        return found


WINDOWS_X64 = {("Windows", "AMD64"): "windows-x86_64"}


class WindowsSdk:
    """A graphics SDK the DirectX build links or bundles, unpacked as shipped.

    These are Windows-only, so `fetch` on another host skips them rather than
    failing: nothing on a Mac or a Linux box has any use for them. None ships a
    tool that reports its own version, so the payload being where the build
    script looks is the whole check.
    """

    reports_version = False
    build_payload = None
    slugs = WINDOWS_X64
    # No pinned digest yet, so `fetch` reports what it got and unpacks it.
    sha256 = None
    watchers = ("build.rs", "crates/concinnity-device/build.rs",
                "crates/concinnity-engine/build.rs", "crates/concinnity-dev/build.rs")

    @classmethod
    def payload(cls):
        return Path(*cls.payload_parts)

    @staticmethod
    def version(payload):
        return None

    @staticmethod
    def alternatives():
        return []


class Agility(WindowsSdk):
    name = "agility"
    release = "1.619.3"
    summary = "Microsoft's D3D12 Agility SDK, which FSR 3 needs"
    # A NuGet package is a zip; the flat container serves it without an API key.
    payload_parts = ("build", "native", "bin", "x64", "D3D12Core.dll")

    @classmethod
    def url(cls, _slug):
        pkg = f"microsoft.direct3d.d3d12.{cls.release}"
        return f"https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/{cls.release}/{pkg}.nupkg"


class FidelityFx(WindowsSdk):
    name = "fidelityfx"
    release = "1.1.4"
    summary = "AMD FidelityFX, for FSR 3 on DirectX"
    payload_parts = ("bin", "amd_fidelityfx_dx12.dll")

    @classmethod
    def url(cls, _slug):
        return ("https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download"
                f"/v{cls.release}/FidelityFX-SDK-v{cls.release}.zip")


class FidelityFxVulkan(WindowsSdk):
    """AMD's Vulkan FSR runtime, rebuilt from the vendored SDK with one fix.

    v1.1.4 declares the FSR3 upscaler's `rw_luma_history` storage image `rgba8`
    in its GLSL callback while the C++ creates the resource as
    R16G16B16A16_FLOAT, so every Vulkan FSR dispatch trips a validation-layer
    format mismatch and the reads and stores are undefined
    (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/161,
    open). Declaring `rgba16f` matches the view, which is what
    `patches/fsr3upscaler_luma_history_rgba16f.patch` records and what the
    regex below applies. Only the shader changes, so the rebuilt DLL is
    ABI-identical to the stock one.

    Nothing upstream ships this, and nothing ever will: SDK v2 dropped Vulkan
    (its readme lists "Vulkan is currently not supported in SDK" and it ships
    dx12 binaries alone), and it deletes the GLSL callbacks this patches. That
    makes v1.1.4 the last release with a Vulkan backend rather than a pin left
    to go stale: raising `FidelityFx.release` past it removes Vulkan FSR from
    the engine, and this refuses to build rather than following it there.
    """

    name = "fidelityfx-vk"
    # Not `FidelityFx.release` but the last one with a Vulkan backend, which is
    # a different fact that happens to agree today. `build_payload` refuses
    # once they part.
    release = "1.1.4"
    summary = "AMD FidelityFX Vulkan runtime, rebuilt with the FSR3 luma_history fix"
    payload_parts = ("bin", "amd_fidelityfx_vk.dll")
    # Built out of the SDK `fetch` already unpacks, so there is one download.
    source = FidelityFx
    url = None

    shader = Path("sdk/include/FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h")
    declared = re.compile(
        r"(binding = FSR3UPSCALER_BIND_UAV_LUMA_HISTORY, )rgba8(\) uniform image2D\s+rw_luma_history)"
    )
    fixed = re.compile(r"rgba16f\) uniform image2D\s+rw_luma_history")

    @classmethod
    def build_payload(cls, slug, args):
        if cls.source.release != cls.release:
            sys.exit(
                f"{cls.source.name} is pinned to {cls.source.release}, and "
                f"{cls.release} is the last release to build a Vulkan runtime from"
            )
        sdk = install_dir(cls.source, slug)
        shader = sdk / cls.shader
        if not shader.is_file():
            sys.exit(
                f"{cls.source.name} is not vendored at {sdk.name}; "
                f"run `vendor.py fetch {cls.source.name}` first"
            )
        require("cmake")
        if not os.environ.get("VULKAN_SDK"):
            print("  warning: VULKAN_SDK is unset, so the shader build may find no glslc")

        cls.fix_shader(shader)
        # ffx-api pulls in the `sdk` subproject, which is what recompiles the
        # shader permutations the fix reaches.
        api = sdk / "ffx-api"
        tree = api / "build-vk"
        cmake("-S", api, "-B", tree, "-G", args.generator, "-A", "x64",
              "-DFFX_API_BACKEND=VK_X64")
        cmake("--build", tree, "--config", "Release", "--parallel")

        built = api / "bin" / "amd_fidelityfx_vk.dll"  # Release carries no postfix
        if not built.is_file():
            sys.exit(f"the build left no {built.name} at {built}")
        install(built, install_dir(cls, slug) / cls.payload())

    @classmethod
    def fix_shader(cls, shader):
        """Declare `rw_luma_history` as its view's format, if it is not already."""
        source = shader.read_text()
        if cls.fixed.search(source):
            print("  shader already declares rgba16f")
            return
        patched, count = cls.declared.subn(r"\1rgba16f\2", source)
        if count != 1:
            sys.exit(
                f"{shader} holds no rgba8 rw_luma_history declaration to fix; "
                f"the SDK layout differs from v{cls.release}"
            )
        shader.write_text(patched)
        print("  patched rw_luma_history: rgba8 -> rgba16f")


class Xess(WindowsSdk):
    name = "xess"
    release = "3.0.1"
    summary = "Intel XeSS"
    payload_parts = ("bin", "libxess.dll")

    @classmethod
    def url(cls, _slug):
        return (f"https://github.com/intel/xess/releases/download"
                f"/v{cls.release}/XeSS_SDK_{cls.release}.zip")


class Streamline(WindowsSdk):
    name = "streamline"
    release = "2.11.1"
    summary = "NVIDIA Streamline, for DLSS"
    payload_parts = ("bin", "x64", "nvngx_dlss.dll")

    @classmethod
    def url(cls, _slug):
        return ("https://github.com/NVIDIA-RTX/Streamline/releases/download"
                f"/v{cls.release}/streamline-sdk-v{cls.release}.zip")


COMPONENTS = {
    c.name: c
    for c in [Slang, Agility, FidelityFx, FidelityFxVulkan, Xess, Streamline]
}


def run_quietly(cmd):
    """`cmd`'s output, or None if it could not be run or failed."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout + done.stderr if done.returncode == 0 else None


def described(component, payload):
    """What to call the vendored copy: its own reported release, or the pin."""
    if not component.reports_version:
        return component.release
    return component.version(payload) or "version unreadable"


def first_line(text):
    if text is None:
        return None
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


def host_slug(component, required=True):
    """The slug this host fetches, or None where the component has no build.

    A bare `fetch` asks for every component, and most hosts can use only some,
    so an absent slug is a skip rather than a failure. Naming the component
    explicitly makes it one, since the caller asked for that one.
    """
    key = (platform.system(), platform.machine())
    slug = component.slugs.get(key)
    if slug is None and required:
        sys.exit(f"{component.name} publishes no release for {key[0]}/{key[1]}")
    return slug


def install_dir(component, slug, release=None):
    return VENDOR / f"{component.name}-{release or component.release}-{slug}"


def digest_error(pins, slug, digest):
    """Why `digest` is not the archive `slug` was pinned to, or None if it is."""
    if pins is None:
        return None
    expected = pins.get(slug)
    if expected is None:
        return f"no pinned sha256 for {slug}; add one before fetching it"
    if digest != expected:
        return f"sha256 {digest} does not match the pinned {expected}"
    return None


def verify(component, slug, archive, url):
    """Refuse an archive that is not what the pin names, before it is unpacked."""
    digest = hashlib.sha256(archive).hexdigest()
    reason = digest_error(component.sha256, slug, digest)
    if reason is not None:
        sys.exit(f"{url}: {reason}")
    state = "unpinned" if component.sha256 is None else "pinned"
    print(f"  {len(archive) / 1e6:.1f} MB, sha256 {digest} ({state})")


def fetch(component, args):
    if component.url is None:
        if args.named:
            sys.exit(f"{component.name} is built, not downloaded -- run `vendor.py build {component.name}`")
        return 0
    slug = host_slug(component, required=args.named)
    if slug is None:
        print(f"{component.name}: no release for this host, skipped")
        return 0
    target = install_dir(component, slug)
    payload = target / component.payload()

    if payload.is_file() and not args.force:
        print(f"{component.name}: already vendored, {target.name} ({described(component, payload)})")
        return 0

    url = component.url(slug)
    print(f"{component.name}: downloading {url}")
    try:
        with urllib.request.urlopen(url) as response:
            archive = response.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"{url}: HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"{url}: {e.reason}")
    verify(component, slug, archive, url)

    version = unpack(component, archive, url, target)
    touch_watchers(component)
    print(f"{component.name}: vendored {target.relative_to(ROOT)} ({version})")
    return 0


def build(component, args):
    """Produce a component the build resolves but no upstream publishes."""
    if component.build_payload is None:
        if args.named:
            sys.exit(f"{component.name} is downloaded, not built -- run `vendor.py fetch {component.name}`")
        return 0
    slug = host_slug(component, required=args.named)
    if slug is None:
        print(f"{component.name}: no build for this host, skipped")
        return 0

    target = install_dir(component, slug)
    if (target / component.payload()).is_file() and not args.force:
        print(f"{component.name}: already built, {target.name}")
        return 0

    print(f"{component.name}: building from {install_dir(component.source, slug).name}")
    component.build_payload(slug, args)
    touch_watchers(component)
    print(f"{component.name}: built {target.relative_to(ROOT)}")
    return 0


def require(tool):
    if shutil.which(tool) is None:
        sys.exit(f"{tool} is not on PATH")


def cmake(*argv):
    done = subprocess.run(["cmake", *(str(a) for a in argv)])
    if done.returncode != 0:
        sys.exit(f"cmake exited {done.returncode}")


def install(built, payload):
    """Put `built` at `payload`, which only exists once the copy is whole."""
    payload.parent.mkdir(parents=True, exist_ok=True)
    incoming = payload.with_suffix(payload.suffix + ".incoming")
    shutil.copy2(built, incoming)
    incoming.replace(payload)


def unpack(component, archive, url, target):
    """Extract `archive` into `target`, replacing it only once it checks out."""
    staging = VENDOR / f".{target.name}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        extract(archive, url, staging)
        # A release unpacks its payload either at the archive root or under one
        # directory named for the release.
        roots = [staging] + [p for p in staging.iterdir() if p.is_dir()]
        source = next((r for r in roots if (r / component.payload()).is_file()), None)
        if source is None:
            sys.exit(f"{url}: contains no {component.payload().as_posix()}")

        # The pin is the point, so a component that can state its release and
        # states the wrong one is discarded here rather than left for the build
        # to resolve. One that cannot has its payload as the only evidence,
        # which the search above already found.
        version = described(component, source / component.payload())
        if component.reports_version and not version.startswith(component.release):
            sys.exit(f"{url}: reports {version!r}, expected {component.release}")

        shutil.rmtree(target, ignore_errors=True)
        source.replace(target)
        return version
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract(archive, url, into):
    # By content, not by name: a NuGet package is a zip called `.nupkg`, and a
    # release asset is free to be named anything at all.
    if archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            z.extractall(into)
        # Zip carries no executable bit.
        for path in into.rglob("*"):
            if path.is_file() and path.suffix in ("", ".exe"):
                path.chmod(path.stat().st_mode | 0o111)
    elif archive[:2] == b"\x1f\x8b":
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as t:
            t.extractall(into, filter="data")
    else:
        sys.exit(f"{url}: not a zip or a gzip archive")


def touch_watchers(component):
    """Make the build scripts that resolve this component re-run.

    They watch `vendor/`, but only once it exists: an absent rerun path reruns a
    build script on every build, so a `vendor/` created after one last ran is
    picked up here instead.
    """
    for name in component.watchers:
        script = ROOT / name
        if script.is_file():
            os.utime(script, None)


def status(component, args):
    slug = host_slug(component, required=args.named)
    print(f"{component.name}: {component.summary}")
    if slug is None:
        print(f"  host      no release for {platform.system()}/{platform.machine()}")
        return 0
    print(f"  host      {slug}")
    print(f"  pinned    {component.release}")
    print(f"  digest    {(component.sha256 or {}).get(slug) or 'unpinned'}")

    vendored = sorted(
        p
        for p in (VENDOR.glob(f"{component.name}-*-{slug}") if VENDOR.is_dir() else [])
        if (p / component.payload()).is_file()
    )
    if vendored:
        for path in vendored:
            mark = "*" if path == install_dir(component, slug) else " "
            print(f"  vendored {mark}{path.name}  {described(component, path / component.payload())}")
    else:
        verb = "build" if component.url is None else "fetch"
        print(f"  vendored  none -- run `scripts/vendor.py {verb} {component.name}`")

    for label, payload in component.alternatives():
        found = component.version(payload) if payload.is_file() else None
        print(f"  {label:<9} {payload}  {found or 'unusable'}")

    return 0


def selftest():
    """Exercise the digest gate. Downloads nothing."""
    pins = {"linux-x86_64": "a" * 64}
    cases = [
        (None, "linux-x86_64", "b" * 64, None),
        (pins, "linux-x86_64", "a" * 64, None),
        (pins, "linux-x86_64", "b" * 64, "does not match the pinned"),
        (pins, "macos-aarch64", "a" * 64, "no pinned sha256"),
    ]
    for component_pins, slug, digest, want in cases:
        got = digest_error(component_pins, slug, digest)
        ok = got is None if want is None else got is not None and want in got
        if not ok:
            sys.exit(f"selftest: {slug} {digest[:8]} gave {got!r}, wanted {want!r}")

    missing = [
        f"{c.name}/{slug}"
        for c in COMPONENTS.values()
        if c.sha256 is not None
        for slug in c.slugs.values()
        if slug not in c.sha256
    ]
    if missing:
        sys.exit(f"selftest: pinned component(s) missing a digest: {', '.join(missing)}")
    print(f"vendor selftest: digest gate OK, {len(cases)} cases")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    def add(name, help_text, run):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("components", nargs="*", metavar="component", choices=None)
        p.set_defaults(run=run)
        return p

    add("fetch", "download the pinned release", fetch).add_argument(
        "--force", action="store_true", help="re-download over an existing copy"
    )
    build_cmd = add("build", "build what no upstream publishes", build)
    build_cmd.add_argument("--force", action="store_true", help="rebuild over an existing copy")
    build_cmd.add_argument(
        "--generator", default="Visual Studio 18 2026", help="CMake generator to build with"
    )
    add("status", "report what the build would pick", status)
    sub.add_parser("selftest", help="check the digest gate, downloading nothing")

    args = parser.parse_args(argv)
    if args.command == "selftest":
        return selftest()
    if args.command is None:
        parser.print_help()
        print("\ncomponents:")
        for component in COMPONENTS.values():
            print(f"  {component.name:<14} {component.summary}")
        return 2

    unknown = [c for c in args.components if c not in COMPONENTS]
    if unknown:
        sys.exit(f"unknown component(s): {', '.join(unknown)}")
    args.named = bool(args.components)
    selected = args.components or list(COMPONENTS)

    for name in selected:
        code = args.run(COMPONENTS[name], args)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
