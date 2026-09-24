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
apply to is skipped: most are downloaded, `fidelityfx-vk` is built, and `dxc`
is downloaded on Windows and Linux and built from the same commit on macOS.

A downloaded archive is checked against a pinned sha256 before it is unpacked,
so a release re-tagged in place fails the fetch rather than reaching a build.

Nothing here runs during `cargo build`: a build script that reached the network
could not build offline, could not build on docs.rs, and would pull a dependency
past Cargo.lock, cargo-vendor and cargo-deny. Vendoring is a setup step, run
once, and the build only ever reads what it left behind.
"""

import argparse
import fnmatch
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"


def is_windows():
    return platform.system() == "Windows"


WINDOWS_X64 = {("Windows", "AMD64"): "windows-x86_64"}


class WindowsSdk:
    """A graphics SDK the DirectX build links or bundles, unpacked as shipped.

    These are Windows-only, so `fetch` on another host skips them rather than
    failing: nothing on a Mac or a Linux box has any use for them. None ships a
    tool that reports its own version, so the payload being where the build
    script looks is the whole check.
    """

    build_payload = None
    slugs = WINDOWS_X64
    # No pinned digest yet, so `fetch` reports what it got and unpacks it.
    sha256 = None
    watchers = ("build.rs", "crates/concinnity-device/build.rs",
                "crates/concinnity-engine/build.rs", "crates/concinnity-dev/build.rs")

    @classmethod
    def payload(cls, _slug):
        return Path(*cls.payload_parts)

    @classmethod
    def keep(cls, _slug):
        return None

    @classmethod
    def report(cls, _exe):
        return None


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
        install(built, install_dir(cls, slug) / cls.payload(slug))

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


class Dxc:
    """The DirectX Shader Compiler every engine shader compiles through.

    One release for every host, since the SPIR-V, DXIL and MSL a build embeds
    and the runtime shader cache's key are all functions of it. Windows and
    Linux take Microsoft's release archive, cut to the compiler, the two
    libraries it loads and the license files. Microsoft publishes no macOS
    build, so a Mac builds the same tagged commit from source: a shallow clone
    of it and three submodules, then `dxc`, `dxcompiler` and `dxildll` in
    Release. That costs a compile measured in minutes (3.5 on a 12-core Apple
    silicon Mac), once per pin; the tree it builds in is deleted afterwards and
    only the installed release stays under `vendor/`.

    Every copy reports the same commit, which is what the shader cache keys on.
    The build counter beside it does not agree even between Microsoft's own
    archives: the Windows one counts the full history, the Linux one a shallow
    clone, as a source build here does too.
    """

    name = "dxc"
    release = "1.9.2607"
    commit = "0d3ee6b551b8fa768fbf825300ebab81047ef6a8"
    summary = "Microsoft's DirectX Shader Compiler, which compiles every engine shader"
    repo = "https://github.com/microsoft/DirectXShaderCompiler"
    slugs = {
        ("Windows", "AMD64"): "windows-x86_64",
        ("Linux", "x86_64"): "linux-x86_64",
        ("Darwin", "arm64"): "macos-aarch64",
    }
    # Upstream's asset names, which follow no scheme from one release to the next.
    assets = {
        "windows-x86_64": "dxc_2026_07_29.zip",
        "linux-x86_64": "linux_dxc_2026_07_29.x86_x64.tar.gz",
    }
    sha256 = {
        "windows-x86_64": "a1dfb116ba3eeae6a1582291b53a8e7bf65ad760676bd3194685c8f7367cd241",
        "linux-x86_64": "55665c87824051ed4774ff3280a79ccbbb7d39243b9736ca5e98222134112d54",
    }
    # Upstream spells one of them LICENCE.
    licenses = "LICEN[CS]E*"
    # The tests' googletest is the fourth, and nothing here builds a test.
    submodules = ("external/SPIRV-Headers", "external/SPIRV-Tools", "external/DirectX-Headers")
    source_licenses = ("LICENSE.TXT", "ThirdPartyNotices.txt")
    watchers = ("crates/concinnity-shader/build.rs",)

    @classmethod
    def url(cls, slug):
        asset = cls.assets.get(slug)
        return asset and f"{cls.repo}/releases/download/v{cls.release}/{asset}"

    @classmethod
    def payload(cls, slug):
        return Path("bin", "x64", "dxc.exe") if slug.startswith("windows") else Path("bin", "dxc")

    @classmethod
    def keep(cls, slug):
        """The compiler, the libraries it loads, and the licenses: the Linux
        archive also carries LLVM's tools and static libraries, 1 GB unpacked."""
        if slug.startswith("windows"):
            return ("bin/x64/dxc.exe", "bin/x64/dxcompiler.dll", "bin/x64/dxil.dll", cls.licenses)
        return ("bin/dxc", "lib/libdxcompiler.so", "lib/libdxil.so", cls.licenses)

    @classmethod
    def report(cls, exe):
        """What a compiler says it is, which proves it loads its library."""
        try:
            done = subprocess.run([str(exe), "--version"], capture_output=True, text=True)
        except OSError as e:
            return f"does not run: {e}"
        if done.returncode != 0:
            return f"does not run: {(done.stderr or done.stdout).strip()}"
        return done.stdout.strip().splitlines()[0]

    @classmethod
    def on_path(cls, slug):
        """The copy a build falls back to when nothing is vendored."""
        found = shutil.which(cls.payload(slug).name)
        return found and Path(found)

    @classmethod
    def build_payload(cls, slug, _args):
        for tool in ("git", "cmake", "ninja"):
            require(tool)
        work = VENDOR / ".build" / f"{cls.name}-{cls.release}"
        src, tree = work / "src", work / "build"
        started = time.monotonic()

        if not (src / ".git").is_dir():
            shutil.rmtree(src, ignore_errors=True)
            git("clone", "--depth", "1", "--branch", f"v{cls.release}", cls.repo, src)
        head = git_output("-C", src, "rev-parse", "HEAD")
        if head != cls.commit:
            sys.exit(f"v{cls.release} resolved to {head}, not the pinned {cls.commit}")
        git("-C", src, "submodule", "update", "--init", "--depth", "1", "--", *cls.submodules)

        # Explicit values before `-C`: the cache script never overrides one.
        cmake("-G", "Ninja", "-S", src, "-B", tree,
              "-DCMAKE_BUILD_TYPE=Release",
              "-DHLSL_INCLUDE_TESTS=OFF", "-DSPIRV_BUILD_TESTS=OFF",
              "-DLLVM_INCLUDE_TESTS=OFF", "-DCLANG_INCLUDE_TESTS=OFF",
              "-C", src / "cmake" / "caches" / "PredefinedParams.cmake")
        cmake("--build", tree, "--target", "dxc", "dxildll")

        target = install_dir(cls, slug)
        staging = VENDOR / f".{target.name}.incoming"
        shutil.rmtree(staging, ignore_errors=True)
        suffix = ".dylib" if slug.startswith("macos") else ".so"
        install(tree / "bin" / "dxc", staging / cls.payload(slug))
        for library in ("libdxcompiler", "libdxil"):
            # The real file under the name the executable loads, not a symlink to it.
            install((tree / "lib" / f"{library}{suffix}").resolve(),
                    staging / "lib" / f"{library}{suffix}")
        for name in cls.source_licenses:
            install(src / name, staging / name)
        shutil.rmtree(target, ignore_errors=True)
        staging.replace(target)

        shutil.rmtree(work)
        if not any(work.parent.iterdir()):
            work.parent.rmdir()
        print(f"  built in {(time.monotonic() - started) / 60:.1f} min")


COMPONENTS = {
    c.name: c
    for c in [Dxc, Agility, FidelityFx, FidelityFxVulkan, Xess, Streamline]
}


def fetched_on(component, slug):
    """Whether `slug` downloads this component rather than building it."""
    return component.url is not None and component.url(slug) is not None


def built_on(component, slug):
    return component.build_payload is not None and not fetched_on(component, slug)


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
    slug = host_slug(component, required=args.named)
    if slug is None:
        print(f"{component.name}: no release for this host, skipped")
        return 0
    if not fetched_on(component, slug):
        if args.named:
            sys.exit(f"{component.name} is built here, not downloaded -- run `vendor.py build {component.name}`")
        print(f"{component.name}: built here, skipped -- `vendor.py build {component.name}` builds it")
        return 0
    target = install_dir(component, slug)
    payload = target / component.payload(slug)

    if payload.is_file() and not args.force:
        print(f"{component.name}: already vendored, {target.name} ({component.release})")
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

    unpack(archive, url, component.payload(slug), target, keeps(component.keep(slug)))
    touch_watchers(component)
    print(f"{component.name}: vendored {target.relative_to(ROOT)} ({component.release})")
    print_report(component, target, slug)
    return 0


def build(component, args):
    """Produce a component the build resolves but no upstream publishes for this host."""
    slug = host_slug(component, required=args.named)
    if slug is None:
        print(f"{component.name}: no build for this host, skipped")
        return 0
    if not built_on(component, slug):
        if args.named:
            sys.exit(f"{component.name} is downloaded here, not built -- run `vendor.py fetch {component.name}`")
        print(f"{component.name}: downloaded here, skipped")
        return 0

    target = install_dir(component, slug)
    if (target / component.payload(slug)).is_file() and not args.force:
        print(f"{component.name}: already built, {target.name}")
        return 0

    print(f"{component.name}: building {component.release}")
    component.build_payload(slug, args)
    touch_watchers(component)
    print(f"{component.name}: built {target.relative_to(ROOT)}")
    print_report(component, target, slug)
    return 0


def print_report(component, root, slug):
    line = component.report(root / component.payload(slug))
    if line is not None:
        print(f"  reports   {line}")


def require(tool):
    if shutil.which(tool) is None:
        sys.exit(f"{tool} is not on PATH")


def cmake(*argv):
    done = subprocess.run(["cmake", *(str(a) for a in argv)])
    if done.returncode != 0:
        sys.exit(f"cmake exited {done.returncode}")


def git(*argv):
    done = subprocess.run(["git", *(str(a) for a in argv)])
    if done.returncode != 0:
        sys.exit(f"git {argv[0]} exited {done.returncode}")


def git_output(*argv):
    done = subprocess.run(["git", *(str(a) for a in argv)], capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"git {' '.join(str(a) for a in argv)}: {done.stderr.strip()}")
    return done.stdout.strip()


def install(built, payload):
    """Put `built` at `payload`, which only exists once the copy is whole."""
    payload.parent.mkdir(parents=True, exist_ok=True)
    incoming = payload.with_suffix(payload.suffix + ".incoming")
    shutil.copy2(built, incoming)
    incoming.replace(payload)


def keeps(patterns):
    """Which archive members to unpack: all of them, or those `patterns` name.

    A pattern is matched against the member's path both as it stands and below
    its first directory, since a release unpacks either at the archive root or
    under one directory named for the release.
    """
    if patterns is None:
        return lambda _name: True

    def wanted(name):
        parts = name.replace("\\", "/").strip("/").split("/")
        paths = ["/".join(parts), "/".join(parts[1:])]
        return any(fnmatch.fnmatchcase(path, pattern) for path in paths for pattern in patterns)

    return wanted


def unpack(archive, url, payload, target, wanted):
    """Extract `archive` into `target`, replacing it only once it checks out."""
    staging = VENDOR / f".{target.name}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        extract(archive, url, staging, wanted)
        # A release unpacks its payload either at the archive root or under one
        # directory named for the release.
        roots = [staging] + [p for p in staging.iterdir() if p.is_dir()]
        source = next((r for r in roots if (r / payload).is_file()), None)
        if source is None:
            sys.exit(f"{url}: contains no {payload.as_posix()}")

        shutil.rmtree(target, ignore_errors=True)
        source.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract(archive, url, into, wanted):
    # By content, not by name: a NuGet package is a zip called `.nupkg`, and a
    # release asset is free to be named anything at all.
    if archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            z.extractall(into, members=[m for m in z.infolist() if wanted(m.filename)])
        # Zip carries no executable bit.
        for path in into.rglob("*"):
            if path.is_file() and path.suffix in ("", ".exe"):
                path.chmod(path.stat().st_mode | 0o111)
    elif archive[:2] == b"\x1f\x8b":
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as t:
            members = [m for m in t.getmembers() if wanted(m.name)]
            t.extractall(into, members=members, filter="data")
    else:
        sys.exit(f"{url}: not a zip or a gzip archive")


def touch_watchers(component):
    """Make the build scripts that resolve this component re-run.

    They watch only the files they resolved, so touching them is what picks up
    a newly installed component.
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
    if built_on(component, slug):
        print("  digest    built here")
    else:
        print(f"  digest    {(component.sha256 or {}).get(slug) or 'unpinned'}")

    vendored = sorted(
        p
        for p in (VENDOR.glob(f"{component.name}-*-{slug}") if VENDOR.is_dir() else [])
        if (p / component.payload(slug)).is_file()
    )
    if vendored:
        for path in vendored:
            mark = "*" if path == install_dir(component, slug) else " "
            print(f"  vendored {mark}{path.name}  {component.release}")
            print_report(component, path, slug)
    else:
        verb = "build" if built_on(component, slug) else "fetch"
        print(f"  vendored  none -- run `scripts/vendor.py {verb} {component.name}`")

    fallback = getattr(component, "on_path", lambda _slug: None)(slug)
    if fallback is not None:
        print(f"  on PATH   {fallback}")
        line = component.report(fallback)
        if line is not None:
            print(f"  reports   {line}")

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
        if fetched_on(c, slug) and slug not in c.sha256
    ]
    if missing:
        sys.exit(f"selftest: pinned component(s) missing a digest: {', '.join(missing)}")

    unsourced = [
        f"{c.name}/{slug}"
        for c in COMPONENTS.values()
        for slug in c.slugs.values()
        if not fetched_on(c, slug) and not built_on(c, slug)
    ]
    if unsourced:
        sys.exit(f"selftest: neither downloaded nor built: {', '.join(unsourced)}")

    wanted = keeps(Dxc.keep("linux-x86_64"))
    members = [
        ("linux_dxc_2026_07_29.x86_x64/bin/dxc", True),
        ("linux_dxc_2026_07_29.x86_x64/lib/libdxcompiler.so", True),
        ("linux_dxc_2026_07_29.x86_x64/LICENCE-MIT.txt", True),
        ("linux_dxc_2026_07_29.x86_x64/LICENSE-LLVM.txt", True),
        ("linux_dxc_2026_07_29.x86_x64/lib/libLLVMSupport.a", False),
        ("linux_dxc_2026_07_29.x86_x64/bin/opt", False),
        ("bin/dxc", True),
    ]
    wanted_windows = keeps(Dxc.keep("windows-x86_64"))
    members_windows = [
        ("bin\\x64\\dxc.exe", True),
        ("bin\\x64\\dxil.dll", True),
        ("bin\\arm64\\dxc.exe", False),
        ("inc\\hlsl\\LICENCE.txt", False),
        ("LICENSE-MS.txt", True),
    ]
    for keep, name, want in [(wanted, *m) for m in members] + [(wanted_windows, *m) for m in members_windows]:
        if keep(name) != want:
            sys.exit(f"selftest: unpacking {name} gave {not want}, wanted {want}")
    if not keeps(None)("anything/at/all"):
        sys.exit("selftest: a component naming no members must unpack all of them")

    kinds = [
        (Dxc, "windows-x86_64", "fetch"),
        (Dxc, "linux-x86_64", "fetch"),
        (Dxc, "macos-aarch64", "build"),
        (Agility, "windows-x86_64", "fetch"),
        (FidelityFxVulkan, "windows-x86_64", "build"),
    ]
    for component, slug, want in kinds:
        got = "fetch" if fetched_on(component, slug) else "build" if built_on(component, slug) else None
        if got != want:
            sys.exit(f"selftest: {component.name}/{slug} would {got}, wanted {want}")

    checked = len(cases) + len(members) + len(members_windows) + len(kinds)
    print(f"vendor selftest: digest gate, unpack filter and sourcing OK, {checked} cases")
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
