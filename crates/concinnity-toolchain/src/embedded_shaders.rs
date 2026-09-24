// What a build without a usable shader compiler means for the engine's shaders.
//
// Both precompile legs -- the Metal libraries and the SPIR-V/DXIL artifacts --
// would generate a lookup answering `None` for every name, and the renderer
// would compile at device init instead, which needs the compiler on whatever
// host runs the binary. Nothing ships that way, so the build stops here
// instead, where the message can still name what to install.

/// Ends the build unless dxc resolves.
pub(crate) fn require_dxc() {
    if let Some(reason) = concinnity_shader::unavailable_reason() {
        panic!("{}", absent_message(reason));
    }
}

/// Reports a host that cannot embed the shaders for a reason no install of dxc
/// would fix, the Metal toolchain being the one that does this.
///
/// A build for the machine building it ends: that binary is one someone runs
/// here, and it would reach for a compiler at device init. A cross build warns
/// and carries on, because the toolchain it would need belongs to the machine
/// that runs the output, not to this one -- which is also what a docs build
/// does, and it never runs anything.
pub(crate) fn cannot_embed(reason: &str) {
    let message = unembeddable_message(reason);
    if building_for_this_host() {
        panic!("{message}");
    }
    println!("cargo:warning={message}");
}

// Whether the binary this build produces runs on the machine producing it.
// Cargo names the target for every build script; the host is whatever this one
// was compiled for.
fn building_for_this_host() -> bool {
    is_native(
        std::env::consts::OS,
        std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref(),
    )
}

fn is_native(host_os: &str, target_os: Option<&str>) -> bool {
    target_os.is_none_or(|target| target == host_os)
}

// The resolver's reason already names what to install, so this only adds what
// the compiler's absence costs the build.
fn absent_message(reason: &str) -> String {
    format!(
        "{reason}\n\nThe engine's shaders are compiled into the binary at build \
         time, and this build would carry none: every backend would compile them \
         at device init instead, which needs a shader compiler on whatever host \
         runs the binary."
    )
}

// What a host that cannot compile the shaders at all is told. Unlike a missing
// dxc there is nothing to install that would change the answer for a cross
// build, so this names the toolchain and leaves the remedy to the reader.
fn unembeddable_message(reason: &str) -> String {
    format!(
        "{reason}: the engine's shaders cannot be compiled here, so this build \
         would carry none and every backend would compile them at device init \
         instead, which needs a shader compiler on whatever host runs the binary."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The resolver's reason is the half that says what to install, so the
    // wording around it may not drop it.
    #[test]
    fn the_message_keeps_the_resolvers_reason() {
        let reason = "dxc not found: install a release";
        assert!(absent_message(reason).contains(reason));
    }

    // The build wanted the shaders embedded, not merely compilable, so the
    // message says what their absence costs.
    #[test]
    fn the_message_names_what_the_absence_costs() {
        assert!(absent_message("reason").contains("device init"));
    }

    // Cross-compiling is what makes a missing toolchain survivable: the machine
    // that runs the output is not this one, so its compiler is not this one's
    // to have. docs.rs is the case in hand -- a Linux container targeting
    // x86_64-apple-darwin, which no Metal toolchain reaches.
    #[test]
    fn a_target_os_other_than_the_hosts_is_a_cross_build() {
        assert!(is_native("macos", Some("macos")));
        assert!(!is_native("linux", Some("macos")));
        assert!(!is_native("macos", Some("windows")));
    }

    // Cargo names the target for every build script, so an absent one is a
    // caller outside a build, not evidence of a cross build.
    #[test]
    fn an_unnamed_target_counts_as_this_host() {
        assert!(is_native("macos", None));
    }

    // The caller's reason is the half naming which toolchain was missing.
    #[test]
    fn the_unembeddable_message_keeps_the_callers_reason() {
        let reason = "Metal compiler not found (full Xcode required)";
        let message = unembeddable_message(reason);
        assert!(message.contains(reason));
        assert!(message.contains("device init"));
    }
}
