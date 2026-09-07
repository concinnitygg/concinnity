// What a build without a usable slangc means for the engine's shaders.
//
// Both precompile legs -- the Metal libraries and the SPIR-V/DXIL artifacts --
// would generate a lookup answering `None` for every name, and the renderer
// would compile at device init instead, which needs slangc on whatever host
// runs the binary. Nothing ships that way, so the build stops here instead,
// where the message can still name what to install.

use concinnity_slang as slang;

/// Ends the build unless a usable slangc resolves, which the version floor
/// makes a compiler new enough as well as a compiler at all.
pub(crate) fn require_slangc() {
    if slang::slangc_path().is_some() {
        return;
    }
    let reason = slang::unavailable_reason().unwrap_or("slangc not found");
    panic!("{}", absent_message(reason));
}

// The resolver's reason already names the compiler it rejected and what to
// install, so this only adds what its absence costs the build.
fn absent_message(reason: &str) -> String {
    format!(
        "{reason}\n\nThe engine's shaders are compiled into the binary at build \
         time, and this build would carry none: every backend would compile them \
         at device init instead, which needs slangc on whatever host runs the \
         binary. In a checkout, `scripts/vendor.py fetch slang` installs the \
         pinned release."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The resolver's reason is the half that says which compiler was rejected
    // and what to install, so the wording around it may not drop it.
    #[test]
    fn the_message_keeps_the_resolvers_reason() {
        let reason = "slangc 2026.13.1 (on PATH) is older than 2026.16";
        assert!(absent_message(reason).contains(reason));
    }

    // A reader who has slangc and still lands here needs to know the build
    // wanted it embedded, not merely present.
    #[test]
    fn the_message_names_what_the_absence_costs() {
        let message = absent_message("reason");
        assert!(message.contains("device init"));
        assert!(message.contains("scripts/vendor.py fetch slang"));
    }
}
