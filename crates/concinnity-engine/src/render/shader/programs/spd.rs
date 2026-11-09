//! The pyramid is two dispatches: one of the phase-1 kernels, chosen by the
//! main pass's sample count, then the tail. The sample count picks a program
//! here rather than a `USE_MSAA` define because the two read different depth
//! resource types, so each is its own entry point.

use super::ShaderProgram;

/// `hiz_spd_msaa` from `hiz_build.hlsl`.
pub static HIZ_SPD_MSAA: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_spd_msaa",
    label: "hiz_spd_msaa.hlsl",
    gates: &["HIZ_SPD_MSAA"],
    msaa: false,
};
/// `hiz_spd_single` from `hiz_build.hlsl`.
pub static HIZ_SPD_SINGLE: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_spd_single",
    label: "hiz_spd_single.hlsl",
    gates: &["HIZ_SPD_SINGLE"],
    msaa: false,
};
/// `hiz_spd_tail` from `hiz_build.hlsl`.
pub static HIZ_SPD_TAIL: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_spd_tail",
    label: "hiz_spd_tail.hlsl",
    gates: &["HIZ_SPD_TAIL"],
    msaa: false,
};

/// Every program in this module.
pub static ALL: &[&ShaderProgram] = &[&HIZ_SPD_MSAA, &HIZ_SPD_SINGLE, &HIZ_SPD_TAIL];

#[cfg(test)]
mod tests {
    #[test]
    fn every_declared_program_is_in_the_table() {
        super::super::declared::assert_table_is_complete(include_str!("spd.rs"));
    }
}
