/// Edge of the square lookup table. 64 matches the resolution the parameterisation
/// was chosen for: finer buys little once `sqrt(1 - cos)` has spread the grazing
/// angles out, and the fit cost grows with the square.
///
/// In its own file because `build.rs` `include!`s it alongside the fitter, while
/// the lib compiles only this half.
pub const LTC_LUT_SIZE: usize = 64;
