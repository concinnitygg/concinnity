// The argv mirrors of engine types. The value-enum derives live here so no
// layer below the command line carries a clap dependency.
use crate::export::BundleFormat;

// The argv face of the export `BundleFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum BundleFormatArg {
    Zip,
    Dir,
}

impl From<BundleFormatArg> for BundleFormat {
    fn from(f: BundleFormatArg) -> Self {
        match f {
            BundleFormatArg::Zip => Self::Zip,
            BundleFormatArg::Dir => Self::Dir,
        }
    }
}
