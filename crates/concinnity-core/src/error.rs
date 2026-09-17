//! The engine's runtime error types.
//!
//! Three seams fail at runtime and each names what it can fail with:
//! [`AssetError`] when a baked blob record does not reconstruct its asset,
//! [`PayloadError`] when compiled payload bytes cannot be read, and
//! [`WorldError`] when a world cannot be built, started, or added to. The
//! first two fold into the third, so a caller that only drives a world handles
//! one type and still reaches the cause through
//! [`source`](core::error::Error::source).
//!
//! A C host gets a flat code instead, built from these at the FFI boundary;
//! see `concinnity-ffi`.

use alloc::boxed::Box;
use alloc::string::String;
use thiserror::Error;

use crate::blob::FrameError;
use crate::ecs::asset_id::AssetIdsExhausted;

/// Why a baked blob record could not be turned back into its asset.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AssetError {
    /// The record's discriminant names no component type this build registers,
    /// which means the blob was written by a different version of the engine.
    #[error("no component type has discriminant {discriminant}")]
    UnknownComponent {
        /// The discriminant the record carried.
        discriminant: u8,
    },

    /// A resource record's kind tag names no resource kind this build has, for
    /// the same reason.
    #[error("no resource kind has tag {tag}")]
    UnknownResourceKind {
        /// The tag the record carried.
        tag: u8,
    },

    /// A record was found for a component that is never stored in a blob, so
    /// it has no baked form to read.
    #[error("{asset} is a runtime-only component and has no baked form")]
    NotStored {
        /// The component the record claimed to be.
        asset: &'static str,
    },

    /// The record's bytes did not decode as the component they were written
    /// for.
    #[error("decoding the baked {asset} failed")]
    Decode {
        /// The component being decoded.
        asset: &'static str,
        /// What the decode reported.
        #[source]
        source: FrameError,
    },
}

/// Why compiled payload bytes could not be read.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PayloadError {
    /// The world has no compiled payloads at all, so no locator resolves.
    #[error("this world has no compiled payloads")]
    NoPayloads,

    /// The locator addresses a blob outside the set the world was loaded from.
    #[error("blob {index} is not one of this world's {count}")]
    NoSuchBlob {
        /// The blob the locator named.
        index: u32,
        /// How many blobs the world has.
        count: u32,
    },

    /// The blob's payload section was released, which is deliberate: a
    /// released section is one every system that needed it already read.
    #[error("blob {index}'s payload section has been released")]
    Released {
        /// The blob whose section is gone.
        index: u32,
    },

    /// The locator's range does not fit the blob's payload section.
    #[error("payload [{offset}, +{len}) does not fit blob {index}'s {section_len}-byte section")]
    OutOfBounds {
        /// The blob the locator named.
        index: u32,
        /// The locator's offset within that blob's payload section.
        offset: u64,
        /// The locator's length.
        len: u64,
        /// How long the section actually is.
        section_len: usize,
    },

    /// Loading the blob's payload section on first access failed. Boxed
    /// because the store is a trait a `no_std` client implements too, so the
    /// concrete cause belongs to whichever implementation reads the bytes.
    #[error("loading blob {index}'s payload section failed")]
    Load {
        /// The blob that would not load.
        index: u32,
        /// What the load reported.
        #[source]
        source: Box<dyn core::error::Error + Send + Sync>,
    },
}

/// Why a world could not be built, started, or added to.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorldError {
    /// Two systems claim one name, which both the ordering edges and the
    /// schedule's lookups key on.
    #[error("a system named `{0}` is already in the schedule")]
    DuplicateSystemName(&'static str),

    /// The loop was started twice. A second start would re-run every system's
    /// `init` over a world already running.
    #[error("this world has already started")]
    AlreadyStarted,

    /// A world declares at most one `EngineDefaults`: which of several applied
    /// would be arbitrary.
    #[error("a world declares at most one EngineDefaults, but this one declares {count}")]
    RepeatedEngineDefaults {
        /// How many were declared.
        count: usize,
    },

    /// Baking something the engine injects into the world failed.
    #[error("baking the engine's {what} failed: {message}")]
    Bake {
        /// What was being baked.
        what: &'static str,
        /// What the bake reported.
        message: String,
    },

    /// The world has named every asset it can.
    #[error(transparent)]
    AssetIds(#[from] AssetIdsExhausted),

    /// A baked record did not reconstruct.
    #[error(transparent)]
    Asset(#[from] AssetError),

    /// A compiled payload did not read.
    #[error(transparent)]
    Payload(#[from] PayloadError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use core::error::Error as _;

    // The cause is what the flat enum used to drop, so every wrapper has to
    // still reach it.
    #[test]
    fn a_decode_failure_keeps_the_frame_error_as_its_source() {
        let source = crate::blob::decode_exact::<String>(&[0xff]).unwrap_err();
        let error = AssetError::Decode {
            asset: "Prop",
            source: source.clone(),
        };

        assert!(error.to_string().contains("Prop"), "{error}");
        let cause = error.source().expect("the frame error is the source");
        assert_eq!(cause.to_string(), source.to_string());
    }

    // A world failure wraps the asset failure transparently: the wrapper adds
    // no link of its own, so the message is the inner one and the chain
    // reaches the frame error in a single hop.
    #[test]
    fn a_world_failure_chains_through_to_the_decode_cause() {
        let source = crate::blob::decode_exact::<u8>(&[1, 2]).unwrap_err();
        let error = WorldError::from(AssetError::Decode {
            asset: "Prop",
            source,
        });

        assert!(error.to_string().contains("Prop"), "{error}");
        let frame = error.source().expect("the frame error");
        assert!(frame.to_string().contains("trailing"), "{frame}");
    }

    // The payload store is a trait a no_std client implements, so its load
    // failure carries whatever that implementation reports.
    #[test]
    fn a_payload_load_failure_keeps_the_cause_the_store_reported() {
        let error = PayloadError::Load {
            index: 3,
            source: Box::new(AssetError::UnknownComponent { discriminant: 9 }),
        };

        assert!(error.to_string().contains('3'), "{error}");
        let cause = error.source().expect("the boxed cause");
        assert!(cause.to_string().contains('9'), "{cause}");
    }

    // Minting is its own narrow seam, and the world failure it folds into
    // reports it verbatim.
    #[test]
    fn an_exhausted_minter_reads_the_same_either_way() {
        let error = WorldError::from(AssetIdsExhausted);
        assert_eq!(error.to_string(), AssetIdsExhausted.to_string());
    }
}
