//! Background clip decoding. One named worker thread receives encoded clip
//! bytes, decodes them with kira, and sends the result back; the output drains
//! completions once per tick. Decode failures, and a decode that panics, travel
//! in the result, so a clip is never left waiting.

use kira::sound::static_sound::StaticSoundData;
use std::io::Cursor;

use concinnity_host::thread::worker::Worker;

pub(crate) struct DecodeResult {
    pub key: u64,
    pub decoded: Result<StaticSoundData, String>,
}

// One clip's encoded bytes, tagged with the caller's key.
struct DecodeRequest {
    key: u64,
    bytes: Vec<u8>,
}

pub(crate) struct DecodeWorker {
    worker: Worker<DecodeRequest, DecodeResult, u64>,
}

impl DecodeWorker {
    // The decode thread, or the error the OS refused it with.
    pub(crate) fn spawn() -> std::io::Result<Self> {
        let worker = Worker::spawn_keyed(
            "audio-decode",
            1,
            |request: &DecodeRequest| request.key,
            |request: DecodeRequest| DecodeResult {
                key: request.key,
                decoded: decode(request.bytes),
            },
        )?;
        Ok(Self { worker })
    }

    // Hand encoded bytes to the worker. False if the worker is gone.
    pub(crate) fn send(&mut self, key: u64, bytes: Vec<u8>) -> bool {
        self.worker.send(DecodeRequest { key, bytes }).is_ok()
    }

    // One decode finished since the last call, if any. Never blocks.
    pub(crate) fn try_recv(&self) -> Option<DecodeResult> {
        self.worker.try_recv().map(|answer| {
            answer.unwrap_or_else(|key| DecodeResult {
                key,
                decoded: Err("the decoder panicked".to_string()),
            })
        })
    }
}

fn decode(bytes: Vec<u8>) -> Result<StaticSoundData, String> {
    StaticSoundData::from_cursor(Cursor::new(bytes)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_testing::fixtures::wav::pcm16_mono;

    // Decoding runs on the worker thread, so wait for its result.
    fn drain_one(worker: &DecodeWorker) -> DecodeResult {
        concinnity_testing::eventually("a decode result", || worker.try_recv())
    }

    #[test]
    fn worker_decodes_valid_bytes_off_thread() {
        let mut worker = DecodeWorker::spawn().expect("thread");
        assert!(worker.send(11, pcm16_mono(44_100, &[0; 64])));
        let result = drain_one(&worker);
        assert_eq!(result.key, 11);
        let data = result.decoded.expect("valid WAV decodes");
        assert!(data.num_frames() > 0);
    }

    #[test]
    fn worker_reports_undecodable_bytes_as_errors() {
        let mut worker = DecodeWorker::spawn().expect("thread");
        assert!(worker.send(3, b"not an audio file".to_vec()));
        let result = drain_one(&worker);
        assert_eq!(result.key, 3);
        assert!(result.decoded.is_err());
    }

    #[test]
    fn drop_joins_the_worker_cleanly() {
        let mut worker = DecodeWorker::spawn().expect("thread");
        worker.send(1, pcm16_mono(44_100, &[0; 16]));
        // Dropping with work possibly still queued must not hang or panic.
        drop(worker);
    }
}
