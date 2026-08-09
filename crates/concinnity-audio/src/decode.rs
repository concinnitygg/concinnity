// concinnity-audio/src/decode.rs
//
// Background clip decoding. One named worker thread receives encoded clip
// bytes over a channel, decodes them with kira, and sends the result back;
// the engine drains completions once per tick. Decode failures travel in the
// result message, so the thread itself never fails. Shutdown lives entirely
// in Drop: dropping the request sender ends the worker's recv loop, then the
// thread is joined, so a world rebuild never leaks the thread.

use std::io::Cursor;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use kira::sound::static_sound::StaticSoundData;

pub(crate) struct DecodeResult {
    pub key: u64,
    pub decoded: Result<StaticSoundData, String>,
}

pub(crate) struct DecodeWorker {
    // `None` only during Drop, which needs to hang up before joining.
    request_tx: Option<Sender<(u64, Vec<u8>)>>,
    results: Receiver<DecodeResult>,
    thread: Option<JoinHandle<()>>,
}

impl DecodeWorker {
    pub(crate) fn spawn() -> Self {
        let (request_tx, requests) = channel::<(u64, Vec<u8>)>();
        let (result_tx, results) = channel();
        let thread = std::thread::Builder::new()
            .name("audio-decode".into())
            .spawn(move || {
                while let Ok((key, bytes)) = requests.recv() {
                    let decoded = decode(bytes);
                    if result_tx.send(DecodeResult { key, decoded }).is_err() {
                        break;
                    }
                }
            })
            .expect("audio decode worker spawn failed");
        Self {
            request_tx: Some(request_tx),
            results,
            thread: Some(thread),
        }
    }

    // Hand encoded bytes to the worker. False if the worker is gone.
    pub(crate) fn send(&self, key: u64, bytes: Vec<u8>) -> bool {
        self.request_tx
            .as_ref()
            .is_some_and(|tx| tx.send((key, bytes)).is_ok())
    }

    // Every decode finished since the last call. Never blocks.
    pub(crate) fn drain(&self) -> Vec<DecodeResult> {
        let mut out = Vec::new();
        while let Ok(result) = self.results.try_recv() {
            out.push(result);
        }
        out
    }
}

impl Drop for DecodeWorker {
    fn drop(&mut self) {
        self.request_tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn decode(bytes: Vec<u8>) -> Result<StaticSoundData, String> {
    StaticSoundData::from_cursor(Cursor::new(bytes)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_wav::pcm_wav_mono;

    // Drain with a deadline: decoding runs on the worker thread, so tests
    // poll briefly instead of assuming completion within one call.
    fn drain_one(worker: &DecodeWorker) -> DecodeResult {
        for _ in 0..500 {
            if let Some(result) = worker.drain().into_iter().next() {
                return result;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("decode worker produced no result within a second");
    }

    #[test]
    fn worker_decodes_valid_bytes_off_thread() {
        let worker = DecodeWorker::spawn();
        assert!(worker.send(11, pcm_wav_mono(64)));
        let result = drain_one(&worker);
        assert_eq!(result.key, 11);
        let data = result.decoded.expect("valid WAV decodes");
        assert!(data.num_frames() > 0);
    }

    #[test]
    fn worker_reports_undecodable_bytes_as_errors() {
        let worker = DecodeWorker::spawn();
        assert!(worker.send(3, b"not an audio file".to_vec()));
        let result = drain_one(&worker);
        assert_eq!(result.key, 3);
        assert!(result.decoded.is_err());
    }

    #[test]
    fn drop_joins_the_worker_cleanly() {
        let worker = DecodeWorker::spawn();
        worker.send(1, pcm_wav_mono(16));
        // Dropping with work possibly still queued must not hang or panic.
        drop(worker);
    }
}
