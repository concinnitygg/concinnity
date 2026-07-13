// src/gfx/graphics_system/settings_writer.rs
//
// Background writer for the persisted `Settings` file. Settings changes are
// applied live on the render thread, but the disk write is handed to this
// dedicated thread so a slow filesystem never stalls a frame. A burst of
// snapshots coalesces: only the newest one queued hits the disk. The thread
// is joined on Drop, after draining the queue, so the final change is always
// flushed before shutdown.

use crate::config::Settings;
use std::sync::mpsc;

pub(crate) struct SettingsWriter {
    tx: Option<mpsc::Sender<Settings>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SettingsWriter {
    pub fn spawn() -> Self {
        Self::with_sink(|cfg| cfg.save())
    }

    // Writer with an injectable persistence sink, so tests never touch the
    // real settings file.
    fn with_sink(sink: impl Fn(&Settings) -> std::io::Result<()> + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel::<Settings>();
        let thread = std::thread::Builder::new()
            .name("cn-settings-writer".into())
            .spawn(move || {
                while let Ok(mut cfg) = rx.recv() {
                    // Coalesce queued snapshots; the newest wins.
                    while let Ok(newer) = rx.try_recv() {
                        cfg = newer;
                    }
                    if let Err(e) = sink(&cfg) {
                        tracing::warn!("settings save failed: {e}");
                    }
                }
            })
            .expect("spawn cn-settings-writer");
        Self {
            tx: Some(tx),
            thread: Some(thread),
        }
    }

    // Queue a snapshot for persistence. Never blocks on disk I/O.
    pub fn save(&self, cfg: Settings) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(cfg);
        }
    }
}

impl Drop for SettingsWriter {
    fn drop(&mut self) {
        // Dropping the sender lets the thread drain what is queued, then exit.
        drop(self.tx.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // Every change queued before shutdown is flushed, and the last snapshot
    // queued is the last one written.
    #[test]
    fn flushes_last_snapshot_before_join() {
        let written: Arc<Mutex<Vec<Settings>>> = Arc::default();
        let sink_log = written.clone();
        let writer = SettingsWriter::with_sink(move |cfg| {
            sink_log.lock().unwrap().push(cfg.clone());
            Ok(())
        });

        let mut first = Settings::default();
        first.graphics.vsync = Some(false);
        let mut last = Settings::default();
        last.graphics.vsync = Some(true);
        writer.save(first);
        writer.save(last);
        drop(writer);

        let written = written.lock().unwrap();
        assert!(!written.is_empty(), "queued snapshots were flushed");
        assert_eq!(
            written.last().unwrap().graphics.vsync,
            Some(true),
            "newest snapshot wins"
        );
    }

    // A failing sink is reported, not propagated: later saves still flush.
    // The test blocks on the sink's call notifications (no sleeps), so each
    // save is observed before the next is queued and nothing coalesces.
    #[test]
    fn sink_error_does_not_stop_the_writer() {
        let (called_tx, called_rx) = mpsc::channel::<u32>();
        let calls = Arc::new(Mutex::new(0u32));
        let seen = calls.clone();
        let writer = SettingsWriter::with_sink(move |_| {
            let mut n = seen.lock().unwrap();
            *n += 1;
            let result = if *n == 1 {
                Err(std::io::Error::other("disk full"))
            } else {
                Ok(())
            };
            let _ = called_tx.send(*n);
            result
        });

        writer.save(Settings::default());
        assert_eq!(called_rx.recv().unwrap(), 1, "first write attempted");
        writer.save(Settings::default());
        assert_eq!(called_rx.recv().unwrap(), 2, "writer survived the failure");
    }
}
