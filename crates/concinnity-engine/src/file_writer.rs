//! Background writer for small persisted files (the settings store, story saves,
//! behavior state).
//! The caller encodes the bytes and queues the write; a Background-role thread
//! applies the queued writes and removals in the order they were queued, so a
//! slow filesystem never stalls the sim thread. A write immediately followed by
//! another to the same file is skipped, since the later one replaces it. The
//! thread starts on the first queued operation and is joined on Drop after it
//! drains the queue, so the last save is always flushed before shutdown.

use concinnity_host::thread::{ThreadRole, set_current_thread_role};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

// One queued file operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileOp {
    Write(PathBuf, Vec<u8>),
    Remove(PathBuf),
}

impl FileOp {
    fn path(&self) -> &Path {
        match self {
            FileOp::Write(path, _) | FileOp::Remove(path) => path,
        }
    }
}

type Sink = Box<dyn FnMut(&FileOp) -> std::io::Result<()> + Send>;

// Operations queued but not yet applied, so a reader can wait them out.
#[derive(Default)]
struct Pending {
    count: Mutex<usize>,
    drained: Condvar,
}

impl Pending {
    fn add(&self) {
        *self.count.lock().unwrap_or_else(|p| p.into_inner()) += 1;
    }

    fn done(&self) {
        let mut count = self.count.lock().unwrap_or_else(|p| p.into_inner());
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.drained.notify_all();
        }
    }

    fn wait(&self) {
        let mut count = self.count.lock().unwrap_or_else(|p| p.into_inner());
        while *count > 0 {
            count = self.drained.wait(count).unwrap_or_else(|p| p.into_inner());
        }
    }
}

struct Running {
    tx: mpsc::Sender<FileOp>,
    thread: std::thread::JoinHandle<()>,
}

pub(crate) struct FileWriter {
    name: &'static str,
    sink: Mutex<Option<Sink>>,
    pending: Arc<Pending>,
    running: OnceLock<Option<Running>>,
}

impl std::fmt::Debug for FileWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWriter")
            .field("name", &self.name)
            .finish()
    }
}

impl FileWriter {
    // A writer whose thread, once started, is called `name`.
    pub(crate) fn new(name: &'static str) -> Self {
        Self::with_sink(name, apply)
    }

    // Writer with an injectable sink, so tests can observe the operations.
    fn with_sink(
        name: &'static str,
        sink: impl FnMut(&FileOp) -> std::io::Result<()> + Send + 'static,
    ) -> Self {
        Self {
            name,
            sink: Mutex::new(Some(Box::new(sink))),
            pending: Arc::default(),
            running: OnceLock::new(),
        }
    }

    // Queue `bytes` to replace the contents of `path`, creating its directory.
    pub(crate) fn write(&self, path: PathBuf, bytes: Vec<u8>) {
        self.queue(FileOp::Write(path, bytes));
    }

    // Queue `value`, encoded as CBOR, to replace the contents of `path`. An
    // encoding failure is logged naming `what`.
    pub(crate) fn write_cbor<T: serde::Serialize>(&self, path: PathBuf, value: &T, what: &str) {
        match crate::cbor_file::encode(value) {
            Ok(bytes) => self.write(path, bytes),
            Err(e) => tracing::warn!("{what} save failed: {e}"),
        }
    }

    // Queue the removal of `path`; a missing file is not an error.
    pub(crate) fn remove(&self, path: PathBuf) {
        self.queue(FileOp::Remove(path));
    }

    // Block until every operation queued so far has been applied, so a read
    // that follows sees them. Returns at once when nothing is queued.
    pub(crate) fn flush(&self) {
        self.pending.wait();
    }

    fn queue(&self, op: FileOp) {
        let Some(running) = self.running.get_or_init(|| self.start()) else {
            self.apply_inline(&op);
            return;
        };
        self.pending.add();
        if let Err(mpsc::SendError(op)) = running.tx.send(op) {
            self.pending.done();
            self.apply_inline(&op);
        }
    }

    // The fallback when no writer thread runs: apply on the caller's thread.
    fn apply_inline(&self, op: &FileOp) {
        if let Err(e) = apply(op) {
            tracing::warn!("{}: {} failed: {e}", self.name, op.path().display());
        }
    }

    // Start the thread, or `None` (operations then apply inline) when the OS
    // refuses one.
    fn start(&self) -> Option<Running> {
        let mut sink = self.sink.lock().unwrap_or_else(|p| p.into_inner()).take()?;
        let (tx, rx) = mpsc::channel::<FileOp>();
        let pending = self.pending.clone();
        let name = self.name;
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                set_current_thread_role(ThreadRole::Background);
                let mut report = |op: &FileOp| {
                    if let Err(e) = sink(op) {
                        tracing::warn!("{name}: {} failed: {e}", op.path().display());
                    }
                    pending.done();
                };
                while let Ok(mut op) = rx.recv() {
                    while let Ok(next) = rx.try_recv() {
                        if next.path() == op.path() {
                            pending.done();
                        } else {
                            report(&op);
                        }
                        op = next;
                    }
                    report(&op);
                }
            });
        match thread {
            Ok(thread) => Some(Running { tx, thread }),
            Err(e) => {
                tracing::warn!("{name}: could not start the writer thread: {e}");
                None
            }
        }
    }
}

impl Drop for FileWriter {
    fn drop(&mut self) {
        // Dropping the sender lets the thread drain what is queued, then exit.
        if let Some(Some(Running { tx, thread })) = self.running.take() {
            drop(tx);
            let _ = thread.join();
        }
    }
}

// Replace the contents of `path` with `bytes`, creating its directory. The
// bytes land in a sibling temporary file renamed over `path` (which replaces an
// existing file on every platform), so a crash mid-write leaves the old file
// whole rather than a torn one.
pub(crate) fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    let written = std::fs::File::create(&temp).and_then(|mut file| {
        file.write_all(bytes)?;
        file.sync_all()
    });
    match written.and_then(|()| std::fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
}

// Apply one operation to the filesystem.
fn apply(op: &FileOp) -> std::io::Result<()> {
    match op {
        FileOp::Write(path, bytes) => write_file(path, bytes),
        FileOp::Remove(path) => match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording() -> (FileWriter, Arc<Mutex<Vec<FileOp>>>) {
        let log: Arc<Mutex<Vec<FileOp>>> = Arc::default();
        let sink_log = log.clone();
        let writer = FileWriter::with_sink("test-writer", move |op| {
            sink_log.lock().unwrap().push(op.clone());
            Ok(())
        });
        (writer, log)
    }

    fn write(path: &str, byte: u8) -> FileOp {
        FileOp::Write(path.into(), vec![byte])
    }

    // Operations on different files land in the order they were queued, and
    // the last one queued before shutdown is flushed by Drop.
    #[test]
    fn operations_apply_in_order_and_flush_on_drop() {
        let (writer, log) = recording();
        writer.write("a".into(), vec![1]);
        writer.write("b".into(), vec![2]);
        writer.remove("a".into());
        writer.write("c".into(), vec![3]);
        drop(writer);
        assert_eq!(
            *log.lock().unwrap(),
            [
                write("a", 1),
                write("b", 2),
                FileOp::Remove("a".into()),
                write("c", 3)
            ]
        );
    }

    // However the queue coalesces, each file's last operation is applied and
    // nothing reorders.
    #[test]
    fn coalescing_keeps_each_files_last_operation_and_the_order() {
        let (writer, log) = recording();
        for i in 0..200u8 {
            writer.write(if i % 3 == 0 { "x" } else { "y" }.into(), vec![i]);
        }
        writer.write("y".into(), vec![255]);
        drop(writer);
        let log = log.lock().unwrap();
        assert_eq!(log.last(), Some(&write("y", 255)));
        let last_x = log.iter().rev().find(|op| op.path() == Path::new("x"));
        assert_eq!(last_x, Some(&write("x", 198)));
        let bytes: Vec<u8> = log
            .iter()
            .filter_map(|op| match op {
                FileOp::Write(_, b) => Some(b[0]),
                FileOp::Remove(_) => None,
            })
            .collect();
        assert!(bytes.windows(2).all(|w| w[0] < w[1]), "in queue order");
    }

    // A flush returns only once everything queued has reached the sink.
    #[test]
    fn flush_waits_for_the_queue() {
        let (writer, log) = recording();
        writer.flush();
        for i in 0..50u8 {
            writer.write(format!("f{i}").into(), vec![i]);
        }
        writer.flush();
        assert_eq!(log.lock().unwrap().len(), 50);
    }

    // A failing write is reported, not propagated: later ones still land.
    #[test]
    fn a_failed_write_does_not_stop_the_writer() {
        let log: Arc<Mutex<Vec<FileOp>>> = Arc::default();
        let sink_log = log.clone();
        let writer = FileWriter::with_sink("test-writer", move |op| {
            if op.path() == Path::new("bad") {
                return Err(std::io::Error::other("disk full"));
            }
            sink_log.lock().unwrap().push(op.clone());
            Ok(())
        });
        writer.write("bad".into(), vec![0]);
        writer.write("good".into(), vec![1]);
        writer.flush();
        assert_eq!(*log.lock().unwrap(), [write("good", 1)]);
    }

    #[test]
    fn the_default_sink_writes_and_removes_files() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("nested/save");
        let writer = FileWriter::new("test-writer");
        writer.write(path.clone(), b"hello".to_vec());
        writer.flush();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        writer.remove(path.clone());
        writer.remove(path.clone());
        writer.flush();
        assert!(!path.exists());
    }

    // A write replaces the old contents and leaves no temporary file beside
    // the target.
    #[test]
    fn a_write_replaces_the_file_and_leaves_no_temporary() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("nested/state");
        write_file(&path, b"old").expect("first write");
        write_file(&path, b"new").expect("second write");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        let names: Vec<_> = std::fs::read_dir(tree.join("nested"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["state"]);
    }
}
