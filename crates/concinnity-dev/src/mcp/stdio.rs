//! Newline-delimited JSON over a byte stream: the framing MCP clients use when
//! they spawn a server as a child process.
//!
//! One JSON message per line in, one per line out, and nothing else on the
//! output stream: a client parses every line it reads, so a stray print there
//! ends the session. Diagnostics go to stderr, which is why they are `eprintln`
//! at the call sites rather than anything routed.

use std::io::{BufRead, Write};

use super::server::{Executor, Server};

/// Serve until `input` reaches EOF.
pub(super) fn serve<E, R, W>(server: &Server<E>, input: R, output: &mut W) -> std::io::Result<()>
where
    E: Executor,
    R: BufRead,
    W: Write,
{
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = server.handle(&line) {
            writeln!(output, "{reply}")?;
            output.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::{Map, Value};

    use super::*;

    // Nothing here reaches a call, so the executor only has to exist.
    struct Never;

    impl Executor for Never {
        fn call(&self, _name: &str, _arguments: &Map<String, Value>) -> Value {
            unreachable!("these messages never reach a tools/call")
        }
    }

    fn transcript(input: &str) -> Vec<Value> {
        let server = Server::new(Never);
        let mut output = Vec::new();
        serve(&server, Cursor::new(input.to_string()), &mut output).expect("stdio serve");
        String::from_utf8(output)
            .expect("output is utf-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is one JSON message"))
            .collect()
    }

    #[test]
    fn each_call_answers_on_its_own_line() {
        let lines = transcript(concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#,
            "\n",
        ));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["id"], 1);
        assert_eq!(lines[1]["id"], 2);
    }

    #[test]
    fn a_notification_writes_nothing() {
        assert!(
            transcript("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
                .is_empty()
        );
    }

    #[test]
    fn blank_lines_are_skipped_and_eof_ends_the_session() {
        let lines = transcript("\n  \n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
        assert_eq!(lines.len(), 1);
    }
}
