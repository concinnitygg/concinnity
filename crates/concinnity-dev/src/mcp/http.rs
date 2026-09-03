//! The Streamable HTTP transport in its stateless form: one JSON-RPC message
//! per request, one response, one connection.
//!
//! `POST /mcp` with a JSON-RPC body answers `application/json` with the
//! response, or 202 with no body when the message was a notification. An
//! `Origin` header naming anything but localhost is refused, which is what the
//! spec asks of a local server so a page in a browser cannot reach it by DNS
//! rebinding. Everything here works over a `BufRead` / `Write` pair, so the
//! whole surface is exercised without a socket.

use std::io::{BufRead, Write};

use super::server::{Executor, Server};

/// The one resource this server exposes.
const PATH: &str = "/mcp";

/// Largest body accepted, so a stray connection cannot exhaust memory.
const MAX_BODY: usize = 1 << 20;

/// Why a request was refused before any protocol work.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Refusal {
    Malformed,
    UnknownPath,
    WrongMethod,
    NoLength,
    ForeignOrigin,
    TooLarge,
}

impl Refusal {
    fn status(self) -> (u16, &'static str) {
        match self {
            Refusal::Malformed => (400, "Bad Request"),
            Refusal::ForeignOrigin => (403, "Forbidden"),
            Refusal::UnknownPath => (404, "Not Found"),
            Refusal::WrongMethod => (405, "Method Not Allowed"),
            Refusal::NoLength => (411, "Length Required"),
            Refusal::TooLarge => (413, "Payload Too Large"),
        }
    }
}

/// What one connection turned out to carry.
#[derive(PartialEq, Eq, Debug)]
pub(super) enum Incoming {
    /// A JSON-RPC body to answer.
    Body(String),
    /// A refusal to answer with instead.
    Refused(Refusal),
    /// The peer closed without sending a request line; nothing to answer.
    Empty,
}

/// Answer one request, then leave the connection to be closed.
pub(super) fn serve<R, W, E>(
    server: &Server<E>,
    input: &mut R,
    output: &mut W,
) -> std::io::Result<()>
where
    R: BufRead,
    W: Write,
    E: Executor,
{
    match read_request(input) {
        Incoming::Empty => Ok(()),
        Incoming::Refused(refusal) => {
            let (code, reason) = refusal.status();
            write_head(output, code, reason, None)?;
            output.flush()
        }
        Incoming::Body(body) => match server.handle(&body) {
            Some(reply) => {
                let text = reply.to_string();
                write_head(output, 200, "OK", Some(text.len()))?;
                output.write_all(text.as_bytes())?;
                output.flush()
            }
            None => {
                write_head(output, 202, "Accepted", None)?;
                output.flush()
            }
        },
    }
}

/// Read one HTTP/1.1 request and report the JSON-RPC body it carried.
pub(super) fn read_request<R: BufRead>(input: &mut R) -> Incoming {
    let Some(start) = read_line(input) else {
        return Incoming::Refused(Refusal::Malformed);
    };
    if start.trim().is_empty() {
        return Incoming::Empty;
    }

    let mut fields = start.split_whitespace();
    let (Some(method), Some(target)) = (fields.next(), fields.next()) else {
        return Incoming::Refused(Refusal::Malformed);
    };

    let mut length: Option<usize> = None;
    let mut origin: Option<String> = None;
    loop {
        let Some(line) = read_line(input) else {
            return Incoming::Refused(Refusal::Malformed);
        };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Incoming::Refused(Refusal::Malformed);
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => match value.parse::<usize>() {
                Ok(n) => length = Some(n),
                Err(_) => return Incoming::Refused(Refusal::Malformed),
            },
            "origin" => origin = Some(value.to_string()),
            _ => {}
        }
    }

    if origin.is_some_and(|value| !is_local_origin(&value)) {
        return Incoming::Refused(Refusal::ForeignOrigin);
    }
    if path_of(target) != PATH {
        return Incoming::Refused(Refusal::UnknownPath);
    }
    if !method.eq_ignore_ascii_case("POST") {
        return Incoming::Refused(Refusal::WrongMethod);
    }
    let Some(length) = length else {
        return Incoming::Refused(Refusal::NoLength);
    };
    if length > MAX_BODY {
        return Incoming::Refused(Refusal::TooLarge);
    }

    let mut body = vec![0u8; length];
    if input.read_exact(&mut body).is_err() {
        return Incoming::Refused(Refusal::Malformed);
    }
    match String::from_utf8(body) {
        Ok(text) => Incoming::Body(text),
        Err(_) => Incoming::Refused(Refusal::Malformed),
    }
}

// One line including its terminator, or None when the stream ended or failed
// mid-line. An empty read is a clean EOF, reported as an empty line.
fn read_line<R: BufRead>(input: &mut R) -> Option<String> {
    let mut line = String::new();
    match input.read_line(&mut line) {
        Ok(_) => Some(line),
        Err(_) => None,
    }
}

// The path of a request target, dropping any query string. An absolute-form
// target (`http://host/mcp`) carries the same path after its authority.
fn path_of(target: &str) -> &str {
    let path = target.split(['?', '#']).next().unwrap_or(target);
    match path.split_once("://") {
        Some((_, rest)) => match rest.find('/') {
            Some(at) => &rest[at..],
            None => "/",
        },
        None => path,
    }
}

// True for an origin that names this machine. Anything else is a page served
// from elsewhere reaching for the debug port, which this server never answers.
fn is_local_origin(origin: &str) -> bool {
    let Some((scheme, rest)) = origin.split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }
    let host = match rest.strip_prefix('[') {
        // An IPv6 literal keeps its brackets; the port follows the closer.
        Some(inside) => match inside.split_once(']') {
            Some((address, _)) => address,
            None => return false,
        },
        None => rest.split(':').next().unwrap_or(rest),
    };
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

fn write_head<W: Write>(
    output: &mut W,
    code: u16,
    reason: &str,
    body_len: Option<usize>,
) -> std::io::Result<()> {
    write!(output, "HTTP/1.1 {code} {reason}\r\n")?;
    match body_len {
        Some(len) => write!(
            output,
            "Content-Type: application/json\r\nContent-Length: {len}\r\n"
        )?,
        None => write!(output, "Content-Length: 0\r\n")?,
    }
    write!(output, "Connection: close\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::{Map, Value, json};

    use super::*;

    // A server whose calls all succeed, so the transport is what a test reads.
    struct Always;

    impl Executor for Always {
        fn call(&self, _name: &str, _arguments: &Map<String, Value>) -> Value {
            super::super::tools::text_result(r#"{"ok":true}"#, false)
        }
    }

    fn request(method: &str, target: &str, headers: &str, body: &str) -> String {
        format!(
            "{method} {target} HTTP/1.1\r\nHost: 127.0.0.1:8777\r\n{headers}Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
    }

    fn incoming(text: &str) -> Incoming {
        read_request(&mut Cursor::new(text.to_string()))
    }

    fn answer(text: &str) -> String {
        let server = Server::new(Always);
        let mut output = Vec::new();
        serve(&server, &mut Cursor::new(text.to_string()), &mut output).expect("serve");
        String::from_utf8(output).expect("the response is utf-8")
    }

    fn body_of(response: &str) -> &str {
        response.split_once("\r\n\r\n").expect("a header block").1
    }

    #[test]
    fn a_post_to_the_endpoint_carries_its_body() {
        let call = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        assert_eq!(
            incoming(&request("POST", PATH, "", call)),
            Incoming::Body(call.to_string())
        );
    }

    #[test]
    fn a_query_string_does_not_change_the_path() {
        let call = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        assert_eq!(
            incoming(&request("POST", "/mcp?session=1", "", call)),
            Incoming::Body(call.to_string())
        );
    }

    #[test]
    fn a_body_without_a_content_length_is_refused() {
        let text = "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n{}";
        assert_eq!(incoming(text), Incoming::Refused(Refusal::NoLength));
    }

    #[test]
    fn another_path_is_not_found() {
        assert_eq!(
            incoming(&request("POST", "/", "", "{}")),
            Incoming::Refused(Refusal::UnknownPath)
        );
    }

    #[test]
    fn another_method_is_not_allowed() {
        for method in ["GET", "DELETE", "PUT"] {
            assert_eq!(
                incoming(&request(method, PATH, "", "{}")),
                Incoming::Refused(Refusal::WrongMethod),
                "{method}"
            );
        }
    }

    #[test]
    fn a_localhost_origin_is_accepted() {
        for origin in [
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1:8777",
            "https://LOCALHOST:1",
            "http://[::1]:8777",
        ] {
            let headers = format!("Origin: {origin}\r\n");
            assert!(
                matches!(
                    incoming(&request("POST", PATH, &headers, "{}")),
                    Incoming::Body(_)
                ),
                "{origin}"
            );
        }
    }

    #[test]
    fn an_origin_from_anywhere_else_is_refused() {
        for origin in [
            "http://evil.example",
            "https://localhost.evil.example",
            "null",
            "http://127.0.0.1.evil.example",
            "file://localhost",
        ] {
            let headers = format!("Origin: {origin}\r\n");
            assert_eq!(
                incoming(&request("POST", PATH, &headers, "{}")),
                Incoming::Refused(Refusal::ForeignOrigin),
                "{origin}"
            );
        }
    }

    #[test]
    fn an_oversized_body_is_refused_before_it_is_read() {
        let text = format!(
            "POST /mcp HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        assert_eq!(incoming(&text), Incoming::Refused(Refusal::TooLarge));
    }

    #[test]
    fn a_truncated_body_is_malformed() {
        let text = "POST /mcp HTTP/1.1\r\nContent-Length: 40\r\n\r\n{}";
        assert_eq!(incoming(text), Incoming::Refused(Refusal::Malformed));
    }

    #[test]
    fn a_closed_connection_carries_nothing_to_answer() {
        assert_eq!(incoming(""), Incoming::Empty);
    }

    #[test]
    fn a_call_answers_two_hundred_with_the_json_response() {
        let response = answer(&request(
            "POST",
            PATH,
            "",
            r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
        ));
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
        assert!(response.contains("Content-Type: application/json\r\n"));
        assert!(response.contains("Connection: close\r\n"));

        let body = body_of(&response);
        assert!(response.contains(&format!("Content-Length: {}\r\n", body.len())));
        let parsed: Value = serde_json::from_str(body).expect("a JSON-RPC response");
        assert_eq!(parsed["id"], json!(1));
        assert_eq!(parsed["result"], json!({}));
    }

    #[test]
    fn a_notification_answers_two_hundred_and_two_with_no_body() {
        let response = answer(&request(
            "POST",
            PATH,
            "",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        ));
        assert!(
            response.starts_with("HTTP/1.1 202 Accepted\r\n"),
            "{response}"
        );
        assert!(response.contains("Content-Length: 0\r\n"));
        assert!(body_of(&response).is_empty());
    }

    #[test]
    fn a_refusal_answers_its_status_with_no_body() {
        let response = answer(&request("GET", PATH, "", ""));
        assert!(
            response.starts_with("HTTP/1.1 405 Method Not Allowed\r\n"),
            "{response}"
        );
        assert!(body_of(&response).is_empty());
    }

    #[test]
    fn a_tool_call_reaches_the_executor_through_the_transport() {
        let response = answer(&request(
            "POST",
            PATH,
            "",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ping"}}"#,
        ));
        let parsed: Value = serde_json::from_str(body_of(&response)).expect("a response");
        assert_eq!(parsed["result"]["isError"], json!(false));
        assert_eq!(parsed["result"]["content"][0]["text"], r#"{"ok":true}"#);
    }
}
