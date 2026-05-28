//! Core message types shared across MiniMCP.
//!
//! Mirrors `minimcp/types.py`. MCP messages are exchanged as JSON strings, and
//! handlers return either a `Message` to send back or a `NoMessage` marker.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// MCP messages are exchanged as JSON-RPC strings.
pub type Message = String;

/// Marker for handler responses that produce no outgoing message.
///
/// See <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoMessage {
    /// Response to a client notification.
    Notification,
}

/// The result of handling a single incoming MCP message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A JSON-RPC message string to send back to the client.
    Message(Message),
    /// No outgoing message (e.g. the input was a notification).
    NoMessage(NoMessage),
}

/// Callback used by transports to push server-to-client messages.
///
/// This is the Rust analog of Python's `Send` type. It is named `SendFn` to
/// avoid colliding with the standard library `Send` marker trait.
pub type SendFn = Arc<dyn Fn(Message) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Resource not found error code, defined by the MCP spec.
///
/// See <https://modelcontextprotocol.io/specification/2025-06-18/server/resources#error-handling>.
pub const RESOURCE_NOT_FOUND: i32 = -32002;
