//! Error hierarchy for MiniMCP.
//!
//! Mirrors `minimcp/exceptions.py`. [`Error`] is the central enum that
//! [`crate::MiniMCP::handle`] maps to JSON-RPC error codes. Parse and JSON-RPC
//! envelope errors are surfaced as [`InvalidMessageError`], which carries a
//! ready-to-send JSON-RPC error response that transports must return to the
//! client.

use thiserror::Error;

use crate::types::{Message, RESOURCE_NOT_FOUND};

/// JSON-RPC error codes used by MiniMCP (subset of the JSON-RPC 2.0 spec).
pub mod codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

/// Errors raised while handling an MCP message.
///
/// Each variant maps to a JSON-RPC error code via [`Error::code`].
#[derive(Debug, Error)]
pub enum Error {
    /// The message is not valid JSON.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),

    /// The message is not a valid JSON-RPC object.
    #[error("invalid JSON-RPC message: {0}")]
    InvalidJsonRpc(String),

    /// The message is not a valid MCP message.
    #[error("invalid MCP message: {0}")]
    InvalidMcpMessage(String),

    /// Handler arguments failed validation.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),

    /// The incoming message was neither a request nor a notification.
    #[error("unsupported message type: {0}")]
    UnsupportedMessageType(String),

    /// No handler is registered for the requested method.
    #[error("method not found: {0}")]
    RequestHandlerNotFound(String),

    /// A requested resource does not exist.
    #[error("resource not found: {0}")]
    ResourceNotFound(String),

    /// Failure while adding, retrieving, or removing a primitive.
    #[error("primitive error: {0}")]
    Primitive(String),

    /// A runtime error occurred while executing a handler.
    #[error("runtime error: {0}")]
    Runtime(String),

    /// Access to the request context failed.
    #[error(transparent)]
    Context(#[from] ContextError),
}

impl Error {
    /// Map the error to its JSON-RPC error code.
    pub fn code(&self) -> i32 {
        match self {
            Error::InvalidJson(_) => codes::PARSE_ERROR,
            Error::InvalidJsonRpc(_) | Error::UnsupportedMessageType(_) => codes::INVALID_REQUEST,
            Error::InvalidMcpMessage(_) | Error::InvalidArguments(_) | Error::Primitive(_) => {
                codes::INVALID_PARAMS
            }
            Error::RequestHandlerNotFound(_) => codes::METHOD_NOT_FOUND,
            Error::ResourceNotFound(_) => RESOURCE_NOT_FOUND,
            Error::Runtime(_) | Error::Context(_) => codes::INTERNAL_ERROR,
        }
    }
}

/// Raised when access to the request context fails.
///
/// Mirrors Python's `ContextError`. Causes include calling a context accessor
/// outside an active handler, or requesting a scope/responder that was not
/// provided for the current request.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ContextError(pub String);

/// Carries a JSON-RPC error response that the transport layer must return.
///
/// Mirrors Python's `InvalidMessageError`. Returned by [`crate::MiniMCP::handle`]
/// only for parse and JSON-RPC envelope failures, where no request id context
/// is available to embed the error in a normal response flow.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct InvalidMessageError {
    /// Human-readable description of the failure.
    pub message: String,
    /// The JSON-RPC error message to send back to the client.
    pub response: Message,
}
