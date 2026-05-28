//! Server-to-client notifications.
//!
//! Mirrors `minimcp/responder.py`. A `Responder` is available to handlers (via
//! the request context) on transports that support bidirectional communication
//! and lets handlers push notifications, such as progress updates, to the
//! client.

use serde_json::{json, Value};

use crate::mcp_types::JSONRPC_VERSION;
use crate::time_limiter::TimeLimiter;
use crate::types::SendFn;

/// Enables message handlers to send notifications back to the client.
///
/// Sending a notification resets the idle timeout so an actively communicating
/// handler is not timed out.
#[derive(Clone)]
pub struct Responder {
    send: SendFn,
    time_limiter: Option<TimeLimiter>,
}

impl Responder {
    /// Create a responder.
    ///
    /// # Arguments
    ///
    /// * `send` - The callback used to transmit messages to the client.
    /// * `time_limiter` - The handler's idle-timeout limiter, if enabled.
    pub fn new(send: SendFn, time_limiter: Option<TimeLimiter>) -> Self {
        Self { send, time_limiter }
    }

    /// Send a JSON-RPC notification with the given method and params.
    ///
    /// The idle timeout is reset before the message is sent.
    pub async fn send_notification(&self, method: &str, params: Value) {
        if let Some(limiter) = &self.time_limiter {
            limiter.reset();
        }

        let message = json!({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params,
        })
        .to_string();

        (self.send)(message).await;
    }

    /// Report progress for the current operation to the client.
    ///
    /// # Arguments
    ///
    /// * `progress_token` - The token supplied by the client in the request metadata.
    /// * `progress` - Current progress value.
    /// * `total` - Optional total value for computing completion percentage.
    /// * `message` - Optional human-readable status message.
    pub async fn report_progress(
        &self,
        progress_token: Value,
        progress: f64,
        total: Option<f64>,
        message: Option<String>,
    ) {
        let mut params = json!({
            "progressToken": progress_token,
            "progress": progress,
        });
        if let Some(total) = total {
            params["total"] = json!(total);
        }
        if let Some(message) = message {
            params["message"] = json!(message);
        }

        self.send_notification("notifications/progress", params)
            .await;
    }
}
