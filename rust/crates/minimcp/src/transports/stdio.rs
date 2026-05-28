//! Stdio transport.
//!
//! Mirrors `minimcp/transports/stdio.py`. Reads newline-delimited JSON-RPC
//! messages from stdin, dispatches each concurrently, and writes responses (and
//! any handler-sent notifications) to stdout. Per the MCP spec, logging must go
//! to stderr so it does not corrupt the stdout message stream.

use std::any::Any;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

use crate::error::InvalidMessageError;
use crate::server::MiniMCP;
use crate::types::{Message, Outcome, SendFn};

/// Stdio transport implementation.
pub struct StdioTransport<S = ()> {
    mcp: Arc<MiniMCP<S>>,
}

impl<S: Any + Send + Sync + 'static> StdioTransport<S> {
    /// Create a stdio transport for the given server.
    pub fn new(mcp: Arc<MiniMCP<S>>) -> Self {
        Self { mcp }
    }

    /// Run the transport until stdin is closed.
    ///
    /// A dedicated writer task owns stdout; all responses and notifications are
    /// funneled through an unbounded channel so concurrent handlers do not
    /// interleave partial lines.
    pub async fn run(&self) -> std::io::Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Message>();

        let writer = tokio::spawn(async move {
            let mut stdout = tokio::io::stdout();
            while let Some(line) = rx.recv().await {
                if stdout.write_all(line.as_bytes()).await.is_err()
                    || stdout.write_all(b"\n").await.is_err()
                    || stdout.flush().await.is_err()
                {
                    break;
                }
            }
        });

        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        while let Some(line) = lines.next_line().await? {
            let trimmed = line.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            let mcp = self.mcp.clone();
            let tx = tx.clone();
            tokio::spawn(dispatch(mcp, trimmed, tx));
        }

        drop(tx);
        let _ = writer.await;
        Ok(())
    }
}

async fn dispatch<S: Any + Send + Sync + 'static>(
    mcp: Arc<MiniMCP<S>>,
    message: Message,
    tx: UnboundedSender<Message>,
) {
    let send_tx = tx.clone();
    let send: SendFn = Arc::new(move |msg: Message| {
        let tx = send_tx.clone();
        Box::pin(async move {
            let _ = tx.send(msg);
        })
    });

    match mcp.handle(message, Some(send), None).await {
        Ok(Outcome::Message(response)) => {
            let _ = tx.send(response);
        }
        Ok(Outcome::NoMessage(_)) => {}
        Err(InvalidMessageError { response, .. }) => {
            let _ = tx.send(response);
        }
    }
}
