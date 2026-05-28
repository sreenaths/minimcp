//! HTTP transport (request/response only).
//!
//! Mirrors `minimcp/transports/http.py`. Exposes the server as an axum
//! [`Router`] that accepts a single JSON-RPC message per POST and returns the
//! response as JSON (or `202 Accepted` for notifications).

use std::any::Any;
use std::sync::Arc;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;

use crate::error::InvalidMessageError;
use crate::server::MiniMCP;
use crate::types::Outcome;

/// HTTP transport implementation backed by axum.
pub struct HttpTransport<S = ()> {
    mcp: Arc<MiniMCP<S>>,
}

impl<S: Any + Send + Sync + 'static> HttpTransport<S> {
    /// Create an HTTP transport for the given server.
    pub fn new(mcp: Arc<MiniMCP<S>>) -> Self {
        Self { mcp }
    }

    /// Build an axum [`Router`] serving the MCP endpoint at `path` (POST only).
    pub fn router(&self, path: &str) -> Router {
        let mcp = self.mcp.clone();
        Router::new().route(
            path,
            post(move |body: String| {
                let mcp = mcp.clone();
                async move { handle_post(mcp, body).await }
            }),
        )
    }
}

async fn handle_post<S: Any + Send + Sync + 'static>(
    mcp: Arc<MiniMCP<S>>,
    body: String,
) -> Response {
    match mcp.handle(body, None, None).await {
        Ok(Outcome::Message(response)) => {
            ([(header::CONTENT_TYPE, "application/json")], response).into_response()
        }
        Ok(Outcome::NoMessage(_)) => StatusCode::ACCEPTED.into_response(),
        Err(InvalidMessageError { response, .. }) => (
            StatusCode::BAD_REQUEST,
            [(header::CONTENT_TYPE, "application/json")],
            response,
        )
            .into_response(),
    }
}
