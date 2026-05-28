//! The [`MiniMCP`] orchestrator.
//!
//! Mirrors `minimcp/minimcp.py`. `MiniMCP` is a stateless entry point: it parses
//! an incoming JSON-RPC message, enforces concurrency and idle-timeout limits,
//! activates the per-request context, dispatches to a registered handler, and
//! formats results and errors back into JSON-RPC messages.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::context::{Context, ContextManager, CURRENT_CONTEXT};
use crate::error::{codes, Error, InvalidMessageError};
use crate::json_rpc::{self, Incoming, ParseError};
use crate::managers::tool::ToolManager;
use crate::mcp_types::{
    Implementation, InitializeResult, ListToolsResult, ServerCapabilities, LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
};
use crate::responder::Responder;
use crate::time_limiter::TimeLimiter;
use crate::types::{Message, NoMessage, Outcome, SendFn};

const DEFAULT_IDLE_TIMEOUT_SECS: i64 = 30;
const DEFAULT_MAX_CONCURRENCY: i64 = 100;

/// Stateless orchestrator for building MCP servers.
///
/// The generic parameter `S` is the optional per-request scope type (auth
/// details, user info, database handles, etc.), reachable inside handlers via
/// `mcp.context.get_scope()`. Use `MiniMCP<()>` when no scope is needed.
///
/// # Example
///
/// ```no_run
/// use minimcp::MiniMCP;
/// use serde_json::json;
///
/// # async fn run() {
/// let mcp = MiniMCP::<()>::new("my-server");
/// mcp.tool
///     .add(
///         "echo",
///         Some("Echo back the input"),
///         json!({"type": "object", "properties": {"value": {"type": "string"}}}),
///         None,
///         |args| async move { Ok(json!({"result": args["value"].clone()})) },
///     )
///     .unwrap();
/// # }
/// ```
pub struct MiniMCP<S = ()> {
    name: String,
    version: String,
    instructions: Option<String>,

    /// Tool registration and execution.
    pub tool: ToolManager,
    /// Accessor for the active request context.
    pub context: ContextManager<S>,

    semaphore: Option<Arc<Semaphore>>,
    idle_timeout: Option<Duration>,
    _scope: PhantomData<fn() -> S>,
}

impl<S> MiniMCP<S> {
    /// Create a server with default limits (idle timeout 30s, max concurrency 100).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "0.0.0".to_string(),
            instructions: None,
            tool: ToolManager::new(),
            context: ContextManager::new(),
            semaphore: Some(Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENCY as usize))),
            idle_timeout: Some(Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS as u64)),
            _scope: PhantomData,
        }
    }

    /// Set the server version reported during the initialize handshake.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the human-readable usage instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set the maximum number of concurrent handlers. Pass `-1` to disable the limit.
    pub fn with_max_concurrency(mut self, max_concurrency: i64) -> Self {
        self.semaphore = if max_concurrency == -1 {
            None
        } else {
            Some(Arc::new(Semaphore::new(max_concurrency.max(1) as usize)))
        };
        self
    }

    /// Set the per-handler idle timeout in seconds. Pass `-1` to disable the timeout.
    pub fn with_idle_timeout(mut self, idle_timeout_secs: i64) -> Self {
        self.idle_timeout = if idle_timeout_secs == -1 {
            None
        } else {
            Some(Duration::from_secs(idle_timeout_secs.max(0) as u64))
        };
        self
    }

    /// The server name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The server version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The server instructions, if any.
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }
}

impl<S: Any + Send + Sync + 'static> MiniMCP<S> {
    /// Handle a single incoming MCP message.
    ///
    /// # Arguments
    ///
    /// * `message` - The raw JSON-RPC message string from the client.
    /// * `send` - Optional callback for server-to-client messages (notifications).
    /// * `scope` - Optional per-request scope made available to handlers.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidMessageError`] for parse and JSON-RPC envelope failures.
    /// All other failures are converted into a JSON-RPC error [`Outcome::Message`].
    pub async fn handle(
        &self,
        message: Message,
        send: Option<SendFn>,
        scope: Option<Arc<S>>,
    ) -> Result<Outcome, InvalidMessageError> {
        let incoming = match json_rpc::parse_incoming(&message) {
            Ok(incoming) => incoming,
            Err(ParseError::Json(msg)) => {
                return Err(self.invalid_message(&message, codes::PARSE_ERROR, &msg))
            }
            Err(ParseError::JsonRpc(msg)) => {
                return Err(self.invalid_message(&message, codes::INVALID_REQUEST, &msg))
            }
        };

        // Enforce concurrency limit for the duration of handling.
        let _permit = match &self.semaphore {
            Some(sem) => Some(sem.acquire().await.expect("semaphore closed")),
            None => None,
        };

        let time_limiter = self.idle_timeout.map(TimeLimiter::new);
        let responder = send.map(|s| Responder::new(s, time_limiter.clone()));
        let erased_scope = scope.map(|s| s as Arc<dyn Any + Send + Sync>);

        let context = Context {
            message: message.clone(),
            scope: erased_scope,
            responder,
        };

        let fut = CURRENT_CONTEXT.scope(context, self.dispatch(incoming));

        let dispatch_result = match self.idle_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => Err(Error::Runtime("handler idle timeout".to_string())),
            },
            None => fut.await,
        };

        match dispatch_result {
            Ok(outcome) => Ok(outcome),
            Err(error) => {
                tracing::error!(error = %error, "error handling message");
                Ok(Outcome::Message(json_rpc::build_error_message(
                    &message,
                    error.code(),
                    &error.to_string(),
                )))
            }
        }
    }

    async fn dispatch(&self, incoming: Incoming) -> Result<Outcome, Error> {
        match incoming {
            Incoming::Request(request) => {
                let response = self.handle_request(request).await?;
                Ok(Outcome::Message(response))
            }
            Incoming::Notification(notification) => {
                // TODO: dispatch client notifications to registered handlers.
                tracing::debug!(method = %notification.method, "received notification");
                Ok(Outcome::NoMessage(NoMessage::Notification))
            }
        }
    }

    async fn handle_request(&self, request: json_rpc::Request) -> Result<Message, Error> {
        match request.method.as_str() {
            "initialize" => {
                let result = self.initialize(&request.params);
                Ok(json_rpc::build_response_message(&request.id, &result))
            }
            "ping" => Ok(json_rpc::build_response_message(&request.id, &json!({}))),
            "tools/list" => {
                let result = ListToolsResult {
                    tools: self.tool.list(),
                };
                Ok(json_rpc::build_response_message(&request.id, &result))
            }
            "tools/call" => {
                let name = request
                    .params
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::InvalidArguments("missing tool name".to_string()))?;
                let args = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let result = self.tool.call(name, args).await?;
                Ok(json_rpc::build_response_message(&request.id, &result))
            }
            other => Err(Error::RequestHandlerNotFound(other.to_string())),
        }
    }

    fn initialize(&self, params: &Value) -> InitializeResult {
        let client_version = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or("");
        let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&client_version) {
            client_version.to_string()
        } else {
            LATEST_PROTOCOL_VERSION.to_string()
        };

        InitializeResult {
            protocol_version: negotiated,
            capabilities: ServerCapabilities {
                tools: Some(json!({})),
                ..Default::default()
            },
            server_info: Implementation {
                name: self.name.clone(),
                version: self.version.clone(),
            },
            instructions: self.instructions.clone(),
        }
    }

    fn invalid_message(&self, raw: &str, code: i32, message: &str) -> InvalidMessageError {
        InvalidMessageError {
            message: message.to_string(),
            response: json_rpc::build_error_message(raw, code, message),
        }
    }
}
