//! Tool registration and execution.
//!
//! Mirrors `minimcp/managers/tool_manager.py`. Tools are registered with a
//! name, JSON Schemas, and an async handler. A handler receives the call
//! arguments as a JSON value and returns the structured-content value that is
//! placed in the tool result.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use serde_json::Value;

use crate::error::Error;
use crate::mcp_types::{CallToolResult, ContentBlock, Tool};

/// A boxed, owned future returned by tool handlers.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Result returned by a tool handler.
///
/// `Ok` carries the structured-content value placed in the tool result; `Err`
/// carries a human-readable error message.
pub type ToolResult = Result<Value, String>;

type ToolHandler = Arc<dyn Fn(Value) -> BoxFuture<ToolResult> + Send + Sync>;

struct Registration {
    tool: Tool,
    handler: ToolHandler,
}

/// Registers and executes MCP tool handlers.
#[derive(Default)]
pub struct ToolManager {
    tools: RwLock<HashMap<String, Registration>>,
}

impl ToolManager {
    /// Create an empty tool manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool handler.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique tool identifier.
    /// * `description` - Optional human-readable description.
    /// * `input_schema` - JSON Schema for the tool input.
    /// * `output_schema` - Optional JSON Schema for the structured output.
    /// * `handler` - Async function mapping call arguments to a structured result.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Primitive`] if a tool with the same name is already registered.
    pub fn add<F, Fut>(
        &self,
        name: &str,
        description: Option<&str>,
        input_schema: Value,
        output_schema: Option<Value>,
        handler: F,
    ) -> Result<Tool, Error>
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        let tool = Tool {
            name: name.to_string(),
            description: description.map(str::to_string),
            input_schema,
            output_schema,
        };

        let boxed: ToolHandler = Arc::new(move |args: Value| {
            let fut = handler(args);
            Box::pin(fut) as BoxFuture<ToolResult>
        });

        let mut tools = self.tools.write().expect("tool registry poisoned");
        if tools.contains_key(name) {
            return Err(Error::Primitive(format!("Tool {name} already registered")));
        }

        tools.insert(
            name.to_string(),
            Registration {
                tool: tool.clone(),
                handler: boxed,
            },
        );
        tracing::debug!(tool = name, "tool added");

        Ok(tool)
    }

    /// Remove a tool by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Primitive`] if the tool is not found.
    pub fn remove(&self, name: &str) -> Result<Tool, Error> {
        let mut tools = self.tools.write().expect("tool registry poisoned");
        tools
            .remove(name)
            .map(|reg| reg.tool)
            .ok_or_else(|| Error::Primitive(format!("Unknown tool: {name}")))
    }

    /// List all registered tools.
    pub fn list(&self) -> Vec<Tool> {
        let tools = self.tools.read().expect("tool registry poisoned");
        tools.values().map(|reg| reg.tool.clone()).collect()
    }

    /// Execute a tool by name with the given arguments.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Primitive`] if the tool is unknown, or [`Error::Runtime`]
    /// if the handler reports a failure.
    pub async fn call(&self, name: &str, args: Value) -> Result<CallToolResult, Error> {
        // Clone the handler out before awaiting so the lock is not held across .await.
        let handler = {
            let tools = self.tools.read().expect("tool registry poisoned");
            match tools.get(name) {
                Some(reg) => reg.handler.clone(),
                None => return Err(Error::Primitive(format!("Unknown tool: {name}"))),
            }
        };

        let structured = handler(args).await.map_err(Error::Runtime)?;

        Ok(CallToolResult {
            content: vec![ContentBlock::Text {
                text: structured.to_string(),
            }],
            structured_content: Some(structured),
            is_error: false,
        })
    }
}
