# MiniMCP (Rust)

A Rust port of [MiniMCP](https://github.com/cloudera/minimcp), a minimal, stateless, and lightweight framework for building Model Context Protocol (MCP) servers.

This is an early scaffold. It reuses the MCP message type definitions (modeled with `serde`) but reimplements message handling in MiniMCP style rather than delegating to an SDK.

## Layout

```text
rust/
├── Cargo.toml                      # workspace
└── crates/
    ├── minimcp/                    # core library crate
    │   └── src/
    │       ├── lib.rs              # public re-exports
    │       ├── server.rs           # MiniMCP + handle()
    │       ├── types.rs            # Message, NoMessage, Outcome, SendFn
    │       ├── error.rs            # error hierarchy + JSON-RPC code mapping
    │       ├── mcp_types.rs        # MCP message type definitions (serde)
    │       ├── json_rpc.rs         # JSON-RPC parse/build helpers
    │       ├── context.rs          # per-request context (task-local)
    │       ├── responder.rs        # server-to-client notifications
    │       ├── time_limiter.rs     # idle timeout
    │       ├── managers/tool.rs    # tool registration/execution
    │       └── transports/         # stdio + http adapters
    └── minimcp-bench-server/       # benchmark server binaries
        └── src/
            ├── lib.rs              # build_server()
            ├── tools.rs            # workload tools + get_memory_usage
            ├── memory.rs           # RSS/maxrss measurement
            └── bin/
                ├── stdio_server.rs
                └── http_server.rs
```

## Mapping to the Python implementation

| Python (`src/minimcp/`) | Rust (`crates/minimcp/src/`) |
|---|---|
| `minimcp.py` (`MiniMCP.handle`) | `server.rs` |
| `types.py` | `types.rs` |
| `exceptions.py` | `error.rs` |
| `utils/json_rpc.py` | `json_rpc.rs` |
| `managers/context_manager.py` | `context.rs` |
| `managers/tool_manager.py` | `managers/tool.rs` |
| `responder.py` | `responder.rs` |
| `time_limiter.py` | `time_limiter.rs` |
| `transports/stdio.py` | `transports/stdio.rs` |
| `transports/http.py` | `transports/http.rs` |
| `mcp.types` (SDK) | `mcp_types.rs` |

## Build and test

```bash
cd rust
cargo build
cargo test
cargo clippy --all-targets
cargo fmt --check
```

Build only the stdio transport (no HTTP / axum):

```bash
cargo build -p minimcp --no-default-features
```

## Running the benchmark servers

The Python benchmark harness in `../benchmarks/` is transport-agnostic: it drives any MCP server over stdio or HTTP using an MCP `ClientSession`. The Rust bench server slots in as another server under test.

Build the release binaries first:

```bash
cd rust
cargo build --release -p minimcp-bench-server
```

This produces:

- `target/release/stdio_server`
- `target/release/http_server`

Each binary exposes the same tools the harness expects: `compute_all_prime_factors`, `io_bound_compute_all_prime_factors`, `noop_tool`, and `get_memory_usage`. The HTTP binary honors `TEST_SERVER_HOST` / `TEST_SERVER_PORT` and serves `/mcp`.

### Wiring into the Python harness

Add a `ServerConfig` for the Rust server alongside the existing ones in `benchmarks/macro/*_mcp_server_benchmark.py`. For stdio, launch the compiled binary as a subprocess; for HTTP, start it on the configured port and connect a `ClientSession`. The Rust server then appears as an additional series in the result JSON and `analyze_results.py` output.

## Status / TODO

- [ ] Prompt and resource managers.
- [ ] Streamable HTTP (SSE) transport.
- [ ] Automatic JSON Schema generation from handler types (parity with `MCPFunc` + Pydantic).
- [ ] Wire `TimeLimiter::reset` to the active timeout deadline.
- [ ] Client notification handler dispatch.
- [ ] `ServerConfig` entries in the Python benchmark harness for the Rust server.
