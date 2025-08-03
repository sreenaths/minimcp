# MiniMCP

A minimal, stateless, and lightweight MCP server designed for easy integration into any Python application.

## Development
### Setup
```
uv init --python=python3.10
uv pip install --dev -e .
```

### Build & Publish
```
python -m build
twine check dist/*
twine upload dist/*
```

### Running demo app

```
uvicorn demo.main:app --reload
```
