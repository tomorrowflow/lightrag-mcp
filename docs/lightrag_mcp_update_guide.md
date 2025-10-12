# LightRAG MCP API Compatibility Update Guide

## Overview

This guide provides precise instructions for updating the `lightrag-mcp` fork to be compatible with LightRAG Server API version **0237** (core v1.4.9.2).

**Current Issue**: The MCP client is using incorrect endpoint paths that don't exist in the API, causing all queries to return `None`.

---

## 1. API Endpoint Mapping Changes

### Critical Path Updates

Update all API endpoint paths in your client code (likely in `src/lightrag_mcp/client.py` or similar):

| Current (Broken) | Correct Endpoint | HTTP Method |
|-----------------|------------------|-------------|
| `/tool_query_document_post` | `/query` | POST |
| `/tool_insert_text_post` | `/documents/text` | POST |
| `/tool_insert_file_post` | `/documents/upload` | POST |
| `/tool_insert_batch_post` | `/documents/texts` | POST |
| Unknown | `/documents/scan` | POST |
| Unknown | `/health` | GET |

---

## 2. Query Tool Updates

### File: `src/lightrag_mcp/client.py` (or equivalent)

#### Current Implementation (Broken):
```python
async def query_document(self, query: str, mode: str = "mix"):
    response = await self.client.post(
        f"{self.base_url}/tool_query_document_post",  # ❌ WRONG
        json={"query": query, "mode": mode}
    )
    return response.json()
```

#### Updated Implementation:
```python
async def query_document(
    self, 
    query: str, 
    mode: str = "mix",
    include_references: bool = True,
    stream: bool = False,
    **kwargs
) -> dict:
    """
    Query the LightRAG knowledge base.
    
    Args:
        query: The question to ask (min 3 characters)
        mode: Query mode - one of: local, global, hybrid, naive, mix, bypass
        include_references: Whether to include source references
        stream: Enable streaming (only for /query/stream endpoint)
        **kwargs: Additional parameters (response_type, top_k, chunk_top_k, etc.)
    
    Returns:
        dict with keys: 'response' (str) and optionally 'references' (list)
    """
    # Validate query length
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    # Build request payload
    payload = {
        "query": query,
        "mode": mode,
        "include_references": include_references,
        "stream": False  # Always false for /query endpoint
    }
    
    # Add optional parameters
    if "response_type" in kwargs:
        payload["response_type"] = kwargs["response_type"]
    if "top_k" in kwargs:
        payload["top_k"] = kwargs["top_k"]
    if "chunk_top_k" in kwargs:
        payload["chunk_top_k"] = kwargs["chunk_top_k"]
    if "max_total_tokens" in kwargs:
        payload["max_total_tokens"] = kwargs["max_total_tokens"]
    if "conversation_history" in kwargs:
        payload["conversation_history"] = kwargs["conversation_history"]
    
    response = await self.client.post(
        f"{self.base_url}/query",  # ✅ CORRECT
        headers=self.headers,
        json=payload
    )
    
    # Handle response
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")
    
    return response.json()
```

### MCP Tool Definition Update

Update the tool schema in your MCP server (likely in `src/lightrag_mcp/tools.py` or `main.py`):

```python
QUERY_TOOL_SCHEMA = {
    "name": "query_document",
    "description": "Query the LightRAG knowledge base using various retrieval modes",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or search query (minimum 3 characters)",
                "minLength": 3
            },
            "mode": {
                "type": "string",
                "enum": ["local", "global", "hybrid", "naive", "mix", "bypass"],
                "default": "mix",
                "description": "Query mode: local (entity-focused), global (pattern analysis), hybrid (combined), naive (vector only), mix (recommended - graph + vector), bypass (direct LLM)"
            },
            "include_references": {
                "type": "boolean",
                "default": True,
                "description": "Include source document references in response"
            },
            "response_type": {
                "type": "string",
                "description": "Desired response format (e.g., 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points')"
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of top entities/relationships to retrieve"
            },
            "chunk_top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of text chunks to retrieve"
            }
        },
        "required": ["query"]
    }
}
```

---

## 3. Insert Document Updates

### Insert Text Implementation

```python
async def insert_text(
    self, 
    text: str, 
    file_source: str = ""
) -> dict:
    """
    Insert text directly into LightRAG storage.
    
    Args:
        text: The text content to insert (minimum 1 character)
        file_source: Optional source identifier for the text
    
    Returns:
        dict with keys: 'status', 'message', 'track_id'
    """
    if len(text) < 1:
        raise ValueError("Text must not be empty")
    
    payload = {
        "text": text,
        "file_source": file_source
    }
    
    response = await self.client.post(
        f"{self.base_url}/documents/text",  # ✅ CORRECT (was /tool_insert_text_post)
        headers=self.headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Insert failed: {response.text}")
    
    result = response.json()
    # Expected keys: status, message, track_id
    return result
```

### Insert Batch Implementation

```python
async def insert_batch(
    self, 
    texts: list[str], 
    file_sources: list[str] = None
) -> dict:
    """
    Insert multiple texts into LightRAG storage.
    
    Args:
        texts: List of text contents (minimum 1 text)
        file_sources: Optional list of source identifiers
    
    Returns:
        dict with keys: 'status', 'message', 'track_id'
    """
    if not texts or len(texts) < 1:
        raise ValueError("Must provide at least one text")
    
    payload = {
        "texts": texts
    }
    
    if file_sources:
        payload["file_sources"] = file_sources
    
    response = await self.client.post(
        f"{self.base_url}/documents/texts",  # ✅ CORRECT
        headers=self.headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Batch insert failed: {response.text}")
    
    return response.json()
```

### Upload File Implementation

```python
async def upload_file(self, file_path: str) -> dict:
    """
    Upload a file to LightRAG input directory for processing.
    
    Args:
        file_path: Path to the file to upload
    
    Returns:
        dict with keys: 'status', 'message', 'track_id'
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        
        # Note: Don't use JSON content-type for file uploads
        headers = {"X-API-Key": self.api_key}
        
        response = await self.client.post(
            f"{self.base_url}/documents/upload",  # ✅ CORRECT
            headers=headers,
            files=files
        )
    
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")
    
    return response.json()
```

---

## 4. Authentication Updates

### Header Configuration

Update your client initialization to properly set authentication:

```python
class LightRAGClient:
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 9621, 
        api_key: str = "no-key"
    ):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        
        # Headers for all requests
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        
        # Initialize HTTP client
        import httpx
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
```

---

## 5. Response Handling Updates

### Query Response Structure

The API returns:
```python
{
    "response": "The generated answer text...",
    "references": [  # Only if include_references=True
        {
            "reference_id": "1",
            "file_path": "/path/to/document.pdf"
        },
        {
            "reference_id": "2", 
            "file_path": "/path/to/another.txt"
        }
    ]
}
```

### Insert Response Structure

The API returns:
```python
{
    "status": "success",  # or "duplicated", "partial_success", "failure"
    "message": "File uploaded successfully. Processing will continue in background.",
    "track_id": "upload_20250729_170612_abc123"
}
```

### Handle these properly in your MCP tool responses:

```python
@self.server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list:
    """Handle MCP tool calls"""
    
    if name == "query_document":
        result = await client.query_document(**arguments)
        
        # Format response for MCP
        response_text = result.get("response", "No response")
        
        # Include references if present
        if "references" in result and result["references"]:
            refs = "\n\n### References\n"
            for ref in result["references"]:
                refs += f"- [{ref['reference_id']}] {ref['file_path']}\n"
            response_text += refs
        
        return [{"type": "text", "text": response_text}]
    
    elif name == "insert_document":
        result = await client.insert_text(**arguments)
        
        # Format response
        response_text = f"Status: {result['status']}\n{result['message']}\nTrack ID: {result['track_id']}"
        
        return [{"type": "text", "text": response_text}]
```

---

## 6. Additional API Endpoints to Implement

### Health Check
```python
async def health_check(self) -> dict:
    """Check LightRAG server health and configuration"""
    response = await self.client.get(
        f"{self.base_url}/health",
        headers=self.headers
    )
    return response.json()
```

### Scan for New Documents
```python
async def scan_documents(self) -> dict:
    """Trigger scanning of input directory for new documents"""
    response = await self.client.post(
        f"{self.base_url}/documents/scan",
        headers=self.headers,
        json={}
    )
    return response.json()
```

### Get Pipeline Status
```python
async def get_pipeline_status(self) -> dict:
    """Get current document processing pipeline status"""
    response = await self.client.get(
        f"{self.base_url}/documents/pipeline_status",
        headers=self.headers
    )
    return response.json()
```

### Track Document Status
```python
async def get_track_status(self, track_id: str) -> dict:
    """Get processing status of documents by tracking ID"""
    response = await self.client.get(
        f"{self.base_url}/documents/track_status/{track_id}",
        headers=self.headers
    )
    return response.json()
```

---

## 7. Testing Your Changes

### Test Script

Create a test file `test_client.py`:

```python
import asyncio
from lightrag_mcp.client import LightRAGClient

async def test_connection():
    client = LightRAGClient(host="192.168.2.16", port=9621, api_key="no-key")
    
    try:
        # Test 1: Health check
        print("Testing health check...")
        health = await client.health_check()
        print(f"✅ Health: {health.get('status')}")
        
        # Test 2: Insert text
        print("\nTesting text insert...")
        insert_result = await client.insert_text(
            text="This is a test document about artificial intelligence and machine learning.",
            file_source="test_script"
        )
        print(f"✅ Insert: {insert_result['status']} - Track ID: {insert_result['track_id']}")
        
        # Test 3: Query
        print("\nTesting query...")
        query_result = await client.query_document(
            query="artificial intelligence",
            mode="mix",
            include_references=True
        )
        print(f"✅ Query response: {query_result['response'][:100]}...")
        if query_result.get('references'):
            print(f"   References: {len(query_result['references'])} sources")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())
```

Run with:
```bash
python test_client.py
```

### Expected Output
```
Testing health check...
✅ Health: healthy

Testing text insert...
✅ Insert: success - Track ID: insert_20251011_123456_abc

Testing query...
✅ Query response: Artificial intelligence (AI) is a branch of computer science focused on creating inte...
   References: 1 sources
```

---

## 8. Key Files to Update

Based on typical MCP server structure, update these files:

### `src/lightrag_mcp/client.py`
- Update all endpoint URLs
- Fix request/response structures
- Add proper authentication headers

### `src/lightrag_mcp/tools.py` (if exists)
- Update tool schemas with correct parameters
- Fix response formatting

### `src/lightrag_mcp/main.py`
- Update MCP tool handlers
- Ensure proper error handling

### `README.md`
- Document the API version compatibility
- Update usage examples with correct parameters

---

## 9. Common Pitfalls to Avoid

1. **Don't forget `stream: false`** in query requests when using `/query` endpoint
2. **File uploads** require different headers (no JSON content-type)
3. **Minimum query length** is 3 characters - validate this
4. **API key** can be passed as header `X-API-Key` OR query param `api_key_header_value`
5. **Response structure changed** - old versions may have had `{status: success, response: None}`, new version has `{response: "text", references: [...]}`

---

## 10. Version Compatibility Check

Add version checking to your client:

```python
async def check_compatibility(self):
    """Check if connected to compatible LightRAG API version"""
    health = await self.health_check()
    
    api_version = health.get("api_version")
    core_version = health.get("core_version")
    
    # Check for compatibility
    if api_version != "0237":
        print(f"⚠️  Warning: API version {api_version} may not be fully compatible")
    
    if core_version and not core_version.startswith("v1.4"):
        print(f"⚠️  Warning: Core version {core_version} may not be fully compatible")
    
    print(f"Connected to LightRAG API v{api_version}, Core {core_version}")
```

---

## Summary of Critical Changes

| Component | Old Behavior | New Behavior |
|-----------|-------------|--------------|
| Query endpoint | `/tool_query_document_post` | `/query` |
| Insert endpoint | `/tool_insert_text_post` | `/documents/text` |
| Response format | `{status, response}` | `{response, references}` |
| Authentication | May have been missing | `X-API-Key` header required |
| Query parameters | Unknown | `query`, `mode`, `include_references`, `stream` |
| Insert parameters | Unknown | `text`, `file_source` |

---

## Testing Checklist

- [ ] Health check works
- [ ] Insert text returns track_id
- [ ] Query returns actual response (not "None")
- [ ] References are included when requested
- [ ] File upload works
- [ ] Batch insert works
- [ ] Error handling works correctly
- [ ] MCP tools are properly registered
- [ ] All 22 tools from README are functional

---

## 11. Docker Deployment (Optional)

If your fork includes Docker support, here's how to containerize the lightrag-mcp server:

### Dockerfile Example

Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy project files
COPY . /app

# Install dependencies
RUN uv pip install --system -e .

# Expose any needed ports (if running HTTP server)
# EXPOSE 8080

# Set default environment variables
ENV LIGHTRAG_HOST=host.docker.internal
ENV LIGHTRAG_PORT=9621
ENV LIGHTRAG_API_KEY=no-key

# Run the MCP server
ENTRYPOINT ["python", "-m", "lightrag_mcp.main"]
CMD ["--host", "${LIGHTRAG_HOST}", "--port", "${LIGHTRAG_PORT}", "--api-key", "${LIGHTRAG_API_KEY}"]
```

### Docker Compose Setup

Create `docker-compose.yml` for running both LightRAG server and MCP together:

```yaml
version: '3.8'

services:
  lightrag-server:
    image: lightrag/server:latest
    container_name: lightrag-server
    ports:
      - "9621:9621"
    volumes:
      - ./rag_storage:/app/data/rag_storage
      - ./inputs:/app/data/inputs
    environment:
      - LLM_BINDING=ollama
      - LLM_BINDING_HOST=http://ollama:11434/
      - LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
      - EMBEDDING_BINDING=ollama
      - EMBEDDING_BINDING_HOST=http://ollama:11434
      - EMBEDDING_MODEL=bge-m3:latest
    networks:
      - lightrag-network

  lightrag-mcp:
    build: .
    container_name: lightrag-mcp
    depends_on:
      - lightrag-server
    environment:
      - LIGHTRAG_HOST=lightrag-server
      - LIGHTRAG_PORT=9621
      - LIGHTRAG_API_KEY=no-key
    networks:
      - lightrag-network
    # For stdio MCP, you may need to configure volume mounts
    # to share with the host MCP client
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - lightrag-network

networks:
  lightrag-network:
    driver: bridge

volumes:
  ollama_data:
```

### Building and Running with Docker

```bash
# Build the MCP server image
docker build -t lightrag-mcp:latest .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f lightrag-mcp

# Stop services
docker-compose down
```

### Using Docker MCP Server with Claude Desktop

Update your `claude_desktop_config.json` to use the containerized version:

```json
{
  "mcpServers": {
    "lightrag-docker": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network", "host",
        "-e", "LIGHTRAG_HOST=localhost",
        "-e", "LIGHTRAG_PORT=9621",
        "-e", "LIGHTRAG_API_KEY=no-key",
        "lightrag-mcp:latest"
      ]
    }
  }
}
```

### Remote Server Configuration

For connecting to a remote LightRAG server (like your 192.168.2.16 setup):

```yaml
# docker-compose.remote.yml
version: '3.8'

services:
  lightrag-mcp:
    build: .
    container_name: lightrag-mcp-remote
    environment:
      - LIGHTRAG_HOST=192.168.2.16
      - LIGHTRAG_PORT=9621
      - LIGHTRAG_API_KEY=no-key
    network_mode: host
```

Run with:
```bash
docker-compose -f docker-compose.remote.yml up -d
```

### Health Check in Docker

Add health check to your Docker setup:

```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://${LIGHTRAG_HOST}:${LIGHTRAG_PORT}/health')" || exit 1
```

### Docker Environment Variables

All configurable options as environment variables:

```bash
# In .env file or docker-compose
LIGHTRAG_HOST=192.168.2.16
LIGHTRAG_PORT=9621
LIGHTRAG_API_KEY=no-key
LOG_LEVEL=INFO
TIMEOUT=30
```

### Troubleshooting Docker Setup

**Issue: Cannot connect to LightRAG server from container**
```bash
# Use host.docker.internal on Mac/Windows
LIGHTRAG_HOST=host.docker.internal

# Use network_mode: host on Linux
network_mode: host

# Or use the container name if in same docker network
LIGHTRAG_HOST=lightrag-server
```

**Issue: MCP stdio communication fails**
```bash
# Ensure proper stdio handling in Dockerfile
# Use unbuffered Python output
ENV PYTHONUNBUFFERED=1
```

---

## Support

For issues specific to this fork:
- Check the OpenAPI spec at `http://192.168.2.16:9621/openapi.json`
- Verify server version at `http://192.168.2.16:9621/health`
- Test endpoints directly with curl before updating MCP code

API Documentation: Your server provides `/docs` endpoint for interactive API testing.

### Additional Resources

- **LightRAG Documentation**: [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- **MCP Protocol**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Docker MCP Catalog**: [https://hub.docker.com/mcp](https://hub.docker.com/mcp)