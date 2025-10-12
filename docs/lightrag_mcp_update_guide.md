# LightRAG MCP Fix Specification

## Executive Summary

**Issue**: The MCP client is using outdated API endpoint paths that return 404 errors, causing all queries to fail with `None` responses.

**Root Cause**: MCP client code uses incorrect endpoints like `/tool_query_document_post` instead of the correct `/query` endpoint. The LightRAG server API (version 0237, core v1.4.9.2) has different endpoint paths than what the MCP client expects.

**Impact**: All MCP tool operations fail, preventing integration with LightRAG knowledge base.

**Solution**: Update MCP client to use correct API endpoints with proper authentication and request/response formats.

## Confirmed API Endpoint Mappings

| Operation | Old Endpoint (Broken) | New Endpoint (Working) | Method |
|-----------|----------------------|----------------------|--------|
| Query Document | `/tool_query_document_post` | `/query` | POST |
| Insert Text | `/tool_insert_text_post` | `/documents/text` | POST |
| Upload File | `/tool_insert_file_post` | `/documents/upload` | POST |
| Insert Batch | `/tool_insert_batch_post` | `/documents/texts` | POST |
| Scan Documents | N/A | `/documents/scan` | POST |
| Health Check | N/A | `/health` | GET |

## Step-by-Step Fix Implementation

### 1. Update Client Initialization

Add proper authentication headers:

```python
class LightRAGClient:
    def __init__(self, host: str = "localhost", port: int = 9621, api_key: str = "no-key"):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        self.client = httpx.AsyncClient(timeout=30.0)
```

### 2. Fix Query Document Method

```python
async def query_document(self, query: str, mode: str = "mix", include_references: bool = True) -> dict:
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long")

    payload = {
        "query": query,
        "mode": mode,
        "include_references": include_references,
        "stream": False
    }

    response = await self.client.post(
        f"{self.base_url}/query",  # ✅ CORRECT
        headers=self.headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")

    return response.json()
```

### 3. Fix Insert Text Method

```python
async def insert_text(self, text: str, file_source: str = "") -> dict:
    if not text:
        raise ValueError("Text cannot be empty")

    payload = {"text": text, "file_source": file_source}

    response = await self.client.post(
        f"{self.base_url}/documents/text",  # ✅ CORRECT
        headers=self.headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"Insert failed: {response.text}")

    return response.json()
```

### 4. Fix File Upload Method

```python
async def upload_file(self, file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        headers = {"X-API-Key": self.api_key}  # No Content-Type for file uploads

        response = await self.client.post(
            f"{self.base_url}/documents/upload",  # ✅ CORRECT
            headers=headers,
            files=files
        )

    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")

    return response.json()
```

### 5. Fix Batch Insert Method

```python
async def insert_batch(self, texts: list[str], file_sources: list[str] = None) -> dict:
    if not texts:
        raise ValueError("Must provide at least one text")

    payload = {"texts": texts}
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

### 6. Update MCP Tool Schema

```python
QUERY_TOOL_SCHEMA = {
    "name": "query_document",
    "description": "Query the LightRAG knowledge base",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 3},
            "mode": {"type": "string", "enum": ["local", "global", "hybrid", "naive", "mix", "bypass"], "default": "mix"},
            "include_references": {"type": "boolean", "default": True}
        },
        "required": ["query"]
    }
}
```

## Response Format Changes

### Query Response
```python
{
    "response": "Generated answer text...",
    "references": [
        {"reference_id": "1", "file_path": "/path/to/document.pdf"}
    ]
}
```

### Insert Response
```python
{
    "status": "success",
    "message": "Operation completed successfully",
    "track_id": "insert_20251011_123456_abc"
}
```

## Testing Procedures

### Validation Script

Create `test_mcp_fix.py`:

```python
import asyncio
from lightrag_mcp.client import LightRAGClient

async def test_fixes():
    client = LightRAGClient(host="192.168.2.16", port=9621, api_key="no-key")

    try:
        # Test 1: Health check
        health = await client.health_check()
        print(f"✅ Health: {health.get('status')}")

        # Test 2: Insert text
        result = await client.insert_text("Test document for MCP fix validation.")
        print(f"✅ Insert: {result['status']} - Track ID: {result['track_id']}")

        # Test 3: Query
        query_result = await client.query_document("test document", include_references=True)
        print(f"✅ Query: {query_result['response'][:50]}...")
        if query_result.get('references'):
            print(f"   References: {len(query_result['references'])} found")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_fixes())
```

### Validation Steps

1. **Run health check**: Verify server connectivity
2. **Test text insertion**: Confirm insert operations work
3. **Test query**: Verify queries return actual responses (not `None`)
4. **Test file upload**: Ensure file operations work
5. **Test MCP tools**: Verify all 22 tools are functional

### Expected Results

```
✅ Health: healthy
✅ Insert: success - Track ID: insert_20251011_123456_abc
✅ Query: Test document refers to a sample text used for...
   References: 1 found
```

## Troubleshooting Guide

### Issue: Still getting 404 errors

**Cause**: Endpoints not updated correctly
**Solution**: Double-check all endpoint URLs match the mapping table exactly

### Issue: Authentication failures

**Cause**: Missing or incorrect API key
**Solution**: Ensure `X-API-Key` header is set correctly

### Issue: Query returns `None`

**Cause**: Using old response format expectations
**Solution**: Update code to access `result['response']` instead of `result['response']`

### Issue: File uploads fail

**Cause**: Wrong Content-Type header for multipart uploads
**Solution**: Remove `Content-Type` from headers when using `files` parameter

### Issue: Connection timeouts

**Cause**: Server not accessible or wrong host/port
**Solution**: Verify server is running and accessible at specified host:port

### Issue: Minimum query length errors

**Cause**: Queries shorter than 3 characters
**Solution**: Add validation to ensure queries meet minimum length requirement

### Common Pitfalls

1. **Stream parameter**: Always set `stream: false` for `/query` endpoint
2. **File uploads**: Use different headers (no JSON Content-Type)
3. **Query validation**: Minimum 3 characters required
4. **Response structure**: New format uses `response` and `references` keys
5. **Authentication**: `X-API-Key` header required for all requests

## Files to Update

- `src/lightrag_mcp/client.py`: Update all endpoint URLs and request formats
- `src/lightrag_mcp/tools.py`: Update tool schemas and response handling
- `src/lightrag_mcp/main.py`: Update MCP tool handlers
- `README.md`: Update usage examples

## Version Compatibility

- **API Version**: 0237
- **Core Version**: v1.4.9.2
- **Tested with**: LightRAG server at 192.168.2.16:9621

## Support

For issues:
- Check `/health` endpoint for server status
- Verify endpoints with direct curl requests
- Compare with OpenAPI spec at `/docs` endpoint