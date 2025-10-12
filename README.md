[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/shemhamforash23-lightrag-mcp-badge.png)](https://mseep.ai/app/shemhamforash23-lightrag-mcp)

# LightRAG MCP Server

**Fully functional LightRAG MCP integration with Docker containerization and remote server support.**

A Model Context Protocol (MCP) server that provides seamless integration between LightRAG (Retrieval-Augmented Generation) API and MCP-compatible AI tools. This server enables AI assistants to interact with LightRAG's knowledge graph and document processing capabilities through standardized MCP tools.

## Description

LightRAG MCP Server acts as a bridge between LightRAG API v0237 and MCP-compatible clients, allowing AI tools to leverage LightRAG's advanced retrieval and knowledge graph capabilities. The server is fully containerized with Docker and supports remote LightRAG server connections.

### Key Features

- **🔍 Semantic Document Queries**: Execute intelligent queries across indexed documents using multiple retrieval modes (local, global, hybrid, naive, mix, bypass)
- **📄 Document Management**: Upload, index, and track document processing status
- **🕸️ Knowledge Graph Operations**: Create, edit, and manage entities and relationships in the knowledge graph
- **📊 Monitoring & Health Checks**: Real-time status monitoring of LightRAG API and document processing pipelines
- **🐳 Docker Containerization**: Complete Docker setup with docker-compose for easy deployment
- **🌐 Remote Server Support**: Connect to remote LightRAG servers (tested with 192.168.2.16:9621)
- **🔧 MCP Protocol Compliance**: Full MCP v1.0 compatibility with stdio transport

### Recent Updates & Fixes

**✅ API Compatibility**: Fully compatible with LightRAG Server API v0237 (core v1.4.9.2)
**✅ Authentication**: Proper X-API-Key header authentication implementation
**✅ Response Formatting**: Correct handling of query responses with references
**✅ Docker Integration**: Complete containerization with health checks and networking
**✅ Remote Server**: Tested and verified with remote LightRAG server configurations

## Installation

### Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **LightRAG Server**: A running LightRAG API server (local or remote)
- **MCP Client**: An MCP-compatible AI tool (Claude Desktop, VSCode extensions, etc.)

### Quick Start with Docker

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shemhamforash23/lightrag-mcp.git
   cd lightrag-mcp
   ```

2. **Configure environment**:
   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env with your LightRAG server details
   # For remote server (192.168.2.16:9621):
   LIGHTRAG_API_HOST=192.168.2.16
   LIGHTRAG_API_PORT=9621
   LIGHTRAG_API_KEY=no-key
   ```

3. **Build and run with Docker Compose**:
   ```bash
   # Build the container
   docker-compose build

   # Start the MCP server
   docker-compose up -d

   # Check logs
   docker-compose logs -f lightrag-mcp
   ```

### Alternative: Direct Installation

If you prefer not to use Docker:

```bash
# Install with uv (recommended)
uv venv --python 3.11
uv pip install -e .

# Or with pip
pip install -e .
```

### MCP Client Configuration

#### For Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lightrag-mcp": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network",
        "host",
        "-e",
        "LIGHTRAG_API_HOST=192.168.2.16",
        "-e",
        "LIGHTRAG_API_PORT=9621",
        "-e",
        "LIGHTRAG_API_KEY=no-key",
        "lightrag-mcp:latest"
      ]
    }
  }
}
```

#### For VSCode Kilo Code Extension

Update your MCP settings in VSCode:

```json
{
  "mcpServers": {
    "lightrag-mcp": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network",
        "host",
        "-e",
        "LIGHTRAG_API_HOST=192.168.2.16",
        "-e",
        "LIGHTRAG_API_PORT=9621",
        "-e",
        "LIGHTRAG_API_KEY=no-key",
        "lightrag-mcp:latest"
      ]
    }
  }
}
```

#### Development Mode (without Docker)

```json
{
  "mcpServers": {
    "lightrag-mcp": {
      "command": "uvx",
      "args": [
        "lightrag-mcp",
        "--host",
        "192.168.2.16",
        "--port",
        "9621",
        "--api-key",
        "no-key"
      ]
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_API_HOST` | `host.docker.internal` | LightRAG server hostname/IP |
| `LIGHTRAG_API_PORT` | `9621` | LightRAG server port |
| `LIGHTRAG_API_KEY` | `no-key` | API authentication key |

### Docker Networking

For remote LightRAG servers, use `network_mode: host` in docker-compose.yml:

```yaml
services:
  lightrag-mcp:
    # ... other config
    network_mode: host
    environment:
      - LIGHTRAG_API_HOST=192.168.2.16
      - LIGHTRAG_API_PORT=9621
```

## Usage Examples

### Basic Query Operations

```python
# Query documents with semantic search
query_result = await query_document(
    query="artificial intelligence applications",
    mode="mix",
    response_type="Multiple Paragraphs",
    top_k=10
)

# Insert text documents
insert_result = await insert_document(
    text="Your document content here...",
    file_source="example.txt"
)

# Upload files for processing
upload_result = await upload_document(
    file_path="/path/to/document.pdf"
)
```

### Docker Container Usage

```bash
# Run one-off commands
docker run --rm -i \
  --network host \
  -e LIGHTRAG_API_HOST=192.168.2.16 \
  -e LIGHTRAG_API_PORT=9621 \
  -e LIGHTRAG_API_KEY=no-key \
  lightrag-mcp:latest \
  --help

# Check container health
docker-compose ps
docker-compose exec lightrag-mcp python -c "print('Container is healthy')"
```

### Knowledge Graph Operations

```python
# Create entities
await create_entities(entities=[
    {
        "entity_name": "Python",
        "entity_type": "PROGRAMMING_LANGUAGE",
        "description": "High-level programming language",
        "source_id": "doc123"
    }
])

# Create relationships
await create_relations(relations=[
    {
        "source": "Python",
        "target": "Django",
        "description": "Python framework for web development",
        "keywords": "web, framework",
        "weight": 0.9
    }
])
```

## Available MCP Tools

### Document Queries
- `query_document`: Execute semantic queries with multiple modes and response formats

### Document Management
- `insert_document`: Add text directly to LightRAG storage
- `upload_document`: Upload files to the /inputs directory for processing
- `insert_file`: Add files directly to storage
- `insert_batch`: Batch insert multiple documents from directories
- `scan_for_new_documents`: Trigger scanning of input directory
- `get_documents`: List all uploaded documents
- `get_pipeline_status`: Monitor document processing pipeline

### Knowledge Graph Operations
- `get_graph_labels`: Retrieve node and relationship types
- `create_entities`: Create multiple entities in parallel
- `edit_entities`: Update existing entities
- `delete_by_entities`: Remove entities by name
- `delete_by_doc_ids`: Delete entities by document ID
- `create_relations`: Create entity relationships
- `edit_relations`: Update relationships
- `merge_entities`: Merge entities with relationship migration

### Monitoring
- `check_lightrag_health`: Verify API connectivity and status

## Troubleshooting

### Common Issues

#### 1. Connection Refused / Cannot Connect to LightRAG Server

**Symptoms**: `Connection refused` or timeout errors

**Solutions**:
- Verify LightRAG server is running: `curl http://192.168.2.16:9621/health`
- Check network connectivity between containers
- For Docker: Use `network_mode: host` or `--network host`
- Verify firewall settings allow port 9621

#### 2. Authentication Errors

**Symptoms**: `401 Unauthorized` or API key errors

**Solutions**:
- Ensure `LIGHTRAG_API_KEY` is set correctly (default: "no-key")
- Check that X-API-Key header is being sent
- Verify API key format (no special characters)

#### 3. Docker Networking Issues

**Symptoms**: Container cannot reach remote LightRAG server

**Solutions**:
```bash
# Use host networking
docker run --network host ...

# Or in docker-compose.yml
services:
  lightrag-mcp:
    network_mode: host

# For Mac/Windows, use host.docker.internal
environment:
  - LIGHTRAG_API_HOST=host.docker.internal
```

#### 4. MCP Timeout Issues

**Symptoms**: Operations timeout after 30 seconds

**Solutions**:
- Increase timeout in client configuration
- Check LightRAG server performance
- Verify network latency to remote server

#### 5. Query Returns "None" or Empty Response

**Symptoms**: Queries return null responses

**Solutions**:
- Verify API compatibility (requires LightRAG v0237+)
- Check query length (minimum 3 characters)
- Ensure documents are properly indexed
- Test with simple queries first

### Docker Troubleshooting Commands

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs lightrag-mcp

# Restart service
docker-compose restart lightrag-mcp

# Rebuild container
docker-compose build --no-cache lightrag-mcp

# Test connectivity from container
docker-compose exec lightrag-mcp curl -f http://192.168.2.16:9621/health
```

### Health Check Verification

```bash
# Test LightRAG server directly
curl -H "X-API-Key: no-key" http://192.168.2.16:9621/health

# Test from MCP container
docker-compose exec lightrag-mcp python -c "
import httpx
response = httpx.get('http://192.168.2.16:9621/health', headers={'X-API-Key': 'no-key'})
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')
"
```

### Performance Optimization

- **Memory**: Container limited to 512MB, increase if needed
- **Network**: Use host networking for better performance
- **Timeout**: Default 30s, adjust based on server response time
- **Batch Operations**: Use bulk operations for multiple entities/relations

## Development

### Installing Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Running Linters

```bash
ruff check src/
mypy src/
```

### Testing the Integration

```bash
# Run the server directly for testing
uvx lightrag-mcp --host 192.168.2.16 --port 9621 --api-key no-key

# Test with sample queries
python -c "
import asyncio
from lightrag_mcp.lightrag_client import LightRAGClient

async def test():
    client = LightRAGClient('http://192.168.2.16:9621', 'no-key')
    result = await client.query(query='test query', mode='mix')
    print('Query result:', result)
    await client.close()

asyncio.run(test())
"
```

## API Compatibility

- **LightRAG Server**: v0237 (core v1.4.9.2) ✅
- **MCP Protocol**: v1.0 ✅
- **Python**: 3.11+ ✅
- **Docker**: 20.10+ ✅

## License

MIT
