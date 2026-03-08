# Modular RAG MCP Server

一个可插拔、可观测的模块化 RAG 服务框架，支持：

- 离线文档摄取：PDF -> Chunk -> Transform -> Embedding -> Upsert
- 在线检索：Dense + Sparse + RRF + 可选 Rerank
- MCP Tools：可供 Codex / Copilot / Claude 调用
- Dashboard：本地 Streamlit 管理与追踪面板
- Evaluation：golden test set 回归评估

详细设计见 `/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/DEV_SPEC.md`。

## 快速开始

### 1. 安装依赖

建议使用 Python `3.10+`，但当前仓库在本地 `3.9` 环境也已通过主要测试。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install pypdf streamlit
```

如果你要使用真实 LLM / Embedding / Rerank provider，还需要按对应后端补装额外依赖。

### 2. 配置 API Key

默认 `config/settings.yaml` 使用本地可离线运行的组合：

- `embedding.provider = hash`
- `vector_store.provider = chroma`
- `rerank.enabled = false`

如果你要启用真实 OpenAI-compatible LLM：

```bash
export OPENAI_BASE_URL="https://api.modelverse.cn/v1"
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="deepseek-ai/DeepSeek-V3.2"
```

### 3. 首次摄取

```bash
.venv/bin/python scripts/ingest.py \
  --collection demo \
  --path tests/data/test_chunking_multimodal.pdf \
  --force
```

再摄取一份纯文本样例：

```bash
.venv/bin/python scripts/ingest.py \
  --collection demo \
  --path tests/data/test_chunking_text.pdf \
  --force
```

### 4. 查询

```bash
.venv/bin/python scripts/query.py \
  --query "分布式 分片" \
  --collection demo \
  --top-k 3 \
  --verbose \
  --no-rerank
```

### 5. 启动 Dashboard

```bash
.venv/bin/python scripts/start_dashboard.py
```

### 6. 运行评估

```bash
.venv/bin/python scripts/evaluate.py \
  --backend custom \
  --test-set tests/fixtures/golden_test_set.json
```

## 配置说明

当前最常用配置位于 `/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/config/settings.yaml`。

主要字段：

- `llm.provider`
  - 例如 `placeholder` / `openai` / `azure` / `ollama`
- `embedding.provider`
  - 例如 `hash` / `openai` / `azure` / `ollama`
- `vector_store.provider`
  - 当前默认 `chroma`
- `retrieval.top_k`
  - 默认检索条数
- `splitter.provider`
  - 当前默认 `recursive`
- `splitter.chunk_size`
  - 单 chunk 目标大小
- `splitter.chunk_overlap`
  - 相邻 chunk 重叠长度
- `ingestion.chunk_refiner.use_llm`
  - 是否在 chunk 清洗阶段启用 LLM
- `rerank.enabled`
  - 是否启用 rerank
- `evaluation.enabled`
  - 是否启用评估能力
- `observability.log_level`
  - 日志等级

## MCP 配置示例

### GitHub Copilot `mcp.json`

```json
{
  "servers": {
    "modular-rag": {
      "command": "/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/.venv/bin/python",
      "args": [
        "src/mcp_server/server.py"
      ],
      "cwd": "/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER"
    }
  }
}
```

### Claude Desktop `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/.venv/bin/python",
      "args": [
        "src/mcp_server/server.py"
      ],
      "cwd": "/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER"
    }
  }
}
```

当前可用工具：

- `query_knowledge_hub`
- `list_collections`
- `get_document_summary`

## Dashboard 使用指南

启动：

```bash
.venv/bin/python scripts/start_dashboard.py
```

当前页面：

1. `Overview`
   - 展示组件配置、文档数、chunk 数、图片数、集合数
2. `Data Browser`
   - 浏览文档、chunk 内容和图片元数据
3. `Ingestion Manager`
   - 上传 PDF 触发摄取，删除已摄入文档
4. `Ingestion Traces`
   - 查看摄取阶段耗时
5. `Query Traces`
   - 查看查询阶段耗时和 query 过滤
6. `Evaluation Panel`
   - 运行 golden set 评估并查看指标

## 已 ingest 数据放在哪里

默认数据目录：

- 向量库：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/data/db/chroma/store.json`
- BM25：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/data/db/bm25/bm25_index.json`
- 图片索引：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/data/db/image_index.db`
- 完整性记录：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/data/db/ingestion_history.db`
- 图片文件：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/data/images`
- Trace 日志：`/Users/chenglun/Desktop/workspace/MODULAR-RAG-MCP-SERVER/logs/traces.jsonl`

如果要重新 ingest 同一文件：

```bash
.venv/bin/python scripts/ingest.py \
  --collection demo \
  --path tests/data/test_chunking_multimodal.pdf \
  --force
```

## 运行测试

### 单元测试

```bash
.venv/bin/pytest -q tests/unit
```

### 集成测试

```bash
.venv/bin/pytest -q tests/integration
```

### E2E 测试

```bash
.venv/bin/pytest -q tests/e2e
```

### 常用分组

```bash
.venv/bin/pytest -q -s tests/integration/test_mcp_server.py
.venv/bin/pytest -q -s tests/e2e/test_mcp_client.py
.venv/bin/pytest -q -s tests/e2e/test_dashboard_smoke.py
.venv/bin/pytest -q -s tests/e2e/test_recall.py
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'main'`

请在项目根目录运行命令，不要在子目录直接执行测试。

### 2. `PdfLoader requires dependency pypdf`

安装：

```bash
pip install pypdf
```

### 3. `urllib3 v2 only supports OpenSSL 1.1.1+`

这是本机 Python + LibreSSL 组合的环境警告，不影响本项目核心功能。若要彻底消除，请切换到带 OpenSSL 的 Python 解释器。

### 4. Dashboard 无法打开

先确认安装了 `streamlit`：

```bash
pip install streamlit
```

### 5. MCP 连接不上

检查三件事：

1. `cwd` 是否指向项目根目录
2. `command` 是否使用了 `.venv/bin/python`
3. 是否能手动执行：

```bash
.venv/bin/python src/mcp_server/server.py
```

## 许可证

MIT
