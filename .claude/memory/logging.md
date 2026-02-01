# Logging Configuration

Part of the unified configuration system. See `configuration.md` for full details.

## Environment Variables

```bash
NLM_PROXY_LOG_LEVEL=INFO
NLM_PROXY_LOG_FILE=~/.nlm-proxy/logs/nlm-proxy.log
NLM_PROXY_LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
NLM_PROXY_LOG_MAX_SIZE=10485760    # 10 MB
NLM_PROXY_LOG_BACKUP_COUNT=5
```

## Logger Hierarchy

```
nlm_proxy              (root)
├── nlm_proxy.mcp      (MCP server)
├── nlm_proxy.openai   (OpenAI proxy)
├── nlm_proxy.api      (API client)
└── nlm_proxy.session  (Session store)
```

## Usage

```bash
# Console only with debug
nlm-proxy serve mcp --debug

# Or via environment
export NLM_PROXY_LOG_LEVEL=DEBUG
nlm-proxy serve mcp
```

## Log Rotation

- Files rotate at `NLM_PROXY_LOG_MAX_SIZE`
- Keeps `NLM_PROXY_LOG_BACKUP_COUNT` backups
- Old backups deleted automatically
