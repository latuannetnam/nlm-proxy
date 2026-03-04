# Project Roadmap & Todo

## 🚧 In Progress
- [Doing] **Agentic Workflow**: Switch to Agent library (langchain/langgraph) for more complex tasks: Memory management, Response caching, Cross-notebook query support: docs\plans\2026-03-03-langchain-refactor-design.md  
- [] Fix all failed, skipped tests
- [] Refactor client.py: file too large
- [] Add support for Cloud-based embedding model (for example OpenAI, Gemini, Cohere, etc): local model work good enough for now
- [ ] Enhance conversation management
- [ ] **Scalability**: Loadbalancing request to multiple NotebookLM account (each Pro account has maximum 500 chat request/day): docs\plans\nlm-proxy-account-pool-specification.md
- [ ] **Pre-processing**: Use LLM task to pre-process in individual notebook models
- [ ] **Authentication**: Support multiple authentication profiles
- [ ] **Async Session Management**: Review FastAPI support for SessionStore and Notebook cache
- [ ] **Smart Router**: Cross-notebook query support
  - Status: Pending (`feature/cross-notebook-query`)
- [ ] **Citations**: Return NotebookLM citation with source links
  - Status: Pending (`feature/openai-proxy-citation`) - Current logic needs improvement (NotebookLM may use internal API)

## ✅ Completed

- [x] **Observability**: Track notebook query -> NotebookLM query analysis (question, answer, notebook, quality metrics)
- [x] **Smart Router**: Enhance NotebookLM selection using metadata (source title + description)
- [x] **Documentation**: Restructure CLAUDE.md with component sections and external references
- [x] **Configuration**: Settings via environment variables and command line arguments
- [x] **Security**: OpenAI endpoint authentication
- [x] **Smart Router**: Basic LLM question routing implementation
- [x] **MS Teams chatbot integration**: docs\plans\notebooklm-chatbot-design.md
- [x] ** Notebook ACL**: Limit notebooks to query based on ACL (for example AD group): docs\plans\per-request-acl-specification.md
- [x] **Performance**Response Cache: docs\plans\2026-03-02-response-cache-implementation.md
- [x] **Monitoring** Montior Response Cache metrics: via API /v1/cache/stats and script scripts\cache-stats.ps1 + Cache Log Analyzer: scripts\cache-log-analyzer.py
- [x] Migrate Claude memory to Antigravity
