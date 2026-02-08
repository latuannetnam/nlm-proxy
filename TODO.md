# Project Roadmap & Todo

## 🚧 In Progress
- [ ] **MS Teams chatbot integration**
- [ ] **Agentic Workflow**: Switch to Agent library (langchain/langgraph) for more complex tasks
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
