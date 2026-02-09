# NotebookLM Chatbot Design Plan

> **Purpose**: Detailed design document for building a multi-channel chatbot (MS Teams, Telegram) that integrates with nlm-proxy to query NotebookLM notebooks with Azure AD authentication.
>
> **Target Repo**: Separate repository (not nlm-proxy)
>
> **Last Updated**: 2025-02-08
>
> **Related Documents**:
> - [Azure App Registration Guide](./azure-app-registration-guide.md) - Step-by-step setup for Azure AD apps
> - [Per-Request ACL Specification](./per-request-acl-specification.md) - Required nlm-proxy changes for ACL support
> - [nlm-proxy Account Pool Specification](./nlm-proxy-account-pool-specification.md) - Multi-account load balancing for nlm-proxy

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Component Design](#component-design)
5. [Authentication & Authorization](#authentication--authorization)
6. [Channel Implementations](#channel-implementations)
7. [nlm-proxy Integration](#nlm-proxy-integration)
8. [Access Control (ACL)](#access-control-acl)
9. [Error Handling & Resilience](#error-handling--resilience)
10. [Deployment](#deployment)
11. [Configuration Reference](#configuration-reference)
12. [Implementation Phases](#implementation-phases)
13. [Testing Strategy](#testing-strategy)
14. [Monitoring & Observability](#monitoring--observability)

---

## Overview

### Goals

- Build a chatbot that serves **50-500 concurrent users**
- Support **MS Teams** (primary) and **Telegram** (secondary) channels
- Integrate with **Azure AD** for corporate user authentication
- Use **nlm-proxy smart routing** to automatically select the best notebook
- Implement **simple ACL**: map AD groups to allowed notebooks
- **Show source**: display which notebook answered the question

### Non-Goals (Out of Scope)

- Per-user NotebookLM accounts (single service account)
- Real-time notebook updates (uses cached summaries)
- Voice/audio input processing
- File upload to NotebookLM via bot

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Azure Bot Service                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐               │
│  │   MS Teams      │   │    Telegram     │   │    WebChat      │               │
│  │   Channel       │   │    Channel      │   │    (future)     │               │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘               │
│           └─────────────────────┼─────────────────────┘                         │
│                                 ▼                                                │
│                     ┌───────────────────────┐                                   │
│                     │  Azure Bot Connector  │                                   │
│                     │  (manages channels)   │                                   │
│                     └───────────┬───────────┘                                   │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │ HTTPS (Bot Framework Protocol)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Bot Backend (Your K8s Cluster)                           │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                           Bot Application                                  │  │
│  │                                                                            │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │  │
│  │  │  Auth Middleware │  │  ACL Service    │  │  Response       │           │  │
│  │  │  (AD validation) │  │  (group→notebook)│  │  Formatter      │           │  │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           │  │
│  │           │                    │                    │                     │  │
│  │           └────────────────────┼────────────────────┘                     │  │
│  │                                ▼                                          │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐    │  │
│  │  │                      Message Handler                              │    │  │
│  │  │  1. Validate user (AD token)                                      │    │  │
│  │  │  2. Check ACL (user groups → allowed notebooks)                   │    │  │
│  │  │  3. Call nlm-proxy with session mapping                           │    │  │
│  │  │  4. Format response with source attribution                       │    │  │
│  │  │  5. Send back to user                                             │    │  │
│  │  └──────────────────────────────────────────────────────────────────┘    │  │
│  │                                │                                          │  │
│  └────────────────────────────────┼──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         nlm-proxy Service                                  │  │
│  │  POST /v1/chat/completions                                                │  │
│  │    model: "knowledge-finder"                                              │  │
│  │    headers: { X-OpenWebUI-Chat-Id: conversation_id }                      │  │
│  │                                                                            │  │
│  │  SmartRouter → classify → select_notebook → query NotebookLM              │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │       Azure AD              │
                    │  (user authentication)      │
                    │  (group membership)         │
                    └─────────────────────────────┘
```

### Component Interaction Flow

```
User sends message in Teams
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Azure Bot Service receives message                           │
│    - Extracts user identity (AAD Object ID)                     │
│    - Forwards to bot backend endpoint                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Bot Backend: Auth Middleware                                  │
│    - Validate Bot Framework token                                │
│    - Extract user info (email, name, groups)                     │
│    - For Teams: automatic SSO                                    │
│    - For Telegram: check linked AD account                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Bot Backend: ACL Check                                        │
│    - Get user's AD groups (via Graph API or cached)              │
│    - Lookup allowed notebooks for those groups                   │
│    - If no access: return "unauthorized" message                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Bot Backend: Query nlm-proxy                                  │
│    - Map Teams conversation_id → nlm-proxy chat_id               │
│    - POST to /v1/chat/completions with model="knowledge-finder"  │
│    - Include allowed_notebooks filter (from ACL)                 │
│    - Stream response                                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. nlm-proxy: Smart Routing                                      │
│    - Classify request (NOTEBOOKLM vs LLM_TASK)                   │
│    - Select best notebook from allowed list                      │
│    - Query NotebookLM with conversation context                  │
│    - Return streamed response with reasoning_content             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Bot Backend: Format & Send Response                           │
│    - Extract notebook name from reasoning_content                │
│    - Format response with source attribution                     │
│    - Send Adaptive Card (Teams) or Markdown (Telegram)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Bot Framework** | Python `botbuilder-python` | Same language as nlm-proxy, mature SDK |
| **Web Framework** | FastAPI or aiohttp | Async support, good performance |
| **LLM Client** | `openai` Python SDK | Direct, minimal overhead, nlm-proxy compatible |
| **Azure AD** | `msal` + Microsoft Graph | Official Microsoft libraries |
| **Caching** | Redis | Session storage, ACL caching |
| **Deployment** | Docker + Kubernetes | As per requirement |

### LLM Client Strategy: OpenAI SDK → LangGraph Migration Path

**Phase 1 (Current Design): OpenAI SDK**

The chatbot uses the OpenAI Python SDK directly to communicate with nlm-proxy. This is the right choice for the initial implementation because:

| Reason | Benefit |
|--------|---------|
| **nlm-proxy is OpenAI-compatible** | Direct SDK usage, no adapters needed |
| **Minimal dependencies** | ~2MB vs ~100MB+ for LangGraph |
| **Simple debugging** | Shallow stack traces, easy to trace issues |
| **Native header/body support** | `extra_headers`, `extra_body` work out of the box |
| **Streaming works natively** | Critical for good UX in chat interfaces |

```python
# Current approach: Direct OpenAI SDK
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://nlm-proxy:8080/v1", api_key="...")

response = await client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": query}],
    stream=True,
    extra_headers={"X-OpenWebUI-Chat-Id": conversation_id},
    extra_body={"metadata": {"allowed_notebooks": allowed_notebooks}},
)
```

**Phase 2 (Future): LangGraph Migration**

When the chatbot needs advanced capabilities, migrate to LangGraph for:

| Future Capability | LangGraph Feature |
|-------------------|-------------------|
| **Conversation memory** | Built-in `ConversationBufferMemory`, `ConversationSummaryMemory` |
| **Context management** | Automatic context window management, token counting |
| **Multi-step workflows** | StateGraph for complex agent flows |
| **Conditional routing** | Graph-based branching logic |
| **Tool calling** | If nlm-proxy adds tool support |
| **Checkpointing** | Pause/resume long conversations |

**Migration Triggers** (when to consider LangGraph):

- [ ] Need to maintain conversation history beyond nlm-proxy's session
- [ ] Need multi-step reasoning (query → refine → query again)
- [ ] Need to call multiple backends (nlm-proxy + other LLMs)
- [ ] Need complex state machines for conversation flows
- [ ] Need built-in tracing (LangSmith) beyond OpenTelemetry

**Migration Strategy:**

1. **Isolate LLM client** - Current `NLMProxyClient` class is already isolated
2. **Abstract interface** - Define `BaseLLMClient` protocol if needed
3. **Swap implementation** - Replace OpenAI SDK with LangChain's `ChatOpenAI` + LangGraph
4. **Preserve headers** - Custom `ChatOpenAI` subclass for `extra_headers` support

```python
# Future approach: LangGraph (when needed)
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

class NLMChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI that supports nlm-proxy headers."""

    def _get_request_headers(self, **kwargs):
        headers = super()._get_request_headers(**kwargs)
        if conversation_id := kwargs.get("conversation_id"):
            headers["X-OpenWebUI-Chat-Id"] = conversation_id
        return headers

# Build stateful graph
graph = StateGraph(ConversationState)
graph.add_node("query", query_nlm_node)
graph.add_node("refine", refine_query_node)  # Future: multi-step
graph.add_edge("query", "refine")
app = graph.compile(checkpointer=RedisCheckpointer())
```

**Recommendation:** Start with OpenAI SDK. Only migrate to LangGraph when you hit specific limitations that justify the added complexity.

### Python Dependencies

```toml
# pyproject.toml
[project]
name = "nlm-chatbot"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "botbuilder-core>=4.14.0",
    "botbuilder-integration-aiohttp>=4.14.0",
    "aiohttp>=3.9.0",
    "openai>=1.0.0",
    "msal>=1.24.0",
    "msgraph-sdk>=1.0.0",
    "redis>=5.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "structlog>=23.0.0",
]

[project.optional-dependencies]
# Future: Add when migrating to LangGraph
langgraph = [
    "langchain>=0.1.0",
    "langchain-openai>=0.0.5",
    "langgraph>=0.0.20",
]

dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "httpx>=0.25.0",  # For testing
]
```

---

## Component Design

### Project Structure

```
nlm-chatbot/
├── src/
│   └── nlm_chatbot/
│       ├── __init__.py
│       ├── main.py                 # Application entrypoint
│       ├── config.py               # Pydantic settings
│       │
│       ├── bot/
│       │   ├── __init__.py
│       │   ├── bot.py              # Main bot class (ActivityHandler)
│       │   ├── middleware/
│       │   │   ├── __init__.py
│       │   │   ├── auth.py         # Authentication middleware
│       │   │   └── logging.py      # Request/response logging
│       │   └── handlers/
│       │       ├── __init__.py
│       │       ├── message.py      # Message handling logic
│       │       └── commands.py     # /help, /status, /link commands
│       │
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── azure_ad.py         # Azure AD token validation
│       │   ├── graph_client.py     # Microsoft Graph API client
│       │   └── telegram_oauth.py   # Telegram ↔ AD linking
│       │
│       ├── acl/
│       │   ├── __init__.py
│       │   ├── service.py          # ACL service (group → notebooks)
│       │   └── models.py           # ACL data models
│       │
│       ├── nlm/
│       │   ├── __init__.py
│       │   ├── client.py           # nlm-proxy OpenAI client
│       │   └── session.py          # Session mapping (conv_id → chat_id)
│       │
│       ├── channels/
│       │   ├── __init__.py
│       │   ├── teams.py            # Teams-specific formatting
│       │   └── telegram.py         # Telegram-specific formatting
│       │
│       └── utils/
│           ├── __init__.py
│           ├── rate_limiter.py     # Rate limiting utilities
│           └── formatting.py       # Response formatting helpers
│
├── config/
│   ├── acl.yaml                    # ACL configuration (group → notebooks)
│   └── prompts/
│       └── system.txt              # System prompts for bot personality
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml          # Local development
│   └── k8s/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       └── secrets.yaml
│
├── tests/
│   ├── conftest.py
│   ├── test_bot.py
│   ├── test_acl.py
│   └── test_nlm_client.py
│
├── pyproject.toml
├── README.md
└── .env.example
```

### Core Classes

#### 1. Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Azure Bot
    app_id: str = Field(..., alias="MICROSOFT_APP_ID")
    app_password: str = Field(..., alias="MICROSOFT_APP_PASSWORD")
    app_tenant_id: str = Field(..., alias="MICROSOFT_APP_TENANT_ID")

    # nlm-proxy
    nlm_proxy_base_url: str = Field("http://localhost:8080/v1", alias="NLM_PROXY_BASE_URL")
    nlm_proxy_api_key: str = Field(..., alias="NLM_PROXY_API_KEY")
    nlm_proxy_model: str = Field("knowledge-finder", alias="NLM_PROXY_MODEL")

    # Azure AD / Graph
    graph_client_id: str = Field(..., alias="GRAPH_CLIENT_ID")
    graph_client_secret: str = Field(..., alias="GRAPH_CLIENT_SECRET")

    # Redis
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")

    # ACL
    acl_config_path: str = Field("config/acl.yaml", alias="ACL_CONFIG_PATH")
    acl_cache_ttl: int = Field(300, alias="ACL_CACHE_TTL")  # 5 minutes

    # Rate Limiting
    max_concurrent_requests: int = Field(10, alias="MAX_CONCURRENT_REQUESTS")
    user_rate_limit: int = Field(30, alias="USER_RATE_LIMIT")  # per minute

    # Features
    show_source_notebook: bool = Field(True, alias="SHOW_SOURCE_NOTEBOOK")

    class Config:
        env_file = ".env"
        extra = "ignore"
```

#### 2. Main Bot Class (`bot/bot.py`)

```python
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes

class NotebookLMBot(ActivityHandler):
    """Main bot handler for NotebookLM queries."""

    def __init__(
        self,
        nlm_client: NLMProxyClient,
        acl_service: ACLService,
        graph_client: GraphClient,
        settings: Settings,
    ):
        self.nlm_client = nlm_client
        self.acl_service = acl_service
        self.graph_client = graph_client
        self.settings = settings

    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages."""
        user_message = turn_context.activity.text

        # 1. Get user identity
        user_info = await self._get_user_info(turn_context)
        if not user_info:
            await turn_context.send_activity("❌ Authentication required.")
            return

        # 2. Check ACL
        allowed_notebooks = await self.acl_service.get_allowed_notebooks(
            user_groups=user_info.groups
        )
        if not allowed_notebooks:
            await turn_context.send_activity(
                "🔒 You don't have access to any knowledge bases. "
                "Contact your administrator."
            )
            return

        # 3. Show typing indicator
        await turn_context.send_activity(
            Activity(type=ActivityTypes.typing)
        )

        # 4. Query nlm-proxy
        try:
            conversation_id = self._get_conversation_id(turn_context)
            response = await self.nlm_client.query(
                message=user_message,
                conversation_id=conversation_id,
                allowed_notebooks=allowed_notebooks,
            )

            # 5. Format and send response
            formatted = self._format_response(
                content=response.content,
                notebook_name=response.notebook_name,
                show_source=self.settings.show_source_notebook,
            )
            await turn_context.send_activity(formatted)

        except RateLimitError:
            await turn_context.send_activity(
                "⏳ Too many requests. Please wait a moment and try again."
            )
        except Exception as e:
            logger.exception("Query failed", error=str(e))
            await turn_context.send_activity(
                "⚠️ Something went wrong. Please try again later."
            )

    async def on_members_added_activity(self, members_added, turn_context: TurnContext):
        """Welcome new users."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "👋 Hello! I'm your NotebookLM assistant. "
                    "Ask me anything about your organization's knowledge bases.\n\n"
                    "Type `/help` for available commands."
                )

    def _get_conversation_id(self, turn_context: TurnContext) -> str:
        """Extract conversation ID for session mapping."""
        # Teams: conversation.id is like "19:abc123@thread.tacv2"
        # Telegram: will be mapped differently
        return turn_context.activity.conversation.id

    async def _get_user_info(self, turn_context: TurnContext) -> UserInfo | None:
        """Extract and validate user information."""
        activity = turn_context.activity

        # For Teams, use AAD object ID to get group membership
        aad_object_id = activity.from_property.aad_object_id
        if aad_object_id:
            return await self.graph_client.get_user_with_groups(aad_object_id)

        # For Telegram, check linked account
        # (implementation in telegram_oauth.py)
        return None

    def _format_response(
        self,
        content: str,
        notebook_name: str | None,
        show_source: bool,
    ) -> Activity:
        """Format response with optional source attribution."""
        if show_source and notebook_name:
            # Add source footer
            formatted_content = f"{content}\n\n---\n📚 *Source: {notebook_name}*"
        else:
            formatted_content = content

        return Activity(
            type=ActivityTypes.message,
            text=formatted_content,
            text_format="markdown",
        )
```

#### 3. nlm-proxy Client (`nlm/client.py`)

```python
from dataclasses import dataclass
from openai import AsyncOpenAI
import asyncio

@dataclass
class QueryResponse:
    """Response from nlm-proxy query."""
    content: str
    notebook_name: str | None
    notebook_id: str | None

class NLMProxyClient:
    """Client for nlm-proxy OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "knowledge-finder",
        max_concurrent: int = 10,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def query(
        self,
        message: str,
        conversation_id: str,
        allowed_notebooks: list[str] | None = None,
    ) -> QueryResponse:
        """Query nlm-proxy with rate limiting."""
        async with self.semaphore:
            return await self._do_query(message, conversation_id, allowed_notebooks)

    async def _do_query(
        self,
        message: str,
        conversation_id: str,
        allowed_notebooks: list[str] | None,
    ) -> QueryResponse:
        """Execute the actual query."""
        # Build extra headers
        headers = {"X-OpenWebUI-Chat-Id": conversation_id}

        # If ACL restricts notebooks, pass as metadata
        # Note: This requires nlm-proxy to support notebook filtering
        extra_body = {}
        if allowed_notebooks:
            extra_body["metadata"] = {"allowed_notebooks": allowed_notebooks}

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            stream=True,
            extra_headers=headers,
            extra_body=extra_body if extra_body else None,
        )

        # Collect streamed response
        content_parts = []
        reasoning_parts = []

        async for chunk in response:
            delta = chunk.choices[0].delta

            # Collect reasoning (contains notebook selection info)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)

            # Collect content
            if delta.content:
                content_parts.append(delta.content)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)

        # Extract notebook name from reasoning
        # Format: "Selected notebook: ML Research (ID: abc-123)"
        notebook_name = self._extract_notebook_name(reasoning)
        notebook_id = self._extract_notebook_id(reasoning)

        return QueryResponse(
            content=content,
            notebook_name=notebook_name,
            notebook_id=notebook_id,
        )

    def _extract_notebook_name(self, reasoning: str) -> str | None:
        """Extract notebook name from reasoning content."""
        import re
        match = re.search(r"Selected notebook: (.+?) \(ID:", reasoning)
        return match.group(1) if match else None

    def _extract_notebook_id(self, reasoning: str) -> str | None:
        """Extract notebook ID from reasoning content."""
        import re
        match = re.search(r"\(ID: ([a-zA-Z0-9-]+)\)", reasoning)
        return match.group(1) if match else None

    async def close(self):
        """Cleanup resources."""
        await self.client.close()
```

#### 4. ACL Service (`acl/service.py`)

```python
import yaml
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class NotebookACL:
    """ACL entry for a notebook."""
    notebook_id: str
    notebook_name: str
    allowed_groups: list[str]

class ACLService:
    """Service for managing notebook access control."""

    def __init__(self, config_path: str, redis_client=None, cache_ttl: int = 300):
        self.config_path = config_path
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self._acl_config = self._load_config()

    def _load_config(self) -> dict:
        """Load ACL configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def reload_config(self):
        """Reload ACL configuration (for hot reload)."""
        self._acl_config = self._load_config()

    async def get_allowed_notebooks(self, user_groups: list[str]) -> list[str]:
        """Get list of notebook IDs user can access based on group membership."""
        # Check cache first
        cache_key = f"acl:{':'.join(sorted(user_groups))}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return cached.split(",")

        # Compute allowed notebooks
        allowed = []
        for notebook in self._acl_config.get("notebooks", []):
            notebook_groups = set(notebook.get("allowed_groups", []))
            user_group_set = set(user_groups)

            # Check if user has any allowed group
            if notebook_groups & user_group_set:
                allowed.append(notebook["id"])

            # Check for wildcard (all authenticated users)
            if "*" in notebook_groups:
                allowed.append(notebook["id"])

        # Cache result
        if self.redis and allowed:
            await self.redis.setex(cache_key, self.cache_ttl, ",".join(allowed))

        return allowed

    def get_notebook_name(self, notebook_id: str) -> str | None:
        """Get notebook display name by ID."""
        for notebook in self._acl_config.get("notebooks", []):
            if notebook["id"] == notebook_id:
                return notebook.get("name", notebook_id)
        return None
```

#### 5. ACL Configuration (`config/acl.yaml`)

```yaml
# Access Control List Configuration
# Maps Azure AD groups to allowed notebooks

notebooks:
  # HR Knowledge Base
  - id: "abc123-hr-notebook-id"
    name: "HR Policies & Procedures"
    description: "Employee handbook, leave policies, benefits"
    allowed_groups:
      - "All Employees"           # AD group name
      - "sg-hr-team"              # Security group

  # Technical Documentation
  - id: "def456-tech-notebook-id"
    name: "Technical Documentation"
    description: "API docs, architecture, coding standards"
    allowed_groups:
      - "sg-engineering"
      - "sg-devops"
      - "sg-tech-leads"

  # Product Knowledge
  - id: "ghi789-product-notebook-id"
    name: "Product Knowledge Base"
    description: "Product features, roadmap, competitive analysis"
    allowed_groups:
      - "sg-product-team"
      - "sg-sales"
      - "sg-customer-success"

  # Public Knowledge (all authenticated users)
  - id: "jkl012-public-notebook-id"
    name: "Company General Information"
    description: "Company policies, office locations, general FAQ"
    allowed_groups:
      - "*"  # Wildcard: all authenticated users

# Default behavior when no notebook matches
defaults:
  # If true, queries without matching notebooks are rejected
  # If false, queries are forwarded without notebook filter
  strict_mode: true

  # Message shown when user has no access
  no_access_message: |
    You don't have access to any knowledge bases.
    Please contact your IT administrator to request access.
```

---

## Authentication & Authorization

### Overview

The chatbot uses **two separate Azure AD app registrations**:

1. **Bot Registration** - For Bot Framework authentication (bot identity)
2. **Graph API Client** - For reading user groups (app-only permissions)

We use **app-only permissions** (not delegated) for the Graph API client because:
- No user consent dialogs required
- Simpler implementation (no SSO token exchange)
- More reliable (no token expiry issues per-user)
- Centralized admin control

### Azure AD App Registrations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Azure AD App Registrations                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ App 1: Bot Registration (created via Azure Bot Service)              │    │
│  │                                                                      │    │
│  │ - App ID: MICROSOFT_APP_ID                                          │    │
│  │ - Secret: MICROSOFT_APP_PASSWORD                                     │    │
│  │ - Purpose: Bot Framework authentication                              │    │
│  │ - Permissions: None (just bot identity)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ App 2: Graph API Client (for reading user groups)                    │    │
│  │                                                                      │    │
│  │ - App ID: GRAPH_CLIENT_ID                                           │    │
│  │ - Secret: GRAPH_CLIENT_SECRET                                        │    │
│  │ - Purpose: Call Microsoft Graph API                                  │    │
│  │ - Permissions (Application, NOT Delegated):                          │    │
│  │   • User.Read.All           - Read user profiles                    │    │
│  │   • GroupMember.Read.All    - Read group memberships                │    │
│  │   • Directory.Read.All      - Read directory data (optional)        │    │
│  │                                                                      │    │
│  │ ⚠️  Requires Admin Consent (one-time, by Azure AD admin)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MS Teams Authentication Flow (App-Only)

When a user sends a message in Teams, authentication happens automatically:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MS Teams Authentication Flow                             │
│                                                                              │
│  ┌─────┐                                                                    │
│  │STEP │  User opens Teams (already signed in with M365 account)            │
│  │  1  │  → Azure AD has issued tokens                                      │
│  └─────┘  → User identity: john@company.com                                  │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  User sends message to bot in Teams                                │
│  │  2  │  "What is our vacation policy?"                                    │
│  └─────┘                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Teams packages message with user context (automatic)              │
│  │  3  │  Includes: aadObjectId, name, tenantId                             │
│  └─────┘  Sends to Azure Bot Service                                         │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Azure Bot Service validates & forwards to your bot                │
│  │  4  │  POST https://your-bot.com/api/messages                            │
│  └─────┘  Authorization: Bearer <bot-framework-token>                        │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Bot extracts user identity from activity (no login needed!)       │
│  │  5  │  aad_object_id = activity.from_property.aad_object_id              │
│  └─────┘  user_name = activity.from_property.name                            │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Bot calls Microsoft Graph (app-only token) to get user groups    │
│  │  6  │  GET /users/{aad_object_id}/memberOf                               │
│  └─────┘  Returns: ["sg-engineering", "All Employees"]                       │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  ACL Service maps groups → allowed notebooks                       │
│  │  7  │  "sg-engineering" → ["tech-docs"]                                  │
│  └─────┘  "All Employees" → ["public-kb", "hr-policies"]                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Bot calls nlm-proxy with ACL filter                               │
│  │  8  │  POST /v1/chat/completions                                         │
│  └─────┘  metadata.allowed_notebooks = ["tech-docs", "public-kb", ...]       │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Response sent back through Bot Service → Teams                    │
│  │  9  │  User sees answer with source attribution                          │
│  └─────┘                                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Users are already authenticated in Teams. The bot receives their Azure AD Object ID automatically - no login dialog, no consent popup, completely seamless.

### Teams Activity Structure

When a message arrives from Teams, the activity contains user identity:

```python
# What the bot receives in turn_context.activity

{
    "type": "message",
    "text": "What is our vacation policy?",
    "from": {
        "id": "29:1abc...",           # Teams-specific user ID
        "name": "John Smith",
        "aadObjectId": "12345-..."    # Azure AD Object ID (key for Graph API!)
    },
    "conversation": {
        "id": "19:abc123@thread.tacv2"  # Use for session mapping
    },
    "channelData": {
        "tenant": {
            "id": "tenant-guid"         # Your organization's tenant
        }
    }
}
```

### Graph Client Implementation (App-Only)

```python
# auth/graph_client.py

from dataclasses import dataclass
from typing import Optional
import httpx
from msal import ConfidentialClientApplication


@dataclass
class UserInfo:
    """User information from Azure AD."""
    aad_object_id: str
    display_name: str
    email: Optional[str]
    groups: list[str]  # List of group display names


class GraphClient:
    """Microsoft Graph API client using app-only authentication.

    Uses application permissions (not delegated) to read user groups.
    Requires admin consent for: User.Read.All, GroupMember.Read.All
    """

    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
    ):
        self.tenant_id = tenant_id
        self.msal_app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
        )
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _get_app_token(self) -> str:
        """Acquire app-only token for Graph API.

        This uses client credentials flow - no user interaction needed.
        Token is cached by MSAL automatically.
        """
        result = self.msal_app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )

        if "access_token" not in result:
            error = result.get("error_description", "Unknown error")
            raise Exception(f"Failed to get Graph API token: {error}")

        return result["access_token"]

    async def get_user_with_groups(
        self,
        aad_object_id: str,
    ) -> UserInfo:
        """Get user info and group memberships from Graph API.

        Args:
            aad_object_id: The user's Azure AD Object ID (from Teams activity)

        Returns:
            UserInfo with display name, email, and list of group names
        """
        token = self._get_app_token()
        client = await self._get_http_client()
        headers = {"Authorization": f"Bearer {token}"}

        # Get user profile
        user_response = await client.get(
            f"{self.GRAPH_API_BASE}/users/{aad_object_id}",
            headers=headers,
        )
        user_response.raise_for_status()
        user_data = user_response.json()

        # Get group memberships (transitive - includes nested groups)
        groups_response = await client.get(
            f"{self.GRAPH_API_BASE}/users/{aad_object_id}/transitiveMemberOf",
            headers=headers,
            params={"$select": "displayName,id", "$top": "999"},
        )
        groups_response.raise_for_status()
        groups_data = groups_response.json()

        # Extract group display names (filter to only groups, not roles)
        group_names = [
            item["displayName"]
            for item in groups_data.get("value", [])
            if item.get("@odata.type") == "#microsoft.graph.group"
            and item.get("displayName")
        ]

        return UserInfo(
            aad_object_id=aad_object_id,
            display_name=user_data.get("displayName", "Unknown"),
            email=user_data.get("mail") or user_data.get("userPrincipalName"),
            groups=group_names,
        )

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
```

### Bot Message Handler with App-Only Auth

```python
# bot/bot.py

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes
import structlog

logger = structlog.get_logger()


class NotebookLMBot(ActivityHandler):
    """Main bot handler with Azure AD app-only authentication."""

    def __init__(
        self,
        nlm_client: NLMProxyClient,
        acl_service: ACLService,
        graph_client: GraphClient,
        settings: Settings,
    ):
        self.nlm_client = nlm_client
        self.acl_service = acl_service
        self.graph_client = graph_client
        self.settings = settings
        self._user_cache: dict[str, UserInfo] = {}  # In-memory cache

    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages from Teams."""
        activity = turn_context.activity

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Extract user identity from Teams activity
        #         (Already authenticated - no login needed!)
        # ═══════════════════════════════════════════════════════════════
        aad_object_id = activity.from_property.aad_object_id
        user_name = activity.from_property.name

        if not aad_object_id:
            # This shouldn't happen for Teams users in the same tenant
            logger.warning("No AAD Object ID in activity", user_name=user_name)
            await turn_context.send_activity(
                "❌ Unable to identify your account. "
                "Please ensure you're signed into Teams with your work account."
            )
            return

        logger.info(
            "message_received",
            user_name=user_name,
            aad_object_id=aad_object_id,
            message_preview=activity.text[:50] if activity.text else "",
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Get user groups via Graph API (with caching)
        # ═══════════════════════════════════════════════════════════════
        try:
            user_info = await self._get_user_info_cached(aad_object_id)
        except Exception as e:
            logger.error("Failed to get user info from Graph", error=str(e))
            await turn_context.send_activity(
                "⚠️ Unable to verify your permissions. Please try again later."
            )
            return

        logger.info(
            "user_groups_retrieved",
            user_name=user_info.display_name,
            groups=user_info.groups,
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Check ACL - get allowed notebooks for this user
        # ═══════════════════════════════════════════════════════════════
        allowed_notebooks = await self.acl_service.get_allowed_notebooks(
            user_groups=user_info.groups
        )

        if not allowed_notebooks:
            logger.warning(
                "user_has_no_access",
                user_name=user_info.display_name,
                groups=user_info.groups,
            )
            await turn_context.send_activity(
                "🔒 You don't have access to any knowledge bases.\n\n"
                f"Your groups: {', '.join(user_info.groups) or 'None'}\n\n"
                "Please contact your administrator for access."
            )
            return

        logger.info(
            "acl_check_passed",
            user_name=user_info.display_name,
            allowed_notebooks=allowed_notebooks,
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Show typing indicator while processing
        # ═══════════════════════════════════════════════════════════════
        await turn_context.send_activity(Activity(type=ActivityTypes.typing))

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Query nlm-proxy with ACL filter
        # ═══════════════════════════════════════════════════════════════
        try:
            conversation_id = activity.conversation.id

            response = await self.nlm_client.query(
                message=activity.text,
                conversation_id=conversation_id,
                allowed_notebooks=allowed_notebooks,  # Per-request ACL filter!
            )

            # ═══════════════════════════════════════════════════════════
            # STEP 6: Format and send response with source attribution
            # ═══════════════════════════════════════════════════════════
            formatted = self._format_response(
                content=response.content,
                notebook_name=response.notebook_name,
            )

            logger.info(
                "response_sent",
                user_name=user_info.display_name,
                notebook_used=response.notebook_name,
                response_length=len(response.content),
            )

            await turn_context.send_activity(formatted)

        except Exception as e:
            logger.exception("query_failed", user_name=user_name, error=str(e))
            await turn_context.send_activity(
                "⚠️ Something went wrong. Please try again later."
            )

    async def _get_user_info_cached(
        self,
        aad_object_id: str,
        cache_ttl: int = 300,  # 5 minutes
    ) -> UserInfo:
        """Get user info with in-memory caching.

        Note: In production, use Redis for distributed caching.
        """
        import time

        cache_key = aad_object_id
        cached = self._user_cache.get(cache_key)

        if cached:
            # Simple TTL check (in production, store timestamp)
            return cached

        user_info = await self.graph_client.get_user_with_groups(aad_object_id)
        self._user_cache[cache_key] = user_info

        return user_info

    def _format_response(
        self,
        content: str,
        notebook_name: str | None,
    ) -> Activity:
        """Format response with source attribution."""
        if self.settings.show_source_notebook and notebook_name:
            formatted_content = f"{content}\n\n---\n📚 *Source: {notebook_name}*"
        else:
            formatted_content = content

        return Activity(
            type=ActivityTypes.message,
            text=formatted_content,
            text_format="markdown",
        )
```

### Security Considerations

| Concern | Mitigation |
|---------|------------|
| **Bot Framework token validation** | Handled automatically by Bot Framework SDK |
| **Tenant isolation** | Verify `tenant_id` matches your organization |
| **Group spoofing** | Groups come from Graph API, not from client |
| **Token security** | App-only tokens cached by MSAL, auto-refresh |
| **Secrets management** | Store in Azure Key Vault, not env vars (production) |
| **Audit trail** | Log user email + query + notebook accessed |
| **Rate limiting** | Implement per-user limits to prevent abuse |

### Admin Consent Setup

To grant admin consent for the Graph API app:

1. Go to Azure Portal → Azure Active Directory → App registrations
2. Select your Graph API app (GRAPH_CLIENT_ID)
3. Go to API permissions
4. Add permissions:
   - Microsoft Graph → Application permissions → User.Read.All
   - Microsoft Graph → Application permissions → GroupMember.Read.All
5. Click "Grant admin consent for [Your Org]"
6. Confirm the permissions show green checkmarks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     API Permissions (after admin consent)                    │
│                                                                              │
│  Permission                  Type           Status                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  User.Read.All               Application    ✅ Granted for [Your Org]       │
│  GroupMember.Read.All        Application    ✅ Granted for [Your Org]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Telegram Authentication (OAuth Account Linking)

Unlike Teams where users are already authenticated, Telegram users must **prove their AD identity** through an OAuth flow. This creates a permanent link between their Telegram ID and Azure AD Object ID.

#### The Identity Bridging Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Teams vs Telegram: Identity Gap                          │
│                                                                              │
│  MS Teams                              Telegram                              │
│  ─────────────────────────────────     ─────────────────────────────────    │
│                                                                              │
│  User signed in with:                  User signed in with:                  │
│  john@company.com                      Phone number / Telegram account       │
│                                                                              │
│  Bot receives:                         Bot receives:                         │
│  • aadObjectId: "12345-..."            • telegram_id: 987654321              │
│  • name: "John Smith"                  • username: "john_smith"              │
│  • email: "john@company.com"           • first_name: "John"                  │
│                                                                              │
│  ✅ Direct link to AD                  ❌ No link to AD                      │
│  ✅ Can get groups immediately         ❓ Who is this user in AD?            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Account Linking Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Telegram OAuth Account Linking                        │
│                                                                              │
│  ┌─────┐                                                                    │
│  │STEP │  User sends /start to Telegram bot                                │
│  │  1  │  Bot shows "Link Work Account" button                              │
│  └─────┘                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  User clicks button → Bot generates OAuth URL                      │
│  │  2  │  • Generate unique state token (UUID)                              │
│  └─────┘  • Store in Redis: state → telegram_user_id (5 min TTL)            │
│     │     • Return Azure AD OAuth URL to user                                │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  User clicks link → Opens Azure AD login in browser                │
│  │  3  │  https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize  │
│  └─────┘                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  User authenticates with work account                              │
│  │  4  │  john@company.com + password + MFA                                 │
│  └─────┘                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Azure AD redirects to your callback URL with auth code            │
│  │  5  │  https://bot.company.com/auth/telegram/callback?code=...&state=... │
│  └─────┘                                                                     │
│     │                                                                        │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Backend handles callback:                                          │
│  │  6  │  • Validate state token (from Redis)                               │
│  └─────┘  • Exchange code for access token                                   │
│     │     • Call Graph API /me to get user info                              │
│     │     • Store permanent mapping: telegram_id → aad_object_id             │
│     ▼                                                                        │
│  ┌─────┐                                                                    │
│  │STEP │  Send success message to user in Telegram                          │
│  │  7  │  "✅ Account linked! Welcome, John Smith!"                          │
│  └─────┘                                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### User Experience

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Telegram Chat                                                               │
│                                                                              │
│  User: /start                                                                │
│                                                                              │
│  Bot:  👋 Welcome to the NotebookLM Assistant!                              │
│                                                                              │
│        To use this bot, you need to link your work account.                 │
│                                                                              │
│        [ 🔗 Link Work Account ]  ← Inline button                            │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (User clicks button)                                                        │
│                                                                              │
│  Bot:  🔐 Please click this link to sign in with your work account:         │
│                                                                              │
│        https://bot.company.com/auth/telegram?state=abc123...                │
│                                                                              │
│        ⏰ This link expires in 5 minutes.                                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  (After successful OAuth)                                                    │
│                                                                              │
│  Bot:  ✅ Account linked successfully!                                       │
│                                                                              │
│        Welcome, John Smith!                                                  │
│        Email: john@company.com                                               │
│                                                                              │
│        You can now ask me questions about your organization's               │
│        knowledge bases.                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### OAuth Service Implementation

```python
# auth/telegram_oauth.py

import uuid
from urllib.parse import urlencode
from dataclasses import dataclass
import httpx


@dataclass
class LinkResult:
    """Result of account linking."""
    success: bool
    telegram_user_id: int
    aad_object_id: str | None = None
    display_name: str | None = None
    email: str | None = None
    error: str | None = None


class TelegramOAuthService:
    """Handles Telegram ↔ Azure AD account linking.

    This service bridges Telegram user IDs to Azure AD Object IDs,
    enabling the bot to look up AD group memberships for ACL enforcement.
    """

    def __init__(
        self,
        redis_client,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ):
        self.redis = redis_client
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    async def generate_link_url(self, telegram_user_id: int) -> str:
        """Generate OAuth URL for Telegram user.

        Args:
            telegram_user_id: Telegram's numeric user ID

        Returns:
            URL the user should click to authenticate with Azure AD
        """
        # Generate unique state token (prevents CSRF)
        state = str(uuid.uuid4())

        # Store state → telegram_user_id mapping (expires in 5 minutes)
        await self.redis.setex(
            f"telegram_link_state:{state}",
            300,  # 5 minutes
            str(telegram_user_id),
        )

        # Build Azure AD OAuth URL
        auth_params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "response_mode": "query",
            "scope": "openid profile email User.Read",
            "state": state,
        }

        auth_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}"
            f"/oauth2/v2.0/authorize?{urlencode(auth_params)}"
        )

        return auth_url

    async def handle_callback(self, code: str, state: str) -> LinkResult:
        """Handle OAuth callback from Azure AD.

        Args:
            code: Authorization code from Azure AD
            state: State token (must match what we generated)

        Returns:
            LinkResult with success status and user info
        """
        # Validate state token
        telegram_user_id_str = await self.redis.get(f"telegram_link_state:{state}")
        if not telegram_user_id_str:
            return LinkResult(
                success=False,
                telegram_user_id=0,
                error="Invalid or expired state token"
            )

        telegram_user_id = int(telegram_user_id_str)

        try:
            async with httpx.AsyncClient() as client:
                # Exchange code for tokens
                token_response = await client.post(
                    f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token",
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "redirect_uri": self.redirect_uri,
                        "grant_type": "authorization_code",
                    },
                )

                if token_response.status_code != 200:
                    return LinkResult(
                        success=False,
                        telegram_user_id=telegram_user_id,
                        error="Token exchange failed"
                    )

                tokens = token_response.json()
                access_token = tokens["access_token"]

                # Get user info from Graph API
                user_response = await client.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if user_response.status_code != 200:
                    return LinkResult(
                        success=False,
                        telegram_user_id=telegram_user_id,
                        error="Failed to get user info"
                    )

                user_data = user_response.json()
                aad_object_id = user_data["id"]
                display_name = user_data.get("displayName", "Unknown")
                email = user_data.get("mail") or user_data.get("userPrincipalName")

            # Store permanent mapping
            await self.redis.set(
                f"telegram_to_aad:{telegram_user_id}",
                aad_object_id,
            )

            # Store reverse mapping (useful for admin tools)
            await self.redis.set(
                f"aad_to_telegram:{aad_object_id}",
                str(telegram_user_id),
            )

            # Clean up state token
            await self.redis.delete(f"telegram_link_state:{state}")

            return LinkResult(
                success=True,
                telegram_user_id=telegram_user_id,
                aad_object_id=aad_object_id,
                display_name=display_name,
                email=email,
            )

        except Exception as e:
            return LinkResult(
                success=False,
                telegram_user_id=telegram_user_id,
                error=str(e)
            )

    async def get_aad_object_id(self, telegram_user_id: int) -> str | None:
        """Get linked AAD Object ID for a Telegram user.

        Returns None if account is not linked.
        """
        return await self.redis.get(f"telegram_to_aad:{telegram_user_id}")

    async def unlink_account(self, telegram_user_id: int) -> bool:
        """Remove the account link."""
        aad_object_id = await self.redis.get(f"telegram_to_aad:{telegram_user_id}")
        if aad_object_id:
            await self.redis.delete(f"telegram_to_aad:{telegram_user_id}")
            await self.redis.delete(f"aad_to_telegram:{aad_object_id}")
            return True
        return False
```

#### Telegram Bot Handler

```python
# bot/telegram_handler.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

class TelegramBotHandler:
    """Handles Telegram bot messages with AD authentication."""

    def __init__(
        self,
        oauth_service: TelegramOAuthService,
        graph_client: GraphClient,
        acl_service: ACLService,
        nlm_client: NLMProxyClient,
    ):
        self.oauth = oauth_service
        self.graph = graph_client
        self.acl = acl_service
        self.nlm = nlm_client

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        telegram_user_id = update.effective_user.id

        # Check if already linked
        aad_id = await self.oauth.get_aad_object_id(telegram_user_id)

        if aad_id:
            await update.message.reply_text(
                "👋 Welcome back! Your account is already linked.\n\n"
                "Just send me a question to get started!\n\n"
                "Commands:\n"
                "/status - Check your linked account\n"
                "/unlink - Unlink your work account"
            )
        else:
            keyboard = [[
                InlineKeyboardButton(
                    "🔗 Link Work Account",
                    callback_data="link_account"
                )
            ]]
            await update.message.reply_text(
                "👋 Welcome to the NotebookLM Assistant!\n\n"
                "To use this bot, you need to link your work account.\n\n"
                "Click the button below to get started:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    async def handle_link_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle link button click."""
        query = update.callback_query
        await query.answer()

        telegram_user_id = update.effective_user.id
        link_url = await self.oauth.generate_link_url(telegram_user_id)

        await query.edit_message_text(
            f"🔐 Please click this link to sign in with your work account:\n\n"
            f"{link_url}\n\n"
            f"⏰ This link expires in 5 minutes."
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages (queries)."""
        telegram_user_id = update.effective_user.id
        message_text = update.message.text

        # Check if linked
        aad_object_id = await self.oauth.get_aad_object_id(telegram_user_id)

        if not aad_object_id:
            keyboard = [[
                InlineKeyboardButton("🔗 Link Work Account", callback_data="link_account")
            ]]
            await update.message.reply_text(
                "🔒 Please link your work account first to use this bot.",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        # Show typing indicator
        await update.message.chat.send_action("typing")

        try:
            # Get user groups (same as Teams flow)
            user_info = await self.graph.get_user_with_groups(aad_object_id)

            # ACL check
            allowed_notebooks = await self.acl.get_allowed_notebooks(
                user_groups=user_info.groups
            )

            if not allowed_notebooks:
                await update.message.reply_text(
                    "🔒 You don't have access to any knowledge bases.\n"
                    "Please contact your administrator."
                )
                return

            # Query nlm-proxy
            response = await self.nlm.query(
                message=message_text,
                conversation_id=f"telegram:{update.message.chat_id}",
                allowed_notebooks=allowed_notebooks,
            )

            # Format response for Telegram
            formatted = self._format_response(response.content, response.notebook_name)
            await update.message.reply_text(formatted, parse_mode="MarkdownV2")

        except Exception as e:
            await update.message.reply_text(
                "⚠️ Something went wrong. Please try again later."
            )

    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show linked account info."""
        telegram_user_id = update.effective_user.id
        aad_object_id = await self.oauth.get_aad_object_id(telegram_user_id)

        if not aad_object_id:
            await update.message.reply_text("❌ No account linked. Use /start to link.")
            return

        user_info = await self.graph.get_user_with_groups(aad_object_id)
        allowed_notebooks = await self.acl.get_allowed_notebooks(user_info.groups)

        await update.message.reply_text(
            f"✅ Account Linked\n\n"
            f"👤 Name: {user_info.display_name}\n"
            f"📧 Email: {user_info.email}\n"
            f"👥 Groups: {len(user_info.groups)}\n"
            f"📚 Accessible notebooks: {len(allowed_notebooks)}"
        )

    async def handle_unlink(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unlink command."""
        telegram_user_id = update.effective_user.id
        success = await self.oauth.unlink_account(telegram_user_id)

        if success:
            await update.message.reply_text(
                "✅ Account unlinked.\n\nUse /start to link a different account."
            )
        else:
            await update.message.reply_text("❌ No account was linked.")

    def _format_response(self, content: str, notebook_name: str | None) -> str:
        """Format response for Telegram MarkdownV2."""
        # Escape special characters
        escaped = self._escape_markdown(content)

        if notebook_name:
            escaped_name = self._escape_markdown(notebook_name)
            return f"{escaped}\n\n———\n📚 _Source: {escaped_name}_"
        return escaped

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#',
                         '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
```

#### OAuth Callback Web Handler

```python
# routes/auth.py

from aiohttp import web

async def telegram_oauth_callback(request: web.Request) -> web.Response:
    """Handle OAuth callback from Azure AD."""
    oauth_service = request.app["telegram_oauth_service"]
    telegram_bot = request.app["telegram_bot"]

    code = request.query.get("code")
    state = request.query.get("state")
    error = request.query.get("error")

    # Handle user cancellation or error
    if error:
        return web.Response(
            text="""
            <html>
            <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                <h1>❌ Authentication Failed</h1>
                <p>You can close this window and try again in Telegram.</p>
            </body>
            </html>
            """,
            content_type="text/html",
        )

    # Process the OAuth callback
    result = await oauth_service.handle_callback(code, state)

    if not result.success:
        return web.Response(
            text=f"""
            <html>
            <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                <h1>❌ Link Failed</h1>
                <p>{result.error}</p>
                <p>Please try again in Telegram with /start</p>
            </body>
            </html>
            """,
            content_type="text/html",
            status=400,
        )

    # Send success message to user via Telegram
    await telegram_bot.send_message(
        chat_id=result.telegram_user_id,
        text=(
            f"✅ Account linked successfully!\n\n"
            f"Welcome, **{result.display_name}**!\n"
            f"Email: {result.email}\n\n"
            f"You can now ask me questions about your organization's "
            f"knowledge bases."
        ),
        parse_mode="Markdown",
    )

    # Show success page in browser
    return web.Response(
        text=f"""
        <html>
        <body style="font-family: Arial; text-align: center; padding-top: 50px;">
            <h1>✅ Account Linked!</h1>
            <p>Welcome, {result.display_name}!</p>
            <p>You can close this window and return to Telegram.</p>
        </body>
        </html>
        """,
        content_type="text/html",
    )
```

#### Azure AD Configuration for Telegram OAuth

For Telegram OAuth, add a **redirect URI** to the Graph API Client app (App 2):

1. Go to Azure Portal → Azure AD → App registrations → NLM Chatbot Graph Client
2. Click **Authentication** in the left menu
3. Click **+ Add a platform** → Select **Web**
4. Add redirect URI: `https://bot.yourcompany.com/auth/telegram/callback`
5. Check **ID tokens** under Implicit grant
6. Click **Save**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Authentication | NLM Chatbot Graph Client                                   │
│                                                                              │
│  Platform configurations                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Web                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Redirect URIs:                                                       │    │
│  │   https://bot.yourcompany.com/auth/telegram/callback                │    │
│  │                                                                      │    │
│  │ Implicit grant and hybrid flows:                                     │    │
│  │   ☑ ID tokens                                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Query Flow After Linking

Once linked, Telegram queries follow the same pattern as Teams:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Telegram Query Flow (After Linking)                      │
│                                                                              │
│  User sends message                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  Bot receives telegram_user_id = 987654321                                   │
│       │                                                                      │
│       ▼                                                                      │
│  Lookup: redis.get("telegram_to_aad:987654321") → "12345-abcd-..."          │
│       │                                                                      │
│       │  (If not linked → "Please link your account first")                 │
│       ▼                                                                      │
│  Get groups via Graph API (same as Teams)                                    │
│       │                                                                      │
│       ▼                                                                      │
│  ACL check → allowed notebooks                                               │
│       │                                                                      │
│       ▼                                                                      │
│  Query nlm-proxy with ACL filter                                             │
│       │                                                                      │
│       ▼                                                                      │
│  Format response (Telegram Markdown) and send                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Teams vs Telegram Comparison

| Aspect | MS Teams | Telegram |
|--------|----------|----------|
| **User identity source** | Azure AD (automatic) | Telegram ID (needs linking) |
| **First-time experience** | Seamless, no login | Must click link + sign in once |
| **Authentication** | Already signed in | OAuth flow required |
| **Session persistence** | N/A (always authenticated) | Stored in Redis (permanent) |
| **Group lookup** | Direct via Graph API | Via linked AAD Object ID |
| **Implementation complexity** | Low | Medium |
| **User friction** | Zero | One-time OAuth (5 min) |

#### Security Considerations for Telegram

| Concern | Mitigation |
|---------|------------|
| **CSRF attacks** | Random UUID state token, stored in Redis, expires in 5 min |
| **Token theft** | HTTPS only, short-lived auth codes |
| **Account hijacking** | Only the Telegram user who initiated linking can complete it |
| **Session fixation** | State token is single-use, deleted after callback |
| **Multiple accounts** | One Telegram → One AD account (overwrite on re-link) |
| **Unlink protection** | Consider requiring re-authentication before unlink |

---

## Channel Implementations

### Teams Channel

Teams uses Adaptive Cards for rich formatting:

```python
# channels/teams.py

from botbuilder.schema import Activity, Attachment
from botbuilder.core import CardFactory

class TeamsFormatter:
    """Teams-specific response formatting."""

    @staticmethod
    def format_response(
        content: str,
        notebook_name: str | None,
        show_source: bool,
    ) -> Activity:
        """Format response as Adaptive Card."""
        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": content,
                    "wrap": True,
                    "markdown": True,
                }
            ],
        }

        if show_source and notebook_name:
            card["body"].append({
                "type": "TextBlock",
                "text": f"📚 Source: {notebook_name}",
                "size": "Small",
                "color": "Accent",
                "separator": True,
            })

        return Activity(
            type="message",
            attachments=[
                Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )
            ],
        )

    @staticmethod
    def format_error(error_type: str, message: str) -> Activity:
        """Format error message."""
        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": f"⚠️ {message}",
                    "wrap": True,
                    "color": "Warning",
                }
            ],
        }

        return Activity(
            type="message",
            attachments=[
                Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )
            ],
        )
```

### Telegram Channel

Telegram uses Markdown V2 formatting:

```python
# channels/telegram.py

class TelegramFormatter:
    """Telegram-specific response formatting."""

    MAX_MESSAGE_LENGTH = 4096

    @classmethod
    def format_response(
        cls,
        content: str,
        notebook_name: str | None,
        show_source: bool,
    ) -> str:
        """Format response for Telegram."""
        # Escape special characters for MarkdownV2
        escaped_content = cls._escape_markdown(content)

        if show_source and notebook_name:
            escaped_name = cls._escape_markdown(notebook_name)
            full_response = f"{escaped_content}\n\n———\n📚 _Source: {escaped_name}_"
        else:
            full_response = escaped_content

        # Truncate if too long
        if len(full_response) > cls.MAX_MESSAGE_LENGTH:
            truncated = full_response[:cls.MAX_MESSAGE_LENGTH - 50]
            full_response = f"{truncated}\n\n_\\[Response truncated\\]_"

        return full_response

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
```

---

## nlm-proxy Integration

### Required nlm-proxy Enhancements

To support ACL-based notebook filtering, nlm-proxy needs a small enhancement:

```python
# In nlm-proxy: openai/server.py

async def handle_smart_routing(request: ChatCompletionRequest, ...):
    # Extract allowed notebooks from metadata
    allowed_notebooks = None
    if request.metadata and "allowed_notebooks" in request.metadata:
        allowed_notebooks = request.metadata["allowed_notebooks"]

    # Pass to router
    decision = await router.route(
        query=user_message,
        allowed_notebooks=allowed_notebooks,  # NEW: filter candidates
    )
```

```python
# In nlm-proxy: openai/router.py

async def select_notebook(
    self,
    query: str,
    allowed_notebooks: list[str] | None = None,  # NEW parameter
) -> tuple[str | None, str]:
    """Select best notebook for query."""
    notebooks = await self._ensure_notebooks_cached()

    # Filter by allowed notebooks if specified
    if allowed_notebooks:
        notebooks = [nb for nb in notebooks if nb.id in allowed_notebooks]

    if not notebooks:
        return None, "No accessible notebooks"

    # ... rest of selection logic
```

### Session Mapping

The bot uses the same session mapping as Open WebUI:

| Bot Platform | Conversation ID Source | Maps To |
|--------------|------------------------|---------|
| Teams | `activity.conversation.id` | `X-OpenWebUI-Chat-Id` header |
| Telegram | `chat.id` | Same header |

This leverages nlm-proxy's existing `SessionStore` for NotebookLM conversation continuity.

---

## Error Handling & Resilience

### Rate Limiting

```python
# utils/rate_limiter.py

import asyncio
from collections import defaultdict
import time

class RateLimiter:
    """Token bucket rate limiter with per-user tracking."""

    def __init__(
        self,
        max_concurrent: int = 10,
        user_limit: int = 30,  # requests per minute
        window_seconds: int = 60,
    ):
        self.global_semaphore = asyncio.Semaphore(max_concurrent)
        self.user_limit = user_limit
        self.window_seconds = window_seconds
        self.user_requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def acquire(self, user_id: str) -> bool:
        """Acquire rate limit slot for user. Returns False if rate limited."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Clean old requests
            self.user_requests[user_id] = [
                t for t in self.user_requests[user_id] if t > cutoff
            ]

            # Check user limit
            if len(self.user_requests[user_id]) >= self.user_limit:
                return False

            # Record request
            self.user_requests[user_id].append(now)

        # Acquire global semaphore
        await self.global_semaphore.acquire()
        return True

    def release(self):
        """Release global semaphore slot."""
        self.global_semaphore.release()
```

### Error Messages

```python
# utils/errors.py

from enum import Enum

class ErrorType(Enum):
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"
    NO_ACCESS = "no_access"
    NOTEBOOK_NOT_FOUND = "notebook_not_found"
    INTERNAL_ERROR = "internal_error"
    NLM_UNAVAILABLE = "nlm_unavailable"

ERROR_MESSAGES = {
    ErrorType.RATE_LIMITED: (
        "⏳ You're sending too many requests. "
        "Please wait a moment and try again."
    ),
    ErrorType.UNAUTHORIZED: (
        "🔐 Authentication required. "
        "Please sign in to use this bot."
    ),
    ErrorType.NO_ACCESS: (
        "🔒 You don't have access to any knowledge bases. "
        "Contact your administrator for access."
    ),
    ErrorType.NOTEBOOK_NOT_FOUND: (
        "📚 I couldn't find relevant information in your accessible knowledge bases. "
        "Try rephrasing your question or check if you have access to the right notebooks."
    ),
    ErrorType.INTERNAL_ERROR: (
        "⚠️ Something went wrong. Please try again later. "
        "If the problem persists, contact support."
    ),
    ErrorType.NLM_UNAVAILABLE: (
        "🔧 The knowledge service is temporarily unavailable. "
        "Please try again in a few minutes."
    ),
}
```

### Retry Logic

```python
# utils/retry.py

import asyncio
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        if exponential:
                            delay = min(delay * 2, max_delay)

            raise last_exception
        return wrapper
    return decorator
```

---

## Deployment

### Dockerfile

```dockerfile
# deploy/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ src/
COPY config/ config/

# Create non-root user
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "-m", "nlm_chatbot.main"]
```

### Docker Compose (Development)

```yaml
# deploy/docker-compose.yml

version: '3.8'

services:
  bot:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
    ports:
      - "3978:3978"
    environment:
      - MICROSOFT_APP_ID=${MICROSOFT_APP_ID}
      - MICROSOFT_APP_PASSWORD=${MICROSOFT_APP_PASSWORD}
      - MICROSOFT_APP_TENANT_ID=${MICROSOFT_APP_TENANT_ID}
      - NLM_PROXY_BASE_URL=http://nlm-proxy:8080/v1
      - NLM_PROXY_API_KEY=${NLM_PROXY_API_KEY}
      - REDIS_URL=redis://redis:6379
      - GRAPH_CLIENT_ID=${GRAPH_CLIENT_ID}
      - GRAPH_CLIENT_SECRET=${GRAPH_CLIENT_SECRET}
    depends_on:
      - redis
      - nlm-proxy
    volumes:
      - ./config:/app/config:ro

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nlm-proxy:
    image: nlm-proxy:latest
    ports:
      - "8080:8080"
    environment:
      - NLM_PROXY_OPENAI_API_KEY=${NLM_PROXY_API_KEY}
      # ... other nlm-proxy env vars

volumes:
  redis-data:
```

### Kubernetes Deployment

```yaml
# deploy/k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlm-chatbot
  labels:
    app: nlm-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlm-chatbot
  template:
    metadata:
      labels:
        app: nlm-chatbot
    spec:
      containers:
        - name: bot
          image: your-registry/nlm-chatbot:latest
          ports:
            - containerPort: 3978
          envFrom:
            - secretRef:
                name: nlm-chatbot-secrets
            - configMapRef:
                name: nlm-chatbot-config
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 3978
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 3978
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: nlm-chatbot
spec:
  type: ClusterIP
  selector:
    app: nlm-chatbot
  ports:
    - port: 3978
      targetPort: 3978
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nlm-chatbot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - bot.yourcompany.com
      secretName: nlm-chatbot-tls
  rules:
    - host: bot.yourcompany.com
      http:
        paths:
          - path: /api/messages
            pathType: Prefix
            backend:
              service:
                name: nlm-chatbot
                port:
                  number: 3978
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MICROSOFT_APP_ID` | Yes | - | Azure Bot registration App ID |
| `MICROSOFT_APP_PASSWORD` | Yes | - | Azure Bot registration password |
| `MICROSOFT_APP_TENANT_ID` | Yes | - | Azure AD tenant ID |
| `NLM_PROXY_BASE_URL` | No | `http://localhost:8080/v1` | nlm-proxy endpoint |
| `NLM_PROXY_API_KEY` | Yes | - | API key for nlm-proxy |
| `NLM_PROXY_MODEL` | No | `knowledge-finder` | Model name for smart routing |
| `GRAPH_CLIENT_ID` | Yes | - | Azure AD app for Graph API |
| `GRAPH_CLIENT_SECRET` | Yes | - | Graph API client secret |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection URL |
| `ACL_CONFIG_PATH` | No | `config/acl.yaml` | Path to ACL configuration |
| `ACL_CACHE_TTL` | No | `300` | ACL cache TTL in seconds |
| `MAX_CONCURRENT_REQUESTS` | No | `10` | Global concurrent request limit |
| `USER_RATE_LIMIT` | No | `30` | Per-user requests per minute |
| `SHOW_SOURCE_NOTEBOOK` | No | `true` | Show source notebook in responses |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### `.env.example`

```bash
# Azure Bot Registration
MICROSOFT_APP_ID=your-bot-app-id
MICROSOFT_APP_PASSWORD=your-bot-password
MICROSOFT_APP_TENANT_ID=your-tenant-id

# nlm-proxy
NLM_PROXY_BASE_URL=http://localhost:8080/v1
NLM_PROXY_API_KEY=your-nlm-proxy-key
NLM_PROXY_MODEL=knowledge-finder

# Azure AD / Graph API (for group membership)
GRAPH_CLIENT_ID=your-graph-app-id
GRAPH_CLIENT_SECRET=your-graph-secret

# Redis (for session & ACL caching)
REDIS_URL=redis://localhost:6379

# ACL
ACL_CONFIG_PATH=config/acl.yaml
ACL_CACHE_TTL=300

# Rate Limiting
MAX_CONCURRENT_REQUESTS=10
USER_RATE_LIMIT=30

# Features
SHOW_SOURCE_NOTEBOOK=true

# Logging
LOG_LEVEL=INFO
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **1.1 Project Setup** | Initialize repo, dependencies, structure | `pyproject.toml`, folder structure |
| **1.2 Configuration** | Pydantic settings, env loading | `config.py`, `.env.example` |
| **1.3 Bot Scaffolding** | Basic Bot Framework setup | `bot.py`, `main.py` |
| **1.4 nlm-proxy Client** | OpenAI SDK client wrapper | `nlm/client.py` |
| **1.5 Local Testing** | Bot Framework Emulator testing | Test scripts |

### Phase 2: Teams Integration (Week 2-3)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **2.1 Azure Bot Registration** | Create Bot in Azure Portal | Bot registration, App ID |
| **2.2 Teams Channel** | Enable Teams channel | Channel configuration |
| **2.3 SSO Setup** | Configure OAuth connection | Graph connection |
| **2.4 User Info Extraction** | Get user identity from turn context | `auth/azure_ad.py` |
| **2.5 Adaptive Cards** | Teams-specific formatting | `channels/teams.py` |

### Phase 3: ACL Implementation (Week 3-4)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **3.1 ACL Configuration** | YAML-based ACL config | `config/acl.yaml` |
| **3.2 Group Lookup** | Microsoft Graph integration | `auth/graph_client.py` |
| **3.3 ACL Service** | Group → notebook mapping | `acl/service.py` |
| **3.4 nlm-proxy Enhancement** | Add notebook filtering support | PR to nlm-proxy |
| **3.5 Caching** | Redis integration for ACL cache | Redis setup |

### Phase 4: Production Hardening (Week 4-5)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **4.1 Rate Limiting** | Implement rate limiter | `utils/rate_limiter.py` |
| **4.2 Error Handling** | User-friendly error messages | Error handling throughout |
| **4.3 Logging** | Structured logging (structlog) | Logging configuration |
| **4.4 Health Checks** | Kubernetes health endpoints | `/health`, `/ready` endpoints |
| **4.5 Dockerization** | Dockerfile, docker-compose | `deploy/` folder |

### Phase 5: Telegram Channel (Week 5-6)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **5.1 Telegram Bot Setup** | Create bot via BotFather | Bot token |
| **5.2 OAuth Flow** | Telegram ↔ AD linking | `auth/telegram_oauth.py` |
| **5.3 Account Linking** | Redis-based account mapping | Linking endpoints |
| **5.4 Telegram Formatter** | Markdown V2 formatting | `channels/telegram.py` |
| **5.5 Integration Testing** | End-to-end Telegram tests | Test suite |

### Phase 6: Deployment & Monitoring (Week 6-7)

| Task | Description | Deliverables |
|------|-------------|--------------|
| **6.1 K8s Manifests** | Deployment, Service, Ingress | `deploy/k8s/` |
| **6.2 Secrets Management** | K8s secrets or external vault | Secrets configuration |
| **6.3 Monitoring** | Prometheus metrics, dashboards | Metrics endpoints |
| **6.4 Documentation** | README, deployment guide | Documentation |
| **6.5 Production Deploy** | Deploy to production cluster | Running bot |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_acl.py

import pytest
from nlm_chatbot.acl.service import ACLService

@pytest.fixture
def acl_service(tmp_path):
    config = tmp_path / "acl.yaml"
    config.write_text("""
notebooks:
  - id: "notebook-1"
    name: "HR Docs"
    allowed_groups: ["sg-hr", "sg-admins"]
  - id: "notebook-2"
    name: "Tech Docs"
    allowed_groups: ["sg-engineering"]
  - id: "notebook-3"
    name: "Public"
    allowed_groups: ["*"]
""")
    return ACLService(str(config))

def test_user_with_hr_group_gets_hr_notebook(acl_service):
    allowed = acl_service.get_allowed_notebooks(["sg-hr"])
    assert "notebook-1" in allowed
    assert "notebook-3" in allowed  # wildcard
    assert "notebook-2" not in allowed

def test_user_with_multiple_groups(acl_service):
    allowed = acl_service.get_allowed_notebooks(["sg-hr", "sg-engineering"])
    assert "notebook-1" in allowed
    assert "notebook-2" in allowed
    assert "notebook-3" in allowed

def test_user_with_no_matching_groups(acl_service):
    allowed = acl_service.get_allowed_notebooks(["sg-sales"])
    assert allowed == ["notebook-3"]  # only wildcard
```

### Integration Tests

```python
# tests/test_nlm_client.py

import pytest
from nlm_chatbot.nlm.client import NLMProxyClient

@pytest.mark.asyncio
async def test_query_with_session_continuity(mock_nlm_proxy):
    client = NLMProxyClient(
        base_url=mock_nlm_proxy.url,
        api_key="test-key",
    )

    # First query
    response1 = await client.query(
        message="What is the vacation policy?",
        conversation_id="conv-123",
    )
    assert response1.content
    assert response1.notebook_name

    # Follow-up query (same conversation)
    response2 = await client.query(
        message="How many days?",
        conversation_id="conv-123",
    )
    assert response2.content

    await client.close()
```

### End-to-End Tests

```python
# tests/test_e2e.py

import pytest
from botbuilder.testing import DialogTestClient
from nlm_chatbot.bot.bot import NotebookLMBot

@pytest.mark.asyncio
async def test_full_query_flow(bot: NotebookLMBot):
    client = DialogTestClient("test", bot)

    # Send message
    reply = await client.send_activity("What is our vacation policy?")

    # Verify response
    assert reply.text is not None
    assert "Source:" in reply.text  # Source attribution
    assert "HR" in reply.text or "vacation" in reply.text.lower()
```

---

## Monitoring & Observability

### Metrics

```python
# metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUESTS_TOTAL = Counter(
    "nlm_chatbot_requests_total",
    "Total requests",
    ["channel", "status"]
)

REQUEST_DURATION = Histogram(
    "nlm_chatbot_request_duration_seconds",
    "Request duration",
    ["channel"]
)

# Rate limiting
RATE_LIMITED_TOTAL = Counter(
    "nlm_chatbot_rate_limited_total",
    "Rate limited requests",
    ["user_id"]
)

# ACL
ACL_DENIALS_TOTAL = Counter(
    "nlm_chatbot_acl_denials_total",
    "ACL denials",
    ["user_group"]
)

# Active sessions
ACTIVE_SESSIONS = Gauge(
    "nlm_chatbot_active_sessions",
    "Active conversation sessions"
)
```

### Logging

```python
# logging_config.py

import structlog

def configure_logging(log_level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

# Usage
logger = structlog.get_logger()

async def on_message_activity(self, turn_context):
    logger.info(
        "message_received",
        user_id=turn_context.activity.from_property.id,
        channel=turn_context.activity.channel_id,
        message_length=len(turn_context.activity.text),
    )
```

### Health Endpoints

```python
# health.py

from aiohttp import web

async def health_check(request):
    """Kubernetes liveness probe."""
    return web.json_response({"status": "healthy"})

async def ready_check(request):
    """Kubernetes readiness probe."""
    # Check dependencies
    checks = {
        "redis": await check_redis(),
        "nlm_proxy": await check_nlm_proxy(),
    }

    all_healthy = all(checks.values())
    status = 200 if all_healthy else 503

    return web.json_response(
        {"status": "ready" if all_healthy else "not_ready", "checks": checks},
        status=status,
    )
```

---

## Appendix

### Azure Bot Registration Steps

1. Go to Azure Portal → Create Resource → Azure Bot
2. Fill in:
   - Bot handle: `nlm-chatbot`
   - Subscription: Your subscription
   - Resource group: Create or select
   - Pricing tier: Standard
   - Microsoft App ID: Create new
3. After creation:
   - Go to Channels → Add Microsoft Teams
   - Go to Configuration → Messaging endpoint: `https://your-domain/api/messages`
   - Note the App ID and create a secret

### Microsoft Graph Permissions

Required permissions for the Graph API app:
- `User.Read` - Read user profile
- `GroupMember.Read.All` - Read group memberships
- `Directory.Read.All` - Read directory data (optional, for group names)

Grant admin consent for these permissions.

### Telegram Bot Setup

1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow prompts to create bot
4. Save the bot token
5. Configure webhook: `https://your-domain/telegram/webhook`

---

## References

- [Bot Framework Python SDK](https://github.com/microsoft/botbuilder-python)
- [Azure Bot Service Documentation](https://docs.microsoft.com/en-us/azure/bot-service/)
- [Microsoft Graph API](https://docs.microsoft.com/en-us/graph/)
- [nlm-proxy Smart Routing](./smart-routing-architecture.md)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
