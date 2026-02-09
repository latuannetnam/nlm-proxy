# nlm-proxy Account Pool Enhancement Specification

> **Type**: Feature Enhancement for nlm-proxy
> **Status**: Ready for Implementation
> **Priority**: Required before Chatbot development
> **Estimated Effort**: 3-4 days
> **Target**: nlm-proxy repository

---

## Summary

Add multi-account load balancing to nlm-proxy, distributing requests across multiple NotebookLM service accounts to handle high concurrency without hitting rate limits.

## Motivation

- NotebookLM has undocumented rate limits
- Single account cannot reliably handle 50-500 concurrent users
- Need automatic failover when accounts are rate-limited
- All consumers (chatbot, Open WebUI, CLI) benefit from this enhancement

## Design Decision

**Load balancing is implemented in nlm-proxy**, not in consumers, because:

| Reason | Benefit |
|--------|---------|
| Clean API | Consumers just call `/v1/chat/completions`, no account awareness |
| Credential isolation | NLM cookies stay in nlm-proxy, not exposed to consumers |
| Reusability | All consumers benefit automatically |
| Single responsibility | nlm-proxy owns NotebookLM infrastructure |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              nlm-proxy                                       │
│                                                                              │
│  POST /v1/chat/completions                                                   │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Account Pool Manager                              │    │
│  │                                                                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │ Account A   │  │ Account B   │  │ Account C   │                  │    │
│  │  │ status: ✅  │  │ status: ✅  │  │ status: ⏳  │                  │    │
│  │  │ requests:45 │  │ requests:32 │  │ cooldown:4m │                  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘                  │    │
│  │         │                │                                           │    │
│  │         └────────────────┤                                           │    │
│  │                          ▼                                           │    │
│  │              Select healthy account (round-robin)                    │    │
│  │                          │                                           │    │
│  └──────────────────────────┼───────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│                    NotebookLMClient(account.cookies)                         │
│                             │                                                │
│                             ▼                                                │
│                        NotebookLM API                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API (No Changes for Consumers)

The API remains unchanged. Consumers don't know about multiple accounts:

```bash
# Same API as before - pool is transparent
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "knowledge-finder", "messages": [...]}'
```

### New Response Headers (Optional)

```
X-NLM-Account-Used: account-a
X-NLM-Pool-Healthy: 3
X-NLM-Pool-Total: 4
```

### New Endpoints (Admin)

```
GET /v1/pool/stats      # Pool statistics
GET /v1/pool/accounts   # Account status list
POST /v1/pool/accounts/{id}/disable   # Disable account
POST /v1/pool/accounts/{id}/enable    # Enable account
```

---

## Configuration

### Account Configuration File

```yaml
# config/accounts.yaml (or ~/.nlm-proxy/accounts.yaml)

accounts:
  - id: account-a
    name: "Service Account A"
    cookies_env: NLM_ACCOUNT_A_COOKIES  # Reference env var

  - id: account-b
    name: "Service Account B"
    cookies_env: NLM_ACCOUNT_B_COOKIES

  - id: account-c
    name: "Service Account C"
    cookies_env: NLM_ACCOUNT_C_COOKIES

pool:
  selection_strategy: round-robin  # round-robin | least-used | random
  cooldown_seconds: 300            # 5 min cooldown on rate limit
  health_check_interval: 60        # Check unhealthy accounts every 60s
  max_retries: 2                   # Retry with different account on failure
```

### Environment Variables

```bash
# Account credentials (one per account)
NLM_ACCOUNT_A_COOKIES="cookie-string-for-account-a"
NLM_ACCOUNT_B_COOKIES="cookie-string-for-account-b"
NLM_ACCOUNT_C_COOKIES="cookie-string-for-account-c"

# Pool settings (optional, defaults shown)
NLM_PROXY_POOL_ENABLED=true
NLM_PROXY_POOL_CONFIG_PATH=~/.nlm-proxy/accounts.yaml
NLM_PROXY_POOL_COOLDOWN_SECONDS=300
NLM_PROXY_POOL_HEALTH_CHECK_INTERVAL=60
```

### Backward Compatibility

If `NLM_PROXY_POOL_ENABLED=false` or no accounts config exists, nlm-proxy uses the existing single-account mode with `NLM_PROXY_COOKIES`.

---

## Implementation

### File Structure

```
src/nlm_proxy/
├── core/
│   ├── pool/
│   │   ├── __init__.py
│   │   ├── manager.py      # AccountPoolManager
│   │   ├── models.py       # NLMAccount, AccountStatus, PoolStats
│   │   └── config.py       # Load accounts from YAML/env
│   └── client.py           # NotebookLMClient (unchanged)
└── openai/
    └── server.py           # Updated to use pool
```

### Key Classes

#### AccountStatus Enum

```python
# core/pool/models.py

from enum import Enum

class AccountStatus(Enum):
    HEALTHY = "healthy"        # Ready for requests
    COOLDOWN = "cooldown"      # Rate limited, waiting
    UNHEALTHY = "unhealthy"    # Failed health checks
    DISABLED = "disabled"      # Manually disabled
```

#### NLMAccount Model

```python
# core/pool/models.py

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class NLMAccount:
    id: str
    name: str
    cookies: str
    csrf_token: str = ""
    session_id: str = ""

    # Runtime state
    status: AccountStatus = AccountStatus.HEALTHY
    request_count: int = 0
    cooldown_until: datetime | None = None
    last_used: datetime | None = None
    consecutive_failures: int = 0
```

#### AccountPoolManager

```python
# core/pool/manager.py

class AccountPoolManager:
    """Manages pool of NotebookLM accounts."""

    def __init__(self, accounts: list[NLMAccount], config: PoolConfig):
        self.accounts = {acc.id: acc for acc in accounts}
        self.config = config
        self._lock = asyncio.Lock()
        self._index = 0  # For round-robin

    async def acquire(self) -> NLMAccount | None:
        """Get a healthy account for a request."""
        async with self._lock:
            self._clear_expired_cooldowns()
            healthy = [a for a in self.accounts.values()
                       if a.status == AccountStatus.HEALTHY]
            if not healthy:
                return None

            # Round-robin selection
            self._index = (self._index + 1) % len(healthy)
            account = healthy[self._index]
            account.request_count += 1
            account.last_used = datetime.utcnow()
            return account

    async def report_success(self, account_id: str):
        """Report successful request."""
        async with self._lock:
            if account_id in self.accounts:
                self.accounts[account_id].consecutive_failures = 0

    async def report_rate_limited(self, account_id: str):
        """Put account in cooldown after 429."""
        async with self._lock:
            if account_id in self.accounts:
                acc = self.accounts[account_id]
                acc.status = AccountStatus.COOLDOWN
                acc.cooldown_until = datetime.utcnow() + timedelta(
                    seconds=self.config.cooldown_seconds
                )

    async def report_failure(self, account_id: str):
        """Report failed request (non-429)."""
        async with self._lock:
            if account_id in self.accounts:
                acc = self.accounts[account_id]
                acc.consecutive_failures += 1
                if acc.consecutive_failures >= 3:
                    acc.status = AccountStatus.UNHEALTHY

    async def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        ...
```

### Server Integration

```python
# openai/server.py

# At startup
if settings.pool_enabled:
    accounts = load_accounts_from_config()
    app.state.pool_manager = AccountPoolManager(accounts, pool_config)
else:
    app.state.pool_manager = None  # Single account mode

# Per request
async def get_nlm_client() -> NotebookLMClient:
    if app.state.pool_manager:
        account = await app.state.pool_manager.acquire()
        if not account:
            raise HTTPException(503, "All accounts busy, try later")
        return NotebookLMClient(
            cookies=account.cookies,
            csrf_token=account.csrf_token,
            session_id=account.session_id,
        ), account.id
    else:
        # Existing single-account logic
        tokens = load_cached_tokens()
        return NotebookLMClient(cookies=tokens.cookies, ...), None

# In request handler
async def handle_request(...):
    client, account_id = await get_nlm_client()
    try:
        result = await client.query(...)
        if account_id:
            await app.state.pool_manager.report_success(account_id)
        return result
    except RateLimitError:
        if account_id:
            await app.state.pool_manager.report_rate_limited(account_id)
        # Retry with different account...
    except Exception as e:
        if account_id:
            await app.state.pool_manager.report_failure(account_id)
        raise
```

---

## Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Request Flow                                    │
│                                                                              │
│  1. Request arrives at /v1/chat/completions                                 │
│       │                                                                      │
│       ▼                                                                      │
│  2. pool_manager.acquire() → Select healthy account                         │
│       │                                                                      │
│       ├── No healthy accounts? → 503 "All accounts busy"                    │
│       │                                                                      │
│       ▼                                                                      │
│  3. Create NotebookLMClient with selected account                           │
│       │                                                                      │
│       ▼                                                                      │
│  4. Execute query                                                            │
│       │                                                                      │
│       ├── Success → report_success() → Return response                      │
│       │                                                                      │
│       ├── 429 Rate Limited → report_rate_limited()                          │
│       │       │                                                              │
│       │       └── Retry with different account (up to max_retries)          │
│       │                                                                      │
│       └── Other Error → report_failure() → Return error                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nlm_pool_accounts_total` | Gauge | Total accounts |
| `nlm_pool_accounts_healthy` | Gauge | Healthy accounts |
| `nlm_pool_accounts_cooldown` | Gauge | Accounts in cooldown |
| `nlm_pool_requests_total` | Counter | Total requests |
| `nlm_pool_rate_limits_total` | Counter | 429 responses |
| `nlm_pool_retries_total` | Counter | Retry attempts |
| `nlm_pool_exhausted_total` | Counter | No accounts available |

### OpenTelemetry Spans

```
nlm_pool.acquire
  ├── account_id: "account-a"
  ├── pool_healthy_count: 3
  └── pool_total_count: 4

nlm_pool.rate_limited
  ├── account_id: "account-b"
  └── cooldown_seconds: 300
```

---

## Testing

### Unit Tests

```python
@pytest.mark.asyncio
async def test_acquire_returns_healthy_account():
    accounts = [create_account("a"), create_account("b")]
    pool = AccountPoolManager(accounts, config)

    account = await pool.acquire()
    assert account is not None
    assert account.status == AccountStatus.HEALTHY

@pytest.mark.asyncio
async def test_acquire_skips_cooldown_accounts():
    accounts = [create_account("a"), create_account("b")]
    pool = AccountPoolManager(accounts, config)

    await pool.report_rate_limited("a")

    account = await pool.acquire()
    assert account.id == "b"

@pytest.mark.asyncio
async def test_acquire_returns_none_when_all_in_cooldown():
    accounts = [create_account("a")]
    pool = AccountPoolManager(accounts, config)

    await pool.report_rate_limited("a")

    account = await pool.acquire()
    assert account is None
```

### Integration Tests

```bash
# Start nlm-proxy with pool enabled
NLM_PROXY_POOL_ENABLED=true nlm-proxy serve openai --port 8080

# Send concurrent requests
for i in {1..20}; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer $API_KEY" \
    -d '{"model": "knowledge-finder", "messages": [...]}' &
done
wait

# Check pool stats
curl http://localhost:8080/v1/pool/stats
```

---

## Setup: Multiple NotebookLM Accounts

### 1. Create Google Accounts

Create or use existing Google accounts for each service account:
- nlm-service-a@gmail.com
- nlm-service-b@gmail.com
- nlm-service-c@gmail.com

### 2. Share Notebooks

In NotebookLM, share each notebook with all service accounts (Editor or Viewer access).

### 3. Extract Credentials

For each account:

```bash
# Login as the account, then run:
nlm-proxy auth extract

# Save the output to environment variable
export NLM_ACCOUNT_A_COOKIES="<output>"
```

### 4. Create Config File

```yaml
# ~/.nlm-proxy/accounts.yaml
accounts:
  - id: account-a
    name: "Service A"
    cookies_env: NLM_ACCOUNT_A_COOKIES
  - id: account-b
    name: "Service B"
    cookies_env: NLM_ACCOUNT_B_COOKIES
```

### 5. Start with Pool Enabled

```bash
NLM_PROXY_POOL_ENABLED=true nlm-proxy serve openai --port 8080
```

---

## Checklist

- [ ] Create `core/pool/models.py` with data models
- [ ] Create `core/pool/manager.py` with AccountPoolManager
- [ ] Create `core/pool/config.py` for loading accounts
- [ ] Update `openai/server.py` to use pool
- [ ] Add retry logic for rate-limited accounts
- [ ] Add pool stats endpoint `/v1/pool/stats`
- [ ] Add response headers with account info
- [ ] Add OpenTelemetry spans for pool operations
- [ ] Add Prometheus metrics
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Update `.env.example`

---

## Commit Message

```
feat(pool): add multi-account load balancing for NotebookLM

Implement AccountPoolManager to distribute requests across multiple
NotebookLM service accounts:

- Round-robin selection among healthy accounts
- Automatic cooldown (5 min) on rate limit (429)
- Health checks for unhealthy accounts
- Retry with different account on failure
- Pool stats endpoint for monitoring
- Backward compatible: single-account mode still works

This enables nlm-proxy to handle 50-500+ concurrent users by
distributing load across multiple NotebookLM accounts.

Co-Authored-By: Claude <noreply@anthropic.com>
```
