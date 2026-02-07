# OTel Tracing Authentication & TLS Enhancement

## Context

The current tracing implementation uses gRPC with `insecure=True`, providing no encryption or authentication. For production deployments with self-hosted OTel Collectors, we need:
- TLS encryption with private CA certificate support
- Bearer token authentication via headers
- Flexibility to skip certificate validation for development
- Protocol choice (gRPC vs HTTP) since gRPC Python doesn't support "TLS + skip verify"

## Configuration Changes

**New settings in `TracingSettings` (`src/nlm_proxy/core/config.py`):**

| Setting | Env Variable | Type | Default | Purpose |
|---------|--------------|------|---------|---------|
| `protocol` | `NLM_PROXY_OTEL_PROTOCOL` | `Literal["grpc", "http"]` | `grpc` | Exporter protocol |
| `api_key` | `NLM_PROXY_OTEL_API_KEY` | `str \| None` | `None` | Bearer token for auth |
| `ca_cert_path` | `NLM_PROXY_OTEL_CA_CERT_PATH` | `str \| None` | `None` | Private CA certificate path |
| `verify_cert` | `NLM_PROXY_OTEL_VERIFY_CERT` | `bool` | `True` | Validate server certificate (HTTP only) |

**Existing setting behavior:**
- `insecure` (existing): `True` = plain text, `False` = TLS enabled

## Behavior Matrix

| Protocol | `insecure` | `verify_cert` | `ca_cert_path` | Result |
|----------|------------|---------------|----------------|--------|
| gRPC | `True` | (ignored) | (ignored) | Plain text |
| gRPC | `False` | (ignored, always validates) | Not set | TLS + system CA |
| gRPC | `False` | (ignored) | Set | TLS + private CA |
| HTTP | `True` | (ignored) | (ignored) | Plain text (`http://`) |
| HTTP | `False` | `False` | (ignored) | TLS, skip validation |
| HTTP | `False` | `True` | Not set | TLS + system CA |
| HTTP | `False` | `True` | Set | TLS + private CA |

**Note:** gRPC Python does not support "TLS + skip verify" - will log warning if attempted.

## Implementation Plan

### Step 1: Update Configuration (`src/nlm_proxy/core/config.py`)

Add new fields to `TracingSettings`:
```python
protocol: Literal["grpc", "http"] = "grpc"
api_key: str | None = None
ca_cert_path: str | None = None
verify_cert: bool = True
```

### Step 2: Add HTTP Exporter Dependency (`pyproject.toml`)

Add to tracing optional dependencies:
```toml
"opentelemetry-exporter-otlp-proto-http",
```

### Step 3: Refactor Exporter Creation (`src/nlm_proxy/core/tracing.py`)

Create helper functions:
- `_create_exporter(settings)` - routes to correct exporter
- `_create_http_exporter(settings, headers)` - HTTP with full TLS options
- `_create_grpc_exporter(settings, headers)` - gRPC with credentials

**HTTP exporter:** Construct full URL from `host:port`:
- `insecure=True` → `http://{endpoint}/v1/traces`
- `insecure=False` → `https://{endpoint}/v1/traces`

**Headers:** Add `Authorization: Bearer {api_key}` when configured.

### Step 4: Add Validation & Logging

Startup validation:
- `ca_cert_path` set but file doesn't exist → raise `FileNotFoundError`
- `ca_cert_path` set but `verify_cert=False` → log warning
- `protocol=grpc` + `verify_cert=False` + `insecure=False` → log warning (not supported)

Logging:
- Log TLS mode and CA source on init
- Log if authentication is enabled

### Step 5: Update Documentation

Files to update:
- `.env.example` - add new env vars
- `.claude/memory/configuration.md` - document new settings
- `.claude/memory/tracing.md` - add TLS/auth section

## Critical Files

- `src/nlm_proxy/core/config.py` - TracingSettings class
- `src/nlm_proxy/core/tracing.py` - init_tracing(), exporter creation
- `pyproject.toml` - dependencies

## Verification

1. **Backward compatibility:** Default config (`protocol=grpc`, `insecure=True`) works unchanged
2. **HTTP + skip verify:** Set `protocol=http`, `insecure=false`, `verify_cert=false`
3. **HTTP + private CA:** Set `protocol=http`, `insecure=false`, `ca_cert_path=/path/to/ca.pem`
4. **gRPC + private CA:** Set `protocol=grpc`, `insecure=false`, `ca_cert_path=/path/to/ca.pem`
5. **Auth header:** Set `api_key=test-token`, verify header sent to collector
6. **Validation:** Test error on missing cert file, warning on invalid combos
