# OpenTelemetry Tracing Guide

This guide explains how to set up and operate the OpenTelemetry tracing feature for NLM Proxy. Tracing provides visibility into request routing decisions, helping you understand how queries are classified and which notebooks are selected.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Infrastructure Setup](#infrastructure-setup)
- [Understanding Traces](#understanding-traces)
- [Querying Trace Data](#querying-trace-data)
- [Troubleshooting](#troubleshooting)

## Overview

The tracing feature instruments the Smart Router to capture:

- **Request classification**: Whether a query is routed to NotebookLM or an external LLM
- **Notebook selection**: Which notebook was chosen and why
- **Timing data**: How long each operation takes
- **Request attributes**: Query text, notebook IDs, and routing decisions
- **Response content**: LLM and NotebookLM responses (truncated, configurable)

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   NLM Proxy     │────▶│  OTel Collector  │────▶│  ClickHouse │
│  (OTLP export)  │     │  (batch + route) │     │  (storage)  │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

**Components:**
- **NLM Proxy**: Exports spans via OTLP gRPC
- **OpenTelemetry Collector**: Receives, batches, and exports traces
- **ClickHouse**: Stores traces with 90-day retention

## Quick Start

### 1. Start the Tracing Infrastructure

```bash
# From the nlm-proxy directory
docker compose -f docker-compose.otel.yml up -d
```

This starts:
- ClickHouse on ports 8123 (HTTP) and 9000 (native)
- OTel Collector on ports 4317 (gRPC) and 4318 (HTTP)

### 2. Enable Tracing

Add to your `.env` file or export as environment variables:

```bash
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
```

### 3. Start NLM Proxy

```bash
nlm-proxy serve openai --port 8080
```

You should see in the logs:
```
[TRACING] OpenTelemetry initialized: endpoint=http://localhost:4317, service=nlm-proxy
[TRACING] FastAPI instrumentation enabled
[TRACING] httpx instrumentation enabled
```

### 4. Send a Test Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

### 5. View Traces

```bash
docker exec -it nlm-clickhouse /bin/bash
docker exec -it nlm-clickhouse clickhouse-client --query \
  "SELECT SpanName, Duration/1000000 as duration_ms
   FROM nlm_traces.otel_traces
   ORDER BY Timestamp DESC
   LIMIT 10 FORMAT Pretty"
```

**Note:** The ClickHouse exporter automatically creates the `otel_traces` table with an optimized schema for OpenTelemetry data.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_OTEL_ENABLED` | `false` | Enable/disable tracing |
| `NLM_PROXY_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint (gRPC) |
| `NLM_PROXY_OTEL_SERVICE_NAME` | `nlm-proxy` | Service name in traces |
| `NLM_PROXY_OTEL_REQUEST_MAX_LENGTH` | `500` | Max chars of user query to store in trace (0 to disable) |
| `NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH` | `1000` | Max chars of response to store in trace (0 to disable) |
| `NLM_PROXY_OTEL_PROTOCOL` | `grpc` | Protocol: `grpc` or `http` |
| `NLM_PROXY_OTEL_INSECURE` | `true` | `true`=plain text, `false`=TLS enabled |
| `NLM_PROXY_OTEL_VERIFY_CERT` | `true` | Skip certificate validation (HTTP only) |
| `NLM_PROXY_OTEL_CA_CERT_PATH` | - | Path to CA certificate for TLS verification |
| `NLM_PROXY_OTEL_API_KEY` | - | Bearer token for collector authentication |

### Configuration File

Add to `.env` or `~/.nlm-proxy/.env`:

```bash
# OpenTelemetry Tracing
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
```

### Programmatic Configuration

```python
from nlm_proxy.core.config import TracingSettings

settings = TracingSettings(
    enabled=True,
    endpoint="http://collector:4317",
    service_name="my-nlm-proxy"
)
```

## Infrastructure Setup

### Docker Compose (Recommended)

The provided `docker-compose.otel.yml` sets up everything:

```bash
# Start services
docker compose -f docker-compose.otel.yml up -d

# Check status
docker compose -f docker-compose.otel.yml ps

# View collector logs
docker logs nlm-otel-collector

# View ClickHouse logs
docker logs nlm-clickhouse

# Stop services
docker compose -f docker-compose.otel.yml down

# Stop and remove data
docker compose -f docker-compose.otel.yml down -v
```

### Manual Setup

#### ClickHouse

1. Start ClickHouse:
```bash
docker run -d --name clickhouse \
  -p 8123:8123 -p 9000:9000 \
  -v clickhouse_data:/var/lib/clickhouse \
  clickhouse/clickhouse-server:24.1
```

2. Create the schema:
```bash
docker exec -i clickhouse clickhouse-client < docker/clickhouse/init.sql
```

#### OTel Collector

1. Create config file (see `docker/otel/config.yaml`)

2. Start collector:
```bash
docker run -d --name otel-collector \
  -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/docker/otel/config.yaml:/etc/otelcol-contrib/config.yaml:ro \
  otel/opentelemetry-collector-contrib:0.96.0 \
  --config=/etc/otelcol-contrib/config.yaml
```

### Cloud Deployment

For production, consider:

- **ClickHouse Cloud**: Managed ClickHouse service
- **Grafana Tempo**: Alternative trace storage
- **Jaeger**: Open-source tracing backend
- **AWS X-Ray**: AWS-native tracing

Update the collector config to export to your chosen backend.

### Secure Setup with TLS and Authentication

For production deployments or when the collector is network-accessible, enable TLS encryption and bearer token authentication.

#### Docker Setup (Recommended)

**Step 1: Generate TLS Certificates**

```bash
# From the nlm-proxy directory
bash docker/otel/generate-certs.sh
```

This creates self-signed certificates in `docker/otel/certs/`:
- `ca.crt` - CA certificate (for client verification)
- `server.crt` - Server certificate
- `server.key` - Server private key

For production, replace with certificates from a trusted CA (Let's Encrypt, DigiCert, etc.).

**Step 2: Generate Bearer Token**

```bash
# Generate a secure random token
openssl rand -base64 32
```

**Step 3: Configure Environment**

Create `docker/otel/.env`:

```bash
# Strong bearer token (must match on client and collector)
OTEL_BEARER_TOKEN=<token-from-step-2>

# Optional: Custom server name for production
OTEL_SERVER_NAME=localhost
```

**Step 4: Start the Secure Stack**

```bash
# Start with secure configuration
docker compose -f docker-compose.otel-secure.yml up -d

# Check collector logs
docker logs nlm-otel-collector

# Verify TLS is enabled (should show "tls" in config)
docker exec nlm-otel-collector cat /etc/otelcol-contrib/config.yaml | grep -A 2 "tls:"
```

**Step 5: Configure NLM Proxy Client**

Add to `.env` or `~/.nlm-proxy/.env`:

```bash
# Enable tracing with TLS and authentication
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy

# TLS configuration
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/docker/otel/certs/ca.crt

# Authentication
NLM_PROXY_OTEL_API_KEY=<same-token-from-step-2>
```

**Step 6: Test the Setup**

```bash
# Start NLM Proxy
nlm-proxy serve openai --port 8080

# Expected in logs:
# [TRACING] OpenTelemetry initialized: endpoint=localhost:4317, service=nlm-proxy

# Send test request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "knowledge-finder", "messages": [{"role": "user", "content": "test"}]}'

# Verify traces
docker exec -it nlm-clickhouse clickhouse-client --query \
  "SELECT COUNT(*) FROM nlm_traces.otel_traces WHERE Timestamp > now() - INTERVAL 1 MINUTE"
```

#### Native Setup (Ubuntu/Linux)

**Step 1: Generate Certificates**

```bash
# Create certificate directory
sudo mkdir -p /etc/otelcol-contrib/certs
cd /etc/otelcol-contrib/certs

# Generate CA private key
sudo openssl genrsa -out ca.key 4096

# Generate CA certificate
sudo openssl req -new -x509 -days 365 -key ca.key -out ca.crt \
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=NLM Proxy CA"

# Generate server private key
sudo openssl genrsa -out server.key 4096

# Generate server CSR
sudo openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=localhost"

# Create SAN configuration
cat <<EOF | sudo tee san.cnf
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF

# Sign server certificate
sudo openssl x509 -req -days 365 -in server.csr \
  -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -extensions v3_req -extfile san.cnf

# Set permissions
sudo chown -R otelcol-contrib:otelcol-contrib /etc/otelcol-contrib/certs
sudo chmod 600 server.key ca.key
sudo chmod 644 server.crt ca.crt

# Clean up
sudo rm server.csr san.cnf ca.srl
```

**Step 2: Update Collector Configuration**

Replace `/etc/otelcol-contrib/config.yaml` with the secure configuration:

```yaml
extensions:
  # Bearer token authentication
  bearertokenauth:
    token: "${env:OTEL_BEARER_TOKEN}"

  health_check:
    endpoint: 0.0.0.0:13133

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        tls:
          cert_file: /etc/otelcol-contrib/certs/server.crt
          key_file: /etc/otelcol-contrib/certs/server.key
          min_version: "1.2"
        auth:
          authenticator: bearertokenauth
      http:
        endpoint: 0.0.0.0:4318
        tls:
          cert_file: /etc/otelcol-contrib/certs/server.crt
          key_file: /etc/otelcol-contrib/certs/server.key
          min_version: "1.2"
        auth:
          authenticator: bearertokenauth

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

exporters:
  clickhouse:
    endpoint: tcp://127.0.0.1:9000
    database: nlm_traces
    traces_table_name: otel_traces
    timeout: 5s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  debug:
    verbosity: normal
    sampling_initial: 5
    sampling_thereafter: 200

service:
  extensions: [bearertokenauth, health_check]

  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [clickhouse, debug]

  telemetry:
    logs:
      level: info
```

**Step 3: Set Bearer Token**

```bash
# Generate token
TOKEN=$(openssl rand -base64 32)

# Set environment variable for systemd service
sudo mkdir -p /etc/systemd/system/otelcol-contrib.service.d
cat <<EOF | sudo tee /etc/systemd/system/otelcol-contrib.service.d/override.conf
[Service]
Environment="OTEL_BEARER_TOKEN=$TOKEN"
EOF

# Save token for client configuration
echo "OTEL_BEARER_TOKEN=$TOKEN" >> ~/.nlm-proxy/.env.otel

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart otelcol-contrib
```

**Step 4: Configure NLM Proxy**

```bash
# Copy CA certificate to client
sudo cp /etc/otelcol-contrib/certs/ca.crt ~/.nlm-proxy/otel-ca.crt
sudo chown $USER:$USER ~/.nlm-proxy/otel-ca.crt

# Add to ~/.nlm-proxy/.env
cat <<EOF >> ~/.nlm-proxy/.env
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=$HOME/.nlm-proxy/otel-ca.crt
NLM_PROXY_OTEL_API_KEY=$(grep OTEL_BEARER_TOKEN ~/.nlm-proxy/.env.otel | cut -d= -f2)
EOF
```

#### Protocol Selection

**gRPC (default):**
- Lower overhead, better performance
- Does NOT support `verify_cert=false` (skip-verify)
- Use for production with proper CA certificates

```bash
NLM_PROXY_OTEL_PROTOCOL=grpc
NLM_PROXY_OTEL_ENDPOINT=localhost:4317
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/ca.crt
```

**HTTP:**
- Supports `verify_cert=false` for self-signed certificates
- Useful for development/testing
- Slightly higher overhead than gRPC

```bash
# HTTP with skip-verify (development)
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_ENDPOINT=localhost:4318
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_VERIFY_CERT=false

# HTTP with CA cert (production)
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_ENDPOINT=collector.example.com:4318
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/etc/ssl/certs/otel-ca.pem
```

#### Security Best Practices

1. **Certificate Management**
   - Use trusted CA certificates in production (Let's Encrypt, DigiCert)
   - Rotate certificates before expiration
   - Store private keys securely (chmod 600)
   - Use Subject Alternative Names (SAN) for multi-domain support

2. **Token Security**
   - Store bearer tokens in secrets manager (AWS Secrets, HashiCorp Vault)
   - Rotate tokens periodically (recommend: 90 days)
   - Use different tokens per environment (dev/staging/prod)
   - Never commit tokens to version control

3. **Network Security**
   - Run collector in private network, not internet-facing
   - Use VPN or VPC peering for remote clients
   - Consider reverse proxy (nginx, Envoy) with rate limiting
   - Enable firewall rules to restrict access

4. **Monitoring**
   - Monitor authentication failures: `docker logs nlm-otel-collector | grep "authentication failed"`
   - Set up alerts for TLS handshake errors
   - Check health endpoint: `curl http://localhost:13133`
   - Monitor collector queue depth and export failures

#### Mutual TLS (mTLS)

For additional security, enable client certificate authentication:

**Collector configuration:**
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        tls:
          cert_file: /etc/otelcol-contrib/certs/server.crt
          key_file: /etc/otelcol-contrib/certs/server.key
          client_ca_file: /etc/otelcol-contrib/certs/ca.crt
          # Require and verify client certificates
          client_auth_type: RequireAndVerifyClientCert
```

**Client configuration:**
```bash
# Generate client certificate (signed by same CA)
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr \
  -subj "/C=US/ST=CA/O=NLM Proxy/CN=nlm-proxy-client"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client.crt -days 365

# Configure NLM Proxy (requires SDK support - check documentation)
NLM_PROXY_OTEL_CLIENT_CERT_PATH=/path/to/client.crt
NLM_PROXY_OTEL_CLIENT_KEY_PATH=/path/to/client.key
```

## Understanding Traces

### Span Hierarchy

Each request creates a trace with nested spans:

```
smart_router.handle_request (parent - full request lifecycle)
├── user_query
├── response_content
├── response_source
│
└── smart_router.route (child - routing decision)
    ├── smart_router.classify (grandchild)
    └── smart_router.select_notebook (grandchild, if NotebookLM)
```

### Span Attributes

#### smart_router.handle_request
| Attribute | Type | Description |
|-----------|------|-------------|
| `user_query` | string | User's query (truncated per `REQUEST_MAX_LENGTH`) |
| `response_content` | string | Response text (truncated per `RESPONSE_MAX_LENGTH`) |
| `response_source` | string | "llm" or "notebooklm" |

#### smart_router.route
| Attribute | Type | Description |
|-----------|------|-------------|
| `request_type` | string | "LLM_TASK" or "NOTEBOOKLM" |
| `notebook_id` | string | Selected notebook ID (if applicable) |
| `routing_reasoning` | string | Explanation of routing decision |

#### smart_router.classify
| Attribute | Type | Description |
|-----------|------|-------------|
| `classification_result` | string | "LLM_TASK" or "NOTEBOOKLM" |
| `llm_model` | string | Model used for classification |

#### smart_router.select_notebook
| Attribute | Type | Description |
|-----------|------|-------------|
| `candidates_count` | int | Number of notebooks considered |
| `selected_notebook_id` | string | Chosen notebook ID |
| `selected_notebook_title` | string | Chosen notebook title |
| `selection_fallback` | bool | True if fell back to first notebook |

### Auto-Instrumented Spans

With FastAPI and httpx instrumentation enabled, you'll also see:

- **HTTP server spans**: Incoming API requests
- **HTTP client spans**: Outgoing calls to NotebookLM and external LLMs

## Querying Trace Data

### ClickHouse Queries

Connect to ClickHouse:
```bash
docker exec -it nlm-clickhouse clickhouse-client
```

#### End-to-End Request Flow

View the complete flow for a single request, showing all spans from HTTP request through routing to response:

```sql
-- Complete request flow grouped by TraceId
SELECT
    TraceId,
    formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as time,
    SpanName,
    round(Duration/1000000, 2) as duration_ms,
    SpanAttributes['user_query'] as query,
    SpanAttributes['classification_result'] as classification,
    SpanAttributes['selected_notebook_title'] as selected_notebook,
    SpanAttributes['candidates_count'] as candidates,
    StatusCode as status
FROM nlm_traces.otel_traces
WHERE TraceId IN (
    SELECT DISTINCT TraceId
    FROM nlm_traces.otel_traces
    WHERE SpanName = 'smart_router.route'
    ORDER BY Timestamp DESC
    LIMIT 1
)
ORDER BY Timestamp
FORMAT PrettyCompact;
```

**Example Output:**
```
┌─TraceId──────────────────────────┬─time─────────────────┬─SpanName────────────────────────┬─duration_ms─┬─query─────────────────────┬─classification─┬─selected_notebook─┬─candidates─┬─status─┐
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ POST /v1/chat/completions       │    51018.59 │                           │                │                   │            │ Unset  │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ smart_router.classify           │     3766.18 │                           │ NOTEBOOKLM     │                   │            │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ smart_router.route              │     4699.97 │ What is machine learning? │                │                   │            │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:24  │ smart_router.select_notebook    │      933.79 │                           │                │ ML Research       │ 4          │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:26:12  │ POST /v1/chat/completions send  │           0 │                           │                │                   │            │ Unset  │
└──────────────────────────────────┴──────────────────────┴─────────────────────────────────┴─────────────┴───────────────────────────┴────────────────┴───────────────────┴────────────┴────────┘
```

This shows the complete request lifecycle:
1. **POST /v1/chat/completions** - HTTP request received (51 seconds total)
2. **smart_router.classify** - LLM classification (3.8 seconds) → Result: NOTEBOOKLM
3. **smart_router.route** - Main routing logic (4.7 seconds)
4. **smart_router.select_notebook** - Notebook selection (0.9 seconds) → Selected: ML Research (4 candidates)
5. **POST /v1/chat/completions send** - HTTP response sent

#### Recent Requests Summary

View a summary of recent requests showing key routing decisions:

```sql
-- Summary view of recent requests
SELECT
    substring(TraceId, 1, 8) as trace,
    formatDateTime(min(Timestamp), '%H:%M:%S') as time,
    any(SpanAttributes['user_query']) as user_query,
    any(SpanAttributes['classification_result']) as classification,
    any(SpanAttributes['selected_notebook_title']) as notebook,
    round(sum(Duration)/1000000, 2) as total_ms
FROM nlm_traces.otel_traces
WHERE SpanName LIKE 'smart_router%'
  AND Timestamp > now() - INTERVAL 1 HOUR
GROUP BY TraceId
ORDER BY min(Timestamp) DESC
LIMIT 10
FORMAT Pretty;
```

This provides a high-level overview of routing activity with one row per request.

#### Recent Traces
```sql
SELECT
    TraceId,
    SpanName,
    Duration/1000000 as duration_ms,
    SpanAttributes['request_type'] as request_type,
    SpanAttributes['notebook_id'] as notebook_id
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.route'
ORDER BY Timestamp DESC
LIMIT 20;
```

#### Average Routing Time by Type
```sql
SELECT
    SpanAttributes['request_type'] as request_type,
    count() as count,
    avg(Duration)/1000000 as avg_duration_ms,
    quantile(0.95)(Duration)/1000000 as p95_duration_ms
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.route'
  AND Timestamp > now() - INTERVAL 1 HOUR
GROUP BY request_type;
```

#### Most Selected Notebooks
```sql
SELECT
    SpanAttributes['selected_notebook_id'] as notebook_id,
    SpanAttributes['selected_notebook_title'] as notebook_title,
    count() as selection_count
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND Timestamp > now() - INTERVAL 24 HOUR
GROUP BY notebook_id, notebook_title
ORDER BY selection_count DESC
LIMIT 10;
```

#### Fallback Rate
```sql
SELECT
    countIf(SpanAttributes['selection_fallback'] = 'true') as fallback_count,
    count() as total_count,
    fallback_count / total_count * 100 as fallback_rate_pct
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND Timestamp > now() - INTERVAL 24 HOUR;
```

#### Error Rate
```sql
SELECT
    SpanName,
    countIf(StatusCode = 'ERROR') as error_count,
    count() as total_count,
    error_count / total_count * 100 as error_rate_pct
FROM nlm_traces.otel_traces
WHERE Timestamp > now() - INTERVAL 1 HOUR
GROUP BY SpanName;
```

#### Trace Details by TraceId
```sql
SELECT
    SpanName,
    Duration/1000000 as duration_ms,
    SpanAttributes,
    StatusCode
FROM nlm_traces.otel_traces
WHERE TraceId = 'your-trace-id-here'
ORDER BY Timestamp;
```

#### Response Content Analysis
```sql
-- View recent requests with their responses
SELECT
    formatDateTime(Timestamp, '%Y-%m-%d %H:%i:%S') as time,
    substring(SpanAttributes['user_query'], 1, 50) as query,
    SpanAttributes['response_source'] as source,
    substring(SpanAttributes['response_content'], 1, 100) as response_preview,
    round(Duration/1000000, 2) as duration_ms
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp > now() - INTERVAL 1 HOUR
ORDER BY Timestamp DESC
LIMIT 20;
```

### Grafana Integration

#### Accessing Grafana

After starting the tracing stack, Grafana is available at:
```
http://localhost:3000
```

**Default credentials:**
- Username: `admin`
- Password: `admin` (change on first login)

#### Pre-configured Dashboard

Navigate to: **Dashboards > NLM Proxy - Routing Analytics**

This dashboard provides:
- **Real-time metrics**: Request volume, average routing time, error rate, P95 latency
- **Classification analytics**: Distribution of NOTEBOOKLM vs LLM_TASK requests
- **Notebook insights**: Most frequently selected notebooks
- **Recent requests**: Detailed table with drill-down to traces

#### Trace Exploration

1. From the "Recent Requests" table, click any **TraceId**
2. This opens the **Explore** tab with a pre-configured query
3. View the complete span hierarchy showing the request flow:
   - HTTP request → smart_router.route → classify → select_notebook

You can also manually explore traces:

1. Navigate to **Explore** tab
2. Select **ClickHouse (nlm_traces)** data source
3. Query to view trace details:
   ```sql
   SELECT * FROM nlm_traces.otel_traces
   WHERE TraceId = 'your-trace-id-here'
   ORDER BY Timestamp
   ```

This shows all spans in the trace with their attributes, timing, and relationships.

## Troubleshooting

### Connection Errors (StatusCode.UNAVAILABLE)

If you see errors like:
```
Failed to export traces to 10.60.5.76:4317, error code: StatusCode.UNAVAILABLE
Transient error StatusCode.UNAVAILABLE encountered while exporting traces
```

The `StatusCode.UNAVAILABLE` error is generic and can indicate:
- **Authentication failure**: Wrong or missing API key/bearer token
- **TLS handshake failure**: Certificate issues, wrong protocol (HTTP vs HTTPS), insecure flag mismatch
- **Network connectivity**: Firewall, wrong endpoint, collector not running
- **Collector configuration error**: Auth extension not loaded, wrong authenticator

**Step 1: Check collector logs for the real error**

```bash
# For systemd service
sudo journalctl -u otelcol-contrib -n 50 --no-pager

# For Docker
docker logs nlm-otel-collector --tail 50

# Watch logs in real-time while testing
sudo journalctl -u otelcol-contrib -f
```

**Common error patterns in collector logs:**

| Error in Collector Logs | Cause | Solution |
|-------------------------|-------|----------|
| `authentication failed` | Wrong bearer token | Check `NLM_PROXY_OTEL_API_KEY` matches collector's `bearertokenauth.token` |
| `tls: first record does not look like a TLS handshake` | Client sending plain HTTP to TLS endpoint | Set `NLM_PROXY_OTEL_INSECURE=false` on client |
| `http: server gave HTTP response to HTTPS client` | Client sending HTTPS to plain HTTP endpoint | Set `NLM_PROXY_OTEL_INSECURE=true` or remove TLS from collector |
| `x509: certificate signed by unknown authority` | Client can't verify server certificate | Set `NLM_PROXY_OTEL_CA_CERT_PATH` or use `VERIFY_CERT=false` (HTTP only) |
| `connection refused` | Collector not running or wrong port | Check `sudo systemctl status otelcol-contrib` or `docker ps` |
| `no such host` | DNS/hostname resolution failed | Use IP address or check `/etc/hosts` |
| `missing authorization header` | Client not sending API key | Set `NLM_PROXY_OTEL_API_KEY` on client |

**Step 2: Test connectivity**

```bash
# Test if collector is listening
nc -zv 10.60.5.76 4317
# Expected: Connection to 10.60.5.76 4317 port [tcp/*] succeeded!

# For TLS endpoints, test with OpenSSL
openssl s_client -connect 10.60.5.76:4317
# Should show certificate details and handshake success
```

**Step 3: Verify client configuration**

```bash
# Print your configuration
env | grep NLM_PROXY_OTEL

# Required variables for TLS + Auth:
# NLM_PROXY_OTEL_ENABLED=true
# NLM_PROXY_OTEL_ENDPOINT=10.60.5.76:4317
# NLM_PROXY_OTEL_INSECURE=false
# NLM_PROXY_OTEL_API_KEY=your-bearer-token
# NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/ca.pem (optional)
```

**Step 4: Test with minimal config first**

Start with plain HTTP (no TLS, no auth) to isolate the issue:

```yaml
# Collector config.yaml (temporary test)
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        # No TLS, no auth

processors:
  batch:

exporters:
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

```bash
# Client .env
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=10.60.5.76:4317
NLM_PROXY_OTEL_INSECURE=true
# No API key

# Restart and test
sudo systemctl restart otelcol-contrib
nlm-proxy serve openai --port 8080
```

If this works, add TLS and auth incrementally to identify the failing component.

**Step 5: Enable debug logging**

```bash
# Client side
nlm-proxy serve openai --port 8080 --debug 2>&1 | tee nlm-proxy.log

# Look for OTLP export details
grep -i "export\|grpc\|unavailable" nlm-proxy.log
```

### Tracing Not Working

**Check if tracing is enabled:**
```bash
echo $NLM_PROXY_OTEL_ENABLED
# Should output: true
```

**Check logs for initialization:**
```bash
nlm-proxy serve openai --debug 2>&1 | grep TRACING
```

Expected output:
```
[TRACING] OpenTelemetry initialized: endpoint=http://localhost:4317, service=nlm-proxy
```

### No Traces in ClickHouse

**Check collector is receiving spans:**
```bash
docker logs nlm-otel-collector 2>&1 | tail -20
```

**Check collector can reach ClickHouse:**
```bash
docker exec nlm-otel-collector wget -q -O- http://clickhouse:8123/ping
```

**Verify ClickHouse table exists:**
```bash
docker exec nlm-clickhouse clickhouse-client --query \
  "SHOW TABLES FROM nlm_traces"
```

### High Memory Usage

The BatchSpanProcessor buffers spans before export. If you're seeing high memory:

1. Reduce batch size in `docker/otel/config.yaml`:
```yaml
processors:
  batch:
    timeout: 500ms
    send_batch_size: 512
```

2. Restart the collector:
```bash
docker compose -f docker-compose.otel.yml restart otel-collector
```

### Slow Exports

If traces are delayed:

1. Check network connectivity between proxy and collector
2. Reduce batch timeout for faster exports
3. Consider using HTTP instead of gRPC if behind a proxy

### Disk Space

ClickHouse data is retained for 90 days by default. To check usage:

```bash
docker exec nlm-clickhouse clickhouse-client --query \
  "SELECT
     formatReadableSize(sum(bytes_on_disk)) as size,
     count() as parts
   FROM system.parts
   WHERE database = 'nlm_traces'"
```

To manually clean old data:
```sql
ALTER TABLE nlm_traces.routing_traces
DELETE WHERE Timestamp < now() - INTERVAL 30 DAY;
```

### Disabling Tracing

To disable tracing without removing configuration:

```bash
export NLM_PROXY_OTEL_ENABLED=false
```

Or remove the environment variables entirely. The proxy will start without tracing overhead.

## Implementation Notes

### Streaming vs Non-Streaming Span Ownership

A critical architectural decision for response tracing: **streaming and non-streaming requests require different span ownership patterns**.

**Problem:** When returning a `StreamingResponse`, the parent function's `with` block exits immediately, closing the span before the generator runs. This means you cannot create a span in the parent and expect it to capture streaming response data.

```python
# WRONG - span closes before streaming starts
async def handle_request(...):
    with tracer.start_as_current_span("my_span") as span:
        return StreamingResponse(my_generator(..., span))
        # <-- span closes HERE, before generator runs!
```

**Solution:** Separate span ownership based on request type:

- **Streaming requests**: The generator function creates and owns the span
- **Non-streaming requests**: The parent function creates and owns the span

```python
# CORRECT - generator owns its span for streaming
async def stream_response(...):
    with tracer.start_as_current_span("handle_request") as span:
        async for chunk in ...:
            accumulated += chunk
            yield chunk
        span.set_attribute("response", accumulated)

async def handle_request(...):
    if request.stream:
        # NO span here - generator owns it
        return StreamingResponse(stream_response(...))

    # Non-streaming: create span here
    with tracer.start_as_current_span("handle_request") as span:
        response = await get_response()
        span.set_attribute("response", response)
        return response
```

**Why this matters:** Creating duplicate spans with the same name in a trace causes `argMax()` aggregation queries (used in Grafana dashboards) to return non-deterministic results—data appears and disappears on refresh.

## Best Practices

1. **Use sampling in production**: For high-traffic deployments, configure sampling in the collector to reduce data volume.

2. **Monitor collector health**: Set up alerts for collector errors and queue depth.

3. **Secure the endpoint**: In production, use TLS and authentication for the OTLP endpoint.

4. **Set appropriate retention**: 90 days is the default; adjust based on your compliance and debugging needs.

5. **Index custom attributes**: If you query specific attributes frequently, add ClickHouse indexes.

## Further Reading

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenTelemetry Collector Configuration](https://opentelemetry.io/docs/collector/configuration/)
- [ClickHouse Documentation](https://clickhouse.com/docs)
