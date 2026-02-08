#!/bin/bash
# Generate self-signed TLS certificates for OTel Collector
#
# Usage:
#   bash docker/otel/generate-certs.sh
#
# Generates:
#   - ca.crt: Certificate Authority (for client-side validation)
#   - server.crt: Server certificate
#   - server.key: Server private key
#
# For production, use certificates from a trusted CA (Let's Encrypt, DigiCert, etc.)

set -e

CERTS_DIR="$(dirname "$0")/certs"
mkdir -p "$CERTS_DIR"

# Load environment variables from .env file if it exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from $ENV_FILE..."
    # Safely export variables from .env (ignoring comments and empty lines)
    set -a
    source <(grep -v '^#' "$ENV_FILE" | grep -v '^\s*$')
    set +a
fi

echo "Generating TLS certificates for OTel Collector..."

# Certificate validity period (days)
DAYS=365

# Server hostname/IP
# For Docker: use 'localhost' if accessing from host, or container name if within Docker network
# For production: use actual domain name (e.g., otel.example.com)
SERVER_NAME="${OTEL_SERVER_NAME:-localhost}"

echo "Server name: $SERVER_NAME"
echo "Certificate validity: $DAYS days"
echo ""

# 1. Generate CA private key
openssl genrsa -out "$CERTS_DIR/ca.key" 4096

# 2. Generate CA certificate
openssl req -new -x509 -days $DAYS -key "$CERTS_DIR/ca.key" -out "$CERTS_DIR/ca.crt" \
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=NLM Proxy CA"

echo "✓ CA certificate generated: $CERTS_DIR/ca.crt"

# 3. Generate server private key
openssl genrsa -out "$CERTS_DIR/server.key" 4096

# 4. Generate server certificate signing request (CSR)
openssl req -new -key "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.csr" \
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=$SERVER_NAME"

# 5. Create SAN (Subject Alternative Name) config for multi-domain support
cat > "$CERTS_DIR/san.cnf" <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = $SERVER_NAME
DNS.2 = localhost
DNS.3 = nlm-otel-collector
IP.1 = 127.0.0.1
EOF

# 6. Sign server certificate with CA
openssl x509 -req -days $DAYS -in "$CERTS_DIR/server.csr" \
  -CA "$CERTS_DIR/ca.crt" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
  -out "$CERTS_DIR/server.crt" \
  -extensions v3_req -extfile "$CERTS_DIR/san.cnf"

echo "✓ Server certificate generated: $CERTS_DIR/server.crt"
echo "✓ Server private key generated: $CERTS_DIR/server.key"

# Clean up temporary files
rm -f "$CERTS_DIR/server.csr" "$CERTS_DIR/san.cnf" "$CERTS_DIR/ca.srl"

# Set permissions
chmod 600 "$CERTS_DIR/server.key" "$CERTS_DIR/ca.key"
chmod 644 "$CERTS_DIR/server.crt" "$CERTS_DIR/ca.crt"

echo ""
echo "✓ Certificate generation complete!"
echo ""
echo "Files created:"
echo "  - $CERTS_DIR/ca.crt       (CA certificate - copy to client for verification)"
echo "  - $CERTS_DIR/server.crt   (Server certificate)"
echo "  - $CERTS_DIR/server.key   (Server private key)"
echo ""
echo "Next steps:"
echo "  1. Copy ca.crt to your NLM Proxy host for client-side verification"
echo "  2. Set NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/ca.crt in client .env"
echo "  3. Set NLM_PROXY_OTEL_INSECURE=false to enable TLS verification"
echo "  4. Generate bearer token: openssl rand -base64 32"
echo "  5. Set OTEL_BEARER_TOKEN in .env (both client and collector)"
echo ""
