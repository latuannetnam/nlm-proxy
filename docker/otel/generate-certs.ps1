# Generate self-signed TLS certificates for OTel Collector (PowerShell)
#
# Usage:
#   .\docker\otel\generate-certs.ps1
#
# Requirements:
#   - OpenSSL (auto-detected from Git for Windows or PATH)
#   - Run from PowerShell terminal

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CertsDir = Join-Path $ScriptDir "certs"

if (-not (Test-Path $CertsDir)) {
    New-Item -ItemType Directory -Force -Path $CertsDir | Out-Null
}

# Load environment variables from .env file if it exists
$EnvFile = Join-Path $ScriptDir ".env"
if (Test-Path $EnvFile) {
    Write-Host "Loading configuration from $EnvFile..."
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line -split "=", 2
            if ($parts.Count -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim().Trim('"').Trim("'")
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
                # Write-Host "  Set $key"
            }
        }
    }
}

Write-Host "Generating TLS certificates for OTel Collector..."

# Auto-detect OpenSSL path
$PossiblePaths = @(
    "openssl",  # Try PATH first
    "C:\Program Files\Git\usr\bin\openssl.exe",
    "C:\Program Files\Git\mingw64\bin\openssl.exe",
    "C:\Program Files\Git\bin\openssl.exe",
    "$env:LOCALAPPDATA\Programs\Git\usr\bin\openssl.exe",
    "$env:LOCALAPPDATA\Programs\Git\mingw64\bin\openssl.exe"
)

$OpenSSLPath = $null
foreach ($Path in $PossiblePaths) {
    if (Get-Command $Path -ErrorAction SilentlyContinue) {
        $OpenSSLPath = $Path
        break
    }
}

if (-not $OpenSSLPath) {
    Write-Error "OpenSSL not found. Please install Git for Windows (https://git-scm.com/download/win) or OpenSSL."
    exit 1
}

Write-Host "Using OpenSSL at: $OpenSSLPath"

# Certificate validity period (days)
$Days = 365

# Server hostname/IP
$ServerName = if ($env:OTEL_SERVER_NAME) { $env:OTEL_SERVER_NAME } else { "localhost" }

Write-Host "Server name: $ServerName"
Write-Host "Certificate validity: $Days days"
Write-Host ""

# 1. Generate CA private key
& $OpenSSLPath genrsa -out "$CertsDir\ca.key" 4096

# 2. Generate CA certificate
& $OpenSSLPath req -new -x509 -days $Days -key "$CertsDir\ca.key" -out "$CertsDir\ca.crt" `
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=NLM Proxy CA"

Write-Host "✓ CA certificate generated: $CertsDir\ca.crt"

# 3. Generate server private key
& $OpenSSLPath genrsa -out "$CertsDir\server.key" 4096

# 4. Generate server certificate signing request (CSR)
& $OpenSSLPath req -new -key "$CertsDir\server.key" -out "$CertsDir\server.csr" `
  -subj "/C=US/ST=CA/L=San Francisco/O=NLM Proxy/CN=$ServerName"

# 5. Create SAN (Subject Alternative Name) config
$SanConfigContent = @"
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = $ServerName
DNS.2 = localhost
DNS.3 = nlm-otel-collector
IP.1 = 127.0.0.1
"@

Set-Content -Path "$CertsDir\san.cnf" -Value $SanConfigContent

# 6. Sign server certificate with CA
& $OpenSSLPath x509 -req -days $Days -in "$CertsDir\server.csr" `
  -CA "$CertsDir\ca.crt" -CAkey "$CertsDir\ca.key" -CAcreateserial `
  -out "$CertsDir\server.crt" `
  -extensions v3_req -extfile "$CertsDir\san.cnf"

Write-Host "✓ Server certificate generated: $CertsDir\server.crt"
Write-Host "✓ Server private key generated: $CertsDir\server.key"

# Clean up temporary files
Remove-Item "$CertsDir\server.csr"
Remove-Item "$CertsDir\san.cnf"
if (Test-Path "$CertsDir\ca.srl") { Remove-Item "$CertsDir\ca.srl" }

# Set permissions using icacls (more reliable than Get-Acl/Set-Acl)
# Restrict access to current user only for private keys
$KeyFiles = @("$CertsDir\server.key", "$CertsDir\ca.key")
foreach ($KeyFile in $KeyFiles) {
    if (Test-Path $KeyFile) {
        try {
            # 1. Reset permissions and disable inheritance (/inheritance:d)
            # 2. Remove all existing permissions (/remove:g *S-1-1-0) - Everyone
            # 3. Grant full control to current user (/grant:r "$($env:USERNAME):F")
            $proc = Start-Process icacls -ArgumentList "`"$KeyFile`" /inheritance:d /grant:r `"$($env:USERNAME):(F)`"" -NoNewWindow -PassThru -Wait
            if ($proc.ExitCode -ne 0) {
                Write-Warning "icacls failed to set permissions on $KeyFile"
            } else {
                Write-Host "  Secured $KeyFile"
            }
        } catch {
            Write-Warning "Could not set exclusive ACLs on $KeyFile. Ensure the folder is secure."
        }
    }
}

Write-Host ""
Write-Host "✓ Certificate generation complete!"
Write-Host ""
Write-Host "Files created:"
Write-Host "  - $CertsDir\ca.crt       (CA certificate - copy to client for verification)"
Write-Host "  - $CertsDir\server.crt   (Server certificate)"
Write-Host "  - $CertsDir\server.key   (Server private key)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy ca.crt to your NLM Proxy host for client-side verification"
Write-Host "  2. Set NLM_PROXY_OTEL_CA_CERT_PATH=$(Join-Path $CertsDir 'ca.crt') in client .env"
Write-Host "  3. Set NLM_PROXY_OTEL_INSECURE=false to enable TLS verification"
Write-Host "  4. Generate bearer token: openssl rand -base64 32"
Write-Host "  5. Set OTEL_BEARER_TOKEN in .env (both client and collector)"
Write-Host ""
