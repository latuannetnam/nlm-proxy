<#
.SYNOPSIS
    Monitor NLM Proxy response cache statistics.
.DESCRIPTION
    Reads .env file for endpoint/credentials and displays cache stats
    in a readable format. Supports one-shot and watch (loop) modes.
.EXAMPLE
    .\scripts\cache-stats.ps1              # One-shot stats
    .\scripts\cache-stats.ps1 -Watch       # Refresh every 5s
    .\scripts\cache-stats.ps1 -Watch -Interval 10  # Every 10s
#>

param(
    [switch]$Watch,
    [int]$Interval = 5
)

# ── Read .env file ──────────────────────────────────────────────────────

function Read-EnvFile {
    $envPaths = @(".env", "$HOME/.nlm-proxy/.env")
    foreach ($path in $envPaths) {
        if (Test-Path $path) {
            Get-Content $path | ForEach-Object {
                if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+?)\s*$') {
                    [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
                }
            }
            return $path
        }
    }
    return $null
}

$envFile = Read-EnvFile

# ── Configuration ───────────────────────────────────────────────────────

$apiKey  = $env:NLM_PROXY_OPENAI_API_KEY
$host_   = if ($env:NLM_PROXY_OPENAI_HOST -and $env:NLM_PROXY_OPENAI_HOST -ne "0.0.0.0") {
    $env:NLM_PROXY_OPENAI_HOST
} else { "localhost" }
$port    = if ($env:NLM_PROXY_OPENAI_PORT) { $env:NLM_PROXY_OPENAI_PORT } else { "8080" }
$baseUrl = "http://${host_}:${port}"

if (-not $apiKey) {
    Write-Host "ERROR: NLM_PROXY_OPENAI_API_KEY not set in .env or environment" -ForegroundColor Red
    exit 1
}

# ── Fetch and display stats ─────────────────────────────────────────────

function Get-CacheStats {
    try {
        $headers = @{ "Authorization" = "Bearer $apiKey" }
        $response = Invoke-RestMethod -Uri "$baseUrl/v1/cache/stats" -Headers $headers -Method Get
        return $response
    } catch {
        Write-Host "ERROR: Cannot reach $baseUrl/v1/cache/stats" -ForegroundColor Red
        Write-Host "  $_" -ForegroundColor DarkGray
        return $null
    }
}

function Show-Stats($stats) {
    Clear-Host

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║       NLM Proxy — Cache Monitor             ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host "  $ts    $baseUrl" -ForegroundColor DarkGray
    Write-Host ""

    if (-not $stats.enabled) {
        Write-Host "  Cache is DISABLED" -ForegroundColor Yellow
        return
    }

    # ── Hit Rate ────────────────────────────────────────────────────────
    $hitRate = $stats.hit_rate
    $barLen  = 30
    $filled  = [math]::Round($hitRate / 100 * $barLen)
    $empty   = $barLen - $filled
    $bar     = ("█" * $filled) + ("░" * $empty)
    $color   = if ($hitRate -ge 70) { "Green" } elseif ($hitRate -ge 40) { "Yellow" } else { "Red" }

    Write-Host "  Hit Rate" -ForegroundColor White -NoNewline
    Write-Host "   $bar " -ForegroundColor $color -NoNewline
    Write-Host "$($hitRate)%" -ForegroundColor $color
    Write-Host ""

    # ── Summary ─────────────────────────────────────────────────────────
    $total = $stats.total_hits + $stats.total_misses
    Write-Host "  SUMMARY" -ForegroundColor White
    Write-Host "  ─────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ("  Total Lookups     {0,8}" -f $total) -ForegroundColor Gray
    Write-Host ("  Total Hits        {0,8}" -f $stats.total_hits) -ForegroundColor Green
    Write-Host ("  Total Misses      {0,8}" -f $stats.total_misses) -ForegroundColor Red
    Write-Host ("  Total Bypasses    {0,8}" -f $stats.total_bypasses) -ForegroundColor Yellow
    Write-Host ""

    # ── Layer Breakdown ─────────────────────────────────────────────────
    Write-Host "  LAYER BREAKDOWN" -ForegroundColor White
    Write-Host "  ─────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ("  L1 Exact Match    {0,8}  hits" -f $stats.l1_hits) -ForegroundColor Green
    Write-Host ("  L2 Near-Exact     {0,8}  hits" -f $stats.l2_hits) -ForegroundColor Cyan
    Write-Host ("  L3 LLM-Verified   {0,8}  hits" -f $stats.l3_hits) -ForegroundColor Blue
    Write-Host ("  L3 LLM-Rejected   {0,8}  misses" -f $stats.l3_misses) -ForegroundColor DarkYellow
    Write-Host ""

    # ── Storage ─────────────────────────────────────────────────────────
    $pct = if ($stats.max_entries -gt 0) {
        [math]::Round($stats.entry_count / $stats.max_entries * 100)
    } else { 0 }

    Write-Host "  STORAGE" -ForegroundColor White
    Write-Host "  ─────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ("  Entries           {0,8}  / {1} ({2}%)" -f $stats.entry_count, $stats.max_entries, $pct) -ForegroundColor Gray
    Write-Host ("  Notebooks         {0,8}" -f $stats.notebook_count) -ForegroundColor Gray
    Write-Host ("  TTL               {0,8}s ({1:N1}h)" -f $stats.ttl_seconds, ($stats.ttl_seconds / 3600)) -ForegroundColor Gray
    Write-Host ("  Semantic          {0,8}" -f $(if ($stats.semantic_enabled) { "enabled" } else { "disabled" })) -ForegroundColor $(if ($stats.semantic_enabled) { "Green" } else { "DarkGray" })

    if ($Watch) {
        Write-Host ""
        Write-Host "  Refreshing every ${Interval}s  (Ctrl+C to stop)" -ForegroundColor DarkGray
    }
}

# ── Main ────────────────────────────────────────────────────────────────

if ($Watch) {
    while ($true) {
        $stats = Get-CacheStats
        if ($stats) { Show-Stats $stats }
        Start-Sleep -Seconds $Interval
    }
} else {
    $stats = Get-CacheStats
    if ($stats) { Show-Stats $stats }
}
