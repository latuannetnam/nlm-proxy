# Authentication

## Method 1: Automated CLI (Recommended)

```bash
nlm-proxy auth extract
```

- Launches Chrome with remote debugging
- Extracts cookies, CSRF token, session ID automatically
- Saves Chrome profile for future headless auth

**Options:**
```bash
nlm-proxy auth extract --port 9223          # Custom port
nlm-proxy auth extract --no-auto-launch     # Use existing Chrome
nlm-proxy auth extract --file               # Manual file import
```

## Method 2: File-Based Import

```bash
nlm-proxy auth extract --file ~/cookies.txt
```

**Manual steps:**
1. Open Chrome → https://notebooklm.google.com
2. F12 → Network tab → filter "batchexecute"
3. Click notebook to trigger request
4. Find "cookie:" header → Copy value
5. Save to file → Run command above

## Method 3: Chrome DevTools MCP (AI Assistants)

```python
# Fast: Extract from network request
navigate_page(url="https://notebooklm.google.com/")
get_network_request(reqid=<batchexecute_request>)
save_auth_tokens(
    cookies=<cookie_header>,
    request_body=<request_body>,
    request_url=<request_url>
)

# Minimal: Cookies only (slower first call)
save_auth_tokens(cookies=<cookie_header>)
```

## Method 4: Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTEBOOKLM_COOKIES` | Yes | Full cookie header |

CSRF token and session ID are auto-extracted.

## Testing

```bash
nlm-proxy auth test
```

## Token Expiration & Auto-Refresh

- **Cookies**: Stable for weeks (Google caps at 400 days)
- **CSRF/Session**: Auto-refreshed on client init and every 30 min (background)
- **Chrome profile**: Persists Google login for headless cookie refresh

### Background Auto-Refresh (default: enabled)

When running `nlm-proxy serve mcp` or `nlm-proxy serve openai`, a background
`AuthRefreshService` automatically refreshes tokens:
- **CSRF/Session**: every 30 min (re-fetches NotebookLM homepage)
- **Cookies**: every 6 h via headless Chrome (requires saved Chrome profile)

Configure via env vars: `NLM_PROXY_AUTH_CSRF_REFRESH_INTERVAL`,
`NLM_PROXY_AUTH_COOKIE_REFRESH_INTERVAL`, `NLM_PROXY_AUTH_AUTO_REFRESH_ENABLED`.

### Manual Refresh

```bash
nlm-proxy auth refresh          # Quick: refresh CSRF + session (~2s)
nlm-proxy auth refresh --full   # Full: CSRF + cookies via headless Chrome (~10s)
```

Re-extract cookies when API calls fail with auth errors and `auth refresh`
does not help (Google session fully expired):

```bash
nlm-proxy auth extract          # Re-login via Chrome
```
