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

## Token Expiration

- **Cookies**: Stable for weeks
- **CSRF/Session**: Auto-refreshed on client init
- **Chrome profile**: Persists Google login

Re-extract cookies when API calls fail with auth errors.
