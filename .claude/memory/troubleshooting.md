# Troubleshooting

## "401 Unauthorized" / "403 Forbidden"

- Cookies or CSRF token expired
- Re-run: `nlm-proxy auth extract`

## "Invalid CSRF token"

- The `at=` value expired
- Re-extract fresh cookies

## Empty notebook list

- Wrong Google account session
- Verify logged into correct account

## Rate limit errors

- Free tier: ~50 queries/day
- Wait until next day or upgrade to Plus

## CLI caching issues

Use direct Python:
```bash
uv run python -m nlm_proxy serve mcp
```

## Auth extraction fails

Try file-based import:
```bash
nlm-proxy auth extract --file
```
