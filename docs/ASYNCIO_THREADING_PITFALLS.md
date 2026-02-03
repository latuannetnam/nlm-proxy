# Asyncio + Threading Pitfalls

This document records critical asyncio issues encountered in this project, their root causes, and solutions. These patterns are common in Python applications that mix asyncio with threading.

## Issue #1: "Event loop is closed" in Background Refresh Thread

**Date Discovered**: 2026-02-03
**Severity**: Critical (causes cache to become empty, breaking smart routing)
**Symptoms**:
- First background refresh fails with "Event loop is closed"
- Subsequent refreshes may succeed or fail intermittently
- Cache expires and becomes empty, causing routing failures

### Root Cause

The issue involves **three interacting problems** when mixing asyncio event loops with threading:

#### Problem 1: `asyncio.run()` in Background Threads

```python
# BAD: Creates and destroys event loop on each call
def _refresh_loop(self):
    while not shutdown:
        asyncio.run(self._fetch_all_summaries())  # Creates new loop, runs, closes
```

When `asyncio.run()` is called repeatedly in a daemon thread, asyncio's global state can become corrupted, causing **alternating success/failure** patterns.

**Solution**: Use a persistent event loop for the thread's lifetime:

```python
# GOOD: Single event loop for entire thread lifetime
def _refresh_loop(self):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not shutdown:
            loop.run_until_complete(self._fetch_all_summaries())
    finally:
        loop.close()
```

#### Problem 2: Global Event Loop Reference Pollution

```python
# BAD: Leaves global reference pointing to closed loop
def _initial_fetch(self):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # Sets global reference
    loop.run_until_complete(...)
    loop.close()  # Global ref still points to closed loop!
```

When another thread calls `asyncio.new_event_loop()`, it may interact with the stale global reference.

**Solution**: Clear the global reference before closing:

```python
# GOOD: Clear global reference before closing
def _initial_fetch(self):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(...)
    finally:
        asyncio.set_event_loop(None)  # Clear global ref first!
        loop.close()
```

#### Problem 3: Async Objects Bound to Wrong Event Loop (THE CRITICAL ONE)

This is the most subtle and critical issue. **Asyncio objects are bound to the event loop where they were first used.**

```python
class NotebookLMClient:
    def __init__(self):
        self._client_lock: asyncio.Lock = asyncio.Lock()  # Bound to current loop!
        self._client: httpx.AsyncClient | None = None     # Will bind on first use
```

When these objects are used in one event loop (initial fetch) and then reused in a different event loop (background thread), they fail with "Event loop is closed".

**Affected Objects**:
- `asyncio.Lock`, `asyncio.Event`, `asyncio.Semaphore`, `asyncio.Queue`
- `httpx.AsyncClient`, `aiohttp.ClientSession`
- Any async context managers or async iterators

**Solution**: Close/reset async clients after use in one event loop before using in another:

```python
# GOOD: Release async resources before switching event loops
def _initial_fetch(self):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(self._fetch_all_summaries())
        # Close client to release async resources bound to this loop
        loop.run_until_complete(self._nlm_client.close())
    finally:
        asyncio.set_event_loop(None)
        loop.close()
```

### Complete Fix Applied

**File**: `src/nlm_proxy/openai/notebook_cache.py`

```python
def _initial_fetch(self) -> None:
    """Blocking fetch at startup to warm the cache."""
    logger.info("[CACHE] Performing initial notebook fetch...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._fetch_all_summaries())
            logger.info(f"[CACHE] Initial fetch complete: {len(self._cache)} notebooks cached")
            # Close the client to release async resources (locks, httpx client)
            # bound to this event loop. They will be recreated fresh in the
            # background thread's event loop on next use.
            loop.run_until_complete(self._nlm_client.close())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    except Exception as e:
        logger.error(f"[CACHE] Initial fetch failed: {e}")

def _refresh_loop(self) -> None:
    """Background thread that refreshes cache before TTL expires."""
    refresh_interval = self._ttl_seconds * 0.8
    logger.debug(f"[CACHE] Refresh interval set to {refresh_interval:.0f}s")

    # Create a persistent event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not self._shutdown.wait(timeout=refresh_interval):
            try:
                logger.debug("[CACHE] Background refresh starting...")
                loop.run_until_complete(self._fetch_all_summaries())
                logger.info(f"[CACHE] Background refresh complete: {len(self._cache)} notebooks")
            except Exception as e:
                logger.error(f"[CACHE] Background refresh failed: {e}")
    finally:
        loop.close()
```

### Debugging Tips

1. **Check for alternating success/failure pattern** in logs - indicates `asyncio.run()` issue
2. **Check if first operation fails but subsequent succeed** - indicates async object binding issue
3. **Add debug logging** to identify which async objects are being reused across loops:
   ```python
   logger.debug(f"[DEBUG] Current loop: {id(asyncio.get_event_loop())}")
   logger.debug(f"[DEBUG] Lock loop: {id(self._lock._loop)}")  # If accessible
   ```

### Prevention Checklist

When mixing asyncio with threading:

- [ ] Never use `asyncio.run()` repeatedly in background threads
- [ ] Use `asyncio.new_event_loop()` + `set_event_loop()` for dedicated thread loops
- [ ] Clear global event loop reference with `set_event_loop(None)` before closing
- [ ] Close async clients (httpx, aiohttp) before switching event loops
- [ ] Be aware that `asyncio.Lock`, `Semaphore`, `Queue` are bound to their creation loop
- [ ] Consider using `asyncio.run_coroutine_threadsafe()` for cross-thread async calls

### Related Python Issues

- [Python Issue #22239](https://bugs.python.org/issue22239) - asyncio.Lock bound to event loop
- [httpx Issue #914](https://github.com/encode/httpx/issues/914) - AsyncClient event loop binding

---

## Future Issues

*(Add new issues below as they are discovered)*
