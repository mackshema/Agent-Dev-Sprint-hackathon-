"""
agents/_rate_limiter.py
Shared rate limiter for all agents calling the Gemini API.
Free tier: gemini-2.5-flash = 5 RPM. With 4 agents all using flash,
we must space calls at least 15 seconds apart to stay under budget.
"""
import asyncio
import time

# Seconds to wait between successive Gemini API calls (any model, any agent)
# 5 RPM = 1 call per 12s. We use 15s for safety margin.
CALL_INTERVAL = 15.0
_last_call: float = 0.0
_lock = asyncio.Lock()


async def rate_limited_call(coro):
    """
    Wrap any async Gemini call with rate limiting.
    Usage:
        result = await rate_limited_call(client.aio.models.generate_content(...))
    But since we can't wrap a coroutine that hasn't been created yet, call like:
        result = await rate_limit_gate()
        result = await client.aio.models.generate_content(...)
    """
    await rate_limit_gate()
    return await coro


async def rate_limit_gate(label: str = ""):
    """Sleep until the next API call is safe to make, then update the timestamp."""
    global _last_call
    async with _lock:
        now = time.monotonic()
        elapsed = now - _last_call
        if elapsed < CALL_INTERVAL:
            wait = CALL_INTERVAL - elapsed
            if label:
                print(f"  [{label}] Rate-limit gate: waiting {wait:.1f}s...")
            else:
                print(f"  [RateLimit] Waiting {wait:.1f}s...")
            await asyncio.sleep(wait)
        _last_call = time.monotonic()
