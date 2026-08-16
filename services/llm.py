"""LLM chain with automatic fallback: Groq (primary) -> Gemini (fallback).

The chain never crashes /analyze: if Groq is exhausted it falls through to
Gemini; only when both fail does it raise. Clients are created lazily so the
service can boot (and serve /health) even without API keys configured.

Also provides `extract_json`, a tolerant JSON extractor for LLM output that may
be wrapped in markdown fences or surrounded by prose.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

# Use the OS trust store for TLS as early as possible. On managed Windows
# machines TLS is often intercepted by a corporate proxy whose root CA lives in
# the Windows store, not in certifi. truststore patches the stdlib ssl module
# globally, which httpx (Groq) and Gemini's REST transport both rely on.
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - best effort; fall back to default CAs
    pass

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.1-flash-lite"

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_TIMEOUT = 60.0

_groq_client = None


def _get_groq():
    """Lazily build the async Groq client."""
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = AsyncGroq(api_key=api_key)
    return _groq_client


async def _gemini_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Gemini's REST API directly with httpx.

    The `google-generativeai` SDK was only ever used with transport="rest", so it
    contributed nothing but weight: it pulls grpc, protobuf and
    google-api-python-client, which together cost ~136 MB on disk and ~32 MB of
    resident memory — over half the Kernel's Python dependencies, for a code path
    that never touched them. This is the same HTTP call the SDK was making.

    Going through httpx also keeps every outbound call on the stdlib ssl module,
    which `truststore` patches above — the reason the SDK was pinned to REST in
    the first place.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    import httpx

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
        response = await client.post(
            GEMINI_ENDPOINT.format(model=GEMINI_MODEL),
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    # Mirror the SDK's `.text`: concatenate the text parts of the first candidate.
    # A response blocked by a safety filter has no parts at all, so this raises
    # rather than silently returning "" — the caller's fallback should see it.
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {str(data)[:200]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        reason = candidates[0].get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no text (finishReason={reason})")
    return text


async def llm_call(prompt: str, max_tokens: int = 1000) -> tuple[str, str]:
    """Call the LLM chain with automatic fallback.

    Returns:
        (response_text, model_used)

    Raises:
        RuntimeError: only when every provider fails.
    """
    last_error: Exception | None = None

    # --- Primary: Groq, 3 attempts with exponential backoff (1s, 2s). --------
    for attempt in range(3):
        try:
            client = _get_groq()
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content, GROQ_MODEL
        except Exception as e:  # noqa: BLE001 - we deliberately fall through
            last_error = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                break  # Groq exhausted -> fall to Gemini

    # --- Fallback: Gemini ----------------------------------------------------
    try:
        text = await _gemini_generate(prompt, max_tokens=max_tokens, temperature=0.1)
        return text, GEMINI_MODEL
    except Exception as e:  # noqa: BLE001
        last_error = e

    raise RuntimeError(f"All LLMs failed: {last_error}")


# --------------------------------------------------------------------------- #
# Provider-targeted calls (no fallback) — used for cross-model corroboration.
# --------------------------------------------------------------------------- #
async def call_groq(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Call Groq directly. Raises on failure (no fallback)."""
    client = _get_groq()
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


async def call_gemini(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Call Gemini directly. Raises on failure (no fallback)."""
    return await _gemini_generate(prompt, max_tokens=max_tokens, temperature=temperature)


# Registry so the graph builder can iterate over available providers by name.
PROVIDERS = {
    GROQ_MODEL: call_groq,
    GEMINI_MODEL: call_gemini,
}


def extract_json(text: str) -> dict | list:
    """Best-effort parse of JSON from an LLM response.

    Handles raw JSON, ```json fenced blocks, and JSON embedded in prose.

    Raises:
        json.JSONDecodeError: if no parseable JSON is found.
    """
    text = text.strip()

    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the first balanced {...} or [...] span.
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise json.JSONDecodeError("No JSON found in LLM response", text, 0)
