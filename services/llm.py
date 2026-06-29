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

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"

_groq_client = None
_gemini_configured = False


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


def _ensure_gemini():
    """Lazily configure the Gemini SDK."""
    global _gemini_configured
    if not _gemini_configured:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        _gemini_configured = True


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
        import google.generativeai as genai

        _ensure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL)
        # Gemini SDK is sync; run it off the event loop.
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text, GEMINI_MODEL
    except Exception as e:  # noqa: BLE001
        last_error = e

    raise RuntimeError(f"All LLMs failed: {last_error}")


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
