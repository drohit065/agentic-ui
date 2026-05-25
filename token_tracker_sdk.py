"""
PrismXAI Token Tracker SDK — Python 3.8+

Requirements:
    pip install requests

Optional (async support):
    pip install httpx

Usage:
    from token_tracker_sdk import TokenTracker, TokenTrackerError

    client = TokenTracker(
        base_url="https://app.prismxai.com:9004",
        api_key=os.environ["TOKEN_TRACKER_API_KEY"],
    )

    # Text chat
    result = client.chat("Summarise this report in 3 bullet points.")
    print(result["reply"])
    print(f"Tokens used: {result['tokenUsage']['totalTokens']}")

    # Vision chat
    import base64
    with open("image.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    result = client.chat("What is in this image?", image={"data": img_b64, "mimeType": "image/jpeg"})

    # Multi-turn conversation
    conv = client.conversation()
    conv.send("Hello, I need help with Python.")
    conv.send("Can you show me a list comprehension example?")

    # Batch processing
    with open("doc.jpg", "rb") as f:
        job = client.batch.create([("doc.jpg", f, "image/jpeg")], prompt="Describe this.")
    result = client.batch.wait(job["id"])

    # Async (requires httpx)
    async def main():
        async with TokenTrackerAsync(base_url="...", api_key="...") as client:
            result = await client.chat("Hello")
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple


try:
    import requests
    from requests import Session
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Error ─────────────────────────────────────────────────────────

class TokenTrackerError(Exception):
    """Raised when the Token Tracker API returns an error response."""

    def __init__(
        self,
        message:    str,
        status_code: int                    = 0,
        response:   Optional[Dict]          = None,
        request_id: Optional[str]           = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response    = response or {}
        self.request_id  = request_id  # correlate with server security/access logs


# ── Sync client ───────────────────────────────────────────────────

class TokenTracker:
    """
    Synchronous Token Tracker client.

    Args:
        base_url:    Root URL of the Token Tracker server.
        api_key:     API key (starts with "tt_"). Generate in the dashboard → API Keys.
        timeout:     Request timeout in seconds (default: 60).
        max_retries: Retry attempts on 429 / 5xx (default: 3).
        on_usage:    Optional callback(token_usage: dict, meta: dict) after each chat call.
    """

    def __init__(
        self,
        base_url:    str                          = "https://app.prismxai.com:9004",
        api_key:     str                          = "",
        timeout:     int                          = 60,
        max_retries: int                          = 3,
        on_usage:    Optional[Callable]           = None,
    ):
        if not _REQUESTS_AVAILABLE:
            raise ImportError("Install 'requests': pip install requests")
        if not api_key:
            raise ValueError("api_key is required")

        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.max_retries = max_retries
        self.on_usage    = on_usage

        self._session = Session()
        self._session.headers.update({
            "Authorization": f"ApiKey {api_key}",
            "Content-Type":  "application/json",
        })

        self.batch = _BatchClient(self)

    def _req(
        self,
        method:  str,
        path:    str,
        body:    Optional[Dict] = None,
        files:   Optional[Any]  = None,
    ) -> Dict[str, Any]:
        req_id  = str(uuid.uuid4())
        url     = f"{self.base_url}/api{path}"
        headers = {"X-Request-ID": req_id}

        attempt = 0
        while True:
            attempt += 1
            try:
                if files is not None:
                    # multipart upload — don't set Content-Type, requests handles it
                    resp = self._session.request(
                        method, url,
                        data=body, files=files,
                        headers=headers,
                        timeout=self.timeout,
                    )
                else:
                    resp = self._session.request(
                        method, url,
                        json=body,
                        headers=headers,
                        timeout=self.timeout,
                    )
            except Exception as exc:
                if attempt > self.max_retries:
                    raise TokenTrackerError(f"Network error: {exc}", request_id=req_id) from exc
                time.sleep(2 ** (attempt - 1))
                continue

            # Retry on 429 or 5xx
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt <= self.max_retries:
                    retry_after = int(resp.headers.get("Retry-After", 0))
                    delay = retry_after if retry_after > 0 else 2 ** (attempt - 1)
                    time.sleep(delay)
                    continue

            if resp.status_code == 204:
                return {}

            data: Dict = {}
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()

            if not resp.ok:
                raise TokenTrackerError(
                    data.get("error", f"HTTP {resp.status_code}"),
                    status_code=resp.status_code,
                    response=data,
                    request_id=data.get("requestId") or resp.headers.get("x-request-id") or req_id,
                )

            # Fire usage hook
            if "tokenUsage" in data and callable(self.on_usage):
                try:
                    self.on_usage(data["tokenUsage"], {"path": path, "method": method})
                except Exception:
                    pass

            return data

    # ── Chat ────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        model:   str             = "gemini-3-flash-preview",
        image:   Optional[Dict]  = None,
    ) -> Dict[str, Any]:
        """
        Send a chat message to an AI model.

        Args:
            message: The user prompt.
            model:   Model ID. Call get_models() for the full list.
            image:   Vision input dict with keys:
                       "data"     — base64-encoded image bytes (no data: prefix)
                       "mimeType" — e.g. "image/jpeg", "image/png"

        Returns:
            dict with keys: reply, tokenUsage, module, provider, recordId
        """
        if not message:
            raise ValueError("message is required")
        body: Dict[str, Any] = {"message": message, "model": model}
        if image and image.get("data") and image.get("mimeType"):
            body["image"] = image
        return self._req("POST", "/chat", body)

    def chat_image_file(
        self,
        message:   str,
        file_path: str,
        mime_type: str           = "image/jpeg",
        model:     str           = "gemini-3-flash-preview",
    ) -> Dict[str, Any]:
        """
        Convenience: read an image file from disk and send it with a message.

        Args:
            message:   The user prompt.
            file_path: Local path to the image file.
            mime_type: MIME type (default: "image/jpeg").
            model:     Model ID.
        """
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return self.chat(message, model=model, image={"data": data, "mimeType": mime_type})

    # ── Stats & history ─────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Token usage stats and recent records for this API key's user."""
        return self._req("GET", "/tokens")

    def get_history(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        Paginated request history.

        Returns:
            dict with keys: records (list), total (int)
        """
        return self._req("GET", f"/users/me/history?limit={limit}&offset={offset}")

    # ── Profile ─────────────────────────────────────────────────

    def get_me(self) -> Dict[str, Any]:
        """Current user profile (id, email, name, role, status)."""
        return self._req("GET", "/users/me")

    # ── Models ──────────────────────────────────────────────────

    def get_models(self) -> List[Dict[str, Any]]:
        """List all available AI models."""
        return self._req("GET", "/config").get("models", [])

    # ── Health ──────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check server health. Returns status, uptime, and service states."""
        return self._req("GET", "/health")

    # ── API key management ───────────────────────────────────────

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List your API keys (metadata only)."""
        return self._req("GET", "/users/me/api-keys").get("keys", [])

    def create_api_key(self, label: str = "SDK Key") -> Dict[str, Any]:
        """
        Generate a new API key. The full key is returned ONCE — store it securely.
        """
        return self._req("POST", "/users/me/api-keys", {"label": label})

    def revoke_api_key(self, key_id: str) -> Dict[str, Any]:
        """Revoke an API key by its ID."""
        return self._req("DELETE", f"/users/me/api-keys/{key_id}")

    # ── BYOLLM token reporting ───────────────────────────────────────

    def track(
        self,
        model:              str,
        provider:           Optional[str]  = None,
        feature:            Optional[str]  = None,
        session_id:         Optional[str]  = None,
        external_id:        Optional[str]  = None,
        prompt_tokens:      int            = 0,
        completion_tokens:  int            = 0,
        cached_tokens:      int            = 0,
        total_tokens:       Optional[int]  = None,
        duration_ms:        Optional[int]  = None,
        metadata:           Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Report token usage from your own LLM call.

        Use this when you call OpenAI / Anthropic / Gemini directly and want
        PrismXAI to record and aggregate the usage data.

        Args:
            model:             Model ID e.g. "gpt-4o", "claude-3-5-sonnet-20241022"
            provider:          "openai" | "anthropic" | "gemini" | ...
            feature:           Your feature/use-case label e.g. "invoice-extraction"
            session_id:        Your session ID for grouping requests
            external_id:       Your request/trace ID
            prompt_tokens:     Input token count
            completion_tokens: Output token count
            cached_tokens:     Cached/read token count
            total_tokens:      Total (computed as prompt + completion if omitted)
            duration_ms:       LLM response latency in milliseconds
            metadata:          Any additional data to store as JSON

        Returns:
            dict with keys: recordId, message, tokens
        """
        if not model:
            raise ValueError("model is required")
        body: Dict[str, Any] = {
            "model":             model,
            "promptTokens":      prompt_tokens,
            "completionTokens":  completion_tokens,
            "cachedTokens":      cached_tokens,
        }
        if provider    is not None: body["provider"]    = provider
        if feature     is not None: body["feature"]     = feature
        if session_id  is not None: body["sessionId"]   = session_id
        if external_id is not None: body["externalId"]  = external_id
        if total_tokens is not None: body["totalTokens"] = total_tokens
        if duration_ms  is not None: body["durationMs"]  = duration_ms
        if metadata     is not None: body["metadata"]    = metadata
        return self._req("POST", "/track", body)

    def track_openai(
        self,
        response:    Any,
        feature:     Optional[str]  = None,
        session_id:  Optional[str]  = None,
        external_id: Optional[str]  = None,
        duration_ms: Optional[int]  = None,
        metadata:    Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Report token usage from an OpenAI API response.

        Args:
            response:    The object returned by openai.chat.completions.create().
                         Supports both the openai library's response object and plain dicts.
        """
        if hasattr(response, "model"):
            model = response.model
        else:
            model = response.get("model", "unknown")

        usage = getattr(response, "usage", None) or response.get("usage") or {}
        if hasattr(usage, "prompt_tokens"):
            pt  = getattr(usage, "prompt_tokens", 0) or 0
            ct  = getattr(usage, "completion_tokens", 0) or 0
            tt  = getattr(usage, "total_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            kc = getattr(details, "cached_tokens", 0) if details else 0
        else:
            pt  = usage.get("prompt_tokens", 0) or 0
            ct  = usage.get("completion_tokens", 0) or 0
            tt  = usage.get("total_tokens", 0) or 0
            kc  = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

        return self.track(
            model=model, provider="openai",
            prompt_tokens=pt, completion_tokens=ct, cached_tokens=kc, total_tokens=tt,
            feature=feature, session_id=session_id, external_id=external_id,
            duration_ms=duration_ms, metadata=metadata,
        )

    def track_anthropic(
        self,
        response:    Any,
        feature:     Optional[str]  = None,
        session_id:  Optional[str]  = None,
        external_id: Optional[str]  = None,
        duration_ms: Optional[int]  = None,
        metadata:    Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Report token usage from an Anthropic API response.

        Args:
            response:    The object returned by anthropic.messages.create().
                         Supports both the anthropic library's response object and plain dicts.
        """
        if hasattr(response, "model"):
            model = response.model
        else:
            model = response.get("model", "unknown")

        usage = getattr(response, "usage", None) or response.get("usage") or {}
        if hasattr(usage, "input_tokens"):
            pt = getattr(usage, "input_tokens", 0) or 0
            ct = getattr(usage, "output_tokens", 0) or 0
            kc = getattr(usage, "cache_read_input_tokens", 0) or 0
        else:
            pt = usage.get("input_tokens", 0) or 0
            ct = usage.get("output_tokens", 0) or 0
            kc = usage.get("cache_read_input_tokens", 0) or 0

        return self.track(
            model=model, provider="anthropic",
            prompt_tokens=pt, completion_tokens=ct, cached_tokens=kc, total_tokens=pt + ct,
            feature=feature, session_id=session_id, external_id=external_id,
            duration_ms=duration_ms, metadata=metadata,
        )

    def get_usage(self) -> Dict[str, Any]:
        """
        Aggregated BYOLLM usage statistics.
        Admins see all users; regular users see their own data only.

        Returns:
            dict with keys: summary, byModel, byFeature, daily
        """
        return self._req("GET", "/usage")

    def get_usage_history(
        self,
        limit:    int            = 50,
        offset:   int            = 0,
        feature:  Optional[str]  = None,
        model:    Optional[str]  = None,
        provider: Optional[str]  = None,
    ) -> Dict[str, Any]:
        """
        Paginated BYOLLM event log.

        Args:
            limit:    Records per page (max 500, default: 50).
            offset:   Pagination offset.
            feature:  Filter by feature label.
            model:    Filter by model name.
            provider: Filter by provider.

        Returns:
            dict with keys: events (list), total, limit, offset
        """
        params = f"limit={limit}&offset={offset}"
        if feature:  params += f"&feature={feature}"
        if model:    params += f"&model={model}"
        if provider: params += f"&provider={provider}"
        return self._req("GET", f"/usage/history?{params}")

    # ── Conversation ─────────────────────────────────────────────

    def conversation(
        self,
        model:     str = "gemini-3-flash-preview",
        max_turns: int = 20,
    ) -> "Conversation":
        """
        Create a stateful multi-turn conversation.
        History is managed locally and formatted into each prompt.
        """
        return Conversation(self, model=model, max_turns=max_turns)

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "TokenTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self._session.close()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


# ── Conversation ──────────────────────────────────────────────────

class Conversation:
    """
    Manages a multi-turn conversation with an AI model.
    History is stored locally; each `send()` appends context to the prompt.
    """

    def __init__(self, client: TokenTracker, model: str, max_turns: int):
        self._client    = client
        self._model     = model
        self._max_turns = max_turns
        self.history: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]

    def send(
        self,
        message:   str,
        image:     Optional[Dict] = None,
        model:     Optional[str]  = None,
    ) -> Dict[str, Any]:
        """
        Send a message and receive a reply. Conversation history is included automatically.

        Args:
            message: The user's message.
            image:   Optional vision input ({"data": base64, "mimeType": "image/jpeg"}).
            model:   Override the conversation's default model for this turn.
        """
        prompt = self._build_prompt(message)
        result = self._client.chat(prompt, model=model or self._model, image=image)

        self.history.append({"role": "user",      "content": message})
        self.history.append({"role": "assistant",  "content": result.get("reply", "")})

        # Trim to max_turns
        if len(self.history) > self._max_turns * 2:
            self.history = self.history[-(self._max_turns * 2):]

        return result

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def _build_prompt(self, new_message: str) -> str:
        if not self.history:
            return new_message
        turns = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.history
        )
        return f"{turns}\nUser: {new_message}"


# ── Batch client ──────────────────────────────────────────────────

class _BatchClient:
    """
    Batch image processing operations.
    Access via `client.batch.create(...)`, `client.batch.wait(...)`, etc.
    """

    def __init__(self, parent: TokenTracker):
        self._p = parent

    def create(
        self,
        files:  List[Tuple[str, Any, str]],
        prompt: str,
        model:  str = "gemini-3-flash-preview",
    ) -> Dict[str, Any]:
        """
        Submit a batch job.

        Args:
            files:  List of (filename, file_object, mime_type) tuples.
            prompt: The prompt to run against every file.
            model:  Model ID.

        Returns:
            Batch job object with id, status, files, createdAt.

        Example:
            with open("doc.jpg", "rb") as f:
                job = client.batch.create(
                    files  = [("doc.jpg", f, "image/jpeg")],
                    prompt = "Summarise the document.",
                )
        """
        multipart_files = [("files", (name, fobj, mime)) for name, fobj, mime in files]
        return self._p._req(
            "POST", "/batch/upload",
            body  = {"prompt": prompt, "model": model},
            files = multipart_files,
        )

    def get(self, job_id: str) -> Dict[str, Any]:
        """Get a batch job's current status and per-file results."""
        return self._p._req("GET", f"/batch/{job_id}")

    def list(self) -> Dict[str, Any]:
        """List all batch jobs for this user."""
        return self._p._req("GET", "/batch")

    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel a pending batch job."""
        return self._p._req("DELETE", f"/batch/{job_id}")

    def stats(self) -> Dict[str, Any]:
        """Queue-wide statistics."""
        return self._p._req("GET", "/batch/stats")

    def wait(
        self,
        job_id:        str,
        poll_interval: float               = 4.0,
        timeout:       float               = 1800.0,
        on_progress:   Optional[Callable]  = None,
    ) -> Dict[str, Any]:
        """
        Poll a batch job until it finishes.

        Args:
            job_id:        The job ID returned by create().
            poll_interval: Seconds between polls (default: 4).
            timeout:       Max seconds to wait (default: 1800 = 30 min).
            on_progress:   Optional callback(job: dict) called on each poll.

        Returns:
            Completed job object.

        Raises:
            TokenTrackerError: If the job doesn't finish within timeout.
        """
        deadline = time.monotonic() + timeout
        terminal = {"completed", "failed", "partial", "cancelled"}
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if callable(on_progress):
                try:
                    on_progress(job)
                except Exception:
                    pass
            if job.get("status") in terminal:
                return job
            time.sleep(poll_interval)
        raise TokenTrackerError(
            f"Batch job {job_id} did not finish within {timeout}s",
        )


# ── Async client (requires httpx) ─────────────────────────────────

try:
    import httpx as _httpx

    class TokenTrackerAsync:
        """
        Async Token Tracker client using httpx.

        Install: pip install httpx

        Usage:
            async with TokenTrackerAsync(base_url="...", api_key="...") as client:
                result = await client.chat("Hello")
                print(result["reply"])
        """

        def __init__(
            self,
            base_url:    str             = "https://app.prismxai.com:9004",
            api_key:     str             = "",
            timeout:     float           = 60.0,
            max_retries: int             = 3,
            on_usage:    Optional[Callable] = None,
        ):
            if not api_key:
                raise ValueError("api_key is required")
            self.base_url    = base_url.rstrip("/")
            self.timeout     = timeout
            self.max_retries = max_retries
            self.on_usage    = on_usage
            self._headers    = {
                "Authorization": f"ApiKey {api_key}",
                "Content-Type":  "application/json",
            }
            self._client: Optional[_httpx.AsyncClient] = None

        async def __aenter__(self) -> "TokenTrackerAsync":
            self._client = _httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
            )
            return self

        async def __aexit__(self, *_: Any) -> None:
            if self._client:
                await self._client.aclose()

        async def _req(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
            import asyncio
            req_id = str(uuid.uuid4())
            url    = f"/api{path}"
            headers = {"X-Request-ID": req_id}

            for attempt in range(1, self.max_retries + 2):
                try:
                    resp = await self._client.request(
                        method, url, json=body, headers=headers
                    )
                except Exception as exc:
                    if attempt > self.max_retries:
                        raise TokenTrackerError(f"Network error: {exc}", request_id=req_id) from exc
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue

                if resp.status_code in (429,) or resp.status_code >= 500:
                    if attempt <= self.max_retries:
                        retry_after = int(resp.headers.get("Retry-After", 0))
                        await asyncio.sleep(retry_after or 2 ** (attempt - 1))
                        continue

                data: Dict = {}
                try:
                    data = resp.json()
                except Exception:
                    resp.raise_for_status()

                if not resp.is_success:
                    raise TokenTrackerError(
                        data.get("error", f"HTTP {resp.status_code}"),
                        status_code=resp.status_code,
                        response=data,
                        request_id=data.get("requestId") or resp.headers.get("x-request-id") or req_id,
                    )

                if "tokenUsage" in data and callable(self.on_usage):
                    try:
                        self.on_usage(data["tokenUsage"], {"path": path, "method": method})
                    except Exception:
                        pass

                return data
            return {}  # unreachable

        async def chat(
            self,
            message: str,
            model:   str            = "gemini-3-flash-preview",
            image:   Optional[Dict] = None,
        ) -> Dict[str, Any]:
            """Async version of chat()."""
            body: Dict[str, Any] = {"message": message, "model": model}
            if image and image.get("data"):
                body["image"] = image
            return await self._req("POST", "/chat", body)

        async def get_stats(self) -> Dict[str, Any]:
            return await self._req("GET", "/tokens")

        async def get_history(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
            return await self._req("GET", f"/users/me/history?limit={limit}&offset={offset}")

        async def get_me(self) -> Dict[str, Any]:
            return await self._req("GET", "/users/me")

        async def get_models(self) -> List[Dict[str, Any]]:
            return (await self._req("GET", "/config")).get("models", [])

        async def health(self) -> Dict[str, Any]:
            return await self._req("GET", "/health")

        async def track(self, model: str, **kwargs: Any) -> Dict[str, Any]:
            """Async version of track()."""
            body: Dict[str, Any] = {"model": model}
            mapping = {
                "provider": "provider", "feature": "feature",
                "session_id": "sessionId", "external_id": "externalId",
                "prompt_tokens": "promptTokens", "completion_tokens": "completionTokens",
                "cached_tokens": "cachedTokens", "total_tokens": "totalTokens",
                "duration_ms": "durationMs", "metadata": "metadata",
            }
            for py_key, json_key in mapping.items():
                if py_key in kwargs and kwargs[py_key] is not None:
                    body[json_key] = kwargs[py_key]
            return await self._req("POST", "/track", body)

        async def get_usage(self) -> Dict[str, Any]:
            """Async version of get_usage()."""
            return await self._req("GET", "/usage")

        async def get_usage_history(
            self,
            limit: int = 50, offset: int = 0,
            feature: Optional[str] = None, model: Optional[str] = None,
            provider: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Async version of get_usage_history()."""
            params = f"limit={limit}&offset={offset}"
            if feature:  params += f"&feature={feature}"
            if model:    params += f"&model={model}"
            if provider: params += f"&provider={provider}"
            return await self._req("GET", f"/usage/history?{params}")

except ImportError:
    # httpx not installed — async client is not available
    class TokenTrackerAsync:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "TokenTrackerAsync requires httpx: pip install httpx"
            )
