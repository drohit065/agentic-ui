import time
import json
import requests
from flask import Flask, request, Response, send_from_directory
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_tracker_sdk import TokenTracker, TokenTrackerError

app = Flask(__name__)

NVIDIA_BASE    = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = "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"

TRACKER_BASE   = "https://app.prismxai.com:9004"
TRACKER_KEY    = "tt_22210984f6fa88fa3ec519512f79f7fccfeeb83519f00a15c9e7715d67e8ce01"
FEATURE_NAME   = "nvidia-chat"

# Mutable container so keys can be updated at runtime without restart
_state = {"nvidia_api_key": NVIDIA_API_KEY, "tracker_key": TRACKER_KEY}

tracker = TokenTracker(base_url=TRACKER_BASE, api_key=TRACKER_KEY)


@app.route("/update-key", methods=["POST"])
def update_key():
    data        = request.get_json() or {}
    nvidia_key  = data.get("nvidia_key",  "").strip()
    tracker_key = data.get("tracker_key", "").strip()
    updated     = []

    if nvidia_key:
        if not nvidia_key.startswith("nvapi-"):
            return {"error": "Invalid NVIDIA key — must start with 'nvapi-'"}, 400
        _state["nvidia_api_key"] = nvidia_key
        print(f"[key] NVIDIA key updated: {nvidia_key[:12]}...")
        updated.append("NVIDIA")

    if tracker_key:
        if not tracker_key.startswith("tt_"):
            return {"error": "Invalid tracker key — must start with 'tt_'"}, 400
        _state["tracker_key"] = tracker_key
        tracker.__init__(base_url=TRACKER_BASE, api_key=tracker_key)
        print(f"[key] Tracker key updated: {tracker_key[:12]}...")
        updated.append("Tracker")

    if not updated:
        return {"error": "Provide at least one key to update"}, 400

    return {"message": f"{' & '.join(updated)} key(s) updated successfully!"}


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/chat/completions", methods=["POST"])
def proxy_chat():
    body       = request.get_json()
    is_stream  = body.get("stream", False)
    model      = body.get("model", "unknown")

    nvidia_headers = {
        "Authorization": f"Bearer {_state['nvidia_api_key']}",
        "Content-Type":  "application/json",
    }

    if is_stream:
        # --- Streaming: collect chunks, stream to client, track after done ---
        def generate():
            full_text         = ""
            prompt_tokens     = 0
            completion_tokens = 0
            total_tokens      = 0
            start_ms          = int(time.time() * 1000)

            try:
                with requests.post(
                    f"{NVIDIA_BASE}/chat/completions",
                    headers=nvidia_headers,
                    json=body,
                    stream=True,
                    timeout=60,
                ) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        decoded = line.decode("utf-8")
                        yield decoded + "\n\n"

                        # Parse SSE to extract token usage from last chunk
                        if decoded.startswith("data: "):
                            chunk_json = decoded[6:].strip()
                            if chunk_json and chunk_json != "[DONE]":
                                try:
                                    chunk = json.loads(chunk_json)
                                    # Accumulate text
                                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        full_text += delta
                                    # NVIDIA sends usage in last chunk
                                    usage = chunk.get("usage")
                                    if usage:
                                        prompt_tokens     = usage.get("prompt_tokens", 0) or 0
                                        completion_tokens = usage.get("completion_tokens", 0) or 0
                                        total_tokens      = usage.get("total_tokens", 0) or 0
                                except Exception:
                                    pass

            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                return

            # Track token usage after streaming completes
            duration_ms = int(time.time() * 1000) - start_ms
            try:
                tracker.track(
                    model=model,
                    provider="nvidia",
                    feature=FEATURE_NAME,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens or (prompt_tokens + completion_tokens),
                    duration_ms=duration_ms,
                )
                print(f"[tracker] model={model} prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
            except TokenTrackerError as e:
                print(f"[tracker] error: {e}")

        return Response(
            generate(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    else:
        # --- Non-streaming ---
        start_ms = int(time.time() * 1000)
        r = requests.post(
            f"{NVIDIA_BASE}/chat/completions",
            headers=nvidia_headers,
            json=body,
            timeout=60,
        )
        duration_ms = int(time.time() * 1000) - start_ms

        if r.ok:
            data = r.json()
            usage = data.get("usage", {})
            try:
                tracker.track(
                    model=model,
                    provider="nvidia",
                    feature=FEATURE_NAME,
                    prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage.get("completion_tokens", 0) or 0,
                    total_tokens=usage.get("total_tokens", 0) or 0,
                    duration_ms=duration_ms,
                )
                print(f"[tracker] model={model} tokens={usage}")
            except TokenTrackerError as e:
                print(f"[tracker] error: {e}")

        return Response(r.content, status=r.status_code, content_type="application/json")


if __name__ == "__main__":
    print("Server running at http://localhost:9088")
    app.run(host="0.0.0.0", port=9088, threaded=True)
