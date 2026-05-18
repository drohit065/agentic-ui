import requests
from flask import Flask, request, Response, send_from_directory
import json
import os

app = Flask(__name__)

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
API_KEY = "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"

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/chat/completions", methods=["POST"])
def proxy_chat():
    body = request.get_json()
    is_stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if is_stream:
        def generate():
            with requests.post(
                f"{NVIDIA_BASE}/chat/completions",
                headers=headers,
                json=body,
                stream=True,
                timeout=60,
            ) as r:
                for line in r.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"

        return Response(
            generate(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )
    else:
        r = requests.post(
            f"{NVIDIA_BASE}/chat/completions",
            headers=headers,
            json=body,
            timeout=60,
        )
        return Response(r.content, status=r.status_code, content_type="application/json")

if __name__ == "__main__":
    print("Server running at http://localhost:9088")
    app.run(host="0.0.0.0", port=9088, threaded=True)
