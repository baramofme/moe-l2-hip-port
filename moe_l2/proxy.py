"""Ollama transparent proxy — intercepts /api/generate requests.

Sits between the user and ollama, adding L2 cache preloading
based on domain prediction before forwarding the request.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import httpx

from .predictor import predict_hybrid, load_mapping, enable_tfidf
from .cache import L2Cache
from .training_flywheel import append_sample, maybe_retrain, training_stats

if False:  # TYPE_CHECKING only
    from .gate import RoutingProfiler  # noqa: F401

logger = logging.getLogger("moe-l2-proxy")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
OLLAMA_BASE = "http://127.0.0.1:11434"
LLAMA_SERVER_BASE = "http://127.0.0.1:11436"

# Headers NOT forwarded to the client (hop-by-hop or internal)
_HOP_BY_HOP = frozenset({
    "content-length", "content-encoding", "transfer-encoding",
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "upgrade", "server",
})


class MoEL2ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to backend with L2 preloading."""

    cache: Optional[L2Cache] = None
    backend_url: str = OLLAMA_BASE  # overridden per-instance by start_proxy
    gate: Optional["RoutingProfiler"] = None  # online gate adaptation (optional)

    # ── Domain prediction & preloading ──────────────────────────

    def _predict_and_preload(self, text: str):
        """Predict domain and trigger expert preload + flywheel sampling."""
        if not self.cache:
            return
        try:
            # 三层预测：关键词 → TF-IDF 分类器 → 语义兜底（提升样本标签质量）
            enable_tfidf()  # 惰性加载，缺 sklearn 时静默降级
            domain = predict_hybrid(text)
            logger.info("Predicted domain: %s", domain)
            expert_map = load_mapping()
            self.cache.preload_domain(domain, expert_map)
            logger.info("Preloaded experts for domain=%s", domain)

            # 门控在线自适应（P2 ④）：请求级信号 → 预热目标域
            if getattr(self, "gate", None) is not None:
                try:
                    self.gate.on_request(domain)
                except Exception as ge:
                    logger.warning("Gate request signal failed (non-fatal): %s", ge)

            # 模式 B 数据飞轮：记录真实流量样本，攒够阈值增量重训
            try:
                append_sample(text, domain)
                maybe_retrain()
            except Exception as fe:
                logger.warning("Flywheel sampling failed (non-fatal): %s", fe)
        except Exception as e:
            # Non-fatal: forwarding still happens even if preload fails
            logger.warning("Preload failed: %s", e)

    # ── Entry points ────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/stats":
            self._handle_stats()
        elif self.path == "/health":
            self._send_json({"status": "ok", "cache_active": self.cache is not None})
        else:
            self._proxy_request("GET")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/generate":
            self._handle_api("generate", body)
        elif self.path == "/api/chat":
            self._handle_api("chat", body)
        elif self.path.startswith("/v1/"):
            # OpenAI API format: pass through, but still do prediction
            try:
                data = json.loads(body)
                if "messages" in data:
                    text = data["messages"][-1].get("content", "")
                elif "prompt" in data:
                    text = data["prompt"]
                else:
                    text = ""
                if text:
                    self._predict_and_preload(text)
            except Exception:
                pass
            self._proxy_request("POST", body)
        else:
            self._proxy_request("POST", body)

    # ── API handlers ────────────────────────────────────────────

    def _handle_api(self, endpoint: str, body: bytes):
        """Handle /api/generate or /api/chat — predict, preload, forward.

        Backend endpoint mapping:
          - ollama (11434):  /api/generate → /api/generate, /api/chat → /api/chat
          - llama-server (11436): /api/generate → /completion,
                                  /api/chat → /v1/chat/completions (OpenAI format)
        """
        try:
            data = json.loads(body)
            if endpoint == "generate":
                text = data.get("prompt", "")
            else:
                messages = data.get("messages", [])
                text = messages[-1].get("content", "") if messages else ""
            if text:
                self._predict_and_preload(text)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning("Failed to parse request body: %s", e)

        # 后端端点映射：llama-server 用 OpenAI 兼容端点，不是 ollama /api/*
        if endpoint == "chat" and "11436" in self.backend_url:
            self._forward_to_backend(body, "/v1/chat/completions")
        elif endpoint == "generate" and "11436" in self.backend_url:
            self._forward_to_backend(body, "/completion")
        else:
            self._forward_to_backend(body, f"/api/{endpoint}")

    # ── Backend forwarding ───────────────────────────────────────

    def _forward_to_backend(self, body: bytes, path: str):
        """Forward request to backend, handling streaming or blocking."""
        url = f"{self.backend_url}{path}"

        try:
            data = json.loads(body)
            is_stream = data.get("stream", True)
        except (json.JSONDecodeError, KeyError):
            is_stream = False

        client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            if is_stream:
                self._forward_stream(client, url, body)
            else:
                self._forward_blocking(client, url, body)
        finally:
            client.close()

    def _forward_stream(self, client: httpx.Client, url: str, body: bytes):
        """SSE-stream the response from backend to the client."""
        try:
            with client.stream(
                "POST",
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                self.send_response(resp.status_code)
                # SSE-specific headers
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                for chunk in resp.iter_bytes():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        except httpx.TimeoutException:
            self._send_error(504, "backend request timed out")

    def _forward_blocking(self, client: httpx.Client, url: str, body: bytes):
        """Non-streaming forward: wait for full response, return as-is."""
        try:
            resp = client.post(
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp.content)
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        except httpx.TimeoutException:
            self._send_error(504, "backend request timed out")

    def _proxy_request(self, method: str, body: bytes = None):
        """Generic HTTP proxy for endpoints like /api/tags, /api/show, etc."""
        url = f"{self.backend_url}{self.path}"
        client = httpx.Client(timeout=30.0)
        try:
            req = client.build_request(method, url, content=body)
            resp = client.send(req)
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp.content)
        except httpx.ConnectError:
            self._send_error(502, f"cannot connect to backend at {self.backend_url}")
        finally:
            client.close()

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        """Send a JSON error response."""
        logger.error(message)
        self._send_json({"error": message}, status)

    def _handle_stats(self):
        """Return cache statistics as JSON (for 'moe-l2 stats')."""
        if not self.cache:
            self._send_json({"error": "cache not initialized"}, status=503)
            return
        try:
            stats = self.cache.stats()
            # 模式 B 数据飞轮状态
            stats.update(training_stats())
            # Return compact version for CLI consumption
            self._send_json(stats)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def start_proxy(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache: Optional[L2Cache] = None,
    backend_url: str = OLLAMA_BASE,
    gate: Optional["RoutingProfiler"] = None,
):
    """Start the moe-l2 proxy server.

    Args:
        host: Listen address.
        port: Listen port.
        cache: Optional L2Cache instance for domain preloading.
        backend_url: Upstream inference server URL.
        gate: Optional RoutingProfiler for online gate adaptation.
    """
    MoEL2ProxyHandler.cache = cache
    MoEL2ProxyHandler.gate = gate
    # Patch the backend URL onto the handler
    MoEL2ProxyHandler.backend_url = backend_url
    import functools
    server = HTTPServer((host, port), MoEL2ProxyHandler)
    server.timeout = 0.5  # allow KeyboardInterrupt to work
    logger.info("moe-l2 proxy listening on %s:%s → %s", host, port, backend_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()
