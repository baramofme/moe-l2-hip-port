"""
Ollama transparent proxy — intercepts /api/generate requests.

Sits between the user and ollama, adding L2 cache preloading
based on domain prediction before forwarding the request.

Phase 2: minimal proxy that forwards all requests.
Cache preloading integration comes after the basic proxy works.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .predictor import predict
from .cache import L2Cache

logger = logging.getLogger("moe-l2-proxy")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
OLLAMA_HOST = "http://127.0.0.1:11434"


class MoEL2ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to ollama with L2 preloading."""

    cache: Optional[L2Cache] = None

    def do_POST(self):
        if self.path == "/api/generate":
            self._handle_generate()
        elif self.path == "/api/chat":
            self._handle_chat()
        else:
            self._forward_request()

    def _handle_generate(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            prompt = data.get("prompt", "")
            domain = predict(prompt)
            logger.info("Predicted domain: %s", domain)

            # Trigger async preload (Phase 2: log only)
            if self.cache:
                expert_ids = []  # TODO: domain_to_expert_ids(domain, ...)
                logger.info(
                    "Would preload %d experts for domain=%s",
                    len(expert_ids), domain,
                )
        except json.JSONDecodeError:
            logger.warning("Failed to decode request body")

        # Forward to ollama (Phase 2: direct pass-through)
        self._forward_to_ollama(body)

    def _handle_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            messages = data.get("messages", [])
            if messages:
                last_msg = messages[-1].get("content", "")
                domain = predict(last_msg)
                logger.info("Predicted domain (chat): %s", domain)
        except json.JSONDecodeError:
            pass

        self._forward_to_ollama(body)

    def _forward_to_ollama(self, body: bytes):
        """Forward request to actual ollama server.

        Phase 2: stub — returns a placeholder response.
        Phase 3: real HTTP forwarding with streaming support.
        """
        # TODO: forward via urllib/httpx to OLLAMA_HOST
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "model": "moe-l2-proxy",
            "response": "[moe-l2 proxy active — forwarding to ollama]",
            "done": True,
        }
        self.wfile.write(json.dumps(response).encode())

    def _forward_request(self):
        """Forward non-generate requests directly."""
        # TODO: generic HTTP proxy
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def start_proxy(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache: Optional[L2Cache] = None,
):
    """Start the moe-l2 proxy server."""
    MoEL2ProxyHandler.cache = cache
    server = HTTPServer((host, port), MoEL2ProxyHandler)
    logger.info("moe-l2 proxy listening on %s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()
