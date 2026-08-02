"""moe-l2 CLI entry point.

Usage:
    moe-l2 start [--model PATH] [--l2-size SIZE] [--port PORT] [--gpu]
    moe-l2 stats [--port PORT]
    moe-l2 download-bins [--release TAG]
    moe-l2 --help
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

import httpx

from . import __version__
from .cache import L2Cache
from .proxy import start_proxy, DEFAULT_PORT, DEFAULT_HOST, LLAMA_SERVER_BASE

logger = logging.getLogger("moe-l2")

# Common GGUF search paths
_GGUF_PATHS = [
    "/opt/data/models/Qwen2.5-MOE-2X1.5B-Q2_k.gguf",
    "/opt/data/models/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf",
    "/opt/data/models/*.gguf",
]

# GitHub release info
_GITHUB_REPO = "yalun753/moe-l2"
_BINS_ASSET_URL = (
    "https://github.com/{repo}/releases/download/{tag}/llama_bins.tar.gz"
)
_DEFAULT_BINS_TAG = "bins-v0.1.1"

# Where the bundled llama-server lives (relative to this file)
_BUNDLE_DIR = Path(__file__).resolve().parent / "bin"
_LLAMA_SERVER_PATH = _BUNDLE_DIR / "llama-server"
_GPU_PORT = 11436  # llama-server listens here; proxy forwards to it


def _find_gguf(path_hint: str) -> str | None:
    """Find a GGUF model file, trying path_hint first, then common paths."""
    if path_hint and path_hint != "auto":
        p = Path(path_hint)
        if p.exists():
            return str(p)
        logger.warning("Specified model not found: %s", path_hint)

    # Try common paths
    for pattern in _GGUF_PATHS:
        matches = sorted(Path("/opt/data/models").glob("*.gguf"))
        if matches:
            return str(matches[0])

    return None


def _parse_l2_size(size_str: str, expert_size: int) -> int:
    """Parse '4GB', '2GB', etc. into slots_per_layer."""
    size_str = size_str.strip().upper()
    if size_str.endswith("GB"):
        total_bytes = float(size_str[:-2]) * (1024**3)
    elif size_str.endswith("MB"):
        total_bytes = float(size_str[:-2]) * (1024**2)
    else:
        raise ValueError(f"Unrecognized size format: {size_str} (use e.g. 4GB)")

    slots = int(total_bytes / expert_size)
    return max(slots, 1)


def _ensure_bins(tag: str | None = None) -> bool:
    """Download bundled binaries if not present. Returns True if ready."""
    if _LLAMA_SERVER_PATH.exists():
        return True
    print(f"  [download-bins] Binary not found at {_BUNDLE_DIR}")
    print(f"  Auto-downloading from GitHub Releases...")
    return _download_bins(tag or _DEFAULT_BINS_TAG)


def _download_bins(tag: str) -> bool:
    """Download and extract the pre-built llama-server + .so bundle."""
    url = _BINS_ASSET_URL.format(repo=_GITHUB_REPO, tag=tag)
    print(f"  Downloading binaries from {url} ...")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  ERROR: Failed to download: {e}")
        return False

    tgz_path = _BUNDLE_DIR / "llama_bins.tar.gz"
    _BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path.write_bytes(resp.content)

    print(f"  Extracting to {_BUNDLE_DIR} ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=_BUNDLE_DIR)

    tgz_path.unlink()  # cleanup

    # Verify
    if _LLAMA_SERVER_PATH.exists():
        size = _LLAMA_SERVER_PATH.stat().st_size
        print(f"  OK: llama-server ({size} bytes)")
        return True
    print(f"  ERROR: llama-server not found after extraction")
    return False


def _start_llama_server(model_path: str, port: int) -> subprocess.Popen:
    """Launch bundled llama-server with GPU support (A3 patch enabled).

    Sets LD_LIBRARY_PATH to the bundled .so files.
    Uses mmap default so expert tensors stay in CPU RAM.
    """
    if not _LLAMA_SERVER_PATH.exists():
        raise FileNotFoundError(
            f"llama-server not found at {_LLAMA_SERVER_PATH}. "
            "Run 'moe-l2 download-bins' or reinstall moe-l2 with GPU support."
        )

    env = os.environ.copy()
    # Point dynamic linker at bundled .so files
    env["LD_LIBRARY_PATH"] = (
        f"{_BUNDLE_DIR}:" + env.get("LD_LIBRARY_PATH", "")
    )
    # Force expert MUL_MAT_ID ops onto GPU even at batch=1 (single-token decode).
    # Default GGML_OP_OFFLOAD_MIN_BATCH is 32 (ggml-cuda.cu), so decode would keep
    # experts on CPU. With host-buffer experts (llama-model-loader moe-l2 patch),
    # sched's MoE expert-copy optimization copies only activated experts → GPU fast
    # path. Measured: DS 12.5→37.5 t/s, Qwen 10→46.8 t/s (RTX 4090, 2026-08-02).
    env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    # mmap default: expert tensors stay in CPU RAM (host buffer), GPU cuBLAS computes them on demand

    cmd = [
        str(_LLAMA_SERVER_PATH),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model_path,
        # Reasonable defaults for consumer GPUs
        "-ngl", "99",       # offload non-expert layers to GPU
        "-c", "8192",       # 8K context — mmap default leaves expert in CPU RAM
    ]

    logger.info("Launching llama-server (A3 GPU): %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _wait_for_llama_server(port: int, timeout: float = 30.0) -> bool:
    """Poll until llama-server responds on /health."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if resp.status_code < 500:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(
        prog="moe-l2",
        description="MoE inference L2 hot-cache scheduler",
    )
    parser.add_argument(
        "--version", action="version", version=f"moe-l2 {__version__}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # moe-l2 start
    start_parser = subparsers.add_parser(
        "start", help="Start the L2 scheduler + proxy"
    )
    start_parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Path to GGUF model file (auto-detect if omitted). "
            "Required for automated n_layers/expert_size detection."
        ),
    )
    start_parser.add_argument(
        "--l2-size",
        type=str,
        default="4GB",
        help="L2 cache size per layer (e.g. 4GB, 2GB). Default: 4GB",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Proxy port (default: {DEFAULT_PORT})",
    )
    start_parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Enable GPU-accelerated inference with bundled llama-server. "
            "Requires CUDA + NVIDIA GPU. Uses the A3 tiered expert scheduler "
            "to fit large MoE models in limited VRAM."
        ),
    )

    # moe-l2 stats
    stats_parser = subparsers.add_parser(
        "stats", help="Show live cache statistics from running proxy"
    )
    stats_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Proxy port (default: {DEFAULT_PORT})",
    )

    # moe-l2 download-bins
    download_parser = subparsers.add_parser(
        "download-bins", help="Download pre-built GPU binaries from GitHub Release"
    )
    download_parser.add_argument(
        "--release",
        type=str,
        default=_DEFAULT_BINS_TAG,
        help=f"GitHub Release tag (default: {_DEFAULT_BINS_TAG})",
    )

    # moe-l2 collect
    collect_parser = subparsers.add_parser(
        "collect", help="Collect MoE routing data → domain_expert_map.json (模式 A)"
    )
    collect_parser.add_argument("--model", required=True, help="Path to GGUF model file")
    collect_parser.add_argument("--llama-cli", default=None, help="Path to llama-cli (supports LLAMA_EXPERT_LOG)")
    collect_parser.add_argument("--output", default=None, help="Output maps dir (default: ~/.moe-l2/maps)")
    collect_parser.add_argument("--domains", nargs="+", default=None, help="Domains to collect (default: all 8)")
    collect_parser.add_argument("--stages", type=int, default=3, help="Prompts per domain (default: 3)")
    collect_parser.add_argument("--tokens", type=int, default=20, help="Gen tokens per prompt (default: 20)")
    collect_parser.add_argument("--timeout", type=int, default=300, help="Per-prompt timeout (default: 300)")

    # moe-l2 embed-map
    embed_parser = subparsers.add_parser(
        "embed-map", help="Embed domain_expert_map.json into a GGUF model (发布形态)"
    )
    embed_parser.add_argument("--model", required=True, help="Path to GGUF model file")
    embed_parser.add_argument("--map", default=None, help="Path to domain_expert_map.json (default: bundled)")
    embed_parser.add_argument("--output", required=True, help="Output GGUF file path")
    embed_parser.add_argument("--keep-original", action="store_true", help="Keep source model (default: delete on success)")

    # moe-l2 version
    subparsers.add_parser(
        "version", help="Show version (alternative to --version)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(message)s",
        )

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "download-bins":
        return cmd_download_bins(args)
    elif args.command == "collect":
        from .collect import cmd_collect
        return cmd_collect(args)
    elif args.command == "embed-map":
        from .gguf_embed import embed_map
        import os as _os
        map_path = args.map or _os.path.join(
            _os.path.dirname(__file__), "data", "domain_expert_map.json"
        )
        out = embed_map(
            args.model, map_path, args.output, keep_original=args.keep_original
        )
        print(f"✅ Embedded domain map → {out}")
        return 0
    elif args.command == "version":
        print(f"moe-l2 {__version__}")
        return 0
    else:
        parser.print_help()
        return 1


def cmd_download_bins(args):
    """Download pre-built GPU binaries from GitHub Release."""
    print(f"moe-l2 {__version__} — downloading GPU binaries")
    print(f"  target:  {_BUNDLE_DIR}")
    ok = _download_bins(args.release)
    return 0 if ok else 1


def cmd_start(args):
    print(f"moe-l2 {__version__} — starting L2 scheduler + proxy")

    # Find model file
    model_path = _find_gguf(args.model)
    if model_path:
        print(f"  model:   {model_path}")
    else:
        print("  model:   (none found — using defaults)")
        model_path = None

    # Parse L2 size
    try:
        l2_size_str = args.l2_size
    except Exception:
        print(f"  invalid L2 size: {args.l2_size}")
        return 1

    # Initialize L2 cache
    cache_kwargs = {}
    if model_path:
        cache_kwargs["model_path"] = model_path

        # Compute slots from expert size directly
        from .gguf_reader import MoEGGUFReader
        reader = MoEGGUFReader(model_path)
        expert_size = reader.per_expert_size()
        slots = _parse_l2_size(l2_size_str, expert_size)
        print(f"  expert:   {expert_size // (1024*1024)} MB each")
        print(f"  L2 size:  {l2_size_str} total → {slots} slots/layer")
        cache_kwargs["slots_per_layer"] = slots

    cache = L2Cache(**cache_kwargs)
    print(f"  layers:   {cache.n_layers}")
    print(f"  slots:    {cache.slots_per_layer}/layer = {cache.n_layers * cache.slots_per_layer} total")

    # ── GPU mode: spawn llama-server ──
    llama_proc: Optional[subprocess.Popen] = None
    backend_url = LLAMA_SERVER_BASE if args.gpu else "http://127.0.0.1:11434"

    if args.gpu:
        if not model_path:
            print("  ERROR: --gpu requires a model path (--model or auto-detect)")
            return 1
        print()
        print("  [GPU mode] Starting bundled llama-server (A3 tiered scheduling)...")

        # Ensure binaries are present
        if not _ensure_bins():
            return 1

        try:
            llama_proc = _start_llama_server(model_path, _GPU_PORT)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return 1

        # Wait for it to be ready
        print("  Waiting for llama-server to start...", end=" ", flush=True)
        ready = _wait_for_llama_server(_GPU_PORT)
        if not ready:
            print("TIMEOUT")
            stdout, stderr = llama_proc.communicate(timeout=5)
            print("  stdout:", stdout.decode(errors="replace")[-500:])
            print("  stderr:", stderr.decode(errors="replace")[-500:])
            llama_proc.kill()
            return 1
        print("READY")
        print(f"  backend:  127.0.0.1:{_GPU_PORT} (llama-server + CUDA + A3)")
        backend_url = f"http://127.0.0.1:{_GPU_PORT}"

    # ── Start proxy ──
    print(f"  proxy:    127.0.0.1:{args.port} → {backend_url}")
    print()
    print("Connect your client to http://127.0.0.1:{}".format(args.port))
    print("Press Ctrl+C to stop")

    try:
        start_proxy(port=args.port, cache=cache, backend_url=backend_url)
    except KeyboardInterrupt:
        pass
    finally:
        if llama_proc is not None:
            logger.info("Shutting down llama-server...")
            llama_proc.terminate()
            try:
                llama_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llama_proc.kill()
                llama_proc.wait()
            logger.info("llama-server stopped")

    return 0


def cmd_stats(args):
    """Fetch and display cache stats from running proxy."""
    url = f"http://{DEFAULT_HOST}:{args.port}/stats"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        print("moe-l2: proxy not running on 127.0.0.1:{}".format(args.port))
        print("  Start it first: moe-l2 start")
        return 1
    except Exception as e:
        print("moe-l2: failed to fetch stats:", e)
        return 1

    if "error" in data:
        print("moe-l2:", data["error"])
        return 1

    # Format the output nicely
    print("moe-l2 cache statistics")
    print(f"  Active domain:    {data.get('active_domain', 'N/A')}")
    print(f"  Requests:         {data.get('total_requests', 0):,}")
    print(f"  Hits:             {data.get('hits', 0):,}")
    print(f"  Misses:           {data.get('misses', 0):,}")
    print(f"  Hit rate:         {data.get('hit_rate_pct', 0)}%")
    print(f"  Utilization:      {data.get('utilization_pct', 0)}%")
    print(f"  Memory used:      {data.get('memory_usage_mb', 0)} MB")
    print(f"  Pinned experts:   {data.get('pinned_experts', 0):,}")
    print(f"  Pending loads:    {data.get('pending_loads', 0):,}")
    print(f"  Loads completed:  {data.get('loads_completed', 0):,}")

    per_layer = data.get("per_layer_lines", [])
    if per_layer:
        # Show first 3 + last 2 to avoid flooding
        shown = per_layer[:3]
        if len(per_layer) > 5:
            shown.append("  ...")
            shown.extend(per_layer[-2:])
        for line in shown:
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
