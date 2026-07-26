"""
moe-l2 CLI entry point.

Usage:
    moe-l2 start [--model NAME] [--l2-size SIZE] [--port PORT]
    moe-l2 stats
    moe-l2 --help
"""

import argparse
import logging
import sys

from . import __version__
from .cache import L2Cache, DEFAULT_SLOTS_PER_LAYER
from .proxy import start_proxy, DEFAULT_PORT

logger = logging.getLogger("moe-l2")


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
    start_parser = subparsers.add_parser("start", help="Start the L2 scheduler")
    start_parser.add_argument(
        "--model", default="auto", help="Model name (auto-detect if omitted)"
    )
    start_parser.add_argument(
        "--l2-size",
        type=str,
        default="4GB",
        help="L2 cache size (e.g., 4GB, 8GB)",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Proxy port (default: {DEFAULT_PORT})",
    )

    # moe-l2 stats
    subparsers.add_parser("stats", help="Show cache statistics")

    # moe-l2 version (also via --version)
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stats":
        return cmd_stats()
    elif args.command == "version":
        print(f"moe-l2 {__version__}")
        return 0
    else:
        parser.print_help()
        return 1


def cmd_start(args):
    logger.info("moe-l2 %s — starting L2 scheduler", __version__)
    logger.info("  model:   %s", args.model)
    logger.info("  L2 size: %s", args.l2_size)
    logger.info("  port:    %d", args.port)

    # Initialize L2 cache
    cache = L2Cache(slots_per_layer=DEFAULT_SLOTS_PER_LAYER)
    logger.info("L2 cache initialized: %d layers × %d slots", cache.n_layers, cache.slots_per_layer)

    # TODO: load domain→expert mapping from GGUF metadata or .json
    # TODO: pin universal experts

    # Start proxy
    logger.info("Starting proxy on 127.0.0.1:%d → ollama", args.port)
    logger.info("Connect your client to http://127.0.0.1:%d", args.port)
    start_proxy(port=args.port, cache=cache)
    return 0


def cmd_stats():
    print("moe-l2: cache statistics")
    print("  (Run 'moe-l2 start' first to initialize the cache)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
