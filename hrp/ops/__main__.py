"""Run HRP ops server: python -m hrp.ops"""

import argparse
import os

from hrp.ops.server import run_server


def main() -> None:
    """Main entry point for ops server."""
    parser = argparse.ArgumentParser(description="HRP Ops Server")
    parser.add_argument(
        "--host",
        default=os.getenv("HRP_OPS_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HRP_OPS_PORT", "8080")),
        help="Bind port (default: 8080)",
    )
    args = parser.parse_args()

    print(f"Starting HRP Ops Server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
