"""Run the HRP HTTP API: ``python -m hrp.api.http``."""

from __future__ import annotations

import argparse
import os

from hrp.api.http.app import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="HRP HTTP/JSON API server")
    parser.add_argument("--host", default=os.getenv("HRP_API_HOST", "0.0.0.0"), help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HRP_API_PORT", "8090")),
        help="Bind port",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
