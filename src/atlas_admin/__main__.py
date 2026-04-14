from __future__ import annotations

import argparse

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(prog="atlas_admin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("atlas_admin.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
