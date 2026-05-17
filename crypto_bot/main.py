"""Entry point: starts trading bot and FastAPI dashboard."""

from __future__ import annotations

import argparse
import asyncio

from crypto_bot._bootstrap import ensure_project_venv

ensure_project_venv()

import uvicorn

from crypto_bot.logging_setup import setup_logging
from crypto_bot.state import bot


async def run_bot_only() -> None:
    await bot.start()
    try:
        while bot.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Crypto trading bot")
    parser.add_argument(
        "--mode",
        choices=["api", "bot", "all"],
        default="all",
        help="api: dashboard only, bot: trading only, all: both",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "bot":
        asyncio.run(run_bot_only())
    elif args.mode == "api":
        uvicorn.run(
            "crypto_bot.frontend.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
    else:
        async def run_all() -> None:
            import threading

            def run_api() -> None:
                uvicorn.run(
                    "crypto_bot.frontend.app:app",
                    host=args.host,
                    port=args.port,
                    reload=False,
                )

            t = threading.Thread(target=run_api, daemon=True)
            t.start()
            await bot.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await bot.stop()

        asyncio.run(run_all())


if __name__ == "__main__":
    main()
