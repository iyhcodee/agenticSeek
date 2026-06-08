"""
Robinhood AI Trading Agent — powered by Claude
================================================
First-time setup (run once):
    python setup_trading.py

Dry-run mode (DEFAULT — reads your account, shows trades, places NOTHING):
    python run_trading_agent.py --once
    python run_trading_agent.py              # loop every 5 minutes

Live trading (real orders — explicit opt-in required):
    python run_trading_agent.py --once --live
    python run_trading_agent.py --live --interval 60

Other options:
    --model claude-opus-4-8     maximum reasoning power
    --prompt "..."              custom trading instruction
    --verbose                   show raw LLM output
"""

import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

from sources.llm_provider import Provider
from sources.agents.trading_agent import TradingAgent
from sources.utility import pretty_print


PROMPT_PATH = "prompts/base/trading_agent.txt"

DEFAULT_PROMPT = (
    "Run your full decision loop: check my portfolio and available buying power, "
    "scan momentum setups on your watchlist, then execute the highest-conviction trade "
    "you see right now. Manage any open losers first — cut anything at or below stop."
)


def load_credentials():
    """Load credentials from .env file, then fall back to existing env vars."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=False)


def check_credentials():
    """Verify required env vars are present. Print clear help if not."""
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("ROBINHOOD_USERNAME"):
        missing.append("ROBINHOOD_USERNAME")
    if not os.getenv("ROBINHOOD_PASSWORD"):
        missing.append("ROBINHOOD_PASSWORD")

    if missing:
        pretty_print("Missing credentials. Run setup first:", color="failure")
        pretty_print("  python setup_trading.py", color="warning")
        pretty_print(f"Missing: {', '.join(missing)}", color="warning")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Robinhood AI trading agent")
    p.add_argument("--once",     action="store_true",
                   help="Run one cycle and exit")
    p.add_argument("--interval", type=int, default=300,
                   help="Seconds between cycles (default 300 = 5 min)")
    p.add_argument("--prompt",   type=str, default=DEFAULT_PROMPT,
                   help="Override the trading instruction")
    p.add_argument("--model",    type=str, default="claude-sonnet-4-6",
                   help="Claude model (default: claude-sonnet-4-6)")
    p.add_argument("--verbose",  action="store_true",
                   help="Show raw LLM output")
    p.add_argument("--live",     action="store_true",
                   help="Place REAL orders. Default is dry-run (no real trades).")
    return p.parse_args()


async def run_cycle(agent: TradingAgent, prompt: str):
    pretty_print("─" * 60, color="status")
    pretty_print(f"Cycle start  {time.strftime('%Y-%m-%d %H:%M:%S')}", color="info")
    answer, _ = await agent.process(prompt)
    pretty_print("─" * 60, color="status")
    pretty_print(answer, color="output")


async def main():
    args = parse_args()
    load_credentials()
    check_credentials()

    if not Path(PROMPT_PATH).exists():
        pretty_print(f"Missing prompt file: {PROMPT_PATH}", color="failure")
        sys.exit(1)

    dry_run = not args.live

    if dry_run:
        pretty_print("=" * 60, color="warning")
        pretty_print("  DRY-RUN MODE — reading your account but placing NO real orders.", color="warning")
        pretty_print("  Add --live to enable real trading.", color="warning")
        pretty_print("=" * 60, color="warning")
    else:
        pretty_print("=" * 60, color="failure")
        pretty_print("  LIVE TRADING MODE — real orders will be placed!", color="failure")
        pretty_print("=" * 60, color="failure")
        confirm = input("  Type YES to confirm live trading: ").strip()
        if confirm != "YES":
            pretty_print("Cancelled. Run without --live for dry-run mode.", color="warning")
            sys.exit(0)

    pretty_print(f"Starting trading agent — model: {args.model}", color="status")

    provider = Provider(
        provider_name="anthropic",
        model=args.model,
        server_address="",
        is_local=False,
    )
    agent = TradingAgent(
        name="TradingBot",
        prompt_path=PROMPT_PATH,
        provider=provider,
        verbose=args.verbose,
        dry_run=dry_run,
    )

    if args.once:
        await run_cycle(agent, args.prompt)
        return

    while True:
        try:
            await run_cycle(agent, args.prompt)
        except KeyboardInterrupt:
            pretty_print("Stopped by user.", color="warning")
            break
        except Exception as e:
            pretty_print(f"Cycle error: {e}", color="failure")

        pretty_print(f"Next cycle in {args.interval}s — Ctrl+C to stop", color="status")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            pretty_print("Stopped.", color="warning")
            break


if __name__ == "__main__":
    asyncio.run(main())
