"""
Robinhood Agentic Trading Runner
---------------------------------
Runs the AI trading agent against the official Robinhood MCP server.

Required environment variables:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export ROBINHOOD_MCP_TOKEN="<token from Robinhood app MCP auth>"

How to get ROBINHOOD_MCP_TOKEN:
  Open the Robinhood app → Account → MCP / Agent settings → copy the token,
  or complete the auth flow at https://agent.robinhood.com/mcp/trading

Usage:
  python run_trading_agent.py                         # loop every 5 minutes
  python run_trading_agent.py --once                  # one cycle and exit
  python run_trading_agent.py --interval 60           # loop every 60 seconds
  python run_trading_agent.py --prompt "Focus on NVDA today"
  python run_trading_agent.py --model claude-opus-4-8 # max reasoning power
  python run_trading_agent.py --summary               # print today's journal and exit
"""

import os
import sys
import time
import asyncio
import argparse

from sources.llm_provider import Provider
from sources.agents.trading_agent import TradingAgent
from sources.tools.trading_journal import daily_summary
from sources.utility import pretty_print


PROMPT_PATH = "prompts/base/trading_agent.txt"

DEFAULT_PROMPT = (
    "Run your full decision loop: check my Agentic portfolio and available buying power, "
    "cancel any stale unfilled orders older than 30 minutes, "
    "scan momentum setups on your watchlist, then execute the highest-conviction trade "
    "you see right now — or explicitly state why no trade meets the bar. "
    "Manage any open losers first: cut anything at or beyond its defined stop level."
)


def parse_args():
    p = argparse.ArgumentParser(description="Robinhood AI trading agent")
    p.add_argument("--once",     action="store_true", help="Run one cycle and exit")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default 300)")
    p.add_argument("--prompt",   type=str, default=DEFAULT_PROMPT)
    p.add_argument("--verbose",  action="store_true")
    p.add_argument("--model",    type=str, default="claude-sonnet-4-6",
                   help="Claude model ID (claude-sonnet-4-6 or claude-opus-4-8)")
    p.add_argument("--summary",  action="store_true", help="Print today's trading journal and exit")
    return p.parse_args()


def check_env():
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY   — get it at https://console.anthropic.com/")
    if not os.getenv("ROBINHOOD_MCP_TOKEN"):
        missing.append("ROBINHOOD_MCP_TOKEN — copy from Robinhood app MCP / Agent settings")
    if missing:
        pretty_print("Missing required environment variables:", color="failure")
        for m in missing:
            pretty_print(f"  export {m}", color="warning")
        sys.exit(1)


async def run_cycle(agent: TradingAgent, prompt: str):
    pretty_print("─" * 64, color="status")
    pretty_print(f"Trading cycle  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", color="info")
    answer, _ = await agent.process(prompt)
    pretty_print("─" * 64, color="status")
    pretty_print(answer, color="output")


async def main():
    args = parse_args()

    if args.summary:
        pretty_print(daily_summary(), color="output")
        return

    check_env()

    if not os.path.exists(PROMPT_PATH):
        pretty_print(f"Prompt file not found: {PROMPT_PATH}", color="failure")
        sys.exit(1)

    pretty_print(f"Starting Robinhood AI trading agent — model: {args.model}", color="status")
    pretty_print("Account scope: Robinhood Agentic account only.", color="info")
    pretty_print("Risk limits: 20% daily loss stop | 35% drawdown halt.", color="info")

    provider = Provider(
        provider_name="anthropic",
        model=args.model,
        server_address="",
        is_local=False,
    )
    agent = TradingAgent(
        name="RobinhoodAgent",
        prompt_path=PROMPT_PATH,
        provider=provider,
        verbose=args.verbose,
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
            pretty_print("Stopped by user.", color="warning")
            break


if __name__ == "__main__":
    asyncio.run(main())
