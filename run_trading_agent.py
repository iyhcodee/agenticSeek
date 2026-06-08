"""
Robinhood Agentic Trading Runner  —  powered by Claude (Anthropic)
--------------------------------------------------------------------
This script runs entirely in this Claude Code session (no local install needed).

Required environment variables — set them with:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export ROBINHOOD_MCP_TOKEN="<token from Robinhood app MCP auth>"

To get ROBINHOOD_MCP_TOKEN:
  On your phone/desktop Robinhood app go to the MCP settings and copy the token,
  OR follow the auth flow at https://agent.robinhood.com/mcp/trading

Usage:
  python run_trading_agent.py                  # loop every 5 minutes
  python run_trading_agent.py --once           # one cycle and exit
  python run_trading_agent.py --interval 60    # loop every 60 seconds
  python run_trading_agent.py --prompt "focus on NVDA options today"
  python run_trading_agent.py --model claude-opus-4-8   # use Opus for max reasoning
"""

import os
import sys
import time
import asyncio
import argparse

from sources.llm_provider import Provider
from sources.agents.trading_agent import TradingAgent
from sources.utility import pretty_print


DEFAULT_PROMPT = (
    "Run your full decision loop: check my portfolio and available buying power, "
    "scan momentum setups on your watchlist, then execute the highest-conviction trade "
    "you see right now. Manage any open losers first — cut anything at or below stop."
)

PROMPT_PATH = "prompts/base/trading_agent.txt"

# Best models for trading decisions, in order of reasoning power
RECOMMENDED_MODELS = {
    "claude-sonnet-4-6": "Fast, smart — good default",
    "claude-opus-4-8":   "Most powerful reasoning — use for complex setups",
}


def parse_args():
    p = argparse.ArgumentParser(description="Robinhood AI trading agent (Claude-powered)")
    p.add_argument("--once",     action="store_true", help="Run one cycle and exit")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default 300)")
    p.add_argument("--prompt",   type=str, default=DEFAULT_PROMPT, help="Trading instruction override")
    p.add_argument("--verbose",  action="store_true", help="Print raw LLM output")
    p.add_argument("--model",    type=str, default="claude-sonnet-4-6",
                   help="Claude model ID (default: claude-sonnet-4-6)")
    return p.parse_args()


def check_env():
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY  →  get it at https://console.anthropic.com/")
    if not os.getenv("ROBINHOOD_MCP_TOKEN"):
        missing.append("ROBINHOOD_MCP_TOKEN  →  copy from Robinhood app MCP settings")
    if missing:
        pretty_print("Missing environment variables:", color="failure")
        for m in missing:
            pretty_print(f"  export {m}", color="warning")
        sys.exit(1)


async def run_cycle(agent: TradingAgent, prompt: str):
    pretty_print("─" * 60, color="status")
    pretty_print(f"Trading cycle  {time.strftime('%Y-%m-%d %H:%M:%S')}", color="info")
    answer, _ = await agent.process(prompt)
    pretty_print("─" * 60, color="status")
    pretty_print(answer, color="output")


async def main():
    args = parse_args()
    check_env()

    if not os.path.exists(PROMPT_PATH):
        pretty_print(f"Prompt file not found: {PROMPT_PATH}", color="failure")
        sys.exit(1)

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
    )

    if args.once:
        await run_cycle(agent, args.prompt)
        return

    while True:
        try:
            await run_cycle(agent, args.prompt)
        except KeyboardInterrupt:
            pretty_print("Stopped.", color="warning")
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
