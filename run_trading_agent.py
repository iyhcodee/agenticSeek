"""
Robinhood Agentic Trading Runner
---------------------------------
Prerequisites (one-time local setup):
  1. In your LOCAL Claude Code CLI terminal run:
       claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
  2. Type /mcp in Claude Code, select robinhood-trading, and authenticate.
  3. Copy the resulting auth token into the ROBINHOOD_MCP_TOKEN env var:
       export ROBINHOOD_MCP_TOKEN="<your token>"

Then run this script:
  python run_trading_agent.py [--once] [--interval 300] [--prompt "trade TSLA calls"]

Flags:
  --once          Run a single trading cycle and exit (default: loop forever)
  --interval N    Seconds between trading cycles (default: 300 = 5 minutes)
  --prompt TEXT   Override the default trading instruction
  --verbose       Enable verbose LLM output
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
    "you see right now. Manage any open losers first."
)

PROMPT_PATH = "prompts/base/trading_agent.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Robinhood agentic trading runner")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--provider", type=str, default="openai",
                   help="LLM provider name (openai, ollama, etc.)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="Model name for the LLM provider")
    p.add_argument("--server-address", type=str, default="127.0.0.1",
                   help="LLM server address (for Ollama/llama.cpp)")
    return p.parse_args()


async def run_cycle(agent: TradingAgent, prompt: str):
    pretty_print("─" * 60, color="status")
    pretty_print(f"Starting trading cycle: {time.strftime('%Y-%m-%d %H:%M:%S')}", color="info")
    answer, _ = await agent.process(prompt)
    pretty_print("─" * 60, color="status")
    pretty_print("Agent decision:", color="output")
    pretty_print(answer, color="output")


async def main():
    args = parse_args()

    if not os.path.exists(PROMPT_PATH):
        pretty_print(f"Prompt file not found: {PROMPT_PATH}", color="failure")
        sys.exit(1)

    token = os.getenv("ROBINHOOD_MCP_TOKEN", "")
    if not token:
        pretty_print(
            "WARNING: ROBINHOOD_MCP_TOKEN not set. "
            "Authenticated Robinhood calls will fail.\n"
            "Run 'claude mcp add robinhood-trading --transport http "
            "https://agent.robinhood.com/mcp/trading' on your local machine first.",
            color="warning",
        )

    provider = Provider(args.provider, args.model, args.server_address)
    agent = TradingAgent(
        name="TradingBot",
        prompt_path=PROMPT_PATH,
        provider=provider,
        verbose=args.verbose,
    )

    pretty_print("Robinhood Trading Agent started.", color="status")

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

        pretty_print(f"Next cycle in {args.interval}s. Press Ctrl+C to stop.", color="status")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            pretty_print("Stopped by user.", color="warning")
            break


if __name__ == "__main__":
    asyncio.run(main())
