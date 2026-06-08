import os
import asyncio

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.robinhoodTradingTool import RobinhoodTradingTool
from sources.memory import Memory


class TradingAgent(Agent):
    """
    Autonomous Robinhood trading agent.
    Uses robin_stocks to fetch live market data and execute trades.
    The LLM drives all decisions via JSON action blocks.
    """

    def __init__(self, name, prompt_path, provider, verbose=False, dry_run=True):
        super().__init__(name, prompt_path, provider, verbose, None)
        self.role = "trading"
        self.type = "trading_agent"
        self.dry_run = dry_run
        self.tools = {
            "robinhood": RobinhoodTradingTool(dry_run=dry_run),
        }
        self.memory = Memory(
            self.load_prompt(prompt_path),
            recover_last_session=False,
            memory_compression=False,
            model_provider=provider.get_model_name(),
        )

    def _build_market_context(self) -> str:
        """Fetch live portfolio state to prepend to each trading cycle."""
        tool = self.tools["robinhood"]
        lines = ["=== LIVE ACCOUNT STATE ==="]
        try:
            portfolio = tool.get_portfolio()
            lines.append(f"Portfolio:\n{portfolio}")
        except Exception as e:
            lines.append(f"Portfolio unavailable: {e}")
        try:
            orders = tool.get_open_orders()
            lines.append(f"Open orders:\n{orders}")
        except Exception as e:
            lines.append(f"Open orders unavailable: {e}")
        lines.append("==========================")
        return "\n".join(lines)

    async def process(self, user_prompt: str, speech_module=None):
        """
        Agentic trading loop.
        Each iteration: inject live context → LLM decides → execute blocks → repeat.
        """
        market_context = self._build_market_context()
        full_prompt = f"{market_context}\n\nInstruction: {user_prompt}"
        self.memory.push("user", full_prompt)

        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            animate_thinking(f"Trading cycle {iteration}/{max_iterations}...", color="status")

            answer, reasoning = await self.llm_request()
            exec_success, feedback = self.execute_modules(answer)
            self.last_answer = self.remove_blocks(answer)

            if len(self.blocks_result) == 0:
                break

            self.blocks_result = []
            self.status_message = f"Cycle {iteration} complete"

            if not exec_success:
                pretty_print(f"Tool error in cycle {iteration}: {feedback}", color="failure")

        self.status_message = "Ready"
        return self.last_answer, reasoning
