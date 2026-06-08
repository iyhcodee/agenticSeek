import os
import asyncio

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.robinhoodMCP import RobinhoodMCPTool
from sources.memory import Memory


class TradingAgent(Agent):
    """
    Autonomous Robinhood trading agent.
    Connects to the Robinhood MCP server to fetch market data and execute trades.
    Uses the LLM to make extreme-risk, high-return trading decisions.
    """

    def __init__(self, name, prompt_path, provider, verbose=False):
        super().__init__(name, prompt_path, provider, verbose, None)
        self.role = "trading"
        self.type = "trading_agent"
        auth_token = os.getenv("ROBINHOOD_MCP_TOKEN", "")
        self.tools = {
            "robinhood": RobinhoodMCPTool(auth_token=auth_token),
        }
        self.memory = Memory(
            self.load_prompt(prompt_path),
            recover_last_session=False,
            memory_compression=False,
            model_provider=provider.get_model_name(),
        )

    def _build_market_context(self) -> str:
        """
        Pull live portfolio and account state to inject into each cycle's prompt.
        Returns a formatted context string.
        """
        tool = self.tools["robinhood"]
        lines = ["=== LIVE MARKET CONTEXT ==="]
        try:
            portfolio = tool.get_portfolio()
            lines.append(f"Portfolio:\n{portfolio}")
        except Exception as e:
            lines.append(f"Portfolio fetch error: {e}")
        try:
            account = tool.get_account_info()
            lines.append(f"Account:\n{account}")
        except Exception as e:
            lines.append(f"Account fetch error: {e}")
        lines.append("===========================")
        return "\n".join(lines)

    async def process(self, user_prompt: str, speech_module=None):
        """
        Main agentic loop.
        Each iteration: inject context → LLM reasons → execute tool blocks → repeat until done.
        """
        market_context = self._build_market_context()
        full_prompt = f"{market_context}\n\nUser instruction: {user_prompt}"
        self.memory.push("user", full_prompt)

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            animate_thinking(f"Trading cycle {iteration}...", color="status")

            answer, reasoning = await self.llm_request()
            exec_success, feedback = self.execute_modules(answer)
            answer_clean = self.remove_blocks(answer)
            self.last_answer = answer_clean

            if len(self.blocks_result) == 0:
                # LLM produced no tool blocks — it's done reasoning
                break

            self.blocks_result = []
            self.status_message = f"Cycle {iteration} complete"

            if not exec_success:
                pretty_print(f"Tool execution error in cycle {iteration}: {feedback}", color="failure")

        self.status_message = "Ready"
        return self.last_answer, reasoning
