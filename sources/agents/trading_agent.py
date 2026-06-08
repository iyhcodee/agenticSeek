import os
import json
import re
import asyncio

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.robinhoodMCP import RobinhoodMCPTool
from sources.tools.trading_journal import (
    log_trade,
    daily_summary,
    update_high_water_mark,
    is_daily_loss_limit_hit,
    is_drawdown_limit_hit,
    get_drawdown_pct,
    get_daily_realized_pnl,
)
from sources.memory import Memory


DAILY_LOSS_LIMIT_PCT = 0.20   # stop new trades if daily loss exceeds 20% of account
DRAWDOWN_STOP_PCT    = 0.35   # halt all trading if account drops 35% from recent high


class TradingAgent(Agent):
    """
    Autonomous Robinhood trading agent.
    Connects to the official Robinhood MCP server, enforces risk limits,
    maintains a trading journal, and uses the LLM for AI-driven decisions.
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
        self._trading_halted = False
        self._halt_reason = ""

    # ---------------------------------------------------------------------- #
    #  Risk guard                                                              #
    # ---------------------------------------------------------------------- #

    def _check_risk_limits(self, account_value: float) -> bool:
        """
        Return True if trading is allowed.
        Side-effects: sets self._trading_halted and self._halt_reason if limits breached.
        """
        if self._trading_halted:
            return False

        hwm = update_high_water_mark(account_value)

        if is_drawdown_limit_hit(account_value, DRAWDOWN_STOP_PCT):
            dd = get_drawdown_pct(account_value) * 100
            self._trading_halted = True
            self._halt_reason = (
                f"DRAWDOWN STOP TRIGGERED: account is down {dd:.1f}% from recent high "
                f"(${hwm:,.2f}). All trading halted. Please review and authorize restart."
            )
            pretty_print(self._halt_reason, color="failure")
            return False

        if is_daily_loss_limit_hit(account_value, DAILY_LOSS_LIMIT_PCT):
            pnl = get_daily_realized_pnl()
            self._halt_reason = (
                f"DAILY LOSS LIMIT HIT: realized P&L today is ${pnl:+.2f}. "
                f"No new trades will be opened for the rest of the day."
            )
            pretty_print(self._halt_reason, color="warning")
            return False

        return True

    # ---------------------------------------------------------------------- #
    #  Market context builder                                                  #
    # ---------------------------------------------------------------------- #

    def _build_market_context(self) -> tuple[str, float]:
        """
        Fetch live portfolio and account data.
        Returns (context_string, account_equity_float).
        """
        tool = self.tools["robinhood"]
        lines = ["=== LIVE MARKET CONTEXT ==="]
        account_value = 0.0

        try:
            portfolio = tool.get_portfolio()
            lines.append(f"Portfolio:\n{portfolio}")
        except Exception as e:
            lines.append(f"Portfolio fetch error: {e}")

        try:
            account_raw = tool.get_account_info()
            lines.append(f"Account:\n{account_raw}")
            # Try to parse equity from account data
            account_value = _parse_equity(account_raw)
        except Exception as e:
            lines.append(f"Account fetch error: {e}")

        # Inject risk status
        pnl = get_daily_realized_pnl()
        dd_pct = get_drawdown_pct(account_value) * 100 if account_value > 0 else 0
        lines.append(f"Daily realized P&L: ${pnl:+.2f}")
        lines.append(f"Drawdown from high: {dd_pct:.1f}%")
        if self._trading_halted or not self._check_risk_limits(account_value):
            lines.append(f"RISK STATUS: TRADING HALTED — {self._halt_reason}")
        else:
            lines.append("RISK STATUS: OK — trading allowed")

        lines.append(f"\n{daily_summary()}")
        lines.append("===========================")
        return "\n".join(lines), account_value

    # ---------------------------------------------------------------------- #
    #  Order logging intercept                                                 #
    # ---------------------------------------------------------------------- #

    def _log_order_from_blocks(self, answer: str) -> None:
        """Scan LLM answer for place_order blocks and log them to the journal."""
        pattern = r"```robinhood\s*([\s\S]*?)```"
        for match in re.finditer(pattern, answer):
            raw = match.group(1).strip()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if cmd.get("action") == "place_order":
                entry = {
                    "type": "order",
                    "symbol": cmd.get("symbol"),
                    "side": cmd.get("side"),
                    "quantity": cmd.get("quantity"),
                    "order_type": cmd.get("order_type", "limit"),
                    "limit_price": cmd.get("limit_price"),
                    "confidence": _extract_confidence(answer),
                }
                log_trade(entry)

    # ---------------------------------------------------------------------- #
    #  Main process loop                                                       #
    # ---------------------------------------------------------------------- #

    async def process(self, user_prompt: str, speech_module=None):
        """
        Main agentic loop:
        inject live context → LLM reasons → execute tool blocks → repeat until done.
        """
        market_context, account_value = self._build_market_context()

        # Build risk-aware prefix
        if self._trading_halted:
            risk_prefix = (
                f"\n\n[SYSTEM RISK ALERT] {self._halt_reason}\n"
                "Do NOT place any new orders. You may review positions and report status only.\n\n"
            )
        elif not self._check_risk_limits(account_value):
            risk_prefix = (
                f"\n\n[SYSTEM RISK ALERT] {self._halt_reason}\n"
                "Do NOT open new positions. You may manage existing positions and report status.\n\n"
            )
        else:
            risk_prefix = ""

        full_prompt = f"{market_context}{risk_prefix}\nUser instruction: {user_prompt}"
        self.memory.push("user", full_prompt)

        max_iterations = 12
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            animate_thinking(f"Trading cycle {iteration}...", color="status")

            answer, reasoning = await self.llm_request()

            # Log any order blocks before executing
            self._log_order_from_blocks(answer)

            exec_success, feedback = self.execute_modules(answer)
            answer_clean = self.remove_blocks(answer)
            self.last_answer = answer_clean

            if len(self.blocks_result) == 0:
                break

            self.blocks_result = []
            self.status_message = f"Cycle {iteration} complete"

            if not exec_success:
                pretty_print(f"Tool execution error in cycle {iteration}: {feedback}", color="failure")

        self.status_message = "Ready"
        return self.last_answer, reasoning


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _parse_equity(account_raw) -> float:
    """Try to extract a numeric equity/portfolio_value from account data."""
    if isinstance(account_raw, dict):
        for key in ("equity", "portfolio_value", "total_value", "market_value"):
            val = account_raw.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    if isinstance(account_raw, str):
        # Attempt to find "equity": "12345.67" style in string
        for pattern in (r'"equity"\s*:\s*"?([\d.]+)"?', r'equity[:\s]+([\d.]+)'):
            m = re.search(pattern, account_raw, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return 0.0


def _extract_confidence(answer: str) -> str:
    """Pull confidence level from the AI decision memo in the answer text."""
    m = re.search(r"confidence level[:\s]+([^\n\-]+)", answer, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "unknown"
