import os
import sys
import json
import time
import uuid
import requests
from typing import Any, Dict, List, Optional

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.tools.tools import Tools


MCP_ENDPOINT = "https://agent.robinhood.com/mcp/trading"


class RobinhoodMCPTool(Tools):
    """
    MCP client for the Robinhood trading MCP server at agent.robinhood.com.
    Implements the MCP HTTP transport (JSON-RPC 2.0).
    """

    def __init__(self, auth_token: str = None):
        super().__init__()
        self.tag = "robinhood"
        self.name = "Robinhood MCP"
        self.description = (
            "Execute trades and fetch market data via the Robinhood MCP server. "
            "Supports get_portfolio, get_quotes, place_order, get_options_chain, "
            "cancel_order, and get_account_info."
        )
        self.endpoint = MCP_ENDPOINT
        self.auth_token = auth_token or os.getenv("ROBINHOOD_MCP_TOKEN", "")
        self.session_id = str(uuid.uuid4())
        self._available_tools: Optional[List[Dict]] = None

    # ------------------------------------------------------------------ #
    #  Low-level MCP transport                                             #
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def _rpc(self, method: str, params: Any = None) -> Dict:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP error {data['error']['code']}: {data['error']['message']}")
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Robinhood MCP request failed: {e}")

    # ------------------------------------------------------------------ #
    #  Tool discovery                                                       #
    # ------------------------------------------------------------------ #

    def list_tools(self) -> List[Dict]:
        if self._available_tools is None:
            result = self._rpc("tools/list")
            self._available_tools = result.get("tools", [])
        return self._available_tools

    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        result = self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        if isinstance(content, list) and content:
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(parts)
        return json.dumps(result)

    # ------------------------------------------------------------------ #
    #  Convenience wrappers                                                #
    # ------------------------------------------------------------------ #

    def get_portfolio(self) -> Dict:
        return self.call_tool("get_portfolio", {})

    def get_account_info(self) -> Dict:
        return self.call_tool("get_account_info", {})

    def get_quotes(self, symbols: List[str]) -> Dict:
        return self.call_tool("get_quotes", {"symbols": symbols})

    def get_options_chain(self, symbol: str, expiration_date: str = None) -> Dict:
        args = {"symbol": symbol}
        if expiration_date:
            args["expiration_date"] = expiration_date
        return self.call_tool("get_options_chain", args)

    def place_order(
        self,
        symbol: str,
        side: str,          # "buy" | "sell"
        quantity: float,
        order_type: str = "market",
        limit_price: float = None,
        stop_price: float = None,
    ) -> Dict:
        args = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
        }
        if limit_price is not None:
            args["limit_price"] = limit_price
        if stop_price is not None:
            args["stop_price"] = stop_price
        return self.call_tool("place_order", args)

    def place_options_order(
        self,
        option_id: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: float = None,
    ) -> Dict:
        args = {"option_id": option_id, "side": side, "quantity": quantity, "order_type": order_type}
        if limit_price is not None:
            args["limit_price"] = limit_price
        return self.call_tool("place_options_order", args)

    def cancel_order(self, order_id: str) -> Dict:
        return self.call_tool("cancel_order", {"order_id": order_id})

    def get_order_status(self, order_id: str) -> Dict:
        return self.call_tool("get_order_status", {"order_id": order_id})

    # ------------------------------------------------------------------ #
    #  Tools interface (agent block execution)                             #
    # ------------------------------------------------------------------ #

    def execute(self, blocks: List[str], safety: bool = False) -> str:
        """
        Execute a JSON block emitted by the LLM.
        Expected block format:
            {
                "action": "place_order" | "get_quotes" | "get_portfolio" | ...,
                <action-specific fields>
            }
        """
        output = ""
        for block in blocks:
            block = block.strip()
            try:
                cmd = json.loads(block)
            except json.JSONDecodeError as e:
                output += f"Error parsing JSON block: {e}\n"
                continue

            action = cmd.pop("action", None)
            if action is None:
                output += "Error: missing 'action' field in block.\n"
                continue

            try:
                if action == "get_portfolio":
                    output += str(self.get_portfolio()) + "\n"
                elif action == "get_account_info":
                    output += str(self.get_account_info()) + "\n"
                elif action == "get_quotes":
                    output += str(self.get_quotes(cmd.get("symbols", []))) + "\n"
                elif action == "get_options_chain":
                    output += str(self.get_options_chain(
                        cmd["symbol"], cmd.get("expiration_date")
                    )) + "\n"
                elif action == "place_order":
                    output += str(self.place_order(
                        cmd["symbol"], cmd["side"], cmd["quantity"],
                        cmd.get("order_type", "market"),
                        cmd.get("limit_price"),
                        cmd.get("stop_price"),
                    )) + "\n"
                elif action == "place_options_order":
                    output += str(self.place_options_order(
                        cmd["option_id"], cmd["side"], cmd["quantity"],
                        cmd.get("order_type", "market"),
                        cmd.get("limit_price"),
                    )) + "\n"
                elif action == "cancel_order":
                    output += str(self.cancel_order(cmd["order_id"])) + "\n"
                elif action == "get_order_status":
                    output += str(self.get_order_status(cmd["order_id"])) + "\n"
                elif action == "list_tools":
                    tools = self.list_tools()
                    output += json.dumps(tools, indent=2) + "\n"
                else:
                    output += f"Error: unknown action '{action}'\n"
            except Exception as e:
                output += f"Error executing '{action}': {e}\n"
        return output.strip()

    def execution_failure_check(self, output: str) -> bool:
        out = output.strip().lower()
        if not out:
            return True
        return "error" in out or "failed" in out or "exception" in out

    def interpreter_feedback(self, output: str) -> str:
        if not output:
            return "Robinhood MCP returned no output."
        return f"Robinhood MCP result:\n{output}"


if __name__ == "__main__":
    tool = RobinhoodMCPTool()
    print("Available tools:", tool.list_tools())
