import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.tools.tools import Tools

# Silence robin_stocks' own verbose logging
logging.getLogger("robin_stocks").setLevel(logging.WARNING)


def _mask(value: str, keep: int = 4) -> str:
    """Mask a sensitive string, showing only the last `keep` chars."""
    if not value or len(value) <= keep:
        return "***"
    return "*" * (len(value) - keep) + value[-keep:]


class RobinhoodTradingTool(Tools):
    """
    Trading tool backed by robin_stocks (Robinhood's API).
    Credentials are read exclusively from environment variables — never from code.

    Required env vars:
        ROBINHOOD_USERNAME   your Robinhood login email
        ROBINHOOD_PASSWORD   your Robinhood password

    Optional:
        ROBINHOOD_MFA_CODE   TOTP code if you use an authenticator app
                             (if not set, the library will send an SMS and prompt)

    Session tokens are cached by robin_stocks at ~/.tokens/robinhood.pickle
    so re-login is only needed when the session expires.
    """

    def __init__(self):
        super().__init__()
        self.tag = "robinhood"
        self.name = "Robinhood Trading Tool"
        self.description = (
            "Execute live trades and fetch market data via Robinhood. "
            "Supports: get_portfolio, get_account_info, get_quotes, "
            "get_options_chain, place_order, place_options_order, cancel_order."
        )
        self._rh = None          # robin_stocks.robinhood module, set after login
        self._logged_in = False

    # ------------------------------------------------------------------ #
    #  Authentication                                                       #
    # ------------------------------------------------------------------ #

    def login(self) -> None:
        """
        Authenticate with Robinhood using env-var credentials.
        Raises EnvironmentError if required vars are missing.
        Raises RuntimeError on auth failure.
        """
        import robin_stocks.robinhood as rh

        username = os.getenv("ROBINHOOD_USERNAME", "").strip()
        password = os.getenv("ROBINHOOD_PASSWORD", "").strip()

        if not username or not password:
            raise EnvironmentError(
                "ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set. "
                "Run setup_trading.py to configure them securely."
            )

        mfa_code = os.getenv("ROBINHOOD_MFA_CODE", "").strip() or None

        try:
            rh.login(
                username=username,
                password=password,
                expiresIn=86400,     # 24-hour session
                by_sms=(mfa_code is None),
                mfa_code=mfa_code,
                store_session=True,  # cache token at ~/.tokens/robinhood.pickle
            )
        except Exception as e:
            raise RuntimeError(f"Robinhood login failed: {e}") from e

        self._rh = rh
        self._logged_in = True

    def ensure_logged_in(self) -> None:
        if not self._logged_in:
            self.login()

    def logout(self) -> None:
        if self._rh and self._logged_in:
            try:
                self._rh.logout()
            except Exception:
                pass
            self._logged_in = False

    # ------------------------------------------------------------------ #
    #  Market data                                                          #
    # ------------------------------------------------------------------ #

    def get_portfolio(self) -> Dict:
        self.ensure_logged_in()
        holdings = self._rh.account.build_holdings()
        profile = self._rh.account.load_portfolio_profile()
        return {
            "holdings": holdings,
            "equity": profile.get("equity"),
            "extended_hours_equity": profile.get("extended_hours_equity"),
            "buying_power": self.get_buying_power(),
        }

    def get_buying_power(self) -> str:
        self.ensure_logged_in()
        account = self._rh.account.load_account_profile()
        return account.get("buying_power", "unknown")

    def get_account_info(self) -> Dict:
        self.ensure_logged_in()
        account = self._rh.account.load_account_profile()
        profile = self._rh.account.load_portfolio_profile()
        return {
            "buying_power": account.get("buying_power"),
            "cash": account.get("cash"),
            "equity": profile.get("equity"),
            "total_return": profile.get("total_return"),
        }

    def get_quotes(self, symbols: List[str]) -> List[Dict]:
        self.ensure_logged_in()
        results = []
        for symbol in symbols:
            try:
                q = self._rh.stocks.get_stock_quote_by_symbol(symbol)
                results.append({
                    "symbol": symbol,
                    "price": q.get("last_trade_price") or q.get("last_extended_hours_trade_price"),
                    "ask": q.get("ask_price"),
                    "bid": q.get("bid_price"),
                    "volume": q.get("volume"),
                    "previous_close": q.get("previous_close"),
                })
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})
        return results

    def get_options_chain(self, symbol: str, expiration_date: str = None,
                          option_type: str = "call") -> List[Dict]:
        self.ensure_logged_in()
        try:
            if expiration_date:
                options = self._rh.options.find_options_by_expiration(
                    symbol, expiration_date, optionType=option_type
                )
            else:
                options = self._rh.options.find_options_by_expiration(
                    symbol, optionType=option_type
                )
            return [
                {
                    "id": o.get("id"),
                    "strike": o.get("strike_price"),
                    "expiration": o.get("expiration_date"),
                    "type": o.get("type"),
                    "ask": o.get("ask_price"),
                    "bid": o.get("bid_price"),
                    "iv": o.get("implied_volatility"),
                    "delta": o.get("delta"),
                    "volume": o.get("volume"),
                    "open_interest": o.get("open_interest"),
                }
                for o in (options or [])
            ]
        except Exception as e:
            return [{"error": str(e)}]

    # ------------------------------------------------------------------ #
    #  Order execution                                                      #
    # ------------------------------------------------------------------ #

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market",
                    limit_price: float = None,
                    stop_price: float = None) -> Dict:
        self.ensure_logged_in()
        try:
            if side == "buy":
                if order_type == "market":
                    result = self._rh.orders.order_buy_market(symbol, quantity)
                elif order_type == "limit" and limit_price:
                    result = self._rh.orders.order_buy_limit(symbol, quantity, limit_price)
                elif order_type == "stop_loss" and stop_price:
                    result = self._rh.orders.order_buy_stop_loss(symbol, quantity, stop_price)
                else:
                    return {"error": f"Invalid buy order config: type={order_type}"}
            elif side == "sell":
                if order_type == "market":
                    result = self._rh.orders.order_sell_market(symbol, quantity)
                elif order_type == "limit" and limit_price:
                    result = self._rh.orders.order_sell_limit(symbol, quantity, limit_price)
                elif order_type == "stop_loss" and stop_price:
                    result = self._rh.orders.order_sell_stop_loss(symbol, quantity, stop_price)
                else:
                    return {"error": f"Invalid sell order config: type={order_type}"}
            else:
                return {"error": f"Unknown side: {side}"}

            return {
                "order_id": result.get("id"),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "type": order_type,
                "state": result.get("state"),
                "created_at": result.get("created_at"),
            }
        except Exception as e:
            return {"error": str(e)}

    def place_options_order(self, symbol: str, expiration_date: str,
                            strike_price: str, option_type: str,
                            side: str, quantity: int,
                            limit_price: float) -> Dict:
        self.ensure_logged_in()
        try:
            if side == "buy":
                result = self._rh.orders.order_buy_option_limit(
                    positionEffect="open",
                    creditOrDebit="debit",
                    price=limit_price,
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration_date,
                    strike=strike_price,
                    optionType=option_type,
                )
            elif side == "sell":
                result = self._rh.orders.order_sell_option_limit(
                    positionEffect="close",
                    creditOrDebit="credit",
                    price=limit_price,
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration_date,
                    strike=strike_price,
                    optionType=option_type,
                )
            else:
                return {"error": f"Unknown side: {side}"}
            return {
                "order_id": result.get("id"),
                "symbol": symbol,
                "side": side,
                "option_type": option_type,
                "strike": strike_price,
                "expiration": expiration_date,
                "quantity": quantity,
                "limit_price": limit_price,
                "state": result.get("state"),
            }
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, order_id: str) -> Dict:
        self.ensure_logged_in()
        try:
            result = self._rh.orders.cancel_stock_order(order_id)
            return {"cancelled": True, "order_id": order_id, "result": result}
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self) -> List[Dict]:
        self.ensure_logged_in()
        try:
            orders = self._rh.orders.get_all_open_stock_orders()
            return [
                {
                    "id": o.get("id"),
                    "symbol": o.get("instrument_data", {}).get("symbol"),
                    "side": o.get("side"),
                    "quantity": o.get("quantity"),
                    "type": o.get("type"),
                    "price": o.get("price"),
                    "state": o.get("state"),
                }
                for o in (orders or [])
            ]
        except Exception as e:
            return [{"error": str(e)}]

    # ------------------------------------------------------------------ #
    #  Tools interface — block execution used by the agent loop            #
    # ------------------------------------------------------------------ #

    def execute(self, blocks: List[str], safety: bool = False) -> str:
        output_parts = []
        for block in blocks:
            block = block.strip()
            try:
                cmd = json.loads(block)
            except json.JSONDecodeError as e:
                output_parts.append(f"Error: invalid JSON block — {e}")
                continue

            action = cmd.pop("action", None)
            if not action:
                output_parts.append("Error: missing 'action' field.")
                continue

            try:
                if action == "get_portfolio":
                    result = self.get_portfolio()
                elif action == "get_account_info":
                    result = self.get_account_info()
                elif action == "get_quotes":
                    result = self.get_quotes(cmd.get("symbols", []))
                elif action == "get_options_chain":
                    result = self.get_options_chain(
                        cmd["symbol"],
                        cmd.get("expiration_date"),
                        cmd.get("option_type", "call"),
                    )
                elif action == "place_order":
                    result = self.place_order(
                        cmd["symbol"], cmd["side"], cmd["quantity"],
                        cmd.get("order_type", "market"),
                        cmd.get("limit_price"),
                        cmd.get("stop_price"),
                    )
                elif action == "place_options_order":
                    result = self.place_options_order(
                        cmd["symbol"], cmd["expiration_date"],
                        cmd["strike_price"], cmd["option_type"],
                        cmd["side"], cmd["quantity"], cmd["limit_price"],
                    )
                elif action == "cancel_order":
                    result = self.cancel_order(cmd["order_id"])
                elif action == "get_open_orders":
                    result = self.get_open_orders()
                else:
                    result = {"error": f"Unknown action: '{action}'"}

                output_parts.append(json.dumps(result, indent=2))

            except KeyError as e:
                output_parts.append(f"Error: missing required field {e} for action '{action}'")
            except Exception as e:
                output_parts.append(f"Error executing '{action}': {e}")

        return "\n---\n".join(output_parts)

    def execution_failure_check(self, output: str) -> bool:
        if not output.strip():
            return True
        low = output.lower()
        return '"error"' in low or "error:" in low

    def interpreter_feedback(self, output: str) -> str:
        if not output:
            return "Robinhood returned no output."
        return f"Robinhood result:\n{output}"


if __name__ == "__main__":
    tool = RobinhoodTradingTool()
    tool.ensure_logged_in()
    print(json.dumps(tool.get_account_info(), indent=2))
