"""
Trading journal — persists trade log, daily P&L, and drawdown tracking.
All data is stored in JSON under logs/trading_journal/.
"""

import os
import json
import time
from datetime import date, datetime
from typing import Dict, List, Optional


JOURNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "trading_journal")


def _ensure_dir():
    os.makedirs(JOURNAL_DIR, exist_ok=True)


def _today() -> str:
    return date.today().isoformat()


def _journal_path(day: str = None) -> str:
    _ensure_dir()
    return os.path.join(JOURNAL_DIR, f"journal_{day or _today()}.json")


def _high_water_path() -> str:
    _ensure_dir()
    return os.path.join(JOURNAL_DIR, "high_water_mark.json")


# --------------------------------------------------------------------------- #
#  Trade logging                                                                #
# --------------------------------------------------------------------------- #

def log_trade(entry: Dict) -> None:
    """Append a trade record to today's journal file."""
    path = _journal_path()
    records: List[Dict] = []
    if os.path.exists(path):
        with open(path, "r") as f:
            records = json.load(f)
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    records.append(entry)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def get_today_trades() -> List[Dict]:
    path = _journal_path()
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def get_trades_for_day(day: str) -> List[Dict]:
    path = _journal_path(day)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Daily realized P&L tracking                                                 #
# --------------------------------------------------------------------------- #

def record_realized_pnl(pnl_usd: float, account_value_at_open: float) -> None:
    """Record a realized P&L event for today."""
    log_trade({
        "type": "realized_pnl",
        "pnl_usd": pnl_usd,
        "account_value_at_open": account_value_at_open,
    })


def get_daily_realized_pnl() -> float:
    """Sum realized P&L events for today."""
    trades = get_today_trades()
    return sum(t.get("pnl_usd", 0) for t in trades if t.get("type") == "realized_pnl")


def is_daily_loss_limit_hit(account_value: float, limit_pct: float = 0.20) -> bool:
    """Return True if today's realized losses exceed limit_pct of current account value."""
    pnl = get_daily_realized_pnl()
    if pnl >= 0:
        return False
    return abs(pnl) >= account_value * limit_pct


# --------------------------------------------------------------------------- #
#  High-water mark / drawdown tracking                                         #
# --------------------------------------------------------------------------- #

def update_high_water_mark(current_value: float) -> float:
    """Update and return the all-time high account value."""
    path = _high_water_path()
    hwm = current_value
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        hwm = max(data.get("high_water_mark", 0), current_value)
    with open(path, "w") as f:
        json.dump({"high_water_mark": hwm, "updated": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    return hwm


def get_high_water_mark() -> float:
    path = _high_water_path()
    if not os.path.exists(path):
        return 0.0
    with open(path, "r") as f:
        return json.load(f).get("high_water_mark", 0.0)


def is_drawdown_limit_hit(current_value: float, limit_pct: float = 0.35) -> bool:
    """Return True if account has dropped limit_pct from its recent high."""
    hwm = get_high_water_mark()
    if hwm <= 0:
        return False
    drawdown = (hwm - current_value) / hwm
    return drawdown >= limit_pct


def get_drawdown_pct(current_value: float) -> float:
    hwm = get_high_water_mark()
    if hwm <= 0:
        return 0.0
    return max(0.0, (hwm - current_value) / hwm)


# --------------------------------------------------------------------------- #
#  Summary report                                                               #
# --------------------------------------------------------------------------- #

def daily_summary() -> str:
    trades = get_today_trades()
    pnl = get_daily_realized_pnl()
    order_trades = [t for t in trades if t.get("type") == "order"]
    lines = [
        f"=== Trading Journal — {_today()} ===",
        f"Orders logged today: {len(order_trades)}",
        f"Realized P&L today:  ${pnl:+.2f}",
    ]
    for t in order_trades:
        ts = t.get("timestamp", "")[:19]
        lines.append(
            f"  [{ts}] {t.get('side','?').upper()} {t.get('quantity','?')} {t.get('symbol','?')} "
            f"@ {t.get('order_type','?')} | confidence={t.get('confidence','?')}"
        )
    return "\n".join(lines)
