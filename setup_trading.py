"""
Secure setup wizard for the Robinhood trading agent.
Saves credentials to .env (which is git-ignored — never committed).

Run once:
    python setup_trading.py

Then run the bot:
    python run_trading_agent.py --once
"""

import os
import sys
import getpass
from pathlib import Path

ENV_FILE = Path(".env")


def read_existing() -> dict:
    existing = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
    return existing


def write_env(values: dict) -> None:
    existing = read_existing()
    existing.update(values)
    lines = ["# Trading agent credentials — DO NOT COMMIT THIS FILE\n"]
    for k, v in existing.items():
        lines.append(f"{k}={v}\n")
    ENV_FILE.write_text("".join(lines))
    # Restrict file permissions to owner-read-only (Unix)
    try:
        ENV_FILE.chmod(0o600)
    except Exception:
        pass


def prompt(label: str, env_key: str, secret: bool = False,
           existing: dict = None) -> str:
    current = (existing or {}).get(env_key, "")
    if current:
        hint = f" [current: {'*' * 8 + current[-4:] if secret else current}]"
    else:
        hint = ""
    if secret:
        value = getpass.getpass(f"  {label}{hint}: ").strip()
    else:
        value = input(f"  {label}{hint}: ").strip()
    return value if value else current


def main():
    print("\n=== Robinhood Trading Agent — Secure Setup ===\n")
    print("Credentials are saved to .env (git-ignored, chmod 600).")
    print("Nothing is sent anywhere except directly to Robinhood and Anthropic.\n")

    existing = read_existing()

    print("[ Anthropic / Claude ]")
    anthropic_key = prompt("Anthropic API key (sk-ant-...)", "ANTHROPIC_API_KEY",
                           secret=True, existing=existing)

    print("\n[ Robinhood ]")
    rh_user = prompt("Robinhood email", "ROBINHOOD_USERNAME",
                     secret=False, existing=existing)
    rh_pass = prompt("Robinhood password", "ROBINHOOD_PASSWORD",
                     secret=True, existing=existing)

    print("\n  MFA / 2FA (leave blank if you want SMS codes instead):")
    rh_mfa = prompt("Authenticator app code (optional)", "ROBINHOOD_MFA_CODE",
                    secret=True, existing=existing)

    values = {
        "ANTHROPIC_API_KEY": anthropic_key,
        "ROBINHOOD_USERNAME": rh_user,
        "ROBINHOOD_PASSWORD": rh_pass,
    }
    if rh_mfa:
        values["ROBINHOOD_MFA_CODE"] = rh_mfa
    elif "ROBINHOOD_MFA_CODE" in existing:
        del existing["ROBINHOOD_MFA_CODE"]

    write_env(values)
    print(f"\nSaved to {ENV_FILE.resolve()} (permissions: 600)")
    print("\nNext step:  python run_trading_agent.py --once")

    # Optionally test Robinhood login right now
    test = input("\nTest Robinhood login now? (y/N): ").strip().lower()
    if test == "y":
        # Load the env vars we just wrote
        for k, v in values.items():
            os.environ[k] = v
        print("Logging in to Robinhood...")
        try:
            from sources.tools.robinhoodTradingTool import RobinhoodTradingTool
            tool = RobinhoodTradingTool()
            tool.login()
            info = tool.get_account_info()
            bp = info.get("buying_power", "?")
            equity = info.get("equity", "?")
            print(f"  Login successful!")
            print(f"  Equity:        ${equity}")
            print(f"  Buying power:  ${bp}")
            tool.logout()
        except Exception as e:
            print(f"  Login failed: {e}")
            print("  Check your username/password and try again.")


if __name__ == "__main__":
    main()
