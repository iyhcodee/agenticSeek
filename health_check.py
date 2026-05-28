#!/usr/bin/env python3
"""
AgenticSeek Health Check
Run: python3 health_check.py
"""

import sys
import os
import configparser
import socket
import shutil
from typing import Optional

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, **kwargs):
        return text

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: list[tuple[str, str, str]] = []  # (label, status, detail)

def _record(label: str, status: str, detail: str = ""):
    results.append((label, status, detail))

def ok(label: str, detail: str = ""):
    _record(label, "PASS", detail)

def fail(label: str, detail: str = ""):
    _record(label, "FAIL", detail)

def warn(label: str, detail: str = ""):
    _record(label, "WARN", detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_config() -> Optional[dict]:
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    if not os.path.exists(config_path):
        fail("config.ini", "file not found")
        return None

    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    required_keys = {
        "MAIN": ["provider_name", "provider_model", "provider_server_address", "is_local"],
        "BROWSER": ["headless_browser"],
    }
    missing = []
    for section, keys in required_keys.items():
        if not cfg.has_section(section):
            missing.append(f"[{section}] section missing")
            continue
        for key in keys:
            if not cfg.has_option(section, key):
                missing.append(f"[{section}] {key}")

    if missing:
        fail("config.ini", "missing keys: " + ", ".join(missing))
        return None

    ok("config.ini", f"provider={cfg['MAIN']['provider_name']}  model={cfg['MAIN']['provider_model']}")
    return {
        "provider": cfg["MAIN"]["provider_name"].strip().lower(),
        "model": cfg["MAIN"]["provider_model"].strip(),
        "server": cfg["MAIN"]["provider_server_address"].strip(),
        "is_local": cfg["MAIN"].get("is_local", "true").strip().lower() == "true",
        "speak": cfg["MAIN"].get("speak", "false").strip().lower() == "true",
        "listen": cfg["MAIN"].get("listen", "false").strip().lower() == "true",
        "headless": cfg["BROWSER"].get("headless_browser", "true").strip().lower() == "true",
    }


def check_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        ok(".env file", "present")
        return True
    warn(".env file", "not found — needed for cloud LLM providers")
    return False


def check_api_key(provider: str):
    cloud_providers = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "dsk_deepseek": "DSK_DEEPSEEK_API_KEY",
    }
    if provider not in cloud_providers:
        return  # local provider, no key needed

    var = cloud_providers[provider]
    # try loading .env manually so we don't depend on dotenv being importable
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_vars: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")

    key = os.environ.get(var) or env_vars.get(var)
    if key:
        ok(f"API key ({var})", f"set ({len(key)} chars)")
    else:
        fail(f"API key ({var})", f"not set — add {var} to .env")


def check_redis():
    try:
        s = socket.create_connection(("127.0.0.1", 6379), timeout=2)
        s.close()
        ok("Redis", "reachable on 127.0.0.1:6379")
    except OSError:
        warn("Redis", "not reachable on 127.0.0.1:6379 — needed for Celery task queue")


def check_ollama(server_address: str):
    if not HAS_REQUESTS:
        warn("Ollama", "requests not installed, skipping check")
        return
    host = f"http://{server_address}" if not server_address.startswith("http") else server_address
    url = host.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            ok("Ollama", f"online — {len(models)} model(s) available")
            return models
        else:
            fail("Ollama", f"responded with HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        fail("Ollama", f"cannot connect to {url}")
    except requests.exceptions.Timeout:
        fail("Ollama", f"timed out connecting to {url}")
    return []


def check_ollama_model(server_address: str, model: str):
    if not HAS_REQUESTS:
        return
    host = f"http://{server_address}" if not server_address.startswith("http") else server_address
    url = host.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            available = [m.get("name", "") for m in resp.json().get("models", [])]
            if any(model in name for name in available):
                ok(f"Ollama model ({model})", "found")
            else:
                warn(f"Ollama model ({model})", f"not found in local models — run: ollama pull {model}")
    except Exception:
        pass  # already reported in check_ollama


def check_searxng():
    if not HAS_REQUESTS:
        warn("SearxNG", "requests not installed, skipping check")
        return
    for url in ["http://127.0.0.1:8080", "http://localhost:8080"]:
        try:
            resp = requests.get(url, timeout=4)
            if resp.status_code == 200:
                ok("SearxNG", f"reachable at {url}")
                return
        except Exception:
            pass
    warn("SearxNG", "not reachable on port 8080 — web search will fail (start via docker-compose)")


def check_chromedriver():
    path = shutil.which("chromedriver")
    if path:
        ok("ChromeDriver", f"found at {path}")
    else:
        # chromedriver-autoinstaller handles this at runtime, so just warn
        warn("ChromeDriver", "not in PATH — chromedriver-autoinstaller will fetch it on first run")


def check_python_packages():
    critical = [
        ("requests", "requests"),
        ("fastapi", "fastapi"),
        ("celery", "celery"),
        ("selenium", "selenium"),
        ("transformers", "transformers"),
        ("ollama", "ollama"),
        ("openai", "openai"),
        ("colorama", "colorama"),
        ("dotenv", "python-dotenv"),
    ]
    missing = []
    for module, pkg in critical:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        fail("Python packages", "missing: " + ", ".join(missing) + " — run: pip install -r requirements.txt")
    else:
        ok("Python packages", "all critical imports available")


def check_work_dir():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.ini")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    work_dir = cfg.get("MAIN", "work_dir", fallback="").strip()
    if not work_dir:
        warn("work_dir", "not set in config.ini — agent file operations may fail")
        return
    if os.path.isdir(work_dir):
        ok("work_dir", f"{work_dir}")
    else:
        warn("work_dir", f"{work_dir} does not exist — create it or update config.ini")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary():
    width = 60
    print()
    print(colored("=" * width, "cyan"))
    print(colored("  AgenticSeek Health Check", "cyan"))
    print(colored("=" * width, "cyan"))

    passes = fails = warns = 0
    for label, status, detail in results:
        if status == "PASS":
            badge = colored("  PASS  ", "white", on_color="on_green")
            passes += 1
        elif status == "FAIL":
            badge = colored("  FAIL  ", "white", on_color="on_red")
            fails += 1
        else:
            badge = colored("  WARN  ", "white", on_color="on_yellow")
            warns += 1

        label_col = label.ljust(28)
        line = f"  {badge}  {label_col}"
        if detail:
            line += f"  {colored(detail, 'dark_grey')}"
        print(line)

    print(colored("-" * width, "cyan"))
    summary = f"  {passes} passed  {warns} warnings  {fails} failed"
    if fails > 0:
        print(colored(summary, "red"))
    elif warns > 0:
        print(colored(summary, "yellow"))
    else:
        print(colored(summary, "green"))
    print(colored("=" * width, "cyan"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(colored("\nRunning agenticSeek health checks...\n", "cyan"))

    check_python_packages()
    cfg = check_config()
    check_env_file()
    check_work_dir()
    check_redis()
    check_searxng()
    check_chromedriver()

    if cfg:
        provider = cfg["provider"]
        check_api_key(provider)

        if provider == "ollama":
            check_ollama(cfg["server"])
            check_ollama_model(cfg["server"], cfg["model"])
        elif provider in ("openai", "deepseek", "google", "together", "huggingface", "openrouter", "dsk_deepseek"):
            ok("LLM provider", f"{provider} (cloud) — connectivity depends on API key")
        elif provider == "lm-studio":
            if HAS_REQUESTS:
                url = f"http://{cfg['server']}/v1/models"
                try:
                    resp = requests.get(url, timeout=4)
                    ok("LM Studio", f"online at {cfg['server']}") if resp.status_code == 200 else fail("LM Studio", f"HTTP {resp.status_code}")
                except Exception:
                    fail("LM Studio", f"cannot connect to {cfg['server']}")
        elif provider == "server":
            if HAS_REQUESTS:
                url = f"http://{cfg['server']}/setup"
                try:
                    requests.get(url, timeout=4)
                    ok("Custom server", f"reachable at {cfg['server']}")
                except Exception:
                    fail("Custom server", f"cannot connect to {cfg['server']}")

    print_summary()
    sys.exit(1 if any(s == "FAIL" for _, s, _ in results) else 0)


if __name__ == "__main__":
    main()
