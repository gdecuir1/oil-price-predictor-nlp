import os
import logging
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import quote, urlparse

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


# --- Security & API Keys ---
CAPTCHA_API_KEY = os.getenv("CAPTCHA_API_KEY")
GOLOGIN_API_TOKEN = os.getenv("GOLOGIN_API_TOKEN")

# Smartproxy / Decodo (residential) — get host, user, and password from the dashboard
SMARTPROXY_HOST = os.getenv("SMARTPROXY_HOST", "gate.smartproxy.com")
SMARTPROXY_PORT = _int_env("SMARTPROXY_PORT", 7000)
# Dashboard username (may already include e.g. -country-us; session suffix is added per boot)
SMARTPROXY_USERNAME = os.getenv("SMARTPROXY_USERNAME", "").strip() or None
SMARTPROXY_PASSWORD = os.getenv("SMARTPROXY_PASSWORD", "").strip() or None
# "session" = append -session-n{node}-b{boot} (default). "last_digit" = only rotate final 0-9
SMARTPROXY_ROTATION_MODE = os.getenv("SMARTPROXY_ROTATION_MODE", "session")

DEFAULT_PROXY = os.getenv("DEFAULT_PROXY", "").strip() or None
# Fill DEFAULT_PROXY for test.py / article_scraper from split variables when not set
if not DEFAULT_PROXY and SMARTPROXY_USERNAME and SMARTPROXY_PASSWORD:
    u = quote(SMARTPROXY_USERNAME, safe="")
    p = quote(SMARTPROXY_PASSWORD, safe="")
    DEFAULT_PROXY = f"http://{u}:{p}@{SMARTPROXY_HOST}:{SMARTPROXY_PORT}"
elif not DEFAULT_PROXY:
    DEFAULT_PROXY = None  # type: ignore[assignment]

# When true, a plain DEFAULT_PROXY URL (no SMARTPROXY_USERNAME) is also used for GoLogin API patch
GOLOGIN_APPLY_DEFAULT_PROXY = os.getenv("GOLOGIN_APPLY_DEFAULT_PROXY", "").lower() in (
    "1",
    "true",
    "yes",
)


def parse_proxy_url(
    url: str,
) -> Optional[Tuple[str, int, str, str]]:
    """Return host, port, username, password from http(s)://user:pass@host:port/"""
    p = urlparse(url)
    if not p.hostname or p.port is None:
        return None
    return (p.hostname, int(p.port), p.username or "", p.password or "")


def get_smartproxy_base_credentials() -> Optional[Tuple[str, int, str, str]]:
    """Host, port, user, and password for Smartproxy, if enabled via env or DEFAULT_PROXY."""
    if SMARTPROXY_USERNAME and SMARTPROXY_PASSWORD:
        return (SMARTPROXY_HOST, SMARTPROXY_PORT, SMARTPROXY_USERNAME, SMARTPROXY_PASSWORD)
    if DEFAULT_PROXY and GOLOGIN_APPLY_DEFAULT_PROXY:
        parsed = parse_proxy_url(DEFAULT_PROXY)
        if parsed:
            return parsed
    return None


def rotate_smartproxy_username(username: str, node_index: int, boot_id: int) -> str:
    """
    Sticky / session rotation: Decodo and Smartproxy keep the same IP while the
    -session-... part is unchanged. Change it (or the last digit) to get a new exit IP
    on each GoLogin browser restart.
    """
    if SMARTPROXY_ROTATION_MODE == "last_digit":
        tail = str(boot_id % 10)
        if username and username[-1].isdigit():
            return username[:-1] + tail
        return f"{username}{tail}"

    if "-session-" in username:
        base, _ = username.rsplit("-session-", 1)
        return f"{base}-session-n{node_index}-b{boot_id:05d}"
    return f"{username}-session-n{node_index}-b{boot_id:05d}"


def build_gologin_smartproxy_payload(node_index: int, boot_id: int) -> Optional[dict[str, Any]]:
    """
    GoLogin `changeProfileProxy` body. Returns None if Smartproxy is not configured
    (set SMARTPROXY_USERNAME and SMARTPROXY_PASSWORD, or DEFAULT_PROXY +
    GOLOGIN_APPLY_DEFAULT_PROXY=1).
    """
    creds = get_smartproxy_base_credentials()
    if not creds:
        return None
    host, port, user, pw = creds
    u2 = rotate_smartproxy_username(user, node_index, boot_id)
    return {
        "mode": "http",
        "host": host,
        "port": port,
        "username": u2,
        "password": pw,
        "changeIpUrl": None,
        "autoProxyRegion": None,
        "torProxyRegion": None,
    }


# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "search_results"
RAW_HTML_DIR = BASE_DIR / "raw_html"
RESULTS_DIR.mkdir(exist_ok=True)
RAW_HTML_DIR.mkdir(exist_ok=True)

# --- Human Mimicry Settings ---
MIN_DELAY_SEC = 3.0
MAX_DELAY_SEC = 8.0
SCROLL_MIN = 300
SCROLL_MAX = 700

# --- Scraper Settings ---
MAX_RETRIES = 3
DEFAULT_MAX_RESULTS = 8
PAGE_TIMEOUT_MS = 60000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
