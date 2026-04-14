import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Security & API Keys ---
CAPTCHA_API_KEY = os.getenv("CAPTCHA_API_KEY")
DEFAULT_PROXY = os.getenv("DEFAULT_PROXY")
GOLOGIN_API_TOKEN = os.getenv("GOLOGIN_API_TOKEN")

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
