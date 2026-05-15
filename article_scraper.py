import csv
import sys
import time
import random
import hashlib
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, Error as PlaywrightError
from playwright_stealth.stealth import Stealth

from config import (
    logger,
    BASE_DIR,
    MIN_DELAY_SEC,
    MAX_DELAY_SEC,
    MAX_RETRIES,
    PAGE_TIMEOUT_MS,
    DEFAULT_PROXY,
)

RAW_ARTICLES_DIR = BASE_DIR / "raw_articles"
RAW_ARTICLES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "search_results"
PARSED_ARTICLES_DIR = BASE_DIR / "parsed_articles"


def launch_chromium(p, launch_opts: dict[str, Any]):
    """Launch a Chromium-based browser for scraping.

    Tries **installed Google Chrome** first (``channel="chrome"``), so you often do not
    need ``playwright install`` on a Mac with Chrome in ``/Applications``. If that
    fails, uses Playwright's downloaded Chromium. If the bundle is missing, prints
    the exact ``python -m playwright install`` command for your interpreter.
    """
    chrome_opts = {**launch_opts, "channel": "chrome"}
    try:
        return p.chromium.launch(**chrome_opts)
    except PlaywrightError as e:
        logger.debug("Launch with channel=chrome failed (%s); trying bundled Chromium.", e)

    try:
        return p.chromium.launch(**launch_opts)
    except PlaywrightError as e:
        err = str(e)
        if "Executable doesn't exist" not in err:
            raise
        logger.error(
            "Playwright browser bundle is missing. Install it for this Python, then retry:\n"
            "  %s -m playwright install chromium\n"
            "Or install Google Chrome and re-run (the scraper tries Chrome first).",
            sys.executable,
        )
        raise


def load_tasks_from_parsed(parsed_dir: Path) -> list[dict[str, Any]]:
    """Build de-duplicated scrape tasks from ``parser.py`` JSON output.

    Filenames match ``ml_model/data/html_extractor.py``: ``article_<md5(url)>.html``.
    Skips placeholder rows, non-http URLs, and Google redirect/cache hosts.
    """
    seen_urls: set[str] = set()
    tasks: list[dict[str, Any]] = []
    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        logger.error("No *.json found under %s", parsed_dir)
        return tasks

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping unreadable JSON %s: %s", jf.name, e)
            continue

        meta_sd = (data.get("metadata") or {}).get("searchDate") or ""

        for article in data.get("articles", []):
            url = (article.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue
            domain = urlparse(url).hostname or ""
            if domain.endswith("google.com") or domain.endswith("gstatic.com"):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            search_date = article.get("searchDate") or meta_sd or ""
            tasks.append(
                {
                    "url": url,
                    "title": article.get("title") or "",
                    "source": article.get("source") or "",
                    "query": article.get("query") or "",
                    "taskQuery": article.get("taskQuery") or "",
                    "searchDate": search_date,
                    "source_file": article.get("sourceFile") or jf.name,
                    "parsed_json": jf.name,
                    "url_hash": hashlib.md5(url.encode("utf-8")).hexdigest(),
                }
            )

    logger.info(
        "Loaded %d unique URL(s) from %d parsed JSON file(s)",
        len(tasks),
        len(json_files),
    )
    return tasks


class ArticleScraper:
    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[str] = None,
        no_proxy: bool = False,
    ):
        self.headless = headless
        if no_proxy:
            self.proxy = None
        else:
            self.proxy = proxy if proxy is not None else DEFAULT_PROXY

    def _parse_proxy(self) -> Optional[dict]:
        """
        Parse a proxy URL like http://user:pass@host:port into a Playwright-compatible
        dict with credentials split out.
        """
        if not self.proxy:
            return None

        parsed = urlparse(self.proxy)
        proxy_dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
        if parsed.username:
            proxy_dict["username"] = parsed.username
        if parsed.password:
            proxy_dict["password"] = parsed.password

        logger.info(
            f"Proxy configured: {parsed.scheme}://{parsed.hostname}:{parsed.port} "
            f"(authenticated: {'yes' if parsed.username else 'no'})"
        )
        return proxy_dict

    def fetch_articles(self, input_csv: str, manifest_name: str):
        csv_path = RESULTS_DIR / input_csv
        if not csv_path.exists():
            logger.error(
                f"Input CSV not found at {csv_path}. Did you run parse_engine.py first?"
            )
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tasks = list(reader)

        downloaded_manifest = []

        # Parse the proxy once
        proxy_dict = self._parse_proxy()

        with sync_playwright() as p:
            launch_opts = {"headless": self.headless}

            # Pass just the server URL at the launch level (no credentials)
            if proxy_dict:
                launch_opts["proxy"] = {"server": proxy_dict["server"]}

            browser = launch_chromium(p, launch_opts)

            for i, task in enumerate(tasks):
                url = task["link"]
                logger.info(f"[{i + 1}/{len(tasks)}] Fetching article: {url[:60]}...")

                for attempt in range(1, MAX_RETRIES + 1):
                    context = None
                    try:
                        # Pass the full proxy dict (with credentials) at the context level
                        context_opts = {
                            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            "ignore_https_errors": True,
                        }
                        if proxy_dict:
                            context_opts["proxy"] = proxy_dict

                        context = browser.new_context(**context_opts)
                        Stealth().use_sync(context)
                        page = context.new_page()

                        page.route(
                            "**/*",
                            lambda route: (
                                route.abort()
                                if route.request.resource_type
                                in ["image", "media", "font"]
                                else route.continue_()
                            ),
                        )

                        page.goto(
                            url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS
                        )
                        time.sleep(random.uniform(2.0, 5.0))

                        html_content = page.content()
                        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
                        file_name = f"article_{url_hash}.html"
                        file_path = RAW_ARTICLES_DIR / file_name

                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(html_content)

                        downloaded_manifest.append(
                            {
                                "original_title": task["title"],
                                "url": url,
                                "source": task["source"],
                                "query": task["query"],
                                "local_file": str(file_path),
                            }
                        )

                        logger.info(f"✓ Saved HTML to {file_name}")
                        break

                    except PlaywrightError as e:
                        logger.warning(f"Failed (Attempt {attempt}/{MAX_RETRIES}): {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(random.uniform(5.0, 10.0))
                    finally:
                        if context:
                            context.close()

                time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))

            browser.close()

        manifest_path = RAW_ARTICLES_DIR / manifest_name
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(downloaded_manifest, f, indent=2)
        logger.info(f"Article scraping complete! Manifest saved to {manifest_path}")

    @staticmethod
    def _manifest_row_parsed(task: dict[str, Any], local_file: Optional[str]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "url": task["url"],
            "original_title": task["title"],
            "source": task["source"],
            "query": task["query"],
            "taskQuery": task.get("taskQuery") or "",
            "searchDate": task.get("searchDate") or "",
            "source_file": task.get("source_file") or "",
            "parsed_json": task.get("parsed_json") or "",
            "local_file": local_file,
        }
        return row

    def fetch_from_parsed(
        self,
        parsed_dir: Path,
        manifest_name: str,
        *,
        force: bool = False,
        limit: int = 0,
    ) -> dict[str, int]:
        """Scrape each unique external article URL into ``article_<md5(url)>.html``.

        Skips URLs whose HTML file already exists (unless *force*). Writes a manifest
        with ``searchDate`` (per row and from JSON metadata) for downstream sorting.
        """
        tasks = load_tasks_from_parsed(parsed_dir)
        if limit > 0:
            tasks = tasks[:limit]
            logger.info("Applied --limit=%d → %d task(s)", limit, len(tasks))

        if not tasks:
            logger.error("No scrapeable tasks from %s", parsed_dir)
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        manifest: list[dict[str, Any]] = []
        downloaded = 0
        skipped = 0
        failed = 0

        proxy_dict = self._parse_proxy()

        with sync_playwright() as p:
            launch_opts: dict[str, Any] = {"headless": self.headless}
            if proxy_dict:
                launch_opts["proxy"] = {"server": proxy_dict["server"]}

            browser = launch_chromium(p, launch_opts)

            for i, task in enumerate(tasks):
                url = task["url"]
                file_name = f"article_{task['url_hash']}.html"
                file_path = RAW_ARTICLES_DIR / file_name

                if not force and file_path.exists():
                    logger.info(
                        f"[{i + 1}/{len(tasks)}] Skip (exists): {file_name} ← {url[:70]}..."
                    )
                    skipped += 1
                    manifest.append(self._manifest_row_parsed(task, file_name))
                    continue

                logger.info(f"[{i + 1}/{len(tasks)}] Fetching article: {url[:70]}...")

                ok = False
                for attempt in range(1, MAX_RETRIES + 1):
                    context = None
                    try:
                        context_opts = {
                            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            "ignore_https_errors": True,
                        }
                        if proxy_dict:
                            context_opts["proxy"] = proxy_dict

                        context = browser.new_context(**context_opts)
                        Stealth().use_sync(context)
                        page = context.new_page()

                        page.route(
                            "**/*",
                            lambda route: (
                                route.abort()
                                if route.request.resource_type
                                in ["image", "media", "font"]
                                else route.continue_()
                            ),
                        )

                        page.goto(
                            url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS
                        )
                        time.sleep(random.uniform(2.0, 5.0))

                        html_content = page.content()
                        file_path.write_text(html_content, encoding="utf-8")

                        logger.info(f"✓ Saved HTML to {file_name}")
                        ok = True
                        downloaded += 1
                        manifest.append(self._manifest_row_parsed(task, file_name))
                        break

                    except PlaywrightError as e:
                        logger.warning(f"Failed (Attempt {attempt}/{MAX_RETRIES}): {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(random.uniform(5.0, 10.0))
                    finally:
                        if context:
                            context.close()

                if not ok:
                    failed += 1
                    manifest.append(self._manifest_row_parsed(task, None))

                time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))

            browser.close()

        manifest_path = RAW_ARTICLES_DIR / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Parsed→raw complete. Manifest: %s (entries=%d)  downloaded=%d skipped=%d failed=%d",
            manifest_path,
            len(manifest),
            downloaded,
            skipped,
            failed,
        )
        return {
            "total": len(tasks),
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch full article HTML (CSV search_results or parsed_articles JSON)."
    )
    parser.add_argument(
        "--from-parsed",
        action="store_true",
        help="Read URLs from parsed_articles/*.json instead of search_results CSV",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=PARSED_ARTICLES_DIR,
        help="Directory of parsed JSON (default: parsed_articles/)",
    )
    parser.add_argument("--input", "-i", type=str, default="final_data.csv")
    parser.add_argument("--manifest", "-m", type=str, default="article_manifest.json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even when the target HTML file already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max articles to process in --from-parsed mode (0 = all)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Do not use DEFAULT_PROXY / residential proxy",
    )
    parser.add_argument(
        "--install-browsers",
        action="store_true",
        help="Run `python -m playwright install chromium` for this interpreter, then continue",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.install_browsers:
        logger.info(
            "Installing Playwright Chromium for %s …",
            sys.executable,
        )
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
        )
    scraper = ArticleScraper(no_proxy=args.no_proxy)
    if args.from_parsed:
        scraper.fetch_from_parsed(
            args.parsed_dir.resolve(),
            args.manifest,
            force=args.force,
            limit=args.limit,
        )
    else:
        scraper.fetch_articles(args.input, args.manifest)
