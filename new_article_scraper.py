"""
new_article_scraper.py – Scrape full article pages whose URLs come from
parsed Google-News search-result JSON files.

Workflow
--------
1. Load every ``*.json`` file in the parsed-articles directory.
2. Collect all article URLs, de-duplicate, and optionally filter by domain.
3. For each URL, open a stealth Playwright browser, fetch the page HTML,
   and save it to the output directory.
4. Write a ``manifest.json`` beside the saved HTML files so downstream
   code can trace each file back to its origin.

Resumable: articles whose HTML file already exists on disk are skipped
unless ``--force`` is passed.

Usage
-----
    # Defaults (reads parsed_articles/, writes to raw_articles/)
    python new_article_scraper.py

    # Custom dirs, verbose, headful browser
    python new_article_scraper.py -i parsed/ -o scraped/ -v --no-headless

    # Quiet mode, force re-download, limit to 20 articles
    python new_article_scraper.py -q --force --limit 20
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, Error as PlaywrightError
from playwright_stealth.stealth import Stealth

from config import (
    BASE_DIR,
    MIN_DELAY_SEC,
    MAX_DELAY_SEC,
    MAX_RETRIES,
    PAGE_TIMEOUT_MS,
    DEFAULT_PROXY,
)

logger = logging.getLogger(__name__)

PARSED_ARTICLES_DIR = BASE_DIR / "parsed_articles"
RAW_ARTICLES_DIR = BASE_DIR / "raw_articles"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(parsed_dir: Path):
    """Read all parsed JSON files and return a de-duplicated list of scrape tasks.

    Each task is a dict with keys ``url``, ``title``, ``source``, ``query``,
    ``source_file``, and ``url_hash`` (MD5 of the URL, used as filename).

    Articles without a valid HTTP(S) URL or whose URL points to a Google
    domain (tracking redirects, cached pages) are silently dropped.

    Parameters
    ----------
    parsed_dir : Path
        Directory containing ``*.json`` files produced by ``parser.py``.

    Returns
    -------
    list[dict]
    """
    seen_urls: set[str] = set()
    tasks: list[dict] = []

    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", parsed_dir)
        return tasks

    logger.info("Loading tasks from %d parsed file(s) in %s", len(json_files), parsed_dir)

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable file: %s", jf.name)
            continue

        for article in data.get("articles", []):
            url = (article.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue

            # Skip Google-internal URLs (redirects, caches, etc.)
            domain = urlparse(url).hostname or ""
            if domain.endswith("google.com") or domain.endswith("gstatic.com"):
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            tasks.append({
                "url": url,
                "title": article.get("title") or "",
                "source": article.get("source") or "",
                "query": article.get("query") or "",
                "taskQuery": article.get("taskQuery") or "",
                "searchDate": article.get("searchDate") or "",
                "source_file": article.get("sourceFile") or jf.name,
                "url_hash": hashlib.md5(url.encode("utf-8")).hexdigest(),
            })

    logger.info("Collected %d unique article URL(s) from %d file(s)", len(tasks), len(json_files))
    return tasks


# ---------------------------------------------------------------------------
# Proxy helper
# ---------------------------------------------------------------------------

def parse_proxy(proxy_url: Optional[str]) -> Optional[dict]:
    """Convert ``http://user:pass@host:port`` into a Playwright proxy dict.

    Returns ``None`` when *proxy_url* is falsy.
    """
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    proxy_dict: dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        proxy_dict["username"] = parsed.username
    if parsed.password:
        proxy_dict["password"] = parsed.password

    logger.info(
        "Proxy configured: %s://%s:%s (auth=%s)",
        parsed.scheme, parsed.hostname, parsed.port,
        "yes" if parsed.username else "no",
    )
    return proxy_dict


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class ArticleScraper:
    """Fetch full article HTML for a list of tasks using Playwright.

    Parameters
    ----------
    output_dir : Path
        Where to save ``article_<hash>.html`` files.
    headless : bool
        Run the browser without a visible window.
    proxy : str | None
        Optional proxy URL.
    force : bool
        Re-download even if the HTML file already exists.
    """

    def __init__(
        self,
        output_dir: Path,
        headless: bool = True,
        proxy: Optional[str] = None,
        force: bool = False,
        no_proxy: bool = False,
    ):
        self.output_dir = output_dir
        self.headless = headless
        self.proxy_url = None if no_proxy else (proxy or DEFAULT_PROXY)
        self.force = force
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- internal helpers ----

    def _block_heavy_resources(self, route):
        """Abort requests for images / media / fonts to speed up scraping."""
        if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
            route.abort()
        else:
            route.continue_()

    def _fetch_one(self, browser, proxy_dict: Optional[dict], task: dict) -> Optional[str]:
        """Attempt to download a single article, retrying on failure.

        Returns the saved file path relative to output_dir on success,
        or ``None`` after all retries are exhausted.
        """
        url = task["url"]
        file_name = f"article_{task['url_hash']}.html"
        date_subdir = task.get("searchDate") or "unknown"
        article_dir = self.output_dir / date_subdir
        article_dir.mkdir(parents=True, exist_ok=True)
        file_path = article_dir / file_name

        if not self.force and file_path.exists():
            logger.debug("Already on disk, skipping: %s", file_name)
            return file_name

        for attempt in range(1, MAX_RETRIES + 1):
            context = None
            try:
                context_opts: dict = {
                    "user_agent": USER_AGENT,
                    "ignore_https_errors": True,
                }
                if proxy_dict:
                    context_opts["proxy"] = proxy_dict

                context = browser.new_context(**context_opts)
                Stealth().use_sync(context)
                page = context.new_page()
                page.route("**/*", self._block_heavy_resources)

                page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

                # Mimic human reading time
                time.sleep(random.uniform(2.0, 5.0))

                html_content = page.content()
                file_path.write_text(html_content, encoding="utf-8")

                relative = f"{date_subdir}/{file_name}"
                logger.info("Saved %s", relative)
                return relative

            except PlaywrightError as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, MAX_RETRIES, url[:80], exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(random.uniform(5.0, 10.0))
            except Exception as exc:
                logger.error("Unexpected error for %s: %s", url[:80], exc)
                break
            finally:
                if context:
                    context.close()

        return None

    # ---- public API ----

    def run(self, tasks: list[dict]):
        """Scrape every task and write a manifest to *output_dir*.

        Parameters
        ----------
        tasks : list[dict]
            Output of :func:`load_tasks`.

        Returns
        -------
        dict
            Summary with keys ``total``, ``downloaded``, ``skipped``, ``failed``.
        """
        if not tasks:
            logger.warning("Nothing to scrape — task list is empty.")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        proxy_dict = parse_proxy(self.proxy_url)
        manifest: list[dict] = []
        downloaded = 0
        skipped = 0
        failed = 0

        with sync_playwright() as pw:
            launch_opts: dict = {"headless": self.headless}
            if proxy_dict:
                launch_opts["proxy"] = {"server": proxy_dict["server"]}

            browser = pw.chromium.launch(**launch_opts)

            for idx, task in enumerate(tasks, start=1):
                url = task["url"]
                file_name = f"article_{task['url_hash']}.html"
                date_subdir = task.get("searchDate") or "unknown"
                file_path = self.output_dir / date_subdir / file_name

                if not self.force and file_path.exists():
                    logger.debug("[%d/%d] Skipping (exists): %s", idx, len(tasks), url[:80])
                    skipped += 1
                    manifest.append(self._manifest_entry(task, f"{date_subdir}/{file_name}"))
                    continue

                logger.info("[%d/%d] Fetching: %s", idx, len(tasks), url[:80])
                saved = self._fetch_one(browser, proxy_dict, task)

                if saved:
                    downloaded += 1
                    manifest.append(self._manifest_entry(task, saved))
                else:
                    failed += 1

                # Polite delay between requests
                time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))

            browser.close()

        # Persist manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Manifest written to %s (%d entries)", manifest_path, len(manifest))

        summary = {
            "total": len(tasks),
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
        }
        logger.info(
            "Done. total=%d  downloaded=%d  skipped=%d  failed=%d",
            summary["total"], summary["downloaded"],
            summary["skipped"], summary["failed"],
        )
        return summary

    @staticmethod
    def _manifest_entry(task: dict, local_file: str) -> dict:
        return {
            "url": task["url"],
            "original_title": task["title"],
            "source": task["source"],
            "query": task["query"],
            "taskQuery": task.get("taskQuery", ""),
            "searchDate": task.get("searchDate", ""),
            "source_file": task["source_file"],
            "local_file": local_file,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape full article pages from parsed search-result JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python new_article_scraper.py\n"
            "  python new_article_scraper.py -v --limit 10\n"
            "  python new_article_scraper.py -q --force --no-headless\n"
        ),
    )

    p.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=PARSED_ARTICLES_DIR,
        help="Directory of parsed JSON files (default: parsed_articles/)",
    )
    p.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=RAW_ARTICLES_DIR,
        help="Directory to save scraped HTML (default: raw_articles/)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max articles to scrape (0 = unlimited, default: 0)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download articles even if they already exist on disk",
    )
    p.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Show the browser window (useful for debugging)",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Override proxy URL (default: use DEFAULT_PROXY from .env)",
    )
    p.add_argument(
        "--no-proxy",
        action="store_true",
        default=False,
        help="Disable proxy entirely (connect directly)",
    )

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="DEBUG-level logging",
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="WARNING-level logging only",
    )

    return p


def configure_logging(verbose: bool = False, quiet: bool = False):
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    tasks = load_tasks(args.input_dir.resolve())
    if not tasks:
        logger.error("No scrapeable URLs found. Exiting.")
        raise SystemExit(1)

    if args.limit > 0:
        tasks = tasks[: args.limit]
        logger.info("Limited to %d task(s) by --limit", len(tasks))

    proxy_value = None  # no proxy at all
    if not args.no_proxy:
        proxy_value = args.proxy  # None means "use DEFAULT_PROXY from config"

    scraper = ArticleScraper(
        output_dir=args.output_dir.resolve(),
        headless=not args.no_headless,
        proxy=proxy_value,
        force=args.force,
        no_proxy=args.no_proxy,
    )
    scraper.run(tasks)


if __name__ == "__main__":
    main()
