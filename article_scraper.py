import csv
import time
import random
import hashlib
import argparse
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse  # <-- Added missing import

from playwright.sync_api import sync_playwright, Error as PlaywrightError
from playwright_stealth import stealth

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


class ArticleScraper:
    def __init__(self, headless: bool = True, proxy: Optional[str] = None):
        self.headless = headless
        self.proxy = proxy or DEFAULT_PROXY

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

            browser = p.chromium.launch(**launch_opts)

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
                        page = context.new_page()

                        stealth(page)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="final_data.csv")
    parser.add_argument("--manifest", "-m", type=str, default="article_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scraper = ArticleScraper()
    scraper.fetch_articles(args.input, args.manifest)
