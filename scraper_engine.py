#!/usr/bin/env python3
import json
import time
import random
import hashlib
import argparse
import math
import requests
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Playwright is used for driving the browser, Stealth masks automated flags.
from playwright.sync_api import sync_playwright, Page, Error as PlaywrightError
from playwright_stealth import Stealth

# Official GoLogin SDK for programmatic headless control
from gologin import GoLogin

# Import environment variables and constants from your config.py
from config import (
    logger,
    RAW_HTML_DIR,
    MIN_DELAY_SEC,
    MAX_DELAY_SEC,
    SCROLL_MIN,
    SCROLL_MAX,
    MAX_RETRIES,
    PAGE_TIMEOUT_MS,
    CAPTCHA_API_KEY,
    DEFAULT_PROXY,
    GOLOGIN_API_TOKEN,
)


# ==============================================================================
# CLASS: HumanBehaviorMimic
# ==============================================================================
class HumanBehaviorMimic:
    @staticmethod
    def delay(min_s: float = MIN_DELAY_SEC, max_s: float = MAX_DELAY_SEC):
        """Pauses execution using a Gaussian (Normal) distribution."""
        mean = (min_s + max_s) / 2
        std_dev = (max_s - min_s) / 4
        sleep_time = abs(random.gauss(mean, std_dev))
        time.sleep(max(min_s, min(sleep_time, max_s)))

    @staticmethod
    def scroll(page: Page):
        """Simulates how a human erratically scrolls down a search results page."""
        for _ in range(random.randint(3, 5)):
            page.mouse.wheel(0, random.randint(SCROLL_MIN, SCROLL_MAX))
            HumanBehaviorMimic.delay(0.5, 1.5)
            if random.random() > 0.7:
                page.mouse.wheel(0, -150)
                HumanBehaviorMimic.delay(0.2, 0.6)


# ==============================================================================
# CLASS: CaptchaManager
# ==============================================================================
class CaptchaManager:
    @staticmethod
    def solve_recaptcha(page: Page, url: str) -> bool:
        """Extracts the site key, sends to CapSolver, waits for token, and injects it."""
        if not CAPTCHA_API_KEY:
            logger.error("CAPTCHA detected, but no API key found in .env!")
            return False

        try:
            site_key_element = page.query_selector(".g-recaptcha")
            if not site_key_element:
                logger.error("Could not locate CAPTCHA sitekey on the page.")
                return False

            site_key = site_key_element.get_attribute("data-sitekey")
            logger.info(
                f"Submitting CAPTCHA task to CapSolver (SiteKey: {site_key[:10]}...)"
            )

            create_task_payload = {
                "clientKey": CAPTCHA_API_KEY,
                "task": {
                    "type": "ReCaptchaV2TaskProxyless",
                    "websiteURL": url,  # Dynamically handles the exact sorry/index URL
                    "websiteKey": site_key,
                },
            }
            res = requests.post(
                "https://api.capsolver.com/createTask", json=create_task_payload
            ).json()

            if res.get("errorId") != 0:
                logger.error(f"Solver API Error: {res.get('errorDescription')}")
                return False

            task_id = res.get("taskId")
            logger.info("Task created. Waiting for AI to solve...")

            token = None
            for _ in range(24):
                time.sleep(5)
                poll_payload = {"clientKey": CAPTCHA_API_KEY, "taskId": task_id}
                poll_res = requests.post(
                    "https://api.capsolver.com/getTaskResult", json=poll_payload
                ).json()

                if poll_res.get("status") == "ready":
                    token = poll_res.get("solution", {}).get("gRecaptchaResponse")
                    break
                elif poll_res.get("status") == "failed":
                    logger.error("Solver failed to solve the CAPTCHA.")
                    return False

            if not token:
                logger.error("Solver timed out.")
                return False

            logger.info("CAPTCHA solved! Injecting token...")

            # Safely inject value and wait for DOM registration
            page.evaluate(
                f"""
                (token) => {{
                    const el = document.getElementById('g-recaptcha-response');
                    if (el) {{
                        el.value = token;
                        el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                }}
                """,
                token,
            )
            time.sleep(2)

            try:
                # Try clicking the submit button first (more human-like)
                submit_button = page.locator(
                    "#captcha-form input[type='submit'], #captcha-form button[type='submit'], input[type='submit'], #submit"
                ).first
                if submit_button.is_visible():
                    submit_button.click()
                else:
                    page.evaluate("document.forms[0].submit();")
            except Exception:
                page.evaluate("document.forms[0].submit();")

            logger.info("Successfully submitted CAPTCHA token.")
            return True

        except Exception as e:
            logger.error(f"Error during CAPTCHA solving process: {e}")
            return False


# ==============================================================================
# CLASS: ScraperEngine
# ==============================================================================
class ScraperEngine:
    def __init__(self, headless: bool = True, proxy: Optional[str] = None):
        self.headless = headless
        self.proxy = proxy or DEFAULT_PROXY

    def fetch_pages_gologin(
        self,
        input_file: Path,
        manifest_file: str,
        base_profile_id: str,
        total_nodes: int = 1,
        node_index: int = 0,
        mean_tasks_per_session: int = 5,
        std_dev_tasks: int = 2,
        output_dir: Optional[Path] = None,
    ):
        target_output_dir = output_dir or RAW_HTML_DIR
        target_output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = target_output_dir.parent / "cache"
        cache_dir.mkdir(exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)

        if total_nodes > 1:
            chunk_size = math.ceil(len(all_tasks) / total_nodes)
            start_idx = node_index * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_tasks))
            tasks = all_tasks[start_idx:end_idx]
            logger.info(
                f"Node {node_index} initialized. Processing {len(tasks)} tasks."
            )
            manifest_path = (
                target_output_dir
                / f"{manifest_file.replace('.json', '')}_node_{node_index}.json"
            )
        else:
            tasks = all_tasks
            logger.info(f"Running single node. Processing all {len(tasks)} tasks.")
            manifest_path = target_output_dir / manifest_file

        seen_hashes = set()
        for cache_file in cache_dir.glob("scraped_hashes_*.txt"):
            with open(cache_file, "r", encoding="utf-8") as f:
                seen_hashes.update(line.strip() for line in f if line.strip())

        logger.info(
            f"Node {node_index} loaded {len(seen_hashes)} previously scraped URLs."
        )

        downloaded_manifest = []
        task_count = 0
        current_batch_limit = 0
        gl_sdk = None

        for i, task in enumerate(tasks):
            url_hash = hashlib.md5(task["url"].encode("utf-8")).hexdigest()
            if url_hash in seen_hashes:
                logger.info(f"[{i + 1}/{len(tasks)}] Skipping cached URL.")
                continue

            logger.info(f"[{i + 1}/{len(tasks)}] Starting task: '{task['query']}'")

            if task_count >= current_batch_limit:
                current_batch_limit = max(
                    1, int(random.gauss(mean_tasks_per_session, std_dev_tasks))
                )

                if gl_sdk:
                    logger.info(
                        "Batch limit reached. Restarting GoLogin profile to flush memory..."
                    )
                    logger.info(
                        f"-> Next session will handle {current_batch_limit} randomized tasks."
                    )
                    try:
                        gl_sdk.stop()
                    except Exception:
                        pass  # Ignore FileNotFoundError if proxy crashed
                    time.sleep(5)

                max_boot_retries = 3
                raw_endpoint = None

                for attempt in range(max_boot_retries):
                    try:
                        logger.info(
                            f"Booting profile {base_profile_id} (Attempt {attempt + 1}/{max_boot_retries})..."
                        )
                        gl_sdk = GoLogin(
                            {
                                "token": GOLOGIN_API_TOKEN,
                                "profile_id": base_profile_id,
                                "extra_params": ["--headless", "--disable-gpu"],
                            }
                        )

                        raw_endpoint = gl_sdk.start()
                        if raw_endpoint:
                            break

                    except Exception as e:
                        logger.error(f"GoLogin proxy boot failed: {e}")
                        if gl_sdk:
                            try:
                                gl_sdk.stop()
                            except Exception:
                                pass  # Ignore FileNotFoundError during failed boots

                        if attempt < max_boot_retries - 1:
                            logger.info(
                                "Waiting 10 seconds before retrying proxy connection..."
                            )
                            time.sleep(10)

                if raw_endpoint and not raw_endpoint.startswith("http"):
                    ws_endpoint = f"http://{raw_endpoint}"
                else:
                    ws_endpoint = raw_endpoint

                task_count = 0

            if not ws_endpoint:
                logger.error("FATAL: Could not get a debugger URL for task.")
                break

            with sync_playwright() as p:
                try:
                    browser = p.chromium.connect_over_cdp(ws_endpoint)
                    context = browser.contexts[0]

                    context.clear_cookies()

                    target_page = None
                    for p_obj in context.pages:
                        if not p_obj.url.startswith("chrome-extension://"):
                            target_page = p_obj
                            break
                    if not target_page:
                        target_page = context.new_page()
                    page = target_page
                    page.bring_to_front()

                    try:
                        page.goto("about:blank")
                        page.evaluate(
                            "window.localStorage.clear(); window.sessionStorage.clear();"
                        )
                    except Exception:
                        pass

                    page.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                    """)

                    logger.info(f"Navigating to task URL: {task['url']}")
                    page.goto(task["url"], wait_until="domcontentloaded")

                    try:
                        page.wait_for_selector(
                            "h3, .g-recaptcha, #captcha-form",
                            state="visible",
                            timeout=PAGE_TIMEOUT_MS,
                        )
                    except Exception as e:
                        logger.warning(f"Timeout waiting for Google DOM to update: {e}")

                    # Check for CAPTCHA
                    if (
                        "sorry/index" in page.url
                        or "Just a moment" in page.title()
                        or page.query_selector(".g-recaptcha")
                    ):
                        logger.warning("Hit a CAPTCHA wall.")

                        # Pass the specific sorry/index URL to CapSolver
                        success = CaptchaManager.solve_recaptcha(page, page.url)
                        if not success:
                            raise PlaywrightError("Failed to bypass CAPTCHA.")

                        logger.info(
                            "CAPTCHA bypassed. Waiting for results page stability..."
                        )
                        try:
                            # Wait for any h3 (result title) or the search results container
                            page.wait_for_selector(
                                "h3, #res, #search", state="visible", timeout=30000
                            )
                            page.wait_for_load_state("networkidle", timeout=30000)
                            logger.info("✓ Results page stabilized!")
                        except Exception as e:
                            logger.error(
                                f"FATAL: CAPTCHA solution integrated but results page failed to load: {e}"
                            )
                            task_count += 1
                            raise PlaywrightError(
                                "Bricked by post-captcha wall stability issue."
                            )

                    HumanBehaviorMimic.delay(3.0, 9.0)
                    HumanBehaviorMimic.scroll(page)

                    html_content = page.content()
                    file_name = f"{url_hash}.html"
                    file_path = target_output_dir / file_name

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    downloaded_manifest.append(
                        {
                            "query": task["query"],
                            "url": task["url"],
                            "local_file": str(file_path),
                        }
                    )
                    logger.info(f"✓ Saved results to {file_name}")

                    hash_file_path = cache_dir / f"scraped_hashes_node_{node_index}.txt"
                    with open(hash_file_path, "a", encoding="utf-8") as f:
                        f.write(url_hash + "\n")

                    seen_hashes.add(url_hash)

                except Exception as e:
                    logger.warning(f"Failed task: {e}")
                    # If the error is related to the browser connection dying,
                    # force an immediate GoLogin reboot on the next loop iteration
                    error_str = str(e).lower()
                    if (
                        "target closed" in error_str
                        or "browser has been closed" in error_str
                        or "connect_over_cdp" in error_str
                    ):
                        logger.error(
                            "Browser crashed mid-session! Forcing early GoLogin restart..."
                        )
                        task_count = current_batch_limit  # This triggers the restart logic on the next loop

                finally:
                    try:
                        page.goto("about:blank")
                        HumanBehaviorMimic.delay(1.0, 3.0)
                        browser.close()
                    except Exception:
                        pass

                task_count += 1

        if gl_sdk:
            logger.info("Stopping final GoLogin profile...")
            try:
                gl_sdk.stop()
            except Exception:
                pass

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(downloaded_manifest, f, indent=2)

        logger.info("Scraping complete!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input tasks.json file"
    )
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        default="download_manifest.json",
        help="Output manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save scraped HTML files (Defaults to RAW_HTML_DIR in config.py)",
    )
    parser.add_argument(
        "--proxy", "-p", type=str, default=None, help="Proxy URL (Overrides .env)"
    )

    parser.add_argument(
        "--total-nodes",
        "-tn",
        type=int,
        default=1,
        help="Total number of cluster nodes running",
    )
    parser.add_argument(
        "--node-index",
        "-ni",
        type=int,
        default=0,
        help="The 0-based index of this specific process",
    )

    parser.add_argument(
        "--gologin",
        "-g",
        type=str,
        required=False,
        default="",
        help="Profile ID of the running GoLogin Anti-Detect profile",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = ScraperEngine(proxy=args.proxy)

    if args.gologin:
        engine.fetch_pages_gologin(
            args.input,
            args.manifest,
            args.gologin,
            args.total_nodes,
            args.node_index,
            output_dir=args.output_dir,
        )
