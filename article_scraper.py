"""
Article HTML fetcher driven by Playwright (synchronous API).

This module implements a **two-mode ingestion pipeline** that downloads the full
DOM HTML of news/article pages and persists it under ``raw_articles/`` using a
**deterministic filename** derived from the article URL. Downstream tooling
(e.g. ``ml_model/data/html_extractor.py``) expects that naming convention:
``article_<md5_hex(url)>.html``.

**High-level architecture**

1. **Configuration** — Runtime knobs (delays, retries, timeouts, default proxy) are
   imported from ``config`` so operational policy stays centralized. This script
   focuses on browser orchestration and I/O.

2. **Browser acquisition** — ``launch_chromium`` prefers the user’s installed
   Google Chrome (Playwright ``channel="chrome"``) so macOS users often avoid
   ``playwright install`` when Chrome exists in ``/Applications``. If that fails,
   it falls back to Playwright-managed Chromium. Missing bundles produce an
   actionable log message referencing ``sys.executable``.

3. **Scraping modes**

   * **CSV mode** (default): ``ArticleScraper.fetch_articles`` reads a
     ``search_results/<input>.csv`` file produced earlier in the project (e.g. by
     a search/parse step). Each row must expose at least ``link``, ``title``,
     ``source``, and ``query`` columns. Every row is attempted unless the CSV
     is missing.

   * **Parsed-JSON mode** (``--from-parsed``): ``load_tasks_from_parsed`` walks
     ``parsed_articles/*.json`` (or ``--parsed-dir``), extracts ``articles[]``
     entries, **de-duplicates by URL across all JSON files**, filters obvious
     non-target hosts (Google redirect/cache domains), and builds in-memory task
     dicts enriched with provenance (which JSON file, ``searchDate``, etc.).
     ``ArticleScraper.fetch_from_parsed`` then scrapes those URLs, supports
     **skip-if-file-exists** idempotency, optional ``--limit``, and ``--force``
     re-download.

4. **Per-request browser hygiene** — For each URL (or retry attempt), a **fresh**
   ``BrowserContext`` is created and closed in a ``finally`` block. Rationale:
   isolates cookies/storage between sites, reduces cross-origin leakage, and
   ensures stealth patches apply per context. A **new page** navigates once per
   successful attempt.

5. **Network shaping** — Optional HTTP(S) proxy: server URL is passed at **browser
   launch**; full dict including credentials is passed at **context** creation
   (Playwright’s documented split). ``page.route("**/*", ...)`` aborts heavy
   resource types (``image``, ``media``, ``font``) to save bandwidth and speed
   ``domcontentloaded`` while keeping HTML/JS/CSS needed for many article shells.

6. **Anti-automation signals** — ``playwright_stealth``’s ``Stealth`` class is
   applied synchronously to each context before page creation. Combined with a
   fixed desktop Chrome user-agent string and randomized post-navigation sleeps,
   this reduces trivial bot fingerprinting (not a guarantee against hardened
   anti-bot stacks).

7. **Resilience** — ``MAX_RETRIES`` wraps transient Playwright failures; backoff
   sleeps are randomized. After each task (success or exhausted retries), an
   inter-task delay ``random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC)`` throttles
   crawl rate.

8. **Artifacts** — Raw HTML files under ``RAW_ARTICLES_DIR``; a **JSON manifest**
   listing metadata and local paths (or ``null`` ``local_file`` on hard failure
   in parsed mode) is written beside them.

**CLI entrypoint** — ``parse_args`` defines flags; ``__main__`` optionally runs
``python -m playwright install chromium`` when ``--install-browsers`` is set,
then instantiates ``ArticleScraper`` and dispatches to the appropriate fetch
method.

**Threading / async** — Entirely synchronous Playwright usage; no asyncio event
loop in this file.
"""

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

# ---------------------------------------------------------------------------
# Path constants (resolved relative to project BASE_DIR from config)
# ---------------------------------------------------------------------------
# RAW_ARTICLES_DIR: persisted HTML + manifest JSON live here; created eagerly so
# later code can assume the directory exists without repeated mkdir checks.
RAW_ARTICLES_DIR = BASE_DIR / "raw_articles"
RAW_ARTICLES_DIR.mkdir(exist_ok=True)

# RESULTS_DIR: default location for CSV input in legacy/search-results workflow.
RESULTS_DIR = BASE_DIR / "search_results"

# PARSED_ARTICLES_DIR: default root for parser-produced *.json when using
# --from-parsed without an explicit --parsed-dir.
PARSED_ARTICLES_DIR = BASE_DIR / "parsed_articles"


def launch_chromium(p, launch_opts: dict[str, Any]):
    """Attach to a Chromium-class browser using Playwright’s sync API.

    **Strategy (ordered attempts)**

    1. Launch with ``channel="chrome"`` merged into *launch_opts*. On macOS this
       typically binds to the user-installed Google Chrome under
       ``/Applications/Google Chrome.app``. Benefits: fewer “missing executable”
       errors for developers who never ran ``playwright install``, and behavior
       closer to a real user Chrome.

    2. If that raises ``PlaywrightError`` (Chrome absent, policy block, etc.),
       log at DEBUG and retry **without** ``channel``, i.e. Playwright’s bundled
       Chromium referenced by the current Playwright version.

    3. If bundled Chromium is not installed, the error string usually contains
       ``Executable doesn't exist``. In that specific case we emit a structured
       ERROR log telling the operator to run
       ``<sys.executable> -m playwright install chromium`` **or** install Chrome
       and rely on attempt (1). The original exception is then re-raised so the
       caller’s context manager can unwind cleanly.

    **Proxy note**

    *launch_opts* may already contain ``proxy={"server": "http://host:port"}`` at
    the browser level (credentials belong on the context, not here—see
    ``ArticleScraper._parse_proxy`` and call sites).

    Args:
        p: The ``sync_playwright()`` manager’s ``Playwright`` instance (exposes
            ``p.chromium``).
        launch_opts: Keyword arguments destined for ``chromium.launch``, except
            we temporarily inject ``channel="chrome"`` for the first try.

    Returns:
        A ``playwright.sync_api.Browser`` instance ready for ``new_context``.

    Raises:
        PlaywrightError: On unrecoverable launch failure, including missing
            Chromium after the error-message branch re-raises.

    Side effects:
        Logging at DEBUG or ERROR depending on the failure path.
    """
    # Shallow copy so we never mutate the caller’s dict when adding channel.
    chrome_opts = {**launch_opts, "channel": "chrome"}
    try:
        return p.chromium.launch(**chrome_opts)
    except PlaywrightError as e:
        # Expected in CI/Linux without Chrome channel — not necessarily fatal.
        logger.debug("Launch with channel=chrome failed (%s); trying bundled Chromium.", e)

    try:
        return p.chromium.launch(**launch_opts)
    except PlaywrightError as e:
        err = str(e)
        # Distinguish “no binary” from other Playwright errors (permissions, bad flags).
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
    """Flatten parser JSON files into a de-duplicated list of scrape tasks.

    **Input schema (per JSON file)**

    Each file is decoded as a dict. The function reads:

    * ``data["metadata"]["searchDate"]`` — optional fallback date string applied
      to articles missing their own ``searchDate``.
    * ``data["articles"]`` — iterable of dicts; each dict is expected to carry
      at least ``url``; optional fields include ``title``, ``source``, ``query``,
      ``taskQuery``, ``searchDate``, ``sourceFile``.

    **Filtering rules (in order)**

    1. Skip rows whose ``url`` is empty/whitespace or does not start with
       ``http`` — filters ``javascript:``, ``mailto:``, and blank SERP slots.
    2. Parse hostname via ``urlparse``; skip ``*.google.com`` and ``*.gstatic.com``
       to avoid pulling Google intermediates instead of publisher pages.
    3. Skip URLs already seen **globally** across files — first occurrence wins,
       preserving stable ordering because ``json_files`` is sorted by name.

    **Output task shape**

    Each task dict includes:

    * ``url`` — canonical string used for HTTP fetch and hashing.
    * ``title``, ``source``, ``query``, ``taskQuery`` — propagated for manifest
      traceability.
    * ``searchDate`` — per-article or metadata fallback or ``""``.
    * ``source_file`` — ``sourceFile`` from JSON if present else basename of JSON.
    * ``parsed_json`` — basename of the originating JSON file.
    * ``url_hash`` — ``hashlib.md5(url.encode("utf-8")).hexdigest()`` without
      the ``article_`` / ``.html`` affixes (callers compose filenames).

    Args:
        parsed_dir: Directory scanned with ``Path.glob("*.json")`` (non-recursive).

    Returns:
        A list of task dicts; empty when no JSON exists or every file is unreadable.

    Side effects:
        Logs ERROR if no JSON matched; WARNING per unreadable file; INFO with
        total unique URLs and file count on success.
    """
    # Global URL set: dedupe across the entire corpus, not just within one file.
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

        # Metadata block is optional; use .get chains to avoid KeyError.
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
    """Coordinates Playwright-based article HTML capture.

    Instances are **lightweight**: they hold headless/proxy preferences and expose
    methods that spin up a full browser for the duration of a batch job, then shut
    it down. There is no long-lived background process inside the class.

    **Proxy handling**

    * ``no_proxy=True`` forces ``self.proxy`` to ``None`` regardless of env/config.
    * Otherwise ``proxy`` constructor argument wins; if omitted, ``DEFAULT_PROXY``
      from ``config`` is used (often a residential or datacenter endpoint string).

    **Typical usage**

    * ``ArticleScraper(headless=True, no_proxy=False).fetch_articles("final_data.csv", "article_manifest.json")``
    * ``ArticleScraper().fetch_from_parsed(Path("parsed_articles"), "manifest.json", force=False, limit=0)``
    """

    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[str] = None,
        no_proxy: bool = False,
    ):
        """Store scraping preferences; no browser is started here.

        Args:
            headless: When True, Chromium runs without a visible window (usual for
                servers). False aids local debugging of selectors or bot walls.
            proxy: Full proxy URL string, e.g.
                ``http://user:pass@host:port``. Passed to ``_parse_proxy`` later.
            no_proxy: If True, unconditionally disables proxying for this
                instance (ignores *proxy* and ``DEFAULT_PROXY``).
        """
        self.headless = headless
        if no_proxy:
            self.proxy = None
        else:
            self.proxy = proxy if proxy is not None else DEFAULT_PROXY

    def _parse_proxy(self) -> Optional[dict]:
        """Convert ``self.proxy`` URL string into Playwright’s proxy dict format.

        Playwright expects roughly::

            {"server": "http://host:port", "username": "...", "password": "..."}

        **Why urlparse**

        Embedded credentials in the authority component must be split so Playwright
        can supply them separately (some transports mishandle userinfo left in
        the raw server URL).

        Returns:
            ``None`` when ``self.proxy`` is falsy. Otherwise a dict always
            containing ``server`` (scheme + host + port). ``username`` / ``password``
            keys appear only when present in the URL.

        Side effects:
            INFO log summarizing scheme/host/port and whether auth was detected.
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
        """CSV pipeline: read ``search_results/<input_csv>`` and fetch each ``link``.

        **Preconditions**

        * ``RESULTS_DIR / input_csv`` must exist. Typical upstream artifact names
          include ``final_data.csv`` (see CLI default). If missing, the method logs
          ERROR and returns immediately — no partial manifest.

        **Row expectations**

        ``csv.DictReader`` requires a header row. Each row dict must at minimum
        contain:

        * ``link`` — absolute article URL to pass to ``page.goto``.
        * ``title``, ``source``, ``query`` — copied into the manifest for lineage.

        **Per-row control flow**

        For each row index *i*:

        1. Log progress ``[i+1/total]``.
        2. Enter retry loop ``1..MAX_RETRIES``:

           a. Build a new ``BrowserContext`` with desktop UA and optional proxy dict.
           b. Apply stealth patches; open a page; install route handler that aborts
              images/media/fonts.
           c. ``goto`` with ``wait_until="domcontentloaded"`` — **not** ``networkidle``,
              which can hang on perpetual analytics beacons.
           d. Sleep 2–5s (uniform) to let client-side hydration settle.
           e. ``page.content()`` serializes the live DOM to a string; MD5 hash of
              *url* determines ``article_<hash>.html`` filename under
              ``RAW_ARTICLES_DIR``.
           f. On success, append manifest entry and ``break`` retry loop.
           g. On ``PlaywrightError``, log WARNING with attempt count; backoff 5–10s
              before next attempt unless attempts exhausted.
           h. ``finally``: close context if created — critical so sockets and
              processes do not leak across retries.

        3. Regardless of outcome, sleep ``MIN_DELAY_SEC..MAX_DELAY_SEC`` between rows
           to avoid hammering targets or proxy endpoints.

        **Post-batch**

        After all rows, the browser is closed and ``downloaded_manifest`` (only
        successful downloads) is JSON-serialized to
        ``RAW_ARTICLES_DIR / manifest_name`` with UTF-8 encoding and pretty indent.

        Args:
            input_csv: Filename relative to ``RESULTS_DIR`` (not an absolute path).
            manifest_name: Output JSON filename placed in ``RAW_ARTICLES_DIR``.

        Returns:
            ``None`` explicitly; manifest path is only logged.

        Side effects:
            Writes HTML files, manifest JSON, extensive INFO/WARNING/ERROR logs.
        """
        csv_path = RESULTS_DIR / input_csv
        if not csv_path.exists():
            logger.error(
                f"Input CSV not found at {csv_path}. Did you run parse_engine.py first?"
            )
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tasks = list(reader)

        # Only successful fetches append here — failed rows leave no manifest record
        # in this legacy path (unlike fetch_from_parsed which records failures).
        downloaded_manifest = []

        # Compute once per batch — proxy_dict is reused for every context.
        proxy_dict = self._parse_proxy()

        with sync_playwright() as p:
            launch_opts = {"headless": self.headless}

            # Launch-level proxy: Playwright docs recommend server-only here when
            # also passing credentials at the context layer below.
            if proxy_dict:
                launch_opts["proxy"] = {"server": proxy_dict["server"]}

            browser = launch_chromium(p, launch_opts)

            for i, task in enumerate(tasks):
                url = task["link"]
                logger.info(f"[{i + 1}/{len(tasks)}] Fetching article: {url[:60]}...")

                for attempt in range(1, MAX_RETRIES + 1):
                    context = None
                    try:
                        # Context receives the **full** proxy dict including auth.
                        context_opts = {
                            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            "ignore_https_errors": True,
                        }
                        if proxy_dict:
                            context_opts["proxy"] = proxy_dict

                        context = browser.new_context(**context_opts)
                        Stealth().use_sync(context)
                        page = context.new_page()

                        # Block heavyweight subresources; keep JS/CSS for SPAs.
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
        """Normalize one manifest record for the ``--from-parsed`` code path.

        **local_file semantics**

        * On success or skip-when-exists: basename string like ``article_<hash>.html``.
          Note: skipped files still record the filename so downstream consumers know
          which artifact corresponds to the URL without probing the filesystem
          prefix.
        * On repeated failure after retries: ``None`` — signals “no new HTML” for
          that URL while retaining the row for auditing/sorting.

        Args:
            task: A dict produced by ``load_tasks_from_parsed`` (must contain
                ``url``, ``title``, ``source``, ``query``; optional keys accessed
                via ``.get``).
            local_file: Basename under ``RAW_ARTICLES_DIR``, or ``None``.

        Returns:
            A plain dict suitable for ``json.dumps`` as one manifest element.
        """
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
        """JSON pipeline: scrape URLs discovered under *parsed_dir*.

        **Task sourcing**

        Delegates to ``load_tasks_from_parsed`` for globbing, validation, dedupe,
        and field normalization. Optional *limit* truncates the in-memory list
        **after** dedupe — useful for smoke tests without editing JSON files.

        **Idempotency**

        For each task, expected HTML path is ``RAW_ARTICLES_DIR/article_<url_hash>.html``.
        If that path exists and *force* is False:

        * No HTTP request is made for that URL.
        * ``skipped`` counter increments.
        * A manifest row is still appended with ``local_file`` set to the basename
          so batch analytics retain one row per intended URL.

        **Fetch loop**

        Mirrors ``fetch_articles``’s Playwright pattern (context per attempt,
        stealth, route blocking, ``domcontentloaded``, randomized settle sleep).
        Differences:

        * Uses ``Path.write_text`` for HTML atomically at the Python layer (single
          write call).
        * On success: ``downloaded`` += 1; manifest row with basename.
        * If all retries fail: ``failed`` += 1; manifest row with ``local_file=None``.

        **Manifest serialization**

        Written with ``ensure_ascii=False`` so titles/snippets with non-ASCII
        characters round-trip legibly in editors.

        Args:
            parsed_dir: Absolute or resolved directory containing ``*.json``.
            manifest_name: Output filename within ``RAW_ARTICLES_DIR``.
            force: When True, ignores existing HTML files and re-fetches every URL.
            limit: Maximum number of tasks to process after dedupe; ``0`` means all.

        Returns:
            Dict with keys ``total``, ``downloaded``, ``skipped``, ``failed`` —
            ``total`` reflects the task list **after** applying *limit* (not the
            pre-limit universe).

        Side effects:
            Creates/overwrites HTML and manifest JSON; structured INFO logs.
        """
        tasks = load_tasks_from_parsed(parsed_dir)
        if limit > 0:
            tasks = tasks[:limit]
            logger.info("Applied --limit=%d → %d task(s)", limit, len(tasks))

        if not tasks:
            logger.error("No scrapeable tasks from %s", parsed_dir)
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        # Full manifest: one entry per task (download, skip, or fail) for reproducible joins.
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
    """Define and parse CLI flags for ``python article_scraper.py``.

    **Modes**

    * Default: legacy CSV ingestion (``fetch_articles``) using ``--input`` relative
      to ``search_results/`` and ``--manifest`` output basename under
      ``raw_articles/``.
    * ``--from-parsed``: JSON ingestion (``fetch_from_parsed``) rooted at
      ``--parsed-dir`` (default ``parsed_articles/`` relative to ``BASE_DIR``).

    **Operational flags**

    * ``--force`` — only affects parsed mode; re-downloads even when HTML exists.
    * ``--limit`` — parsed mode only; caps task count after dedupe.
    * ``--no-proxy`` — constructs ``ArticleScraper(no_proxy=True)``.
    * ``--install-browsers`` — before scraping, invokes
      ``<sys.executable> -m playwright install chromium`` with ``check=True`` so
      install failure aborts the process before any URL work.

    Returns:
        An ``argparse.Namespace`` consumed by the ``__main__`` guard.

    Raises:
        SystemExit: On ``-h/--help`` or argparse validation errors (standard library
            behavior).
    """
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
    # CLI orchestration: optional browser bootstrap, then mode dispatch.
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
    # headless defaults True in ArticleScraper.__init__; proxy follows config unless --no-proxy.
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
