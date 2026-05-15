"""
parser.py – Extract structured article data from scraped Google News HTML pages.

Supports four complementary extraction strategies applied in priority order:
  1. JSON-LD structured data (highest fidelity)
  2. OpenGraph / <meta> page-level article metadata
  3. Google News card DOM selectors (div.SoaBEf, g-card, article)
  4. Generic card heuristics (article, .post, .story, .card, etc.)

Duplicate articles (keyed by URL or title) are merged so that later
strategies fill in fields the earlier ones missed.

Usage
-----
    # Parse oil_raw_articles/ → parsed_articles/ (default paths)
    python parser.py

    # Custom paths with verbose logging
    python parser.py -i /path/to/html_dir -o /path/to/output_dir -v

    # Quiet mode – only warnings and errors
    python parser.py -q

    # Keep raw JSON-LD / card text in the output
    python parser.py --include-raw
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Article @type values recognised in JSON-LD blocks
_ARTICLE_LD_TYPES = frozenset({
    "NewsArticle",
    "Article",
    "BlogPosting",
    "ReportageNewsArticle",
})


# ---------------------------------------------------------------------------
# Primary entry points
# ---------------------------------------------------------------------------

def parse_news_html(html, source_file=None, include_raw=False, min_title_length=8, task_meta=None):
    """Parse a full HTML page and return all extractable article records.

    Four extraction strategies run in sequence.  Results are de-duplicated by
    URL (falling back to title), with later extractions merging non-empty
    fields into earlier ones.

    Parameters
    ----------
    html : str
        Raw HTML content of the page.
    source_file : str | None
        Filename of the originating HTML file (stored in metadata for tracing).
    include_raw : bool
        If True, keep the ``raw`` JSON-LD node and ``rawCardText`` on each
        article.  Useful for debugging selectors.
    min_title_length : int
        Articles whose title is shorter than this (and have no URL) are
        dropped as noise.

    Returns
    -------
    dict
        ``{"metadata": {…}, "count": int, "articles": [dict, …]}``
    """
    soup = BeautifulSoup(html, "html.parser")

    metadata = extract_page_metadata(soup, source_file, task_meta=task_meta)
    articles_by_key = {}

    # Strategy 1 – JSON-LD structured data (highest fidelity)
    for item in extract_json_ld_articles(soup):
        add_article(articles_by_key, normalize_article(item, metadata))

    # Strategy 2 – page-level OpenGraph / meta tags
    page_article = extract_page_article(soup, metadata)
    if page_article:
        add_article(articles_by_key, page_article)

    # Strategy 3 – Google News DOM card selectors
    for item in extract_google_news_cards(soup, metadata):
        add_article(articles_by_key, item)

    # Strategy 4 – generic card heuristics (broadest, lowest fidelity)
    for item in extract_generic_cards(soup, metadata):
        add_article(articles_by_key, item)

    articles = [
        strip_raw(a) if not include_raw else a
        for a in articles_by_key.values()
        if len(a.get("title") or "") >= min_title_length or a.get("url")
    ]

    return {
        "metadata": metadata,
        "count": len(articles),
        "articles": articles,
    }


def parse_file(path, include_raw=False, task_meta=None):
    """Read a single HTML file from disk and parse it.

    Parameters
    ----------
    path : str | Path
        Path to the ``.html`` file.
    include_raw : bool
        Forwarded to :func:`parse_news_html`.
    task_meta : dict | None
        Optional ``{"taskQuery": str, "searchDate": str}`` from the task file
        that generated this HTML, used to embed task provenance in each article.

    Returns
    -------
    dict
        Same structure as :func:`parse_news_html`.
    """
    path = Path(path)
    return parse_news_html(
        path.read_text(encoding="utf-8", errors="ignore"),
        source_file=path.name,
        include_raw=include_raw,
        task_meta=task_meta,
    )


def parse_folder(folder, pattern="*.html", include_raw=False):
    """Parse every HTML file in *folder* and return a flat list of articles.

    Parameters
    ----------
    folder : str | Path
        Directory containing HTML files.
    pattern : str
        Glob pattern for matching files (default ``*.html``).
    include_raw : bool
        Forwarded to :func:`parse_file`.

    Returns
    -------
    list[dict]
        All extracted article dicts, concatenated across files.
    """
    results = []
    for path in Path(folder).glob(pattern):
        parsed = parse_file(path, include_raw=include_raw)
        results.extend(parsed["articles"])
    return results


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_page_metadata(soup, source_file=None, task_meta=None):
    """Build a metadata dict describing the page itself (not individual articles).

    Fields include the canonical URL, page title, site name, language, and —
    for Google Search result pages — the original search query.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM of the page.
    source_file : str | None
        Originating filename for traceability.
    task_meta : dict | None
        Optional ``{"taskQuery": str, "searchDate": str}`` from the task file
        that generated this HTML.

    Returns
    -------
    dict
    """
    canonical = attr(soup, 'link[rel="canonical"]', "href")
    page_title = meta(soup, "og:title") or text(soup, "title")
    task_meta = task_meta or {}

    return {
        "sourceFile": source_file,
        "pageTitle": clean(page_title),
        "canonicalUrl": canonical,
        "siteName": clean(meta(soup, "og:site_name")),
        "description": clean(meta(soup, "description") or meta(soup, "og:description")),
        "language": soup.html.get("lang") if soup.html else None,
        "query": extract_search_query(soup),
        "taskQuery": task_meta.get("taskQuery"),
        "searchDate": task_meta.get("searchDate"),
        "scrapedAt": datetime.now(timezone.utc).isoformat(),
    }


def extract_search_query(soup):
    """Recover the original search query from a Google Search result page.

    Tries two approaches:
      1. The value of the ``<input name="q">`` or ``<textarea name="q">``
         element (the search box).
      2. Stripping the trailing ``" - Google Search"`` from ``<title>``.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM.

    Returns
    -------
    str | None
    """
    q = attr(soup, 'input[name="q"]', "value") or text(soup, 'textarea[name="q"]')
    if q:
        return q

    title = text(soup, "title")
    if title:
        return re.sub(r"\s*-\s*Google Search\s*$", "", title, flags=re.I)

    return None


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def extract_json_ld_articles(soup):
    """Extract articles from ``<script type="application/ld+json">`` blocks.

    Only nodes whose ``@type`` is one of the recognised article types
    (NewsArticle, Article, BlogPosting, ReportageNewsArticle) are returned.
    Nested ``@graph`` arrays are flattened before inspection.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM.

    Returns
    -------
    list[dict]
        Raw (un-normalised) article dicts with a ``"raw"`` key holding the
        original JSON-LD node.
    """
    out = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.get_text(strip=True))
        except Exception:
            logger.debug("Skipping malformed JSON-LD block")
            continue

        for node in flatten_json_ld(data):
            types = node.get("@type")
            if isinstance(types, str):
                types = [types]
            if not isinstance(types, list):
                types = []

            if any(t in _ARTICLE_LD_TYPES for t in types):
                out.append({
                    "title": node.get("headline") or node.get("name"),
                    "url": node.get("url") or extract_main_entity_url(node.get("mainEntityOfPage")),
                    "snippet": node.get("description"),
                    "source": safe_get(node, ["publisher", "name"]),
                    "author": extract_author(node.get("author")),
                    "publishedAt": node.get("datePublished"),
                    "modifiedAt": node.get("dateModified"),
                    "image": extract_image(node.get("image")),
                    "section": node.get("articleSection"),
                    "keywords": normalize_keywords(node.get("keywords")),
                    "raw": node,
                })

    logger.debug("JSON-LD extraction found %d article(s)", len(out))
    return out


def flatten_json_ld(data):
    """Recursively flatten JSON-LD data that may contain ``@graph`` arrays.

    A single JSON-LD ``<script>`` block can hold one object, a list of
    objects, or objects that themselves contain ``@graph`` arrays.  This
    function normalises all of those into a flat list of dicts.

    Parameters
    ----------
    data : dict | list
        Parsed JSON-LD payload.

    Returns
    -------
    list[dict]
    """
    items = data if isinstance(data, list) else [data]
    out = []

    for item in items:
        if not isinstance(item, dict):
            continue
        if "@graph" in item:
            out.extend(flatten_json_ld(item["@graph"]))
        else:
            out.append(item)

    return out


def extract_page_article(soup, metadata):
    """Extract a single article from page-level OpenGraph / meta tags.

    This covers pages that are themselves a single article (e.g. a news
    story opened directly) rather than a search-results listing.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM.
    metadata : dict
        Page metadata (used as fallback for missing fields).

    Returns
    -------
    dict | None
        Normalised article, or ``None`` if neither a title nor a URL could
        be found.
    """
    title = meta(soup, "article:title") or meta(soup, "og:title") or text(soup, "h1")
    url = meta(soup, "og:url") or metadata.get("canonicalUrl")

    if not title and not url:
        return None

    return normalize_article({
        "title": title,
        "url": url,
        "snippet": meta(soup, "og:description") or meta(soup, "description"),
        "source": meta(soup, "og:site_name"),
        "author": meta(soup, "article:author"),
        "publishedAt": meta(soup, "article:published_time"),
        "modifiedAt": meta(soup, "article:modified_time"),
        "section": meta(soup, "article:section"),
        "image": meta(soup, "og:image"),
    }, metadata)


def extract_google_news_cards(soup, metadata):
    """Extract articles from Google News card DOM elements.

    Targets the class names and tag names Google uses for news result cards:
    ``div.SoaBEf``, ``<g-card>``, and ``<article>``.  These selectors are
    fragile and may need updating when Google changes its markup.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM.
    metadata : dict
        Page metadata for normalisation.

    Returns
    -------
    list[dict]
        Normalised article dicts.
    """
    cards = []
    cards.extend(soup.select("div.SoaBEf"))
    cards.extend(soup.select("g-card"))
    cards.extend(soup.select("article"))

    out = []

    for card in unique_nodes(cards):
        link = select_one(card, "a.WlydOe[href], a[href]")
        time_el = select_one(card, "[data-ts], time, .OSrXXb, .rbYSKb")

        out.append(normalize_article({
            "title": text(card, ".n0jPhd, h3, h2, [role='heading']"),
            "url": attr(link, None, "href"),
            "snippet": text(card, ".UqSP2b, .GI74Re, .st, p"),
            "source": text(card, ".MgUUmf span, cite, .CEMjEf"),
            "publishedAt": attr(time_el, None, "datetime"),
            "timestamp": to_number(attr(time_el, None, "data-ts")),
            "timeText": clean(time_el.get_text(" ", strip=True)) if time_el else None,
            "image": attr(card, "img", "src") or attr(card, "img", "data-src"),
            "rawCardText": clean(card.get_text(" ", strip=True)),
        }, metadata))

    logger.debug("Google News card extraction found %d card(s)", len(out))
    return out


def extract_generic_cards(soup, metadata):
    """Broad heuristic extraction using common article/card class names.

    Acts as a catch-all for pages that are neither Google News results nor
    single-article pages.  Selectors target semantic tags (``<article>``)
    and common CMS class names (``.post``, ``.story``, ``.card``, etc.).

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed DOM.
    metadata : dict
        Page metadata for normalisation.

    Returns
    -------
    list[dict]
        Normalised article dicts.
    """
    selectors = [
        "article",
        '[itemtype*="NewsArticle"]',
        '[itemtype*="Article"]',
        ".article",
        ".post",
        ".story",
        ".card",
        ".result",
        ".news",
    ]

    nodes = []
    for selector in selectors:
        nodes.extend(soup.select(selector))

    out = []

    for node in unique_nodes(nodes):
        link = select_one(node, "a[href]")
        time_el = select_one(node, "time, [datetime], [data-ts]")

        out.append(normalize_article({
            "title": (
                text(node, "h1, h2, h3, [class*='title'], [class*='headline']")
                or clean(link.get_text(" ", strip=True)) if link else None
            ),
            "url": attr(link, None, "href"),
            "snippet": text(node, "p, [class*='summary'], [class*='snippet'], [class*='description']"),
            "source": text(node, "[class*='source'], [class*='publisher'], [class*='site']"),
            "author": text(node, "[class*='author'], [rel='author']"),
            "publishedAt": attr(time_el, None, "datetime"),
            "timestamp": to_number(attr(time_el, None, "data-ts")),
            "timeText": clean(time_el.get_text(" ", strip=True)) if time_el else None,
            "image": attr(node, "img", "src") or attr(node, "img", "data-src"),
            "rawCardText": clean(node.get_text(" ", strip=True)),
        }, metadata))

    logger.debug("Generic card extraction found %d card(s)", len(out))
    return out


# ---------------------------------------------------------------------------
# Article normalisation & de-duplication
# ---------------------------------------------------------------------------

def normalize_article(article, metadata):
    """Map a raw extraction dict into the canonical article schema.

    Resolves relative URLs against the page's canonical URL, extracts the
    bare domain, and fills in the site name / query from page metadata when
    the article itself lacks them.

    Parameters
    ----------
    article : dict
        Raw fields from any extraction strategy.
    metadata : dict
        Page-level metadata for fallback values.

    Returns
    -------
    dict
        Article with all canonical keys present (some may be ``None``).
    """
    url = absolutize(article.get("url"), metadata.get("canonicalUrl"))

    return {
        "title": clean(article.get("title")),
        "url": url,
        "domain": get_domain(url),
        "source": clean(article.get("source")) or metadata.get("siteName"),
        "snippet": clean(article.get("snippet")),
        "author": clean(article.get("author")),
        "publishedAt": clean(article.get("publishedAt")),
        "modifiedAt": clean(article.get("modifiedAt")),
        "timestamp": article.get("timestamp"),
        "timeText": clean(article.get("timeText")),
        "image": absolutize(article.get("image"), metadata.get("canonicalUrl")),
        "section": clean(article.get("section")),
        "keywords": article.get("keywords") or [],
        "query": metadata.get("query"),
        "taskQuery": metadata.get("taskQuery"),
        "searchDate": metadata.get("searchDate"),
        "sourceFile": metadata.get("sourceFile"),
        "rawCardText": article.get("rawCardText"),
        "raw": article.get("raw"),
    }


def add_article(article_map, article):
    """Insert or merge an article into the de-duplication map.

    The key is the article URL (preferred) or title.  When a duplicate is
    found the two records are merged: non-empty fields from the new article
    overwrite ``None``/empty values in the existing one.

    Parameters
    ----------
    article_map : dict
        Mutable map of key → article dict.
    article : dict
        Article to insert or merge.
    """
    key = article.get("url") or article.get("title")
    if not key:
        return

    existing = article_map.get(key)
    if not existing:
        article_map[key] = article
        return

    merged = dict(existing)
    for k, v in article.items():
        if v not in (None, "", []):
            merged[k] = v
    article_map[key] = merged


def strip_raw(article):
    """Return a copy of *article* without debugging-only fields.

    Removes ``raw`` (the full JSON-LD node) and ``rawCardText`` (all
    visible text from the card DOM node) to keep output compact.
    """
    return {k: v for k, v in article.items() if k not in {"raw", "rawCardText"}}


# ---------------------------------------------------------------------------
# DOM / text helpers
# ---------------------------------------------------------------------------

def clean(value):
    """Collapse whitespace and strip a string, returning ``None`` if empty."""
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def text(root, selector):
    """Return the cleaned visible text of the first element matching *selector*.

    Parameters
    ----------
    root : Tag | None
        Parent element to search within.
    selector : str | None
        CSS selector.  If ``None``, the text of *root* itself is returned.

    Returns
    -------
    str | None
    """
    if root is None:
        return None
    el = root.select_one(selector) if selector else root
    return clean(el.get_text(" ", strip=True)) if el else None


def attr(root, selector, name):
    """Return a cleaned attribute value from the first matching element.

    Parameters
    ----------
    root : Tag | None
        Parent element.
    selector : str | None
        CSS selector, or ``None`` to read from *root* directly.
    name : str
        Attribute name (e.g. ``"href"``, ``"content"``).

    Returns
    -------
    str | None
    """
    if root is None:
        return None
    el = root.select_one(selector) if selector else root
    return clean(el.get(name)) if el and el.has_attr(name) else None


def meta(soup, name):
    """Shortcut to read a ``<meta>`` tag's ``content`` by property or name.

    Tries ``property`` first (OpenGraph convention), then ``name``
    (standard HTML convention).
    """
    return (
        attr(soup, f'meta[property="{name}"]', "content")
        or attr(soup, f'meta[name="{name}"]', "content")
    )


def select_one(root, selector):
    """Null-safe ``root.select_one(selector)``."""
    return root.select_one(selector) if root else None


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def to_number(value):
    """Coerce *value* to ``int`` or ``float``, returning ``None`` on failure."""
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def absolutize(url, base=None):
    """Resolve a potentially relative *url* against *base*.

    Returns the original *url* unchanged if resolution fails or *base* is
    not provided.
    """
    if not url:
        return None
    try:
        return urljoin(base or "", url)
    except Exception:
        return url


def get_domain(url):
    """Extract the bare domain (no ``www.`` prefix) from a URL."""
    if not url:
        return None
    try:
        host = urlparse(url).hostname
        return host[4:] if host and host.startswith("www.") else host
    except Exception:
        return None


def unique_nodes(nodes):
    """De-duplicate a list of BeautifulSoup Tag objects by identity.

    Multiple CSS selectors can match the same DOM node.  This ensures each
    node is only processed once by tracking ``id(node)``.
    """
    seen = set()
    out = []
    for node in nodes:
        ident = id(node)
        if ident not in seen:
            seen.add(ident)
            out.append(node)
    return out


# ---------------------------------------------------------------------------
# JSON-LD field helpers
# ---------------------------------------------------------------------------

def extract_author(author):
    """Normalise a JSON-LD ``author`` field into a plain string.

    Handles strings, lists of authors, and ``{"name": …}`` dicts.
    """
    if not author:
        return None
    if isinstance(author, str):
        return author
    if isinstance(author, list):
        return ", ".join(filter(None, [extract_author(a) for a in author]))
    if isinstance(author, dict):
        return author.get("name")
    return None


def extract_image(image):
    """Normalise a JSON-LD ``image`` field into a single URL string.

    Handles plain URL strings, ``{"url": …}`` / ``{"contentUrl": …}``
    dicts, and arrays (takes the first element).
    """
    if not image:
        return None
    if isinstance(image, str):
        return image
    if isinstance(image, list):
        return extract_image(image[0]) if image else None
    if isinstance(image, dict):
        return image.get("url") or image.get("contentUrl")
    return None


def normalize_keywords(keywords):
    """Normalise a JSON-LD ``keywords`` field into a list of strings.

    Accepts a list, a comma-separated string, or ``None``.
    """
    if not keywords:
        return []
    if isinstance(keywords, list):
        return [clean(k) for k in keywords if clean(k)]
    return [clean(k) for k in str(keywords).split(",") if clean(k)]


def extract_main_entity_url(value):
    """Pull a URL out of a JSON-LD ``mainEntityOfPage`` value.

    The field can be a plain URL string or a ``{"@id": …}`` / ``{"url": …}``
    dict.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@id") or value.get("url")
    return None


def safe_get(obj, path):
    """Traverse nested dicts along *path* keys, returning ``None`` on miss.

    Example: ``safe_get(node, ["publisher", "name"])`` safely reads
    ``node["publisher"]["name"]``.
    """
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    """Construct the :class:`argparse.ArgumentParser` for the CLI.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Parse scraped Google News HTML files into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python parser.py                        # defaults\n"
            "  python parser.py -i raw/ -o out/ -v     # custom dirs, verbose\n"
            "  python parser.py -q --include-raw       # quiet + keep raw data\n"
        ),
    )

    p.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "oil_raw_articles",
        help="Directory containing raw HTML files (default: oil_raw_articles/)",
    )
    p.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "parsed_articles",
        help="Directory to write parsed JSON files (default: parsed_articles/)",
    )
    p.add_argument(
        "--pattern",
        default="*.html",
        help="Glob pattern for input files (default: *.html)",
    )
    p.add_argument(
        "--include-raw",
        action="store_true",
        default=False,
        help="Keep raw JSON-LD nodes and card text in the output",
    )
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Directory containing tasks_YYYY-MM-DD.json files (e.g. oil_tasks/). "
             "When provided, embeds taskQuery and searchDate into every parsed article.",
    )

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging (show per-strategy extraction counts, etc.)",
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except warnings and errors",
    )

    return p


def configure_logging(verbose=False, quiet=False):
    """Set up the root logger for console output.

    Parameters
    ----------
    verbose : bool
        If True, set level to DEBUG.
    quiet : bool
        If True, set level to WARNING.  Ignored if *verbose* is also True.
    """
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
    """CLI entry point: parse all HTML files in a directory to JSON.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename → task metadata lookup when task provenance args are provided.
    # HTML files are named md5(url).html by the scraper, so we hash each task URL
    # directly rather than relying on the manifest (which may be incomplete).
    file_to_task = {}
    if args.tasks_dir:
        import hashlib
        tasks_dir = args.tasks_dir.resolve()
        for task_file in sorted(tasks_dir.glob("tasks_*.json")):
            try:
                for task in json.loads(task_file.read_text(encoding="utf-8")):
                    url_hash = hashlib.md5(task["url"].encode("utf-8")).hexdigest()
                    fname = f"{url_hash}.html"
                    file_to_task[fname] = {
                        "taskQuery": task["query"],
                        "searchDate": task["search_date"],
                    }
            except Exception as exc:
                logger.warning("Could not read task file %s: %s", task_file.name, exc)
        logger.info("Task index built: %d file(s) linked to tasks.", len(file_to_task))

    html_files = sorted(input_dir.glob(args.pattern))
    if not html_files:
        logger.error("No files matching '%s' found in %s", args.pattern, input_dir)
        raise SystemExit(1)

    logger.info(
        "Parsing %d file(s) from %s -> %s", len(html_files), input_dir, output_dir
    )

    total_articles = 0
    errors = 0

    for html_path in html_files:
        try:
            task_meta = file_to_task.get(html_path.name)
            parsed = parse_file(html_path, include_raw=args.include_raw, task_meta=task_meta)
            out_path = output_dir / (html_path.stem + ".json")
            out_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            n = parsed["count"]
            total_articles += n
            logger.info("  %s -> %d article(s)", html_path.name, n)
        except Exception as exc:
            errors += 1
            logger.error("  FAILED %s: %s", html_path.name, exc)

    logger.info(
        "Done. %d article(s) from %d file(s) (%d error(s)).",
        total_articles, len(html_files), errors,
    )


if __name__ == "__main__":
    main()
