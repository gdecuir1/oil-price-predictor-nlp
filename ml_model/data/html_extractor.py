"""
HTML Article Extractor
======================

Converts raw HTML article files into clean plain-text suitable for
tokenisation.  Supports multiple extraction backends with automatic
fallback so that even poorly-structured pages yield *something* usable.

Extraction pipeline (per article):
    1. Try the configured primary backend (trafilatura by default).
    2. If that yields < ``min_chars`` of text, fall back to
       BeautifulSoup with aggressive tag stripping.
    3. If still insufficient, attempt ``readability-lxml``.
    4. Finally, regex-strip all tags as a last resort.
    5. Normalise whitespace, remove boilerplate, and optionally
       prepend extracted metadata (title, date, source) as a
       structured header that the model can attend to.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HTMLArticleExtractor:
    """Extracts clean article text from raw HTML files.

    The extractor is stateless — each call to :meth:`extract` or
    :meth:`extract_all` processes files independently, making it safe
    to use from multiple threads / DataLoader workers.

    Args:
        primary_backend: Preferred extraction library.
            One of ``"trafilatura"``, ``"bs4"``, ``"readability"``.
        min_chars: Minimum acceptable character count after extraction.
            If the primary backend returns fewer characters, fallbacks
            are tried.
        include_metadata_header: If ``True``, prepend a short structured
            header (title, source, date) to the extracted body text so
            the transformer can leverage metadata during encoding.
        parsed_articles_dir: Path to the ``parsed_articles/`` folder
            containing JSON metadata.  Used to look up titles, sources,
            and dates for the metadata header.

    Example::

        extractor = HTMLArticleExtractor(
            primary_backend="trafilatura",
            parsed_articles_dir=Path("../parsed_articles"),
        )
        articles = extractor.extract_all(Path("../raw_articles"))
    """

    # Tags whose entire subtree is useless for article text
    _STRIP_TAGS = {
        "script", "style", "nav", "footer", "header", "aside",
        "form", "iframe", "noscript", "svg", "button", "input",
        "select", "textarea", "menu", "menuitem",
    }

    # Regex for collapsing runs of whitespace after extraction
    _WHITESPACE_RE = re.compile(r"[ \t]+")
    _BLANK_LINES_RE = re.compile(r"\n{3,}")
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    def __init__(
        self,
        primary_backend: str = "trafilatura",
        min_chars: int = 100,
        include_metadata_header: bool = True,
        parsed_articles_dir: Optional[Path] = None,
    ) -> None:
        """Initialise the extractor with the chosen backend and settings."""
        self.primary_backend = primary_backend
        self.min_chars = min_chars
        self.include_metadata_header = include_metadata_header
        self.parsed_articles_dir = parsed_articles_dir

        # Pre-load a lookup from article filenames to parsed metadata
        self._metadata_index: Dict[str, Dict[str, Any]] = {}
        if parsed_articles_dir and parsed_articles_dir.exists():
            self._build_metadata_index(parsed_articles_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, html_path: Path) -> Optional[Dict[str, Any]]:
        """Extract clean text from a single HTML file.

        Args:
            html_path: Absolute or relative path to an HTML file.

        Returns:
            A dictionary with keys:
                - ``"text"``: cleaned article body text.
                - ``"title"``: article title (may be ``None``).
                - ``"source"``: source domain or publication name.
                - ``"filename"``: original HTML filename.
                - ``"char_count"``: length of the cleaned text.
                - ``"extraction_method"``: which backend succeeded.
            Returns ``None`` if no usable text could be extracted.
        """
        html_path = Path(html_path)
        if not html_path.exists():
            logger.warning("File not found: %s", html_path)
            return None

        raw_html = html_path.read_text(encoding="utf-8", errors="replace")

        # Try backends in priority order
        text, method = self._extract_with_fallbacks(raw_html)

        if not text or len(text.strip()) < self.min_chars:
            logger.debug(
                "Skipping %s — extracted text too short (%d chars)",
                html_path.name,
                len(text) if text else 0,
            )
            return None

        # Clean up the extracted text
        text = self._normalise_text(text)

        # Look up metadata from parsed articles
        meta = self._lookup_metadata(html_path.name)
        title = meta.get("title") if meta else self._extract_title(raw_html)
        source = meta.get("source") or meta.get("domain") if meta else None

        # Optionally prepend a structured header
        if self.include_metadata_header:
            header = self._build_metadata_header(title, source, meta)
            text = f"{header}\n\n{text}" if header else text

        return {
            "text": text,
            "title": title,
            "source": source,
            "filename": html_path.name,
            "char_count": len(text),
            "extraction_method": method,
        }

    def extract_all(self, articles_dir: Path) -> List[Dict[str, Any]]:
        """Extract text from every HTML file in a directory.

        Args:
            articles_dir: Directory containing ``*.html`` files.

        Returns:
            List of extraction result dicts (see :meth:`extract`).
            Files that fail extraction are silently skipped.
        """
        articles_dir = Path(articles_dir)
        html_files = sorted(articles_dir.glob("*.html"))
        logger.info(
            "Extracting text from %d HTML files in %s",
            len(html_files),
            articles_dir,
        )

        results: List[Dict[str, Any]] = []
        for html_path in html_files:
            result = self.extract(html_path)
            if result is not None:
                results.append(result)

        logger.info(
            "Successfully extracted %d / %d articles (%.1f%%)",
            len(results),
            len(html_files),
            100.0 * len(results) / max(len(html_files), 1),
        )
        return results

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _extract_with_fallbacks(self, raw_html: str) -> Tuple[str, str]:
        """Try each extraction backend in order until one succeeds.

        Args:
            raw_html: Full HTML source string.

        Returns:
            Tuple of (extracted_text, method_name).  ``extracted_text``
            may be empty if all backends fail.
        """
        # Ordered list of (method_name, callable) pairs
        backends = [
            (self.primary_backend, self._get_backend(self.primary_backend)),
            ("bs4", self._extract_bs4),
            ("readability", self._extract_readability),
            ("regex", self._extract_regex),
        ]

        # Deduplicate — don't retry the primary backend as a fallback
        seen = set()
        for method_name, fn in backends:
            if method_name in seen:
                continue
            seen.add(method_name)

            try:
                text = fn(raw_html)
                if text and len(text.strip()) >= self.min_chars:
                    return text.strip(), method_name
            except Exception as exc:
                logger.debug("Backend '%s' failed: %s", method_name, exc)

        return "", "none"

    def _get_backend(self, name: str):
        """Return the extraction callable for a backend name.

        Args:
            name: One of ``"trafilatura"``, ``"bs4"``, ``"readability"``.

        Returns:
            Bound method implementing that backend.
        """
        mapping = {
            "trafilatura": self._extract_trafilatura,
            "bs4": self._extract_bs4,
            "readability": self._extract_readability,
            "regex": self._extract_regex,
        }
        return mapping.get(name, self._extract_trafilatura)

    @staticmethod
    def _extract_trafilatura(html: str) -> str:
        """Use trafilatura for high-quality main-content extraction.

        Trafilatura is purpose-built for extracting the main textual
        content of web pages, ignoring navigation, ads, and boilerplate.

        Args:
            html: Raw HTML source.

        Returns:
            Extracted plain text.
        """
        import trafilatura

        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        return result or ""

    def _extract_bs4(self, html: str) -> str:
        """BeautifulSoup-based extraction with aggressive tag stripping.

        Removes scripts, styles, navs, footers, and other non-content
        elements, then extracts the remaining visible text.

        Args:
            html: Raw HTML source.

        Returns:
            Extracted plain text.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove all non-content tags
        for tag_name in self._STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Try to find the main content container
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("div", class_=re.compile(
                r"(article|content|post|story|entry|body)", re.I
            ))
            or soup.body
            or soup
        )

        return main_content.get_text(separator="\n", strip=True)

    @staticmethod
    def _extract_readability(html: str) -> str:
        """Use readability-lxml for Mozilla Readability-style extraction.

        This mirrors the algorithm Firefox uses for its Reader View.

        Args:
            html: Raw HTML source.

        Returns:
            Extracted plain text (HTML tags stripped from readability output).
        """
        from readability import Document

        doc = Document(html)
        # readability returns simplified HTML; strip remaining tags
        summary_html = doc.summary()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def _extract_regex(self, html: str) -> str:
        """Last-resort extraction via regex tag stripping.

        Blunt but reliable — strips every HTML tag and collapses whitespace.
        Produces noisy output but guarantees *some* text from any HTML.

        Args:
            html: Raw HTML source.

        Returns:
            Plain text with all HTML tags removed.
        """
        text = self._HTML_TAG_RE.sub(" ", html)
        return self._normalise_text(text)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _build_metadata_index(self, parsed_dir: Path) -> None:
        """Construct a lookup from article filenames to parsed metadata.

        Reads all JSON files in ``parsed_articles/`` and indexes every
        article record by the MD5 hash of its URL — the same hash used
        to name the corresponding HTML file in ``raw_articles/``.

        Args:
            parsed_dir: Path to the parsed_articles directory.
        """
        for json_path in parsed_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for article in data.get("articles", []):
                    url = article.get("url")
                    if url:
                        # Reconstruct the expected HTML filename
                        url_hash = hashlib.md5(url.encode()).hexdigest()
                        key = f"article_{url_hash}.html"
                        self._metadata_index[key] = article
            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("Failed to parse %s: %s", json_path.name, exc)

        logger.info(
            "Metadata index built: %d articles mapped",
            len(self._metadata_index),
        )

    def _lookup_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Look up parsed metadata for an article by its HTML filename.

        Args:
            filename: Name of the HTML file (e.g. ``article_abc123.html``).

        Returns:
            Metadata dict from the parsed JSON, or ``None`` if not found.
        """
        return self._metadata_index.get(filename)

    @staticmethod
    def _extract_title(html: str) -> Optional[str]:
        """Extract the <title> tag as a fallback when metadata is missing.

        Args:
            html: Raw HTML source.

        Returns:
            Title string, or ``None`` if not found.
        """
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.I)
        if match:
            title = match.group(1).strip()
            title = re.sub(r"\s+", " ", title)
            return title if title else None
        return None

    @staticmethod
    def _build_metadata_header(
        title: Optional[str],
        source: Optional[str],
        meta: Optional[Dict[str, Any]],
    ) -> str:
        """Build a structured text header from available metadata.

        The header is prepended to the article body so the transformer
        can attend to metadata features (title, source, date) alongside
        the main content.

        Args:
            title: Article title.
            source: Publication name or domain.
            meta: Full metadata dict from parsed articles.

        Returns:
            Multi-line header string, or empty string if no metadata.
        """
        parts: List[str] = []
        if title:
            parts.append(f"TITLE: {title}")
        if source:
            parts.append(f"SOURCE: {source}")
        if meta:
            date = meta.get("publishedAt") or meta.get("timeText")
            if date:
                parts.append(f"DATE: {date}")
            snippet = meta.get("snippet")
            if snippet:
                parts.append(f"SUMMARY: {snippet}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Text normalisation
    # ------------------------------------------------------------------

    def _normalise_text(self, text: str) -> str:
        """Clean up extracted text by collapsing whitespace and trimming.

        Args:
            text: Raw extracted text.

        Returns:
            Normalised text string.
        """
        # Collapse horizontal whitespace (preserve newlines)
        text = self._WHITESPACE_RE.sub(" ", text)
        # Collapse excessive blank lines
        text = self._BLANK_LINES_RE.sub("\n\n", text)
        # Strip leading/trailing whitespace on each line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        return text.strip()
