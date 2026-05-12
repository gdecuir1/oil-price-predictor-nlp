import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def parse_news_html(html, source_file=None, include_raw=False, min_title_length=8):
    soup = BeautifulSoup(html, "html.parser")

    metadata = extract_page_metadata(soup, source_file)
    articles_by_key = {}

    for item in extract_json_ld_articles(soup):
        add_article(articles_by_key, normalize_article(item, metadata))

    page_article = extract_page_article(soup, metadata)
    if page_article:
        add_article(articles_by_key, page_article)

    for item in extract_google_news_cards(soup, metadata):
        add_article(articles_by_key, item)

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


def extract_page_metadata(soup, source_file=None):
    canonical = attr(soup, 'link[rel="canonical"]', "href")
    page_title = meta(soup, "og:title") or text(soup, "title")

    return {
        "sourceFile": source_file,
        "pageTitle": clean(page_title),
        "canonicalUrl": canonical,
        "siteName": clean(meta(soup, "og:site_name")),
        "description": clean(meta(soup, "description") or meta(soup, "og:description")),
        "language": soup.html.get("lang") if soup.html else None,
        "query": extract_search_query(soup),
        "scrapedAt": datetime.now(timezone.utc).isoformat(),
    }


def extract_search_query(soup):
    q = attr(soup, 'input[name="q"]', "value") or text(soup, 'textarea[name="q"]')
    if q:
        return q

    title = text(soup, "title")
    if title:
        return re.sub(r"\s*-\s*Google Search\s*$", "", title, flags=re.I)

    return None


def extract_json_ld_articles(soup):
    out = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.get_text(strip=True))
        except Exception:
            continue

        for node in flatten_json_ld(data):
            types = node.get("@type")
            if isinstance(types, str):
                types = [types]
            if not isinstance(types, list):
                types = []

            if any(t in {"NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle"} for t in types):
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

    return out


def flatten_json_ld(data):
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

    return out


def extract_generic_cards(soup, metadata):
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

    return out


def normalize_article(article, metadata):
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
        "sourceFile": metadata.get("sourceFile"),
        "rawCardText": article.get("rawCardText"),
        "raw": article.get("raw"),
    }


def add_article(article_map, article):
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
    return {k: v for k, v in article.items() if k not in {"raw", "rawCardText"}}


def clean(value):
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def text(root, selector):
    if root is None:
        return None
    el = root.select_one(selector) if selector else root
    return clean(el.get_text(" ", strip=True)) if el else None


def attr(root, selector, name):
    if root is None:
        return None
    el = root.select_one(selector) if selector else root
    return clean(el.get(name)) if el and el.has_attr(name) else None


def meta(soup, name):
    return (
        attr(soup, f'meta[property="{name}"]', "content")
        or attr(soup, f'meta[name="{name}"]', "content")
    )


def to_number(value):
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def absolutize(url, base=None):
    if not url:
        return None
    try:
        return urljoin(base or "", url)
    except Exception:
        return url


def get_domain(url):
    if not url:
        return None
    try:
        host = urlparse(url).hostname
        return host[4:] if host and host.startswith("www.") else host
    except Exception:
        return None


def extract_author(author):
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
    if not keywords:
        return []
    if isinstance(keywords, list):
        return [clean(k) for k in keywords if clean(k)]
    return [clean(k) for k in str(keywords).split(",") if clean(k)]


def extract_main_entity_url(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@id") or value.get("url")
    return None


def safe_get(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def select_one(root, selector):
    return root.select_one(selector) if root else None


def unique_nodes(nodes):
    seen = set()
    out = []
    for node in nodes:
        ident = id(node)
        if ident not in seen:
            seen.add(ident)
            out.append(node)
    return out


def parse_file(path, include_raw=False):
    path = Path(path)
    return parse_news_html(
        path.read_text(encoding="utf-8", errors="ignore"),
        source_file=path.name,
        include_raw=include_raw,
    )


def parse_folder(folder, pattern="*.html"):
    results = []
    for path in Path(folder).glob(pattern):
        parsed = parse_file(path)
        results.extend(parsed["articles"])
    return results


if __name__ == "__main__":
    parsed = parse_file("example.html")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))