import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


def generate_dates(start_date: str, end_date: str, interval_days: int) -> List[str]:
    """Generate list of dates in YYYY-MM-DD format."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    return dates


def build_google_news_url(query: str, date_str: str, max_results: int) -> str:
    """Builds the search URL for a specific date window."""
    after = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    before = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    safe_query = query.replace(" ", "+")
    return f"https://www.google.com/search?q={safe_query}+after:{after}+before:{before}&tbm=nws&hl=en&gl=us&num={max_results * 2}"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Google News Search URLs")
    parser.add_argument(
        "--queries",
        "-q",
        nargs="+",
        required=True,
        help="List of search queries (e.g., -q 'AAPL' 'crypto news')",
    )
    parser.add_argument(
        "--start_date", "-s", type=str, required=True, help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end_date", "-e", type=str, required=True, help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=7, help="Days between searches"
    )
    parser.add_argument(
        "--max_results", "-m", type=int, default=8, help="Max results per search"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("target_urls.json"),
        help="Output JSON file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dates = generate_dates(args.start_date, args.end_date, args.interval)

    tasks: List[Dict] = []

    for query in args.queries:
        for date_str in dates:
            url = build_google_news_url(query, date_str, args.max_results)
            tasks.append(
                {
                    "query": query,
                    "search_date": date_str,
                    "url": url,
                    "max_results": args.max_results,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Generated {len(tasks)} target URLs and saved to {args.output}")


if __name__ == "__main__":
    main()
