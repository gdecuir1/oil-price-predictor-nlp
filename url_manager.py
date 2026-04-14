import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

from config import logger, DEFAULT_MAX_RESULTS


class URLGenerator:
    """Handles the creation of targeted Google News search URLs."""

    @staticmethod
    def generate_dates(start_date: str, end_date: str, interval_days: int) -> List[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []

        while start <= end:
            dates.append(start.strftime("%Y-%m-%d"))
            start += timedelta(days=interval_days)
        return dates

    @staticmethod
    def build_url(query: str, date_str: str, max_results: int) -> str:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        after = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        before = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        safe_query = query.replace(" ", "+")

        return (
            f"https://www.google.com/search?q={safe_query}"
            f"+after:{after}+before:{before}&tbm=nws&hl=en&gl=us&num={max_results * 2}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Google News Search Tasks")
    parser.add_argument(
        "--queries",
        "-q",
        type=Path,
        required=True,
        help="File containing the list of search queries",
    )
    parser.add_argument(
        "--start", "-s", type=str, required=True, help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", "-e", type=str, required=True, help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=7, help="Days between searches"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("tasks.json"), help="Output JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = URLGenerator()

    dates = generator.generate_dates(args.start, args.end, args.interval)
    tasks: List[Dict] = []

    # Fetch the queries
    with open(args.queries, "r") as f:
        queries = [line.strip() for line in f if line.strip()]

    for query in queries:
        for date_str in dates:
            tasks.append(
                {
                    "query": query,
                    "search_date": date_str,
                    "url": generator.build_url(query, date_str, DEFAULT_MAX_RESULTS),
                    "max_results": DEFAULT_MAX_RESULTS,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    logger.info(f"Generated {len(tasks)} target URLs and saved to {args.output}")
