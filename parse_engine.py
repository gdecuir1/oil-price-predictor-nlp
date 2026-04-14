import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

from config import logger, RESULTS_DIR, RAW_HTML_DIR


class OfflineParser:
    """Parses locally saved HTML files rapidly using BeautifulSoup."""

    @staticmethod
    def parse_google_news_html(html_content: str, max_results: int) -> list:
        soup = BeautifulSoup(html_content, "lxml")
        results = []

        # Google News card containers
        cards = soup.select("div[data-n-a-href], div.WlydOe, article")

        for card in cards[: max_results * 2]:
            try:
                # Find Title
                title_elem = card.select_one("h3, .JtKRv")
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # Find Link
                link_elem = card.select_one("a")
                if not link_elem or not link_elem.has_attr("href"):
                    continue
                link = link_elem["href"]

                # Clean up Google's redirect URLs
                if link.startswith("/url?"):
                    link = link.split("url=")[1].split("&")[0]
                if not link.startswith("http"):
                    link = f"https://www.google.com{link}"

                # Find Source
                source_elem = card.select_one("span, .CEMjEf, .wEwyrc")
                source = source_elem.get_text(strip=True) if source_elem else "Unknown"

                results.append(
                    {
                        "title": title,
                        "link": link,
                        "source": source,
                        "parsed_at": datetime.now().isoformat(),
                    }
                )

                if len(results) >= max_results:
                    break

            except Exception as e:
                logger.debug(f"Skipped a card due to parse error: {e}")
                continue

        return results

    def process_manifest(self, manifest_file: str, output_csv: str):
        manifest_path = RAW_HTML_DIR / manifest_file
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        all_results = []
        logger.info(f"Starting offline parse of {len(manifest)} local files...")

        for item in manifest:
            local_file = Path(item["local_file"])

            if not local_file.exists():
                logger.warning(f"File missing, skipping: {local_file}")
                continue

            with open(local_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            extracted_data = self.parse_google_news_html(
                html_content, item["max_results"]
            )

            for row in extracted_data:
                row["query"] = item["query"]
                row["search_date"] = item["search_date"]
                all_results.append(row)

        if all_results:
            csv_path = RESULTS_DIR / output_csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            logger.info(f"Successfully parsed {len(all_results)} records to {csv_path}")
        else:
            logger.warning(
                "No data extracted. You may need to update your BeautifulSoup CSS selectors."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Offline HTML Parser")
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        default="download_manifest.json",
        help="Input manifest file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="final_data.csv", help="Output CSV name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parser = OfflineParser()
    parser.process_manifest(args.manifest, args.output)
