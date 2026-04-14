from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()
proxy_url = os.getenv("DEFAULT_PROXY")
parsed = urlparse(proxy_url)
proxy_dict = {
    "server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}",
    "username": parsed.username,
    "password": parsed.password,
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, proxy=proxy_dict)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()

    print("Navigating to IP checker...")
    page.goto("https://ipinfo.io", wait_until="domcontentloaded")
    print(page.inner_text("body"))

    browser.close()
