"""Visit the hosted app like a person would, and wake it up if it fell asleep.

Streamlit Community Cloud puts free apps to sleep after a stretch without visitors.
A sleeping app shows a "get this app back up" button instead of the app, and a plain
HTTP request doesn't press it, so this uses a real (headless) browser.

Run by .github/workflows/keep-awake.yml. Exits non-zero if the app never loads, so a
broken deployment shows up as a failed workflow run.
"""

import os
import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://trend-following-lab.streamlit.app/").rstrip("/") + "/"
# Text that only exists once the app is running (a navigation label of this app).
LOADED_TEXT = os.environ.get("LOADED_TEXT", "Phase 1 · Build")
WAIT_SECONDS = 300  # waking a sleeping app can take a few minutes


def app_loaded(page) -> bool:
    """Streamlit Cloud runs the app in an iframe at <url>/~/+/ inside the main page."""
    return any(
        "/~/+/" in frame.url and frame.get_by_text(LOADED_TEXT).count() > 0 for frame in page.frames
    )


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)

    wake = page.get_by_role("button", name=re.compile(r"get this app back up", re.I))
    try:
        wake.wait_for(timeout=20_000)
        wake.click()  # stay on this page afterwards: navigating away cancels the wake-up
        print("App was asleep: pressed the wake-up button.")
    except PlaywrightTimeout:
        print("No wake-up button: app was awake.")

    deadline = time.monotonic() + WAIT_SECONDS
    while not app_loaded(page):
        if time.monotonic() > deadline:
            page.screenshot(path="keep-awake.png", full_page=True)
            print(f"App did not finish loading within {WAIT_SECONDS} s: {URL}")
            sys.exit(1)
        page.wait_for_timeout(5_000)

    print(f"App is up: {URL}")
    browser.close()
