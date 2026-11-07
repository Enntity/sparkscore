# SPDX-License-Identifier: AGPL-3.0-only
"""Render the SparkScore cards from cards.html with headless Chromium.

    python3 docs/assets/render_cards.py

Writes banner.png (README hero), social-preview.png (GitHub social preview,
1280 x 640) and social-preview.jpg next to this file. Needs Playwright's
Chromium (`pip install playwright && playwright install chromium`) and network
access for the two Google Fonts the cards use.
"""
import asyncio
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
CARDS = {"banner": (2172, 724), "social": (1280, 640)}


async def render():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for card, (w, h) in CARDS.items():
            page = await browser.new_page(viewport={"width": w, "height": h})
            await page.goto(f"{(HERE / 'cards.html').as_uri()}?card={card}")
            await page.wait_for_selector("body[data-ready]")
            problems = await page.evaluate("document.body.dataset.problems")
            if problems:
                print(f"{card}: {problems}")
            out = HERE / ("banner.png" if card == "banner" else "social-preview.png")
            await page.screenshot(path=str(out))
            print(out)
        await browser.close()


def main():
    asyncio.run(render())
    png, jpg = HERE / "social-preview.png", HERE / "social-preview.jpg"
    # A JPEG under 1 MB for sites that limit upload size; same pixels.
    subprocess.run(["python3", "-c", "import sys;from PIL import Image;Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2],quality=92)",
                    str(png), str(jpg)], check=True)
    print(jpg, os.path.getsize(jpg), "bytes")


if __name__ == "__main__":
    main()
