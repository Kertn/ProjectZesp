import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser
import async_timeout
import os

INPUT_FILE = 'Data/links.txt'  # Your uploaded file with 250+ URLs
OUTPUT_FILE = 'Data/doc_links.txt'  # Output file with PDF/DOCX links
DOMAIN_PREFIX = 'https://al.edu.pl/wnit'  # From your example; used for filtering/normalization
USER_AGENT = 'SimpleCrawler/1.0 (+https://example.com)'  # From your code
DELAY_BETWEEN_REQUESTS = 1  # Seconds; increase if site rate-limits
TIMEOUT_SECONDS = 20  # Per request


# Function to normalize URLs (from your code)
def normalize_url(base, link):
    joined = urljoin(base, link)
    clean, _ = urldefrag(joined)
    return clean.rstrip("/")


# Async function to get robots.txt parser (adapted from your code)
async def get_robot_parser(session, base_url):
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        async with session.get(robots_url) as resp:
            if resp.status == 200:
                text = await resp.text()
                rp = robotparser.RobotFileParser()  # Corrected reference
                rp.parse(text.splitlines())
                return rp
    except Exception:
        pass
    rp = robotparser.RobotFileParser()  # Corrected reference
    rp.parse([])
    return rp


# Async function to fetch and scrape a single page for PDF/DOCX links
async def scrape_page_for_docs(session, url, rp, semaphore):
    async with semaphore:  # Limit concurrent requests
        if not rp.can_fetch(USER_AGENT, url):
            print(f"[DISALLOWED by robots.txt] {url}")
            return []

        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        try:
            async with async_timeout.timeout(TIMEOUT_SECONDS):
                async with session.get(url, headers={"User-Agent": USER_AGENT}) as resp:
                    if resp.status == 200 and "text/html" in resp.headers.get("Content-Type", ""):
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        doc_links = []
                        for a in soup.find_all('a', href=True):
                            href = a['href'].strip()
                            if href.lower().endswith(('.pdf', '.docx')):
                                absolute_url = normalize_url(url, href)
                                if absolute_url.startswith(DOMAIN_PREFIX):  # Filter to domain
                                    doc_links.append(absolute_url)
                        print(f"[SUCCESS] Found {len(doc_links)} docs on {url}")
                        return doc_links
                    else:
                        print(f"[SKIPPED] Non-HTML or bad status on {url}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
        return []


# Main async function
async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file '{INPUT_FILE}' not found. Upload it to Kaggle and try again.")
        return

    # Read URLs from input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(urls)} URLs from '{INPUT_FILE}'.")

    async with aiohttp.ClientSession() as session:
        rp = await get_robot_parser(session, DOMAIN_PREFIX)

        # Semaphore to limit concurrent requests (e.g., 5 at a time to avoid overwhelming the server or Kaggle limits)
        semaphore = asyncio.Semaphore(5)

        # Run scraping tasks asynchronously
        tasks = [scrape_page_for_docs(session, url, rp, semaphore) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all doc links, flattening the list
        all_doc_links = set()  # Use set to avoid duplicates
        for result in results:
            if isinstance(result, list):
                all_doc_links.update(result)

    # Write to output file (one link per line)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for link in sorted(all_doc_links):
            f.write(link + '\n')

    print(f"Done! {len(all_doc_links)} unique PDF/DOCX links saved to '{OUTPUT_FILE}'. Download it from Kaggle.")


# Run the async main function (works in Jupyter/Kaggle)
await main()