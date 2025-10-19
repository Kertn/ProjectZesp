import aiohttp
import asyncio
import async_timeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse
from urllib import robotparser


START_URL = "https://al.edu.pl/wnit"
DOMAIN_PREFIX = "https://al.edu.pl/wnit"  # stricter domain filter
OUTPUT_FILE = "Data/links.txt"
USER_AGENT = "SimpleCrawler/1.0 (+https://example.com)"
DELAY_BETWEEN_REQUESTS = 1  # seconds

visited = set()
found_links = set()
queue = asyncio.Queue()




def normalize_url(base, link):
    joined = urljoin(base, link)
    clean, _ = urldefrag(joined)
    return clean.rstrip("/")

async def fetch(session, url):
    await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    try:
        async with async_timeout.timeout(20):
            async with session.get(url, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status == 200 and "text/html" in resp.headers.get("Content-Type", ""):
                    return await resp.text()
    except Exception:
        pass
    return None

async def get_robot_parser(session, url):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        async with session.get(robots_url) as resp:
            if resp.status == 200:
                text = await resp.text()
                rp = robotparser.RobotFileParser()
                rp.parse(text.splitlines())
                return rp
    except Exception:
        pass
    rp = robotparser.RobotFileParser()
    rp.parse([])
    return rp

async def crawl():
    async with aiohttp.ClientSession() as session:
        rp = await get_robot_parser(session, START_URL)
        await queue.put(START_URL)

        while not queue.empty():
            current_url = await queue.get()

            if current_url in visited or not current_url.startswith(DOMAIN_PREFIX):
                continue

            if not rp.can_fetch(USER_AGENT, current_url):
                print(f"[DISALLOWED] {current_url}")
                continue

            print(f"[CRAWLING] {current_url}")
            visited.add(current_url)
            found_links.add(current_url)

            html = await fetch(session, current_url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a['href'].strip()
                if href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                try:
                    absolute_url = normalize_url(current_url, href)
                    if absolute_url.startswith(DOMAIN_PREFIX) and absolute_url not in visited:
                        await queue.put(absolute_url)
                except Exception:
                    continue

        print(f"\n✅ Done! {len(found_links)} unique links found.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for link in sorted(found_links):
                f.write(link + "\n")


await crawl()