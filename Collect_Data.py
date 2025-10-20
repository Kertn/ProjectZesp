import asyncio
import aiohttp
import os
from langchain.document_loaders import AsyncHtmlLoader, PyPDFLoader, Docx2txtLoader  # Reverted to PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader, Docx2txtLoader
# from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Configuration (Updated: Larger chunks for more context)
WEB_LINKS_FILE = 'Data/links.txt'
DOC_LINKS_FILE = 'Data/doc_links.txt'
DOWNLOAD_DIR = 'Data/docs'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
DELAY_BETWEEN_REQUESTS = 1
CONCURRENCY_LIMIT = 5
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500


# Step 1: Read links (same)
def read_links(file_path):
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Skipping.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


web_links = read_links(WEB_LINKS_FILE)
doc_links = read_links(DOC_LINKS_FILE)
print(f"Loaded {len(web_links)} web links and {len(doc_links)} doc links.")


# Step 2: Async load web content (UPDATED: Clean HTML to plain text using BeautifulSoup)
async def load_web_content(session, url, semaphore):
    async with semaphore:
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        try:
            loader = AsyncHtmlLoader([url])
            docs = await loader.aload()

            # New: Clean raw HTML to extract plain text
            for doc in docs:
                soup = BeautifulSoup(doc.page_content, 'html.parser')
                # Extract text, remove scripts/styles, and clean up
                for script in soup(["script", "style", "nav", "footer", "header"]):  # Remove boilerplate tags
                    script.extract()
                clean_text = soup.get_text(separator=' ', strip=True)  # Get clean text
                doc.page_content = clean_text  # Replace raw HTML with clean text
                doc.metadata['source'] = url

            print(f"[SUCCESS] Loaded and cleaned web: {url}")
            return docs
        except Exception as e:
            print(f"[ERROR] Failed web load {url}: {e}")
            return []


# Step 3: Async download and load doc content (Reverted to PyPDFLoader as requested)
async def load_doc_content(session, url, semaphore):
    async with semaphore:
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    filename = urlparse(url).path.split('/')[-1]
                    local_path = os.path.join(DOWNLOAD_DIR, filename)
                    with open(local_path, 'wb') as f:
                        f.write(content)

                    ext = os.path.splitext(filename)[1].lower()
                    if ext == '.pdf':
                        loader = PyPDFLoader(local_path)
                    elif ext == '.docx':
                        loader = Docx2txtLoader(local_path)
                    else:
                        raise ValueError(f"Unsupported file type: {ext} for {filename}")

                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = url
                    print(f"[SUCCESS] Loaded doc: {url}")
                    return docs
                else:
                    raise Exception(f"Bad status: {response.status}")
        except Exception as e:
            print(f"[ERROR] Failed doc load {url}: {e}")
            return []


# Main async function (same)
async def load_and_unite():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    all_docs = []

    async with aiohttp.ClientSession() as session:
        web_tasks = [load_web_content(session, url, semaphore) for url in web_links]
        doc_tasks = [load_doc_content(session, url, semaphore) for url in doc_links]
        results = await asyncio.gather(*(web_tasks + doc_tasks), return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)

    print(f"Unite complete: {len(all_docs)} documents loaded.")
    return all_docs


documents = asyncio.run(load_and_unite())

# Step 4: Divide into chunks (with updated sizes)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# Step 5: Store (focus on FAISS)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local('faiss_index_1')
print("Stored as FAISS vector store: 'faiss_index' folder")

print("Rebuild complete! Now re-run your RAG cells.")