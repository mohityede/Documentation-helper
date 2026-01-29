from dotenv import load_dotenv

import os
import ssl
import certifi
import asyncio
from typing import List,Dict,Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract,TavilyMap
from langchain_core.documents import Document

import logger

load_dotenv()

# logger.log_header("Initial setup")
# # configure SSL context to use certifi certificates
# logger.log_info("SSL context creating...")
# ssl_context=ssl.create_default_context(cafile=certifi.where())
# os.environ["SSL_CERT_FILE"]=certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

# logger.log_info("gemini embeddings creating...")
# embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=1024)

# logger.log_info("vector store creating...")
# vectorStore=PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        logger.log_info(f"🔄 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs.")
        docs = await tavily_extract.ainvoke(
            input={"urls": urls, "extract_depth": "advanced"}
        )
        extracted_docs_count = len(docs.get("results", []))
        if extracted_docs_count > 0:
            logger.log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {extracted_docs_count} documents")
        else:
            logger.log_error(f"TavilyExtract: Batch {batch_num} failed to extract any documents, {docs}")
        return docs
    except Exception as e:
        logger.log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    logger.log_header("DOCUMENT EXTRACTION PHASE")
    logger.log_info(f"🔧 TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches")

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            logger.log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    logger.log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        logger.log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages

async def main():
    """Main async function to orchestrate the entire process."""
    logger.log_header("DOCUMENTATION INGESTION PIPELINE")
    logger.log_info("🔍 TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/overview")

    site_map = tavily_map.invoke("https://docs.langchain.com/oss/python/langchain/overview")
    logger.log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site")

    # Split URLs into batches of 20
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    logger.log_info(f"📋 URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches")

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    logger.log_success(f"TavilyCrawl: successfully crawled {len(all_docs)} URLs from documantation site")

if __name__ == "__main__":
    asyncio.run(main())