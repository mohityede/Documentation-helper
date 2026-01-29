from dotenv import load_dotenv

import os
import ssl
import certifi
import asyncio

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document

import logger

load_dotenv()

logger.log_header("Initial setup")
# configure SSL context to use certifi certificates
logger.log_info("SSL context creating...")
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

logger.log_info("gemini embeddings creating...")
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=1024)

logger.log_info("vector store creating...")
vectorStore=PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])

tavily_crawl = TavilyCrawl()

async def main():
    """Main async function to orchestrate the entire process."""
    logger.log_header("DOCUMENTATION INGESTION PIPELINE")
    logger.log_info("🔍 TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/overview")

    crawl_result = tavily_crawl.invoke(input={
        "url":"https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth":3,
        "extract_depth":"advanced",
        "instructions":"Documantation relevent to AI agent"
    })
    all_docs = [Document(page_content=res['raw_content'],metadata={"source":res['url']}) for res in crawl_result['results']]
    logger.log_success(f"TavilyCrawl: successfully crawled {len(all_docs)} URLs from documantation site")

if __name__ == "__main__":
    asyncio.run(main())