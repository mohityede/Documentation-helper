from dotenv import load_dotenv

import os
import ssl
import certifi
import asyncio
from typing import List
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

import logger

load_dotenv()

logger.log_header("Initial setup")
# configure SSL context to use certifi certificates
logger.log_info("SSL context creating...")
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

logger.log_info("gemini embeddings creating...")
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=2048)

logger.log_info("vector store creating...")
vectorStore=PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])

tavily_crawl = TavilyCrawl()

async def index_docs_async(documents:List[Document],batch_size:int=50):
    """Process documents in batches asynchronously."""
    logger.log_header("VECTOR STORAGE PHASE")
    logger.log_info(f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store")

    #create batches
    batches = [documents[i:i+batch_size] for i in range(0,len(documents),batch_size)]
    logger.log_info(f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    #process all batches asynchronously
    def add_batch(batch:List[Document],batch_num:int):
        try:
            vectorStore.add_documents(batch)
            logger.log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)")
        except Exception as e:
            logger.log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    #process batches concurrently
    tasks=[add_batch(batch,i+1) for i,batch in enumerate(batches)]
    # results= await asyncio.gather(*tasks,return_exceptions=True)

    #count successfull batches
    success = sum(1 for result in tasks if result is True)

    if success==len(batches):
        logger.log_success(f"VectorStore Indexing: All batches processed successfully! ({success}/{len(batches)})")
    else:
        logger.log_warning(f"VectorStore Indexing: Processed {success}/{len(batches)} batches successfully")


async def main():
    """Main async function to orchestrate the entire process."""
    logger.log_header("DOCUMENTATION INGESTION PIPELINE")
    logger.log_info("🔍 TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/overview")

    crawl_result = tavily_crawl.invoke(input={
        "url":"https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth":4,
        "extract_depth":"advanced",
        # "instructions":"Documantation relevent to AI agent"
    })
    all_docs = [Document(page_content=res['raw_content'],metadata={"source":res['url']}) for res in crawl_result['results']]
    logger.log_success(f"TavilyCrawl: successfully crawled {len(all_docs)} URLs from documantation site")

    logger.log_header("DOCUMENT CHUNKING PHASE")
    logger.log_info(f"✂️ Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(all_docs)
    logger.log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    #processing documents asynchronously
    await index_docs_async(documents=splitted_docs,batch_size=60)

    logger.log_header("PIPELINE COMPLETE")
    logger.log_success("🎉 Documentation ingestion pipeline finished successfully!")
    logger.log_info("📊 Summary:")
    logger.log_info(f"   • Pages crawled: {len(crawl_result)}")
    logger.log_info(f"   • Documents extracted: {len(all_docs)}")
    logger.log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())