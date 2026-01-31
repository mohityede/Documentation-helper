import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initializing Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=2048)

# Initializing vector store
vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)

# Initializing Chat model
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrived_docs=vectorStore.as_retriever().invoke(input=query,k=4)

    serialized = "\n".join((f"source: {doc.metadata.get('source','unkown')} \ncontent: {doc.page_content}") for doc in retrived_docs)

    # returing both serialized content and raw documents
    return serialized,retrived_docs

def run_llm(query:str)-> Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent= create_agent(model=model,tools=[retrieve_context],system_prompt=system_prompt)

    # building massage list
    messages=[{"role":"user","content":query}]

    # Invoke the agent
    response = agent.invoke({"messages":messages})

    # extract answer from last AI message
    answer = response["messages"][-1].content

    # extract context documents from ToolMessage artifacts
    context_docs=[]
    for message in response["messages"]:
        # check if thise ToolMassage with artifact
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):
            # The artifact should contain the list of document objects
            if isinstance(message,list):
                context_docs.extend(message.artifact)

    return {
        "answer":answer,
        "context":context_docs
    }

if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)

