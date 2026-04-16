import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient

load_dotenv()


def run_ingestion():
    print("-- NATIVE INGESTION STARTING --")


    print("Connecting to Google Gemini Embeddings...")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


    print("Loading PDFs...")
    loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)
    print(f"Prepared {len(docs)} chunks.")

    print("Connecting to Pinecone...")
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    index = pc.Index(index_name)

    print("Uploading vectors directly...")
    for i, doc in enumerate(docs):

        vector = embeddings_model.embed_query(doc.page_content)

        index.upsert(vectors=[(
            f"id-{i}",
            vector,
            {"text": doc.page_content, "source": doc.metadata.get("source", "unknown")}
        )])

    print("✅ SUCCESS: Data is live in Pinecone!")


if __name__ == "__main__":
    run_ingestion()