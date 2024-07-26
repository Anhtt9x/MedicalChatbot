from src.helper import text_spliter, load_data ,download_embedding_model
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index("medicalchatbot")

documents = load_data("data")

docs = text_spliter(documents)

embeddings = download_embedding_model()

docsearch = PineconeVectorStore.from_texts([doc.page_content for doc in docs], embeddings, index_name="medicalchatbot")