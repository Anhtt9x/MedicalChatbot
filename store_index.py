from src.helper import text_spliter, load_data ,download_embedding_model
from langchain_pinecone import PineconeVectorStore 
from dotenv import load_dotenv
import os


load_dotenv()
os.environ['PINECONE_API_KEY'] = "aa851085-206d-4788-aa69-74176f639ba8"


documents = load_data("data")

docs = text_spliter(documents)

embeddings = download_embedding_model()


docsearch=PineconeVectorStore.from_texts([doc.page_content for doc in docs],
              embedding=embeddings,index_name="medicalchatbot")