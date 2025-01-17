from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
def load_data(data):
    data = DirectoryLoader(data, loader_cls=PyPDFLoader, glob= "*.txt")
    documents = data.load()
    return documents

def text_spliter(data):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_spliter.split_documents(data)
    return docs

def download_embedding_model():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding

