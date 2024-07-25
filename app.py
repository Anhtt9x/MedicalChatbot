from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pinecone
from src.prompt import *

load_dotenv()

app = Flask(__name__)

# Lấy API key từ biến môi trường
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Tải mô hình embedding
embedding = download_embedding_model()

# Khởi tạo Pinecone
pc =pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index(name="medicalchatbot",
                 host="https://medicalchatbot-52q17kl.svc.aped-4627-b74a.pinecone.io")
# Khởi tạo PineconeVectorStore
doc_search = PineconeVectorStore(index_name="medicalchatbot", embedding=embedding)

# Định nghĩa PromptTemplate
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Khởi tạo mô hình LLM
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama",
                    config={"max_new_tokens": 512, "temperature": 0.7},
                    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin")

# Khởi tạo RetrievalQA
qa = RetrievalQA(
    retriever=doc_search.as_retriever(),
    combine_documents_chain="stuff",
    llm=llm,
    combine_documents_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
