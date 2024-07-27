from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pinecone
from src.prompt import *

load_dotenv()
os.environ['PINECONE_API_KEY'] = "aa851085-206d-4788-aa69-74176f639ba8"
app = Flask(__name__)

# Lấy API key từ biến môi trường
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Tải mô hình embedding
embedding = download_embedding_model()


# Khởi tạo PineconeVectorStore
doc_search = PineconeVectorStore.from_existing_index(index_name="medicalchatbot",embedding=embedding)

# Định nghĩa PromptTemplate
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Khởi tạo mô hình LLM
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama",
                    config={"max_new_tokens": 512, "temperature": 0.7},
                    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin")

# Khởi tạo RetrievalQA
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=doc_search.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt})

@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
