from flask import Flask , render_template, jsonify, request
from src.helper import download_embedding_model
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

app = Flask(__name__)

pinecone_api_key = os.getenv("PINECONE_API_KEY")

embedding = download_embedding_model()

pinecone.Pinecone(api_key=pinecone_api_key)

doc_search = PineconeVectorStore.from_existing_index(index_name="medicalchatbot",
                                                     embedding=embedding)

Prompt = PromptTemplate(template=prompt_template,input_variables=["context", "question"])

chain_type_kwargs = {"prompt":Prompt}

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", config={"max_new_tokens": 512, "temperature":0.7},
                    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin")

qa = RetrievalQA(llm=llm,
                 chain_type_kwargs=chain_type_kwargs,
                 retriever= doc_search.as_retriever(),
                 chain_type="stuff",
                 return_source_documents=True)


@app.route("/")
def home():
    return render_template("chat.html")


if __name__ == "__main__":
    app.run(debug=False)