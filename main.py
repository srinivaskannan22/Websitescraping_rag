from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
load_dotenv()

class notepad:
    
    def __init__(self,pageurl):
        self.pageurl=pageurl
        pc = Pinecone(api_key=os.getenv('PINECONE_API'))
        index = pc.Index("rag3")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_stores = PineconeVectorStore(index=index, embedding=embeddings)
    
    def datainjection(self):
        loader = WebBaseLoader(web_paths=[self.pageurl])
        docs=loader.load()
        return docs

    def chunking(self):
        splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        data=self.datainjection()
        chunk=splitter.split_documents(data)
        return chunk

    def vector_store(self):
        docs=self.chunking()
        self.vector_stores.add_documents(documents=docs)
        return "success"
    
    def similarity(self,content):
        reteriver=self.vector_stores.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.4},
            )
        return reteriver.invoke(content)
    
    def rag(self,content):
        similarity=self.similarity(content=content)
        client = InferenceClient(
            provider="fireworks-ai",
            api_key=os.getenv("HF_TOKEN"),
        )

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=[
                {
                    "role": "user",
                    "content": f"you are rag agent please tell me the answer based on the {similarity} and question{content}"
                }
            ],
        )
        return completion.choices[0].message

