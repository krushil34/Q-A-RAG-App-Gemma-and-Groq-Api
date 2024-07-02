import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain  
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

# load the groq Api key
groq_api_key=os.environ['GROQ_API_KEY']
# =os.getenv("GROQ_API_KEY")

if 'vector' not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="phi3")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter()
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50] )
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma-7b-it")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)

retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)

query= st.text_input("Input your promt here")

if query:
    start= time.process_time()
    response=retriever_chain.invoke({"input":query})
    print("Response time :",time.process_time()-start)
    st.write( response['answer'])











