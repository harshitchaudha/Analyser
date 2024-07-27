import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')

def extract_pdf_text(pdf_files):
    combined_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

template = """
You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer. If you don't know the context, then don't answer the question.

context: \n{context}\n

question: \n{question}\n
Answer:"""

def initialize_conversation_chain():
    ai_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chain = load_qa_chain(ai_model, chain_type="stuff", prompt=prompt)
    return chain

def load_vector_store(index_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(index_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store

def handle_user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = load_vector_store("faiss_index")
    documents = vector_store.similarity_search(question)

    conversation_chain = initialize_conversation_chain()
    response = conversation_chain({"input_documents":documents, "question":question})

    st.write("Answer: ", response["output_text"])

st.set_page_config(
    page_title="PDF Analyzer",
    page_icon=':books:',
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("Upload PDF")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing PDF..."):
            raw_text = extract_pdf_text(uploaded_pdfs)
            text_chunks = split_text_into_chunks(raw_text)
            create_vector_store(text_chunks)
            st.success("VectorDB Upload Successful!")

def main():
    st.title("LLM GenAi ChatBot")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("🖇️Chat with PDF Analyzer 🗞️")
    st.markdown("<hr>", unsafe_allow_html=True)
    question = st.text_input("", placeholder="Ask a question")
    st.markdown("<hr>", unsafe_allow_html=True)

    if question:
        handle_user_input(question)

if __name__ == "__main__":
    main()
