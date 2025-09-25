import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

import langchain_groq
dir(langchain_groq)

load_dotenv()
api_key=os.getenv("groq_API_KEY")


persistant_directory="./"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb=Chroma(persist_directory=persistant_directory,embedding_function=embeddings)
retriever=vectordb.as_retriever(search_kwargs={"k":3})

uploaded_file=st.file_uploader("Upload a document", type=["pdf", "txt"])


if uploaded_file:
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    st.success("PDF loaded successfully!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    vectordb.add_documents(docs)
    st.success("Document added to vector database!")

query = st.text_input("Ask a question about the document:")

if query:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.7,api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    result = qa_chain.invoke(query)
    st.markdown(f"**Answer:** {result['result']}")
































# prompt="suggest me a skill that is in demand"
# #response = llm.invoke(prompt)
# #print(response)


# template="give me 3 career skills that are in demand in {year}"
# prompt_template = PromptTemplate.from_template(template)

# chain= prompt_template | llm | StrOutputParser()

# response = chain.invoke({"year":2000})
# print(response)