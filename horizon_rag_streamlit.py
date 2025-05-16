import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import pickle

# --- SETUP ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- STREAMLIT UI ---
st.set_page_config(page_title="Horizon Navigator 2025", layout="wide")
st.title("Horizon Europe 2025 – Document Assistant")
st.markdown("Ask a question based on the official Horizon Europe 2025 work programmes. Answers will reference the relevant PDFs.")

user_question = st.text_input("Your question:")

# --- LOAD VECTORSTORE ---
# ✅ NE legyen paraméter — így nem lesz cache error
@st.cache_resource
def load_vectorstore():
    vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return vectorstore

if user_question:
    with st.spinner("Searching documents..."):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(user_question, k=3)

        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        st.markdown("### Answer")
        st.write(response)

        st.markdown("---")
        st.markdown("### Source Documents")
        for doc in docs:
            file = doc.metadata.get('source_file', 'Unknown file')
            page = doc.metadata.get('page', 'Unknown')
            try:
                page = int(page) + 1  # Convert to 1-indexed
            except:
                pass
            st.markdown(f"**File:** {file} – **Page:** {page}")
            st.markdown(f"> {doc.page_content[:500]}...")
