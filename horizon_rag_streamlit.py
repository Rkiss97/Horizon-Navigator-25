import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# --- SETUP ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Load PDFs from folder and attach filename to metadata ---
def load_pdfs(folder_path="docs"):
    all_pages = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["source_file"] = filename
            all_pages.extend(pages)
    return all_pages

# --- Split pages and embed using OpenAI, create FAISS index ---
def create_vectorstore(pages):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(pages)

    # Ensure metadata is preserved during text splitting
    for i, split_doc in enumerate(split_docs):
        source_file = pages[i // 10].metadata.get("source_file", "Unknown file")
        split_doc.metadata["source_file"] = source_file

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(split_docs, embeddings)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Horizon Navigator 2025", layout="wide")
st.title("Horizon Europe 2025 – Document Assistant")
st.markdown("Ask a question based on the official Horizon Europe 2025 work programmes. Answers will reference the relevant PDFs.")

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Searching documents..."):
        pages = load_pdfs()
        vectorstore = create_vectorstore(pages)
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
