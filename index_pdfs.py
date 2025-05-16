# index_pdfs.py

import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def index_documents():
    folder_path = "docs"
    all_pages = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            pages = loader.load_and_split()
            for p in pages:
                p.metadata['source'] = filename
            all_pages.extend(pages)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # A vectorstore mentése fájlba
    os.makedirs("vectorstore", exist_ok=True)
    with open("vectorstore/index.faiss", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    index_documents()
