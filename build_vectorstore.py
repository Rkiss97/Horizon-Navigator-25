# build_vectorstore.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

# Optional: set your API key if not using Streamlit
os.environ["OPENAI_API_KEY"] = "sk-..."  # VAGY használd streamlit.secrets ha szükséges

def load_and_split_pdfs(folder_path="docs"):
    all_chunks = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(full_path)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["source_file"] = filename  # Add filename to metadata
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
    return all_chunks

# Load PDF chunks
print("Loading and splitting PDFs...")
documents = load_and_split_pdfs()

# Embed and index
print("Generating embeddings...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Save index locally
print("Saving FAISS index to ./vectorstore...")
vectorstore.save_local("vectorstore")

print("Vectorstore built and saved.")
