import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Set API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure the Streamlit app layout
st.set_page_config(page_title="Horizon Navigator '25", layout="wide")
st.title("Horizon Navigator '25 by poltextLAB")
st.markdown(
    "An AI-powered assistant for exploring Horizon Europe 2025 calls, rules, "
    "and funding conditions — straight from the official work programme documents."
)

# Prompt a dokumentum-alapú QA-hoz (stuff chain)
prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers questions **only** based on the provided EU Horizon Europe 2025 documents.
If the answer is not in the documents, say that you cannot find it in the work programme.

Question:
{question}

Relevant documents:
{context}
"""
)

# Text input field for user questions
user_question = st.text_input("Your question:")

# Load the pre-built vectorstore from local files (no parameters to avoid caching issues)
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )

# If a question has been entered, perform retrieval and answering
if user_question:
    with st.spinner("Searching documents..."):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(user_question, k=3)

        # LLM + dokumentum-chain (ez váltja ki a load_qa_chain-t)
        llm = ChatOpenAI(
            model="gpt-5", 
            temperature=0,
        )

        chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )

        # A chain a dokumentumok listáját "context" kulcs alatt várja
        response = chain.invoke(
            {
                "context": docs,
                "question": user_question,
            }
        )

        # Display the answer
        st.markdown("### Answer")
        st.write(response)

        # Display the source documents
        st.markdown("---")
        st.markdown("### Source Documents")
        for doc in docs:
            file = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", "Unknown")
            try:
                page = int(page) + 1  # Convert from 0-based to 1-based index
            except Exception:
                pass
            st.markdown(f"**File:** {file} – **Page:** {page}")
            st.markdown(f"> {doc.page_content[:500]}...")
