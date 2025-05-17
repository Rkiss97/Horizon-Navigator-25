import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os

# Set API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure the Streamlit app layout
st.set_page_config(page_title="Horizon Navigator 2025", layout="wide")
st.title("Horizon Navigator WP2025 by poltextLAB")
st.markdown("An AI-powered assistant for exploring Horizon Europe 2025 calls, rules, and funding conditions — straight from the official work programme documents.")

# Text input field for user questions
user_question = st.text_input("Your question:")

# Load the pre-built vectorstore from local files (no parameters to avoid caching issues)
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# If a question has been entered, perform retrieval and answering
if user_question:
    with st.spinner("Searching documents..."):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(user_question, k=3)

        # Load QA chain and generate a response
        chain = load_qa_chain(ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0), chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        # Display the answer
        st.markdown("### Answer")
        st.write(response)

        # Display the source documents
        st.markdown("---")
        st.markdown("### Source Documents")
        for doc in docs:
            file = doc.metadata.get('source_file', 'Unknown file')
            page = doc.metadata.get('page', 'Unknown')
            try:
                page = int(page) + 1  # Convert from 0-based to 1-based index
            except:
                pass
            st.markdown(f"**File:** {file} – **Page:** {page}")
            st.markdown(f"> {doc.page_content[:500]}...")
