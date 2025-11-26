import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="Horizon Navigator '25", layout="wide")
st.title("Horizon Navigator '25 by poltextLAB")
st.markdown(
    "An AI-powered assistant for exploring Horizon Europe 2025 calls, rules, "
    "and funding conditions — straight from the official work programme documents."
)

user_question = st.text_input("Your question:")

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )

if user_question:
    with st.spinner("Searching documents..."):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(user_question, k=3)

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an assistant that answers questions strictly based on the provided Horizon Europe 2025 work programme documents.
If the answer is not clearly contained in these documents, say that the work programme does not provide this information.

Question:
{user_question}

Relevant excerpts from the work programme:
{context}
"""

        llm = ChatOpenAI(
            model="gpt-5",  
            temperature=0,
        )

        llm_response = llm.invoke(prompt)

        st.markdown("### Answer")
        st.write(llm_response.content)

        st.markdown("---")
        st.markdown("### Source Documents")
        for doc in docs:
            file = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", "Unknown")
            try:
                page = int(page) + 1
            except Exception:
                pass
            st.markdown(f"**File:** {file} – **Page:** {page}")
            st.markdown(f"> {doc.page_content[:500]}...")
