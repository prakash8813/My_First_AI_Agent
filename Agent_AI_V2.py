import os
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()

# Check if API key is loaded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not set. Please set it in your .env or environment variables.")
    st.stop()

# Set for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------------
# 2️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="📄 Policy Chat Agent", page_icon="📚", layout="wide")
st.title("📄 Company Policy Chat Agent")
st.markdown("Ask questions about your company policies and get AI-based answers with sources!")

# -------------------------------
# 3️⃣ Load and cache the agent
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_agent(folder_path="./Policies", persist_dir="policy_db"):
    if not os.path.exists(folder_path):
        st.error(f"❌ Folder not found: {folder_path}")
        return None

    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx") and not file.startswith("~$"):
            loader = Docx2txtLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())

    if not docs:
        st.warning("⚠️ No policy documents found in the folder!")
        return None

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store in Chroma vectorstore
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    # Custom prompt template with polite fallback
    custom_prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions strictly based on company policy documents provided.
If the answer is not in the documents, respond politely with:

"I'm sorry, I could not find a relevant answer in the company policies. Please contact HR for clarification."

Question: {question}
Context from documents:
{context}
Answer:""",
        input_variables=["question", "context"]
    )
    
    # Create chain with custom prompt
    stuff_chain = load_qa_with_sources_chain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        prompt=custom_prompt
    )
    
    # Create QA chain
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain,
        return_source_documents=True
    )

    return qa_chain

qa_chain = load_agent()
if qa_chain is None:
    st.stop()

# -------------------------------
# 4️⃣ Session state for chat history
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# 5️⃣ User query form
# -------------------------------
with st.form("query_form"):
    query = st.text_input("Ask a question about company policy:")
    submitted = st.form_submit_button("Get Answer")

if submitted and query.strip():
    with st.spinner("🔍 Searching policies..."):
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        st.session_state.history.append({"query": query, "answer": answer, "sources": sources})

# -------------------------------
# 6️⃣ Display chat history
# -------------------------------
for chat in reversed(st.session_state.history):
    st.markdown(f"**Q:** {chat['query']}")
    st.markdown(f"**A:** {chat['answer']}")
    st.markdown(f"*Sources:* {', '.join(chat['sources'])}")
    st.markdown("---")
