"""
Agentic RAG System with Streamlit UI
Uses: PDF Processing, Tavily Search, and OpenRouter LLM
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import HumanMessage, SystemMessage

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Agentic RAG System")
st.markdown("""
### Intelligent Document Q&A with Web Search Fallback
Upload a PDF and ask questions. The AI will:
1. 📄 First search your PDF for answers
2. 🌐 Then use Tavily web search if needed
""")

# Sidebar for API keys
with st.sidebar:
    st.header("⚙️ Configuration")

    # Load API keys from environment variables or allow manual input
    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Get your key from openrouter.ai (or set in .env file)"
    )

    tavily_api_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Get your key from tavily.com (or set in .env file)"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Settings")

    model_name = st.selectbox(
        "Select Model",
        [
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-70b-instruct"
        ],
        index=2
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. Enter your API keys
    2. Upload a PDF document
    3. Ask questions naturally
    4. Get intelligent answers!
    """)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None


@st.cache_resource
def get_embeddings():
    """Get or create embeddings model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def process_pdf(pdf_file) -> tuple:
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            st.error("Could not extract text from PDF")
            return None, 0

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # Get cached embeddings
        embeddings = get_embeddings()

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Clean up temp file
        os.unlink(tmp_path)

        return vectorstore, len(splits)

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0


def search_pdf(vectorstore, query: str, k: int = 4) -> str:
    """Search the PDF and return relevant context"""
    if vectorstore is None:
        return None
    
    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return None
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        st.error(f"Error searching PDF: {str(e)}")
        return None


def search_web(tavily_key: str, query: str) -> str:
    """Search the web using Tavily"""
    try:
        tavily = TavilySearchResults(api_key=tavily_key, max_results=3)
        results = tavily.run(query)
        return str(results)
    except Exception as e:
        return f"Web search error: {str(e)}"


def get_answer(llm, question: str, pdf_context: Optional[str], web_context: Optional[str]) -> str:
    """Get answer from LLM using available context"""
    
    # Build context section
    context_parts = []
    sources_used = []
    
    if pdf_context:
        context_parts.append(f"=== DOCUMENT CONTENT ===\n{pdf_context}")
        sources_used.append("PDF")
    
    if web_context:
        context_parts.append(f"=== WEB SEARCH RESULTS ===\n{web_context}")
        sources_used.append("Web")
    
    context = "\n\n".join(context_parts) if context_parts else "No context available."
    
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.

RULES:
1. Use the DOCUMENT CONTENT first if it contains relevant information
2. Use WEB SEARCH RESULTS as supplementary or if the document doesn't cover the topic
3. Be specific and cite which source you're using (PDF or Web)
4. If neither source has the answer, say so clearly
5. Keep answers concise but comprehensive"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\n---\n\nQuestion: {question}")
    ]
    
    response = llm.invoke(messages)
    
    # Add source indicator
    source_label = " & ".join(sources_used) if sources_used else "AI"
    return f"{response.content}\n\n*Source: {source_label}*"


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file:
        if not st.session_state.pdf_processed or st.button("🔄 Reprocess PDF"):
            with st.spinner("Processing PDF... (first time may take a minute to download embedding model)"):
                vectorstore, num_chunks = process_pdf(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.success(f"✅ PDF processed! Created {num_chunks} text chunks.")
                else:
                    st.error("Failed to process PDF")

with col2:
    st.header("📊 Status")
    if st.session_state.pdf_processed:
        st.success("✅ PDF Ready")
    else:
        st.info("⏳ No PDF loaded")

    if openrouter_api_key:
        st.success("✅ OpenRouter Connected")
    else:
        st.warning("⚠️ OpenRouter key needed")

    if tavily_api_key:
        st.success("✅ Tavily Connected")
    else:
        st.warning("⚠️ Tavily key needed")

# Chat interface
st.markdown("---")
st.header("💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about your document or anything else..."):

    # Validate API keys
    if not openrouter_api_key:
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar")
        st.stop()

    if not tavily_api_key:
        st.error("⚠️ Please enter your Tavily API key in the sidebar")
        st.stop()

    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                # Initialize LLM
                llm = ChatOpenAI(
                    openai_api_key=openrouter_api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name=model_name,
                    temperature=temperature,
                    default_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Agentic RAG System"
                    }
                )
                
                # Step 1: Search PDF first
                pdf_context = None
                if st.session_state.vectorstore:
                    pdf_context = search_pdf(st.session_state.vectorstore, question)
                
                # Step 2: Search web (always, for supplementary info)
                web_context = search_web(tavily_api_key, question)
                
                # Step 3: Generate answer
                response = get_answer(llm, question, pdf_context, web_context)
                
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with LangChain, OpenRouter, and Tavily |
    <a href='https://openrouter.ai' target='_blank'>Get OpenRouter API Key</a> |
    <a href='https://tavily.com' target='_blank'>Get Tavily API Key</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
