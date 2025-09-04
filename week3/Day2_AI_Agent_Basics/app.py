import os
from io import StringIO
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import asyncio
import nest_asyncio

# Patch nested event loops (important for Streamlit + async libs)
nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----------------------------
# Custom CSS for UI enhancements with dark/light theme support
# ----------------------------

def inject_custom_css():
    st.markdown("""
    <style>
    /* Main styling with theme compatibility */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border: 1px solid var(--border-color);
    }
    
    .stChatMessage [data-testid="stMarkdownContainer"] {
        padding: 0.5rem 1rem;
    }
    
    /* User message */
    [data-testid="stChatMessage"]:has(> div > div:nth-child(1) > div > div > div:nth-child(2)) {
        background-color: var(--user-msg-bg);
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"]:has(> div > div:nth-child(2) > div > div > div:nth-child(2)) {
        background-color: var(--assistant-msg-bg);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: var(--button-bg);
        color: var(--button-text);
        border: 1px solid var(--button-border);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--button-hover-bg);
        color: var(--button-text);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
        color: var(--expander-color);
    }
    
    .streamlit-expanderContent {
        background-color: var(--expander-content-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0;
        border-top: 1px solid var(--border-color);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--header-color);
    }
    
    /* Card-like containers */
    .card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        background-color: var(--card-bg);
        margin-bottom: 1.5rem;
        border: 1px solid var(--card-border);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: var(--status-active);
    }
    
    .status-inactive {
        background-color: var(--status-inactive);
    }
    
    /* Tag styling for agent labels */
    .agent-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .salary-tag {
        background-color: var(--salary-tag-bg);
        color: var(--salary-tag-text);
        border: 1px solid var(--salary-tag-border);
    }
    
    .insurance-tag {
        background-color: var(--insurance-tag-bg);
        color: var(--insurance-tag-text);
        border: 1px solid var(--insurance-tag-border);
    }
    
    /* Chat input styling */
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: var(--chat-input-bg);
        padding: 1rem 0;
        z-index: 999;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 1px dashed var(--border-color);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: var(--metric-bg);
        border: 1px solid var(--metric-border);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Light theme variables */
    [data-theme="light"] {
        --background-color: #f8f9fa;
        --user-msg-bg: #e6f7ff;
        --assistant-msg-bg: #f9f9f9;
        --button-bg: #4f8bf9;
        --button-text: white;
        --button-border: #4f8bf9;
        --button-hover-bg: #3a7de8;
        --expander-color: #4f8bf9;
        --expander-content-bg: #f8f9fa;
        --border-color: #e6e6e6;
        --header-color: #1f3d7a;
        --card-bg: white;
        --card-border: #e6e6e6;
        --status-active: #28a745;
        --status-inactive: #dc3545;
        --salary-tag-bg: #e6f7ff;
        --salary-tag-text: #0066cc;
        --salary-tag-border: #99c2ff;
        --insurance-tag-bg: #fff2e6;
        --insurance-tag-text: #cc5500;
        --insurance-tag-border: #ffcc99;
        --chat-input-bg: rgba(255,255,255,0.9);
        --metric-bg: #f8f9fa;
        --metric-border: #e6e6e6;
    }
    
    /* Dark theme variables */
    [data-theme="dark"] {
        --background-color: #0e1117;
        --user-msg-bg: #1e3a5f;
        --assistant-msg-bg: #2d2d2d;
        --button-bg: #4f8bf9;
        --button-text: white;
        --button-border: #4f8bf9;
        --button-hover-bg: #3a7de8;
        --expander-color: #4f8bf9;
        --expander-content-bg: #1a1a1a;
        --border-color: #3d3d3d;
        --header-color: #a3b9e0;
        --card-bg: #1a1a1a;
        --card-border: #3d3d3d;
        --status-active: #28a745;
        --status-inactive: #dc3545;
        --salary-tag-bg: #1e3a5f;
        --salary-tag-text: #99c2ff;
        --salary-tag-border: #4f8bf9;
        --insurance-tag-bg: #5c3d17;
        --insurance-tag-text: #ffcc99;
        --insurance-tag-border: #cc9966;
        --chat-input-bg: rgba(14,17,23,0.9);
        --metric-bg: #1a1a1a;
        --metric-border: #3d3d3d;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--button-bg);
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# App-wide constants & templates
# ----------------------------

DEFAULT_SALARY_TEXT = """
Salaries are typically structured with a monthly pay that sums to an annual total.
Annual salary = monthly salary × 12. Deductions may include income tax, provident fund,
professional tax, and other withholdings. Gross salary is the total before deductions.
Net (take-home) salary = Gross salary − total deductions. Some companies pay bonuses
or allowances (HRA, travel) that can be taxable or partially exempt.
"""

DEFAULT_INSURANCE_TEXT = """
An insurance policy usually defines coverage, premiums, and the claim process.
Coverage may include hospitalization, outpatient care, and prescription medicines.
The premium is the amount you pay periodically (monthly or yearly). Deductibles or
co-pays can apply. To file a claim: notify the insurer, submit required documents
(bills, prescriptions, discharge summary), and follow the approval process. Exclusions
(e.g., cosmetic procedures) are not covered.
"""

SALARY_AGENT_SYSTEM = (
    "You are the Salary Agent. ONLY answer questions about salary structure, monthly vs annual, deductions, gross vs net, allowances, bonuses, taxes, and payroll calculations. If the user asks about insurance or anything else, say briefly that you only handle salary topics, and suggest they ask the Insurance Agent."
)

INSURANCE_AGENT_SYSTEM = (
    "You are the Insurance Agent. ONLY answer questions about insurance coverage, premiums, claims, deductibles/co-pays, and exclusions. If the user asks about salary or anything else, say briefly that you only handle insurance topics, and suggest they ask the Salary Agent."
)

COORDINATOR_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        """
You are a routing classifier. Read the user question and output one token exactly: SALARY or INSURANCE.
Choose SALARY for salary/payroll/deductions/compensation math. Choose INSURANCE for policy/coverage/premium/claims.
Question: {question}
Answer with exactly one word: SALARY or INSURANCE.
        """.strip()
    ),
)

# Answer formatting template for RAG agents
RAG_ANSWER_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "system"],
    template=(
        """
{system}
Use the provided context to answer concisely and helpfully. If the context lacks details, say so and provide a best-effort general answer within your domain.

Context:
{context}

Question: {question}
Helpful answer:
        """.strip()
    ),
)


# ----------------------------
# Utility: Build / update the vector store
# ----------------------------

def build_vector_store(texts: List[str], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs: List[Document] = []
    for t in texts:
        for chunk in splitter.split_text(t):
            docs.append(Document(page_content=chunk))
    vs = FAISS.from_documents(docs, embedding=embeddings)
    return vs


# ----------------------------
# Coordinator (router)
# ----------------------------

def classify_query(llm: ChatGoogleGenerativeAI, question: str) -> str:
    """Return 'SALARY' or 'INSURANCE' using a minimal LLM call. Falls back to keywords."""
    try:
        msg = COORDINATOR_CLASSIFIER_PROMPT.format(question=question)
        out = llm.invoke(msg)
        label = (out.content or "").strip().upper()
        if label.startswith("SALARY"):
            return "SALARY"
        if label.startswith("INSURANCE"):
            return "INSURANCE"
    except Exception:
        pass
    # Fallback keyword heuristic
    q = question.lower()
    salary_kw = ["salary", "pay", "annual", "monthly", "deduction", "pf", "allowance", "bonus", "ctc", "take-home", "net", "gross"]
    insurance_kw = ["insurance", "policy", "premium", "coverage", "claim", "cashless", "deductible", "co-pay", "exclusion"]
    if any(k in q for k in salary_kw) and not any(k in q for k in insurance_kw):
        return "SALARY"
    return "INSURANCE"


# ----------------------------
# Agent builders
# ----------------------------

def build_rag_chain(llm: ChatGoogleGenerativeAI, retriever, system_prompt: str) -> RetrievalQA:
    # LangChain RetrievalQA with a custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": RAG_ANSWER_TEMPLATE.partial(system=system_prompt),
        },
        return_source_documents=True,
    )
    return qa_chain


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Multi-Agent RAG (Gemini): Salary + Insurance", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Multi-Agent RAG System with Gemini AI"
    }
)
load_dotenv()

# Inject custom CSS
inject_custom_css()

with st.sidebar:
    st.header("⚙️ Configuration")
    
    
    # API key status
    api_status = "active" if os.getenv("GOOGLE_API_KEY") else "inactive"
    st.markdown(f"""
    <div class="card">
        <h4>API Status <span class="status-indicator status-{api_status}"></span></h4>
        <p>Google Generative AI API key is <strong>{'configured' if os.getenv("GOOGLE_API_KEY") else 'missing'}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file or environment.")

    st.divider()
    st.subheader("📚 Knowledge Base")
    
    st.markdown("""
    <div class="card">
        <p>Upload your knowledge files or use the built-in sample content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    salary_file = st.file_uploader("Salary Documentation", type=["txt"], key="salary_upl", 
                                  help="Upload a text file with salary information")
    insurance_file = st.file_uploader("Insurance Documentation", type=["txt"], key="ins_upl", 
                                     help="Upload a text file with insurance information")

    use_defaults = st.checkbox("Use built-in sample content", value=True, 
                              help="Use default knowledge base if files aren't uploaded")

    if st.button("🔄 Build/Update Knowledge Base", type="primary", use_container_width=True):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide GOOGLE_API_KEY in the .env file to build the vector store.")
        else:
            sal_text = DEFAULT_SALARY_TEXT
            ins_text = DEFAULT_INSURANCE_TEXT
            if salary_file is not None:
                sal_text = salary_file.read().decode("utf-8", errors="ignore")
            elif not use_defaults:
                st.warning("Salary file not provided; using empty content.")
                sal_text = ""

            if insurance_file is not None:
                ins_text = insurance_file.read().decode("utf-8", errors="ignore")
            elif not use_defaults:
                st.warning("Insurance file not provided; using empty content.")
                ins_text = ""

            try:
                with st.spinner("Building knowledge base..."):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vs = build_vector_store([sal_text, ins_text], embeddings)
                    st.session_state["vectorstore"] = vs
                    st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": 4})
                
                st.success("Knowledge base updated successfully! ✅")
                
                # Show file status
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Salary Content", "Uploaded" if salary_file else "Default" if use_defaults else "None")
                with col2:
                    st.metric("Insurance Content", "Uploaded" if insurance_file else "Default" if use_defaults else "None")

            except Exception as e:
                st.error(f"Error building knowledge base: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role, content}
if "vectorstore" not in st.session_state:
    # lazy-build a default VS so the app is usable immediately
    try:
        if os.getenv("GOOGLE_API_KEY"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vs = build_vector_store([DEFAULT_SALARY_TEXT, DEFAULT_INSURANCE_TEXT], embeddings)
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": 4})
    except Exception:
        pass

# Main content area
st.title("🤖 Multi-Agent RAG System")
st.caption("Two specialized AI agents working together with Retrieval-Augmented Generation")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 System Info", "🛠️ Settings"])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col2:
        st.subheader("🧪 Try These Examples")
        
        st.markdown("""
        <div class="card">
            <p>Click a question to test the system:</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💵 How do I calculate annual salary from monthly?", use_container_width=True):
            st.session_state["prefill"] = "How do I calculate annual salary from monthly?"
        if st.button("🏥 What's typically covered in health insurance?", use_container_width=True):
            st.session_state["prefill"] = "What's typically covered in health insurance?"
        if st.button("📊 What's the difference between gross and net salary?", use_container_width=True):
            st.session_state["prefill"] = "What's the difference between gross and net salary?"
        if st.button("🩺 How do I file an insurance claim?", use_container_width=True):
            st.session_state["prefill"] = "How do I file an insurance claim?"
        
        st.divider()
        
        # System status
        st.subheader("📈 System Status")
        
        vs_status = "active" if "vectorstore" in st.session_state else "inactive"
        api_status = "active" if os.getenv("GOOGLE_API_KEY") else "inactive"
        
        st.markdown(f"""
        <div class="card">
            <p><span class="status-indicator status-{vs_status}"></span> Knowledge Base: <strong>{'Loaded' if vs_status == 'active' else 'Not ready'}</strong></p>
            <p><span class="status-indicator status-{api_status}"></span> API Connection: <strong>{'Active' if api_status == 'active' else 'Inactive'}</strong></p>
            <p><span class="status-indicator status-active"></span> Coordinator: <strong>Ready</strong></p>
            <p><span class="status-indicator status-active"></span> Salary Agent: <strong>Ready</strong></p>
            <p><span class="status-indicator status-active"></span> Insurance Agent: <strong>Ready</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.subheader("💬 Chat with the Agents")
        
        # Show chat history with improved styling
        chat_container = st.container()
        with chat_container:
            for m in st.session_state["messages"]:
                with st.chat_message(m["role"]):
                    # Add agent tag for assistant messages
                    if m["role"] == "assistant":
                        # Try to detect which agent responded
                        if "salary" in m["content"].lower() and "insurance" not in m["content"].lower():
                            st.markdown(f"<span class='agent-tag salary-tag'>Salary Agent</span>", unsafe_allow_html=True)
                        elif "insurance" in m["content"].lower() and "salary" not in m["content"].lower():
                            st.markdown(f"<span class='agent-tag insurance-tag'>Insurance Agent</span>", unsafe_allow_html=True)
                    st.markdown(m["content"])

        prefill = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
        user_input = st.chat_input(placeholder=prefill or "Ask about salary or insurance...", key="chat_input")
        if prefill and not user_input:
            # allow user to just press Enter
            user_input = prefill

        if user_input:
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("Please provide GOOGLE_API_KEY in the .env file to ask questions.")
            else:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                
                # Re-render the chat to show the new message
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(user_input)

                # Build LLM and agents lazily when needed
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

                retriever = st.session_state.get("retriever")
                if retriever is None:
                    try:
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        vs = build_vector_store([DEFAULT_SALARY_TEXT, DEFAULT_INSURANCE_TEXT], embeddings)
                        retriever = vs.as_retriever(search_kwargs={"k": 4})
                        st.session_state["vectorstore"] = vs
                        st.session_state["retriever"] = retriever
                    except Exception as e:
                        st.error(f"Error initializing knowledge base: {str(e)}")
                        st.stop()

                salary_agent = build_rag_chain(llm, retriever, SALARY_AGENT_SYSTEM)
                insurance_agent = build_rag_chain(llm, retriever, INSURANCE_AGENT_SYSTEM)

                # Route the query
                with st.spinner("Analyzing your question..."):
                    route = classify_query(llm, user_input)
                    
                # Show which agent is responding
                if route == "SALARY":
                    agent_tag = "salary-tag"
                    agent_name = "Salary Agent"
                else:
                    agent_tag = "insurance-tag"
                    agent_name = "Insurance Agent"
                
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(f"<span class='agent-tag {agent_tag}'>{agent_name}</span>", unsafe_allow_html=True)
                        
                        with st.spinner(f"Consulting {agent_name}..."):
                            if route == "SALARY":
                                res = salary_agent({"query": user_input})
                            else:
                                res = insurance_agent({"query": user_input})

                            answer = res.get("result", "Sorry, I couldn't generate an answer.")
                            st.markdown(answer)

                            # Show retrieved context in an expandable section
                            with st.expander("View source materials used for this response"):
                                srcs = res.get("source_documents", []) or []
                                if not srcs:
                                    st.info("No specific sources were retrieved for this question.")
                                for i, d in enumerate(srcs, 1):
                                    st.markdown(f"**Excerpt {i}:**")
                                    st.info(d.page_content)

                st.session_state["messages"].append({"role": "assistant", "content": answer})
                
                # Scroll to bottom after new message
                st.rerun()

with tab2:
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📋 About This System</h3>
            <p>This application demonstrates a multi-agent RAG (Retrieval-Augmented Generation) system with:</p>
            <ul>
                <li>Two specialized agents for salary and insurance queries</li>
                <li>Automatic routing based on question content</li>
                <li>Knowledge retrieval from uploaded documents</li>
                <li>Conversational memory within each session</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>🔧 How It Works</h3>
            <ol>
                <li>User asks a question</li>
                <li>Coordinator classifies the question as Salary or Insurance related</li>
                <li>Appropriate agent is selected</li>
                <li>Agent searches knowledge base for relevant information</li>
                <li>Agent formulates response using retrieved context</li>
                <li>Response is displayed to user with source materials</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎯 Agent Capabilities</h3>
            <h4>Salary Agent</h4>
            <ul>
                <li>Salary structure and calculations</li>
                <li>Monthly vs annual conversions</li>
                <li>Deductions and taxes</li>
                <li>Gross vs net pay</li>
                <li>Allowances and bonuses</li>
            </ul>
            
            <h4>Insurance Agent</h4>
            <ul>
                <li>Policy coverage details</li>
                <li>Premium calculations</li>
                <li>Claims process</li>
                <li>Deductibles and co-pays</li>
                <li>Exclusions and limitations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("Settings & Configuration")
    
    st.markdown("""
    <div class="card">
        <h3>⚙️ Advanced Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model settings
        st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"], index=0,
                    help="Select which Gemini model to use")
        
        st.slider("Temperature", 0.0, 1.0, 0.2, 0.1,
                 help="Controls randomness of responses. Lower = more deterministic")
        
    with col2:
        # Retrieval settings
        st.slider("Retrieval chunks", 1, 10, 4, 1,
                 help="Number of text chunks to retrieve for context")
        
        st.slider("Chunk size", 200, 1200, 800, 100,
                 help="Size of text chunks for the knowledge base")
    
    st.divider()
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    
    # Export chat history
    if st.button("📥 Export Chat History", use_container_width=True):
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state["messages"]])
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name="rag_chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("No chat history to export")

st.divider()

st.markdown("""
<div class="card" style="text-align: center;">
    <p>Built with ❤️ using Streamlit, LangChain, and Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)