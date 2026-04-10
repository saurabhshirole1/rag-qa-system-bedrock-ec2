
# --- Standard Python libraries ---
import os           
import tempfile     
import streamlit as st
from bedrock_rag import query_knowledge_base, upload_document_to_s3
from config import APP_TITLE, APP_ICON, S3_BUCKET_NAME

# SECTION 1: Page Configuration
st.set_page_config(
    page_title=APP_TITLE,         
    page_icon=APP_ICON,           
    layout="wide",                
    initial_sidebar_state="expanded"  
)

# SECTION 2: Custom CSS Styling
st.markdown("""
    <style>
        /* Style the main chat container */
        .main { padding-top: 2rem; }
        
        /* Style citation boxes */
        .citation-box {
            background-color: #f0f2f6;
            border-left: 4px solid #FF6B35;
            padding: 10px;
            margin: 5px 0;
            border-radius: 0 5px 5px 0;
            font-size: 0.85em;
        }
        
        /* Style the header */
        .header-text { color: #1a1a2e; }
    </style>
""", unsafe_allow_html=True)

# SECTION 3: Sidebar

with st.sidebar:
    
    st.title("Settings & Info")
    st.divider() 
    st.subheader("Knowledge Base")
    st.info(f"S3 Bucket: `{S3_BUCKET_NAME}`")
    
    st.divider()
    
    # --- Document Upload Section ---
    st.subheader("Upload Documents")
    st.write("Upload new documents to add to the knowledge base.")
    
    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=["pdf", "txt", "docx"],  
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_file is not None:
        if st.button("Upload to S3"):
            with st.spinner("Uploading..."):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                success = upload_document_to_s3(temp_path, uploaded_file.name)
                
                if success:
                    st.success(f"Uploaded! Now sync your Bedrock KB.")
                else:
                    st.error("Upload failed. Check logs.")
    
    st.divider()
    st.subheader("How to use")
    st.markdown("""
    1. Type your question below
    2. Press Enter or click Send
    3. View the AI answer
    4. Expand 'Sources' to see citations
    """)
    
    # --- Clear chat button ---
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# SECTION 4: Main Page Header
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Ask questions about company documents — get accurate, cited answers powered by Amazon Bedrock RAG.")
# SECTION 5: Session State — Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
# SECTION 6: Display All Previous Chat Messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# SECTION 7: Chat Input Box
if question := st.chat_input("Ask a question about company documents..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # --- Query Bedrock and display answer ---
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):

            result = query_knowledge_base(question)
        
        if result["success"]:
            st.markdown(result["answer"])
            
            if result["citations"]:
                with st.expander(f"📎 View Sources ({len(result['citations'])} documents used)"):
                    for i, citation in enumerate(result["citations"], start=1):
                        source_path = citation["source"]
                        file_name   = source_path.split("/")[-1]
                        st.markdown(f"**Source {i}: `{file_name}`**")
                        st.markdown(f"> {citation['excerpt']}")
                        
                        if i < len(result["citations"]):
                            st.divider()  
            else:
                st.warning("No specific citations found for this answer.")
        
        else:
            # Something went wrong — show the error
            st.error(f"Error: {result['answer']}")
    
    # --- Save assistant response to session state ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })