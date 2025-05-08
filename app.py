import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from PyPDF2 import PdfReader

# Initialize SentenceTransformer and Groq client
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#client = Groq(api_key="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")                  this is used when you run on your system.If you want to deploy on streamlit then use following code
api_key=st.secrets['key']
client = Groq(api_key=api_key)


# Global variables
documents = []
document_embeddings = []

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Function to split long text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Update knowledge base
def update_knowledge_base(text):
    global documents, document_embeddings
    chunks = split_text_into_chunks(text)
    documents.extend(chunks)
    document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Retrieve relevant context from stored embeddings
def retrieve(query, top_k=1):
    if not documents or len(document_embeddings) == 0:
        return None
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)
    return documents[hits[0][0]['corpus_id']] if hits and hits[0] else None

# Generate response using Groq's LLM
def generate_response(query, context):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # or try "mixtral-8x7b" or "llama3-8b-8192" instead of gemma-7b-it
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="📄 PDF Chatbot with Groq", layout="wide")
st.title("📄 PDF Chatbot using Groq + RAG")
st.markdown("Upload a PDF file and ask questions based on its content.")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("📚 Reading and indexing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        if text:
            update_knowledge_base(text)
            st.success("✅ PDF content processed and indexed.")
        else:
            st.error("❌ No text found in PDF.")

# Ask a question
question = st.text_input("💬 Ask your question here:")
if question:
    with st.spinner("🔎 Retrieving context..."):
        context = retrieve(question)
    if context:
        with st.spinner("🤖 Generating answer..."):
            answer = generate_response(question, context)
            st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("⚠️ No relevant context found to answer your question.")
