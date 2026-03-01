import os
import pypdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Set API Key (Note: API Key is used for Google GenAI in main.py, here we use local embeddings as requested fallback)

def create_embeddings(pdf_path="Machine Learning (1).pdf", index_path="faiss_index"):
    """
    Loads a PDF using pypdf, splits it into chunks, generates embeddings 
    using Sentence-BERT (all-MiniLM-L6-v2), and saves the FAISS index locally.
    """
    print(f"Loading PDF from {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"PDF loaded. Pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print(f"Extracted {len(text)} characters.")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    print("Splitting text into chunks...")
    
    # Custom splitting logic to avoid potential import hangs with complex splitters
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    if len(text) > 0:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
    else:
        print("Warning: No text extracted from PDF.")
        return
    
    # Create documents from text
    docs = [Document(page_content=chunk) for chunk in chunks]
    print(f"Split into {len(docs)} chunks.")

    print("Generating embeddings and creating FAISS index...")
    try:
        # Using Sentence-BERT embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Save the index locally
        vectorstore.save_local(index_path)
        print(f"FAISS index saved to {index_path}/")
    except Exception as e:
        print(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    create_embeddings()
