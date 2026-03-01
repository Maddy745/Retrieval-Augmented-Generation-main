import os
from langchain_community.vectorstores import FAISS

def get_retriever(index_path="faiss_index", k=3):
    """
    Loads the FAISS index and returns a retriever object.
    """
    if not os.path.exists(index_path):
        print(f"Error: Index path {index_path} not found. Please run vec.py first.")
        return None

    # Load embeddings using Sentence-BERT
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the vector store
    # allow_dangerous_deserialization is needed for loading local pickle files created by older versions or manually
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k}) # Retrieve top k relevant chunks
    return retriever

if __name__ == "__main__":
    retriever = get_retriever()
    if retriever:
        print("Retriever loaded successfully.")
