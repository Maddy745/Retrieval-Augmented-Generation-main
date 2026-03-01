import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough
from vecre import get_retriever
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

# Set API Key
api_key = "AIzaSyCZc-UrYeqg8Sa5FxFr9Y7Ut_OH2TsiyUc"
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Embeddings globally to avoid reloading for every similarity check
from langchain_huggingface import HuggingFaceEmbeddings
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def semantic_similarity(text1, text2):
    """Calculates cosine similarity between embeddings of two texts."""
    emb1 = eval_embeddings.embed_query(text1)
    emb2 = eval_embeddings.embed_query(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

def evaluate_answer(answer, context):
    """Evaluates an answer against the reference context using ROUGE and Similarity."""
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # Treating context as the 'reference'
    scores = scorer.score(context, answer)
    
    # Semantic Similarity
    similarity = semantic_similarity(answer, context)
    
    return scores, similarity

def clean_content(content):
    """
    Parses the content to return a clean string.
    Handles string, list of strings, and list of dicts (Gemini response format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    return str(content)

import time
import random

import re

def retry_invoke(chain, input_data):
    """Retries the chain invocation with smart backoff upon 429 errors."""
    max_retries = 3
    base_delay = 10 
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Try to extract wait time from error message
                match = re.search(r"retry in (\d+(\.\d+)?)s", str(e))
                if match:
                    delay = float(match.group(1)) + 1.0 # Add 1s buffer
                    print(f"Rate limit hit. API requests to wait {delay:.2f}s...")
                else:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit. Retrying in {delay:.2f}s...")
                
                time.sleep(delay)
            else:
                raise e
    return chain.invoke(input_data) # Final attempt

def main():
    print("Initializing RAG System...")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Initialize Retriever
    # Using k=5 to retrieve more context for better accuracy
    retriever = get_retriever(k=5)
    if not retriever:
        return

    # RAG Chain
    rag_template = """You are a helpful assistant. Use ONLY the following pieces of context to answer the question at the end.
    Do not use your own internal knowledge. If the answer is not in the context, say "I cannot find the answer in the document."
    Keep the answer very simple, concise, and to the point. Maximum 2-3 sentences.
    
    Context: {context}
    
    Question: {question}
    Answer:"""
    rag_prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])
    
    # Updated RAG Chain to take context as string
    rag_chain = (
        rag_prompt
        | llm
    )

    # Non-RAG Chain (Simple Generation)
    non_rag_template = """Answer the following question based on your general knowledge.
    Keep the answer very simple, concise, and to the point. Maximum 2-3 sentences.
    
    Question: {question}
    Answer:"""
    non_rag_prompt = PromptTemplate(template=non_rag_template, input_variables=["question"])
    non_rag_chain = non_rag_prompt | llm

    print("\nSystem Ready! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            break

        print("\n--- Generating Answers ---")
        
        # Get RAG Response
        print("Fetching Context and Generating RAG Answer...")
        rag_answer = ""
        context_str = ""
        try:
            # Manually retrieve context to use for evaluation
            docs = retriever.invoke(question) # Using .invoke() for best practice
            context_str = "\n".join([doc.page_content for doc in docs])
            
            rag_response = retry_invoke(rag_chain, {"context": context_str, "question": question})
            raw_content = rag_response.content if hasattr(rag_response, 'content') else str(rag_response)
            rag_answer = clean_content(raw_content)
        except Exception as e:
            rag_answer = f"Error generating RAG answer: {e}"
        
        # Get Non-RAG Response
        print("Fetching Non-RAG Answer...")
        non_rag_answer = ""
        try:
            non_rag_response = retry_invoke(non_rag_chain, {"question": question})
            raw_content = non_rag_response.content if hasattr(non_rag_response, 'content') else str(non_rag_response)
            non_rag_answer = clean_content(raw_content)
        except Exception as e:
            non_rag_answer = f"Error generating Non-RAG answer: {e}"

        print(f"\n[Retrieved Context (First 200 chars)]:\n{context_str[:200]}...\n")
        print(f"[RAG Answer]:\n{rag_answer}\n")
        print(f"[Non-RAG Answer]:\n{non_rag_answer}\n")
        
        # Evaluation
        if not context_str or "Error generating" in rag_answer or "Error generating" in non_rag_answer:
            print("--- Evaluation Skipped due to Generation Error or Missing Context ---")
        else:
            print("--- Evaluation (Against Context) ---")
            try:
                print("\n[RAG vs Context]")
                rag_scores, rag_sim = evaluate_answer(rag_answer, context_str)
                print(f"  Semantic Similarity: {rag_sim:.4f}")
                print(f"  ROUGE-1 F1: {rag_scores['rouge1'].fmeasure:.4f}")
                print(f"  ROUGE-L F1: {rag_scores['rougeL'].fmeasure:.4f}")

                print("\n[Non-RAG vs Context]")
                non_rag_scores, non_rag_sim = evaluate_answer(non_rag_answer, context_str)
                print(f"  Semantic Similarity: {non_rag_sim:.4f}")
                print(f"  ROUGE-1 F1: {non_rag_scores['rouge1'].fmeasure:.4f}")
                print(f"  ROUGE-L F1: {non_rag_scores['rougeL'].fmeasure:.4f}")
            except Exception as e:
                print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
