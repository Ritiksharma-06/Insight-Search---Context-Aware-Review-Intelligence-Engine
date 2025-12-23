import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# -------------------------------
# 1. API Key & Pinecone Setup
# -------------------------------
PINECONE_API_KEY = "pcsk_4Bsv1R_2W4YXEMLjaKMziE3wbZjWb4pkrUW5VxHqdHx5sBMbTChaBRotFtxmtiveCjYa8L"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "amazon-reviews-index"

# -------------------------------
# 2. Load Embedding Model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 3. Clean & Embed Text
# -------------------------------
def clean_text(text):
    return re.sub(r"<.*?>", "", str(text))

def get_embedding(text):
    return model.encode(clean_text(text)).tolist()

def shorten_review(text, max_words=60):
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def get_relevance(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# 4. Search Function
# -------------------------------
def search_reviews(query, top_k=5, summarize=True):
    query_embedding = get_embedding(query)
    index = pc.Index(index_name)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    processed_results = []
    for match in results['matches']:
        score = match['score']
        if score < 0.6:
            continue
        text = match['metadata']['text']
        if summarize:
            text = shorten_review(text)
        processed_results.append({
            "Score": round(score, 2),
            "Relevance": get_relevance(score),
            "Review": text
        })
    return processed_results

# -------------------------------
# 5. Save Search Logs
# -------------------------------
def save_search_log(query, results, log_file="search_logs.csv"):
    if '/' in log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "query", "score", "relevance", "review"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            writer.writerow([timestamp, query, r["Score"], r["Relevance"], r["Review"]])

# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Semantic Search - Food Reviews", layout="centered")
st.title("🍪 context-aware-review-intelligence")

query = st.text_input("Enter your query:")
top_k = st.slider("Number of results", 1, 10, 5)
summarize = st.checkbox("Summarize long reviews", value=True)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = search_reviews(query, top_k=top_k, summarize=summarize)
        if results:
            save_search_log(query, results)
            st.success(f"Found {len(results)} relevant results:")
            for r in results:
                st.markdown(f"""
                **Score**: {r['Score']} ({r['Relevance']})  
                **Review**: {r['Review']}  
                ---
                """)
        else:
            st.warning("No relevant results found.")

# -------------------------------
# 7. Show Past Logs (Optional)
# -------------------------------
if st.checkbox("📄 Show past queries"):
    try:
        logs_df = pd.read_csv("search_logs.csv")
        st.dataframe(logs_df.tail(10))  # Show last 10 searches
    except FileNotFoundError:
        st.info("No search logs found yet.")
