import os
import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


# --------------------
# 1. Set Pinecone API Key and initialize client
# --------------------
PINECONE_API_KEY = 'pcsk_4Bsv1R_2W4YXEMLjaKMziE3wbZjWb4pkrUW5VxHqdHx5sBMbTChaBRotFtxmtiveCjYa8L'
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'amazon-reviews-index'

# --------------------
# 2. Load CSV file with data
# --------------------
csv_file = 'Reviews.csv'
df = pd.read_csv(csv_file)

if 'Text' not in df.columns:
    raise ValueError("CSV file must contain a 'Text' column")

df = df.dropna(subset=['Text']).sample(n=2000, random_state=42).reset_index(drop=True)

# --------------------
# 3. Load sentence-transformers model
# --------------------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Switched to faster model

# --------------------
# 4. Clean text
# --------------------
def clean_text(text):
    return re.sub(r'<.*?>', '', str(text))

def get_embedding(text):
    return model.encode(clean_text(text)).tolist()

# --------------------
# 5. Create Pinecone index if it does not exist
# --------------------
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # for all-MiniLM-L6-v2
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# --------------------
# 6. Generate embeddings and prepare data for Pinecone upsert
# --------------------
print("Generating embeddings...")
df['clean_text'] = df['Text'].apply(clean_text)
df['embedding'] = df['clean_text'].apply(get_embedding)

to_upsert = [
    {
        'id': str(i),
        'values': row.embedding,
        'metadata': {'text': row.clean_text}
    }
    for i, row in df.iterrows()
]

# --------------------
# 7. Upsert data to Pinecone in batches
# --------------------
print("Uploading embeddings to Pinecone...")
batch_size = 100
for i in tqdm(range(0, len(to_upsert), batch_size)):
    batch = to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

print("Upload complete.")

# --------------------
# 8. Add relevance label
# --------------------
def get_relevance(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

# --------------------
# 9. Shorten long reviews (optional)
# --------------------
def shorten_review(text, max_words=60):
    words = text.split()
    return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')

# --------------------
# 10. Semantic search function
# --------------------
def search_reviews(query, top_k=5, summarize=True):
    print(f"\n🔎 Searching for: {query}\n")
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    for match in results['matches']:
        score = match['score']
        if score < 0.6:
            continue
        text = match['metadata']['text']
        if summarize:
            text = shorten_review(text)
        print(f"Score: {score:.2f} ({get_relevance(score)})")
        print(f"Review: {text}")
        print("-" * 80)

# --------------------
# 11. Example query
# --------------------
if __name__ == "__main__":
    search_reviews("Does users like chips?", top_k=5)
