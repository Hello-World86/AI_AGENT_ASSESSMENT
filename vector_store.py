from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag import POLICY_DOCUMENTS
import numpy as np

#Initialize vectorizer and compute embeddings for policy documents
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

#Convert policy documents into a TF-IDF vectors
policy_texts = [doc["content"] for doc in POLICY_DOCUMENTS]
policy_vectors = vectorizer.fit_transform(policy_texts)

def search(query, k=1):
    """Search for the most relevant policy document based on cosine similarity."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, policy_vectors)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    # Get indices of top-k most similar documents
    I = np.argsort(similarities)[::-1][:k]
    
    return [POLICY_DOCUMENTS[i] for i in I]