# The vector store is based on TF-IDF policy document search

import logging
import numpy as np
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Policy documents stored directly here to avoid circular import with rag.py
POLICY_DOCUMENTS = [
    {
        "id": "standard_return_policy",
        "content": (
            "Standard customers have a 30-day return window and the return window begins from the delivery date. "
            "If a Standard customer's order gets delayed, they will receive a $20 credit as compensation."
        ),
        "metadata": {"category": "return_policy", "tier": "Standard"},
    },
    {
        "id": "vip_return_policy",
        "content": (
            "VIP customers have a 60-day return window. "
            "If a VIP customer's order is delayed or damaged by any means, they will receive a full refund as compensation. "
            "The return window for VIP customers also begins from the delivery date."
        ),
        "metadata": {"category": "return_policy", "tier": "VIP"},
    },
]

#Computing TF-IDF vectors for the policy documents at module load time. This allows us to have a simple vector store for retrieval without needing an external database or service.
try:
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=1,
        max_df=0.95
    )

    policy_texts = [doc.get("content", "") for doc in POLICY_DOCUMENTS]

    if not policy_texts or all(not text for text in policy_texts):
        logger.error("No valid policy documents found")
        policy_vectors = None
    else:
        policy_vectors = vectorizer.fit_transform(policy_texts)
        logger.info(f"Vectorizer initialized with {len(POLICY_DOCUMENTS)} policy documents")

except Exception as e:
    logger.error(f"Error initializing vectorizer: {str(e)}")
    vectorizer = None
    policy_vectors = None


def search(query: str, k: int = 3) -> Dict[str, any]:
    """Searches for the most relevant policy documents based on cosine similarity."""
    try:
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to search")
            return {"error": "Query must be a non-empty string"}

        if not vectorizer or policy_vectors is None:
            logger.error("Vectorizer not initialized")
            return {"error": "Vector store not initialized"}

        query = query.strip()
        if not query:
            return {"error": "Query cannot be empty"}

        logger.debug(f"Searching for: {query}")

        # Transform query and compute similarities
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, policy_vectors)[0]

        # Get indices of top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:k]

        # Filter out results with very low similarity
        min_similarity_threshold = 0.0
        relevant_docs = []

        for idx in top_indices:
            if similarities[idx] >= min_similarity_threshold:
                relevant_docs.append(POLICY_DOCUMENTS[idx])

        if not relevant_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            return {"context": "No matching policies found", "similarity_score": 0.0}

        # Concatenate content from top results
        combined_context = "\n\n".join([
            doc.get("content", "") for doc in relevant_docs
        ])

        logger.info(f"Found {len(relevant_docs)} relevant policies for query")

        return {
            "context": combined_context,
            "documents_count": len(relevant_docs),
            "top_similarity_score": float(similarities[top_indices[0]])
        }

    except ValueError as e:
        logger.error(f"Vectorization error: {str(e)}")
        return {"error": f"Search failed: {str(e)}"}

    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}", exc_info=True)
        return {"error": f"Search error: {str(e)}"}
