from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
"Standard customers have a 30-day return window and get a $20 credit for delays.",
"VIP customers have a 60-day return window and get full refunds for delays or damage."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


def search(query):

    query_embedding = model.encode([query])

    _, I = index.search(np.array(query_embedding), k=1)

    return documents[I[0][0]]