from sentence_transformers import SentenceTransformer
import numpy as np
from ingestion import build_novel_rows
import pickle

class NovelVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loaded embedding model: {model_name}...")
        self.model= SentenceTransformer(model_name)
        self.chunks= []
        self.embeddings = None
        self.metadata=[]

    def build_from_novels(self):
        print("Loding and chuking novels..")
        rows= build_novel_rows()

        for book, chunk_id, content in rows:
            self.chunks.append(content)
            self.metadata.append({
                "book":book,
                "chunk_id": chunk_id,
                "content" : content
            })

        print(f"Creating embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"Vector store built with {len(self.chunks)} chunks")

    def search(self, query, book_name=None, k=10):

        query_embedding = self.model.encode([query])[0]

        if book_name:
            indices = [i for i, m in enumerate(self.metadata) if m['book'] == book_name]
            filtered_embeddings = self.embeddings[indices]
            filtered_metadata = [self.metadata[i] for i in indices]
        else:
            filtered_embeddings = self.embeddings
            filtered_metadata = self.metadata

        similarity = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)

        )

        top_indices = np.argsort(similarity)[-k:][::-1]

        results = []
        for idx in top_indices:
            result = filtered_metadata[idx].copy()
            result['similarity'] = float(similarity[idx])
            results.append(result)

        return results
    
    def save(self, path="Data/vector_store.pkl"):
        print(f"saving vector store to {path}")
        with open(path, 'wb') as f:
            pickle.dump({
                "chunks" : self.chunks,
                "embeddings" : self.embeddings,
                "metadata" : self.metadata
            },f)
        print("saved")

    def load(self, path="Data/vector_store.pkl"):
        print(f"Loading vector store from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        print(f"Loaded {len(self.chunks)} chunks")  

from pathlib import Path

def test_retrieval():
    """Test the retrieval system"""
    store = NovelVectorStore()
    
    if Path("Data/vector_store.pkl").exists():
        store.load()
    else:
        store.build_from_novels()
        store.save()
    
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    test_query = "character's childhood and early life experiences"
    print(f"\nQuery: {test_query}")
    print(f"Searching in Monte Cristo...\n")
    
    results = store.search(test_query, book_name='monte', k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (similarity: {result['similarity']:.3f}) ---")
        print(f"Book: {result['book']}, Chunk: {result['chunk_id']}")
        print(f"Content preview: {result['content'][:200]}...")


if __name__ == "__main__":
    test_retrieval()






        