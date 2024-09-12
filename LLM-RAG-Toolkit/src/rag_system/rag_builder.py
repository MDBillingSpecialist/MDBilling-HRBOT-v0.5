from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import logging

logger = logging.getLogger(__name__)

class RAGBuilder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None

    def build_rag_system(self, processed_document: Dict[str, Any]) -> Dict[str, Any]:
        chunks = processed_document['segments']
        embeddings = self.create_embeddings(chunks)
        self.build_index(embeddings)
        
        return {
            "file_path": processed_document.get('file_path', 'Unknown'),
            "chunks": chunks,
            "embeddings": embeddings,
            "index": self.index,
            "metadata": processed_document.get('metadata', {}),
            "toc": processed_document.get('toc', {}),
            "semantic_data": processed_document.get('semantic_data', {})
        }

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [chunk['content'] for chunk in chunks]
        return self.model.encode(texts)

    def build_index(self, embeddings: np.ndarray):
        self.index = IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [{"chunk_id": int(i), "distance": float(d)} for i, d in zip(indices[0], distances[0])]

    def get_relevant_chunks(self, query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        retrieved = self.retrieve(query, k)
        return [chunks[result['chunk_id']] for result in retrieved]