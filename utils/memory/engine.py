"""
TrialAgents Memory Bank - Search Engine

FAISS-based vector similarity search
Supports semantic search for ErrorCase and SuccessCase
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Optionally import FAISS, use fallback if unavailable
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using simple cosine similarity fallback.")


@dataclass
class SearchResult:
    """Search result"""
    case_id: str
    case_type: str                          # "error" or "success"
    similarity: float
    data: Any                               # ErrorCase or SuccessCase


class MemoryEngine:
    """
    Memory search engine

    Uses FAISS for efficient vector similarity search
    """

    def __init__(
        self,
        embedding_dim: int = 384,           # Default embedding dimension
        store=None,                          # MemoryStore instance
        index_path: str = None,
    ):
        """
        Initialize search engine

        Args:
            embedding_dim: Embedding vector dimension
            store: MemoryStore instance
            index_path: Index save path
        """
        self.embedding_dim = embedding_dim
        self.store = store
        self.embedding_model = None          # Lazy loading

        # Index path
        if index_path is None:
            from utils.data.paths import get_data_path
            index_path = os.path.join(get_data_path(), "memory", "indices")

        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)

        # FAISS index
        self.error_index = None
        self.success_index = None

        # ID mapping
        self.error_ids: List[str] = []
        self.success_ids: List[str] = []

        # Initialize or load index
        self._init_or_load_index()

    def _init_or_load_index(self):
        """Initialize or load index"""
        error_index_file = os.path.join(self.index_path, "error_index.faiss")
        success_index_file = os.path.join(self.index_path, "success_index.faiss")
        error_ids_file = os.path.join(self.index_path, "error_ids.json")
        success_ids_file = os.path.join(self.index_path, "success_ids.json")

        if FAISS_AVAILABLE:
            # Try to load existing index
            if os.path.exists(error_index_file) and os.path.exists(error_ids_file):
                try:
                    self.error_index = faiss.read_index(error_index_file)
                    with open(error_ids_file, "r") as f:
                        self.error_ids = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load error index: {e}")
                    self._create_new_indexes()

                # Check dimension match
                if self.error_index is not None and self.error_index.d != self.embedding_dim:
                    print(f"Warning: Error index dimension mismatch ({self.error_index.d} vs {self.embedding_dim}). Recreating.")
                    self._create_new_indexes()
            else:
                self._create_new_indexes()

            if os.path.exists(success_index_file) and os.path.exists(success_ids_file):
                try:
                    self.success_index = faiss.read_index(success_index_file)
                    with open(success_ids_file, "r") as f:
                        self.success_ids = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load success index: {e}")
            else:
                if self.success_index is None:
                    self.success_index = self._create_index()
        else:
            # Create simple index when FAISS is unavailable
            self._create_new_indexes()

    def _create_new_indexes(self):
        """Create new indices"""
        if FAISS_AVAILABLE:
            self.error_index = self._create_index()
            self.success_index = self._create_index()
            self.error_ids = []
            self.success_ids = []

    def _create_index(self):
        """Create FAISS index"""
        if FAISS_AVAILABLE:
            # Use inner product index (requires normalized embeddings)
            return faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        return None

    def _get_embedding_model(self):
        """Lazily load embedding model"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use lightweight model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not available. Using fallback embedding.")
                self.embedding_model = None
        return self.embedding_model

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector

        Args:
            text: Input text

        Returns:
            Embedding vector (embedding_dim,)
        """
        model = self._get_embedding_model()

        if model is not None:
            embedding = model.encode([text], normalize_embeddings=True)[0]
            return embedding.astype('float32')

        # Fallback: simple bag-of-words embedding
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding (term frequency vector)"""
        # Use fixed dimension
        dim = self.embedding_dim
        words = text.lower().split()
        vec = np.zeros(dim, dtype='float32')

        for i, word in enumerate(words[:dim]):
            vec[i % dim] += 1

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def _get_patient_text(self, case) -> str:
        """Extract search text from case"""
        if hasattr(case, 'patient_profile'):
            return case.patient_profile.to_text()
        return str(case)

    def add_error_case(self, error_case) -> bool:
        """
        Add error case to index

        Args:
            error_case: ErrorCase instance

        Returns:
            Whether successful
        """
        try:
            # Generate embedding
            text = self._get_patient_text(error_case)
            text += f" | Error type: {error_case.error_type.value}"
            text += f" | Root cause: {error_case.root_cause}"
            embedding = self._embed_text(text)

            # Get or assign ID
            if error_case.case_id in self.error_ids:
                idx_id = self.error_ids.index(error_case.case_id)
            else:
                idx_id = len(self.error_ids)
                self.error_ids.append(error_case.case_id)

            # Add to FAISS index
            if FAISS_AVAILABLE and self.error_index is not None:
                self.error_index.add_with_ids(
                    np.array([embedding]).astype('float32'),
                    np.array([idx_id])
                )

            return True
        except Exception as e:
            print(f"Error adding error case to index: {e}")
            return False

    def add_success_case(self, success_case) -> bool:
        """
        Add success case to index

        Args:
            success_case: SuccessCase instance

        Returns:
            Whether successful
        """
        try:
            # Generate embedding
            text = self._get_patient_text(success_case)
            text += f" | Reasoning: {success_case.reasoning_chain[:200]}"
            text += f" | Triggers: {', '.join(success_case.key_triggers[:3])}"
            embedding = self._embed_text(text)

            # Get or assign ID
            if success_case.case_id in self.success_ids:
                idx_id = self.success_ids.index(success_case.case_id)
            else:
                idx_id = len(self.success_ids)
                self.success_ids.append(success_case.case_id)

            # Add to FAISS index
            if FAISS_AVAILABLE and self.success_index is not None:
                self.success_index.add_with_ids(
                    np.array([embedding]).astype('float32'),
                    np.array([idx_id])
                )

            return True
        except Exception as e:
            print(f"Error adding success case to index: {e}")
            return False

    def search_similar_errors(
        self,
        patient_profile_text: str,
        k: int = 5,
        error_types: List[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar error cases

        Args:
            patient_profile_text: Patient feature text
            k: Number of results to return
            error_types: List of error types to filter by

        Returns:
            List of similar error cases
        """
        if self.store is None:
            return []

        # Generate query embedding
        query_embedding = self._embed_text(patient_profile_text)

        results = []

        if FAISS_AVAILABLE and self.error_index is not None and len(self.error_ids) > 0:
            # FAISS search
            search_k = min(k * 2, len(self.error_ids))  # Search more to allow filtering
            distances, indices = self.error_index.search(
                np.array([query_embedding]).astype('float32'),
                search_k
            )

            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.error_ids):
                    continue

                case_id = self.error_ids[idx]
                error_case = self.store.get_error(case_id)

                if error_case is None:
                    continue

                # Filter by error type
                if error_types and error_case.error_type.value not in error_types:
                    continue

                # Convert to similarity (0-1)
                similarity = float(dist) if dist >= 0 else 0.0
                similarity = min(1.0, max(0.0, similarity))

                results.append(SearchResult(
                    case_id=case_id,
                    case_type="error",
                    similarity=similarity,
                    data=error_case,
                ))

                if len(results) >= k:
                    break
        else:
            # Fallback: simple text matching
            all_errors = self.store.get_all_errors()

            for error_case in all_errors:
                if error_types and error_case.error_type.value not in error_types:
                    continue

                # Simple similarity calculation
                text1 = self._get_patient_text(error_case)
                sim = self._cosine_similarity_simple(patient_profile_text, text1)

                results.append(SearchResult(
                    case_id=error_case.case_id,
                    case_type="error",
                    similarity=sim,
                    data=error_case,
                ))

            # Sort and return top-k
            results.sort(key=lambda x: x.similarity, reverse=True)
            results = results[:k]

        return results

    def search_similar_successes(
        self,
        patient_profile_text: str,
        k: int = 3,
        min_confidence: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar success cases

        Args:
            patient_profile_text: Patient feature text
            k: Number of results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of similar success cases
        """
        if self.store is None:
            return []

        # Generate query embedding
        query_embedding = self._embed_text(patient_profile_text)

        results = []

        if FAISS_AVAILABLE and self.success_index is not None and len(self.success_ids) > 0:
            # FAISS search
            search_k = min(k * 2, len(self.success_ids))
            distances, indices = self.success_index.search(
                np.array([query_embedding]).astype('float32'),
                search_k
            )

            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.success_ids):
                    continue

                case_id = self.success_ids[idx]
                success_case = self.store.get_success(case_id)

                if success_case is None:
                    continue

                # Filter by confidence
                if success_case.confidence < min_confidence:
                    continue

                similarity = float(dist) if dist >= 0 else 0.0
                similarity = min(1.0, max(0.0, similarity))

                results.append(SearchResult(
                    case_id=case_id,
                    case_type="success",
                    similarity=similarity,
                    data=success_case,
                ))

                if len(results) >= k:
                    break
        else:
            # Fallback
            all_successes = self.store.get_all_successes()

            for success_case in all_successes:
                if success_case.confidence < min_confidence:
                    continue

                text1 = self._get_patient_text(success_case)
                sim = self._cosine_similarity_simple(patient_profile_text, text1)

                results.append(SearchResult(
                    case_id=success_case.case_id,
                    case_type="success",
                    similarity=sim,
                    data=success_case,
                ))

            results.sort(key=lambda x: x.similarity, reverse=True)
            results = results[:k]

        return results

    def _cosine_similarity_simple(self, text1: str, text2: str) -> float:
        """Simple cosine similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def save_index(self):
        """Save indices to disk"""
        if not FAISS_AVAILABLE:
            return

        error_index_file = os.path.join(self.index_path, "error_index.faiss")
        success_index_file = os.path.join(self.index_path, "success_index.faiss")
        error_ids_file = os.path.join(self.index_path, "error_ids.json")
        success_ids_file = os.path.join(self.index_path, "success_ids.json")

        if self.error_index is not None:
            faiss.write_index(self.error_index, error_index_file)
        with open(error_ids_file, "w") as f:
            json.dump(self.error_ids, f)

        if self.success_index is not None:
            faiss.write_index(self.success_index, success_index_file)
        with open(success_ids_file, "w") as f:
            json.dump(self.success_ids, f)

    def rebuild_index(self):
        """Rebuild indices from store"""
        if self.store is None:
            return

        self._create_new_indexes()

        # Rebuild error index
        for error_case in self.store.get_all_errors():
            self.add_error_case(error_case)

        # Rebuild success index
        for success_case in self.store.get_all_successes():
            self.add_success_case(success_case)

        self.save_index()
