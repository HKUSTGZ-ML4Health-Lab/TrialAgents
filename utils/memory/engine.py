"""
TrialAgents Memory Bank - 检索引擎

基于 FAISS 的向量相似度检索
支持 ErrorCase 和 SuccessCase 的语义检索
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 可选导入 FAISS，如果不可用则使用备选方案
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using simple cosine similarity fallback.")


@dataclass
class SearchResult:
    """检索结果"""
    case_id: str
    case_type: str                          # "error" or "success"
    similarity: float
    data: Any                               # ErrorCase or SuccessCase


class MemoryEngine:
    """
    记忆检索引擎

    使用 FAISS 进行高效的向量相似度检索
    """

    def __init__(
        self,
        embedding_dim: int = 384,           # 默认embedding维度
        store=None,                          # MemoryStore实例
        index_path: str = None,
    ):
        """
        初始化检索引擎

        Args:
            embedding_dim: embedding向量维度
            store: MemoryStore实例
            index_path: 索引保存路径
        """
        self.embedding_dim = embedding_dim
        self.store = store
        self.embedding_model = None          # 延迟加载

        # 索引路径
        if index_path is None:
            from utils.data.paths import get_data_path
            index_path = os.path.join(get_data_path(), "memory", "indices")

        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)

        # FAISS 索引
        self.error_index = None
        self.success_index = None

        # ID映射
        self.error_ids: List[str] = []
        self.success_ids: List[str] = []

        # 初始化或加载索引
        self._init_or_load_index()

    def _init_or_load_index(self):
        """初始化或加载索引"""
        error_index_file = os.path.join(self.index_path, "error_index.faiss")
        success_index_file = os.path.join(self.index_path, "success_index.faiss")
        error_ids_file = os.path.join(self.index_path, "error_ids.json")
        success_ids_file = os.path.join(self.index_path, "success_ids.json")

        if FAISS_AVAILABLE:
            # 尝试加载已有索引
            if os.path.exists(error_index_file) and os.path.exists(error_ids_file):
                try:
                    self.error_index = faiss.read_index(error_index_file)
                    with open(error_ids_file, "r") as f:
                        self.error_ids = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load error index: {e}")
                    self._create_new_indexes()

                # 检查维度是否匹配
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
            # 无FAISS时创建简单索引
            self._create_new_indexes()

    def _create_new_indexes(self):
        """创建新索引"""
        if FAISS_AVAILABLE:
            self.error_index = self._create_index()
            self.success_index = self._create_index()
            self.error_ids = []
            self.success_ids = []

    def _create_index(self):
        """创建FAISS索引"""
        if FAISS_AVAILABLE:
            # 使用内积索引（需要归一化embedding）
            return faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        return None

    def _get_embedding_model(self):
        """延迟加载embedding模型"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 使用轻量级模型
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not available. Using fallback embedding.")
                self.embedding_model = None
        return self.embedding_model

    def _embed_text(self, text: str) -> np.ndarray:
        """
        将文本转为embedding向量

        Args:
            text: 输入文本

        Returns:
            embedding向量 (embedding_dim,)
        """
        model = self._get_embedding_model()

        if model is not None:
            embedding = model.encode([text], normalize_embeddings=True)[0]
            return embedding.astype('float32')

        # Fallback: 简单的词袋embedding
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """简单的fallback embedding（词频向量）"""
        # 使用固定维度
        dim = self.embedding_dim
        words = text.lower().split()
        vec = np.zeros(dim, dtype='float32')

        for i, word in enumerate(words[:dim]):
            vec[i % dim] += 1

        # L2归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def _get_patient_text(self, case) -> str:
        """从案例中提取用于检索的文本"""
        if hasattr(case, 'patient_profile'):
            return case.patient_profile.to_text()
        return str(case)

    def add_error_case(self, error_case) -> bool:
        """
        添加错误案例到索引

        Args:
            error_case: ErrorCase实例

        Returns:
            是否成功
        """
        try:
            # 生成embedding
            text = self._get_patient_text(error_case)
            text += f" | Error type: {error_case.error_type.value}"
            text += f" | Root cause: {error_case.root_cause}"
            embedding = self._embed_text(text)

            # 获取或分配ID
            if error_case.case_id in self.error_ids:
                idx_id = self.error_ids.index(error_case.case_id)
            else:
                idx_id = len(self.error_ids)
                self.error_ids.append(error_case.case_id)

            # 添加到FAISS索引
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
        添加成功案例到索引

        Args:
            success_case: SuccessCase实例

        Returns:
            是否成功
        """
        try:
            # 生成embedding
            text = self._get_patient_text(success_case)
            text += f" | Reasoning: {success_case.reasoning_chain[:200]}"
            text += f" | Triggers: {', '.join(success_case.key_triggers[:3])}"
            embedding = self._embed_text(text)

            # 获取或分配ID
            if success_case.case_id in self.success_ids:
                idx_id = self.success_ids.index(success_case.case_id)
            else:
                idx_id = len(self.success_ids)
                self.success_ids.append(success_case.case_id)

            # 添加到FAISS索引
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
        检索相似的错误案例

        Args:
            patient_profile_text: 患者特征文本
            k: 返回数量
            error_types: 筛选的错误类型列表

        Returns:
            相似错误案例列表
        """
        if self.store is None:
            return []

        # 生成query embedding
        query_embedding = self._embed_text(patient_profile_text)

        results = []

        if FAISS_AVAILABLE and self.error_index is not None and len(self.error_ids) > 0:
            # FAISS检索
            search_k = min(k * 2, len(self.error_ids))  # 多检索一些以便过滤
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

                # 按错误类型过滤
                if error_types and error_case.error_type.value not in error_types:
                    continue

                # 转换为相似度 (0-1)
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
            # Fallback: 简单文本匹配
            all_errors = self.store.get_all_errors()

            for error_case in all_errors:
                if error_types and error_case.error_type.value not in error_types:
                    continue

                # 简单相似度计算
                text1 = self._get_patient_text(error_case)
                sim = self._cosine_similarity_simple(patient_profile_text, text1)

                results.append(SearchResult(
                    case_id=error_case.case_id,
                    case_type="error",
                    similarity=sim,
                    data=error_case,
                ))

            # 排序并返回top-k
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
        检索相似的成功案例

        Args:
            patient_profile_text: 患者特征文本
            k: 返回数量
            min_confidence: 最低置信度

        Returns:
            相似成功案例列表
        """
        if self.store is None:
            return []

        # 生成query embedding
        query_embedding = self._embed_text(patient_profile_text)

        results = []

        if FAISS_AVAILABLE and self.success_index is not None and len(self.success_ids) > 0:
            # FAISS检索
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

                # 按置信度过滤
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
        """简单的余弦相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def save_index(self):
        """保存索引到磁盘"""
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
        """从存储重建索引"""
        if self.store is None:
            return

        self._create_new_indexes()

        # 重建错误索引
        for error_case in self.store.get_all_errors():
            self.add_error_case(error_case)

        # 重建成功索引
        for success_case in self.store.get_all_successes():
            self.add_success_case(success_case)

        self.save_index()
