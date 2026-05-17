"""
TrialAgents 检索模块

从 m-agents/utils/retrieve_utils.py 迁移至此
提供基于MedCPT的医学文档检索功能
"""

import os
import json
from typing import List, Dict, Tuple

import faiss
import torch
import numpy as np
import tqdm
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

from .hparams import nct_config

# 配置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = nct_config.get('gpuid', '6')

# 语料库名称映射
corpus_names = {
    "MOC": ["msd", "pubmed"],
    "PubMed": ["pubmed"],
    "MSD": ["msd"],
    "Textbooks": ["textbooks"],
    "No": ["no_corpus"],
}

# 检索器名称映射
retriever_names = {
    "MedCPT": "ncbi/MedCPT-Query-Encoder",
}

# 检索器路径映射
retriever_paths = {
    "MedCPT": "retriever/sentence_transformers/ncbi_MedCPT-Query-Encoder",
}


class CustomizeSentenceTransformer(SentenceTransformer):
    """
    自定义SentenceTransformer，使用CLS pooling替代MEAN pooling
    """

    def _load_auto_model(self, model_name_or_path, **kwargs):
        """
        创建Transformer + CLS Pooling模型
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(
    chunk_dir: str,
    index_dir: str,
    model_name: str,
    **kwargs
) -> int:
    """
    将文本块嵌入为向量并保存

    Args:
        chunk_dir: 文本块目录
        index_dir: 索引保存目录
        model_name: 模型名称
        **kwargs: 其他参数

    Returns:
        嵌入向量的维度
    """
    save_dir = os.path.join(index_dir, "embedding")
    model = CustomizeSentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    fnames = sorted([
        fname for fname in os.listdir(chunk_dir)
        if fname.endswith(".json") or fname.endswith(".jsonl")
    ])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(
                save_dir,
                fname.replace(".json", ".npy") if fname.endswith(".json") else fname.replace(".jsonl", ".npy")
            )
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue

            with open(fpath, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()

            if fname.endswith(".jsonl"):
                texts = [json.loads(item) for item in file_content.split('\n') if item.strip()]
            else:
                data = json.loads(file_content)
                if isinstance(data, list):
                    texts = data
                else:
                    texts = [data]

            # 提取title和content字段
            text_pairs = []
            for item in texts:
                title = item.get("title", "")
                content = item.get("content", "")
                if not content:
                    content = item.get("text", item.get("abstract", ""))
                text_pairs.append([title, content])

            texts = text_pairs
            embed_chunks = model.encode(texts, **kwargs)
            np.save(save_path, embed_chunks)

        embed_chunks = model.encode([""], **kwargs)

    return embed_chunks.shape[-1]


def construct_index(index_dir: str, h_dim: int = 768):
    """
    构建FAISS索引

    Args:
        index_dir: 索引目录
        h_dim: 嵌入维度

    Returns:
        FAISS索引对象
    """
    metadata_path = os.path.join(index_dir, "metadatas.jsonl")

    # 清空并重新创建元数据文件
    with open(metadata_path, 'w') as f:
        f.write("")

    index = faiss.IndexFlatIP(h_dim)
    current_index = 0

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        embedding_path = os.path.join(index_dir, "embedding", fname)
        curr_embed = np.load(embedding_path)

        index.add(curr_embed)

        source_name = fname.replace(".npy", "")
        with open(metadata_path, 'a+', encoding='utf-8') as f:
            for i in range(len(curr_embed)):
                metadata_entry = {
                    'index': i,
                    'source': source_name,
                    'global_index': current_index
                }
                f.write(json.dumps(metadata_entry) + '\n')
                current_index += 1

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    print(f"Index constructed with {index.ntotal} vectors and {current_index} metadata entries")

    return index


class Retriever:
    """
    单个检索器，加载FAISS索引和元数据
    """

    def _load_metadatas(self, metadata_path: str) -> List[dict]:
        """安全加载metadata文件"""
        metadatas = []
        try:
            with open(metadata_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            metadatas.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                            continue
        except FileNotFoundError:
            print(f"Warning: Metadata file not found: {metadata_path}")
        return metadatas

    def __init__(
        self,
        retriever_name: str = "ncbi/MedCPT-Query-Encoder",
        retriever_path: str = None,
        corpus_name: str = "textbooks",
        corpus_dir: str = "/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/corpus",
        **kwargs
    ):
        self.retriever_name = retriever_name
        self.retriever_path = retriever_path
        self.corpus_name = corpus_name
        self.corpus_dir = corpus_dir

        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)

        self.chunk_dir = os.path.join(self.corpus_dir, self.corpus_name, "chunk")
        self.index_dir = os.path.join(
            self.corpus_dir,
            self.corpus_name,
            "index",
            self.retriever_name.replace("Query-Encoder", "Article-Encoder")
        )

        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            print("[In progress] Loading the {:s} corpus with the {:s} retriever...".format(
                self.corpus_name,
                self.retriever_name.replace("Query-Encoder", "Article-Encoder")
            ))
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.metadatas = self._load_metadatas(os.path.join(self.index_dir, "metadatas.jsonl"))
            print("[Finished] Corpus loading finished!")
        else:
            print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(
                self.corpus_name,
                self.retriever_name.replace("Query-Encoder", "Article-Encoder")
            ))
            if os.path.exists(self.retriever_path.replace("Query-Encoder", "Article-Encoder")):
                h_dim = embed(
                    chunk_dir=self.chunk_dir,
                    index_dir=self.index_dir,
                    model_name=self.retriever_path.replace("Query-Encoder", "Article-Encoder"),
                    **kwargs
                )
            else:
                h_dim = embed(
                    chunk_dir=self.chunk_dir,
                    index_dir=self.index_dir,
                    model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"),
                    **kwargs
                )
            print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
            self.index = construct_index(index_dir=self.index_dir, h_dim=h_dim)
            print("[Finished] Corpus indexing finished!")
            self.metadatas = self._load_metadatas(os.path.join(self.index_dir, "metadatas.jsonl"))

        if os.path.exists(self.retriever_path):
            self.embedding_function = CustomizeSentenceTransformer(
                self.retriever_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.embedding_function = CustomizeSentenceTransformer(
                self.retriever_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        self.embedding_function.eval()

    def get_relevant_documents(
        self,
        question: str,
        k: int = 3,
        **kwargs
    ) -> Tuple[List[Dict[str, str]], List[np.ndarray], List[float]]:
        """
        获取与问题相关的文档

        Args:
            question: 查询问题
            k: 返回文档数量
            **kwargs: 其他参数

        Returns:
            (texts, embeddings, scores)
        """
        assert type(question) == str
        question = [question]

        with torch.no_grad():
            query_embed = self.embedding_function.encode(question, **kwargs)

        res_ = self.index.search(query_embed, k=k)

        # 安全地获取索引
        valid_indices = []
        valid_scores = []
        for idx, i in enumerate(res_[1][0]):
            if 0 <= i < len(self.metadatas):
                valid_indices.append(self.metadatas[i])
                valid_scores.append(res_[0][0][idx])
            else:
                print(f"Warning: Index {i} out of range for metadatas (length: {len(self.metadatas)})")

        if not valid_indices:
            print("Warning: No valid indices found, returning empty results")
            return [], [], []

        texts = self.idx2txt(valid_indices)
        embeddings = self.idx2embedding(valid_indices)

        return texts, embeddings, valid_scores

    def idx2txt(self, indices: List[dict]) -> List[Dict[str, str]]:
        """
        将索引转换为文本

        Args:
            indices: 索引列表

        Returns:
            文档文本列表
        """
        texts = []
        for i in indices:
            if os.path.exists(os.path.join(self.chunk_dir, i["source"] + ".json")):
                with open(os.path.join(self.chunk_dir, i["source"] + ".json"), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and i["index"] < len(data):
                    texts.append(data[i["index"]])
                else:
                    texts.append(data[0] if isinstance(data, list) else data)
            else:
                with open(os.path.join(self.chunk_dir, i["source"] + ".jsonl"), 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    if i["index"] < len(lines):
                        texts.append(json.loads(lines[i["index"]]))
                    else:
                        texts.append(json.loads(lines[0]) if lines else {})
        return texts

    def idx2embedding(self, indices: List[dict]) -> List[np.ndarray]:
        """
        将索引转换为嵌入向量

        Args:
            indices: 索引列表

        Returns:
            嵌入向量列表
        """
        return [
            np.load(os.path.join(self.index_dir, "embedding", i["source"] + ".npy"))[i["index"]]
            for i in indices
        ]


class RetrievalSystem:
    """
    检索系统入口，整合多个检索器
    """

    def __init__(
        self,
        retriever_name: str = "MedCPT",
        corpus_name: str = "MSD",
        corpus_dir: str = "/home/chunyu.lei/DEAP/SELF_DATA/mas/m-agents/corpus"
    ):
        assert retriever_name in retriever_names, f"Unknown retriever: {retriever_name}"
        assert corpus_name in corpus_names, f"Unknown corpus: {corpus_name}"

        self.retriever_name = retriever_names[retriever_name]
        self.retriever_path = retriever_paths[retriever_name]
        self.corpus_names = corpus_names[corpus_name]
        self.retrievers = [
            Retriever(self.retriever_name, self.retriever_path, corpus_name, corpus_dir)
            for corpus_name in self.corpus_names
        ]

    def retrieve(
        self,
        question: str,
        k: int = 1
    ) -> Tuple[List[Dict[str, str]], List[np.ndarray], List[float]]:
        """
        检索相关文档

        Args:
            question: 查询问题
            k: 每个检索器返回的数量

        Returns:
            (texts, embeddings, scores)
        """
        assert type(question) == str

        retrieve_texts = []
        retrieve_embeddings = []
        retrieve_scores = []

        for retriever in self.retrievers:
            texts, embeddings, scores = retriever.get_relevant_documents(question, k=k)
            retrieve_texts += texts
            retrieve_embeddings += embeddings
            retrieve_scores += scores

        return retrieve_texts, retrieve_embeddings, retrieve_scores

    def retrieve_with_metadata(
        self,
        query: str,
        k: int = 5
    ) -> dict:
        """
        带元数据的检索（增强方法）

        Args:
            query: 查询字符串
            k: 返回结果数量

        Returns:
            包含检索结果和元数据的字典
        """
        texts, scores, indices = self.retrieve(query, k=k)
        return {
            "texts": texts,
            "scores": scores,
            "query": query,
            "k": k,
        }


__all__ = [
    'CustomizeSentenceTransformer',
    'embed',
    'construct_index',
    'Retriever',
    'RetrievalSystem',
    'corpus_names',
    'retriever_names',
    'retriever_paths',
]
