"""
RRF 融合器
==========
Reciprocal Rank Fusion，将多个检索源（向量、关键词、知识图谱）的结果融合排序。

算法: RRF_score(doc) = sum(1 / (k + rank_i + 1))  for each source i
k=60 是常用常数，排名靠前的文档获得更高权重。
"""

from collections import defaultdict
from typing import List, Dict


def doc_key(doc: dict) -> str:
    """生成文档唯一标识"""
    return doc.get("id", "") or doc.get("content", "")[:100]


class RRFFuser:
    """
    RRF 融合器。将多个检索结果列表按 Reciprocal Rank Fusion 算法融合。

    使用方式:
        fuser = RRFFuser(k_const=60)
        fused = fuser.fuse(vector_results, keyword_results, graph_results)
    """

    def __init__(self, k_const: int = 60):
        self.k_const = k_const

    def fuse(self, *doc_lists: List[dict]) -> List[dict]:
        """
        融合多个文档列表。

        参数:
            *doc_lists: 多个文档列表，每个元素是包含 id/content/similarity 的 dict

        返回:
            list[dict]: 融合后的文档列表，按 RRF 分数降序
        """
        scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, dict] = {}

        for docs in doc_lists:
            if not docs:
                continue
            for rank, doc in enumerate(docs):
                key = doc_key(doc)
                # RRF: 1 / (k + rank + 1)
                scores[key] += 1.0 / (self.k_const + rank + 1)
                store.setdefault(key, doc)

        ranked_keys = sorted(store.keys(), key=lambda k: scores[k], reverse=True)
        return [store[k] for k in ranked_keys]

    def fuse_with_scores(self, *doc_score_pairs) -> List[tuple]:
        """
        融合带原始分数的检索结果。

        参数:
            *doc_score_pairs: 每个元素是 (docs_list, scores_list) 的元组

        返回:
            list[tuple]: (doc, fused_score) 列表
        """
        fused_scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, dict] = {}

        for docs, scores in doc_score_pairs:
            if not docs:
                continue
            for rank, (doc, orig_score) in enumerate(zip(docs, scores)):
                key = doc_key(doc)
                rrf = 1.0 / (self.k_const + rank + 1)
                # 原始分数 50% + RRF 50%
                fused_scores[key] += orig_score * 0.5 + rrf * 0.5
                store.setdefault(key, doc)

        ranked = sorted(store.keys(), key=lambda k: fused_scores[k], reverse=True)
        return [(store[k], fused_scores[k]) for k in ranked]


class HybridRetriever:
    """
    v4 混合检索器，结合向量检索和 BM25 关键词检索。

    使用方式:
        hybrid = HybridRetriever(
            vector_retriever=knowledge_retriever,
            bm25_corpus=docs,
            reranker=semantic_reranker,
        )
        results = hybrid.retrieve("查询", top_k=5)
    """

    def __init__(
        self,
        vector_retriever=None,
        bm25_corpus: list[dict] = None,
        reranker=None,
    ):
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self._bm25_docs: list[str] = []
        if bm25_corpus:
            self._bm25_docs = [d.get("text", "") for d in bm25_corpus]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """混合检索：向量检索 + 关键词检索 → RRF 融合 → 重排序。"""
        results = []

        # 向量检索
        if self.vector_retriever:
            try:
                vec_results = self.vector_retriever.retrieve(query, top_k=top_k)
                results.extend(vec_results)
            except Exception:
                pass

        # BM25 关键词检索
        if self._bm25_docs:
            bm25_results = self._bm25_search(query, top_k)
            for idx, score in bm25_results:
                results.append({
                    "id": f"bm25_{idx}",
                    "content": self._bm25_docs[idx],
                    "similarity": score,
                    "metadata": {},
                })

        # RRF 融合去重
        fuser = RRFFuser(k_const=60)
        results = fuser.fuse(results)

        # 重排序
        if self.reranker and results:
            try:
                results = self.reranker.rerank(query, results, top_n=top_k)
            except Exception:
                pass

        return results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple]:
        """简单的 BM25 风格关键词搜索。"""
        query_terms = set(query)
        scored = []
        for idx, doc in enumerate(self._bm25_docs):
            score = sum(1 for t in query_terms if t in doc)
            if score > 0:
                scored.append((idx, score / max(len(query_terms), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
