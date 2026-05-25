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
