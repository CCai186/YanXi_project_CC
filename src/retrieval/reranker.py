"""
重排器
======
对检索结果进行二次排序，提升最相关文档的排名。

目前使用简单的关键词密度排序，后续可升级为 Cross-Encoder 模型。
"""

from typing import List


class SimpleReranker:
    """
    简单重排器。根据查询词在文档中的命中程度重新排序。

    使用方式:
        reranker = SimpleReranker()
        reranked = reranker.rerank("外卖", docs, top_n=3)
    """

    def rerank(self, query: str, documents: List[dict], top_n: int = 5) -> List[dict]:
        """
        对文档列表重新排序。

        排序策略:
        1. 查询词在文档中出现的次数越多，排名越靠前
        2. 原有相似度作为辅助参考

        参数:
            query: 查询文本
            documents: 文档列表
            top_n: 返回数量

        返回:
            list[dict]: 重排后的 top_n 文档
        """
        if not documents:
            return []

        # 提取查询关键词
        query_terms = set(query)

        scored = []
        for doc in documents:
            content = doc.get("content", "")
            # 关键词命中数
            term_hits = sum(1 for t in query_terms if t in content)
            # 原始相似度
            orig_score = doc.get("similarity", 0.5)
            # 综合得分: 关键词命中 60% + 原始相似度 40%
            combined = term_hits * 0.6 + orig_score * 0.4
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
