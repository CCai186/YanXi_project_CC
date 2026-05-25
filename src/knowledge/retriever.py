"""
RAG 检索模块
------------
提供诈骗知识库的语义检索接口。采用双模式设计：
  - 语义模式：sentence-transformers + ChromaDB（模型可用时）
  - 关键词模式：TF-IDF 文本匹配（模型不可用时自动降级，零依赖）

两种模式对调用方透明，统一通过 retrieve() 接口使用。

检索流程:
    用户 Query → 向量化/分词 → 相似度搜索 → 返回 Top-K 案例

使用方式:
    from src.knowledge.retriever import ScamKnowledgeRetriever

    retriever = ScamKnowledgeRetriever(config)
    results = retriever.retrieve("您好，我是公安局的...")
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 预处理文档路径（keyword 模式回退用）
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "scam_documents.json"


class ScamKnowledgeRetriever:
    """
    诈骗知识库检索器。

    自动选择最优检索模式:
    1. 优先使用 ChromaDB 语义检索（需先运行 embedder.py 构建向量库）
    2. 若向量库不可用或模型加载失败，降级为 TF-IDF 关键词检索
    """

    def __init__(self, config: dict):
        """
        初始化检索器。

        参数:
            config: 全局配置字典
        """
        rag_cfg = config.get("rag", {})

        self.top_k = rag_cfg.get("retrieval_top_k", 5)
        self.embedding_model_name = rag_cfg.get(
            "embedding_model", "BAAI/bge-small-zh-v1.5"
        )
        chroma_dir = rag_cfg.get("chroma_persist_dir", "./data/chroma_db")
        self.collection_name = rag_cfg.get("collection_name", "scam_knowledge")
        self.chroma_persist_dir = str(PROJECT_ROOT / chroma_dir.lstrip("./"))

        # --- 检测可用模式 ---
        self._mode: str = "keyword"  # 默认 keyword，语义模式按需懒加载
        self._embedding_model = None
        self._chroma_collection = None
        self._documents_cache: Optional[list[dict]] = None
        self._semantic_checked = False  # 是否已经尝试过语义模式

        logger.info(f"检索器已就绪 (模式={self._mode}, top_k={self.top_k})")

    def _init_semantic_mode(self):
        """
        尝试初始化 ChromaDB + sentence-transformers 语义检索模式。
        任一步骤失败则保持 keyword 模式。
        """
        # 检查 ChromaDB 是否存在
        chroma_path = Path(self.chroma_persist_dir)
        if not chroma_path.exists() or not any(chroma_path.iterdir()):
            logger.info("ChromaDB 未构建，使用关键词检索模式")
            return

        # 尝试加载 sentence-transformers
        try:
            # 自动设置 HuggingFace 镜像
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                logger.debug("已设置 HF_ENDPOINT=https://hf-mirror.com")

            from sentence_transformers import SentenceTransformer

            logger.info(f"加载 Embedding 模型: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding 模型加载成功 (维度={self._embedding_model.get_sentence_embedding_dimension()})")

        except Exception as e:
            logger.warning(f"Embedding 模型加载失败: {e}")
            logger.info("降级为关键词检索模式")
            return

        # 尝试连接 ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma_collection = client.get_collection(self.collection_name)
            logger.info(f"ChromaDB 连接成功 (collection={self.collection_name})")

        except Exception as e:
            logger.warning(f"ChromaDB 连接失败: {e}")
            logger.info("降级为关键词检索模式")
            return

        # 全部成功
        self._mode = "semantic"

    def _load_documents(self) -> list[dict]:
        """
        加载预处理文档（keyword 模式的回退数据源）。

        返回:
            list[dict]: 文档列表
        """
        if self._documents_cache is not None:
            return self._documents_cache

        if not DOCUMENTS_PATH.exists():
            logger.warning(f"文档文件不存在: {DOCUMENTS_PATH}")
            self._documents_cache = []
            return []

        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            self._documents_cache = json.load(f)

        logger.debug(f"从文件加载了 {len(self._documents_cache)} 篇文档")
        return self._documents_cache

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fraud_only: bool = False,
    ) -> list[dict]:
        """
        根据用户查询检索最相似的案例。

        参数:
            query: 用户的查询文本
            top_k: 返回数量，默认使用配置值
            filter_fraud_only: 是否只返回诈骗案例

        返回:
            list[dict]: 每个元素包含 id, content, similarity, metadata
        """
        k = top_k or self.top_k

        # 懒加载语义模式（仅在首次检索时尝试，避免 init 时 segfault）
        if not self._semantic_checked:
            self._semantic_checked = True
            try:
                self._init_semantic_mode()
            except Exception as e:
                logger.info(f"语义模式不可用，使用关键词检索: {e}")

        if self._mode == "semantic":
            try:
                return self._semantic_retrieve(query, k, filter_fraud_only)
            except Exception as e:
                logger.warning(f"语义检索失败，降级为关键词: {e}")
                self._mode = "keyword"

        return self._keyword_retrieve(query, k, filter_fraud_only)

    def _semantic_retrieve(
        self, query: str, top_k: int, filter_fraud_only: bool
    ) -> list[dict]:
        """
        ChromaDB 语义检索。
        """
        query_embedding = self._embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        where_filter = {"label": "fraud"} if filter_fraud_only else None

        chroma_results = self._chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if chroma_results["ids"] and chroma_results["ids"][0]:
            for i, doc_id in enumerate(chroma_results["ids"][0]):
                content = chroma_results["documents"][0][i]
                metadata = chroma_results["metadatas"][0][i]
                distance = chroma_results["distances"][0][i]
                similarity = round(1.0 / (1.0 + distance), 4)

                results.append({
                    "id": doc_id,
                    "content": content,
                    "similarity": similarity,
                    "metadata": metadata,
                })

        return results

    def _keyword_retrieve(
        self, query: str, top_k: int, filter_fraud_only: bool
    ) -> list[dict]:
        """
        TF-IDF 风格的关键词检索（无需模型，零依赖回退方案）。

        算法:
        1. 对 query 和每篇文档做中文分词（用 jieba 或简单正则分词）
        2. 计算 query 和文档的关键词重叠度 (Jaccard + TF 加权)
        3. 按得分排序，返回 top_k
        """
        documents = self._load_documents()
        if not documents:
            return []

        # --- 中文分词 ---
        query_tokens = _tokenize(query)

        # --- 计算每篇文档的得分 ---
        scored = []
        for doc in documents:
            # 过滤条件
            if filter_fraud_only and doc.get("metadata", {}).get("label") != "fraud":
                continue

            doc_text = doc.get("content", "")
            doc_tokens = _tokenize(doc_text)

            # 计算 TF-IDF 风格的得分
            score = _compute_keyword_score(query_tokens, doc_tokens)

            if score > 0:
                scored.append((score, doc))

        # --- 按得分降序排序 ---
        scored.sort(key=lambda x: x[0], reverse=True)

        # --- 格式化结果 ---
        results = []
        for score, doc in scored[:top_k]:
            results.append({
                "id": doc.get("id", ""),
                "content": doc.get("content", ""),
                "similarity": round(min(score, 1.0), 4),
                "metadata": doc.get("metadata", {}),
            })

        return results

    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        将检索结果格式化为上下文文本，方便注入 LLM Prompt。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            str: 格式化后的上下文文本
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "（未找到相似案例）"

        lines = ["以下是与当前来电内容相似的已知案例，请参考这些案例进行判断：", ""]

        for i, r in enumerate(results, 1):
            label = "【诈骗】" if r["metadata"].get("label") == "fraud" else "【正常】"
            similarity = r["similarity"]
            lines.append(f"--- 案例 {i} (匹配度: {similarity:.0%}, {label}) ---")
            lines.append(r["content"][:500])
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 关键词检索算法
# ============================================================

# 诈骗检测相关的关键词权重表（人工标注，提升关键术语的匹配权重）
SCAM_KEYWORD_WEIGHTS = {
    # 高危词 — 出现这些词大幅度加分
    "转账": 3.0, "安全账户": 3.0, "保证金": 3.0, "手续费": 2.5,
    "验证码": 3.0, "银行卡号": 3.0, "密码": 2.5, "身份证号": 3.0,
    "通缉令": 3.0, "逮捕": 3.0, "涉嫌犯罪": 3.0, "洗钱": 3.0,
    "公安局": 2.5, "检察院": 2.5, "法院": 2.5, "公检法": 2.5,
    "资金冻结": 2.5, "冻结": 2.0, "社保卡": 2.5, "医保": 2.0,
    "冒充": 2.0, "诈骗": 2.0, "骗": 1.5,
    # 中危词
    "退款": 2.0, "赔偿": 2.0, "中奖": 2.5, "奖金": 2.0,
    "个人所得税": 2.0, "公证费": 2.0, "入会费": 2.0,
    "客服": 1.5, "注销": 2.0, "征信": 2.0, "逾期": 2.0,
    "稳赚": 2.5, "高收益": 2.5, "理财": 1.5, "投资": 1.5,
    "贷款": 2.0, "无抵押": 2.0, "低息": 2.0,
    "刷单": 2.5, "兼职": 1.5, "日赚": 2.5, "垫付": 2.5,
    "杀猪盘": 3.0, "网恋": 2.0, "博彩": 2.5,
    # 正常词 — 出现这些词降低诈骗嫌疑
    "外卖": -1.5, "快递": -1.5, "包裹": -1.5, "美团": -1.5,
    "饿了么": -1.5, "顺丰": -1.5, "取餐": -1.5, "放门口": -1.0,
    "楼下": -1.0, "门卫": -1.0, "物业": -1.0, "开会": -1.0,
    "吃饭": -1.0, "回家": -1.0,
}


def _tokenize(text: str) -> list[str]:
    """
    中文分词。优先用 jieba，不可用时用字符级 bigram + 关键词匹配。

    参数:
        text: 待分词文本

    返回:
        list[str]: 分词结果
    """
    # 尝试用 jieba（如果已安装）
    try:
        import jieba
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        pass

    # --- 简易分词：基于标点和空格切分 + 2-gram ---
    # 把所有标点替换为空格，再按空白切分
    clean = text
    for ch in "，。！？、；：（）\"\"''[]{}【】,;:!?\t\n\r":
        clean = clean.replace(ch, " ")
    segments = [s.strip() for s in clean.split() if s.strip()]

    tokens = []
    for seg in segments:
        # 直接添加整个段作为 token
        tokens.append(seg)
        # 添加 2-gram（字符对）
        if len(seg) >= 2:
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i + 2])
        # 添加单个字符
        tokens.extend(list(seg))

    return tokens


def _compute_keyword_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """
    计算 query 和文档的关键词匹配得分。

    得分策略:
    1. 基础分: Jaccard 相似度（共有词 / 总词数）
    2. 加权分: 查 SCAM_KEYWORD_WEIGHTS 表，匹配关键术语加权
    3. 最终得分 = 基础分 * 0.3 + 加权分 * 0.7

    参数:
        query_tokens: query 的分词列表
        doc_tokens: 文档的分词列表

    返回:
        float: 0~1 之间的得分
    """
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)

    if not query_set or not doc_set:
        return 0.0

    # --- Jaccard 相似度 ---
    intersection = query_set & doc_set
    union = query_set | doc_set
    jaccard = len(intersection) / len(union) if union else 0

    # --- 关键词加权 ---
    weight_score = 0.0
    max_possible_weight = 0.0

    for token in query_set:
        if token in SCAM_KEYWORD_WEIGHTS:
            w = SCAM_KEYWORD_WEIGHTS[token]
            max_possible_weight += abs(w)
            if token in doc_set:
                weight_score += w
                # 额外加分：多个高危词同时命中
                if w >= 2.5 and token in doc_set:
                    weight_score += w * 0.5

    # --- 合成最终得分 ---
    # 归一化权重分为 0~1
    if max_possible_weight > 0:
        normalized_weight = max(0.0, weight_score) / max_possible_weight
    else:
        normalized_weight = 0.0

    # 组合得分
    final_score = jaccard * 0.3 + normalized_weight * 0.7

    return final_score


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试 RAG 检索功能:
        python -m src.knowledge.retriever
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    test_config = {
        "rag": {
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "chroma_persist_dir": "./data/chroma_db",
            "collection_name": "scam_knowledge",
            "retrieval_top_k": 5,
        },
        "logging": {"level": "INFO"},
    }

    retriever = ScamKnowledgeRetriever(test_config)

    test_queries = [
        ("诈骗", "您好，我是公安局的，您涉嫌洗钱案件，请提供银行卡号和密码"),
        ("外卖", "喂您好，您的外卖到了，放在门口还是下来取？"),
        ("诈骗", "恭喜您中奖了！奖金10万元，请先缴纳个人所得税2000元"),
    ]

    for category, query in test_queries:
        print(f"\n{'='*60}")
        print(f"[{category}] {query[:60]}...")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)

        for r in results:
            print(f"\n  匹配度: {r['similarity']:.0%}")
            print(f"  标签: {r['metadata'].get('label', 'N/A')}")
            print(f"  类型: {r['metadata'].get('fraud_type', 'N/A')}")
            print(f"  内容: {r['content'][:120]}...")
