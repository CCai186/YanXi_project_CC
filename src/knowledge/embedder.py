"""
文本向量化与 ChromaDB 存储模块
-------------------------------
将预处理后的文档用 Embedding 模型向量化，存入 ChromaDB 向量数据库，
构建诈骗电话检测的 RAG 知识库。

技术选型:
- Embedding 模型: BAAI/bge-small-zh-v1.5
  - 中文优化，1024 维向量
  - 模型体积小 (~100MB)，推理速度快
  - 在中文语义相似度任务上表现优秀
- 向量数据库: ChromaDB
  - 轻量级，本地持久化，无需外部服务
  - 支持 HNSW 索引，检索速度快
  - Python 原生集成

使用方式:
    # 构建知识库（首次运行或数据更新后执行）
    python -m src.knowledge.embedder

    # 在代码中使用:
    from src.knowledge.embedder import KnowledgeBaseBuilder
    builder = KnowledgeBaseBuilder(config)
    collection = builder.build()  # 返回 ChromaDB collection 对象
"""

import json
import time
from pathlib import Path

import os

# 自动设置 HuggingFace 镜像（国内用户必须）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 预处理文档路径
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "scam_documents.json"


class KnowledgeBaseBuilder:
    """
    知识库构建器，负责将文档向量化并存入 ChromaDB。

    工作流程:
    1. 加载预处理后的文档 (JSON)
    2. 加载 Embedding 模型
    3. 逐批将文档向量化
    4. 存入 ChromaDB collection
    5. 持久化到本地磁盘

    后续检索使用 retriever.py 模块。
    """

    def __init__(self, config: dict):
        """
        初始化知识库构建器。

        参数:
            config: 全局配置字典，主要读取 rag 部分
        """
        rag_cfg = config.get("rag", {})

        self.embedding_model_name = rag_cfg.get(
            "embedding_model", "BAAI/bge-small-zh-v1.5"
        )
        self.chroma_persist_dir = str(
            PROJECT_ROOT / rag_cfg.get("chroma_persist_dir", "./data/chroma_db").lstrip("./")
        )
        self.collection_name = rag_cfg.get("collection_name", "scam_knowledge")
        self.chunk_size = rag_cfg.get("chunk_size", 512)
        self.chunk_overlap = rag_cfg.get("chunk_overlap", 50)

        # 延迟加载
        self._embedding_model = None
        self._chroma_client = None

    def load_embedding_model(self):
        """
        加载中文 Embedding 模型。

        首次运行会自动下载模型 (~100MB)，下载后缓存于本地。
        """
        if self._embedding_model is not None:
            return

        logger.info(f"正在加载 Embedding 模型: {self.embedding_model_name}...")
        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"Embedding 模型加载完成 (维度={self._embedding_model.get_sentence_embedding_dimension()}) ✓")

    def _get_chroma_client(self) -> chromadb.PersistentClient:
        """
        获取或创建 ChromaDB 持久化客户端。

        返回:
            chromadb.PersistentClient: ChromaDB 客户端实例
        """
        if self._chroma_client is None:
            # 确保持久化目录存在
            Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB 已连接: {self.chroma_persist_dir}")
        return self._chroma_client

    def build(self, documents_path: str | None = None, batch_size: int = 100):
        """
        构建知识库：加载文档 → 向量化 → 存入 ChromaDB。

        如果 collection 已存在，会先删除再重建（全量构建）。

        参数:
            documents_path: 预处理文档 JSON 路径，不传则用默认路径
            batch_size: 每批向量化的文档数量（避免一次性加载太多导致 OOM）

        返回:
            chromadb.Collection: 构建好的向量集合
        """
        start_time = time.time()

        # --- 1. 加载文档 ---
        path = Path(documents_path) if documents_path else DOCUMENTS_PATH
        logger.info(f"正在加载文档: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"文档文件不存在: {path}\n"
                f"请先运行: python -m src.knowledge.dataset_loader"
            )

        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        logger.info(f"已加载 {len(documents)} 篇文档")

        # --- 2. 加载 Embedding 模型 ---
        self.load_embedding_model()

        # --- 3. 准备 ChromaDB ---
        client = self._get_chroma_client()

        # 如果 collection 已存在，删除（全量重建策略）
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"已删除旧的 collection: {self.collection_name}")
        except Exception:
            pass  # collection 不存在，无需删除

        collection = client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "诈骗电话检测知识库 - TeleAntiFraud-28k",
                "embedding_model": self.embedding_model_name,
                "document_count": str(len(documents)),
            },
        )
        logger.info(f"已创建 collection: {self.collection_name}")

        # --- 4. 批量向量化并存入 ---
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info(f"开始向量化，共 {total_batches} 批 (每批 {batch_size} 篇)...")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(documents))
            batch = documents[start:end]

            # 4a. 提取文档内容和元数据
            ids = [doc["id"] for doc in batch]
            contents = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]

            # 4b. 批量向量化（sentence-transformers 内部已优化 batch 推理）
            logger.debug(f"  向量化第 {batch_idx+1}/{total_batches} 批 ({len(batch)} 篇)...")
            embeddings = self._embedding_model.encode(
                contents,
                normalize_embeddings=True,  # L2 归一化，方便余弦相似度计算
                show_progress_bar=False,
            ).tolist()

            # 4c. 存入 ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

            if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                logger.info(f"  进度: {end}/{len(documents)} ({end/len(documents)*100:.0f}%)")

        elapsed = time.time() - start_time
        logger.info(f"知识库构建完成！")
        logger.info(f"  文档总数: {len(documents)}")
        logger.info(f"  collection: {self.collection_name}")
        logger.info(f"  存储路径: {self.chroma_persist_dir}")
        logger.info(f"  耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

        return collection

    def get_collection(self) -> chromadb.Collection:
        """
        获取已构建的 ChromaDB collection（只读访问）。

        返回:
            chromadb.Collection: 已存在的向量集合

        异常:
            RuntimeError: 如果 collection 尚未构建
        """
        client = self._get_chroma_client()
        try:
            return client.get_collection(self.collection_name)
        except Exception:
            raise RuntimeError(
                f"Collection '{self.collection_name}' 不存在，请先运行:\n"
                f"  python -m src.knowledge.embedder"
            )


# ============================================================
# 独立运行入口
# ============================================================
if __name__ == "__main__":
    """
    运行知识库构建:
        python -m src.knowledge.embedder

    前置条件: 已运行 dataset_loader.py 生成了 documents JSON 文件
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    # 测试配置
    test_config = {
        "rag": {
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "chroma_persist_dir": "./data/chroma_db",
            "collection_name": "scam_knowledge",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_top_k": 5,
        },
        "logging": {"level": "INFO"},
    }

    builder = KnowledgeBaseBuilder(test_config)

    # 检查文档文件是否存在
    if not DOCUMENTS_PATH.exists():
        logger.error(f"文档文件不存在: {DOCUMENTS_PATH}")
        logger.info("请先下载数据集并生成文档: python -m src.knowledge.dataset_loader")
        exit(1)

    builder.build()
