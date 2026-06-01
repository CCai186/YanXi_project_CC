"""
配置校验与管理模块 v4
======================
提供配置文件加载、校验、环境变量覆盖、默认值填充。

改进点:
  - 环境变量覆盖（如 DEEPSEEK_API_KEY）
  - 必填字段校验
  - 默认值填充
  - 所有模块的配置项

使用方式:
    from src.core.config import load_and_validate_config

    config = load_and_validate_config("config.yaml")
"""

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 配置默认值
DEFAULTS = {
    "llm": {
        "api_key": "",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "temperature": 0.3,
        "max_tokens": 2048,
        "timeout": 30,
        "retry": {
            "max_retries": 3,
            "base_delay": 1.0,
        },
    },
    "stt": {
        "model_size": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "language": "zh",
        "sample_rate": 16000,
        "vad_threshold": 0.02,
        "silence_duration": 2.0,
        "device_index": None,
    },
    "tts": {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
    },
    "rag": {
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "chroma_persist_dir": "./data/chroma_db",
        "collection_name": "scam_knowledge",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_top_k": 5,
    },
    "orchestrator": {
        "max_conversation_rounds": 10,
        "scam_confidence_threshold": 0.7,
    },
    "habit": {
        "persist_path": "./data/habits/habit_store.json",
        "auto_detect": True,
    },
    "notification": {
        "persist_path": "./data/notifications",
    },
    "call_log": {
        "persist_dir": "./data/call_logs",
    },
    "recorder": {
        "persist_dir": "./data/recordings",
    },
    "caller_profile": {
        "persist_path": "./data/caller_profiles.json",
    },
    "conversation_memory": {
        "max_short_term": 50,
        "context_window": 10,
    },
    "hybrid_retrieval": {
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "rrf_k": 60,
        "reranker_semantic_weight": 0.6,
        "reranker_keyword_weight": 0.4,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "./data/yanxi.log",
    },
}

# 必填字段路径
REQUIRED_FIELDS = [
    ("llm", "api_key"),
]

# 环境变量映射: 环境变量名 → 配置路径
ENV_OVERRIDES = {
    "DEEPSEEK_API_KEY": ("llm", "api_key"),
    "DEEPSEEK_BASE_URL": ("llm", "base_url"),
    "DEEPSEEK_MODEL": ("llm", "model"),
    "YANXI_LOG_LEVEL": ("logging", "level"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并字典，override 的值覆盖 base 的值。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """用环境变量覆盖配置。"""
    for env_var, config_path in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            d = config
            for key in config_path[:-1]:
                d = d.setdefault(key, {})
            d[config_path[-1]] = value
            logger.debug(f"环境变量覆盖: {'.'.join(config_path)} = {value[:8]}...")

    return config


def _validate_config(config: dict) -> list[str]:
    """校验配置，返回警告列表。"""
    warnings = []

    for field_path in REQUIRED_FIELDS:
        d = config
        for key in field_path:
            d = d.get(key, {}) if isinstance(d, dict) else {}
        if not d or d in ("", "your-deepseek-api-key-here", "sk-your-key"):
            warnings.append(
                f"配置项 {'.'.join(field_path)} 未设置或为占位符，"
                f"LLM 相关功能将不可用"
            )

    return warnings


def load_and_validate_config(config_path: str = "config.yaml") -> dict:
    """
    加载、合并默认值、环境变量覆盖、校验配置。

    参数:
        config_path: 配置文件路径（相对于项目根目录）

    返回:
        dict: 完整的配置字典
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / config_path

    # 1. 从默认值开始
    config = _deep_merge({}, DEFAULTS)

    # 2. 加载用户配置文件（如果存在）
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
        logger.debug(f"已加载配置文件: {full_path}")
    else:
        logger.warning(f"配置文件不存在: {full_path}，使用默认配置")

    # 3. 环境变量覆盖
    config = _apply_env_overrides(config)

    # 4. 校验
    warnings = _validate_config(config)
    for w in warnings:
        logger.warning(f"⚠️ 配置警告: {w}")

    return config
