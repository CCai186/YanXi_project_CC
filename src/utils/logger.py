"""
日志工具模块
-----------
提供统一的日志格式，同时输出到控制台和文件。
记录每次通话的完整流程，方便调试和问题追踪。

使用方式:
    from src.utils.logger import setup_logger, load_config
    logger = setup_logger(__name__)
    config = load_config("config.yaml")
"""

import logging
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    加载 YAML 配置文件，返回字典。

    参数:
        config_path: 配置文件路径，默认项目根目录下的 config.yaml

    返回:
        dict: 解析后的配置字典
    """
    # 找到项目根目录（YanXi/）
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def setup_logger(name: str, config: dict | None = None) -> logging.Logger:
    """
    创建并配置一个 logger 实例。
    - 同时输出到控制台（INFO 级别）和文件（DEBUG 级别）
    - 控制台使用 Rich 友好的格式，文件包含完整时间戳和模块信息

    参数:
        name: logger 名称，通常传入 __name__
        config: 配置字典，不传则使用默认值

    返回:
        logging.Logger: 配置好的 logger 对象
    """
    # --- 解析配置 ---
    if config is None:
        config = {}

    logging_config = config.get("logging", {})
    log_level_str = logging_config.get("level", "INFO")
    log_file = logging_config.get("file", "./logs/call_assistant.log")
    log_format = logging_config.get(
        "format",
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- 创建 logger ---
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # logger 本身设最低级别，由 handler 控制输出级别
    logger.propagate = False  # 不向上级 logger 重复传递

    # 避免重复添加 handler（热加载场景）
    if logger.handlers:
        return logger

    # --- 控制台 Handler（INFO 级别）---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level_str, logging.INFO))
    console_fmt = logging.Formatter("%(message)s")  # 终端简洁输出
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # --- 文件 Handler（DEBUG 级别）---
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
