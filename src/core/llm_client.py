"""
LLM 统一客户端
==============
封装 OpenAI 兼容 API 调用，提供：
  - 自动重试（指数退避）
  - 超时控制
  - JSON 输出解析
  - 降级兜底（API 不可用时返回默认结果）

所有 Agent 统一通过此客户端调用 LLM，避免各自实例化 OpenAI 客户端。

使用方式:
    from src.core.llm_client import LLMClient

    client = LLMClient(config)
    result = client.chat(messages=[...], response_type="json")
"""

import json
import time
from typing import Any, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """统一的 LLM 调用客户端，带重试和降级机制。"""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.client = OpenAI(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        )
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.max_tokens = llm_cfg.get("max_tokens", 2048)
        self.timeout = llm_cfg.get("timeout", 30)  # 秒

        # 重试配置
        retry_cfg = llm_cfg.get("retry", {})
        self.max_retries = retry_cfg.get("max_retries", 3)
        self.base_delay = retry_cfg.get("base_delay", 1.0)  # 首次重试等待秒数

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_type: str = "text",
        default_response: Any = None,
    ) -> str:
        """
        发送聊天请求，带自动重试。

        参数:
            messages: OpenAI 格式的消息列表
            temperature: 温度参数，None 则使用默认
            max_tokens: 最大 token 数，None 则使用默认
            response_type: "text" 或 "json"（json 模式会尝试解析输出）
            default_response: 所有重试失败后的降级返回值

        返回:
            str: LLM 回复文本；json 模式返回解析后的 dict/原始字符串
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "timeout": self.timeout,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content.strip()

                if response_type == "json":
                    return self._parse_json(text)
                return text

            except APITimeoutError as e:
                last_error = e
                logger.warning(f"LLM 请求超时 (第{attempt}次)")
            except RateLimitError as e:
                last_error = e
                logger.warning(f"LLM 限流 (第{attempt}次)")
            except APIError as e:
                last_error = e
                logger.warning(f"LLM API 错误 (第{attempt}次): {e}")
            except Exception as e:
                last_error = e
                logger.error(f"LLM 未知错误 (第{attempt}次): {e}")

            # 指数退避等待
            if attempt < self.max_retries:
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.info(f"等待 {delay:.1f}s 后重试...")
                time.sleep(delay)

        # 所有重试失败
        logger.error(f"LLM 调用失败，已重试 {self.max_retries} 次: {last_error}")
        if default_response is not None:
            return default_response
        if response_type == "json":
            return {}
        return ""

    def _parse_json(self, text: str) -> Any:
        """尝试从 LLM 输出中提取 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 或 [ ... ]
        match = re.search(r"[\[{][\s\S]*?[}\]]", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"JSON 解析失败，返回原始文本: {text[:100]}")
        return text

    def close(self):
        """关闭客户端连接。"""
        self.client.close()
