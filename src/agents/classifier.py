"""
三级来电分类器
==============
关键词匹配(快速) → RAG检索(辅助) → LLM增强(兜底)
每级仅在上一级置信度不足时触发，节省 LLM 调用。

同时提供无意义文本检测，过滤纯数字、重复字符等无效 STT 输出。
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.agents.call_types import CallType, get_call_type
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ClassifyResult:
    """分类结果"""
    type_id: str                    # 类型ID
    call_type: CallType             # 完整类型定义
    confidence: float               # 置信度 0-1
    method: str                     # 方法: keyword/rag/llm/default
    matched_keywords: list = field(default_factory=list)
    reason: str = ""


# ============================================================
# 关键词匹配表（16 类，按优先级排序）
# ============================================================

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    # --- 高危：诈骗 ---
    "scam": [
        "转账", "汇款", "安全账户", "打钱", "银行卡号", "密码",
        "验证码", "短信码", "通缉令", "逮捕令", "涉嫌犯罪",
        "公安", "检察院", "法院传票", "洗钱",
    ],
    "scam_risk": [
        "中奖", "恭喜", "免费领取", "领奖", "奖金",
        "账户异常", "涉嫌", "案件", "配合调查",
        "客服退款", "订单异常", "双倍退款",
        "换号", "我是你领导",
    ],

    # --- 业务类 ---
    "food_delivery": [
        "外卖", "饿了么", "美团", "送餐", "取餐", "骑手", "小哥",
        "餐", "送达", "取餐码", "订单号",
    ],
    "express": [
        "快递", "包裹", "顺丰", "中通", "圆通", "韵达", "菜鸟",
        "驿站", "快递柜", "取件", "取快递",
    ],
    "taxi_arrived": [
        "打车", "车到了", "网约车", "司机", "接驾", "滴滴",
    ],

    # --- 推销/广告 ---
    "telemarketing": [
        "推销", "优惠", "套餐", "升级", "办理", "了解一下",
        "推荐", "推广", "课程", "保险", "理财", "办卡",
        "免费体验", "活动", "介绍",
    ],
    "game_promo": [
        "周年庆", "游戏", "回归", "福利", "礼包", "玩家",
    ],

    # --- 重要联系人 ---
    "family": [
        "妈妈", "爸爸", "妈", "爸", "老婆", "老公", "儿子", "女儿",
        "家里", "家人", "奶奶", "爷爷",
    ],
    "leader": [
        "领导", "老板", "经理", "总经理", "董事",
    ],
    "friend": [
        "朋友", "同学", "哥们", "闺蜜", "兄弟", "姐妹",
        "吃饭", "聚会", "喝酒", "出去玩", "好久不见",
        "最近怎么样", "在干嘛",
    ],
    "colleague": [
        "同事", "技术", "资料", "文档", "对接", "协作",
        "开会", "会议",
    ],
    "client": [
        "客户", "报价", "合同", "方案", "需求", "合作",
    ],
    "bank": [
        "银行", "信用卡", "还款", "账单",
    ],
    "interview": [
        "面试", "offer", "入职", "岗位", "简历", "通知",
    ],
}


class ThreeTierClassifier:
    """
    三级分类器：关键词 → RAG → LLM

    使用方式:
        classifier = ThreeTierClassifier(config)
        result = classifier.classify("我是美团外卖的，你的餐到了")
        print(result.call_type.name)  # "外卖配送"
    """

    def __init__(self, config: dict, retriever=None):
        """
        初始化分类器。

        参数:
            config: 全局配置
            retriever: RAG 检索器（可选，用于二级检索分类）
        """
        self.config = config
        self.retriever = retriever
        self._llm_enabled = bool(config.get("llm", {}).get("api_key", ""))

    def classify(self, text: str) -> ClassifyResult:
        """
        三级分类入口。依次尝试关键词→RAG→LLM。

        参数:
            text: 来电语音转文字

        返回:
            ClassifyResult: 分类结果
        """
        if not text or not text.strip():
            return ClassifyResult(
                type_id="meaningless",
                call_type=get_call_type("meaningless"),
                confidence=0.0,
                method="none",
            )

        # 第 0 步：无意义检测
        if is_meaningless(text):
            return ClassifyResult(
                type_id="meaningless",
                call_type=get_call_type("meaningless"),
                confidence=0.95,
                method="meaningless_detect",
            )

        # 第 1 步：关键词匹配（降低门槛，关键词比 RAG 更可靠）
        kw_result = _keyword_match(text)
        if kw_result and kw_result.confidence >= 0.7:
            logger.info(f"分类: {kw_result.call_type.name} (关键词, 置信度={kw_result.confidence:.0%})")
            return kw_result

        # 第 2 步：RAG 检索辅助（仅关键词完全没匹配时才用）
        if kw_result and kw_result.confidence >= 0.55:
            # 关键词有弱匹配 → 信任关键词，不用 RAG（RAG 可能有偏）
            logger.info(f"分类: {kw_result.call_type.name} (关键词弱匹配, 置信度={kw_result.confidence:.0%})")
            return kw_result

        rag_result = self._rag_classify(text, kw_result)
        if rag_result and rag_result.confidence >= 0.7:
            logger.info(f"分类: {rag_result.call_type.name} (RAG, 置信度={rag_result.confidence:.0%})")
            return rag_result

        # 第 3 步：LLM 增强（仅在关键词和 RAG 都不确定时）
        if self._llm_enabled and (not kw_result or kw_result.confidence < 0.7):
            llm_result = self._llm_classify(text)
            if llm_result and llm_result.confidence >= 0.7:
                logger.info(f"分类: {llm_result.call_type.name} (LLM, 置信度={llm_result.confidence:.0%})")
                return llm_result

        # 兜底
        if kw_result:
            logger.info(f"分类: {kw_result.call_type.name} (关键词, 低置信度={kw_result.confidence:.0%})")
            return kw_result

        return ClassifyResult(
            type_id="general",
            call_type=get_call_type("general"),
            confidence=0.3,
            method="default",
            reason="无法确定来电类型",
        )

    def _rag_classify(self, text: str, kw_result: Optional[ClassifyResult] = None) -> Optional[ClassifyResult]:
        """通过 RAG 检索辅助分类"""
        if self.retriever is None:
            return None

        try:
            results = self.retriever.retrieve(text, top_k=5)
            if not results:
                return None

            # 统计检索结果中的类型分布
            from collections import Counter
            type_ids = []
            for r in results:
                fraud_type = r.get("metadata", {}).get("fraud_type", "")
                label = r.get("metadata", {}).get("label", "")
                if label == "fraud":
                    type_ids.append("scam")
                else:
                    # 用关键词在文档内容中匹配类型
                    content = r.get("content", "")
                    for tid, keywords in CATEGORY_KEYWORDS.items():
                        if any(kw in content for kw in keywords[:3]):
                            type_ids.append(tid)
                            break
                    else:
                        type_ids.append("general")

            if not type_ids:
                return None

            counter = Counter(type_ids)
            top_type_id, count = counter.most_common(1)[0]
            confidence = min(0.85, 0.5 + (count / len(type_ids)) * 0.35)

            return ClassifyResult(
                type_id=top_type_id,
                call_type=get_call_type(top_type_id),
                confidence=confidence,
                method="rag",
                reason=f"检索到 {count}/{len(type_ids)} 条匹配",
            )

        except Exception as e:
            logger.debug(f"RAG 分类失败: {e}")
            return None

    def _llm_classify(self, text: str) -> Optional[ClassifyResult]:
        """通过 LLM 分类（最后一层兜底）"""
        try:
            from openai import OpenAI
            from src.agents.call_types import CALL_TYPES

            llm_cfg = self.config.get("llm", {})
            client = OpenAI(
                api_key=llm_cfg.get("api_key", ""),
                base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
            )

            type_list = "\n".join([
                f"- {ct.type_id}: {ct.name}"
                for ct in list(CALL_TYPES.values())[:10]
            ])

            prompt = f"""判断以下来电内容属于哪种类型，只输出类型ID。

来电: "{text}"

可选类型:
{type_list}
- general: 其他

只输出一个类型ID："""

            response = client.chat.completions.create(
                model=llm_cfg.get("model", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )

            type_id = response.choices[0].message.content.strip().lower()
            call_type = get_call_type(type_id)

            return ClassifyResult(
                type_id=call_type.type_id,
                call_type=call_type,
                confidence=0.9,
                method="llm",
            )

        except Exception as e:
            logger.warning(f"LLM 分类失败: {e}")
            return None


# ============================================================
# 关键词匹配
# ============================================================

def _keyword_match(text: str) -> Optional[ClassifyResult]:
    """
    关键词快速匹配。
    诈骗类关键词权重更高，一条命中即可；其他类需多条命中。
    """
    best_type_id = ""
    best_score = 0
    best_keywords = []

    for type_id, keywords in CATEGORY_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text]

        if not matched:
            continue

        # 诈骗类：命中 1 条即高分
        if type_id in ("scam", "scam_risk"):
            score = min(0.98, 0.85 + len(matched) * 0.05)
        # 推销类：需 2 条
        elif type_id in ("telemarketing", "game_promo"):
            if len(matched) >= 2:
                score = min(0.90, 0.70 + len(matched) * 0.08)
            else:
                score = 0.55
        # 业务类：需 2 条
        elif type_id in ("food_delivery", "express", "taxi_arrived"):
            if len(matched) >= 2:
                score = min(0.92, 0.75 + len(matched) * 0.08)
            else:
                score = 0.60
        # 联系人类：需 1-2 条
        else:
            score = min(0.88, 0.60 + len(matched) * 0.10)

        if score > best_score:
            best_score = score
            best_type_id = type_id
            best_keywords = matched

    if best_score >= 0.6:
        return ClassifyResult(
            type_id=best_type_id,
            call_type=get_call_type(best_type_id),
            confidence=best_score,
            method="keyword",
            matched_keywords=best_keywords,
            reason=f"关键词匹配: {', '.join(best_keywords[:5])}",
        )

    return None


# ============================================================
# 无意义文本检测
# ============================================================

def is_meaningless(text: str) -> bool:
    """
    检测 STT 输出是否为无意义内容。
    过滤场景：纯数字、重复字符、计数序列、空/过短文本。
    """
    text = text.strip()
    if len(text) < 2:
        return True

    clean = re.sub(r'\s+', '', text)
    if len(clean) < 2:
        return True

    # 连续数字 >= 15 位
    if re.search(r'\d{15,}', clean):
        return True

    # 同一个单字重复 >= 3 次且占比 >= 40%
    for ch in set(clean):
        count = clean.count(ch)
        if count >= 3 and count / len(clean) >= 0.4:
            return True

    # 同一个 2-3 字词组重复 >= 3 次，覆盖过半文本
    for wlen in [2, 3]:
        seen = set()
        for i in range(len(clean) - wlen + 1):
            word = clean[i:i + wlen]
            if word in seen:
                continue
            seen.add(word)
            if clean.count(word) >= 3 and len(word) * clean.count(word) >= len(clean) * 0.5:
                return True

    # 中文数字 + 阿拉伯数字占比 >= 60%
    numerals = set('一二三四五六七八九十百千万亿零两')
    if len(clean) >= 3:
        numeral_count = sum(1 for ch in clean if ch in numerals or ch.isdigit())
        if numeral_count / len(clean) >= 0.6:
            meaningful = {'送餐', '取餐', '取件', '打钱', '汇款',
                         '快递', '外卖', '到了', '面试', '开会', '同学'}
            if not any(kw in clean for kw in meaningful):
                return True

    return False


# ============================================================
# 独立测试
# ============================================================
if __name__ == "__main__":
    classifier = ThreeTierClassifier({"llm": {}}, retriever=None)

    tests = [
        "我是美团外卖的，你的餐到了",
        "你涉嫌洗钱，请配合调查",
        "喂，在干嘛呢",
        "妈，我今天晚上回家吃饭",
        "领导，明天有个会议",
        "你的快递到了，在菜鸟驿站",
        "123 456 789 012 345 678",
        "你好，我们这边有个课程推荐",
    ]

    for t in tests:
        r = classifier.classify(t)
        print(f"{r.call_type.emoji} [{r.call_type.name}] 置信度:{r.confidence:.0%} 方法:{r.method}")
        print(f"   '{t}'")
        print()
