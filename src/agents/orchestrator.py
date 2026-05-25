"""
主调度器 (Orchestrator) v2
==========================
基于三级分类器 + 16 种来电类型 + 预设处理规则的新调度系统。

流程:
  STT文本 → 三级分类(关键词→RAG→LLM) → 查找处理规则 → 执行动作

动作类型:
  - reject:   直接拒接（诈骗/推销/游戏推广）
  - forward:  转接机主（面试通知/重要来电）
  - proxy:    AI 代接对话（外卖/快递/打车）
  - record:   记录留言（同事/客户/银行）
  - ask:      追问信息（家人/朋友/领导/普通）

相比 v1 的改进:
  - 分类从 4 类扩展到 16 类
  - 关键词+RAG 分类不消耗 LLM token（仅低置信度时才调 LLM）
  - 预设处理规则替代动态生成，回复更稳定
"""

import json
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

from src.agents.call_types import (
    CallType, CallAction, get_call_type, get_reply, get_action,
)
from src.agents.classifier import ThreeTierClassifier
from src.agents.business_handler import BusinessHandler
from src.agents.urgent_forwarder import UrgentForwarder
from src.utils.logger import setup_logger
from src.utils.presence import get_presence

logger = setup_logger(__name__)


class OrchestratorState(TypedDict):
    """LangGraph 全局状态"""
    call_text: str
    conversation_history: list[dict]
    type_id: str
    call_type_name: str
    confidence: float
    classify_method: str
    final_action: str
    final_message: str
    agent_reply: str
    business_result: dict
    urgent_result: dict


class CallOrchestrator:
    """
    来电调度器 v2。
    用三级分类器 + 预设规则替代 LLM 动态判断。
    """

    def __init__(self, config: dict):
        self.config = config

        # 初始化三级分类器
        self.classifier = ThreeTierClassifier(config)

        # 尝试注入 RAG 检索器（初始化时不加载模型，安全）
        try:
            from src.knowledge.retriever import ScamKnowledgeRetriever
            self.classifier.retriever = ScamKnowledgeRetriever(config)
            logger.info("RAG 检索器已注入分类器")
        except Exception as e:
            logger.warning(f"RAG 检索器注入失败（将仅用关键词+LLM分类）: {e}")

        # 子 Agent
        self.business_handler = BusinessHandler(config)
        self.urgent_forwarder = UrgentForwarder(config)

        # LangGraph
        self.graph = self._build_graph()
        logger.info("Orchestrator v2 已就绪 (16类分类 + 三级分类器)")

    def _build_graph(self) -> StateGraph:
        """构建简化版状态图：分类 → 路由 → 动作"""
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("classify", self._classify_node)
        workflow.add_node("action", self._action_node)

        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "action")
        workflow.add_edge("action", END)

        return workflow.compile()

    # ============================================================
    # 节点实现
    # ============================================================

    def _classify_node(self, state: OrchestratorState) -> dict:
        """
        节点 1: 三级分类。
        用关键词→RAG→LLM 确定来电类型。
        """
        call_text = state.get("call_text", "")
        logger.info("=" * 50)
        logger.info(f"📍 分类: {call_text[:80]}...")

        result = self.classifier.classify(call_text)

        logger.info(f"  → {result.call_type.emoji} {result.call_type.name} "
                     f"(置信度={result.confidence:.0%}, 方法={result.method})")

        return {
            "type_id": result.type_id,
            "call_type_name": result.call_type.name,
            "confidence": result.confidence,
            "classify_method": result.method,
        }

    def _action_node(self, state: OrchestratorState) -> dict:
        """
        节点 2: 执行动作。
        根据来电类型 + 机主状态，执行对应处理动作。
        """
        type_id = state.get("type_id", "general")
        call_text = state.get("call_text", "")
        presence = get_presence()
        mode = presence.get_mode()
        call_type = get_call_type(type_id)

        # 确定最终动作
        action = get_action(call_type, mode)

        logger.info("=" * 50)
        logger.info(f"📍 动作: {action} (类型={call_type.name}, 机主={presence.get_summary()})")

        # --- reject: 直接拒接 ---
        if action == "reject":
            notification = call_type.notification.format(call_text=call_text[:200])
            return {
                "final_action": "reject",
                "final_message": notification,
                "agent_reply": "",
            }

        # --- forward: 转接机主 ---
        if action == "forward":
            return {
                "final_action": "forward",
                "final_message": f"{call_type.emoji} {call_type.name} → 转接机主:\n内容: {call_text[:200]}",
                "agent_reply": "好的，马上帮您转接机主～",
            }

        # --- proxy: AI 代接对话（外卖/快递/打车）---
        if action == "proxy":
            reply_template = get_reply(call_type, mode)
            conv_state = self.business_handler.start_conversation()
            result = self.business_handler.process_turn(
                caller_text=call_text,
                collected_info=conv_state["collected_info"],
                history=conv_state["history"],
            )
            if result.get("is_complete"):
                return {
                    "business_result": result,
                    "final_action": "summary_card",
                    "final_message": result.get("summary_text", call_type.notification.format(call_text=call_text[:200])),
                    "agent_reply": result.get("agent_text", ""),
                }
            return {
                "business_result": result,
                "final_action": "continue_conversation",
                "final_message": "待后续对话补充信息",
                "agent_reply": result.get("agent_text", ""),
            }

        # --- record: 记录留言 ---
        if action == "record":
            reply = get_reply(call_type, mode)
            if not reply:
                reply = "好的，机主现在不方便接电话，有什么需要转达的吗？"
            return {
                "final_action": "general_reply",
                "final_message": f"{call_type.emoji} {call_type.name}:\n{call_text[:200]}",
                "agent_reply": reply,
                "conversation_history": [
                    {"role": "user", "content": call_text},
                    {"role": "assistant", "content": reply},
                ],
            }

        # --- ask: 追问信息 ---
        if action in ("ask", "continue_conversation"):
            reply = get_reply(call_type, mode)
            if not reply:
                reply = "请问您是哪位？找机主有什么事吗？"
            return {
                "final_action": "continue_conversation",
                "final_message": f"{call_type.emoji} {call_type.name} (待确认)",
                "agent_reply": reply,
                "conversation_history": [
                    {"role": "user", "content": call_text},
                    {"role": "assistant", "content": reply},
                ],
            }

        # --- 兜底 ---
        prefix = presence.get_reply_prefix()
        return {
            "final_action": "general_reply",
            "final_message": f"来电: {call_text[:200]}",
            "agent_reply": f"{prefix}已帮您转达，稍后会回您～",
        }

    # ============================================================
    # 运行入口
    # ============================================================

    def run(self, call_text: str) -> dict:
        """
        运行完整的来电处理流程。

        参数:
            call_text: 来电语音转文字文本

        返回:
            dict: 处理结果
        """
        if not call_text or not call_text.strip():
            return {
                "final_action": "general_reply",
                "final_message": "未收到来电内容",
                "agent_reply": "",
            }

        logger.info("")
        logger.info("╔══════════════════════════════════════════╗")
        logger.info("║    AI 语音通话助手 v2 — 开始处理来电      ║")
        logger.info(f"║ 来电: {call_text[:50]}...")
        logger.info("╚══════════════════════════════════════════╝")

        initial_state: OrchestratorState = {
            "call_text": call_text,
            "conversation_history": [],
            "type_id": "",
            "call_type_name": "",
            "confidence": 0.0,
            "classify_method": "",
            "final_action": "",
            "final_message": "",
            "agent_reply": "",
            "business_result": {},
            "urgent_result": {},
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"调度器异常: {e}")
            return {
                "final_action": "error",
                "final_message": f"系统异常: {str(e)[:200]}",
                "agent_reply": "系统异常，请稍后再试。",
            }

        result = {
            "final_action": final_state.get("final_action", "error"),
            "final_message": final_state.get("final_message", ""),
            "agent_reply": final_state.get("agent_reply", ""),
            "type_id": final_state.get("type_id", ""),
            "call_type_name": final_state.get("call_type_name", ""),
            "confidence": final_state.get("confidence", 0.0),
            "classify_method": final_state.get("classify_method", ""),
            "full_state": final_state,
        }

        logger.info(f"处理完成 → {result['final_action']} "
                     f"({result.get('call_type_name', '?')})")
        return result

    def resume_conversation(self, previous_result: dict, new_caller_text: str) -> dict:
        """
        恢复多轮对话。关键改进：每轮都重新分类，不盲目继承旧类型。

        参数:
            previous_result: 上一轮结果
            new_caller_text: 新的话音文字

        返回:
            dict: 处理结果（与 run() 格式相同）
        """
        logger.info("")
        logger.info("╔══════════════════════════════════════════╗")
        logger.info(f"║  📞 继续对话: {new_caller_text[:50]}...")
        logger.info("╚══════════════════════════════════════════╝")

        # 【重要】重新分类：对方的新回复可能包含关键信息
        classify_result = self.classifier.classify(new_caller_text)
        type_id = classify_result.type_id
        call_type = classify_result.call_type
        logger.info(f"  重新分类: {call_type.name} (置信度={classify_result.confidence:.0%})")

        presence = get_presence()
        mode = presence.get_mode()
        full_state = previous_result.get("full_state", {})

        # 如果新分类是外卖/快递/打车 → 切到业务处理
        if type_id in ("food_delivery", "express", "taxi_arrived"):
            action = get_action(call_type, mode)
            if action == "proxy":
                # 延续已有的对话上下文，不新开 start_conversation
                prev_business = full_state.get("business_result", {})
                collected_info = prev_business.get("collected_info", {
                    "call_type": "", "company": "", "item_description": "",
                    "location": "", "contact_person": "", "additional_notes": "",
                })
                history = prev_business.get("history", [])
                # 如果这是首轮业务对话，先记录来电者已经说了什么
                if not history:
                    history = [{"role": "user", "content": new_caller_text}]
                else:
                    history.append({"role": "user", "content": new_caller_text})

                result = self.business_handler.process_turn(
                    caller_text=new_caller_text,
                    collected_info=collected_info,
                    history=history,
                )
                return {
                    "final_action": "summary_card" if result.get("is_complete") else "continue_conversation",
                    "final_message": result.get("summary_text", call_type.notification.format(call_text=new_caller_text[:200])),
                    "agent_reply": result.get("agent_text", ""),
                    "type_id": type_id,
                    "call_type_name": call_type.name,
                    "full_state": {**full_state, "business_result": result},
                }

        # 如果新分类是诈骗/推销 → 直接拒接
        if type_id in ("scam", "scam_risk", "telemarketing", "game_promo"):
            return {
                "final_action": "reject",
                "final_message": call_type.notification.format(call_text=new_caller_text[:200]),
                "agent_reply": "",
                "type_id": type_id,
                "call_type_name": call_type.name,
            }

        # 追问类：用 LLM 判断下一步
        prefix = presence.get_reply_prefix()
        history = full_state.get("conversation_history", [])
        conversation_stage = len(history) // 2 + 1

        action, reply = self._decide_next_action(
            new_caller_text, call_type, conversation_stage, prefix
        )

        history.append({"role": "user", "content": new_caller_text})
        history.append({"role": "assistant", "content": reply})

        return {
            "final_action": action,
            "final_message": f"{call_type.emoji} {call_type.name}:\n{new_caller_text[:200]}",
            "agent_reply": reply,
            "type_id": type_id,
            "call_type_name": call_type.name,
            "full_state": {**full_state, "conversation_history": history},
        }

    def _decide_next_action(
        self, text: str, call_type: CallType, stage: int, prefix: str
    ) -> tuple[str, str]:
        """
        根据对方回复决定下一步动作。
        优先用 LLM 判断，失败则用规则。

        返回:
            (action, reply)
        """
        try:
            from openai import OpenAI

            llm_cfg = self.config.get("llm", {})
            client = OpenAI(
                api_key=llm_cfg.get("api_key", ""),
                base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
            )

            prompt = f"""来电者说："{text}"
来电类型：{call_type.name}
对话阶段：已回答身份+来意
机主状态：{prefix}

判断下一步：
- 如果是推销/广告/商业目的 → 拒绝，说"机主现在不方便"
- 如果是正常社交（吃饭聚会聊天约）且机主有空 → 转接
- 如果机主忙/免打扰 → 告知状态并结束
- 不确定 → 再问一句

输出（竖线分隔）：
转接|（转接语）
拒绝|（拒绝语）
追问|（追问语）
结束|（结束语）"""

            response = client.chat.completions.create(
                model=llm_cfg.get("model", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )

            output = response.choices[0].message.content.strip()
            if "|" in output:
                parts = output.split("|", 1)
                a = parts[0].strip()
                r = parts[1].strip()
                if "转接" in a:
                    return "forward", r
                if "追问" in a:
                    return "continue_conversation", r
                if "拒绝" in a:
                    return "general_reply", r
                return "general_reply", r

        except Exception as e:
            logger.warning(f"下一步判断失败: {e}")

        return "general_reply", f"{prefix}已帮您转达，稍后会回您～"


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.logger import load_config

    config = load_config("config.yaml")
    orch = CallOrchestrator(config)

    tests = [
        "我是美团外卖的，你的餐到楼下了",
        "您好，我是市公安局的，你涉嫌洗钱案件",
        "喂，在干嘛呢",
        "领导，明天下午有个会议",
        "你的快递到菜鸟驿站了",
        "你好，我们这边有一个课程推荐",
        "妈，我今天晚上回家吃饭",
    ]

    for t in tests:
        result = orch.run(t)
        print(f"\n来电: {t}")
        print(f"  类型: {result.get('call_type_name', '?')} "
              f"(置信度={result.get('confidence',0):.0%}, "
              f"方法={result.get('classify_method','?')})")
        print(f"  动作: {result['final_action']}")
        print(f"  回复: {result['agent_reply'][:60]}")
