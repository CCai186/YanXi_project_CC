"""
主调度器 (Orchestrator) v4 — 完整版
====================================
集成所有模块的 LangGraph 多智能体调度系统。

模块集成:
  ✅ LangGraph 状态图驱动
  ✅ HabitLearner — 习惯学习
  ✅ CardBuilder — 通知卡片
  ✅ CallLogger — 通话记录
  ✅ CallRecorder — 通话录音 (模块A)
  ✅ CallerProfile — 来电者画像 (模块B)
  ✅ ConversationMemory — 对话记忆 (模块D)
  ✅ HybridRetriever — 混合检索 (模块F)
  ✅ KnowledgeExpander — 知识库扩展 (模块F)
  ✅ LLMClient — 统一 LLM 调用

流程:
  STT文本 → classify → lookup_profile → infer_presence → route → action → notify
               ↑              ↑
        CallerProfile    HabitLearner
        ConversationMemory
"""

import time
import uuid
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.agents.call_types import (
    CallType, CallAction, get_call_type, get_reply, get_action,
)
from src.agents.classifier import ThreeTierClassifier
from src.agents.business_handler import BusinessHandler
from src.agents.urgent_forwarder import UrgentForwarder
from src.agents.conversation_memory import ConversationMemory
from src.core.llm_client import LLMClient
from src.habit.habit_learner import HabitLearner
from src.knowledge.caller_profile import CallerProfileStore, CallerProfile
from src.knowledge.knowledge_expander import KnowledgeExpander
from src.notification.card_builder import CardBuilder, NotificationStore
from src.store.call_logger import CallLogger
from src.utils.logger import setup_logger
from src.utils.presence import get_presence

logger = setup_logger(__name__)


class OrchestratorState(TypedDict, total=False):
    """LangGraph 全局状态"""
    # 输入
    call_text: str
    caller_number: str
    # 分类结果
    type_id: str
    call_type_name: str
    confidence: float
    classify_method: str
    # 机主状态
    presence_mode: str
    presence_reason: str
    # 来电者画像
    caller_profile: Optional[dict]
    # 动作
    final_action: str
    final_message: str
    agent_reply: str
    # 通知卡片
    notification_card: Optional[dict]
    # 对话记忆
    conversation_memory: Optional[dict]


class CallOrchestrator:
    """
    基于 LangGraph 的多智能体调度器 v4。

    完整流程:
      来电文本 → 分类 → 画像查询 → 习惯推断 → 路由 → 动作 → 通知
    """

    def __init__(self, config: dict):
        self.config = config

        # --- 核心组件 ---
        self.llm_client = LLMClient(config)
        self.classifier = ThreeTierClassifier(config)
        self.business_handler = BusinessHandler(config)
        self.urgent_forwarder = UrgentForwarder(config)
        self.card_builder = CardBuilder()

        # --- 模块A: 通话录音 ---
        recorder_cfg = config.get("recorder", {})
        try:
            from src.voice.recorder import CallRecorder
            self.recorder = CallRecorder(
                persist_dir=recorder_cfg.get("persist_dir", "./data/recordings")
            )
        except Exception as e:
            logger.warning(f"录音模块初始化失败（将跳过录音功能）: {e}")
            self.recorder = None

        # --- 模块B: 来电者画像 ---
        profile_cfg = config.get("caller_profile", {})
        self.caller_profile_store = CallerProfileStore(
            persist_path=profile_cfg.get("persist_path", "./data/caller_profiles.json")
        )

        # --- 模块D: 对话记忆 ---
        mem_cfg = config.get("conversation_memory", {})
        self.conversation_memory = ConversationMemory(
            max_short_term=mem_cfg.get("max_short_term", 50)
        )

        # --- 模块F: 知识库增强 ---
        self.knowledge_expander = KnowledgeExpander()
        self._init_enhanced_retrieval(config)

        # --- 习惯学习 ---
        self.habit_learner = HabitLearner(config)

        # --- 通知系统 ---
        notif_cfg = config.get("notification", {})
        self.notification_store = NotificationStore(
            persist_path=notif_cfg.get("persist_path", "./data/notifications")
        )

        # --- 通话记录 ---
        log_cfg = config.get("call_log", {})
        self.call_logger = CallLogger(
            persist_dir=log_cfg.get("persist_dir", "./data/call_logs")
        )

        # --- 机主状态 ---
        self.presence = get_presence()

        # --- 构建 LangGraph ---
        self.graph = self._build_graph()
        logger.info("CallOrchestrator v4 初始化完成（含模块ABDF）")

    def _init_enhanced_retrieval(self, config: dict) -> None:
        """初始化增强检索系统（模块F）。
        注意：sentence-transformers 在部分环境下会导致 segfault，
        因此默认仅使用关键词模式，语义模式需显式启用。
        """
        # 默认使用基础检索（关键词模式），避免 sentence-transformers segfault
        self.embedder = None
        self.retriever = None
        self.reranker = None
        self.hybrid_retriever = None

        # 检查是否启用语义模式
        rag_cfg = config.get("rag", {})
        if not rag_cfg.get("enable_semantic", False):
            logger.info("检索系统: 关键词模式（语义模式未启用，在 config.yaml 的 rag.enable_semantic 设为 true 可开启）")
            return

        try:
            from src.knowledge.embedder import Embedder
            from src.knowledge.retriever import KnowledgeRetriever

            self.embedder = Embedder(config)
            self.retriever = KnowledgeRetriever(config, self.embedder)

            # 扩展知识库
            expand_count = self.knowledge_expander.expand_all(self.retriever)
            logger.info(f"知识库扩展: 新增 {expand_count} 条知识")

            # 构建混合检索
            all_docs = self.knowledge_expander.get_all_documents()
            from src.retrieval.reranker import SemanticReranker
            self.reranker = SemanticReranker(embedder=self.embedder)
            from src.retrieval.fusion import HybridRetriever
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.retriever,
                bm25_corpus=all_docs,
                reranker=self.reranker,
            )

            # 更新分类器的检索器
            if hasattr(self.classifier, 'retriever'):
                self.classifier.retriever = self.hybrid_retriever

            logger.info("增强检索系统初始化完成（语义模式）")
        except Exception as e:
            logger.warning(f"增强检索系统初始化失败（将使用关键词模式）: {e}")
            self.embedder = None
            self.retriever = None
            self.reranker = None
            self.hybrid_retriever = None

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图。"""
        graph = StateGraph(OrchestratorState)

        # 添加节点
        graph.add_node("classify", self._node_classify)
        graph.add_node("lookup_profile", self._node_lookup_profile)
        graph.add_node("infer_presence", self._node_infer_presence)
        graph.add_node("route", self._node_route)
        graph.add_node("handle_scam", self._node_handle_scam)
        graph.add_node("handle_business", self._node_handle_business)
        graph.add_node("handle_urgent", self._node_handle_urgent)
        graph.add_node("handle_normal", self._node_handle_normal)
        graph.add_node("notify", self._node_notify)

        # 设置入口
        graph.set_entry_point("classify")

        # 边
        graph.add_edge("classify", "lookup_profile")
        graph.add_edge("lookup_profile", "infer_presence")
        graph.add_edge("infer_presence", "route")
        graph.add_conditional_edges(
            "route",
            self._route_by_type,
            {
                "scam": "handle_scam",
                "business": "handle_business",
                "urgent": "handle_urgent",
                "normal": "handle_normal",
            },
        )
        graph.add_edge("handle_scam", "notify")
        graph.add_edge("handle_business", "notify")
        graph.add_edge("handle_urgent", "notify")
        graph.add_edge("handle_normal", "notify")
        graph.add_edge("notify", END)

        return graph.compile()

    # ============================================================
    # LangGraph 节点
    # ============================================================

    def _node_classify(self, state: dict) -> dict:
        """分类节点：三级分类器。"""
        text = state.get("call_text", "")
        if not text:
            return {"type_id": "general", "call_type_name": "未知", "confidence": 0.0, "classify_method": "empty"}

        result = self.classifier.classify(text)
        return {
            "type_id": result.type_id,
            "call_type_name": result.call_type.name,
            "confidence": result.confidence,
            "classify_method": result.method,
        }

    def _node_lookup_profile(self, state: dict) -> dict:
        """画像查询节点：查看来电者历史（模块B）。"""
        caller_number = state.get("caller_number", "")
        if not caller_number:
            return {"caller_profile": None}

        profile = self.caller_profile_store.lookup(caller_number)
        if profile:
            logger.info(f"来电者画像: {caller_number} 信任={profile.trust_score:.2f} "
                        f"标签={profile.tags} 来电={profile.call_count}次")
            # 设置长期记忆
            self.conversation_memory.set_long_term(profile.to_dict())
            return {"caller_profile": profile.to_dict()}
        return {"caller_profile": None}

    def _node_infer_presence(self, state: dict) -> dict:
        """习惯推断节点：根据习惯推断机主状态。"""
        mode, reason = self.habit_learner.infer_presence_mode()
        if mode != "free":
            # 使用 UserPresence.set() 方法
            self.presence.set(mode, reason)
        return {"presence_mode": mode, "presence_reason": reason}

    def _node_route(self, state: dict) -> dict:
        """路由节点：根据分类结果 + 机主状态 + 来电者画像决定路由。"""
        type_id = state.get("type_id", "general")
        presence_mode = state.get("presence_mode", "free")
        profile_dict = state.get("caller_profile")

        # 黑名单直接走诈骗
        if profile_dict and profile_dict.get("is_blacklisted"):
            logger.info("来电者在黑名单中，路由到诈骗处理")
            return {"final_action": "reject"}

        # 白名单直接走转接
        if profile_dict and profile_dict.get("is_whitelisted"):
            logger.info("来电者在白名单中，优先转接")
            if type_id in ("scam",):
                return {"final_action": "reject"}
            return {"final_action": "forward"}

        # 机主繁忙/免打扰时，只有紧急来电转接
        if presence_mode in ("busy", "dnd"):
            if type_id in ("scam",):
                return {"final_action": "reject"}
            if type_id in ("family", "urgent"):
                return {"final_action": "forward"}
            # 其他一律代接
            return {"final_action": "proxy"}

        # 正常模式
        call_type = get_call_type(type_id)
        action = get_action(call_type, presence_mode)
        return {"final_action": action}

    def _route_by_type(self, state: dict) -> str:
        """条件路由：根据 type_id 选择处理节点。"""
        type_id = state.get("type_id", "general")
        if type_id in ("scam", "scam_risk", "telemarketing", "game_promo"):
            return "scam"
        if type_id in ("food_delivery", "express", "taxi_arrived", "bank"):
            return "business"
        if type_id in ("family", "leader", "urgent"):
            return "urgent"
        return "normal"

    def _node_handle_scam(self, state: dict) -> dict:
        """诈骗处理节点。"""
        text = state.get("call_text", "")
        confidence = state.get("confidence", 0.8)

        # 使用 LLMClient 获取详细分析
        try:
            analysis = self.llm_client.chat(
                messages=[{
                    "role": "user",
                    "content": f"分析以下来电是否为诈骗，简要说明理由：\n{text}"
                }],
                response_type="text",
                default_response="疑似诈骗电话",
            )
        except Exception:
            analysis = "疑似诈骗电话"

        # 更新来电者画像
        caller_number = state.get("caller_number", "")
        if caller_number:
            profile = self.caller_profile_store.get_or_create(caller_number)
            profile.add_call("scam", confidence)
            self.caller_profile_store.update(profile)

        # 生成诈骗拦截卡片
        type_id = state.get("type_id", "scam")
        card = self.card_builder.build_scam_log(
            scam_type=state.get("call_type_name", "诈骗拦截"),
            reason=analysis[:100] if analysis else "疑似诈骗",
            confidence=confidence,
        )

        return {
            "agent_reply": "",
            "notification_card": card.to_dict(),
        }

    def _node_handle_business(self, state: dict) -> dict:
        """业务处理节点（外卖/快递/银行等）。"""
        text = state.get("call_text", "")
        type_id = state.get("type_id", "general")
        presence_mode = state.get("presence_mode", "free")

        # 更新对话记忆
        self.conversation_memory.add_user_message(text)

        # 从对话中提取关键信息更新工作记忆
        self._extract_business_info(text, type_id)

        # 使用 BusinessHandler 生成回复
        try:
            result = self.business_handler.process_turn(
                caller_text=text,
                collected_info=self.conversation_memory.working_memory.to_dict(),
                history=self.conversation_memory.short_term,
            )
            reply = result.get("agent_text", "") if isinstance(result, dict) else str(result)
        except Exception:
            call_type = get_call_type(type_id)
            reply = get_reply(call_type, presence_mode)

        self.conversation_memory.add_assistant_message(reply)

        # 检查对话是否完成
        is_complete = result.get("is_complete", False) if isinstance(result, dict) else True

        # 只在对话完成时生成业务卡片
        wm = self.conversation_memory.working_memory
        card = None
        if is_complete:
            if type_id in ("food_delivery", "express"):
                card = self.card_builder.build_delivery_card(
                    call_type="外卖" if type_id == "food_delivery" else "快递",
                    company=wm.caller_company or "未知平台",
                    item=wm.purpose_detail or "物品",
                    location=wm.delivery_location or "待确认",
                    notes=wm.delivery_notes or "",
                )
            else:
                card = self.card_builder.build_message_card(
                    caller=wm.caller_identity or "来电者",
                    message=wm.caller_purpose or text[:100],
                    relationship=state.get("call_type_name", "业务"),
                )

        # 更新来电者画像
        caller_number = state.get("caller_number", "")
        if caller_number:
            profile = self.caller_profile_store.get_or_create(caller_number)
            profile.add_call(type_id, state.get("confidence", 0.8))
            self.caller_profile_store.update(profile)

        return {
            "agent_reply": reply,
            "notification_card": card.to_dict() if card else None,
            "final_action": "summary_card" if is_complete else "continue_conversation",
        }

    def _node_handle_urgent(self, state: dict) -> dict:
        """紧急来电处理节点。"""
        text = state.get("call_text", "")
        presence_mode = state.get("presence_mode", "free")

        self.conversation_memory.add_user_message(text)

        try:
            result = self.urgent_forwarder.assess(text)
            should_forward = result.get("should_forward", False)
            urgency = result.get("urgency_level", "high")
        except Exception:
            should_forward = False
            urgency = "high"

        # 决定回复和动作
        if should_forward:
            reply = result.get("agent_text", "好的，我马上通知机主。")
            card = self.card_builder.build_urgent_alert(
                caller=self.conversation_memory.working_memory.caller_identity or "来电者",
                reason=text[:100],
                urgency_level=urgency,
            )
            effective_action = "forward"
        else:
            # 非紧急：空闲模式用友好模板，忙碌/免打扰用 forwarder 的回复
            card = None
            effective_action = "continue_conversation"
            if presence_mode == "free":
                call_type = get_call_type(state.get("type_id", "general"))
                reply = get_reply(call_type, "free")
            else:
                reply = result.get("agent_text", "")

        self.conversation_memory.add_assistant_message(reply)

        # 更新画像
        caller_number = state.get("caller_number", "")
        if caller_number:
            profile = self.caller_profile_store.get_or_create(caller_number)
            profile.add_call("urgent", state.get("confidence", 0.8))
            self.caller_profile_store.update(profile)

        return {
            "agent_reply": reply,
            "notification_card": card.to_dict() if card else None,
            "final_action": effective_action,
        }

    def _node_handle_normal(self, state: dict) -> dict:
        """普通来电处理节点。"""
        text = state.get("call_text", "")
        type_id = state.get("type_id", "general")
        presence_mode = state.get("presence_mode", "free")
        final_action = state.get("final_action", "forward")

        self.conversation_memory.add_user_message(text)

        if final_action == "proxy":
            # 代接模式：礼貌回复并留言
            reply = "您好，机主现在不方便接听电话。请问您有什么事，我可以帮忙转达。"
            self.conversation_memory.add_assistant_message(reply)

            card = self.card_builder.build_message_card(
                caller=self.conversation_memory.working_memory.caller_identity or "来电者",
                message=text[:100],
                relationship="普通来电",
            )
            # proxy 模式也是问问题 → 继续对话
            effective_action = "continue_conversation"
        elif final_action == "general_reply":
            # 真正的终结动作：礼貌结束
            call_type = get_call_type(type_id)
            reply = get_reply(call_type, presence_mode)
            self.conversation_memory.add_assistant_message(reply)
            card = None
            effective_action = "general_reply"
        else:
            # forward/ask/continue_conversation → 先问清楚再决定
            call_type = get_call_type(type_id)
            reply = get_reply(call_type, presence_mode)
            self.conversation_memory.add_assistant_message(reply)
            card = None
            effective_action = "continue_conversation"

        # 更新画像
        caller_number = state.get("caller_number", "")
        if caller_number:
            profile = self.caller_profile_store.get_or_create(caller_number)
            profile.add_call(type_id, state.get("confidence", 0.5))
            self.caller_profile_store.update(profile)

        return {
            "agent_reply": reply,
            "notification_card": card.to_dict() if card else None,
            "final_action": effective_action,
        }

    def _node_notify(self, state: dict) -> dict:
        """通知节点：保存卡片 + 记录通话 + 更新录音信息。"""
        card_dict = state.get("notification_card")
        if card_dict:
            try:
                from src.notification.card_builder import NotificationCard
                card = NotificationCard(**card_dict)
                self.notification_store.save(card)
            except Exception as e:
                logger.warning(f"保存通知卡片失败: {e}")

        # 记录通话
        try:
            session_id = self.call_logger.start_session(
                caller_number=state.get("caller_number", ""),
                caller_text=state.get("call_text", ""),
            )
            self.call_logger.log_classification(
                session_id,
                state.get("type_id", "general"),
                state.get("call_type_name", "未知"),
                state.get("confidence", 0.0),
                state.get("classify_method", "unknown"),
            )
            self.call_logger.log_action(
                session_id,
                state.get("final_action", "unknown"),
                state.get("agent_reply", ""),
            )
            self.call_logger.end_session(
                session_id,
                notification_card=card_dict,
            )

            # 更新录音关联信息
            if self.recorder:
                try:
                    self.recorder.update_recording_info(
                        session_id,
                        call_type=state.get("type_id", ""),
                        final_action=state.get("final_action", ""),
                    )
                except Exception as e:
                    logger.debug(f"更新录音信息跳过: {e}")
        except Exception as e:
            logger.warning(f"记录通话失败: {e}")

        return {}

    # ============================================================
    # 辅助方法
    # ============================================================

    def _extract_business_info(self, text: str, type_id: str) -> None:
        """从对话文本中提取业务关键信息，更新工作记忆。"""
        wm = self.conversation_memory.working_memory

        # 根据类型设置默认身份
        if type_id == "food_delivery" and not wm.caller_identity:
            wm.caller_identity = "外卖配送员"
        elif type_id == "express" and not wm.caller_identity:
            wm.caller_identity = "快递员"

        # 尝试提取平台名
        platforms = ["美团", "饿了么", "顺丰", "京东", "中通", "圆通", "韵达", "申通", "极兔"]
        for p in platforms:
            if p in text and not wm.caller_company:
                wm.caller_company = p
                break

        # 尝试提取配送地点
        location_keywords = ["放", "放在", "送到", "在", "门口", "楼下", "驿站", "快递柜", "前台"]
        for kw in location_keywords:
            idx = text.find(kw)
            if idx >= 0 and not wm.delivery_location:
                wm.delivery_location = text[idx:idx + 20]
                break

        # 设置目的
        if type_id == "food_delivery" and not wm.caller_purpose:
            wm.caller_purpose = "送外卖"
        elif type_id == "express" and not wm.caller_purpose:
            wm.caller_purpose = "送快递"

    # ============================================================
    # 公共接口
    # ============================================================

    def run(self, call_text: str, caller_number: str = "") -> dict:
        """
        处理一次来电。

        参数:
            call_text: STT 识别的来电文本
            caller_number: 来电号码

        返回:
            dict: 处理结果
        """
        # 重置对话记忆
        self.conversation_memory.reset()

        # 开始录音
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        if self.recorder:
            try:
                self.recorder.start(call_id, caller_number=caller_number)
            except Exception as e:
                logger.debug(f"开始录音失败: {e}")

        # 执行 LangGraph
        initial_state = {
            "call_text": call_text,
            "caller_number": caller_number,
        }
        try:
            result = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}")
            result = {
                "final_action": "error",
                "final_message": str(e),
                "agent_reply": "",
            }

        # 停止录音
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                logger.debug(f"停止录音失败: {e}")

        return dict(result)

    def resume_conversation(self, previous_result: dict, new_text: str) -> dict:
        """
        继续多轮对话。保持上下文，只需重新分类并处理新文本。

        参数:
            previous_result: 上一轮 run() 或 resume_conversation() 的返回值
            new_text: 来电者新说的话

        返回:
            dict: 处理结果
        """
        logger.info(f"继续对话: {new_text[:50]}...")

        prev_action = previous_result.get("final_action", "")
        prev_type = previous_result.get("type_id", "general")

        # 重新分类
        classify_result = self.classifier.classify(new_text)
        new_type = classify_result.type_id
        logger.info(f"  重新分类: {classify_result.call_type.name} (置信度={classify_result.confidence:.0%})")

        # 如果新分类变成诈骗/推销 → 立即拒接
        if new_type in ("scam", "scam_risk", "telemarketing", "game_promo"):
            caller_number = previous_result.get("caller_number", "")
            if caller_number:
                profile = self.caller_profile_store.get_or_create(caller_number)
                profile.add_call(new_type, classify_result.confidence)
                self.caller_profile_store.update(profile)
            card = self.card_builder.build_scam_log(
                scam_type=classify_result.call_type.name,
                reason="多轮对话中检测到诈骗特征",
                confidence=classify_result.confidence,
            )
            return {
                "final_action": "reject",
                "type_id": new_type,
                "call_type_name": classify_result.call_type.name,
                "confidence": classify_result.confidence,
                "classify_method": classify_result.method,
                "agent_reply": "",
                "notification_card": card.to_dict(),
            }

        # 如果是业务类对话，继续业务处理
        if new_type in ("food_delivery", "express", "taxi_arrived", "bank"):
            self.conversation_memory.add_user_message(new_text)
            self._extract_business_info(new_text, new_type)

            try:
                biz_result = self.business_handler.process_turn(
                    caller_text=new_text,
                    collected_info=self.conversation_memory.working_memory.to_dict(),
                    history=self.conversation_memory.short_term,
                )
                reply = biz_result.get("agent_text", "") if isinstance(biz_result, dict) else str(biz_result)
                is_complete = biz_result.get("is_complete", False) if isinstance(biz_result, dict) else True
            except Exception:
                call_type = get_call_type(new_type)
                reply = get_reply(call_type, self.presence.get_mode())
                is_complete = True

            self.conversation_memory.add_assistant_message(reply)

            card = None
            if is_complete:
                wm = self.conversation_memory.working_memory
                if new_type in ("food_delivery", "express"):
                    card = self.card_builder.build_delivery_card(
                        call_type="外卖" if new_type == "food_delivery" else "快递",
                        company=wm.caller_company or "未知平台",
                        item=wm.purpose_detail or "物品",
                        location=wm.delivery_location or "待确认",
                        notes=wm.delivery_notes or "",
                    )
                else:
                    card = self.card_builder.build_message_card(
                        caller=wm.caller_identity or "来电者",
                        message=wm.caller_purpose or new_text[:100],
                        relationship=classify_result.call_type.name,
                    )

            return {
                "final_action": "summary_card" if is_complete else "continue_conversation",
                "type_id": new_type,
                "call_type_name": classify_result.call_type.name,
                "confidence": classify_result.confidence,
                "classify_method": classify_result.method,
                "agent_reply": reply,
                "notification_card": card.to_dict() if card else None,
            }

        # 如果是熟人/家人/领导来电 → 继续追问或转接
        if new_type in ("friend", "family", "leader", "colleague", "client", "general"):
            self.conversation_memory.add_user_message(new_text)

            # 尝试用紧急转接评估
            try:
                ug_result = self.urgent_forwarder.assess(
                    new_text,
                    history=self.conversation_memory.short_term,
                )
                reply = ug_result.get("agent_text", "")
                should_forward = ug_result.get("should_forward", False)
                urgency = ug_result.get("urgency_level", "low")
            except Exception:
                reply = "好的，我知道了。请问还有其他事吗？"
                should_forward = False
                urgency = "low"

            self.conversation_memory.add_assistant_message(reply)

            if should_forward:
                card = self.card_builder.build_urgent_alert(
                    caller=self.conversation_memory.working_memory.caller_identity or "来电者",
                    reason=new_text[:100],
                    urgency_level=urgency,
                )
                return {
                    "final_action": "forward",
                    "type_id": new_type,
                    "call_type_name": classify_result.call_type.name,
                    "confidence": classify_result.confidence,
                    "classify_method": classify_result.method,
                    "agent_reply": reply,
                    "notification_card": card.to_dict(),
                }
            else:
                # 追问模式：检查对话轮数，超过3轮就结束
                turns = len(self.conversation_memory.short_term) // 2
                if turns >= 3:
                    card = self.card_builder.build_message_card(
                        caller=self.conversation_memory.working_memory.caller_identity or "来电者",
                        message=new_text[:100],
                        relationship=classify_result.call_type.name,
                    )
                    return {
                        "final_action": "general_reply",
                        "type_id": new_type,
                        "call_type_name": classify_result.call_type.name,
                        "confidence": classify_result.confidence,
                        "classify_method": classify_result.method,
                        "agent_reply": reply,
                        "notification_card": card.to_dict(),
                    }
                return {
                    "final_action": "continue_conversation",
                    "type_id": new_type,
                    "call_type_name": classify_result.call_type.name,
                    "confidence": classify_result.confidence,
                    "classify_method": classify_result.method,
                    "agent_reply": reply,
                }

        # 其他类型：当作新对话结束
        return {
            "final_action": "general_reply",
            "type_id": new_type,
            "call_type_name": classify_result.call_type.name,
            "confidence": classify_result.confidence,
            "classify_method": classify_result.method,
            "agent_reply": "",
        }

    def learn_habit(self, user_input: str) -> dict:
        """
        机主习惯学习接口。

        参数:
            user_input: 机主的自然语言输入

        返回:
            dict: 学习结果
        """
        return self.habit_learner.learn_from_conversation(user_input)

    def end_activity(self, keyword: str = "") -> Optional[str]:
        """结束当前活动。"""
        return self.habit_learner.end_current_activity(keyword)

    def get_habits_summary(self) -> str:
        """获取习惯摘要。"""
        return self.habit_learner.get_habits_summary()

    def get_today_stats(self) -> dict:
        """获取今日通话统计。"""
        return self.call_logger.get_today_stats()

    def get_caller_profile(self, phone_number: str) -> Optional[CallerProfile]:
        """查询来电者画像。"""
        return self.caller_profile_store.lookup(phone_number)

    def blacklist_caller(self, phone_number: str) -> None:
        """将号码加入黑名单。"""
        self.caller_profile_store.set_blacklist(phone_number)

    def whitelist_caller(self, phone_number: str) -> None:
        """将号码加入白名单。"""
        self.caller_profile_store.set_whitelist(phone_number)

    def get_profile_stats(self) -> dict:
        """获取画像统计。"""
        return self.caller_profile_store.get_stats()

    def get_recent_recordings(self, limit: int = 10) -> list:
        """获取最近的录音列表。"""
        if self.recorder:
            try:
                return self.recorder.list_recordings(limit=limit)
            except Exception:
                pass
        return []

    def get_recent_notifications(self, limit: int = 10) -> list:
        """获取最近的通知卡片。"""
        return self.notification_store.load_recent(limit)


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.core.config import load_and_validate_config

    config = load_and_validate_config("config.yaml")
    orch = CallOrchestrator(config)

    # 测试来电处理
    tests = [
        ("13800001111", "我是美团外卖的，你的餐到楼下了"),
        ("13800002222", "您好，我是市公安局的，你涉嫌洗钱案件"),
        ("13800003333", "妈，我今天晚上回家吃饭"),
        ("13800004444", "你的快递到菜鸟驿站了"),
        ("13800005555", "领导，明天下午有个紧急会议"),
        ("13800001111", "美团外卖，你上一单的餐到了"),
    ]

    for number, text in tests:
        result = orch.run(text, caller_number=number)
        print(f"\n来电: {text}")
        print(f"  号码: {number}")
        print(f"  类型: {result.get('call_type_name', '?')} "
              f"(置信度={result.get('confidence', 0):.0%})")
        print(f"  动作: {result.get('final_action')}")
        print(f"  回复: {result.get('agent_reply', '')[:60]}")
        if result.get('notification_card'):
            print(f"  📬 卡片: {result['notification_card'].get('title', '')}")

        # 查看画像
        profile = orch.get_caller_profile(number)
        if profile:
            print(f"  👤 画像: 信任={profile.trust_score:.2f} "
                  f"来电={profile.call_count}次 标签={profile.tags}")

    # 测试习惯学习
    print("\n" + "=" * 50)
    print("习惯学习测试:")
    habit_result = orch.learn_habit("我要自习一下午")
    print(f"  回复: {habit_result['reply']}")
    print(f"\n{orch.get_habits_summary()}")

    # 统计
    print(f"\n今日统计: {orch.get_today_stats()}")
    print(f"画像统计: {orch.get_profile_stats()}")
