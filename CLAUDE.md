# YanXi - AI 语音通话助手 v4

## 项目简介
基于 LangGraph 多智能体的 AI 语音通话助手，用于代接来电。
三大功能：诈骗电话拒接、业务来电处理（外卖/快递等）、紧急来电转接。
v4 新增：习惯学习、来电者画像、对话记忆、通知卡片、通话记录。

## 技术栈
- 语音识别: Faster-Whisper large-v3 (GPU, float16)
- 语音合成: Edge-TTS (zh-CN-XiaoxiaoNeural)
- 大模型: DeepSeek V4 Pro API (OpenAI 兼容接口)
- 多Agent框架: LangGraph (8节点状态图)
- 向量数据库: ChromaDB (RAG)
- 分类器: 三级分类 (关键词→RAG→LLM)，16 种来电类型

## 每次代码改进后必须做的事
**改进完代码后，必须在 `docs/changelog.md` 里追加一条记录**，包含：
- 问题是什么
- 原理分析
- 具体改了什么文件
- 效果对比

格式参考已有条目。

## 配置文件
- 全局配置: `config.yaml` (API Key、STT/TTS 参数、RAG 参数等 12 个配置段)
- API Key 在 `config.yaml` 的 `llm.api_key` 字段
- 支持环境变量覆盖: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `YANXI_LOG_LEVEL`

## 运行方式
```
python src/main.py           # 语音模式（按 Enter 录音，打字输命令）
python src/main.py --text    # 文本模式（/habit /profile /stats 等命令）
python src/main.py --test    # 测试模式
```

## 关键模块
- `src/agents/orchestrator.py` - 主调度器 v4（8节点 LangGraph）
- `src/agents/classifier.py` - 三级分类器（关键词→RAG→LLM）
- `src/agents/call_types.py` - 16 种来电类型定义+处理规则
- `src/agents/business_handler.py` - 业务处理（外卖/快递多轮对话）
- `src/agents/urgent_forwarder.py` - 紧急转接评估
- `src/agents/scam_detector.py` - 诈骗检测（RAG+LLM）
- `src/agents/conversation_memory.py` - 三层对话记忆（短期/工作/长期）
- `src/core/config.py` - 配置加载+校验+默认值+环境变量覆盖
- `src/core/llm_client.py` - 统一 LLM 客户端（重试+超时+JSON解析）
- `src/voice/stt.py` - 语音识别
- `src/voice/tts.py` - 语音合成
- `src/voice/recorder.py` - 通话录音管理
- `src/knowledge/retriever.py` - RAG检索（语义+关键词双模式）
- `src/knowledge/embedder.py` - 文本向量化 + ChromaDB
- `src/knowledge/caller_profile.py` - 来电者画像+信任评分+黑白名单
- `src/knowledge/knowledge_expander.py` - 内置知识库（52条）
- `src/knowledge/dataset_loader.py` - 数据集加载
- `src/habit/habit_learner.py` - 习惯学习+状态推断
- `src/notification/card_builder.py` - 通知卡片系统（4种类型）
- `src/store/call_logger.py` - 通话记录持久化（JSONL）
- `src/retrieval/fusion.py` - RRF融合+混合检索
- `src/retrieval/reranker.py` - 简单重排+语义重排
- `src/utils/presence.py` - 机主状态管理（free/busy/dnd/driving）
- `src/utils/logger.py` - 日志工具+配置加载

## 已知注意事项
- HuggingFace 被墙，所有模块已自动设置 `HF_ENDPOINT=https://hf-mirror.com`
- PyTorch 必须用 CUDA 版（`pip install torch --index-url https://download.pytorch.org/whl/cu121`）
- Windows 中文环境强制 `sys.stdout.reconfigure(encoding="utf-8")` 避免 GBK 报错
- STT 的 VAD 阈值很低（0.02），因为用户麦克风音量小
- 语义检索（`rag.enable_semantic`）首次需下载模型 ~100MB，网络不好建议关闭
- pyarrow 版本锁在 21.0.0，高版本与 torch MKL DLL 冲突导致 segfault
- numpy 版本锁在 1.x，2.x 与 scipy/scikit-learn 不兼容
