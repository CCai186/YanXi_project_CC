# YanXi - AI 语音通话助手

## 项目简介
基于 LangGraph 多智能体的 AI 语音通话助手，用于代接来电。
三大功能：诈骗电话拒接、业务来电处理（外卖/快递等）、紧急来电转接。

## 技术栈
- 语音识别: Faster-Whisper large-v3 (GPU, float16)
- 语音合成: Edge-TTS (zh-CN-XiaoxiaoNeural)
- 大模型: DeepSeek V4 Pro API (OpenAI 兼容接口)
- 多Agent框架: LangGraph
- 向量数据库: ChromaDB (RAG)
- 数据集: TeleAntiFraud-28k

## 每次代码改进后必须做的事
**改进完代码后，必须在 `docs/changelog.md` 里追加一条记录**，包含：
- 问题是什么
- 原理分析  
- 具体改了什么文件
- 效果对比

格式参考已有条目。

## 配置文件
- 全局配置: `config.yaml` (API Key、STT/TTS 参数、RAG 参数)
- API Key 在 `config.yaml` 的 `llm.api_key` 字段

## 运行方式
```
python src/main.py           # 语音模式
python src/main.py --text    # 文本模式（支持 /busy /free /dnd /driving）
python src/main.py --test    # 测试模式
```

## 关键模块
- `src/agents/orchestrator.py` - 主调度器（统一评估LLM调用 + 多轮对话）
- `src/agents/scam_detector.py` - 诈骗检测
- `src/agents/business_handler.py` - 业务处理（外卖/快递）
- `src/agents/urgent_forwarder.py` - 紧急转接
- `src/voice/stt.py` - 语音识别
- `src/voice/tts.py` - 语音合成
- `src/knowledge/retriever.py` - RAG检索（语义+关键词双模式）
- `src/knowledge/dataset_loader.py` - 数据集加载（HF/本地/内置）
- `src/utils/presence.py` - 机主状态管理（free/busy/dnd/driving）

## 已知注意事项
- HuggingFace 被墙，所有模块已自动设置 `HF_ENDPOINT=https://hf-mirror.com`
- PyTorch 必须用 CUDA 版（`pip install torch --index-url https://download.pytorch.org/whl/cu121`）
- Windows 中文环境可能出现 GBK 编码问题，文件避免中文注释在 requirements.txt 中
- STT 的 VAD 阈值很低（0.02），因为用户麦克风音量小
