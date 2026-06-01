# 言犀 — AI 智能通话助手 v4

基于 LangGraph 多智能体的 AI 语音通话助手，在你无法接电话时代为处理。**16 种来电类型智能识别** + **机主状态感知** + **习惯学习** + **来电者画像** + **通知卡片**。

## 核心能力

| 场景 | 示例 | 处理 |
|------|------|------|
| 诈骗识别 | "你涉嫌洗钱，请配合调查" | 直接拒接 + 诈骗卡片 |
| 外卖/快递 | "美团外卖，餐到楼下了" | AI 多轮对话 → 信息卡片通知 |
| 熟人/朋友 | "喂，在干嘛呢" | 问清来意 → 判断是否转接 |
| 家人紧急 | "妈住院了，快过来" | 紧急卡片 + 立即转接 |
| 推销拦截 | "英语课程推荐了解一下" | 自动拒接 |
| 习惯学习 | "我要开会一小时" | 自动切换忙碌 + 一小时自动恢复 |

## 支持的来电类型（16 种）

诈骗电话 / 疑似诈骗 / 外卖配送 / 快递取件 / 打车到达 / 推销电话 / 游戏推广 / 家人来电 / 领导来电 / 熟人问候 / 同事协作 / 客户来电 / 银行来电 / 面试通知 / 无意义输入 / 其他

## 快速开始

### 1. 环境

- Python 3.12+
- NVIDIA GPU + CUDA 12.1（语音模式需要，文本模式不需要）
- Windows / macOS / Linux

### 2. 安装

```bash
git clone <repo-url>
cd YanXi

# 安装依赖
pip install -r requirements.txt

# 可选：构建 RAG 知识库（首次使用）
python -m src.knowledge.dataset_loader --sample
python -m src.knowledge.embedder
```

### 3. 配置 API Key

编辑 `config.yaml` 第 9 行，填入 DeepSeek API Key：

```yaml
llm:
  api_key: "sk-your-deepseek-api-key"  # https://platform.deepseek.com/
```

也支持环境变量：`set DEEPSEEK_API_KEY=sk-xxx`（Windows）或 `export DEEPSEEK_API_KEY=sk-xxx`（macOS/Linux）

### 4. 启动

```bash
python src/main.py           # 语音模式（按 Enter 录音，打字输入命令）
python src/main.py --text    # 文本模式（纯打字模拟来电）
python src/main.py --test    # 测试模式（预设用例）
```

## 交互命令

语音模式和文本模式都支持以下命令：

| 命令 | 说明 |
|------|------|
| `/habit 我要自习一下午` | 学习习惯，自动设置状态 |
| `/habit 每天晚上11点到7点睡觉` | 周期性习惯 |
| `/habit end 自习` | 结束当前活动 |
| `/habits` | 查看已记录的习惯 |
| `/profile 13800001111` | 查看来电者画像 |
| `/blacklist 13800001111` | 加入黑名单 |
| `/whitelist 13800001111` | 加入白名单 |
| `/stats` | 查看今日通话统计 |
| `/recordings` | 查看最近录音 |
| `/notifications` | 查看最近通知卡片 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

## 语音模式交互

```
>> /habit 我要开会一小时    ← 打字设置习惯
>>                          ← 按 Enter 开始录音
[录音中...]
[STT] 喂你好我是送外卖的
回复: 你好，请问放门口可以吗？
>>                          ← 再按 Enter 继续对话
```

支持语音状态切换（对着麦克风说）：
- "我要开会了" / "开始开会" → 忙碌模式
- "别打扰我" / "我要睡了" → 免打扰
- "我要开车了" → 开车模式
- "我忙完了" / "有空了" → 恢复空闲

## 文本模式交互

```
>> 13800001111 我是美团外卖的，餐到楼下了  ← 带号码模拟来电
>> 你好，我是公安局的，你涉嫌洗钱           ← 不带号码也可
>> /stats                                  ← 查看统计
```

## v4 新特性

| 模块 | 功能 |
|------|------|
| 习惯学习 (HabitLearner) | 从对话中学习日程，自动调整机主状态 |
| 来电者画像 (CallerProfile) | 号码识别 + 信任评分 + 自动黑白名单 |
| 对话记忆 (ConversationMemory) | 三层记忆（短期/工作/长期），多轮上下文 |
| 通知卡片 (CardBuilder) | 4 种卡片类型（外卖/留言/紧急/诈骗） |
| 通话记录 (CallLogger) | JSONL 持久化，按日期归档 |
| 通话录音 (CallRecorder) | 录音元数据管理 |
| 知识库增强 (KnowledgeExpander) | 52 条内置诈骗话术+业务模板+紧急场景 |
| 统一 LLM 客户端 | 自动重试 + 超时 + JSON 解析 |
| 配置校验 | 默认值填充 + 环境变量覆盖 + 必填校验 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 多Agent框架 | LangGraph |
| 大模型 | DeepSeek (OpenAI 兼容 API) |
| 语音识别 | Faster-Whisper large-v3 (GPU, float16) |
| 语音合成 | Edge-TTS (zh-CN-XiaoxiaoNeural) |
| 向量数据库 | ChromaDB (语义检索) |
| Embedding | BAAI/bge-small-zh-v1.5 (512维) |
| 分类器 | 三级分类 (关键词 → RAG → LLM) |
| 检索融合 | RRF + 语义重排序 |

## 项目结构

```
YanXi/
├── config.yaml                    # 全局配置（先填 API Key）
├── requirements.txt               # Python 依赖
├── README.md                      # 项目说明
├── CLAUDE.md                      # AI 辅助开发说明
├── setup.bat                      # Windows 安装脚本
├── src/
│   ├── main.py                    # 主入口（语音/文本/测试）
│   ├── core/                      # 核心模块
│   │   ├── config.py              # 配置加载+校验+环境变量
│   │   └── llm_client.py          # 统一 LLM 客户端（重试/超时）
│   ├── agents/                    # 智能体
│   │   ├── orchestrator.py        # 主调度器（8节点 LangGraph）
│   │   ├── classifier.py          # 三级分类器
│   │   ├── call_types.py          # 16 种来电类型定义
│   │   ├── business_handler.py    # 业务来电处理（外卖/快递）
│   │   ├── urgent_forwarder.py    # 紧急来电评估+转接
│   │   ├── scam_detector.py       # 诈骗检测（RAG+LLM）
│   │   └── conversation_memory.py # 三层对话记忆
│   ├── voice/                     # 语音模块
│   │   ├── stt.py                 # 语音识别（Whisper）
│   │   ├── tts.py                 # 语音合成（Edge-TTS）
│   │   └── recorder.py            # 通话录音管理
│   ├── knowledge/                 # 知识库
│   │   ├── retriever.py           # RAG 检索（语义+关键词双模式）
│   │   ├── embedder.py            # 文本向量化 + ChromaDB
│   │   ├── dataset_loader.py      # 数据集加载
│   │   ├── caller_profile.py      # 来电者画像+信任评分
│   │   └── knowledge_expander.py  # 内置知识库扩展
│   ├── notification/              # 通知系统
│   │   └── card_builder.py        # 通知卡片生成+持久化
│   ├── store/                     # 存储模块
│   │   └── call_logger.py         # 通话记录（JSONL）
│   ├── retrieval/                 # 检索增强
│   │   ├── fusion.py              # RRF 融合 + 混合检索
│   │   └── reranker.py            # 语义重排序
│   ├── habit/                     # 习惯学习
│   │   └── habit_learner.py       # 习惯记录+状态推断
│   └── utils/                     # 工具
│       ├── logger.py              # 日志+配置加载
│       └── presence.py            # 机主状态管理
├── data/                          # 数据目录（运行时生成）
│   ├── chroma_db/                 # ChromaDB 向量库
│   ├── call_logs/                 # 通话记录
│   ├── notifications/             # 通知卡片
│   ├── habits/                    # 习惯数据
│   ├── recordings/                # 录音索引
│   └── processed/                 # 预处理文档
└── docs/                          # 文档
    ├── architecture.md
    └── changelog.md
```

## 配置说明

`config.yaml` 中所有配置项都有默认值，只需填 `llm.api_key` 即可运行。完整配置段：

| 配置段 | 说明 |
|--------|------|
| `llm` | DeepSeek API（key/url/model/temperature/重试） |
| `stt` | 语音识别（Whisper 模型/VAD/设备） |
| `tts` | 语音合成（Edge-TTS 声音/语速） |
| `rag` | 知识库（Embedding 模型/ChromaDB/语义开关） |
| `orchestrator` | 调度参数（最大轮次/诈骗阈值） |
| `habit` | 习惯学习（持久化路径/自动检测） |
| `notification` | 通知卡片持久化路径 |
| `call_log` | 通话记录持久化路径 |
| `recorder` | 录音存储路径 |
| `caller_profile` | 画像数据持久化路径 |
| `conversation_memory` | 记忆参数（短期记忆/上下文窗口） |
| `hybrid_retrieval` | 混合检索权重参数 |
| `logging` | 日志级别/格式/路径 |

环境变量覆盖：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`YANXI_LOG_LEVEL`

## 注意事项

- HuggingFace 被墙，所有模块已自动设置 `HF_ENDPOINT=https://hf-mirror.com`
- Windows 中文环境建议用 `python src/main.py` 直接运行（代码内已设置 UTF-8 输出）
- PyTorch 需 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu121`
- 语义检索（`rag.enable_semantic`）首次加载模型约需下载 ~100MB，网络不好建议关闭
- STT 的 VAD 阈值默认 0.02，安静环境下可能需调高

## 许可

内部研究项目
