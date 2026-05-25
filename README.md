# AI 语音通话助手 — Voice Call Assistant

基于 LangGraph 多智能体的 AI 语音通话助手。在电话响铃之前代接来电，**16 种来电类型智能识别** + **机主状态感知** + **RAG 诈骗知识库**。

## 核心能力

| 场景 | 示例 | 处理 |
|------|------|------|
| 🚫 诈骗识别 | "你涉嫌洗钱，请配合调查" | 直接拒接 |
| 🍜 外卖/快递 | "美团外卖，餐到楼下了" | AI 对话收集信息→卡片通知 |
| 📞 朋友来电 | "喂，在干嘛呢" | 问清来意→判断是否转接 |
| 🔔 紧急转接 | "妈住院了，快过来" | 立即转接机主 |
| 📢 推销拦截 | "英语课程推荐了解一下" | 自动拒接 |

## 支持的来电类型（16 种）

诈骗电话 / 疑似诈骗 / 外卖配送 / 快递取件 / 打车到达 / 推销电话 / 游戏推广 / 家人来电 / 领导来电 / 熟人问候 / 同事协作 / 客户来电 / 银行来电 / 面试通知 / 无意义输入 / 其他

## 快速开始

### 1. 克隆

```bash
git clone <repo-url>
cd YanXi
```

### 2. 一键安装

**Windows**:
```bash
setup.bat
```

**macOS/Linux**:
```bash
pip install -r requirements.txt
python -m src.knowledge.dataset_loader --sample
python -m src.knowledge.embedder
```

### 3. 配置 API Key

编辑 `config.yaml`：
```yaml
llm:
  api_key: "sk-your-deepseek-api-key"  # https://platform.deepseek.com/
```

### 4. 启动

```bash
python src/main.py           # 语音模式（空格键录音）
python src/main.py --text    # 文本模式（打字模拟）
python src/main.py --test    # 测试模式
```

## 机主状态命令

| 命令 | 效果 |
|------|------|
| `/busy 开会 30分钟` | 忙碌模式，30分钟后自动恢复 |
| `/dnd 睡觉了` | 免打扰，仅极端紧急转接 |
| `/driving` | 开车模式，全代接 |
| `/free` | 恢复空闲 |

## 技术栈

- 多Agent: LangGraph
- LLM: DeepSeek V4 Pro (OpenAI 兼容 API)
- STT: Faster-Whisper large-v3 (GPU)
- TTS: Edge-TTS (免费)
- RAG: ChromaDB + BGE Embedding + 91 条内置样本
- 分类: 三级分类器 (关键词→RAG→LLM)

## 项目结构

```
YanXi/
├── setup.bat               # Windows 一键安装
├── config.yaml              # 配置文件（先填 API Key）
├── requirements.txt         # Python 依赖
├── src/
│   ├── main.py              # 主入口
│   ├── agents/              # 智能体（调度器/分类器/对话/转接）
│   ├── voice/               # 语音模块（STT/TTS）
│   ├── knowledge/           # RAG 知识库
│   ├── retrieval/           # RRF 融合 + 重排
│   └── utils/               # 日志/配置/状态管理
└── docs/                    # 文档

## 项目结构

```
YanXi/
├── requirements.txt          # Python 依赖
├── config.yaml               # 全局配置文件
├── README.md                 # 项目说明（本文件）
├── data/                     # 数据目录
│   └── processed/            # 处理后的数据
├── src/                      # 源代码
│   ├── main.py               # 入口程序
│   ├── voice/                # 语音模块
│   │   ├── stt.py            # 语音识别(语音→文字)
│   │   └── tts.py            # 语音合成(文字→语音)
│   ├── knowledge/            # 知识库模块
│   │   ├── dataset_loader.py # 数据集加载与预处理
│   │   ├── embedder.py       # 文本向量化 + ChromaDB 存储
│   │   └── retriever.py      # RAG 检索接口
│   ├── agents/               # 智能体模块
│   │   ├── orchestrator.py   # 主调度器(LangGraph 状态机)
│   │   ├── scam_detector.py  # 诈骗电话检测 Agent
│   │   ├── business_handler.py # 业务来电处理 Agent
│   │   └── urgent_forwarder.py # 紧急来电转接 Agent
│   └── utils/                # 工具模块
│       └── logger.py         # 日志工具
└── docs/                     # 文档
    └── architecture.md       # 架构详解
```

## 许可

内部研究项目
