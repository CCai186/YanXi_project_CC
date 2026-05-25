"""
AI 语音通话助手 — 主入口
=========================
启动语音助手，监听麦克风输入，通过多 Agent 调度器处理来电，
并将 Agent 的回复通过 TTS 播报。

运行方式:
    python src/main.py               # 默认模式：监听麦克风
    python src/main.py --text        # 文本模式：命令行输入文字模拟
    python src/main.py --test        # 测试模式：运行预设测试用例

完整流程:
    麦克风输入 → STT 识别 → Orchestrator 调度
        ├── 诈骗 → 拒接（不响应）
        ├── 业务 → 对话提取信息 → 生成卡片
        ├── 紧急 → 转接通知
        └── 普通 → 礼貌回复
    → TTS 播报 Agent 回复
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import CallOrchestrator
from src.utils.logger import setup_logger, load_config
from src.voice.tts import SpeechSynthesizer

logger = setup_logger(__name__)


def print_banner():
    """打印欢迎横幅。"""
    banner = r"""
╔══════════════════════════════════════════════════╗
║        AI 语音通话助手 — Voice Call Assistant     ║
║                                                  ║
║  📞 诈骗检测 → 拒接                                ║
║  📦 业务来电 → 对话 → 信息卡片                     ║
║  🔔 紧急来电 → 转接机主                            ║
║                                                  ║
║  技术栈: LangGraph + DeepSeek + Whisper + EdgeTTS ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


def run_text_mode(orchestrator: CallOrchestrator):
    """
    文本模式：通过命令行输入文字来模拟来电，不涉及真实麦克风。
    用于调试和测试 Agent 逻辑。

    支持多轮对话：当上一轮返回 continue_conversation 时，
    下一轮会保持上下文，直接继续对话，不再重复进行诈骗检测和意图分类。

    支持机主状态命令:
      /free              设为空闲
      /busy 开会 30min   设忙碌+原因+时长
      /dnd 睡觉了        设免打扰
      /driving           设开车模式
      /status            查看当前状态

    参数:
        orchestrator: 已初始化的调度器
    """
    from src.utils.presence import get_presence

    presence = get_presence()

    print("\n📝 文本模式 — 直接输入来电内容，输入 'quit' 退出")
    print("💡 提示: 支持多轮对话 | /busy /free /dnd /driving 切换机主状态\n")

    # 会话状态
    previous_result = None

    while True:
        try:
            # 显示当前状态标签
            status_tag = presence.get_summary()
            prompt = f"📞 [机主:{status_tag}] 来电内容: "
            call_text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if call_text.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not call_text:
            continue

        # --- 处理状态切换命令 ---
        if call_text.startswith("/"):
            handled = _handle_presence_command(call_text, presence)
            if handled:
                previous_result = None  # 状态切换，重置会话
            continue

        # --- 判断是继续上一轮对话还是新来电 ---
        if previous_result and previous_result.get("final_action") == "continue_conversation":
            print("(上下文已保持，直接继续对话...)")
            result = orchestrator.resume_conversation(previous_result, call_text)
        else:
            result = orchestrator.run(call_text)

        previous_result = result
        _display_result(result)

        if result.get("final_action") in ("summary_card", "reject", "forward", "general_reply"):
            previous_result = None


def _handle_presence_command(cmd: str, presence) -> bool:
    """
    处理机主状态切换命令。返回 True 表示已处理。

    参数:
        cmd: 以 / 开头的命令字符串
        presence: UserPresence 实例

    返回:
        bool: 是否识别并处理了命令
    """
    parts = cmd.split(maxsplit=2)
    action = parts[0].lower()

    if action == "/status":
        print(f"\n📊 当前状态: {presence.get_summary()}")
        print(f"   模式: {presence.get_mode()}")
        if presence.get_reason():
            print(f"   原因: {presence.get_reason()}")
        return True

    if action == "/free":
        presence.reset()
        print("✅ 已切换为「空闲」模式 — 正常处理所有来电\n")
        return True

    if action == "/busy":
        reason = parts[1] if len(parts) > 1 else ""
        duration = 0
        if reason:
            import re
            dur_match = re.search(r'(\d+)\s*(分钟|min|分)', reason)
            if dur_match:
                duration = int(dur_match.group(1))
                reason = re.sub(r'\d+\s*(分钟|min|分)', '', reason).strip()
        presence.set("busy", reason, duration)
        duration_str = f" ({duration}分钟)" if duration > 0 else ""
        print(f"✅ 已切换为「忙碌」模式{duration_str}: {reason}\n")
        return True

    if action == "/dnd":
        reason = parts[1] if len(parts) > 1 else ""
        presence.set("dnd", reason)
        print(f"✅ 已切换为「免打扰」模式: {reason}\n")
        return True

    if action == "/driving":
        presence.set("driving", "", duration_min=0)
        print("✅ 已切换为「开车中」模式\n")
        return True

    print(f"⚠️ 未知命令: {action}")
    print("   可用命令: /free  /busy <原因> <时长>  /dnd <原因>  /driving  /status\n")
    return True


def _detect_presence_from_speech(text: str, presence) -> bool:
    """
    从语音识别的文字中检测机主状态切换意图。
    如果检测到，直接更新 presence，返回 True。
    如果只是普通对话，返回 False。

    支持的语音触发词:
      "我要开会了" / "开始开会" / "我要忙了"                → busy
      "别打扰我" / "我要睡了" / "免打扰"                     → dnd
      "我要开车了" / "在开车" / "路上"                       → driving
      "我好了" / "忙完了" / "有空了" / "没事了"              → free

    参数:
        text: 语音识别的文本
        presence: UserPresence 实例

    返回:
        bool: 是否触发了状态切换
    """
    import re
    text_lower = text.lower().replace(" ", "")

    # --- 免打扰 ---
    dnd_patterns = [
        "别打扰我", "我要睡了", "我要休息", "免打扰", "不要打扰",
        "睡觉了", "睡了", "休息了", "别吵我", "静音",
        "开启免打扰", "打开免打扰",
    ]
    for pat in dnd_patterns:
        if pat in text_lower or pat in text:
            reason = text.replace(pat, "").strip()[:20]
            presence.set("dnd", reason if reason else "休息中")
            print(f"✅ 语音触发: 免打扰模式 → {presence.get_summary()}\n")
            return True

    # --- 开车 ---
    driving_patterns = ["我要开车", "在开车", "开车了", "开车中", "驾驶中", "上路了", "我在开车"]
    for pat in driving_patterns:
        if pat in text_lower or pat in text:
            presence.set("driving")
            print(f"✅ 语音触发: 开车模式\n")
            return True

    # --- 忙碌 ---
    busy_patterns = [
        "我要开会", "开会了", "开始开会", "我要忙", "忙了",
        "我要工作", "工作了", "我要学习", "学习了", "上课了",
        "有会", "开会中", "在开会", "在忙", "忙着", "有事",
    ]
    for pat in busy_patterns:
        if pat in text_lower or pat in text:
            reason = text.replace(pat, "").strip()[:20]
            presence.set("busy", reason if reason else "")
            print(f"✅ 语音触发: 忙碌模式 → {presence.get_summary()}\n")
            return True

    # --- 恢复空闲 ---
    free_patterns = [
        "我好了", "忙完了", "有空了", "没事了", "结束了",
        "开完会", "下课了", "下班了", "忙好了",
        "恢复", "取消免打扰", "关闭免打扰",
        "我回来了", "回来了",
    ]
    for pat in free_patterns:
        if pat in text_lower or pat in text:
            presence.reset()
            print(f"✅ 语音触发: 恢复空闲\n")
            return True

    return False


async def run_voice_mode(orchestrator: CallOrchestrator, tts: SpeechSynthesizer):
    """
    语音模式：用麦克风录音 → STT 识别 → 调度器处理 → TTS 播报回复。

    支持语音切换机主状态:
      "我要开会了"     → 忙碌模式
      "别打扰我"       → 免打扰
      "我要开车了"     → 开车模式
      "我忙完了"       → 恢复空闲

    参数:
        orchestrator: 已初始化的调度器
        tts: 语音合成器
    """
    from src.voice.stt import SpeechRecognizer
    from src.utils.presence import get_presence

    config = load_config("config.yaml")
    recognizer = SpeechRecognizer(config)
    presence = get_presence()

    print("\n🎤 语音模式 — 对麦克风说话，系统将自动识别和处理\n")
    print("💡 文字命令: /busy /free /dnd /driving /status  |  直接按 Enter = 开始录音模拟来电\n")
    print("按 Ctrl+C 退出\n")

    # 语音模式的会话上下文
    previous_result = None

    try:
        while True:
            # 显示当前状态，接受文字命令或按 Enter 录音
            status_tag = presence.get_summary()
            cmd = input(f"\n[机主:{status_tag}] 输入命令或按 Enter 录音: ").strip()

            # --- 文字命令模式 ---
            if cmd:
                if cmd.startswith("/"):
                    _handle_presence_command(cmd, presence)
                    previous_result = None
                else:
                    # 直接把输入文字当作来电内容（不需要录音）
                    if previous_result and previous_result.get("final_action") == "continue_conversation":
                        result = orchestrator.resume_conversation(previous_result, cmd)
                    else:
                        result = orchestrator.run(cmd)
                    previous_result = result
                    _display_result(result)
                    if result.get("final_action") in ("summary_card", "reject", "forward", "general_reply"):
                        previous_result = None
                continue

            # --- 语音录音模式 ---
            from src.agents.classifier import is_meaningless

            call_text = recognizer.listen_with_keyboard(timeout=60)

            if not call_text:
                print("⚠️ 未识别到语音，请重试")
                continue

            print(f"\n📝 识别结果: {call_text}")

            # 无意义检测
            if is_meaningless(call_text):
                print("⚠️ 无意义输入（杂音/重复），请重试")
                continue

            # 检测是否是状态切换命令
            if _detect_presence_from_speech(call_text, presence):
                previous_result = None
                continue

            # 3. 运行调度器（保持上下文）
            print("🔄 正在分析处理...")
            if previous_result and previous_result.get("final_action") == "continue_conversation":
                result = orchestrator.resume_conversation(previous_result, call_text)
            else:
                result = orchestrator.run(call_text)

            # 3. TTS 播报 Agent 回复
            agent_reply = result.get("agent_reply", "")
            if agent_reply:
                print(f"\n🔊 播放回复: {agent_reply}")
                await tts.speak(agent_reply)

            # 4. 显示处理结果并更新上下文
            _display_result(result)
            previous_result = result

            # 对话结束则重置上下文
            if result.get("final_action") in ("summary_card", "reject", "forward", "general_reply"):
                previous_result = None

    except KeyboardInterrupt:
        print("\n\n再见！")
    finally:
        recognizer.close()
        tts.close()


def run_test_mode(orchestrator: CallOrchestrator):
    """
    测试模式：运行预设的测试用例，验证三个场景。

    参数:
        orchestrator: 已初始化的调度器
    """
    print("\n🧪 测试模式 — 运行预设测试用例\n")

    test_cases = [
        ("🚫 诈骗场景", "您好，我是市公安局的，你涉嫌一起洗钱案件，请配合调查，把你的身份证号和银行卡号告诉我。"),
        ("📦 外卖场景", "喂，你好，我是美团外卖的，你的餐到了，现在在楼下，你下来拿还是给你放门卫？"),
        ("🔔 紧急场景", "喂？我是你妈，你爸刚才在家摔倒了，好像骨折了，我们正在去市医院的路上，你赶紧过来！"),
        ("📞 普通推销", "你好，我是XX教育的课程顾问，想跟您介绍一下我们的英语培训课程，现在有优惠活动..."),
    ]

    for i, (label, text) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {label}")
        print(f"来电: {text[:60]}...")
        print(f"{'='*60}")

        result = orchestrator.run(text)
        _display_result(result)

    print("\n✅ 全部测试完成")


def _display_result(result: dict):
    """
    在终端中显示处理结果。
    根据 final_action 展示不同的信息。

    参数:
        result: 调度器返回的结果字典
    """
    action = result.get("final_action", "unknown")
    agent_reply = result.get("agent_reply", "")

    call_type = result.get("call_type_name", "")
    confidence = result.get("confidence", 0)
    method = result.get("classify_method", "")

    print(f"\n{'─'*50}")
    print(f"📊 处理结果: {action}")
    if call_type:
        print(f"🏷️  来电类型: {call_type} (置信度={confidence:.0%}, {method})")
    if agent_reply:
        print(f"🤖 Agent 回复: {agent_reply}")
    print(f"{'─'*50}")

    if action == "reject":
        print("🚫 诈骗电话 — 已拒接！")
        sr = result.get("scam_result", {})
        print(f"   类型: {sr.get('scam_type', '未知')}")
        print(f"   置信度: {sr.get('confidence', 0):.0%}")
        print(f"   理由: {sr.get('reason', '')}")

    elif action == "summary_card":
        print("📋 业务来电 — 信息已记录")
        print(result.get("final_message", ""))

    elif action == "forward":
        # 判断是紧急转接还是朋友来电转接
        intent = result.get("intent", "")
        if intent == "urgent":
            print("🔔 紧急来电 — 正在转接机主！")
        else:
            print("📞 朋友来电 — 正在转接机主")
        print(result.get("final_message", ""))

    elif action == "general_reply":
        print("📝 普通来电 — 已代接")
        print(result.get("final_message", ""))

    elif action == "continue_conversation":
        print("💬 需要继续对话...")

    else:
        print(f"⚠️ 未知动作: {action}")
        print(result)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """
    主入口函数。解析命令行参数，选择运行模式。
    """
    parser = argparse.ArgumentParser(
        description="AI 语音通话助手 — 智能代接电话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/main.py              语音模式（默认）
  python src/main.py --text       文本模式（命令行输入）
  python src/main.py --test       测试模式（预设用例）
        """,
    )
    parser.add_argument(
        "--text", action="store_true",
        help="文本模式：通过命令行输入文字模拟来电"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="测试模式：运行预设的测试用例"
    )
    args = parser.parse_args()

    # --- 打印横幅 ---
    print_banner()

    # --- 加载配置 ---
    logger.info("正在加载配置...")
    config = load_config("config.yaml")

    # 检查 API Key
    api_key = config.get("llm", {}).get("api_key", "")
    if api_key in ("", "your-deepseek-api-key-here", "sk-your-key"):
        logger.warning("⚠️ 未配置 DeepSeek API Key！")
        logger.warning("请在 config.yaml 中设置 llm.api_key")
        logger.info("")
        logger.info("获取 API Key: https://platform.deepseek.com/")
        if not args.test and not args.text:
            logger.info("切换为测试模式（不需要 API Key 的文本演示）...")
            args.text = True

    # --- 初始化调度器 ---
    logger.info("正在初始化调度器和子 Agent...")
    orchestrator = CallOrchestrator(config)

    # --- 选择运行模式 ---
    if args.test:
        run_test_mode(orchestrator)
    elif args.text:
        run_text_mode(orchestrator)
    else:
        # 语音模式：需要麦克风和扬声器
        tts = SpeechSynthesizer(config)
        asyncio.run(run_voice_mode(orchestrator, tts))


if __name__ == "__main__":
    main()
