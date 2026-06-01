"""
AI 语音通话助手 — 主入口 v4 (完整版)
=====================================
集成所有模块的完整启动入口。

模块:
  ✅ LangGraph 多智能体调度
  ✅ 习惯学习 (HabitLearner)
  ✅ 通知卡片 (CardBuilder)
  ✅ 通话记录 (CallLogger)
  ✅ 通话录音 (CallRecorder) — 模块A
  ✅ 来电者画像 (CallerProfile) — 模块B
  ✅ 对话记忆 (ConversationMemory) — 模块D
  ✅ 知识库增强 (KnowledgeExpander + HybridRetriever) — 模块F

运行方式:
    python src/main.py               # 默认模式：监听麦克风
    python src/main.py --text        # 文本模式：命令行输入文字模拟
    python src/main.py --test        # 测试模式：运行预设测试用例

交互命令 (文本模式):
    /habit 我要自习一下午          → 学习习惯
    /habit 我每天晚上11点到7点睡觉  → 学习周期性习惯
    /habits                       → 查看已记录的习惯
    /habit end 自习                → 结束当前活动
    /profile 13800001111          → 查看来电者画像
    /blacklist 13800001111        → 加入黑名单
    /whitelist 13800001111        → 加入白名单
    /stats                       → 查看今日统计
    /recordings                  → 查看最近录音
    /notifications               → 查看最近通知
    /help                        → 显示帮助
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Windows 中文环境强制 UTF-8 输出，避免 GBK 编码报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import CallOrchestrator
from src.core.config import load_and_validate_config
from src.utils.logger import setup_logger
from src.voice.tts import SpeechSynthesizer

logger = setup_logger(__name__)


def print_banner():
    """打印欢迎横幅。"""
    banner = r"""
╔════════════════════════════════════════════════════════════╗
║          言犀 - AI 智能通话助手 v4 (完整版)                   ║
║                                                            ║
║  [诈骗检测] -> 拦截拒接                                      ║
║  [业务来电] -> 对话 -> 信息卡片                               ║
║  [紧急来电] -> 转接机主                                      ║
║  [习惯学习] -> 自动调整来电策略                               ║
║  [通话录音] -> 事后回溯                                      ║
║  [来电画像] -> 号码识别 + 信任评分                            ║
║  [对话记忆] -> 多轮上下文                                    ║
║  [知识增强] -> 混合检索 + 扩展知识库                          ║
║                                                            ║
║  输入 /help 查看所有命令                                     ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """打印帮助信息。"""
    help_text = """
╔════════════════════════════════════════════════════════════╗
║                      命令帮助                              ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  来电模拟:                                                  ║
║    直接输入文字模拟来电内容                                   ║
║    格式: [号码] 内容                                        ║
║    示例: 13800001111 我是美团外卖的                          ║
║    示例: 我是快递员 (无号码)                                 ║
║                                                            ║
║  习惯学习:                                                  ║
║    /habit 我要自习一下午                                    ║
║    /habit 我每天晚上11点到7点睡觉                            ║
║    /habit end 自习                                          ║
║    /habits                                                 ║
║                                                            ║
║  来电者画像:                                                ║
║    /profile 13800001111                                    ║
║    /blacklist 13800001111                                  ║
║    /whitelist 13800001111                                  ║
║                                                            ║
║  统计与记录:                                                ║
║    /stats                    今日通话统计                    ║
║    /recordings               最近录音列表                    ║
║    /notifications            最近通知卡片                    ║
║                                                            ║
║  其他:                                                      ║
║    /help                     显示帮助                       ║
║    /quit                     退出                           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def handle_command(text: str, orchestrator: CallOrchestrator) -> bool:
    """
    处理用户命令。

    返回:
        bool: True 表示继续运行，False 表示退出
    """
    text = text.strip()
    if not text:
        return True

    # 退出
    if text in ("/quit", "/exit", "/q"):
        print("再见！")
        return False

    # 帮助
    if text == "/help":
        print_help()
        return True

    # 习惯学习
    if text.startswith("/habit "):
        user_input = text[7:].strip()
        if user_input.lower().startswith("end"):
            keyword = user_input[3:].strip()
            result = orchestrator.end_activity(keyword)
            if result:
                print(f"[OK] 已结束活动: {result}")
            else:
                print("[INFO] 没有正在进行的活动")
        else:
            result = orchestrator.learn_habit(user_input)
            print(f"[HABIT] {result.get('reply', '已记录')}")
            if result.get("mode_change"):
                print(f"   状态变化: {result['mode_change']}")
        return True

    # 查看习惯
    if text == "/habits":
        print(orchestrator.get_habits_summary())
        return True

    # 查看画像
    if text.startswith("/profile "):
        number = text[9:].strip()
        profile = orchestrator.get_caller_profile(number)
        if profile:
            print(f"[PROFILE] 号码: {profile.phone_number}")
            print(f"   来电次数: {profile.call_count}")
            print(f"   信任分数: {profile.trust_score:.2f}")
            print(f"   标签: {profile.tags}")
            print(f"   黑名单: {'是' if profile.is_blacklisted else '否'}")
            print(f"   白名单: {'是' if profile.is_whitelisted else '否'}")
            if profile.contact_name:
                print(f"   联系人: {profile.contact_name}")
            if profile.notes:
                print(f"   备注: {profile.notes}")
        else:
            print(f"[INFO] 未找到号码 {number} 的画像")
        return True

    # 黑名单
    if text.startswith("/blacklist "):
        number = text[11:].strip()
        orchestrator.blacklist_caller(number)
        print(f"[BLACKLIST] 已将 {number} 加入黑名单")
        return True

    # 白名单
    if text.startswith("/whitelist "):
        number = text[11:].strip()
        orchestrator.whitelist_caller(number)
        print(f"[WHITELIST] 已将 {number} 加入白名单")
        return True

    # 统计
    if text == "/stats":
        stats = orchestrator.get_today_stats()
        profile_stats = orchestrator.get_profile_stats()
        print(f"[STATS] 今日通话统计:")
        print(f"   总计: {stats.get('total', 0)} 次")
        print(f"   按动作: {stats.get('by_action', {})}")
        print(f"   按类型: {stats.get('by_type', {})}")
        print(f"[STATS] 来电者画像统计:")
        print(f"   总画像: {profile_stats.get('total_profiles', 0)} 个")
        print(f"   黑名单: {profile_stats.get('blacklisted', 0)} 个")
        print(f"   白名单: {profile_stats.get('whitelisted', 0)} 个")
        print(f"   平均信任: {profile_stats.get('avg_trust', 0):.2f}")
        return True

    # 录音列表
    if text == "/recordings":
        recordings = orchestrator.get_recent_recordings(5)
        if recordings:
            print(f"[RECORDINGS] 最近录音:")
            for r in recordings:
                print(f"   {r.call_id} | {r.duration_sec:.1f}s | "
                      f"{r.call_type or '?'} | {r.filepath}")
        else:
            print("[INFO] 暂无录音记录")
        return True

    # 通知列表
    if text == "/notifications":
        notifications = orchestrator.get_recent_notifications(5)
        if notifications:
            print(f"[NOTIFICATIONS] 最近通知:")
            for n in notifications:
                print(f"   {n.title} | {n.body[:30]} | {n.priority}")
        else:
            print("[INFO] 暂无通知记录")
        return True

    # 未知命令
    if text.startswith("/"):
        print(f"[?] 未知命令: {text}，输入 /help 查看帮助")
        return True

    return True  # 不是命令，继续处理


def parse_caller_input(text: str) -> tuple[str, str]:
    """
    解析来电输入，提取号码和内容。

    格式: [号码] 内容
    示例: "13800001111 我是美团外卖的" → ("13800001111", "我是美团外卖的")
    示例: "我是快递员" → ("", "我是快递员")
    """
    import re
    match = re.match(r'^(\d{5,15})\s+(.+)$', text)
    if match:
        return match.group(1), match.group(2)
    return "", text


def run_text_mode(orchestrator: CallOrchestrator):
    """文本交互模式。"""
    print("\n[TEST] 文本模式已启动（输入 /help 查看命令）\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            if not handle_command(user_input, orchestrator):
                break
            continue

        # 解析来电
        caller_number, call_text = parse_caller_input(user_input)

        # 处理来电
        result = orchestrator.run(call_text, caller_number=caller_number)

        # 显示结果
        print(f"\n[RESULT] 处理结果:")
        print(f"   类型: {result.get('call_type_name', '?')} "
              f"(置信度={result.get('confidence', 0):.0%}, "
              f"方法={result.get('classify_method', '?')})")
        print(f"   动作: {result.get('final_action', '?')}")

        presence_mode = result.get('presence_mode', 'free')
        if presence_mode != 'free':
            print(f"   机主状态: {presence_mode} ({result.get('presence_reason', '')})")

        if result.get('agent_reply'):
            print(f"   言犀: {result['agent_reply']}")

        if result.get('notification_card'):
            card = result['notification_card']
            print(f"   卡片: {card.get('title', '')} - {card.get('body', '')[:40]}")

        # 显示画像信息
        if caller_number:
            profile = orchestrator.get_caller_profile(caller_number)
            if profile and profile.call_count > 0:
                print(f"   画像: 信任={profile.trust_score:.2f} "
                      f"来电={profile.call_count}次 标签={profile.tags}")

        print()


def run_test_mode(orchestrator: CallOrchestrator):
    """测试模式：运行预设测试用例。"""
    print("\n[TEST] 运行测试用例...\n")

    tests = [
        ("13800001111", "我是美团外卖的，你的餐到楼下了", "food_delivery"),
        ("13800002222", "您好，我是市公安局的，你涉嫌洗钱案件", "scam"),
        ("13800003333", "你的快递到菜鸟驿站了", "express"),
        ("13800004444", "妈，我今天晚上回家吃饭", "family"),
        ("13800005555", "领导，明天下午有个紧急会议", "leader"),
        ("13800001111", "美团外卖，你上一单的餐到了", "food_delivery"),  # 同号二次
        ("13800002222", "你好，我是检察院的", "scam"),  # 诈骗号二次
    ]

    for number, text, expected in tests:
        result = orchestrator.run(text, caller_number=number)
        actual = result.get('type_id', '?')
        action = result.get('final_action', '?')
        reply = result.get('agent_reply', '')[:50]
        card = result.get('notification_card')
        card_info = f'[CARD] {card["title"]}' if card else ''

        match = '[OK]' if expected in actual or actual in expected else '[?]'
        print(f"{match} [{number}] {text[:25]}...")
        print(f"   分类: {actual} | 动作: {action} | {card_info}")
        print(f"   回复: {reply}...")

        # 画像
        profile = orchestrator.get_caller_profile(number)
        if profile:
            print(f"   信任={profile.trust_score:.2f} 来电={profile.call_count}次 "
                  f"标签={profile.tags}")
        print()

    # 测试习惯学习
    print("=" * 60)
    print("[HABIT] 习惯学习测试:")
    print("=" * 60)
    habit_result = orchestrator.learn_habit("我要自习一下午")
    print(f"回复: {habit_result.get('reply', '')}")
    print(f"\n{orchestrator.get_habits_summary()}")

    # 测试黑名单
    print("\n" + "=" * 60)
    print("[BLACKLIST] 黑名单测试:")
    print("=" * 60)
    orchestrator.blacklist_caller("13800002222")
    print("已将诈骗号码加入黑名单")

    # 再次来电
    result = orchestrator.run("你好，我是检察院的", caller_number="13800002222")
    print(f"黑名单号码来电 → 动作: {result.get('final_action')}")

    # 统计
    print("\n" + "=" * 60)
    print("[STATS] 统计:")
    print("=" * 60)
    stats = orchestrator.get_today_stats()
    print(f"今日通话: {stats.get('total', 0)} 次")
    print(f"按动作: {stats.get('by_action', {})}")
    print(f"按类型: {stats.get('by_type', {})}")

    profile_stats = orchestrator.get_profile_stats()
    print(f"画像总数: {profile_stats.get('total_profiles', 0)}")
    print(f"黑名单: {profile_stats.get('blacklisted', 0)}")
    print(f"白名单: {profile_stats.get('whitelisted', 0)}")

    print("\n[DONE] 测试完成！")


async def run_voice_mode(orchestrator: CallOrchestrator, tts: SpeechSynthesizer):
    """语音模式：输入命令或按 Enter 录音。"""
    try:
        from src.voice.stt import SpeechRecognizer
        stt = SpeechRecognizer(orchestrator.config)
    except Exception as e:
        logger.error(f"STT 初始化失败: {e}")
        logger.info("请检查 faster-whisper 是否正确安装")
        return

    from src.utils.presence import get_presence
    from src.agents.classifier import is_meaningless
    presence = get_presence()

    print("\n[VOICE] 语音模式已启动")
    print("  输入文字命令直接执行，按 Enter 开始录音模拟来电")
    print("  命令: /habit /habits /profile /stats /help /quit")
    print("  按 Ctrl+C 退出\n")

    previous_result = None  # 多轮对话上下文

    while True:
        try:
            # 先接受文字输入，按 Enter = 开始录音
            cmd = input(">> ").strip()

            # --- 文字命令 ---
            if cmd:
                if cmd.startswith("/"):
                    if not handle_command(cmd, orchestrator):
                        break
                    previous_result = None
                    continue
                else:
                    # 直接输入文字当作来电内容（不录音）
                    text = cmd
                    if previous_result and previous_result.get("final_action") == "continue_conversation":
                        result = orchestrator.resume_conversation(previous_result, text)
                    else:
                        result = orchestrator.run(text)
                    previous_result = result
                    if result.get('agent_reply'):
                        print(f"回复: {result['agent_reply']}")
                        await tts.speak(result['agent_reply'])
                    _display_voice_result(result)
                    if result.get("final_action") in ("summary_card", "reject", "forward", "general_reply"):
                        previous_result = None
                    continue

            # --- 按 Enter 开始录音 ---
            print("[录音中...]")
            text = stt.listen(timeout=60.0)
            if not text:
                print("[INFO] 未识别到语音")
                continue

            print(f"[STT] {text}")

            # 无意义检测
            if is_meaningless(text):
                print("[INFO] 无意义输入，跳过")
                continue

            # 语音状态切换检测
            if _detect_presence_from_speech(text, presence):
                previous_result = None
                continue

            # 多轮对话
            if previous_result and previous_result.get("final_action") == "continue_conversation":
                result = orchestrator.resume_conversation(previous_result, text)
            else:
                result = orchestrator.run(text)

            previous_result = result

            if result.get('agent_reply'):
                print(f"回复: {result['agent_reply']}")
                await tts.speak(result['agent_reply'])

            _display_voice_result(result)

            if result.get("final_action") in ("summary_card", "reject", "forward", "general_reply"):
                previous_result = None

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理出错: {e}")


def _display_voice_result(result: dict):
    """显示语音模式处理结果。"""
    action = result.get("final_action", "?")
    type_name = result.get("call_type_name", "?")
    print(f"[{action}] 类型={type_name} 置信度={result.get('confidence', 0):.0%}")
    card = result.get('notification_card')
    if card:
        print(f"  卡片: {card.get('title', '')}")


def _detect_presence_from_speech(text: str, presence) -> bool:
    """
    从语音识别文字中检测机主状态切换意图。
    支持的语音触发词:
      "我要开会了"/"开始开会"/"我要忙了" → busy
      "别打扰我"/"我要睡了"/"免打扰" → dnd
      "我要开车了"/"在开车"/"路上" → driving
      "我好了"/"忙完了"/"有空了"/"没事了" → free
    """
    text_lower = text.lower().replace(" ", "")

    dnd_patterns = [
        "别打扰我", "我要睡了", "我要休息", "免打扰", "不要打扰",
        "睡觉了", "睡了", "休息了", "别吵我", "静音",
    ]
    for pat in dnd_patterns:
        if pat in text_lower or pat in text:
            presence.set("dnd", "休息中")
            print(f"[PRESENCE] 语音触发: 免打扰模式")
            return True

    driving_patterns = ["我要开车", "在开车", "开车了", "开车中", "驾驶中", "上路了", "我在开车"]
    for pat in driving_patterns:
        if pat in text_lower or pat in text:
            presence.set("driving")
            print(f"[PRESENCE] 语音触发: 开车模式")
            return True

    busy_patterns = [
        "我要开会", "开会了", "开始开会", "我要忙", "忙了",
        "我要工作", "工作了", "我要学习", "学习了", "上课了",
        "有会", "开会中", "在开会", "在忙", "忙着", "有事",
    ]
    for pat in busy_patterns:
        if pat in text_lower or pat in text:
            presence.set("busy", "")
            print(f"[PRESENCE] 语音触发: 忙碌模式")
            return True

    free_patterns = [
        "我好了", "忙完了", "有空了", "没事了", "结束了",
        "开完会", "下课了", "下班了", "忙好了",
        "恢复", "取消免打扰", "关闭免打扰",
        "我回来了", "回来了",
    ]
    for pat in free_patterns:
        if pat in text_lower or pat in text:
            presence.reset()
            print(f"[PRESENCE] 语音触发: 恢复空闲")
            return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="言犀 — AI 智能通话助手 v4",
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
    config = load_and_validate_config("config.yaml")

    # 检查 API Key
    api_key = config.get("llm", {}).get("api_key", "")
    if api_key in ("", "your-deepseek-api-key-here", "sk-your-key"):
        logger.warning("⚠️ 未配置 DeepSeek API Key！")
        logger.warning("请在 config.yaml 中设置 llm.api_key 或设置环境变量 DEEPSEEK_API_KEY")
        logger.info("")
        logger.info("获取 API Key: https://platform.deepseek.com/")
        if not args.test and not args.text:
            logger.info("切换为文本模式（不需要 API Key 的基础功能仍可用）...")
            args.text = True

    # --- 初始化调度器 ---
    logger.info("正在初始化调度器和子模块...")
    try:
        orchestrator = CallOrchestrator(config)
        logger.info("✅ 所有模块初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        logger.info("尝试使用基础模式...")
        sys.exit(1)

    # --- 选择运行模式 ---
    if args.test:
        run_test_mode(orchestrator)
    elif args.text:
        run_text_mode(orchestrator)
    else:
        tts = SpeechSynthesizer(config)
        asyncio.run(run_voice_mode(orchestrator, tts))


if __name__ == "__main__":
    main()
