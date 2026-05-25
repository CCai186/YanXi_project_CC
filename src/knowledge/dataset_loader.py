"""
数据集加载与预处理模块
-----------------------
支持多种数据源加载诈骗电话检测数据：
  1. HuggingFace (hf-mirror.com 镜像)
  2. Kaggle 本地下载 (data/raw/ 目录)
  3. 内置样例数据（无需下载，直接可用）

数据来源:
  - Kaggle: https://www.kaggle.com/datasets/divyanshsharma23/research28k
  - HuggingFace: JimmyMa99/TeleAntiFraud

使用方式:
  python -m src.knowledge.dataset_loader             # 自动选择最优来源
  python -m src.knowledge.dataset_loader --sample    # 使用内置样例
  python -m src.knowledge.dataset_loader --local     # 从本地 data/raw/ 加载

输出:
  将处理后的文档保存到 data/processed/scam_documents.json
"""

import csv
import json
import os
import time
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 输出路径
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "scam_documents.json"

# Kaggle 本地数据目录 (用户手动下载后放这里)
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================
# 内置样例数据 (无需联网，立即可用)
# 覆盖了 12 种常见诈骗类型和正常通话场景
# ============================================================

BUILTIN_SCAM_SAMPLES = [
    # --- 冒充公检法 ---
    {
        "sentence": "你好，我是市公安局刑侦支队的王警官，警号054328。你名下的一张银行卡涉嫌一起洗钱案件，现在需要你配合我们的调查。请你提供身份证号码和银行卡号，我们需要核实你的资金流向。如果不配合的话，我们只能发通缉令了。",
        "label": "fraud",
        "fraud_type": "冒充公检法",
        "reasoning": "公检法机关不会通过电话办案，不会索要银行卡号和密码，更不会以'发通缉令'威胁。这是典型的冒充公检法诈骗话术。"
    },
    {
        "sentence": "喂您好，我是最高人民法院的，您有一个传票没有签收，案件编号是粤0612刑初字第384号。您涉嫌一起跨境洗钱案，现在需要把您的资金转到我们指定的安全账户进行核查，核查完毕后会原路退还。",
        "label": "fraud",
        "fraud_type": "冒充公检法",
        "reasoning": "法院传票不会通过电话通知，更不会要求转账到'安全账户'。'安全账户'是诈骗常用概念，公检法机关不存在此物。"
    },
    {
        "sentence": "您好，这边是市公安局出入境管理科，我们发现您的护照有异常出境记录，涉嫌非法出入境。请添加我们的QQ号，视频做笔录，并缴纳5000元保证金，否则将限制您出境。",
        "label": "fraud",
        "fraud_type": "冒充公检法",
        "reasoning": "公安机关不会通过QQ视频做笔录，不会要求缴纳任何形式的保证金。这是冒充公检法诈骗的变种。"
    },
    # --- 冒充客服 ---
    {
        "sentence": "您好，我是京东金融的客服，工号J00852。您之前开通的京东白条服务，由于利率超过国家规定，现在需要帮您注销，否则会影响您的个人征信。请您下载一个'会议通'APP，我来指导您操作。",
        "label": "fraud",
        "fraud_type": "冒充客服",
        "reasoning": "正规平台客服不会以'影响征信'来威胁用户下载第三方APP。这是冒充客服+虚假征信诈骗的典型手法，目的是诱导安装远程控制软件。"
    },
    {
        "sentence": "喂，您好，我是淘宝商城的客服。您在3月15日购买的商品因为质量问题需要召回退款。请添加我们的QQ群，扫二维码填写银行卡信息，我们会双倍退款给您。",
        "label": "fraud",
        "fraud_type": "冒充客服",
        "reasoning": "正规退款会原路返回到支付账户，不需要添加QQ群或填写银行卡信息。'双倍退款'是利用贪小便宜心理的诈骗手法。"
    },
    {
        "sentence": "您好，这里是移动通信客服，您的手机号码涉嫌发送大量垃圾短信，将在2小时后被停机。如需申诉请按9转人工处理，我们将核实您的身份信息。",
        "label": "fraud",
        "fraud_type": "冒充客服",
        "reasoning": "运营商发送的停机通知通常通过短信而非电话，且不会在电话中索要身份信息。这是冒充运营商客服的社会工程学攻击。"
    },
    # --- 投资理财诈骗 ---
    {
        "sentence": "您好，我是华泰证券的投资顾问小陈。我们最近推出了一款稳赚不赔的量化交易产品，月化收益15%，最低投入5万起。很多客户都已经赚到钱了，您要不要了解一下？我们有专业的操盘团队，保证本金安全。",
        "label": "fraud",
        "fraud_type": "投资理财诈骗",
        "reasoning": "任何承诺'稳赚不赔'、月收益高达15%的投资产品都是诈骗。正规金融机构不允许承诺保本保收益。"
    },
    {
        "sentence": "大哥您好，我是做虚拟货币合约的，我们有内部消息，下一波比特币会大涨。现在入金1万，一个月变10万。我们有自己的交易所，充值和提现都非常方便，您先下载我们的APP看看。",
        "label": "fraud",
        "fraud_type": "投资理财诈骗",
        "reasoning": "'内部消息'和'自有交易所'是虚拟货币杀猪盘的典型特征，目的是诱导用户将资金转入诈骗团伙控制的平台。"
    },
    # --- 贷款诈骗 ---
    {
        "sentence": "您好，我是平安信贷的客户经理。我们近期推出了一款无抵押、无担保、低利率的贷款产品，额度最高50万，最快当天放款。您只需要先缴纳2000元的手续费和保证金，就可以马上办理。",
        "label": "fraud",
        "fraud_type": "贷款诈骗",
        "reasoning": "正规贷款机构不会在放款前要求缴纳手续费或保证金。'先交钱后放款'是所有贷款诈骗的共同特征。"
    },
    # --- 兼职刷单诈骗 ---
    {
        "sentence": "招聘！招聘！在家手机操作，日赚300-800元，无需经验。只要你有手机，会网购就可以。工作内容是帮商家刷销量，每单佣金20-50元，多劳多得。先交299元入会费，当天回本。",
        "label": "fraud",
        "fraud_type": "兼职刷单诈骗",
        "reasoning": "'日赚300-800'远超合理范围，且要求先交入会费，这是典型的兼职刷单诈骗。正规兼职不会要求预付任何费用。"
    },
    {
        "sentence": "姐妹，我这边有一个超简单的兼职，就是帮电商平台的商家点点赞、刷刷好评，一单3-5块钱。你现在先充100块钱进去做任务，做完马上提现，五分钟就能赚回来。我做了很久了一直很稳。",
        "label": "fraud",
        "fraud_type": "兼职刷单诈骗",
        "reasoning": "先充值再做任务是刷单诈骗的标准模式。前期小额返利诱使受害者加大投入，最终无法提现。"
    },
    # --- 中奖诈骗 ---
    {
        "sentence": "您好，恭喜您的手机号码在'星光大道'节目抽奖活动中被选中为幸运观众，获得了我们的一等奖，奖金为30万元加笔记本电脑一台！请您先缴纳个人所得税和公证费共计3800元，我们将在24小时内将奖金打入您的账户。",
        "label": "fraud",
        "fraud_type": "中奖诈骗",
        "reasoning": "正规中奖不需要先缴纳税费，税费在奖金发放时直接扣除。要求先交'个人所得税'和'公证费'是中奖诈骗的标准话术。"
    },
    # --- 杀猪盘/交友诈骗 ---
    {
        "sentence": "宝宝，我在新加坡做IT，最近发现了一个博彩网站的漏洞，每天固定时间投注稳赢。我已经帮好几个朋友赚了钱。你把钱转给我，我帮你操作，保证一周翻倍。我们是恋人，我怎么会骗你呢？",
        "label": "fraud",
        "fraud_type": "杀猪盘诈骗",
        "reasoning": "网恋对象推荐博彩/投资是杀猪盘诈骗的标准模式。'漏洞稳赢'是虚构的，目的是利用感情操控受害者转账。"
    },
    # --- 虚假绑架/事故诈骗 ---
    {
        "sentence": "喂！你是XXX的家长吗？你儿子在学校出事了，现在正在市中心医院抢救，需要马上做手术。你先转3万块钱过来，账户是6222xxxx，快快快，晚了就来不及了！",
        "label": "fraud",
        "fraud_type": "虚假事故诈骗",
        "reasoning": "医院不会用催促转账的方式通知家属。遇到此类情况应先联系学校或医院官方电话核实，切勿在慌乱中转账。"
    },
    {
        "sentence": "爸！妈！救我！我被绑架了！他们说不给钱就砍我的手！快给他们转10万块钱！",
        "label": "fraud",
        "fraud_type": "虚假绑架诈骗",
        "reasoning": "利用AI语音合成技术模拟亲人声音是新兴诈骗手法。遇到此类电话应冷静核实，先联系本人或报警，不要直接转账。"
    },
    # --- 更多正常通话样本 (平衡知识库，避免 RAG 误判) ---
    {
        "sentence": "喂，在干嘛呢？晚上有没有空，一起吃个饭？",
        "label": "normal", "fraud_type": "",
        "reasoning": "朋友闲聊约饭，正常的社交来电。"
    },
    {
        "sentence": "喂，最近怎么样？好久没联系了，周末聚聚？",
        "label": "normal", "fraud_type": "",
        "reasoning": "朋友问候+聚会邀请，正常的社交来电。"
    },
    {
        "sentence": "在吗？我是你大学同学，好久不见了，加个微信吧。",
        "label": "normal", "fraud_type": "",
        "reasoning": "同学联系，正常的社交来电。"
    },
    {
        "sentence": "喂，我是隔壁老王啊，你家水管好像漏水了，赶紧回来看看。",
        "label": "normal", "fraud_type": "",
        "reasoning": "邻居通知家中问题，正常的紧急来电，非诈骗。"
    },
    {
        "sentence": "你好，我是滴滴司机，我到定位的地方了，你在哪？",
        "label": "normal", "fraud_type": "",
        "reasoning": "网约车司机到达通知，正常的业务来电。"
    },
    {
        "sentence": "喂，我是你表哥，晚上去外婆家吃饭，你来不来？",
        "label": "normal", "fraud_type": "",
        "reasoning": "亲戚约饭，正常的家庭社交来电。"
    },
    {
        "sentence": "您好，这边是XX公司HR，恭喜您通过了面试，想和您确认一下入职时间。",
        "label": "normal", "fraud_type": "",
        "reasoning": "正常的面试通知来电。"
    },
    {
        "sentence": "喂老板，明天那个客户会议改到下午3点了，我发您邮箱了。",
        "label": "normal", "fraud_type": "",
        "reasoning": "正常的工作沟通来电。"
    },
    {
        "sentence": "您好，我是小区物业，楼下要停水检修，大概2小时，请提前备水。",
        "label": "normal", "fraud_type": "",
        "reasoning": "正常的物业通知来电。"
    },
    {
        "sentence": "喂，是我，你在哪？我到了，怎么没看到你？",
        "label": "normal", "fraud_type": "",
        "reasoning": "朋友约见面的正常来电。"
    },

    # --- 正常通话样本 (用于对比学习) ---
    {
        "sentence": "喂你好，我是顺丰快递的，有你一个包裹到了，你现在在家吗？我给你送上来还是放在菜鸟驿站？",
        "label": "normal",
        "fraud_type": "",
        "reasoning": "正常的快递通知电话，询问送货方式，不索要敏感信息，没有威胁或利诱。"
    },
    {
        "sentence": "你好，我是美团外卖的，你的餐到了，我放在楼下外卖柜了，取餐码发到你手机上了，请及时取餐哦。",
        "label": "normal",
        "fraud_type": "",
        "reasoning": "正常的外卖送达通知，告知取餐方式，没有异常要求。"
    },
    {
        "sentence": "喂大舅，我是小明啊。我妈让我问问你，周末有没有空来家里吃饭，她说好久没见你了。",
        "label": "normal",
        "fraud_type": "",
        "reasoning": "正常的亲友问候电话，有明确的身份关系，话题自然，没有异常。"
    },
    {
        "sentence": "您好，我是小区物业的，楼下水管爆了需要紧急抢修，您家里需要暂时停水大概2个小时，跟您说一声，请提前接点水备用。",
        "label": "normal",
        "fraud_type": "",
        "reasoning": "正常的物业通知电话，有具体事由，不索要钱财或敏感信息。"
    },
    {
        "sentence": "喂老板，我是小刘啊。明天的会议资料我已经准备好了，发您邮箱了，您看一下有什么需要改的没？",
        "label": "normal",
        "fraud_type": "",
        "reasoning": "正常的工作沟通电话，有明确的上下级关系，话题符合日常工作场景。"
    },
    # --- 更多诈骗变种 ---
    {
        "sentence": "你好，我这边是社保局的。你的社保卡在上海市有一笔异常报销记录，涉嫌骗保。如果不及时处理，你的社保卡将被停用，还会影响你以后看病报销。请你提供一下你的社保卡号和密码，我们帮你核实。",
        "label": "fraud",
        "fraud_type": "冒充社保局",
        "reasoning": "社保局不会通过电话索要社保卡密码。这是冒充社保局的诈骗，利用人们对社保卡停用的恐慌心理套取个人信息。"
    },
    {
        "sentence": "您好，我是航空公司客服。您预订的明天飞北京的CA1234航班因天气原因取消了，我们可以帮您改签或退票。改签需要您支付200元的手续费，退票可以全额退款但需要您提供银行卡信息。",
        "label": "fraud",
        "fraud_type": "冒充航空公司",
        "reasoning": "航班取消改签是常见诈骗手法。诈骗分子通过非法获取的航班信息增加可信度。正规改签不收取额外费用。"
    },
    {
        "sentence": "哥！我是阿强啊！我换了新号码，之前那个不用了。我现在急需一笔钱周转，能不能借我5万块钱？三天就还你。我发你一个新卡号。",
        "label": "fraud",
        "fraud_type": "冒充熟人",
        "reasoning": "冒充熟人换号借钱是常见诈骗。应通过其他渠道（微信、面见）核实对方身份后再决定是否借钱。"
    },
    {
        "sentence": "恭喜你被我们公司选中为品牌体验官！不需要交任何费用，只需要每天在朋友圈分享我们的产品，每个月就能拿3000元底薪+提成。加我微信了解一下详情。",
        "label": "fraud",
        "fraud_type": "传销/拉人头诈骗",
        "reasoning": "以'品牌体验官'为名的拉人头传销模式，先免费后通过各种方式诱导付费或发展下线。"
    },
    {
        "sentence": "您好，我是ETC客服中心。您的ETC已经过期，需要重新认证，否则明天就上不了高速了。请点击短信里的链接进行认证，输入您的车牌号和银行卡信息即可。",
        "label": "fraud",
        "fraud_type": "ETC诈骗",
        "reasoning": "ETC过期短信链接诈骗。短信中的链接是钓鱼网站，目的是获取银行卡信息。官方ETC认证不会通过短信链接进行。"
    },
    {
        "sentence": "您好，我是学而思的班主任王老师。恭喜您的孩子通过了我们的入学测试！现在报名暑期班可享受早鸟优惠，原价12800，现在只要3980，仅限今天！请把学费转到这个账户。",
        "label": "fraud",
        "fraud_type": "教育诈骗",
        "reasoning": "冒充教育机构以优惠为由诱导快速转账。'仅限今天'制造紧迫感。正规机构会通过官方渠道收费。"
    },
    {
        "sentence": "您好，这边是疫情防控中心。根据大数据排查，您是密切接触者，需要集中隔离。请提供您的身份证号和家庭住址，我们会安排专车接您去隔离点。在此之前需要缴纳5000元的隔离押金。",
        "label": "fraud",
        "fraud_type": "涉疫诈骗",
        "reasoning": "疫情防控机构不会通过电话索要隔离押金。这是利用公众对疫情的恐慌进行的诈骗。"
    },
    {
        "sentence": "你好我是快递公司的，你有一个国际快递被海关扣了，里面发现可疑物品。需要你配合调查并缴纳清关罚款5000元，否则将移交公安机关处理。请扫描我发给你的付款码。",
        "label": "fraud",
        "fraud_type": "海关/快递诈骗",
        "reasoning": "海关不会通过个人转账方式收取罚款。这是虚构国际快递问题的诈骗手法。"
    },

    # ================================================================
    # 扩展样本 — 覆盖16种来电类型，平衡知识库（减少RAG偏向）
    # ================================================================

    # --- 正常社交/熟人 (friend) ---
    {"sentence": "喂，在干嘛呢？晚上有没有空，一起吃个饭？", "label": "normal", "fraud_type": "", "reasoning": "朋友闲聊约饭，正常的社交来电。"},
    {"sentence": "喂，最近怎么样？好久没联系了，周末聚聚？", "label": "normal", "fraud_type": "", "reasoning": "朋友问候+聚会邀请。"},
    {"sentence": "在吗？我是你大学同学，好久不见了，加个微信吧。", "label": "normal", "fraud_type": "", "reasoning": "同学联系，正常的社交。"},
    {"sentence": "兄弟，晚上撸串去不去？老地方。", "label": "normal", "fraud_type": "", "reasoning": "朋友约吃饭，口语化社交。"},
    {"sentence": "嘿，周末有空吗？一起去打球。", "label": "normal", "fraud_type": "", "reasoning": "朋友约运动，日常社交。"},
    {"sentence": "在吗在吗？我刚看到个超好笑的视频发你了。", "label": "normal", "fraud_type": "", "reasoning": "好友闲聊分享，正常社交。"},
    {"sentence": "喂，是我，你在哪？我到了怎么没看到你。", "label": "normal", "fraud_type": "", "reasoning": "朋友约见面，正常来电。"},
    {"sentence": "生日快乐！今天怎么安排的？", "label": "normal", "fraud_type": "", "reasoning": "朋友生日祝福，正常社交。"},

    # --- 正常家庭 (family) ---
    {"sentence": "妈，我今天晚上回家吃饭。", "label": "normal", "fraud_type": "", "reasoning": "子女通知回家，正常家庭通话。"},
    {"sentence": "喂，儿子，你周末回不回家？妈给你做好吃的。", "label": "normal", "fraud_type": "", "reasoning": "母亲关心子女，正常家庭通话。"},
    {"sentence": "老爸，我下周三的火车，到站你来接我一下。", "label": "normal", "fraud_type": "", "reasoning": "家人接站安排，正常通话。"},
    {"sentence": "老婆，今天加班晚点回去，不用等我吃饭了。", "label": "normal", "fraud_type": "", "reasoning": "夫妻日常沟通。"},
    {"sentence": "喂，我是你表哥，晚上去外婆家吃饭，你来不来？", "label": "normal", "fraud_type": "", "reasoning": "亲戚约饭，正常家庭社交。"},
    {"sentence": "姐，爸妈让你这周末回来一趟，家里有点事。", "label": "normal", "fraud_type": "", "reasoning": "家人通知事务，正常。"},

    # --- 正常工作 (leader/colleague/client) ---
    {"sentence": "喂老板，明天那个客户会议改到下午3点了，我发您邮箱了。", "label": "normal", "fraud_type": "", "reasoning": "正常的工作沟通。"},
    {"sentence": "你好我是你同事小王，那个需求文档你那边有没有模板？", "label": "normal", "fraud_type": "", "reasoning": "同事协作，正常。"},
    {"sentence": "喂，我这边是XX公司的，上回的方案您看过了吗？方便沟通一下吗？", "label": "normal", "fraud_type": "", "reasoning": "客户跟进，正常商务来电。"},
    {"sentence": "领导，项目进度报告我已经发你了，你看一下有没有问题。", "label": "normal", "fraud_type": "", "reasoning": "工作汇报，正常。"},
    {"sentence": "大家好，明天上午10点开周会，请大家准时参加。", "label": "normal", "fraud_type": "", "reasoning": "会议通知，正常工作。"},
    {"sentence": "经理，客户那边说明天合同就能签了。", "label": "normal", "fraud_type": "", "reasoning": "工作进度汇报。"},

    # --- 外卖/快递/打车 (business) ---
    {"sentence": "你好我是美团骑手，你的餐到了，在楼下了。", "label": "normal", "fraud_type": "", "reasoning": "正常外卖送达通知。"},
    {"sentence": "喂，你的外卖到了，放在楼下外卖柜了，取餐码发你手机上了。", "label": "normal", "fraud_type": "", "reasoning": "外卖柜取餐通知。"},
    {"sentence": "你好，我是饿了么骑手，你的订单到东门了，请问放哪里？", "label": "normal", "fraud_type": "", "reasoning": "外卖员询问放置地点。"},
    {"sentence": "顺丰快递，有你一个包裹，你在家吗？我给你送上来。", "label": "normal", "fraud_type": "", "reasoning": "快递送货上门。"},
    {"sentence": "你好快递到了，放菜鸟驿站可以吗？", "label": "normal", "fraud_type": "", "reasoning": "快递放驿站询问。"},
    {"sentence": "我是圆通快递的，有个快递麻烦你下来取一下。", "label": "normal", "fraud_type": "", "reasoning": "快递通知取件。"},
    {"sentence": "中通快递，你的包裹到楼下了，下来拿还是放快递柜？", "label": "normal", "fraud_type": "", "reasoning": "快递员询问放置方式。"},
    {"sentence": "你好我是滴滴司机，我到定位的地方了，你在哪？", "label": "normal", "fraud_type": "", "reasoning": "网约车到达通知。"},
    {"sentence": "喂，我是网约车司机，我到小区门口了，你出来吧。", "label": "normal", "fraud_type": "", "reasoning": "网约车司机通知。"},
    {"sentence": "师傅你好，我是XX家政的，约了今天下午2点上门保洁，我到小区了。", "label": "normal", "fraud_type": "", "reasoning": "家政服务上门通知。"},
    {"sentence": "你好我是XX维修的，你预约的空调维修，我现在过去方便吗？", "label": "normal", "fraud_type": "", "reasoning": "维修服务确认时间。"},
    {"sentence": "喂，你的快递给你放传达室了，记得拿一下。", "label": "normal", "fraud_type": "", "reasoning": "快递存放通知。"},

    # --- 面试/bank/物业 (important) ---
    {"sentence": "您好，这边是XX公司HR，恭喜您通过了面试，想和您确认一下入职时间。", "label": "normal", "fraud_type": "", "reasoning": "正常的面试通过通知。"},
    {"sentence": "你好我是XX银行的，您的信用卡账单这个月有异常消费，想跟您核实一下。", "label": "normal", "fraud_type": "", "reasoning": "银行核实消费，正常业务。"},
    {"sentence": "您好，我是小区物业，楼下要停水检修，大概2小时，请提前备水。", "label": "normal", "fraud_type": "", "reasoning": "正常的物业通知。"},
    {"sentence": "你好，我是你楼上邻居，你家水管好像漏水了，水都渗下来了。", "label": "normal", "fraud_type": "", "reasoning": "邻居通知紧急情况，正常。"},
    {"sentence": "你好，这边是XX教育，您之前咨询的课程现在有优惠活动，想跟您确认一下是否还有兴趣？", "label": "normal", "fraud_type": "", "reasoning": "客户回访，介于推销和正常之间。"},

    # --- 更多诈骗变种 ---
    {"sentence": "您好，我是京东客服，您购买的商品因为质量问题要给您双倍退款，请加QQ填写退款信息。", "label": "fraud", "fraud_type": "冒充客服", "reasoning": "客服退款+QQ引导，典型诈骗。"},
    {"sentence": "喂，我是你领导，现在有个急事需要用钱，你先转5万到这个账户，明天还你。", "label": "fraud", "fraud_type": "冒充领导", "reasoning": "冒充领导借钱，典型诈骗。"},
    {"sentence": "您好，您的社保卡在上海有一笔异常消费，已被冻结。请提供您的身份证号解冻。", "label": "fraud", "fraud_type": "冒充社保", "reasoning": "社保诈骗，索要个人信息。"},
    {"sentence": "你好，我是XX借贷平台的，你有一笔贷款逾期未还，已被列入失信名单。请马上还款到这个账户。", "label": "fraud", "fraud_type": "贷款诈骗", "reasoning": "虚假贷款催收诈骗。"},
    {"sentence": "喂，我这边是微信客服，你的微信百万保障即将到期，需要续费，否则每月扣800块。", "label": "fraud", "fraud_type": "冒充客服", "reasoning": "微信百万保障诈骗，近期高发。"},
    {"sentence": "您好，这边是疫情防控中心，您是密切接触者，需要集中隔离。请缴纳5000元隔离押金。", "label": "fraud", "fraud_type": "涉疫诈骗", "reasoning": "冒充防疫人员骗取押金。"},
    {"sentence": "恭喜您成为我们平台的幸运用户！现在只需支付99元运费，就可以免费领取一部最新款手机。", "label": "fraud", "fraud_type": "虚假中奖", "reasoning": "虚假中奖+小额运费骗取信任。"},
    {"sentence": "我是你的大学同学张伟啊，换号了。现在在医院急需用钱，能借我3000吗？", "label": "fraud", "fraud_type": "冒充熟人", "reasoning": "冒充熟人借钱诈骗。"},

    # --- 推销/广告 ---
    {"sentence": "你好，我是XX保险公司的，我们最近推出了一款新的健康险，特别适合您这个年龄段。", "label": "fraud", "fraud_type": "推销电话", "reasoning": "保险推销，商业来电。"},
    {"sentence": "家长您好，我们是XX教育的，孩子马上放暑假了，我们的暑期班现在有早鸟优惠，仅限今天。", "label": "fraud", "fraud_type": "推销电话", "reasoning": "教育培训推销。"},
    {"sentence": "你好，这边是XX银行信用卡中心，您符合我行白金卡办理条件，现在办理免年费。", "label": "normal", "fraud_type": "", "reasoning": "银行推销，虽然烦人但不算诈骗。"},
    {"sentence": "您好，我是XX房产中介的，您之前看过的那个小区有一套急售房源，价格比市场低20%，有没有兴趣？", "label": "normal", "fraud_type": "", "reasoning": "房产中介推销。"},
    {"sentence": "你好，对编程感兴趣吗？我们这边有一个免费的Python体验课，想不想了解一下？", "label": "fraud", "fraud_type": "推销电话", "reasoning": "编程培训推销，带有诱导性。"},
    {"sentence": "尊敬的会员您好，您的年度会员即将到期，现在续费可享受8折优惠。", "label": "normal", "fraud_type": "", "reasoning": "会员续费提醒，正常商业通知。"},

    # --- 无意义/杂音 ---
    {"sentence": "喂？喂？能听到吗？", "label": "normal", "fraud_type": "", "reasoning": "信号测试，正常。"},
    {"sentence": "嗯...那个...算了没事了。", "label": "normal", "fraud_type": "", "reasoning": "对方临时改变主意，正常。"},
]


def load_from_huggingface() -> list[dict]:
    """
    从 HuggingFace 加载 TeleAntiFraud-28k 数据集。
    自动尝试国内镜像 (hf-mirror.com)，失败则报错。

    返回:
        list[dict]: 原始记录列表
    """
    from datasets import load_dataset

    # 先尝试镜像站
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        logger.info("尝试使用 HuggingFace 镜像站 (hf-mirror.com)...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        logger.info("正在从 HuggingFace 加载 TeleAntiFraud-28k 数据集...")
        dataset = load_dataset("JimmyMa99/TeleAntiFraud")
    except Exception as e:
        logger.error(f"HuggingFace 加载失败: {e}")
        raise

    logger.info(f"数据集加载成功！划分: {list(dataset.keys())}")

    all_records = []
    for split_name, split_data in dataset.items():
        logger.info(f"  处理 {split_name}: {len(split_data)} 条")
        for record in split_data:
            all_records.append({
                "sentence": record.get("sentence", ""),
                "label": record.get("label", ""),
                "fraud_type": record.get("fraud_type", ""),
                "reasoning": record.get("reasoning", ""),
                "split": split_name,
            })

    logger.info(f"总计: {len(all_records)} 条记录")
    return all_records


def load_from_kaggle_local(data_dir: str | None = None) -> list[dict]:
    """
    从本地目录加载从 Kaggle 下载的数据集。

    支持的文件格式:
    - CSV 文件: 自动检测分隔符和列名
    - JSON/JSONL 文件: 自动解析
    - Parquet 文件: 通过 pandas 加载
    - 音频文件 (.wav/.mp3): 仅记录文件名，文本需配合 CSV 元数据

    Kaggle 下载步骤:
    1. 打开 https://www.kaggle.com/datasets/divyanshsharma23/research28k
    2. 点击 Download 按钮下载 zip 文件
    3. 解压到 YanXi/data/raw/ 目录下
    4. 运行 python -m src.knowledge.dataset_loader --local

    参数:
        data_dir: 数据目录路径，默认为 project_root/data/raw/

    返回:
        list[dict]: 原始记录列表
    """
    target_dir = Path(data_dir) if data_dir else LOCAL_DATA_DIR

    if not target_dir.exists():
        raise FileNotFoundError(
            f"本地数据目录不存在: {target_dir}\n"
            f"请从 Kaggle 下载数据集并解压到此目录:\n"
            f"  https://www.kaggle.com/datasets/divyanshsharma23/research28k"
        )

    # 列出所有文件
    all_files = list(target_dir.rglob("*"))
    logger.info(f"本地数据目录: {target_dir}")
    logger.info(f"  找到 {len(all_files)} 个文件")
    for f in all_files[:20]:  # 只显示前 20 个
        logger.info(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    records = []

    # 1. 尝试加载 CSV 文件 (最常见的数据集格式)
    csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
    for csv_file in csv_files:
        records.extend(_parse_csv_file(csv_file))

    # 2. 尝试加载 JSON/JSONL 文件
    json_files = [f for f in all_files if f.suffix.lower() in (".json", ".jsonl")]
    for json_file in json_files:
        records.extend(_parse_json_file(json_file))

    # 3. 尝试加载 Parquet 文件
    parquet_files = [f for f in all_files if f.suffix.lower() == ".parquet"]
    for pq_file in parquet_files:
        records.extend(_parse_parquet_file(pq_file))

    # 4. 兼容 HuggingFace 下载到本地的 arrow 格式 (data-*.arrow)
    arrow_files = [f for f in all_files if f.suffix.lower() == ".arrow"]
    if arrow_files:
        records = _parse_arrow_files(target_dir)

    if not records:
        raise RuntimeError(
            f"未能从 {target_dir} 中解析出任何数据。\n"
            f"支持的格式: CSV, JSON, JSONL, Parquet\n"
            f"请确认解压后的文件格式正确。"
        )

    logger.info(f"成功加载 {len(records)} 条本地记录")
    return records


def _parse_csv_file(filepath: Path) -> list[dict]:
    """解析 CSV 文件，自动检测列名映射。"""
    logger.info(f"解析 CSV: {filepath.name}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        # 先读第一行判断分隔符
        first_line = f.readline()
        f.seek(0)

        # 检测分隔符
        delimiter = "," if first_line.count(",") > first_line.count("\t") else "\t"

        reader = csv.DictReader(f, delimiter=delimiter)
        # 获取列名（转小写方便匹配）
        columns = [c.lower().strip() for c in reader.fieldnames] if reader.fieldnames else []
        logger.info(f"  列名: {columns}")

        records = []
        for row in reader:
            # 列名映射: 将常见列名映射到标准字段
            sentence = _find_field(row, ["sentence", "text", "content", "transcript", "utterance", "对话", "文本", "内容"])
            label = _find_field(row, ["label", "is_fraud", "is_scam", "fraud_label", "label_cn", "标签", "类别"])
            fraud_type = _find_field(row, ["fraud_type", "scam_type", "category", "type", "诈骗类型", "类型"])
            reasoning = _find_field(row, ["reasoning", "reason", "analysis", "explanation", "分析", "推理"])

            if sentence:
                records.append({
                    "sentence": sentence,
                    "label": label or "",
                    "fraud_type": fraud_type or "",
                    "reasoning": reasoning or "",
                    "split": "kaggle_local",
                })

        logger.info(f"  从 CSV 解析出 {len(records)} 条记录")
        return records


def _parse_json_file(filepath: Path) -> list[dict]:
    """解析 JSON/JSONL 文件。"""
    logger.info(f"解析 JSON: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        # 先判断是 JSON 数组还是 JSONL
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON 数组
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            # JSONL (每行一个 JSON 对象)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not data:
        return []

    # 检查是否是 HuggingFace datasets 格式 (有 features 字段的 Dataset dict)
    if isinstance(data, dict) and "dataset" in str(data).lower():
        data = _extract_from_dict_format(data)

    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sentence = _find_field(item, ["sentence", "text", "content", "transcript", "utterance"])
        if sentence:
            records.append({
                "sentence": str(sentence),
                "label": str(_find_field(item, ["label", "is_fraud", "is_scam", "fraud_label"]) or ""),
                "fraud_type": str(_find_field(item, ["fraud_type", "scam_type", "category", "type"]) or ""),
                "reasoning": str(_find_field(item, ["reasoning", "reason", "analysis", "explanation"]) or ""),
                "split": "kaggle_local",
            })

    logger.info(f"  从 JSON 解析出 {len(records)} 条记录")
    return records


def _parse_parquet_file(filepath: Path) -> list[dict]:
    """解析 Parquet 文件。"""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，跳过 Parquet 解析")
        return []

    logger.info(f"解析 Parquet: {filepath.name}")
    df = pd.read_parquet(filepath)

    columns = [c.lower().strip() for c in df.columns]
    logger.info(f"  列名: {columns}")

    records = []
    for _, row in df.iterrows():
        sentence = _find_field_in_series(row, columns, ["sentence", "text", "content", "transcript"])
        if sentence and not pd.isna(sentence):
            records.append({
                "sentence": str(sentence),
                "label": str(_find_field_in_series(row, columns, ["label", "is_fraud", "is_scam"]) or ""),
                "fraud_type": str(_find_field_in_series(row, columns, ["fraud_type", "scam_type", "category"]) or ""),
                "reasoning": str(_find_field_in_series(row, columns, ["reasoning", "reason", "analysis"]) or ""),
                "split": "kaggle_local",
            })

    logger.info(f"  从 Parquet 解析出 {len(records)} 条记录")
    return records


def _parse_arrow_files(data_dir: Path) -> list[dict]:
    """解析 HuggingFace datasets 缓存的 Arrow 文件。"""
    try:
        from datasets import Dataset
    except ImportError:
        return []

    logger.info("检测到 Arrow 文件，尝试用 datasets 库加载...")
    try:
        ds = Dataset.load_from_disk(str(data_dir))
    except Exception:
        # 尝试加载子目录
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                try:
                    ds = Dataset.load_from_disk(str(subdir))
                    break
                except Exception:
                    continue
        else:
            return []

    records = []
    for item in ds:
        records.append({
            "sentence": item.get("sentence", item.get("text", "")),
            "label": item.get("label", ""),
            "fraud_type": item.get("fraud_type", ""),
            "reasoning": item.get("reasoning", ""),
            "split": "local_arrow",
        })

    logger.info(f"  从 Arrow 加载 {len(records)} 条记录")
    return records


def _extract_from_dict_format(data: dict) -> list[dict]:
    """从 HuggingFace dataset dict 格式中提取数据列表。"""
    # 可能是 {"train": [...], "test": [...]} 这种结构
    result = []
    for key, value in data.items():
        if isinstance(value, list):
            result.extend(value)
        elif isinstance(value, dict):
            result.extend(_extract_from_dict_format(value))
    return result if result else [data]


def _find_field(row: dict, candidates: list[str]) -> str | None:
    """从 dict 行中查找候选字段名（大小写不敏感）。"""
    row_lower = {k.lower().strip(): v for k, v in row.items()}
    for cand in candidates:
        cand_lower = cand.lower().strip()
        if cand_lower in row_lower:
            val = row_lower[cand_lower]
            if val and str(val).strip() and str(val).strip().lower() not in ("nan", "none", "null", ""):
                return str(val).strip()
    return None


def _find_field_in_series(row, columns: list[str], candidates: list[str]) -> str | None:
    """从 pandas Series 行中查找候选字段名。"""
    for cand in candidates:
        cand_lower = cand.lower().strip()
        for col in columns:
            if col == cand_lower:
                val = row[col]
                if not pd.isna(val) and str(val).strip():
                    return str(val).strip()
    return None


def load_builtin_samples() -> list[dict]:
    """加载内置样例数据（无需联网）。"""
    logger.info(f"使用内置样例数据: {len(BUILTIN_SCAM_SAMPLES)} 条")
    for sample in BUILTIN_SCAM_SAMPLES:
        sample["split"] = "builtin"
    return list(BUILTIN_SCAM_SAMPLES)


# ============================================================
# 预处理（所有来源共用）
# ============================================================

def preprocess_records(records: list[dict]) -> list[dict]:
    """
    对原始记录进行预处理，生成适合 RAG 检索的标准文档格式。

    处理步骤:
    1. 过滤空文本：去除 sentence 为空的记录
    2. 文本清洗：去除多余空白和特殊字符
    3. 生成 RAG 文档块：将 sentence + fraud_type + reasoning 拼接为标准文本
    4. 添加元数据：保留标签、诈骗类型、来源划分

    参数:
        records: 原始数据集记录列表

    返回:
        list[dict]: 处理后的文档列表，每个文档包含:
            - id: 唯一标识
            - content: 供 embedding 和检索使用的文本内容
            - metadata: 元数据（label, fraud_type, split）
    """
    logger.info(f"开始预处理 {len(records)} 条记录...")

    documents = []
    fraud_count = 0
    normal_count = 0
    empty_count = 0

    for i, record in enumerate(records):
        sentence = (record.get("sentence") or "").strip()

        # 跳过空文本
        if not sentence:
            empty_count += 1
            continue

        # 清洗文本：合并多个空白为单个空格
        sentence = " ".join(sentence.split())

        # 获取标签和诈骗类型
        label = record.get("label", "")
        fraud_type = record.get("fraud_type", "")
        reasoning = (record.get("reasoning") or "").strip()

        # 判断是否为诈骗记录
        is_fraud = _is_fraud_label(label, fraud_type, sentence)
        if is_fraud:
            fraud_count += 1
        else:
            normal_count += 1

        # --- 构建 RAG 文档内容 ---
        parts = [f"【通话内容】{sentence}"]

        if fraud_type:
            parts.append(f"【类型】{fraud_type}")

        if reasoning:
            parts.append(f"【分析】{reasoning}")

        content = "\n".join(parts)

        # --- 构建文档对象 ---
        doc = {
            "id": f"scam_{i:06d}",
            "content": content,
            "metadata": {
                "label": "fraud" if is_fraud else "normal",
                "fraud_type": str(fraud_type),
                "split": record.get("split", ""),
                "source": record.get("split", "unknown"),
                "original_sentence": sentence[:200],
            },
        }
        documents.append(doc)

    logger.info(f"预处理完成:")
    logger.info(f"  有效文档: {len(documents)}")
    logger.info(f"  诈骗案例: {fraud_count}")
    logger.info(f"  正常通话: {normal_count}")
    logger.info(f"  过滤空记录: {empty_count}")

    return documents


def _is_fraud_label(label, fraud_type: str, sentence: str) -> bool:
    """
    综合判断一条记录是否为诈骗。

    参数:
        label: 原始标签值
        fraud_type: 诈骗类型字段
        sentence: 通话文本

    返回:
        bool: True=诈骗, False=正常
    """
    # 数字标签
    if isinstance(label, (int, float)):
        return bool(int(label))

    # 字符串标签
    label_str = str(label).lower().strip()
    if label_str in ("fraud", "scam", "1", "true", "yes"):
        return True
    if label_str in ("normal", "legitimate", "0", "false", "no"):
        return False

    # 通过 fraud_type 判断
    if fraud_type and str(fraud_type).lower() not in ("", "none", "normal", "nan"):
        return True

    # 关键词兜底
    scam_keywords = [
        "诈骗", "骗局", "转账", "安全账户", "验证码", "银行卡号",
        "公检法", "法院传票", "涉嫌犯罪", "资金冻结", "保证金",
    ]
    sentence_lower = sentence.lower()
    match_count = sum(1 for kw in scam_keywords if kw in sentence_lower)
    return match_count >= 2


def save_documents(documents: list[dict], output_path: Path = OUTPUT_PATH):
    """
    将处理后的文档保存为 JSON 文件。

    参数:
        documents: 处理后的文档列表
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"文档已保存: {output_path} ({file_size_mb:.1f} MB)")


# ============================================================
# 统一入口
# ============================================================

def load_dataset(source: str = "auto") -> list[dict]:
    """
    统一的数据加载入口，根据 source 参数选择加载方式。

    参数:
        source: 数据来源
            - "auto":   自动选择最优来源 (HuggingFace > 本地文件 > 内置样例)
            - "sample": 仅使用内置样例
            - "local":  从本地 data/raw/ 加载
            - "hf":     仅从 HuggingFace 加载

    返回:
        list[dict]: 原始记录列表
    """
    if source == "sample":
        return load_builtin_samples()

    if source == "local":
        try:
            return load_from_kaggle_local()
        except FileNotFoundError as e:
            logger.warning(str(e))
            logger.info("自动回退到内置样例数据...")
            return load_builtin_samples()

    if source == "hf":
        return load_from_huggingface()

    # "auto" 模式：依次尝试
    sources_tried = []

    # 1. 先尝试本地数据
    if LOCAL_DATA_DIR.exists():
        try:
            records = load_from_kaggle_local()
            logger.info("✅ 使用本地数据")
            return records
        except Exception as e:
            sources_tried.append(f"local ({e})")

    # 2. 再尝试 HuggingFace
    try:
        records = load_from_huggingface()
        logger.info("✅ 使用 HuggingFace 数据")
        return records
    except Exception as e:
        sources_tried.append(f"hf ({e})")

    # 3. 兜底：内置样例
    logger.warning(f"外部数据源均不可用: {'; '.join(sources_tried)}")
    logger.info("✅ 使用内置样例数据")
    return load_builtin_samples()


# ============================================================
# 独立运行入口
# ============================================================
if __name__ == "__main__":
    """
    运行数据集加载和预处理:
      python -m src.knowledge.dataset_loader              # 自动选择最优来源
      python -m src.knowledge.dataset_loader --sample     # 使用内置样例 (推荐，无需联网)
      python -m src.knowledge.dataset_loader --local      # 从本地 data/raw/ 加载
    """
    import argparse

    parser = argparse.ArgumentParser(description="诈骗知识库数据集加载器")
    parser.add_argument("--sample", action="store_true", help="使用内置样例数据")
    parser.add_argument("--local", action="store_true", help="从本地 data/raw/ 加载")
    parser.add_argument("--hf", action="store_true", help="仅从 HuggingFace 加载")
    args = parser.parse_args()

    start_time = time.time()

    # 确定数据来源
    if args.sample:
        source = "sample"
    elif args.local:
        source = "local"
    elif args.hf:
        source = "hf"
    else:
        source = "auto"

    logger.info(f"数据来源: {source}")

    # 1. 加载
    records = load_dataset(source)

    # 2. 预处理
    documents = preprocess_records(records)

    # 3. 保存
    save_documents(documents)

    elapsed = time.time() - start_time
    logger.info(f"全部完成，耗时: {elapsed:.1f} 秒")
    logger.info(f"下一步: python -m src.knowledge.embedder  # 构建向量库")
