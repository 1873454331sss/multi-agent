import os
import json
import operator
from datetime import datetime
from typing import Annotated, List, Dict, Any, TypedDict
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
import requests

from database import save_analysis

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
if not BOCHA_API_KEY:
    raise ValueError("请在 .env 文件中设置 BOCHA_API_KEY")

# ============================================================
# 0. 行业配置（新能源车企私有化）
# ============================================================

# 竞品配置（人工维护，每个品牌对应 3-5 个核心竞品）
BRAND_COMPETITORS = {
    "蔚来": ["理想", "小鹏", "特斯拉"],
    "理想": ["蔚来", "小鹏", "特斯拉"],
    "小鹏": ["蔚来", "理想", "特斯拉"],
    "特斯拉": ["蔚来", "理想", "小鹏"],
    "比亚迪": ["特斯拉", "蔚来", "理想"],
    "极氪": ["蔚来", "理想", "特斯拉"],
    "问界": ["理想", "蔚来", "小鹏"],
    "智己": ["蔚来", "理想", "极氪"],
    "阿维塔": ["蔚来", "理想", "问界"],
    "岚图": ["蔚来", "理想", "极氪"],
    "哪吒": ["零跑", "小鹏", "比亚迪"],
    "零跑": ["哪吒", "小鹏", "比亚迪"],
}

DEFAULT_COMPETITORS = ["理想", "小鹏", "特斯拉"]

# 风险词库（按安全/质量/服务/监管四类组织）
RISK_KEYWORDS = [
    # 安全类
    "电池起火", "自动驾驶事故", "刹车失灵", "安全气囊", "碰撞测试", "失控", "自燃", "爆炸",
    # 质量类
    "续航虚标", "充电故障", "OTA召回", "异响", "漏水", "生锈", "车机卡顿", "品控",
    # 服务类
    "投诉", "售后差", "维修贵", "交付延迟", "承诺不兑现",
    # 监管类
    "召回", "监管", "调查", "处罚", "约谈",
]

# 严重词库（命中直接处置级）
SEVERE_KEYWORDS = [
    "电池起火", "自动驾驶事故", "刹车失灵", "安全气囊", "碰撞测试", "失控", "自燃", "爆炸",
]

# ============================================================
# 1. 工具类
# ============================================================

class LocalDataTools:
    @staticmethod
    def load_data(query: str) -> List[Dict]:
        url = "https://api.bocha.cn/v1/web-search"
        headers = {
            "Authorization": f"Bearer {BOCHA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": 5
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('code') != 200:
                return []
            results = data.get('data', {}).get('webPages', {}).get('value', [])
            articles = []
            for item in results[:5]:
                articles.append({
                    "title": item.get("name", "无标题"),
                    "content": item.get("summary", item.get("snippet", "暂无摘要")),
                    "source": item.get("displayUrl", "未知来源"),
                    "time": datetime.now().strftime("%Y-%m-%d"),
                    "url": item.get("url", ""),
                    "siteName": item.get("siteName", "")
                })
            seen = set()
            unique_articles = []
            for item in articles:
                title = item.get("title", "")
                if title not in seen:
                    seen.add(title)
                    unique_articles.append(item)
            return unique_articles
        except Exception:
            return []

    @staticmethod
    def _call_llm(prompt: str, system: str = "你是分析专家，只返回JSON。", max_tokens: int = 500) -> Dict:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        return json.loads(content)

    @staticmethod
    def analyze_sentiment(full_text: str, brand: str) -> Dict:
        prompt = f"""分析以下文本的情感倾向，严格按JSON返回。
主品牌：{brand}
文本：{full_text[:2000]}

返回格式：
{{
  "overall": {{"label": "positive/negative/neutral", "score": 0.85}},
  "brand": {{"label": "positive/negative/neutral", "score": 0.2, "evidence_count": 3}}
}}
其中 overall 是整体情感，brand 是主品牌相关语句的情感，evidence_count 是支撑主品牌情感的语句数量。
只返回JSON，不要解释。"""
        try:
            return LocalDataTools._call_llm(prompt, "你是情感分析专家，只返回JSON。", 300)
        except Exception:
            return LocalDataTools._fallback_sentiment(full_text)

    @staticmethod
    def _fallback_sentiment(text: str) -> Dict:
        positive_keywords = ["强劲", "突破", "领先", "增长", "积极", "升级", "出色", "优秀", "创新", "超越", "反响热烈",
                             "显著"]
        negative_keywords = ["下滑", "下降", "危机", "负面", "投诉", "质量", "召回", "诉讼", "压力", "监管", "调查",
                             "挑战"]
        pos = sum(1 for kw in positive_keywords if kw in text)
        neg = sum(1 for kw in negative_keywords if kw in text)
        total = pos + neg
        if total == 0:
            label, score = "neutral", 0.5
        else:
            score = pos / total
            label = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        return {
            "overall": {"label": label, "score": round(score, 2)},
            "brand": {"label": label, "score": round(score, 2), "evidence_count": total}
        }

    @staticmethod
    def generate_report(data: Dict) -> str:
        sentiment = data.get("sentiment", {})
        competitors = data.get("competitors", [])
        comparison = data.get("comparison", [])
        risks = data.get("risks", [])
        raw_data = data.get("raw_data", [])
        risk_level = data.get("risk_level", "low")
        arbitration = data.get("arbitration", {})
        alert_level = data.get("alert_level", "无")
        ticket = data.get("ticket", {})

        lines = []
        lines.append("# 📊 舆情告警报告")
        lines.append("")
        lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("**数据来源**：博查 Web Search API")
        lines.append(f"**告警等级**：{alert_level}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 一、数据概览")
        lines.append("")
        lines.append(f"- 共采集 **{len(raw_data)}** 条相关资讯")
        for idx, item in enumerate(raw_data, 1):
            title = item.get('title', '未知标题')
            src = item.get('siteName', item.get('source', '未知来源'))
            lines.append(f"  {idx}. {title}（{src}）")
        lines.append("")
        lines.append("## 二、情感分析")
        lines.append("")
        label_map = {"positive": "🟢 积极", "neutral": "🟡 中性", "negative": "🔴 消极"}
        overall = sentiment.get("overall", {})
        brand = sentiment.get("brand", {})
        lines.append(f"- **整体倾向**：{label_map.get(overall.get('label', 'neutral'), '未知')}")
        lines.append(
            f"- **主品牌倾向**：{label_map.get(brand.get('label', 'neutral'), '未知')}（证据 {brand.get('evidence_count', 0)} 条）")
        lines.append("")
        lines.append("## 三、竞品分析")
        lines.append("")
        if comparison:
            lines.append("| 品牌 | 资讯条数 | 情感倾向 |")
            lines.append("| :--- | :--- | :--- |")
            for row in comparison:
                lines.append(
                    f"| {row.get('brand', '—')} | {row.get('article_count', 0)} | {label_map.get(row.get('sentiment', 'neutral'), '未知')} |"
                )
        else:
            lines.append("- 暂未发现明显竞品")
        lines.append("")
        lines.append("## 四、风险预警")
        lines.append("")
        if risks:
            lines.append(f"- ⚠️ 发现 **{len(risks)}** 个风险信号：")
            for risk in risks:
                lines.append(f"  - {risk}")
            lines.append(f"- **风险等级**：{risk_level}")
        else:
            lines.append("- ✅ 当前未检测到明显风险信号")
        lines.append("")
        lines.append("## 五、综合结论")
        lines.append("")
        if arbitration.get("final_label"):
            lines.append(f"- **仲裁结果**：{arbitration.get('final_label')}")
            lines.append(f"- **仲裁理由**：{arbitration.get('reason')}")
        else:
            if alert_level == "处置":
                conclusion = "舆情存在严重风险，建议启动应急响应。"
            elif alert_level == "预警":
                conclusion = "舆情存在一定风险，建议重点监测。"
            elif alert_level == "关注":
                conclusion = "舆情出现风险信号，建议持续关注。"
            else:
                conclusion = "整体舆情态势良好，建议继续保持关注。"
            lines.append(f"> {conclusion}")
        lines.append("")
        lines.append("## 六、告警与工单")
        lines.append("")
        lines.append(f"- **告警等级**：{alert_level}")
        if ticket and ticket.get("ticket_id"):
            lines.append(f"- **工单号**：{ticket.get('ticket_id')}")
            lines.append(f"- **推送对象**：{ticket.get('assignee')}")
            lines.append(f"- **时效**：{ticket.get('deadline')}")
            lines.append(f"- **建议动作**：{ticket.get('suggestion')}")
        else:
            lines.append("- 无工单生成（关注级或无明显风险）")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*本报告由多Agent舆情告警系统自动生成，仅供参考。*")
        return "\n".join(lines)


# ============================================================
# 2. LangChain Tools
# ============================================================

@tool
def search_web(query: str) -> list:
    """搜索网页，输入查询词，返回搜索结果列表。用于获取品牌舆情数据。"""
    return LocalDataTools.load_data(query)


@tool
def check_alert_level(risk_keywords: list, sentiment_label: str, severe_keywords: list) -> str:
    """根据风险关键词和情感倾向，判断告警等级。新能源车企：电池安全、自动驾驶事故直接处置级。"""
    if severe_keywords:
        return "处置"
    if len(risk_keywords) >= 5:
        return "处置"
    elif len(risk_keywords) >= 2:
        return "预警"
    elif len(risk_keywords) >= 1 or sentiment_label == "negative":
        return "关注"
    return "无"


@tool
def create_ticket(brand: str, alert_level: str, evidence: list) -> dict:
    """根据告警等级和证据，生成工单。关注级不生成，预警级和处置级生成。"""
    if alert_level == "关注" or alert_level == "无":
        return {"ticket": None, "action": "自动归档"}
    ticket = {
        "ticket_id": f"TICKET_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "brand": brand,
        "level": alert_level,
        "evidence": evidence,
        "assignee": "舆情专员" if alert_level == "预警" else "公关/法务/业务",
        "deadline": "当天" if alert_level == "预警" else "2小时",
        "suggestion": "持续关注" if alert_level == "预警" else "启动应急响应"
    }
    return {"ticket": ticket, "action": "已生成"}


# ============================================================
# 3. LangGraph 状态定义
# ============================================================

def merge_dicts(a: Dict, b: Dict) -> Dict:
    merged = a.copy()
    merged.update(b)
    return merged


class AgentState(TypedDict):
    query: str
    brand: str
    main_query: str
    competitors: List[str]
    error: str
    sentiment_result: Dict[str, Any]
    competitor_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    arbitration: Dict[str, Any]
    conflict: bool
    alert_level: str
    ticket: Dict[str, Any]
    final_report: str
    status: Annotated[Dict[str, str], merge_dicts]
    error_log: Annotated[List[str], operator.add]
    raw_data: Annotated[List[Dict], operator.add]
    competitor_raw_data: Annotated[List[Dict], operator.add]


# ============================================================
# 4. Skill 懒加载器
# ============================================================

class SkillLoader:
    def __init__(self):
        self.tools = LocalDataTools()

    @lru_cache(maxsize=10)
    def load(self, skill_name: str):
        skill_map = {
            "load_data": self.tools.load_data,
            "sentiment": self.tools.analyze_sentiment,
            "report": self.tools.generate_report,
        }
        if skill_name not in skill_map:
            raise ValueError(f"未知的Skill: {skill_name}")
        return skill_map[skill_name]


skills = SkillLoader()


# ============================================================
# 5. Agent 节点
# ============================================================

def preprocess_agent(state: AgentState) -> Dict[str, Any]:
    """预处理节点：清洗输入，从配置里取竞品列表"""
    query = state["query"].strip()
    if not query or len(query) < 2:
        return {"error": "请输入品牌名", "status": {"preprocess": "failed"}}

    competitors = BRAND_COMPETITORS.get(query, DEFAULT_COMPETITORS)

    return {
        "brand": query,
        "main_query": f"{query} 用户评价 投诉 新闻",
        "competitors": competitors,
        "status": {"preprocess": "completed"}
    }


def search_node(state: AgentState) -> Dict[str, Any]:
    if state.get("error"):
        return {"status": {"search": "skipped"}}
    try:
        main_data = search_web.invoke({"query": state["main_query"]})
        main_data = [
            item for item in main_data
            if "官网" not in item.get("siteName", "") and "官网" not in item.get("title", "")
        ]
        return {
            "raw_data": main_data,
            "status": {"search": "completed"}
        }
    except Exception as e:
        return {"status": {"search": "failed"}, "error_log": [f"搜索失败: {str(e)}"]}


def sentiment_agent(state: AgentState) -> Dict[str, Any]:
    """情感分析：一次 LLM 调用，返回整体 + 主品牌"""
    try:
        raw_data = state.get("raw_data", [])
        if not raw_data:
            return {"status": {"sentiment": "failed"}, "error_log": ["情感分析失败: 无数据"]}

        full_text = " ".join([item.get("content", "") for item in raw_data])
        brand = state.get("brand", "")
        result = skills.load("sentiment")(full_text, brand)

        overall = result.get("overall", {"label": "neutral", "score": 0.5})
        brand_result = result.get("brand", {"label": "neutral", "score": 0.5, "evidence_count": 0})

        return {
            "sentiment_result": {
                "overall": overall,
                "brand": brand_result
            },
            "status": {"sentiment": "completed"}
        }
    except Exception as e:
        return {"status": {"sentiment": "failed"}, "error_log": [f"情感分析失败: {str(e)}"]}


def competitor_agent(state: AgentState) -> Dict[str, Any]:
    """竞品分析：分别搜每个竞品的舆情，对比资讯条数和情感倾向"""
    try:
        competitors = state.get("competitors", [])
        if not competitors:
            return {"status": {"competitor": "skipped"}}

        comparison = []
        for c in competitors:
            data = search_web.invoke({"query": f"{c} 用户评价 投诉 新闻"})
            all_text = " ".join([item.get("content", "") for item in data])
            sentiment = skills.load("sentiment")(all_text, c)
            comparison.append({
                "brand": c,
                "article_count": len(data),
                "sentiment": sentiment.get("brand", {}).get("label", "neutral")
            })

        return {
            "competitor_result": {
                "competitors": competitors,
                "comparison": comparison
            },
            "status": {"competitor": "completed"}
        }
    except Exception as e:
        return {"status": {"competitor": "failed"}, "error_log": [f"竞品分析失败: {str(e)}"]}


def risk_agent(state: AgentState) -> Dict[str, Any]:
    """风险检测：行业定制关键词 + 否定词检测"""
    try:
        raw_data = state.get("raw_data", [])
        if not raw_data:
            return {"status": {"risk": "skipped"}, "error_log": ["风险检测跳过: 无数据"]}

        all_text = " ".join([item.get("content", "") for item in raw_data])
        negation_words = ["没有", "未", "无", "不", "非", "没"]

        found = []
        evidence = []
        severe_found = []
        for kw in RISK_KEYWORDS:
            if kw in all_text:
                idx = all_text.find(kw)
                context = all_text[max(0, idx - 10):idx + len(kw) + 10]
                if any(neg in context for neg in negation_words):
                    continue
                found.append(kw)
                if kw in SEVERE_KEYWORDS:
                    severe_found.append(kw)
                for item in raw_data:
                    if kw in item.get("content", ""):
                        evidence.append({"keyword": kw, "source": item.get("siteName", "未知")})
                        break

        level = "critical" if len(found) > 4 else "high" if len(found) > 2 else "medium" if len(found) > 0 else "low"

        return {
            "risk_result": {
                "risks": found,
                "severe_keywords": severe_found,
                "level": level,
                "count": len(found),
                "evidence": evidence[:5]
            },
            "status": {"risk": "completed"}
        }
    except Exception as e:
        return {"status": {"risk": "failed"}, "error_log": [f"风险检测失败: {str(e)}"]}


def arbiter_node(state: AgentState) -> Dict[str, Any]:
    """冲突处理：情感 vs 风险，比较证据条数"""
    sentiment = state.get("sentiment_result", {})
    risk = state.get("risk_result", {})

    brand_label = sentiment.get("brand", {}).get("label", "neutral")
    risk_level = risk.get("level", "low")
    sentiment_evidence = sentiment.get("brand", {}).get("evidence_count", 0)
    risk_evidence = len(risk.get("evidence", []))

    conflict = False
    arbitration = {}

    if brand_label == "positive" and risk_level in ["high", "critical"]:
        conflict = True
        if sentiment_evidence > risk_evidence:
            arbitration = {"final_label": "cautious_positive", "reason": "情感正面证据多于风险信号，综合判断为谨慎乐观"}
        elif risk_evidence > sentiment_evidence:
            arbitration = {"final_label": "cautious_negative", "reason": "风险信号证据多于情感正面，综合判断为谨慎消极"}
        else:
            arbitration = {"final_label": "controversial", "reason": "证据相当，存在争议，建议人工介入"}
    elif brand_label == "negative" and risk_level == "low":
        conflict = True
        arbitration = {"final_label": "cautious_negative", "reason": "情感负面但风险信号低，建议持续关注"}

    return {"conflict": conflict, "arbitration": arbitration, "status": {"arbiter": "completed"}}


def alert_node(state: AgentState) -> Dict[str, Any]:
    """告警判断：规则判断告警等级"""
    risk = state.get("risk_result", {})
    sentiment = state.get("sentiment_result", {})
    brand_label = sentiment.get("brand", {}).get("label", "neutral")

    alert_level = check_alert_level.invoke({
        "risk_keywords": risk.get("risks", []),
        "sentiment_label": brand_label,
        "severe_keywords": risk.get("severe_keywords", [])
    })

    return {"alert_level": alert_level, "status": {"alert": "completed"}}


def ticket_node(state: AgentState) -> Dict[str, Any]:
    """工单生成：预警/处置级生成，关注级不生成"""
    alert_level = state.get("alert_level", "无")
    brand = state.get("brand", "")
    risk = state.get("risk_result", {})

    result = create_ticket.invoke({
        "brand": brand,
        "alert_level": alert_level,
        "evidence": risk.get("evidence", [])
    })

    return {"ticket": result.get("ticket") or {}, "status": {"ticket": "completed"}}


def report_agent(state: AgentState) -> Dict[str, Any]:
    """报告生成"""
    if state.get("error"):
        return {
            "final_report": f"# 📊 舆情告警报告\n\n⚠️ {state['error']}",
            "status": {"report": "completed"}
        }

    raw_data = state.get("raw_data", [])
    if not raw_data:
        return {
            "final_report": "# 📊 舆情告警报告\n\n⚠️ **暂时无法获取数据**\n\n博查搜索服务暂时不可用。请稍后重试。",
            "status": {"report": "completed"}
        }

    report = skills.load("report")({
        "sentiment": state.get("sentiment_result", {}),
        "competitors": state.get("competitor_result", {}).get("competitors", []),
        "comparison": state.get("competitor_result", {}).get("comparison", []),
        "risks": state.get("risk_result", {}).get("risks", []),
        "risk_level": state.get("risk_result", {}).get("level", "low"),
        "raw_data": raw_data,
        "arbitration": state.get("arbitration", {}),
        "alert_level": state.get("alert_level", "无"),
        "ticket": state.get("ticket", {})
    })
    return {"final_report": report, "status": {"report": "completed"}}


# ============================================================
# 6. 构建图
# ============================================================

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("preprocess", preprocess_agent)
    graph.add_node("search", search_node)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("competitor", competitor_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("arbiter", arbiter_node)
    graph.add_node("alert", alert_node)
    graph.add_node("ticket", ticket_node)
    graph.add_node("report", report_agent)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "search")
    graph.add_edge("search", "sentiment")
    graph.add_edge("search", "competitor")
    graph.add_edge("search", "risk")
    graph.add_edge("sentiment", "arbiter")
    graph.add_edge("competitor", "arbiter")
    graph.add_edge("risk", "arbiter")
    graph.add_edge("arbiter", "alert")
    graph.add_edge("alert", "ticket")
    graph.add_edge("ticket", "report")
    graph.add_edge("report", END)

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# 7. 运行函数
# ============================================================

async def run_full_analysis(query: str) -> Dict[str, Any]:
    graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "brand": "",
        "main_query": "",
        "competitors": [],
        "error": "",
        "sentiment_result": {},
        "competitor_result": {},
        "risk_result": {},
        "arbitration": {},
        "conflict": False,
        "alert_level": "无",
        "ticket": {},
        "final_report": "",
        "status": {},
        "error_log": [],
        "raw_data": [],
        "competitor_raw_data": [],
    }

    config = {"configurable": {"thread_id": f"thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    final_state = await graph.ainvoke(initial_state, config=config)

    result = {
        "query": final_state["query"],
        "report": final_state.get("final_report", ""),
        "status": final_state.get("status", {}),
        "data": {
            "articles": final_state.get("raw_data", []),
            "sentiment": final_state.get("sentiment_result", {}),
            "competitors": final_state.get("competitor_result", {}).get("competitors", []),
            "comparison": final_state.get("competitor_result", {}).get("comparison", []),
            "risks": final_state.get("risk_result", {}).get("risks", []),
            "risk_level": final_state.get("risk_result", {}).get("level", "low"),
            "arbitration": final_state.get("arbitration", {}),
            "alert_level": final_state.get("alert_level", "无"),
            "ticket": final_state.get("ticket", {})
        }
    }

    await save_analysis(query, result)
    return result