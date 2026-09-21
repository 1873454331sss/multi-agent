import os
import json
import operator
from datetime import datetime
from typing import Annotated, List, Dict, Any, Literal, TypedDict
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
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
            return articles
        except Exception:
            return []

    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
            prompt = f"""分析以下文本的整体情感倾向，严格按JSON返回。
文本：{text[:1500]}
返回格式：{{"sentiment": "positive", "score": 0.85}} 或 negative / neutral"""
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": "你是情感分析专家，只返回JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            result = json.loads(resp.choices[0].message.content.strip())
            return {
                "label": result.get("sentiment", "neutral"),
                "score": result.get("score", 0.5),
                "source": "deepseek-v4-flash"
            }
        except Exception:
            return LocalDataTools._fallback_sentiment(text)

    @staticmethod
    def _fallback_sentiment(text: str) -> Dict:
        positive_keywords = ["强劲", "突破", "领先", "增长", "积极", "升级", "出色", "优秀", "创新", "超越", "反响热烈", "显著"]
        negative_keywords = ["下滑", "下降", "危机", "负面", "投诉", "质量", "召回", "诉讼", "压力", "监管", "调查", "挑战"]
        pos = sum(1 for kw in positive_keywords if kw in text)
        neg = sum(1 for kw in negative_keywords if kw in text)
        total = pos + neg
        if total == 0:
            return {"label": "neutral", "score": 0.5, "source": "fallback-keyword"}
        score = pos / total
        label = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        return {"label": label, "score": round(score, 2), "source": "fallback-keyword"}

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        pool = ["小米", "华为", "特斯拉", "苹果", "比亚迪", "宁德时代", "理想", "蔚来", "小鹏", "吉利"]
        return [e for e in pool if e in text][:5]

    @staticmethod
    def generate_report(data: Dict) -> str:
        sentiment = data.get("sentiment", {})
        competitors = data.get("competitors", [])
        risks = data.get("risks", [])
        raw_data = data.get("raw_data", [])
        risk_level = data.get("risk_level", "low")

        lines = []
        lines.append("# 📊 舆情分析报告")
        lines.append("")
        lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        source = sentiment.get("source", "unknown")
        lines.append(f"**分析引擎**：{'DeepSeek V4 Flash' if source == 'deepseek-v4-flash' else '关键词分析(备用)'}")
        lines.append("**数据来源**：博查 Web Search API")
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
        lines.append(f"- **整体倾向**：{label_map.get(sentiment.get('label', 'neutral'), '未知')}")
        lines.append(f"- **情感得分**：{sentiment.get('score', 0.5)}（越接近1越积极）")
        lines.append("")
        lines.append("## 三、竞品识别")
        lines.append("")
        if competitors:
            lines.append(f"- 发现 **{len(competitors)}** 个相关竞品：{', '.join(competitors)}")
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
        if sentiment.get('label') == 'positive' and not risks:
            conclusion = "整体舆情态势良好，建议继续保持关注，可考虑加大投入。"
        elif sentiment.get('label') == 'positive' and risks:
            conclusion = "舆情总体积极，但存在一定风险点，建议重点监测风险信号并制定应对预案。"
        elif sentiment.get('label') == 'negative' and risks:
            conclusion = "舆情偏消极且存在风险，建议启动应急响应机制，积极应对市场关切。"
        elif risk_level == "critical":
            conclusion = "⚠️ 检测到严重风险信号，建议立即启动应急预案，组织专项会议讨论应对策略。"
        else:
            conclusion = "舆情态势复杂，建议人工介入做进一步深度分析。"
        lines.append(f"> {conclusion}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*本报告由多Agent舆情分析系统自动生成，仅供参考。*")
        return "\n".join(lines)


# ============================================================
# 2. LangGraph 状态定义
# ============================================================

def merge_dicts(a: Dict, b: Dict) -> Dict:
    merged = a.copy()
    merged.update(b)
    return merged


class AgentState(TypedDict):
    query: str
    input_type: Literal["brand", "industry", "url", "default"]
    sentiment_result: Dict[str, Any]
    competitor_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    final_report: str
    status: Annotated[Dict[str, str], merge_dicts]
    retry_count: Annotated[Dict[str, int], merge_dicts]
    error_log: Annotated[List[str], operator.add]
    raw_data: Annotated[List[Dict], operator.add]


# ============================================================
# 3. Skill 懒加载器
# ============================================================

class SkillLoader:
    def __init__(self):
        self.tools = LocalDataTools()

    @lru_cache(maxsize=10)
    def load(self, skill_name: str):
        skill_map = {
            "load_data": self.tools.load_data,
            "sentiment": self.tools.analyze_sentiment,
            "entity": self.tools.extract_entities,
            "report": self.tools.generate_report,
        }
        if skill_name not in skill_map:
            raise ValueError(f"未知的Skill: {skill_name}")
        return skill_map[skill_name]


skills = SkillLoader()


# ============================================================
# 4. Agent 节点
# ============================================================

def supervisor_agent(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    if query.startswith(("http://", "https://")):
        input_type = "url"
    elif any(kw in query for kw in ["小米", "华为", "特斯拉", "苹果", "比亚迪", "蔚来", "理想"]):
        input_type = "brand"
    elif any(kw in query for kw in ["行业", "趋势", "市场", "发展"]):
        input_type = "industry"
    else:
        input_type = "default"
    return {"input_type": input_type, "status": {"supervisor": "completed"}}


def sentiment_agent(state: AgentState) -> Dict[str, Any]:
    try:
        raw_data = skills.load("load_data")(state["query"])
        if not raw_data:
            return {"status": {"sentiment": "failed"}, "error_log": ["情感分析失败: 博查搜索无数据"]}
        full_text = " ".join([item.get("content", "") for item in raw_data])
        result = skills.load("sentiment")(full_text)
        return {"raw_data": raw_data, "sentiment_result": result, "status": {"sentiment": "completed"}}
    except Exception as e:
        return {"status": {"sentiment": "failed"}, "error_log": [f"情感分析失败: {str(e)}"]}


def competitor_agent(state: AgentState) -> Dict[str, Any]:
    try:
        raw_data = state.get("raw_data", [])
        if not raw_data:
            raw_data = skills.load("load_data")(state["query"])
        if not raw_data:
            return {"status": {"competitor": "failed"}, "error_log": ["竞品分析失败: 无数据"]}
        all_text = " ".join([item.get("content", "") for item in raw_data])
        entities = skills.load("entity")(all_text)
        return {
            "raw_data": raw_data,
            "competitor_result": {"competitors": entities[:5], "count": len(entities)},
            "status": {"competitor": "completed"}
        }
    except Exception as e:
        return {"status": {"competitor": "failed"}, "error_log": [f"竞品分析失败: {str(e)}"]}


def risk_agent(state: AgentState) -> Dict[str, Any]:
    try:
        raw_data = state.get("raw_data", [])
        if not raw_data:
            raw_data = skills.load("load_data")(state["query"])
        if not raw_data:
            return {"status": {"risk": "failed"}, "error_log": ["风险检测失败: 无数据"]}
        all_text = " ".join([item.get("content", "") for item in raw_data])
        risk_keywords = ["下滑", "下降", "危机", "负面", "投诉", "质量", "召回", "诉讼", "监管", "压力", "调查", "挑战", "审查", "停止"]
        found = list(set([kw for kw in risk_keywords if kw in all_text]))
        level = "critical" if len(found) > 4 else "high" if len(found) > 2 else "medium" if len(found) > 0 else "low"
        return {
            "raw_data": raw_data,
            "risk_result": {"risks": found, "level": level, "count": len(found)},
            "status": {"risk": "completed"}
        }
    except Exception as e:
        return {"status": {"risk": "failed"}, "error_log": [f"风险检测失败: {str(e)}"]}


def report_agent(state: AgentState) -> Dict[str, Any]:
    raw_data = state.get("raw_data", [])
    if not raw_data:
        return {
            "final_report": "# 📊 舆情分析报告\n\n⚠️ **暂时无法获取数据**\n\n博查搜索服务暂时不可用。请稍后重试。",
            "status": {"report": "completed"}
        }
    report = skills.load("report")({
        "sentiment": state.get("sentiment_result", {}),
        "competitors": state.get("competitor_result", {}).get("competitors", []),
        "risks": state.get("risk_result", {}).get("risks", []),
        "risk_level": state.get("risk_result", {}).get("level", "low"),
        "raw_data": raw_data
    })
    return {"final_report": report, "status": {"report": "completed"}}


# ============================================================
# 5. 构建图
# ============================================================

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("competitor", competitor_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "sentiment")
    graph.add_edge("supervisor", "competitor")
    graph.add_edge("supervisor", "risk")
    graph.add_edge("sentiment", "report")
    graph.add_edge("competitor", "report")
    graph.add_edge("risk", "report")
    graph.add_edge("report", END)

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# 6. 运行函数
# ============================================================

async def run_full_analysis(query: str) -> Dict[str, Any]:
    graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "input_type": "default",
        "sentiment_result": {},
        "competitor_result": {},
        "risk_result": {},
        "final_report": "",
        "status": {},
        "retry_count": {},
        "error_log": [],
        "raw_data": [],
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
            "risks": final_state.get("risk_result", {}).get("risks", []),
            "risk_level": final_state.get("risk_result", {}).get("level", "low")
        }
    }

    await save_analysis(query, result)
    return result