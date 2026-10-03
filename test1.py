# 单独跑一条否定用例
case = {
    "name": "否定_0",
    "category": "否定",
    "raw_data": [{"content": "蔚来汽车没有发生电池起火，官方已澄清", "siteName": "测试", "title": "测试标题", "url": ""}],
    "expected": {"alert_level": "无"}
}

from analyzer import risk_agent, alert_node, arbiter_node, ticket_node

state = {
    "raw_data": case["raw_data"],
    "brand": "测试品牌",
    "sentiment_result": {"brand": {"label": "negative", "evidence_count": 1}},
}
risk_result = risk_agent(state)
print("风险结果:", risk_result)
state["risk_result"] = risk_result.get("risk_result", {})
alert_result = alert_node(state)
print("告警等级:", alert_result)