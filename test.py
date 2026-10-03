"""
一个脚本，跑一次，生成两个文件：
- test_cases.json：300 条测试用例
- test_results.json：测试结果 + 统计汇总
"""

import json
import random
from typing import Dict, List

from analyzer import (
    risk_agent,
    arbiter_node,
    alert_node,
    ticket_node,
)


# ============================================================
# 1. 测试数据池
# ============================================================

BRANDS = ["蔚来", "理想", "小鹏", "特斯拉", "比亚迪", "极氪", "问界", "智己"]
RISKS = ["投诉", "质量", "续航虚标", "充电故障", "异响", "漏水", "售后差", "维修贵"]
SEVERES = ["电池起火", "自动驾驶事故", "刹车失灵", "安全气囊", "失控", "自燃"]
POSITIVES = ["销量增长", "用户满意", "技术创新", "口碑良好", "交付提升"]


# ============================================================
# 2. 测试用例配置
# ============================================================

TEST_CONFIG = {
    "正常负面": 80,
    "正常正面": 60,
    "严重风险": 50,
    "否定": 50,
    "冲突": 60,
}


# ============================================================
# 3. 生成用例
# ============================================================

def make_raw_data(content: str) -> List[Dict]:
    return [{"content": content, "siteName": "测试", "title": "测试标题", "url": ""}]


def build_test_cases() -> List[Dict]:
    cases = []
    for category, count in TEST_CONFIG.items():
        for i in range(count):
            if category == "正常负面":
                brand = random.choice(BRANDS)
                risk = random.choice(RISKS)
                content = f"{brand}汽车被投诉{risk}问题，用户反映强烈"
                expected = {"alert_level": ["关注", "预警"]}

            elif category == "正常正面":
                brand = random.choice(BRANDS)
                positive = random.choice(POSITIVES)
                content = f"{brand}汽车{positive}，用户评价积极"
                expected = {"alert_level": "无"}

            elif category == "严重风险":
                brand = random.choice(BRANDS)
                severe = random.choice(SEVERES)
                content = f"{brand}汽车发生{severe}，引发广泛关注"
                expected = {"alert_level": "处置"}

            elif category == "否定":
                brand = random.choice(BRANDS)
                severe = random.choice(SEVERES)
                content = f"{brand}汽车没有发生{severe}，官方已澄清"
                expected = {"alert_level": "无"}

            elif category == "冲突":
                brand = random.choice(BRANDS)
                positive = random.choice(POSITIVES)
                severe = random.choice(SEVERES)
                content = f"{brand}汽车{positive}，但{severe}事故频发"
                expected = {"alert_level": "处置"}

            else:
                continue

            cases.append({
                "name": f"{category}_{i}",
                "category": category,
                "raw_data": make_raw_data(content),
                "expected": expected
            })
    return cases


# ============================================================
# 4. 跑测试
# ============================================================

def run_single_case(case: Dict) -> Dict:
    # 情感标签：正常正面、否定、冲突场景 = positive，其他 = negative
    sentiment_label = "positive" if case["category"] in ["正常正面", "否定", "冲突"] else "negative"

    state = {
        "raw_data": case["raw_data"],
        "brand": "测试品牌",
        "sentiment_result": {
            "brand": {
                "label": sentiment_label,
                "evidence_count": 1
            }
        },
    }
    risk_result = risk_agent(state)
    state["risk_result"] = risk_result.get("risk_result", {})
    arbiter_result = arbiter_node(state)
    state["arbitration"] = arbiter_result.get("arbitration", {})
    alert_result = alert_node(state)
    state["alert_level"] = alert_result.get("alert_level", "无")
    ticket_result = ticket_node(state)
    state["ticket"] = ticket_result.get("ticket", {})

    return {
        "name": case["name"],
        "category": case["category"],
        "expected": case["expected"],
        "actual": {
            "alert_level": state["alert_level"],
            "ticket_id": state["ticket"].get("ticket_id", ""),
        }
    }


def is_alert_correct(expected: Dict, actual: str) -> bool:
    """判断告警等级是否匹配（支持列表）"""
    exp = expected.get("alert_level")
    if exp is None:
        return False
    if isinstance(exp, list):
        return actual in exp
    return exp == actual


def run_all_tests():
    # 1. 生成用例
    cases = build_test_cases()
    with open("test_cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(cases)} 条测试用例 → test_cases.json")

    # 2. 跑测试
    results = []
    for case in cases:
        try:
            results.append(run_single_case(case))
        except Exception as e:
            print(f"[失败] {case['name']}: {e}")

    # 3. 统计
    total = len(results)
    alert_correct = sum(
        1 for r in results
        if is_alert_correct(r["expected"], r["actual"]["alert_level"])
    )
    ticket_correct = sum(
        1 for r in results
        if bool(r["actual"]["ticket_id"]) == (r["actual"]["alert_level"] in ["预警", "处置"])
    )

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if is_alert_correct(r["expected"], r["actual"]["alert_level"]):
            categories[cat]["correct"] += 1

    summary = {
        "total": total,
        "alert_accuracy": round(alert_correct / total * 100, 1) if total else 0,
        "ticket_accuracy": round(ticket_correct / total * 100, 1) if total else 0,
        "by_category": {
            cat: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"] * 100, 1)
            }
            for cat, stats in categories.items()
        },
        "results": results
    }

    # 4. 写结果
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5. 打印
    print("=" * 60)
    print(f"测试用例总数：{total}")
    print(f"告警准确率：{alert_correct}/{total} = {summary['alert_accuracy']}%")
    print(f"工单生成准确率：{ticket_correct}/{total} = {summary['ticket_accuracy']}%")
    print("=" * 60)
    print("\n按类别统计：")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} = {stats['accuracy']}%")
    print("\n结果已写入 test_results.json")


if __name__ == "__main__":
    run_all_tests()