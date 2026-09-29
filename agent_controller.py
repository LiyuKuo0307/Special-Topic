# -*- coding: utf-8 -*-
"""
AI Agent 決策控制層（對應第六節）：依內容特性與傳播狀況動態規劃查核流程，
而非固定步驟。實作第六節「觸發條件 → Agent 對應動作」表的決策邏輯。
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AgentPlan:
    """Agent 依當前狀態決定要啟用哪些分析模組與策略，並記錄決策理由（供 Dashboard 展示）。"""
    expand_keywords: bool = False
    run_evidence_lookup: bool = False
    run_propagation: bool = False
    mark_insufficient: bool = False
    platform_weight: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)


def plan_actions(topic: str, rows: list, analysis: dict, min_rows: int = 8) -> AgentPlan:
    """
    對應第六節動態決策情境表。回傳一份 AgentPlan（純函式，方便單元測試）。
    """
    plan = AgentPlan()

    # 1) 蒐集到的資料量不足 → 擴展搜尋關鍵字
    if analysis.get("total_count", 0) < min_rows:
        plan.expand_keywords = True
        plan.decisions.append(("資料量不足", "自動擴展搜尋關鍵字，擴大蒐集範圍"))

    # 2) 內容疑似含未經證實之具體事實陳述 → 啟動 LLM 證據彙整
    sus = analysis.get("suspicious_comments", [])
    if any(s.get("has_verifiable_fact") for s in sus) or sus:
        plan.run_evidence_lookup = True
        plan.decisions.append(("疑似具體事實陳述", "啟動 LLM 證據彙整，比對外部可信來源"))

    # 3) 短時間內留言量異常暴增 → 啟動傳播路徑分析
    trend = analysis.get("daily_trend", [])
    if any(t.get("is_anomaly") for t in trend):
        plan.run_propagation = True
        plan.decisions.append(("聲量異常暴增", "啟動傳播路徑分析，標記潛在擴散事件"))

    # 5) 特定平台討論度顯著偏高 → 提高該平台權重
    plat_counts = {}
    for r in rows:
        plat_counts[r.get("platform", "unknown")] = plat_counts.get(r.get("platform", "unknown"), 0) + 1
    if plat_counts:
        total = sum(plat_counts.values())
        for p, c in plat_counts.items():
            w = round(c / total, 2)
            plan.platform_weight[p] = w
            if w >= 0.6:
                plan.decisions.append((f"平台 {p} 討論度偏高", f"提高其權重至 {w}"))
    return plan


def note_insufficient(plan: AgentPlan, fact_check: dict) -> AgentPlan:
    """4) 現有可信來源不足以判定 → 標註證據不足、提示人工查核。"""
    from .schema import CRED_INSUFFICIENT, CRED_NEEDS_REVIEW
    if fact_check and fact_check.get("credibility_label") in (CRED_INSUFFICIENT, CRED_NEEDS_REVIEW):
        plan.mark_insufficient = True
        plan.decisions.append(("可信來源不足", "標註為證據不足，提示需人工查核，避免武斷判定"))
    return plan
