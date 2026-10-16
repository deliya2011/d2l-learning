"""算一段任务的用时与比值，避免心算错（2026-10-17 记错过一次）。

用法：
    .venv/bin/python tools/elapsed.py 01:00:32 01:12:09 10
    # 起点 终点 预算分钟  → 印出实际用时与比值

起点与终点同一天，按北京时间；跨天时加 --date 指定任一端的日期（默认今天）：
    .venv/bin/python tools/elapsed.py 18:03:42 18:24:11 40

规则：**只用起止时刻相减**，不要把心跳 job 到点的时刻混进来——那是上一次记错的原因。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

FMT = "%H:%M:%S"


def parse(s: str) -> datetime:
    today = datetime.now().date()
    return datetime.combine(today, datetime.strptime(s, FMT).time())


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        print(__doc__)
        return 2
    start, end, budget = parse(args[0]), parse(args[1]), float(args[2])
    if end < start:  # 跨过午夜
        end += timedelta(days=1)

    sec = (end - start).total_seconds()
    minutes, seconds = int(sec // 60), int(round(sec % 60))
    ratio = sec / (budget * 60)

    print(f"起点 {args[0]}  终点 {args[1]}")
    print(f"实际 {minutes} 分 {seconds} 秒（{sec:.0f} 秒）")
    print(f"预算 {budget:g} 分钟  比值 {ratio:.2f}")

    if ratio < 0.7:
        print("→ 偏快（连续低于 0.7 时按 AGENTS.md：加难度或压节奏）")
    elif ratio > 1.5:
        print("→ 偏慢（先查讲义清不清楚、题目有没有超纲，再考虑状态）")
    else:
        print("→ 在预算附近")
    return 0


if __name__ == "__main__":
    sys.exit(main())
