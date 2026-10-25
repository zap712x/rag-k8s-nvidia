"""回归测试集：固化典型问题，验证每次改动后系统行为不退化。

判定标准：
- expect="answer"：回答中不应出现拒答话术，且应引用来源
- expect="refuse"：回答中应出现拒答话术
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from generator.generate import answer

REFUSE_TEXT = "根据现有资料无法回答"

# (问题, 期望行为, 期望命中的来源关键词)
CASES = [
    ("What is a node in Kubernetes?", "answer", "nodes"),
    ("What is a cluster in Kubernetes?", "answer", None),
    ("What is the difference between a node and a cluster?", "answer", None),
    ("How to deploy an application on Kubernetes?", "answer", "deployment"),
    ("如何制作红烧肉？", "refuse", None),
]


def run():
    passed = failed = 0
    for q, expect, keyword in CASES:
        time.sleep(21)
        ans, hits = answer(q)
        refused = REFUSE_TEXT in ans

        ok = (expect == "refuse") == refused
        # answer 类问题额外检查：期望的来源是否进入 Top-3
        if ok and expect == "answer" and keyword:
            ok = any(keyword in h["source"].lower() for h in hits)

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += (not ok)
        print(f"[{status}] {q}")
        print(f"       期望={expect} 实际={'拒答' if refused else '回答'} "
              f"Top-1={hits[0]['source']} ({hits[0]['distance']:.3f})")

    print(f"\n结果：{passed} 通过 / {failed} 失败 / 共 {len(CASES)} 条")


if __name__ == "__main__":
    run()
