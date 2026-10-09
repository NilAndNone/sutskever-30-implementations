"""Grade a submission and record progress.

  python3 grade.py SUBMISSION.json MANUAL.json [--summary SUMMARY.md]

SUBMISSION.json: the artifact db `submissions/<doc>` body ({sessionId, answers:{id:{ans,idk,conf}}}).
  Old v1 page submissions (version "baseline_v1", ids S1..Q4) are mapped to baseline-A automatically.
MANUAL.json: {itemId: {"score": x, "feedback": "..."}} — required for fill/short, optional
  feedback for choice items. Choice items are auto-scored.
Writes results/<sessionId>.json (push to db `results/<sessionId>`), updates progress.json/progress.md.
"""
import argparse, json, os
from collections import defaultdict
from evallib import *

def score_choice(it, key, ans):
    corr = key["answer"]
    if it["type"] == "single":
        return it["points"] if ans == corr else 0
    a, c = set(ans or ""), set(corr)
    if a == c: return it["points"]
    if a and a < c: return 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission"); ap.add_argument("manual"); ap.add_argument("--summary")
    a = ap.parse_args()
    sub = json.load(open(a.submission)); manual = json.load(open(a.manual))
    if "sessionId" not in sub and sub.get("version") == "baseline_v1":
        sub["sessionId"] = "baseline-A"
        sub["answers"] = {"A-" + k: v for k, v in sub["answers"].items()}
    sess = json.load(open(os.path.join(SESSIONS, sub["sessionId"] + ".json")))
    items, keys = load_items(), load_keys()
    res_items, total, mx = [], 0, 0
    byC = defaultdict(lambda: {"got": 0, "max": 0}); byD = defaultdict(lambda: {"got": 0, "max": 0})
    mis, shaky = [], []
    for q in sess["items"]:
        it, k = items[q["id"]], keys[q["id"]]
        s = sub["answers"].get(q["id"], {}); ans = "" if s.get("idk") else (s.get("ans") or "")
        m = manual.get(q["id"], {})
        if it["type"] in ("single", "multi"):
            sc = score_choice(it, k, ans)
        else:
            if "score" not in m: raise SystemExit(f"missing manual score for {q['id']}")
            sc = m["score"]
        conf = s.get("conf"); flag = None
        if conf == 3 and sc < it["points"] / 2: flag = "误解"; mis.append(q["id"])
        elif conf == 1 and sc == it["points"]: flag = "不稳"; shaky.append(q["id"])
        total += sc; mx += it["points"]
        for d, kk in ((byC, it["cluster"]), (byD, it["difficulty"])):
            d[kk]["got"] += sc; d[kk]["max"] += it["points"]
        res_items.append({"id": q["id"], "label": q["id"], "stem": it["stem"] + ("\n\n" + "\n".join(f"{o['k']}. {o['t']}" for o in it["opts"]) if it.get("opts") else ""),
                          "yours": "不会" if s.get("idk") else ans, "conf": conf,
                          "answer": k.get("answer") or k.get("rubric", ""), "explanation": k.get("explanation", ""),
                          "feedback": m.get("feedback", ""), "score": sc, "max": it["points"], "flag": flag,
                          "cluster": it["cluster"], "difficulty": it["difficulty"]})
    result = {"sessionId": sess["id"], "title": sess["title"], "kind": sess["kind"], "form": sess.get("form"),
              "gradedAt": now(), "total": round(total, 2), "max": mx, "byCluster": dict(byC), "byDifficulty": dict(byD),
              "misconceptions": mis, "shaky": shaky,
              "summary": open(a.summary).read() if a.summary else "", "items": res_items}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", sess["id"] + ".json")
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
    prog = load_progress()
    prog["sessions"] = [x for x in prog["sessions"] if x["sessionId"] != sess["id"]]
    prog["sessions"].append({k: result[k] for k in ("sessionId", "title", "kind", "form", "gradedAt", "total", "max", "byCluster", "byDifficulty", "misconceptions", "shaky")})
    for r in res_items:
        h = prog["itemHistory"].setdefault(r["id"], [])
        h[:] = [x for x in h if x["sessionId"] != sess["id"]] + [{"sessionId": sess["id"], "score": r["score"], "max": r["max"], "conf": r["conf"]}]
    save_progress(prog); write_md(prog)
    print(out, f"{total}/{mx}", "misconceptions:", mis, "shaky:", shaky)

def write_md(prog):
    ck = [s for s in prog["sessions"] if s["kind"] == "checkpoint"]
    pr = [s for s in prog["sessions"] if s["kind"] == "practice"]
    L = ["# 学习进度（Sutskever 30 测评）", "", "由 `tools/grade.py` 自动生成，请勿手改。", "",
         "## 测评卷（基线与复测，各卷配比相同，可直接比较）", ""]
    if ck:
        L.append("| 主题 | " + " | ".join(f"{s['title']}（{s['gradedAt'][:10]}）" for s in ck) + " |")
        L.append("|---|" + "---|" * len(ck))
        for c, name in CLUSTERS.items():
            row = []
            for s in ck:
                v = s["byCluster"].get(c)
                row.append(f"{100*v['got']/v['max']:.0f}% ({v['got']}/{v['max']})" if v and v["max"] else "—")
            L.append(f"| {name} | " + " | ".join(row) + " |")
        L.append("| **总分** | " + " | ".join(f"**{s['total']}/{s['max']}**" for s in ck) + " |")
        L += ["", "按难度：", ""]
        for s in ck:
            L.append(f"- {s['title']}：" + "，".join(f"{d} {100*v['got']/v['max']:.0f}%" for d, v in sorted(s["byDifficulty"].items())))
            L.append(f"  - 误解（C3 却答错）：{', '.join(s['misconceptions']) or '无'}；不稳（C1 却答对）：{', '.join(s['shaky']) or '无'}")
    else:
        L.append("还没有批改过的测评卷。")
    dg = [s for s in prog["sessions"] if s["kind"] == "diagnostic"]
    if dg:
        L += ["", "## 前置知识诊断", "", "| 层 | " + " | ".join(f"{s['title']}（{s['gradedAt'][:10]}）" for s in dg) + " |", "|---|" + "---|" * len(dg)]
        for c, name in PREREQ.items():
            row = []
            for s in dg:
                v = s["byCluster"].get(c)
                row.append(f"{100*v['got']/v['max']:.0f}% ({v['got']}/{v['max']})" if v and v["max"] else "—")
            L.append(f"| {name} | " + " | ".join(row) + " |")
    L += ["", "## 练习记录", ""]
    if pr:
        L += ["| 场次 | 日期 | 得分 | 涉及主题 |", "|---|---|---|---|"]
        for s in pr:
            L.append(f"| {s['title']} | {s['gradedAt'][:10]} | {s['total']}/{s['max']} | " + "、".join(CLUSTERS.get(c, c) for c in s["byCluster"]) + " |")
    else:
        L.append("还没有练习记录。")
    open(PROGRESS_MD, "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
