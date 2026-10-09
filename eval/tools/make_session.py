"""Build a session (no answers) from the bank.

  python3 make_session.py checkpoint --form B
  python3 make_session.py practice --n 12 [--clusters rnn,attn] [--seed 1]

Writes sessions/<id>.json; push it to the artifact db as sessions/<id> and set config/active.
"""
import argparse, json, os, random, datetime
from collections import defaultdict
from evallib import *

def cluster_rates(prog):
    got, mx = defaultdict(float), defaultdict(float)
    for s in prog["sessions"][-6:]:          # recent window, checkpoints and practice alike
        for c, v in s["byCluster"].items():
            got[c] += v["got"]; mx[c] += v["max"]
    return {c: (got[c] / mx[c] if mx[c] else None) for c in CLUSTERS}

def diff_mix(rate):
    if rate is None: return {"L1": .25, "L2": .5, "L3": .25}
    if rate < .5:   return {"L1": .4, "L2": .5, "L3": .1}
    if rate < .8:   return {"L1": .15, "L2": .5, "L3": .35}
    return {"L1": 0, "L2": .4, "L3": .6}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["checkpoint", "practice"])
    ap.add_argument("--form"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--clusters"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    items, prog = load_items(), load_progress()
    rnd = random.Random(a.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    if a.kind == "checkpoint":
        chosen = [it for it in items.values() if it.get("pool") == "heldout" and it.get("form") == a.form]
        if not chosen: raise SystemExit(f"no held-out items for form {a.form}")
        done = [s for s in prog["sessions"] if s.get("form") == a.form]
        if done: print(f"WARNING: form {a.form} was already used in {done[0]['sessionId']}; scores won't be a fresh measurement")
        sid = f"checkpoint-{a.form}-{stamp}"
        sess = {"id": sid, "title": f"复测 · {a.form} 卷", "kind": "checkpoint", "form": a.form, "timeLimitMin": 120,
                "note": f"平行测评卷 {a.form}，与基线卷 A 主题和难度配比相同，分数可直接比较。"}
        order = {"single": 0, "multi": 1, "fill": 2, "short": 3}
        chosen.sort(key=lambda it: (order[it["type"]], int(''.join(ch for ch in it["id"].split('-')[-1] if ch.isdigit()) or 0)))
    else:
        rates = cluster_rates(prog)
        allowed = set(a.clusters.split(",")) if a.clusters else set(CLUSTERS)
        pool = [it for it in items.values() if it.get("pool") == "practice" and it["cluster"] in allowed]
        hist = prog["itemHistory"]
        weight = {c: (1 - (rates[c] if rates[c] is not None else .5)) + .15 for c in allowed}
        # allocate counts per cluster proportional to weakness
        tot = sum(weight.values()); quota = {c: max(1, round(a.n * weight[c] / tot)) for c in allowed}
        while sum(quota.values()) > a.n: quota[max(quota, key=quota.get)] -= 1
        chosen = []
        for c, q in quota.items():
            mix = diff_mix(rates[c]); cand = [it for it in pool if it["cluster"] == c]
            def seen(it):   # 0 unseen < 1 missed last time < 2 mastered last time
                h = hist.get(it["id"], [])
                return 0 if not h else (1 if h[-1]["score"] < h[-1]["max"] else 2)
            for _ in range(q):
                if not cand: break
                levels = [d for d in mix if mix[d] > 0 and any(it["difficulty"] == d for it in cand)] or list({it["difficulty"] for it in cand})
                d = rnd.choices(levels, weights=[mix.get(x, 0) or 1e-6 for x in levels])[0]
                opts = [it for it in cand if it["difficulty"] == d]
                best = min(seen(it) for it in opts)
                pick = rnd.choice([it for it in opts if seen(it) == best])
                chosen.append(pick); cand.remove(pick)
        rnd.shuffle(chosen)
        order = {"single": 0, "multi": 1, "fill": 2, "short": 3}
        chosen.sort(key=lambda it: order[it["type"]])
        sid = f"practice-{stamp}"
        sess = {"id": sid, "title": f"练习 · {stamp[:8]}", "kind": "practice", "form": None,
                "timeLimitMin": max(15, 4 * len(chosen)),
                "note": "按最近得分偏低的主题抽题。交卷批改后可在「成绩与解析」里看解析。"}
    sess["items"] = [public(it) for it in chosen]
    os.makedirs(SESSIONS, exist_ok=True)
    path = os.path.join(SESSIONS, sid + ".json")
    json.dump(sess, open(path, "w"), ensure_ascii=False)
    from collections import Counter
    print(path, len(chosen), "items", dict(Counter(it["cluster"] for it in chosen)), dict(Counter(it["difficulty"] for it in chosen)))

if __name__ == "__main__":
    main()
