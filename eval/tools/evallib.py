"""Shared helpers for the Sutskever 30 question bank."""
import json, glob, os, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "bank")
# Answer keys and unused held-out forms live outside the public repo.
PRIVATE = os.environ.get("EVAL_PRIVATE_DIR", "/mnt/project-files/eval")
GRADING = os.environ.get("EVAL_GRADING_DIR", os.path.join(PRIVATE, ".grading"))
SESSIONS = os.path.join(ROOT, "sessions")
PROGRESS_JSON = os.path.join(ROOT, "progress.json")
PROGRESS_MD = os.path.join(ROOT, "progress.md")
CLUSTERS = {"rnn": "序列 / RNN", "cnn": "CNN / 残差", "attn": "Attention", "rel": "关系 / 图 / 记忆",
            "info": "生成 / 信息论", "scale": "规模 / 系统", "retr": "检索 / 长上下文"}
PREREQ = {"pre-la": "线性代数", "pre-calc": "微积分 / 梯度", "pre-prob": "概率 / 信息论基础", "pre-py": "Python / numpy", "pre-ml": "机器学习基础", "pre-nn": "神经网络基础"}
PUBLIC_FIELDS = ("id", "type", "difficulty", "papers", "cluster", "points", "stem", "opts", "tags")

def load_items():
    out = {}
    files = glob.glob(os.path.join(BANK, "items_*.json")) + glob.glob(os.path.join(PRIVATE, "bank", "items_*.json"))
    for f in sorted(set(files), key=os.path.basename):
        for it in json.load(open(f)):
            out[it["id"]] = it
    return out

def load_keys():
    out = {}
    for f in sorted(glob.glob(os.path.join(GRADING, "key_*.json"))):
        for k in json.load(open(f)):
            out[k["id"]] = k
    return out

def load_progress():
    if os.path.exists(PROGRESS_JSON):
        return json.load(open(PROGRESS_JSON))
    return {"sessions": [], "itemHistory": {}}

def save_progress(p):
    json.dump(p, open(PROGRESS_JSON, "w"), ensure_ascii=False, indent=1)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

def public(it):
    return {k: it.get(k) for k in PUBLIC_FIELDS}
