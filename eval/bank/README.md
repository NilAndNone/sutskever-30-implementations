# Sutskever 30 题库

- `items_*.json`：题目（不含答案）。字段：id, form（A/B/…，练习题为 null）, pool（heldout 测评卷 / practice 练习池）, type（single/multi/fill/short）, difficulty（L1/L2/L3）, papers（论文编号）, cluster, tags, points, stem（markdown，可含 ``` 代码块）, opts（[{k,t}]，仅选择题）。
- 答案与评分细则不在仓库里，放在 `$EVAL_PRIVATE_DIR/.grading/`（默认 `/mnt/project-files/eval/.grading/`），批改前不要看。
- 主题簇 cluster：rnn（P2,3,4,21,26）、cnn（P7,10,11,15）、attn（P6,13,14）、rel（P8,12,16,18,20）、info（P1,5,17,19,23,24,25）、scale（P9,22,27）、retr（P28,29,30）。
- 测评卷（heldout）只用于基线和复测，从不出现在练习里；各卷按同一份蓝图（主题簇 × 题型 × 难度）配平，分数可直接比较。

## 当前规模（2026-10-09）

| 集合 | 文件 | 题数 | 用途 |
|---|---|---|---|
| 测评卷 A | items_A.json | 39（100 分） | 基线 |
| 测评卷 B | 不在仓库（`$EVAL_PRIVATE_DIR/bank/items_B.json`） | 39（100 分） | 第一次复测，蓝图与 A 完全一致；考完后再提交 |
| 练习池 | items_practice.json | 98（7 个主题簇各 14 题） | 按薄弱主题抽题练习 |

工具：`../tools/make_session.py` 抽题组卷，`../tools/grade.py` 批改并更新 `../progress.md`。
