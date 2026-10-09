# 测评与 baseline

这个目录记录 Sutskever 30 学习进度的题库、考卷和成绩。

## Baseline（2026-10-09）

| 测评 | 得分 | 结果 |
|---|---|---|
| 论文测评 · A 卷（39 题） | 6 / 100 | `results/baseline-A.json` |
| 前置知识诊断 P0（25 题） | 17 / 56 | `results/prereq-P0.json` |

分主题明细见 `progress.md`（由 `tools/grade.py` 自动生成）。

## 什么不在仓库里

仓库是公开的，所以下面两类文件只放在私有目录 `$EVAL_PRIVATE_DIR`（默认 `/mnt/project-files/eval`）：

- `.grading/`：所有答案与评分细则。
- `bank/items_B.json`：还没考过的测评卷 B。提前看到题目，复测分数就不可比了。考完后再提交。

`tools/evallib.py` 会同时从仓库的 `bank/` 和私有目录的 `bank/` 读题，从 `$EVAL_GRADING_DIR`（默认 `$EVAL_PRIVATE_DIR/.grading`）读答案。

已经考过的 A 卷和 P0 的答案会随 `results/*.json` 一起公开，这两套卷以后只做参考，不再用于计分。
