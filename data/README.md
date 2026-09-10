# 数据治理与物化边界

本目录不提交下载的原始题目、训练样本、checkpoint 或缓存。可提交的是
source registry、manifest、数据治理文档与小型合成测试夹具。

## 首期批准来源

`../configs/data/sources.yaml` 固定以下首期来源的 Hugging Face revision、许可证和引用：

- OpenR1-Math-220k、DAPO-Math-17k、NuminaMath-TIR：训练候选；
- Omni-MATH：冻结评测来源。

MATH、MATH-500 和 AIME 不在首期可执行 registry 内：前两者的访问或衍生
许可证需要在下载前复核，AIME 竞赛题也须先确认内容使用权限。它们只能作为
待法律/许可复核的外部评测候选，不能直接混入训练或冻结测试集。

## 构建与审计

在完成下载权限和字段抽样检查后，使用固定 revision 构建：

```bash
uv run python scripts/data/build_dataset.py \
  --registry configs/data/sources.yaml \
  --output-dir data/processed/v1 \
  --manifest data/manifests/v1.json \
  --seed 20260910

uv run python scripts/data/audit_dataset.py --manifest data/manifests/v1.json
```

构建会将每个来源的 revision、加载/保留/隔离数量、去重统计、split task-ID 哈希
和 Parquet 文件哈希写入 manifest。`quarantine.jsonl` 保存被拒绝的行及机器可读
原因，便于复核而不改变原始下载内容。

任何来源、字段映射、许可证或 revision 的变更都必须先更新 registry、重建
manifest，并在提交前重新运行审计与数据契约测试。
