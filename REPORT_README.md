# ProRL 课程报告代码与实验归档

本目录只保留课程报告需要的 Baseline、M1、M2、M3、原生 Pareto、M4 及 M4 机制验证。其余早期候选方案均已从入口、训练器和辅助函数中删除。

## 整理顺序

1. **Baseline**：原始 ProRL，`method_mode=none`。
2. **M1 难度课程学习**：按初始目标排名从易到难调整样本权重。
3. **M2 精英轨迹记忆**：保存每个输入当前最优轨迹并加入回放损失。
4. **M3 鲁棒奖励归一化**：使用有界 `tanh` 压缩异常奖励 z-score。
5. **原生 Pareto 与 M4 轻量对比**：标准非支配 front rank 直接替换位置优势，对比 M4 的组内支配分数叠加方式。
6. **M4 Pareto 组内相对优势**：正式 50 epoch 结果与完整实现。
7. **五组机制验证**：轻量 Baseline、M4 标准版本、M4-Shuffled、M4-Reverse、M4-Length-normalized。

## 代码位置

- `prorl_variants.py`
  - `pareto_front_scores`：原生 Pareto 非支配分层。
  - `pareto_rollout_scores`：M4 组内支配数减被支配数。
  - `curriculum_sample_weights`：M1。
  - `EliteTrajectoryMemory`：M2。
  - `bounded_normalization`：M3。
- `trainer_RL_prorl.py`
  - `method_mode=none`：Baseline。
  - `pareto_advantage` 分支：原生 Pareto、M4 和机制控制。
  - `elite_memory` 分支：M2 的读取、回放与更新。
  - `difficulty_curriculum` 分支：M1 的 loss 权重。
  - `robust_normalization` 分支：M3 的奖励归一化。
- `Proactive_RL_prorl.py`：只暴露以上方法所需的命令行参数。

## 编号脚本

```text
scripts/report/01_run_baseline.sh
scripts/report/02_run_m1_curriculum.sh
scripts/report/03_run_m2_elite_memory.sh
scripts/report/04_run_m3_robust_normalization.sh
scripts/report/05_run_pareto_light_compare.sh
scripts/report/06_run_m4_pareto.sh
scripts/report/07_run_m4_mechanism_checks.sh
```

正式实验统一使用 ML-1M、seed 1、50 epoch 和原模型超参数。轻量实验统一使用 seed 1、10 epoch、每 epoch 100 个训练 batch、20 个评估 batch，并采用 validation 监控后一次性 test 评估。

运行前需将数据集和预训练权重放回原 README 约定的位置，或设置：

```bash
PRETRAINED_CKPT=/absolute/path/to/checkpoint.pth GPU_ID=7 \
  bash scripts/report/01_run_baseline.sh
```

## 原始输出

```text
outputs/raw/full_50epoch/
  baseline/
  m1_curriculum/
  m2_elite_memory/
  m3_robust_normalization/
  m4_pareto/

outputs/raw/light_10epoch/
  baseline/
  vanilla_front/
  m4_normal/
  m4_shuffled/
  m4_reverse/
  m4_length_normalized/
```

每个目录保留已有的 `console.log`、`results/*.json`、日志和检查点目录。原始状态文件保存在 `outputs/raw/source_status/`。这些日志是实验发生时的只读证据，内部可能保留旧运行代号；对外脚本、目录、汇总表与报告统一采用 M1--M4 口径。

## 汇总表

- `outputs/summary/all_results.md`：11 个唯一实验条目的统一大总表。
- `outputs/summary/all_results.csv`：便于 Excel 分析。
- `outputs/summary/all_results.json`：保留结构化字段与原始文件路径。

重新生成：

```bash
python scripts/report/collect_all_results.py
```

正式 50 epoch 与轻量协议训练预算不同，只能在各自表内比较，不能直接比较绝对数值。
