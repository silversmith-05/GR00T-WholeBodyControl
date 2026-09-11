# Inspire 数据合并与训练划分

使用数据采集环境运行专用入口。它只排除 `discarded_episode_indices` 中的完整轨迹，保留正常等待手部确认的帧、SMPL 缺失或陈旧记录、拇指旋转和所有原始时间戳。合并不会插值、切段、重新编码视频或启动机器人。

```bash
.venv_data_collection/bin/python gear_sonic/scripts/merge_inspire_datasets.py \
  --dataset-path \
    outputs/2026-09-10-22-26-48 \
    outputs/2026-09-11-15-10-34 \
    outputs/2026-09-11-16-13-22 \
  --output-path processed_datasets/pick_up_ball_inspire_v1 \
  --eval-count 8 \
  --split-seed 42
```

输入顺序决定批次顺序；各批次内按原始 episode 编号排序。评估条数按各批次保留条数的比例分配，余数按大小补齐，同余数按输入顺序处理。然后按批次顺序使用一个 `numpy.default_rng(42)` 从原始编号中无放回采样。

上述三个批次会分别留出 1、5、2 条评估轨迹，原始编号为：

- `2026-09-10-22-26-48`：2。
- `2026-09-11-15-10-34`：37、50、57、67、77。
- `2026-09-11-16-13-22`：17、30。

输出目录包含三个独立的 LeRobot v2.1 数据集：

| 目录 | 轨迹数 | 帧数 | 视频数 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `all/` | 82 | 82,086 | 246 | 完整合并档案 |
| `train/` | 74 | 73,860 | 222 | 正式训练输入 |
| `eval/` | 8 | 8,226 | 24 | 独立评估 |

每个数据集只重写 `episode_index` 和全局 `index`，其他 Parquet 列与原始列保持一致。`meta/episodes_stats.jsonl` 沿用未修改字段的原始统计，重算两个索引字段的统计。`info.json` 的 `splits` 分别为 `all`、`train`、`validation`，范围均从本目录的 episode 0 开始。

`meta/source_episodes.jsonl` 记录每条输出轨迹的原始目录、原始编号、完整合并集编号、训练划分及源文件 SHA-256。根目录的 `merge_report.json` 记录配置、总数、保留的诊断帧数、全部已使用源文件的 SHA-256 和验收结果。

脚本在写入前检查配置、字段、任务、文件、帧数和已有统计；输出目录已存在时拒绝覆盖。全部产物先写入同文件系统的临时目录，完成 Parquet 逐列对比、视频 SHA-256 验证、来源映射验证、源文件未变验证和当前 LeRobot v2.1 元数据加载后，才发布最终目录。失败时清理本次临时产物，保留原始输入。

正式 GR00T 训练只指向 `train/`，评估使用 `eval/`。本工具不生成 GR00T 的 `stats.json`：应在配置好 66 维动作后从 `train/` 生成训练统计，评估复用训练／checkpoint 的归一化参数。不要把 `all/` 用作训练输入后再将 `eval/` 当作独立评估。

离线测试：

```bash
.venv_data_collection/bin/python -m unittest gear_sonic.tests.test_merge_inspire_datasets -v
```
