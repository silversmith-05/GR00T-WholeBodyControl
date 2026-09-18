# 双臂轨迹通过 G1 Encoder 重放

入口为根目录的 `start_arm_replay.sh`。不需要 Pico，不控制 Dex3 或因时手。
当前实现标识为 `ARM_REPLAY_G1_ENCODER_V1`；启动器会拒绝使用旧的直接覆盖电机目标版本。

## 控制路径

```text
录制的双臂 q → 连续位置/速度参考 ─┐
                                 ├→ G1 Encoder → 64D token
IDLE planner 的腿腰及根部参考 ────┘                ↓
                          机器人实时状态 → SONIC policy
                                                 ↓
                            29 个电机的 q_target、固定 PD → LowCmd
```

只替换**编码器运动参考**中对应硬件索引 15–28 的双臂位置和速度，映射到 IsaacLab
顺序后送入 G1 Encoder（mode 0）。腿腰参考和根部姿态来自原 IDLE planner。
全身 29 个最终电机目标都由原 SONIC 策略产生，**不再覆盖策略输出的双臂 q_target**。
动作历史保留策略实际输出。底层 `dq_target=0`、`tau_ff=0`，Kp/Kd 沿用原设置。

编码器每个控制周期读取当前及未来 `0, 0.1, ..., 0.9` 秒的 10 帧 q/dq 参考，
各帧使用对应时刻的双臂轨迹；参考朝向使用现有的实时机器人姿态归一化逻辑。
因此在线生成 token，不把离线固定 token 当作与机器人状态无关的指令。
50 Hz 策略更新、500 Hz `LowCommandWriter` 发送保持不变。

腿腰仍由 SONIC 控制，其数值可能随双臂动作及平衡反馈改变，不保证等于原来纯 IDLE 输出。
策略跟踪能力需要实机验证；编码器缺失、G1 模式配置不匹配或参考采集失败时停止，
不会退回其他编码模式而忽略双臂轨迹。

## 当前记录与插值

`recordings/metadata.json` 和 `recordings/samples.csv` 包含左右臂各 7 个关节的
`rt/lowstate` 反馈：93,518 帧，约 1000 Hz，时长 93.538575563 秒。

准备阶段默认在相对时间 `0, 0.02, 0.04, ...` 取最近的原始采样点，并保留末帧，
得到 4,678 帧。时间基准为 `received_monotonic_ns`，不因跳帧改变动作时长。
运行时对这些位置使用保形三次 Hermite 插值，速度使用同一插值曲线的解析导数。
首末端速度为零，内部斜率采用同号相邻割线的加权调和平均，局部极值处斜率为零，
避免插值越过相邻位置范围。参考速度是编码器输入，不是下发到电机的速度前馈。

原始 `dq` 和 `tau_est` 不作为命令或参考速度使用。记录的是实测轨迹，不是原来的
SONIC `q_target`；现在将这条实测轨迹作为期望运动，让策略生成新的控制输入。

## 离线准备和检查

```bash
cd /home/pku/GR00T-WholeBodyControl
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2

# 只生成降采样轨迹和报告
./start_arm_replay.sh eno2 --prepare-only

# 生成轨迹并打印部署命令
./start_arm_replay.sh eno2 --dry-run

# 检查数据、q/dq 插值和 G1 编码配置；不创建 G1Deploy、不初始化 DDS
./start_arm_replay.sh eno2 --check
```

派生文件为 `outputs/arm_replay/arms.csv` 和 `outputs/arm_replay/report.json`，
不修改 `recordings`。报告包含采样误差、关节范围、文件 SHA-256、插值及编码参数。
文件检查验证角度有限性、URDF 范围、时间递增及采样间隙不超过 0.1 秒。
`--check` 不运行机器人或神经网络，不能证明实机跟踪精度。

## 操作者执行

在 a100 上使用原命令：

```bash
cd /home/pku/GR00T-WholeBodyControl
./start_arm_replay.sh eno2
```

1. 脚本检查数据、编译，再显示原 `deploy.sh` 的部署确认提示。
2. **确认部署后，机器人执行原有约 3 秒的默认姿态初始化。** 此阶段尚未重放。
3. 看到 `Init Done` 后按 `]`，启动 IDLE planner 和 SONIC。
4. 参考时间轴先等待 5 秒，再用 3 秒从实时 planner 双臂参考过渡到录制起点。
5. 按原时间轴播放 93.54 秒，再用 3 秒过渡回实时 IDLE planner 参考；之后持续站立。
6. 按 `O`（大小写均可）走原停控路径，发送阻尼命令。再次重放需重新启动程序。

参考过渡使用五次平滑权重，参考速度包含权重变化项。编码器有 0.9 秒前瞻，
策略可以在参考阶段边界之前开始调整；日志阶段描述的是参考时间轴，不是实际动作起止。
专用输入只处理 `]` 和 `O`，不响应行走或模式切换按键。

其他记录及降采样频率可指定，支持 10–50 Hz：

```bash
./start_arm_replay.sh eno2 --recording /absolute/path/to/recording --hz 25 --check
```

`--motor-kp-scale` / `--motor-kd-scale` 沿用原语义。此入口不能与禁用腿腰的
`--only-arms-output` 混用。

## 两类误差日志

约 50 Hz 采样，每秒输出窗口统计，阶段切换时输出剩余窗口：

- **`[ArmReplayTracking]`**：`motion_reference_at_feedback`，期望运动角减去实测角。
  在反馈接收时间求轨迹参考；正式重放阶段直接对应录制轨迹的连续插值。
  过渡阶段使用当前 planner 参考与录制轨迹的混合值。
- **`[ArmReplayPD]`**：`previous_control_target`，上一周期 SONIC 最终电机目标减去实测角。
  该偏差可用于产生支撑力矩，不应作为动作跟踪误差。

两类日志都只统计双臂，字段为：

- `t` / `phase`：参考时间轴秒数，以及 `blend_in`、`playing`、`blend_out`。
- `frames` / `skipped`：有效 / 排除的反馈帧数。
- `mae_rad` / `rmse_rad`：窗口内 14 个关节的 MAE / RMSE。
- `left_rmse_rad` / `right_rmse_rad`：左右臂各自 RMSE。
- `max_abs_rad` / `max_abs_deg`：最大绝对误差，弧度 / 度。
- `worst` / `sdk`：最大误差关节及硬件索引；`error_rad`、`target_rad`、
  `actual_rad` 为该峰值的带符号误差、对应参考/命令角和实测角。

反馈超过 100 ms、重复接收快照、非有限值会被排除；PD 统计还排除接收时间早于
命令生成时间的反馈。接收时间不是机器人采集时间，两种统计均不补偿通信延迟，
也不构成电机接收命令的确认。无有效数据时显示 `no_valid_samples`。

正常结束或中途停止时，两类日志各输出一次 `summary` 及每个关节的统计。
汇总只包含 `playing`，排除等待和过渡；中止标记 `reason=stopped`。
日志输出到终端，可随标准输出保存，不要求启用原有 CSV 状态日志。
