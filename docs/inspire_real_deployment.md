# G1 + Inspire：GR00T N1.7 有线真机部署

入口：`start_inspire_inference_wired.sh`。保留原来的 Dex3 和 Pico 入口；本入口使用 Inspire 二值动作接口。模型已完成 A100 重载、8 条轨迹离线评估和轻量客户端通信验证。现场运动、完整实时观测联调及 SONIC 并行运行性能仍需现场验收。

## 1. 拓扑与模型约定

所有控制进程运行在 A100 工作站。Python 环境由启动器选择，无需在 Python 3.10 控制环境安装 GR00T。

| 组件 | 默认配置 |
| --- | --- |
| 模型服务 | `/home/pku/workspace/Isaac-GR00T/.venv/bin/python`，Python 3.12，`127.0.0.1:5550` |
| checkpoint | `/home/pku/workspace/Isaac-GR00T/models/inspire/checkpoint-10000` |
| 推理客户端、手部驱动 | 本仓库 `.venv_teleop/bin/python`，Python 3.10 |
| SONIC 输入、反馈 | 本机 `5556`、`5557`，键盘 `5580` |
| 三路相机 | `192.168.123.164:5555`，与采集相同的 ZMQ 解码器 |
| 左手、右手 | `192.168.123.211:6000`、`192.168.123.210:6000` |
| SONIC 文件 | `gear_sonic_deploy/policy/release/model_encoder.onnx`、`model_decoder.onnx`、`observation_config.yaml` |

输入状态顺序固定为 `left_leg[6] + right_leg[6] + waist[3] + left_arm[7] + right_arm[7] + projected_gravity[3] + left_hand[6] + right_hand[6]`，共 44 维。身体转换复用采集的 RobotModel，手部直接使用 `angle_act` 的设备标度。三路 RGB 图像为 `ego_view`、`left_wrist`、`right_wrist`，每路 `(480,640,3)`，语言字段为 `annotation.human.task_description`。

模型反归一化后输出 `motion_token (1,40,64)` 和 `hand (1,40,2)`；手部顺序为左、右，`>=0.5` 闭合，否则张开。动作是 absolute/non-EEF；SMPL 和独立的拇指旋转动作不进入推理接口。模型处理器使用 checkpoint 的训练统计。

身体使用 ZMQ v4 `pose` 消息，只包含 `token_state` 和 `frame_index`，不发送 Dex3 字段。启动 SONIC 时显式传入 `--disable-dex3-hands`；客户端还会核对广播的 Dex3 开关、release 文件路径和 50 Hz 频率。

`gear_sonic_deploy/deploy.sh` 补充了 `--zmq-port` 和 `--zmq-out-port` 校验及透传，原默认值不变。模型服务和 SONIC 都使用启动器选定的 GPU。

## 2. 先做离线检查

```bash
cd /home/pku/GR00T-WholeBodyControl
./start_inspire_inference_wired.sh check

# 加上 GPU 重载和录制观测推理
./start_inspire_inference_wired.sh check --gpu-check \
  --output-dir outputs/inspire_check_gpu_01
```

`check` 不连接机器人。它检查 SDK、模型分片及索引、模态配置、训练统计、SONIC 文件，并导出初始化 token。默认来源为训练集 episode 0、frame 0，可用 `--initial-episode`、`--initial-frame` 覆盖；`initial_token.json` 记录原始 Parquet 路径及 SHA256。

自动选择本机 A100 的 GPU UUID；其他机器用 `--cuda-visible-devices` 指定 GPU 编号或 UUID。每次使用新输出目录；默认自动创建带时间戳的目录，不向已有日志目录追加实验。

## 3. 相机、手部只读检查

```bash
cd /home/pku/GR00T-WholeBodyControl
.venv_teleop/bin/python gear_sonic/scripts/probe_inspire_sensors.py \
  --mode observe --duration 12 \
  --output-dir outputs/inspire_sensor_check_01
```

该入口只读取反馈，不启动 SONIC、不建立身体动作发布器、不写入手部寄存器。`sensor_probe.json` 分别报告相机名称、形状、新鲜度、身体流、配置和手部反馈。若 SONIC 尚未发布 `g1_debug`，整体 `passed=false`、原因为身体反馈缺失；这不代表相机离线。

2026-09-12 复查时三路相机均有数据，最近接收距今约 55 ms，两手反馈有效；左右拇指实测均为 `0`。没有身体流和 `robot_config`，尚未进行完整实时预测。检查过程未发送动作。报告在 `outputs/inspire_deploy_validation_10000/sensors_retry/sensor_probe.json`。

## 4. 实时只读预测

```bash
./start_inspire_inference_wired.sh observe \
  --prompt 'pick up the ball' \
  --duration 60 \
  --output-dir outputs/inspire_observe_01
```

启动器创建 tmux 的 `server`、`client`、`keyboard` 窗口，终端打印 attach 命令。`observe` 不启动或停止 SONIC，不创建身体动作发布器，手部驱动禁止写入。已有 SONIC 必须提供匹配 release 配置、关闭 Dex3 的新鲜 `g1_debug`。没有身体流时等待并记录缺失原因，不以零状态替代。

当前 C++ 只在现场启动控制后发布 `g1_debug`，所以首次上机不能仅靠 `observe` 自动获得身体反馈。需要现场人员建立反馈来源；不要为了让检查通过而无准备地启动身体控制，也不要同时启动 Pico 或另一个占用手部锁的客户端。

观测新鲜度使用接收端单调时钟，并要求每路相机的源时间戳和身体序号持续前进。不会把同一缓存帧的重复读取视为新帧；机器人和工作站时钟没有被假定为同步。日志中的源端图像延迟仅供诊断，跨机时钟误差会影响它。

## 5. 现场启用真机

以下命令由现场操作人员执行。`real` 会启动硬件控制组件，但不会自动发送键盘启动、初始化或模型启用命令。

```bash
./start_inspire_inference_wired.sh real \
  --model-path /home/pku/workspace/Isaac-GR00T/models/inspire/checkpoint-10000 \
  --prompt 'pick up the ball' \
  --output-dir outputs/inspire_real_01
```

启动顺序：离线检查 → 模型服务 → 录制输入预热及 30 次轻量 RPC 性能检查 → 键盘和客户端 → SONIC。性能检查的 95 分位超过更新预算时，不创建硬件控制窗口。SONIC 的 `deploy.sh` 仍保留原有现场确认。

在 tmux 的 `keyboard` 窗口输入下列字符并按回车：

| 输入 | 行为 |
| --- | --- |
| `k`（首次） | 核对 SONIC 广播配置后启动 planner，开始身体控制；等待身体反馈 |
| `i` | 从新鲜 C++ 当前 token 向训练初始 token 默认过渡 1 秒；结束后为 PAUSED |
| `p` | 初始化完成后启用模型；再次输入则暂停模型发布 |
| `t pick up the ball` | 修改提示词，清空旧动作和旧请求代次，真实模式需要再次 `p` |
| `s` / `f` | 记录本次任务成功 / 失败，不自动停止控制 |
| `o`、`q` 或控制已启动时再次 `k` | 锁定停控，走 C++ stop 路径并退出客户端，需要重启恢复 |

没有有效当前 token 时，`i` 被拒绝，不跳到训练姿态。首次 `k` 后有最多 10 秒等待 C++ 首次发布身体数据的初始化期限；正常运行后适用下面的反馈超时规则。

**普通暂停不保证机器人静止。** PAUSED 仅停止模型动作发布，SONIC 仍在运行。停控走现有 C++ 退出和阻尼关闭路径，不是可自动恢复的暂停。客户端退出或收到终止信号时，若已启动 C++，也会发送 stop。不能把杀掉推理发布进程视为可靠的独立急停。

### 拇指和手部

启动时只读取。现场手动对齐左右拇指后，每次 `p` 启用都会读取当时的新鲜第六路角度，并作为本轮保持目标。初始化 `i` 不写手；不会采用独立测试工具的默认 `339`。启动及每秒状态日志显示实测值、保持值及反馈有效性，重新启用时重新读取。

| 目标 | 前五路角度 |
| --- | --- |
| 张开 | `850,850,850,850,1000` |
| 闭合 | `250,250,250,250,300` |

速度 200、力阈值 500，复用 `inspire_hand_controller.py` 的统一预设。VLA 使用 `begin_policy_session()` / `update_policy()`；首次闭合可直接执行，无需 Pico 的先松后按。相同目标不重复写入，仅保留最新目标。身体与手部共用动作序号，这表示目标派发对齐，不表示网络写入或电机到位时间相同。

等待写入确认属于正常状态。通信失败、写入不明、反馈过期或会话失效都会撤销待写目标，停止控制，不自动重放；断线后必须重新现场启用。`write_current` 表示最新写入得到确认，并不等于手已经物理到位。

### 时序和故障

模型输出 40 步（0.8 秒），50 Hz 主循环，默认每 16 步（320 ms）发起一次异步请求。依据观测接收时刻扣除已过去的动作步；过期结果拒绝使用，动作耗尽不无限重复最后一帧。暂停、提示词切换和初始化都会使此前请求失效，包括稍后才返回的结果。

以下情况锁定停控并记录原因：身体或任一路相机超过 250 ms 无新数据、手部反馈超过 1 秒、手部断连或反馈无效、非有限预测、token 越界、预测超过窗口、动作耗尽、手部写入故障。连续 5 次预测延迟超过配置的更新预算也会停控，要求先检查性能。

手部反馈超过 500 ms 但未超过 1 秒时继续运行，在 `client.log` 中记录 `hand_feedback_timeout`，包含左右手和反馈年龄；每次超时过程只告警一次，恢复后记录 `hand_feedback_recovered`。反馈校验在读取手部快照之后取当前时间，避免后台更新造成负年龄误报。

同一有效策略会话内，新手部目标替换正在写入的旧目标时，取消旧目标剩余寄存器写入，随后执行最新目标，不增加故障计数、不退出。`client.log` 记录 `hand_write_cancelled` 和 `superseded=True`。输入过期、会话失效、断连及写入结果不确定仍保留故障处理；故障计数变化时额外记录 `hand_fault` 手部快照。

`--execution-horizon` 可修改更新间隔，不改变模型 40 步预测长度。首版的实际 SONIC 并行负载、50 Hz 循环和输入分布须通过只读联调检查；录制输入的 GPU 性能不等同于完整控制链路性能。

## 6. 覆盖参数和日志

启动器支持 `--gr00t-repo`、`--model-path`、`--dataset-root`、`--output-dir`、`--prompt`、`--cuda-visible-devices`、`--deploy-interface`、`--initial-pose-blend-duration` 以及各组件的 `--*-host` / `--*-port`。Inspire 使用 `--inspire-left-ip`、`--inspire-right-ip`、`--inspire-port`。本拓扑要求模型、身体通信及键盘地址为本机回环地址；相机及手部端点可覆盖。默认自动识别唯一的 `192.168.123.x` 有线网卡，存在歧义时明确报错。

每次运行目录包含：

- `deployment_manifest.json`：实际命令、参数、GPU、SONIC 文件 SHA256。
- `configs/`：模型配置、模态配置、训练归一化统计、SONIC observation 配置及手部预设。
- `initial_token.json`、`checkpoint_report.json`、`check.log`：初始化来源和离线检查。
- `warmup/rpc_report.json`、`warmup.log`：真实通信输出形状、预热及 30 次延迟。
- `server.log`、`deploy.log`（real）、`client.log`、`keyboard.log`：各进程输出。
- `client/events.jsonl`：模式、请求延迟、状态范围、动作序号、原始手部预测、二值目标、手部写入确认和故障。
- `client/session_report.json`：结束时汇总延迟、循环频率、故障和操作人员记录的任务结果。

诊断文件不写入采集数据集。退出客户端不会自动关闭模型服务；确认硬件已按停控流程退出后，按终端打印的会话名清理此次 tmux 会话即可。预热失败时只会留下模型服务窗口供查看日志。

## 7. 本次验证及复验

2026-09-12，当前 checkpoint 的全部 8 条评估轨迹、8,226 帧完成离线评估：帧加权 MSE `0.0020085975`，MAE `0.0278362132`；各轨迹指标有限并有图像。报告：`outputs/inspire_deploy_validation_10000/eval/eval_report.json`。

同一 A100 上，Python 3.10 轻量客户端对 Python 3.12 官方模型服务完成 30 次录制观测请求，输出 `(1,40,64)`、`(1,40,2)` 均有限。预热后 p50 `72.31 ms`、p95 `194.36 ms`。报告：`outputs/inspire_deploy_validation_10000/rpc/rpc_report.json`。未同时运行 SONIC，未发送硬件动作。

```bash
# CPU 合约、调度、故障和真实 SDK + 模拟传输回归
.venv_teleop/bin/python -m unittest \
  gear_sonic.tests.test_inspire_inference \
  gear_sonic.tests.test_inspire_hand_controller -q
```

测试覆盖：44 维状态顺序、三路图像、官方 RPC 请求封装、v4 token 协议、二值阈值、首次闭合、目标去重、拇指保持、断线和未确认写入、过期帧、动作耗尽、旧请求失效、初始化过渡、只读模式不输出、停控锁定及性能超预算。

尚未完成的现场项：真实身体反馈加入后的完整只读预测、与 SONIC 同时运行的循环和延迟验收、初始姿态、手部开闭及闭环抓球试验。真实成功率应由现场成功、失败和中止记录统计；离线误差不能代替任务成功率。
