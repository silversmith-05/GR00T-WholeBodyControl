# NX 本机 Pico + 因时手遥操作

入口为仓库根目录的 `start_inspire_teleop.sh`。默认在当前机器同时启动：

- `deploy` 窗格：SONIC 身体控制，`zmq_manager` 输入、ZMQ 反馈，自动关闭 Dex3 驱动。
- `teleop` 窗格：Pico manager 接收 XRoboToolkit 数据，控制因时手并发布身体遥操作目标。

使用已有 SONIC 模型和 `.venv_teleop`，无需 VLA checkpoint 或 A100；此入口不启动模型服务、相机服务或训练数据记录。需要示范采集时仍使用 `start_wired_real.sh`。

## 启动

```bash
cd /home/unitree/GR00T-WholeBodyControl

# 只打印命令，不导入 SDK、不启动进程
./start_inspire_teleop.sh --dry-run

# 只检查本地依赖和 C++ 二进制，不连接设备
./start_inspire_teleop.sh --check

# 现场启动；无需必填 mode，默认 real
./start_inspire_teleop.sh

# 也可以指定机器人有线网卡
./start_inspire_teleop.sh eth0
```

脚本建立 `sonic_inspire_teleop` tmux 会话，默认选中 `deploy` 窗格。在该窗格完成原 `deploy.sh` 的现场确认；不会自动发送确认或运动启动按键。用 `Ctrl+b` 再按方向键切换窗格，`Ctrl+b d` 离开会话，`tmux attach -t sonic_inspire_teleop` 返回。离开会话不会停止控制，退出前按原遥操作流程停控。

身体启停和模式切换沿用 Pico manager：首次 `A+B+X+Y` 启动，`A+X` 切到全身 POSE；运行中 `A+B+X+Y` 停控。因时手沿用现有逻辑：进入允许的模式后先松开 trigger 和面键建立基线，再按 trigger 闭合、松开放开。左手 X/Y、右手 A/B 调整拇指旋转，单击和长按规则详见 [因时手控制说明](inspire_hand_data_collection.md#左右拇指旋转操作)。

## 双臂输入模式：only-arms-detect

`--only-arms-detect` 沿用原版 **SMPL 双臂跟踪**和因时手控制，只将人的腿、脊柱、颈部及整体朝向输入替换为中立参考。保留肩、肘、腕信息，不再将 A+X 跟踪替换成 VR3PT。第一次 AXBY 仍启动 IDLE 站立；A+X 跟踪期间使用原版 SMPL 编码模式，腿腰由全身策略跟随中立参考并参与平衡，**不是持续运行 IDLE planner，也不是固定实际关节角**。

B+Y 只冻结双臂需要部署端支持新的 `arm_position` 字段。首次使用前，在实际运行 C++ deploy 的机器上更新源码并编译（NX 上需本机编译）：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
```

```bash
./start_inspire_teleop.sh --only-arms-detect --check
./start_inspire_teleop.sh --only-arms-detect --dry-run
./start_inspire_teleop.sh --only-arms-detect
```

操作顺序：

1. 第一次 `A+B+X+Y` 启动 IDLE 站立，与原入口一致。
2. 松开按键，待机器人站稳后按 `A+X` 进入原版 SMPL 跟踪路径；再按 `A+X` 返回 IDLE。肩、肘、腕使用原局部旋转转换、FK、时间插值和手腕分解，不做 VR3PT 的实测手腕重校准。
3. SMPL 跟踪时按 `B+Y`，切到原 `PLANNER_FROZEN_UPPER_BODY` 状态，只下发当前实测双臂共 **14 个关节**作为固定参考；腿腰的位置和速度参考由 IDLE planner 提供。再次 `B+Y` 返回 SMPL 跟踪。长按组合键只切换一次；缺少有效机器人关节反馈时不进入冻结。
4. 左摇杆按下仍是原版可选的 VR3PT 子模式，与 A+X 的 SMPL 跟踪不同：从 IDLE/冻结进入，再按返回来源模式；VR3PT 中 A+X 或 B+Y 均返回 SMPL。VR3PT 仅提供手腕/neck 约束，肘部姿态不保证与 SMPL 相同。进入 VR3PT 时仍需实测反馈重校准。SMPL 中左菜单键暂停/恢复也保留原规则。
5. 两手 trigger、拇指旋转沿用原规则：冻结期间取消新手部动作，恢复跟踪后先松键建立基线。因时手的原采集有效性规则不变。
6. `A+B+X+Y` 仍为停控，优先于模式切换。

SMPL 输入先按原版由全局旋转计算局部旋转，仅保留 SMPL 关节 `13,14,16,17,18,19,20,21`（左右锁骨、肩、肘、腕），其他局部旋转归零，并设定产生直立根朝向的参考旋转。随后运行原版 FK，保持肩肘腕几何一致；没有修改手腕为相对躯干坐标系。站立/冻结期间 planner 输入仍固定为 `IDLE`、零移动、初始朝向、默认速度/高度；SMPL 期间禁止摇杆叠加转向。人的腿腰动作不会成为下发的腿腰参考。

跟踪复用原版 `pose` 协议 v3，C++ 使用原 SMPL encoder mode 2，无需新增模型或修改 VLA token 协议。冻结继续使用 14 维 `arm_position` 字段，C++ 在写入策略参考时跳过腰。普通模式仍发送原来的 17 维 `upper_body_position`。本模式不需要新的 C++ 开关，也不会传入 `--only-arms-output`。本机启动器会检查部署二进制是否支持新字段。两种模式互斥：

| 模式 | 人体输入 | 腿腰电机 |
| --- | --- | --- |
| 默认 | 原全身跟踪及模式切换 | 原全身控制 |
| `--only-arms-detect` | 原版 SMPL 双臂 + 灵巧手，腿腰输入为中立参考 | 保留全身策略控制，关节角允许调整 |
| `--only-arms-output` | 原遥操作输入 | 禁用，吊装使用 |

分机运行时把 `--only-arms-detect` 传给 `--component teleop`，deploy 保持全身输出，并在部署机器上完成上述编译；teleop 单机检查无法验证远端二进制。不要在另一台机器上启用 `--only-arms-output`；单独的 `--component deploy` 不接受 detect 参数。

采集入口对应 `--pico-only-arms-detect`，例如：

```bash
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --pico-only-arms-detect \
  --hand-backend inspire --enable-hand-control \
  --camera-host 192.168.123.164 --record-wrist-cameras \
  --task-prompt "pick up the ball" --dataset-name only_arms_detect_ball
```

先停掉已有控制会话，再启动采集入口。`start_wired_real.sh` 不转发参数，使用它时需把 `--pico-only-arms-detect` 加入脚本内的 Python 参数列表。本模式要求 `--pico-manager` 与 `--deploy-input-type zmq_manager`，不能同时使用 `--deploy-only-arms-output` 或 `--pico-waist-tracking`。

采集格式保持不变：A+X/B+Y 返回跟踪时 `teleop.stream_mode=1`，原采集器记录 `teleop.smpl_pose` / `teleop.smpl_joints`；冻结仍为 `3`，手动进入 VR3PT 才是 `5`。全身 `observation.state` / `action.wbc` 仍为 29 维。`action.hand` / `action.thumb_rotation` 不变。已做离线几何、消息、状态切换和采集兼容性验证；肘部外扩是否消除、SMPL 中立参考下的站立效果仍需实机确认。

## 吊装双臂输出模式：only-arms-output

新增可选 `--only-arms-output`；不传时仍为原全身控制。先在实际运行 C++ deploy 的机器上更新源码并重新编译（NX 上需本机编译，不能复制工作站的 x86 二进制）：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
./start_inspire_teleop.sh --only-arms-output --check
./start_inspire_teleop.sh --only-arms-output --dry-run

# 完成吊装后启动；也可在参数中指定 eth0 等网卡
./start_inspire_teleop.sh --only-arms-output
```

`--check` 和 `--dry-run` 不连接设备。启动器在创建会话前检查二进制是否支持新开关；`deploy.sh` 也会在构建后检查，避免旧程序忽略参数而运行全身控制。运行时 deploy 会打印 `Body motor output: only-arms-output`。分机使用时把参数传给 `--component deploy`；单独的 `--component teleop` 不接受该参数。

| 硬件索引 | 部位 | 双臂模式下的下发 |
| --- | --- | --- |
| 0–11 | 双腿 | `mode=0`，`q/dq/tau/kp/kd=0` |
| 12–14 | 腰 | `mode=0`，`q/dq/tau/kp/kd=0` |
| 15–21 | 左臂，含腕关节 | 保留原控制 |
| 22–28 | 右臂，含腕关节 | 保留原控制 |
| 独立手部连接 | 左右因时手 | 保留原 Pico 控制 |

身体下发路径是 `Pico manager → ZMQ → C++ SONIC 策略（50 Hz）→ MotorCommand 缓冲 → LowCommandWriter（500 Hz）→ DDS rt/lowcmd`。屏蔽在 `motor_output.hpp::pack_motor_commands()` 实施，每条 DDS 消息在计算 CRC 前处理，覆盖初始化插值、等待启动、正常控制和退出阻尼。即使上游仍为腿腰计算目标、设置非零增益，腿腰也只会收到禁用且各字段清零的命令。DDS 仍发布完整包，“不输出”指禁用对应电机，不是停止发整包。关节状态订阅保持原样。

因时手走 `Pico → InspireHandController → SDK/Modbus TCP`，不经过上述身体输出；C++ 的 Dex3 仍禁用。`--read-only-hands` 可额外关闭因时手写入。

此模式按机器人吊装使用：腿腰无位置保持、无主动支撑、无退出阻尼。双臂仍会执行原有约 3 秒初始化插值，启停和 POSE 切换按原按键流程进行。SONIC 仍读取全身状态并运行原全身策略；本改动没有把双臂改为独立 IK，吊装姿态下的跟随质量尚需实机验证。

### 记录数据

`start_inspire_teleop.sh` 本身仍不启动记录。需要采集时，在原 `launch_data_collection.py` 命令中增加 `--deploy-only-arms-output`，例如（相机地址按现场设置）：

```bash
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --deploy-only-arms-output \
  --hand-backend inspire --enable-hand-control \
  --camera-host 192.168.123.164 --record-wrist-cameras \
  --task-prompt "pick up the ball" --dataset-name only_arms_output_ball
```

采集启动器会自行启动 deploy 和 teleop，应先按原停控流程退出此前的遥操作会话，避免重复控制进程。`start_wired_real.sh` 当前不转发命令行参数；如使用该脚本，请把 `--deploy-only-arms-output` 加入脚本内的 Python 参数列表。

采集格式不变：`observation.state` / `action.wbc` 仍是 29 维；腿腰实测状态仍记录，但 `action.wbc` 的腿腰部分是策略输出，**没有下发执行**。新建单独数据集，并在双臂训练时按关节名称选择左右臂各 7 维，不能把腿腰动作列当成已执行示范。数据集的 RobotModel 顺序与 DDS 硬件索引不应混用，不能直接假定数据集切片也是 `[15:29]`。因时手的 `action.hand` 和 `action.thumb_rotation` 保留原语义。本开关不自动裁剪数据维度，也不添加采集格式字段。

## 默认参数与覆盖

| 项目 | 默认值 |
| --- | --- |
| 模式、输入源 | 真机 `real`、Pico `xrt` |
| 左手、右手 | `192.168.123.211:6000`、`192.168.123.210:6000` |
| 身体指令、反馈 | 本机 `5556`、`5557` |
| 手部控制 | 启用；`--read-only-hands` 可改成只读手部反馈 |
| 五指闭合 | `250 250 250 250 300`，小指到拇指弯曲 |
| 拇指单击步长、长按速率 | 左右均 `10`、`50` 标度单位/秒 |
| 身体模型、planner | 沿用 `deploy.sh` 默认值 |

```bash
./start_inspire_teleop.sh real \
  --inspire-left-ip 192.168.123.211 \
  --inspire-right-ip 192.168.123.210 \
  --inspire-close-angles 250 250 250 250 300
```

身体参数支持 `--cp`、`--obs-config`、`--planner`、`--motion-data`、`--motor-kp-scale`、`--motor-kd-scale`；路径按原 `deploy.sh` 的目录规则解析。完整参数见 `--help`。`--read-only-hands` 只禁用手部写入，不禁用身体控制。

如以后把 Pico 接收移到工作站，可分别使用 `--component deploy --teleop-host <工作站IP>` 和 `--component teleop --state-host <NX-IP>`。当前 NX 同机部署无需这些参数。

## 因时 SDK 环境

控制器要求 `.venv_teleop` 中安装 `inspire-rh56e2==0.3.0`，其依赖是 `pymodbus==3.11.4`。本机已有 SDK 源码时，从源码安装，例如：

```bash
uv pip install --python .venv_teleop/bin/python \
  /home/unitree/inspire_hand_ws/workspace/Inspire_RH56DFTP_sdk
./start_inspire_teleop.sh --check
```

`--check` 仅验证本地依赖和二进制开关支持，不验证 Pico 数据、手部网络连接或真机动作。SDK 安装和离线测试不发送硬件指令。
