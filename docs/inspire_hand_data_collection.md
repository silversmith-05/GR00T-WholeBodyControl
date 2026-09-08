# Pico trigger / 拇指旋转 → Inspire RH56E2-T1 → SONIC 数据采集

本接入在工作站 Python 遥操作进程中管理左右手；NX 继续运行相机服务器。
数据记录程序只订阅手部反馈，不创建 SDK 客户端。控制与采集验证使用模拟传输、生成图像和离线文件；后续已通过独立入口完成两手的实机只读连通性检查。未运行实机遥操作或发送实机运动目标，未修改设备或网络配置。

## 环境与设备

复用现有 `.venv_teleop`，其中已确认可导入 `inspire-rh56e2==0.3.0`，基础依赖为 `pymodbus==3.11.4`。不需要重建环境或运行安装脚本。SDK 在启动 Inspire 后端时才导入；Dex3 后端不要求安装该 SDK。

| 参数 | 左手 | 右手 |
| --- | --- | --- |
| 地址 | `192.168.123.211:6000` | `192.168.123.210:6000` |
| 型号 | RH56E2-T1 | RH56E2-T1 |
| 控制输入 | 左 trigger | 右 trigger |
| Modbus TCP unit_id | 255 | 255 |

`unit_id` 是 Modbus TCP Unit Identifier，不是内部 HAND_ID。固定使用 `byte_layout="packed_little"`、`tactile_order="big"`，没有串口、UDP 或自行编码的寄存器协议。

| 二值目标 | 用户定义 | SDK 预设 | 六路角度标度 |
| --- | --- | --- | --- |
| 0 | 放开 | release（SDK 自定义预设） | `1000 1000 1000 1000 1000 θ` |
| 1 | 五指闭合 | close（SDK 自定义预设） | `250 250 250 250 300 θ` |

Pico 模式中 `θ` 是该手独立保持的拇指旋转目标，来自新鲜实测基线和后续短按/长按增减，不再被 trigger 重置到 339。下方独立键盘工具也使用五指闭合，但仍使用固定旋转 339；不要在调好 Pico 旋转后用它测试开合并期待旋转保持。

六路顺序：小指、无名指、中指、食指、拇指弯曲、拇指旋转。速度、力阈值均为六路 `200`。角度字段是设备标度，不是弧度或角度制。保留此前修正过的 `0=放开、1=闭合` 含义；预设修订 3 将原两指捏取改为五指更深收拢。SDK 0.3.0 自带预设未修改；应用使用其自定义预设和 `Command` 接口。普通开合/短按依次写力阈值、速度、角度。长按在本连接明确确认过力阈值/速度为 200 后，只写变化的角度；每次写入均经过 SDK 保护检查。

## 约 4 厘米硬球的五指目标

旧目标前三路为 1000，实际只让食指和拇指弯曲；食指 592、拇指 720 留出的空间也较大。新默认目标将四指设为 250、拇指弯曲设为 300，五指在同一条开合指令中收拢，旋转保持各手单独选定的位置。trigger 仍是二值开合，不做随按压程度的插值，也没有新增需要学习的连续五路弯曲动作。

这些数值是针对用户所述约 4 厘米硬球的**待现场标定起始值**，不是由球直径计算出的精确姿态。可以用 `--inspire-close-angles 250 250 250 250 300` 指定五路终点；顺序同上，整数范围 0..999，值越小弯曲越深。两个手使用同一套五路闭合标度，旋转和旋转速度各自独立。启动器会把相同闭合参数传给控制和记录程序。修改闭合终点时使用新数据集名称，避免一个二值动作对应两套不同闭合姿态。

现场先在手张开的状态下用旋转按键调整拇指，使其朝向四指围成的抓取空间，再主动按 trigger 收拢。若仍有较大空隙，先区分“实测尚未接近目标”与“到达目标后仍留空隙”，再微调闭合标度；不要把目标已写入当作已经到位，也不要用盲目提高力阈值来替代姿态调整。本次没有发送实机动作，尚未确认抓球成功。

## 先单独测试灵巧手（不启动全身控制）

新增 `gear_sonic/scripts/test_inspire_hand.py`，直接复用相同 SDK、预设校验和设备锁，不启动 Pico、相机、数据采集或 tmux。先退出占用该手的遥操作/其他手驱动，再从仓库根目录运行。

只连接左右手，各读取一次 `angle_act` 后退出，不发送运动目标：

```bash
.venv_teleop/bin/python gear_sonic/scripts/test_inspire_hand.py --hand both
```

手动控制右手（左手不建立连接）：

```bash
.venv_teleop/bin/python gear_sonic/scripts/test_inspire_hand.py --hand right --enable-control
```

手动控制左手：

```bash
.venv_teleop/bin/python gear_sonic/scripts/test_inspire_hand.py --hand left --enable-control
```

启动后没有开合动作。输入以下命令并回车：

- `0`：放开，发送用户定义的 release 预设。
- `1`：闭合，发送用户定义的 close 预设。
- `s`：读取实际角度，并在已有目标时显示实测与目标的差值。
- `q`：退出并断开连接，不发退出开合动作。

每次手动目标发送之前先读取当前角度；读失败便退出。成功写入后显示“寄存器写入已确认”，并读取实际角度；手尚未到位时可稍后输入 `s` 再看。重复相同目标不重复写寄存器。写入失败/超时/中断会退出且不自动重试或重连。此工具的键盘命令是主动的一次性操作，不模拟 Pico trigger，也不需要 trigger 的首次松开基线。

默认只读。`--enable-control` 必须明确选择 `--hand left` 或 `--hand right`，不能同时驱动两手。可用 `--host`、`--port` 覆盖单手端点；默认地址与上表一致。

独立入口测试：

```bash
.venv_teleop/bin/python -m unittest gear_sonic.tests.test_inspire_hand_standalone gear_sonic.tests.test_inspire_hand_controller -v
```

独立工具的 9 项交互/只读/异常/自定义五指目标测试使用假 Modbus transport；完整结果见下方。键盘工具也接受 `--inspire-close-angles` 五个值，发送前会打印实际采用的目标。

## 启动

从仓库根目录执行（以下命令会启动实机全身控制流程，应留到现场验收时运行）：

```bash
python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "pick up the ball" \
  --hand-backend inspire \
  --enable-hand-control \
  --inspire-left-ip 192.168.123.211 \
  --inspire-right-ip 192.168.123.210 \
  --inspire-port 6000 \
  --inspire-left-thumb-step 10 \
  --inspire-right-thumb-step 10 \
  --inspire-left-thumb-hold-rate 50 \
  --inspire-right-thumb-hold-rate 50 \
  --inspire-close-angles 250 250 250 250 300 \
  --inspire-left-thumb-min 0 \
  --inspire-left-thumb-max 1000 \
  --inspire-right-thumb-min 0 \
  --inspire-right-thumb-max 1000 \
  --dataset-name inspire_ball_v4
```

`--hand-backend inspire` 自动向 C++ 传递 `--disable-dex3-hands`，并向遥操作和记录进程传递一致的后端选择。C++ 默认仍启用 Dex3；Inspire 模式完全跳过 Dex3 初始化、读状态、初始开合、周期发命令及键盘手部目标应用。全身控制算法保留。

本工作站已用下面的命令重新编译 C++，它只编译，不运行部署：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
```

记录端还要求收到 `robot_config.dex3_hands_enabled=false`，防止误用旧二进制或漏传开关。手动拆分启动时，必须同时使用：C++ `--disable-dex3-hands`、Pico `--manager --hand-backend inspire --enable-hand-control`、记录端 `--hand-backend inspire`。自定义 `--inspire-close-angles` 时，Pico 和记录端必须传入同样的五个值，否则反馈会被拒收。

省略 `--enable-hand-control` 时，Inspire 只连接并读取角度，不发运动目标；这种状态不能作为正常手部示范。原命令或显式 `--hand-backend dex3` 保留旧后端，不受 `--enable-hand-control` 管理。Inspire 不支持 legacy 单独 pose 启动入口，也不允许通过 `launch_data_collection.py --sim` 连接真实手；离线验证使用下方测试。

请为本版使用新的 `--dataset-name`。记录端会检查后端、FPS、feature、相机配置、动作语义和本数据集的五指闭合参数，拒绝混用 Dex3、旧两指预设、缺少旋转动作或不同闭合参数的数据。

## 左右拇指旋转操作

| 手柄 | 单击 | 结果 |
| --- | --- | --- |
| 左 | X | 左拇指旋转目标减去左步长 |
| 左 | Y | 左拇指旋转目标加上左步长 |
| 右 | A | 右拇指旋转目标减去右步长 |
| 右 | B | 右拇指旋转目标加上右步长 |

默认步长均为 **10 个设备标度**。增减的是设备数值，不定义未经实机核验的内旋/外旋方向。通过 `--inspire-left-thumb-step` 和 `--inspire-right-thumb-step` 独立调小或调大，例如左侧 5、右侧 20；参数在启动时设置，不占用额外按键。值必须为 1..1000 的整数。`min/max` 是每手独立的软件限幅，默认 0..1000 只是协议范围，不代表已经标定的适用抓取范围。设定范围外的实测位置会使该手无法准备控制并报告原因，不自动跳到边界。

**短按**（不到 0.6 秒）在松开时走一次上述步长。**长按**同一键达到 0.6 秒后，按该手的速度连续生成小步目标，松开不会额外补一次短按。默认 `--inspire-left-thumb-hold-rate 50`、`--inspire-right-thumb-hold-rate 50`，单位是设备标度/秒，不是角度/秒；可独立设为 1..50，例如 10 更慢。修改后需重启遥操作进程；旧命令中显式指定的 `20` 应改为 `50` 或省略对应参数。默认每步约 5 标度，最多每秒 10 步，和短按步长独立。慢通信只会使旋转更慢，不积攒未来目标或恢复后追赶。长按目标向当前方向最多领先实测旋转 10 标度，实测滞后或过期时等待，避免受阻后积攒大幅目标。

手势从第一次面键按下持续到四个面键全部松开；期间出现第二个面键或左 grip 就停止长按，并取消待发生的短按，必须全部松开后重新操作。原 A+X、B+Y 模式切换、A+B / X+Y 行走模式调整、A+B+X+Y 启停、左 grip+A/B 录制/丢弃仍执行原逻辑。**如果先独自按住一个键超过长按门槛、已经开始旋转，再补按组合键，已经发出的旋转无法撤回**；组合键的各键应在长按门槛前按下。

进入允许控制的模式后，先松开该手 trigger，并让 A/B/X/Y 和左 grip 全部松开，然后再操作。启动时已按住的键不会进入长按或在松开时触发短按。模式切换、暂停、输入缺失/过期、断线、退出会清除长按状态及未完成单击；快速重连后继续按着旧键也不重放，须全部松开再重新按下。

启动、模式切换或重连后，从新的 `angle_act[5]` 选取旋转基线，这一步不写寄存器。单击可以先于第一次主动抓取：此时只给第六路旋转目标，前五路使用 SDK 的 `-1` 不动作标记（本机 RH56E2 用户手册 §2.6.11 和 SDK 模拟器均有该定义）。有了明确开合目标后，后续旋转与 trigger 统一组合为“当前开合前五路 + 该手旋转目标”，因此旋转不会丢失开合意图，开合也不会复位旋转。

持续保持不变的目标、到达软件边界后继续向外操作均不重复写入。长按只有产生新小步时才写角度。放开键会停止生成后续目标；已经发到设备的小步可能仍在完成，不发送反向补偿或自动复位。

## 操作与保护逻辑

1. 启动、连接、重连都只读取角度，不自动放开或闭合。
2. 仅 `POSE` 和 `PLANNER_VR_3PT` 模式接受手部输入。`OFF`、`POSE_PAUSE`、普通 `PLANNER` 和 `PLANNER_FROZEN_UPPER_BODY` 都取消待执行命令，保持现有模式含义。
3. 每次启动、模式切换、输入失效或重连后，每只手先获得新的有效角度反馈，再观察到一次有效的松开 trigger（`<=0.5`）。这一步只建立基线，不发放开目标。随后新按下（`>0.5`）才发闭合；再松开才发放开。
4. 因此，启动时已按住 trigger 不会运动。现场开始录制前，应在受控条件下分别完成两手的上述主动操作，使两侧目标都有明确来源，再停在所需起始目标上。
5. 持续保持目标不重复写寄存器。每手只有一个最新待执行目标，过时目标被替换，不积累队列。
6. 每只手拥有独立连接和工作线程，同一只手的读写、启用写入及关闭均串行。网络等待不占用 Pico/全身主循环；另一只手不受该连接等待影响。
7. 默认输入有效期 250 ms、角度有效期 500 ms、SDK 单次通信超时 200 ms、角度轮询 20 Hz、断线后只读重连间隔 1 s。输入设备时间戳不前进时，不会用主循环时间刷新输入有效期。
8. 暂停、过期、断线、关闭都会清空命令并撤销准备状态。写入超时记录 `unconfirmed`，不重试、不重放；下一次控制仍需新的松开/按下。已经发出的寄存器写入无法撤回，取消检查会阻止后续尚未发送的写入。
9. 每个 IP/端口有一个 `/tmp/sonic-inspire-<IP>-<port>.lock` 进程锁，重复启动本驱动会报错。锁文件保留，锁随进程关闭释放。外部 SDK 调试程序不遵守此锁，因此不要在驱动运行时另开其他硬件客户端。
10. Ctrl+C、SIGTERM、SIGHUP（例如关闭 tmux）及原有停止路径会取消双手任务、等待正在执行的有界通信结束，再断开连接。SDK `close()` 只断开连接，代码不调用 `close_hand()` 进行退出动作。

已有 A+X、B+Y、左摇杆按键切换模式，A+B+X+Y 启停，以及 **A+左 grip 录制切换、B+左 grip 丢弃录制** 的逻辑保留。

## 反馈与训练数据

手部反馈以 `inspire_hand ` 前缀加 UTF-8 JSON，通过原 Pico PUB 端口 5556 发布，最高 50 Hz。当前协议为 `schema_version=4`，带型号、五指目标、旋转/长按配置、发布时刻与故障累计计数。记录端只订阅该消息，拒绝版本 1/2/3 的旧反馈。更新后需重新启动 Python 遥操作和记录进程，避免继续运行旧代码。

Inspire 数据集：

| 字段 | 维度 | 含义 |
| --- | --- | --- |
| `action.hand` | 2 | `[left,right]` 二值目标；每帧保留，即使没有新写入 |
| `action.thumb_rotation` | 2 | `[left,right]` 拇指旋转目标，0..1000 设备标度；每帧保留 |
| `hand.thumb_target_valid` | 2 | 旋转目标有效性，须与二值目标、反馈及写入有效性一起使用 |
| `hand.thumb_step/thumb_min/thumb_max` | 各 2 | 该帧左右步长和软件限幅，允许两侧分别配置 |
| `hand.thumb_hold_rate/thumb_hold_active` | 各 2 | 左右长按目标速率（标度/秒）及长按是否生效 |
| `hand.thumb_hold_cancel_count` | 2 | 长按在角度发送前被取消的累计次数，已确认的上一角度目标仍有效 |
| `observation.state` / `action.wbc` | 各 29 | 仅 G1 全身关节；顺序由 RobotModel 派生 |
| `hand.trigger` | 2 | 左右模拟 trigger 原值 |
| `hand.angle_act` | 12 | 左六路、右六路的独立 `angle_act` 实测标度 |
| `hand.angle_target` | 12 | 左六路、右六路组合目标；不是实测反馈，未知弯曲目标为 -1 |
| `hand.close_angles` | 10 | 左五路、右五路的本数据集固定闭合参数；不含旋转 |
| `hand.connected/input_valid/armed/target_valid/angle_valid` | 各 2 | 分别表示连接、输入、准备状态、目标来源、反馈有效性 |
| `hand.at_target/at_target_valid` | 各 2 | 六路实测与目标差值均不超过 20 标度的判断及其有效性 |
| `hand.command_id/write_id/write_target` | 各 2 | 当前目标编号、最近写入编号及目标，避免异步应答被归给新目标 |
| `hand.write_thumb_rotation` | 2 | 最近写入对应的旋转目标；不等于实测角度 |
| `hand.write_status/write_time/write_current` | 各 2 | 最近写入结果、Unix 时刻、当前目标是否已获得写入确认 |
| `hand.fault_count` | 2 | 故障累计计数，防止短暂故障因消息丢失被漏掉 |
| `hand.input_time/input_monotonic` | 各 2 | 输入采集时间（Unix / 工作站 monotonic） |
| `hand.angle_time/angle_monotonic/angle_elapsed_ms` | 各 2 | 反馈读取时间及通信耗时 |
| `hand.frame_time/frame_monotonic` | 各 1 | 本记录帧获取手部快照时刻 |
| `hand.published_time/published_monotonic/publication_valid` | 各 1 | 控制程序发布时刻及消息有效性 |
| `hand.training_valid/episode_fault` | 各 1 | 本帧训练掩码、本帧是否发生需丢弃 episode 的条件 |
| `observation.capture_time/capture_time_valid` | 各 4 | 全身数据在工作站的接收时刻、ego/左腕/右腕相机源时间戳及有效性 |

`action.hand=-1` 或 `action.thumb_rotation=-1` 表示尚无明确目标，必须结合有效性排除。仅初始化旋转或只使用旋转按键，不会凭空生成二值开合动作，仍需在录制前建立左右两手的明确开合目标。失效后可保留上次角度用于诊断，但 `angle_valid=false`，不能当成新反馈。`at_target` 比较实测与动态组合后的六路目标，不拿固定 339 比较；它不等同于设备运动完成保证。写入确认仅表示寄存器应答成功。只读 `angle_act`，从不依赖完整 `read_state()`，因此绕开已知电流寄存器 1594 范围错误。

手部快照与本次采集取得的图像、全身数据放在同一 LeRobot 帧中，采用最新有效样本保持。LeRobot `timestamp` 仍为原有的帧号/FPS；源采集时刻另行保存。这不是硬件同步采样。相机源时间来自 NX，若需要按绝对采集时刻做更严格的离线对齐，应核验 NX 与工作站的时钟偏差；本次没有调整时钟或设备配置。未启用腕相机时，对应时间戳为 -1 且无效，不伪造采集时间。

`meta/modality.json` 的 `action.hand` 映射为 `{"start":0,"end":2,"original_key":"action.hand"}`，另有 `action.thumb_rotation` 映射为 `{"start":0,"end":2,"original_key":"action.thumb_rotation"}`。两者均是训练动作：只训练二值开合无法重现录制中的独立旋转。没有 `left_hand_joints/right_hand_joints` 动作或 Dex3 手部 state 切片。六路实测角度、trigger 和有效性仍是辅助数据，默认不作为训练输入。

`meta/info.json.script_config.hand` 记录型号、顺序、实际选用的闭合模板、旋转/长按规则、状态码和有效性；当前 `preset_revision=3`、`schema_version=4`。每帧另外保存实际配置，记录端不连接硬件。一个数据集的五指终点固定；不同终点不能续录到同一个数据集。`TypedLeRobotDataset` 支持本版自定义闭合模板，拒绝旧版两指/缺字段数据；本次不自动修改历史数据。

长按每个小步只在收到寄存器确认后推进 `action.thumb_rotation`；在途候选值记录于 `hand.write_thumb_rotation`，在途帧 `hand.training_valid=false`。这仍然是目标而非实测，实测只来自 `hand.angle_act`。普通开合/短按继续记录当前待执行目标。写入超时保留候选目标和未确认结果、使示范失效，不自动重试。长按在角度发送之前取消（可能只确认了力/速度为 200）时，仅增加 `thumb_hold_cancel_count`，保留上一角度写入结果；不会把正常松键误记为运动失败。

写入状态码：`0=none`、`1=pending`、`2=confirmed`、`3=unconfirmed`、`4=failed`、`5=cancelled（可能部分写入）`。新的目标尚未确认时，`hand.training_valid=false`，但正常排队/在途写入本身不会丢弃整段录制。

录制期间发生输入失效、暂停、未知目标、断线、角度过期、反馈格式错误、写入失败/未确认/部分取消或故障计数变化，会将整段 episode 标记为 discarded。帧和辅助数据仍落盘，`meta/info.json.discarded_episode_indices` 记录排除项；后续恢复不会取消标记。仓库的 `TypedLeRobotDataset` 加载 Inspire 数据时会排除这些 episode，并仅用保留 episode 的统计量。其他训练数据读取器也必须排除该列表，并使用 `hand.training_valid` 对尚未确认的帧/动作 chunk 做掩码，不能直接训练所有 parquet 行。

## 离线验证

以下测试不会连接硬件、启动 Pico、调用启动脚本 main 或创建 tmux 会话：

```bash
.venv_teleop/bin/python -m unittest gear_sonic.tests.test_inspire_hand_controller gear_sonic.tests.test_inspire_hand_standalone -v
HF_DATASETS_CACHE=/tmp/sonic-inspire-test-cache HF_HUB_OFFLINE=1 \
  .venv_data_collection/bin/python -m unittest gear_sonic.tests.test_inspire_data_collection -v
bash -n gear_sonic_deploy/deploy.sh
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
```

控制测试使用真实 0.3.0 SDK 和假 Modbus transport，覆盖左右映射、两套精确预设、二值阈值、首次输入、去重、慢读取隔离、取消、断线重连、写入超时与拒绝、反馈真实性、锁及退出。数据测试验证两种后端的全量字段和维度、消息解析、失效标记、启动参数传递、数据集兼容性，并实际写入/读回临时 parquet 与生成的视频，确认异常 episode 被排除。

## 只读连通性检查

2026-09-07 已在允许网络访问后执行独立入口的 `--hand both` 只读检查，两侧均成功：

| 手 | angle_act 实测 | 读取耗时 |
| --- | --- | --- |
| 左手 | `997 1000 996 595 719 323` | 约 0.5 ms |
| 右手 | `999 995 994 500 723 301` | 约 0.8 ms |

这些是该次读取的实测值，不表示已经完成预设动作；没有启用写入或发送运动目标。此前沙箱禁止网络访问产生的 `Operation not permitted` 不是设备不通。

也可在其他手驱动退出后，使用下面的原始适配层只读检查；它使用同一进程锁，没有 Pico 或全身控制进程，也不会启用写入：

```bash
.venv_teleop/bin/python - <<'PY'
import json
import time
from gear_sonic.utils.teleop.inspire_hand_controller import InspireHandController
hands = InspireHandController(enabled=False)
try:
    hands.start()
    time.sleep(2)
    print(json.dumps(hands.snapshot(), ensure_ascii=False, indent=2))
finally:
    hands.close()
PY
```

检查两手各自的 `host`、`connected`、`angle_valid`、`angle` 和 `error`。连接成功或读回角度均不代表运动验收通过。若不通，先记录错误，不修改设备 IP、HAND_ID、网卡、路由或 Modbus 配置。

## 本次交付检查结果

- 五指抓球与长按版：42 项控制/按键/SDK 假传输测试、9 项独立工具测试、12 项数据与启动参数测试，共 63 项通过。
- 覆盖五指默认/自定义目标、短按与长按门槛、左右独立速率、小步限速、实测滞后等待、慢通信不补发、松键/组合键/暂停/过期/退出取消、断线不重放、未确认写入、实测与目标区分、长按在途帧屏蔽、五指配置不一致拒收、不同闭合数据集拒绝续录，以及真实 parquet/video 写入读回。
- 基础 Inspire 接入时 C++ `g1_deploy_onnx_ref` 已编译成功；二进制位于 `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`，未执行。本次拇指映射只修改 Python 和文档，未再修改或编译 C++。
- 修改的 Python 文件通过语法编译，`deploy.sh` 通过 shell 语法检查，`git diff --check` 通过。
- 原有 `gear_sonic/tests/test_input_readers.py` 无法导入 `build_body_pose_sample` 和 `decode_msgpack_byte_multi_array`；检查未修改的 HEAD 版本确认这两个函数原本就缺失。本次没有修复这个独立的历史测试问题，也没有将其计为通过。
- 保留用户原有 `.gitignore` 的 `.venv` 忽略项、未跟踪的 `gear_sonic_deploy/policy/low_latency/` 和 `run_sim.md`，未重建虚拟环境、修改 tmux 会话或设备配置。

## 修改文件

| 文件 | 内容 |
| --- | --- |
| `gear_sonic/scripts/launch_data_collection.py` | 后端、端点、控制启用、旋转短按步长/长按速率/限幅及五指闭合参数透传、启动前检查 |
| `gear_sonic/scripts/pico_manager_thread_server.py` | trigger/按键/模式接入、反馈发布、信号与退出清理，Inspire 不发布 Dex3 手目标 |
| `gear_sonic/utils/teleop/inspire_hand_controller.py`（新增） | 五指固定目标、短按/慢速长按过滤、实测滞后限制、独立旋转、双线程、取消/去重/准备状态、进程锁 |
| `gear_sonic/utils/teleop/input_readers.py` | IsaacTeleop 样本附带对应控制器数据，避免缺失输入被当作松开 |
| `gear_sonic/scripts/run_data_exporter.py` | 接收反馈、每帧目标与有效性、时间戳、异常 episode、配置兼容检查 |
| `gear_sonic/utils/data_collection/inspire_hand.py`（新增） | 手部协议验证、feature、元数据、消费端有效期判断 |
| `gear_sonic/data/features_sonic_vla.py` | Inspire 的 29 路 body 数据、两路二值开合及两路旋转动作/训练映射 |
| `gear_sonic/data/exporter.py` | 布尔字段读取、Inspire discarded episode 排除和统计量重算 |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` | Dex3 禁用开关及发布该配置；保留全身控制算法 |
| `gear_sonic_deploy/deploy.sh` | 透传 Dex3 禁用开关 |
| `gear_sonic/tests/test_inspire_hand_controller.py`（新增） | 42 项按键、SDK 与控制模拟测试 |
| `gear_sonic/tests/test_inspire_data_collection.py`（新增） | 12 项数据、视频/parquet、启动参数测试 |
| `docs/inspire_hand_data_collection.md`（新增） | 本使用与验收文档 |

后续独立测试增加 `gear_sonic/scripts/test_inspire_hand.py` 和 `gear_sonic/tests/test_inspire_hand_standalone.py`；`inspire_hand_controller.py` 提取共享预设校验和进程锁函数，遥操作行为保持不变。


## 开合方向修订检查

根据操作者反馈修正两套物理预设，trigger 仍为 `>0.5 → action 1`、`<=0.5 → action 0`，首次松开准备逻辑和手臂跟随方式均保持不变。独立键盘工具也使用同一套新预设，发送前会显示实际六路目标。

开合方向修订时运行过 31 项测试，随后短按旋转版为 49 项；本版增加五指抓球和慢速长按后为上方 63 项。测试直接断言下发数值，SDK 内置预设保持原样。本次未进行实机运动复测或抓球验收。
