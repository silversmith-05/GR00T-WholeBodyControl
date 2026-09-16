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
