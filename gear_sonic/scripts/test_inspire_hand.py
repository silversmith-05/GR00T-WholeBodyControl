"""Standalone RH56E2-T1 check. No Pico, cameras, ZMQ or full-body deployment.

Default: read angle_act once and disconnect. --enable-control permits interactive
0/1 commands for ONE selected hand; opening the session never sends a target.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import signal
import time

from gear_sonic.utils.teleop.inspire_hand_controller import (
    CLOSE_ANGLES, HAND_FORCE, acquire_hand_lock, hand_preset, validate_inspire_sdk, validate_close_angles,
)

HOSTS = {"left": "192.168.123.211", "right": "192.168.123.210"}
LABELS = {"left": "左手", "right": "右手"}


def run_session(side, *, host=None, port=6000, enable_control=False, timeout=0.3,
                client_factory=None, input_fn=input, output_fn=print, lock_dir="/tmp", close_angles=CLOSE_ANGLES):
    """One SDK client and one ownership lock, closed on all exit paths."""
    from inspire_rh56e2 import HandClient, HandConfig, WriteUnconfirmed

    validate_inspire_sdk()
    close_angles = validate_close_angles(close_angles)
    presets = [hand_preset(target, close_angles=close_angles) for target in (0,1)]
    config = HandConfig(host=host or HOSTS[side], port=port, unit_id=255, model="RH56E2-T1",
                        byte_layout="packed_little", tactile_order="big", timeout=timeout)
    label = f"{LABELS[side]} {config.host}:{config.port}"
    last_target = None

    def show_angles(hand):
        reading = hand.read_state_field("angle_act")
        output_fn(f"[{label}] angle_act={list(reading.values)} "
                  f"timestamp={reading.timestamp:.6f} elapsed_ms={reading.elapsed_ms:.1f}")
        if last_target is not None:
            delta = [a - b for a, b in zip(reading.values, presets[last_target].angle)]
            output_fn(f"目标={last_target}（{'放开' if last_target == 0 else '闭合'}），"
                      f"实测减目标={delta}；六路均在 ±20 标度内={all(abs(d) <= 20 for d in delta)}")

    try:
        with ExitStack() as resources:
            resources.enter_context(acquire_hand_lock(config.host, config.port, lock_dir))
            hand = (client_factory or HandClient)(config)
            resources.callback(hand.close)  # disconnect only, never close_hand()
            output_fn(f"[{label}] 连接中；unit_id=255，仅 Modbus TCP。")
            hand.connect()
            show_angles(hand)
            if not enable_control:
                output_fn(f"[{label}] 只读检查成功，没有发送运动目标。")
                return 0

            output_fn(f"[{label}] 已进入手动测试；启动未发送开合。")
            output_fn(f"输入 0=放开、1=闭合、s=读取实际角度、q=退出，然后回车。速度为六路 200，力阈值为六路 {HAND_FORCE}。")
            while True:
                try:
                    command = input_fn(f"{LABELS[side]} [0/1/s/q]> ").strip().lower()
                except EOFError:
                    return 0
                if command == "q":
                    return 0
                if command == "s":
                    show_angles(hand)
                    continue
                if command not in {"0", "1"}:
                    output_fn("请输入 0、1、s 或 q；未发送任何命令。")
                    continue
                target = int(command)
                if target == last_target:
                    output_fn("目标与上次确认写入相同，不重复写入；可输入 s 查看实际角度。")
                    continue
                # Verify a current read before accepting this explicit user action.
                # Any failure exits; this tester does not reconnect or replay commands.
                show_angles(hand)
                deadline = time.monotonic() + 2.0

                def guard():
                    if time.monotonic() > deadline:
                        raise TimeoutError("手动命令已过期，停止后续写入")

                try:
                    output_fn(f"[{label}] 发送目标 {target}，角度标度={list(presets[target].angle)}")
                    hand.enable_writes()
                    hand.apply_preset(presets[target], guard=guard)
                except WriteUnconfirmed as exc:
                    output_fn(f"[{label}] 写入结果未确认，不重试：{exc}")
                    return 2
                except KeyboardInterrupt:
                    output_fn(f"[{label}] 写入过程中被中断，结果未确认，不重试。")
                    return 2
                except Exception as exc:
                    output_fn(f"[{label}] 写入失败或部分写入，不重试：{exc}")
                    return 2
                finally:
                    hand.enable_writes(False)
                last_target = target
                output_fn(f"[{label}] 目标 {target} 的寄存器写入已确认；不表示实际运动到位。")
                show_angles(hand)
    except KeyboardInterrupt:
        output_fn(f"[{label}] 已停止测试并断开连接；退出不发送开合。")
        return 130
    except Exception as exc:
        output_fn(f"[{label}] 检查失败，已退出且不自动重试：{exc}")
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--host", help="Override selected hand's IPv4 address (single hand only)")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--enable-control", action="store_true", help="Allow manual 0/1 input; no startup motion")
    parser.add_argument("--inspire-close-angles", type=int, nargs=5, default=CLOSE_ANGLES,
                        help="Fixed five bend targets; rotation remains 339 in this keyboard tool")
    args = parser.parse_args(argv)
    if args.hand == "both" and (args.enable_control or args.host):
        parser.error("运动测试或 --host 必须明确选择 --hand left 或 --hand right")

    def stop_on_signal(*_):
        raise KeyboardInterrupt

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for sig in handlers:
            signal.signal(sig, stop_on_signal)
        sides = ("left", "right") if args.hand == "both" else (args.hand,)
        results = [run_session(side, host=args.host, port=args.port,
                               enable_control=args.enable_control, timeout=args.timeout,
                               close_angles=args.inspire_close_angles)
                   for side in sides]
        return max(results)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
