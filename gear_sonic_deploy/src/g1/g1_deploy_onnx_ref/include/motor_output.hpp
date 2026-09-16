#pragma once

#include <unitree/idl/hg/LowCmd_.hpp>

#include "robot_parameters.hpp"

// This is the final output boundary for initialization, policy control and
// shutdown damping alike. Indices here are hardware indices, not policy order.
inline void pack_motor_commands(const MotorCommand& command, bool only_arms_output,
                                unitree_hg::msg::dds_::LowCmd_& output) {
  for (size_t i = 0; i < output.motor_cmd().size(); ++i) {
    auto& motor = output.motor_cmd().at(i);
    motor = {};  // Disabled mode and zero targets/gains/torque, including unused slots.
    const bool is_arm = i >= LeftShoulderPitch && i <= RightWristYaw;
    if (i >= G1_NUM_MOTOR || (only_arms_output && !is_arm)) {
      continue;
    }
    motor.mode() = 1;
    motor.tau() = command.tau_ff.at(i);
    motor.q() = command.q_target.at(i);
    motor.dq() = command.dq_target.at(i);
    motor.kp() = command.kp.at(i);
    motor.kd() = command.kd.at(i);
  }
}
