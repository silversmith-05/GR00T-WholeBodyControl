#include <gtest/gtest.h>

#include "motor_output.hpp"

namespace {
MotorCommand make_command(int phase) {
  MotorCommand command;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    if (phase == 2) {  // Shutdown requests damping for every joint upstream.
      command.kd[i] = 8.0f;
    } else {
      command.q_target[i] = 0.01f * (i + 1);
      command.kp[i] = 20.0f + i;
      command.kd[i] = 1.0f + i;
      if (phase == 1) {  // Include nonzero feedforward/velocity to catch leakage.
        command.tau_ff[i] = 0.1f * (i + 1);
        command.dq_target[i] = 0.02f * (i + 1);
      }
    }
  }
  return command;
}
}  // namespace

TEST(MotorOutput, OnlyArmsOutputDisablesLegsWaistAndUnusedSlotsInEveryPhase) {
  for (int phase = 0; phase < 3; ++phase) {
    SCOPED_TRACE(phase);  // Initialization / running / shutdown damping.
    const auto command = make_command(phase);
    unitree_hg::msg::dds_::LowCmd_ output;
    // Simulate a reused message containing previously enabled motors.
    for (auto& motor : output.motor_cmd()) {
      motor = unitree_hg::msg::dds_::MotorCmd_{1, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6};
    }
    output.mode_pr() = 1;
    output.mode_machine() = 7;
    pack_motor_commands(command, true, output);

    int enabled = 0;
    for (size_t i = 0; i < output.motor_cmd().size(); ++i) {
      SCOPED_TRACE(i);
      const auto& motor = output.motor_cmd()[i];
      if (i >= 15 && i <= 28) {
        ++enabled;
        EXPECT_EQ(motor.mode(), 1);
        EXPECT_FLOAT_EQ(motor.q(), command.q_target[i]);
        EXPECT_FLOAT_EQ(motor.dq(), command.dq_target[i]);
        EXPECT_FLOAT_EQ(motor.tau(), command.tau_ff[i]);
        EXPECT_FLOAT_EQ(motor.kp(), command.kp[i]);
        EXPECT_FLOAT_EQ(motor.kd(), command.kd[i]);
      } else {
        EXPECT_EQ(motor, unitree_hg::msg::dds_::MotorCmd_{});
      }
    }
    EXPECT_EQ(enabled, 14);
    EXPECT_EQ(output.mode_pr(), 1);
    EXPECT_EQ(output.mode_machine(), 7);
  }
}

TEST(MotorOutput, DefaultFullBodyPreservesAll29CommandsInEveryPhase) {
  for (int phase = 0; phase < 3; ++phase) {
    SCOPED_TRACE(phase);
    const auto command = make_command(phase);
    unitree_hg::msg::dds_::LowCmd_ output;
    pack_motor_commands(command, false, output);
    for (size_t i = 0; i < output.motor_cmd().size(); ++i) {
      SCOPED_TRACE(i);
      const auto& motor = output.motor_cmd()[i];
      if (i < 29) {
        EXPECT_EQ(motor.mode(), 1);
        EXPECT_FLOAT_EQ(motor.q(), command.q_target[i]);
        EXPECT_FLOAT_EQ(motor.dq(), command.dq_target[i]);
        EXPECT_FLOAT_EQ(motor.tau(), command.tau_ff[i]);
        EXPECT_FLOAT_EQ(motor.kp(), command.kp[i]);
        EXPECT_FLOAT_EQ(motor.kd(), command.kd[i]);
      } else {
        EXPECT_EQ(motor, unitree_hg::msg::dds_::MotorCmd_{});
      }
    }
  }
}
