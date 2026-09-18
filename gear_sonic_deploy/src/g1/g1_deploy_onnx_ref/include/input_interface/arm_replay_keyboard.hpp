#pragma once

#include "keyboard_handler.hpp"

// Reuse the original IDLE planner and start/stop handling. No headset, hand
// driver, walking keys, or second DDS publisher is needed for recorded arms.
class ArmReplayKeyboard : public SimpleKeyboard {
 public:
  void update() override {
    start_control = false;
    stop_control = false;
    planner_use_movement_mode = LocomotionMode::IDLE;
    movement_momentum = 0.0;
    char ch;
    while (ReadStdinChar(ch)) {
      if (ch == ']') {
        use_planner = true;
        start_control = true;
      }
      if (ch == 'o' || ch == 'O') stop_control = true;
    }
  }
};
