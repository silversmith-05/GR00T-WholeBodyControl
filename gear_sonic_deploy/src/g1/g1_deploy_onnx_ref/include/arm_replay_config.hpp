#pragma once

#include <stdexcept>
#include "observation_config.hpp"

namespace arm_replay {
// Run before DDS initialization; a token-only or SMPL-only policy cannot replay
// the joint reference. Do not let encoder fallback silently ignore the arms.
inline void validate_encoder_config(const FullObservationConfig& config) {
  const auto enabled = [](const auto& observations, const std::string& name) {
    return std::any_of(observations.begin(), observations.end(), [&](const auto& obs) {
      return obs.name == name && obs.enabled;
    });
  };
  if (config.encoder.dimension != 64 || !enabled(config.observations, "token_state"))
    throw std::runtime_error("G1 arm replay requires an enabled 64D token_state");
  const auto mode = std::find_if(config.encoder.encoder_modes.begin(), config.encoder.encoder_modes.end(),
      [](const auto& m) { return m.mode_id == 0 && m.name == "g1"; });
  if (mode == config.encoder.encoder_modes.end())
    throw std::runtime_error("G1 arm replay requires encoder mode g1=0");
  const auto required = [&](const std::string& name) {
    return enabled(config.encoder.encoder_observations, name) &&
           std::find(mode->required_observations.begin(), mode->required_observations.end(), name) !=
               mode->required_observations.end();
  };
  if (!required("encoder_mode_4") || !required("motion_joint_positions_10frame_step5") ||
      !required("motion_joint_velocities_10frame_step5") ||
      !(required("motion_anchor_orientation_10frame_step5") ||
        required("motion_anchor_orientation_heading_10frame_step5") ||
        required("motion_anchor_orientation_refheading_10frame_step5")))
    throw std::runtime_error("G1 arm replay requires 10-frame step-5 joint q/dq and anchor orientation observations");
}
}  // namespace arm_replay
