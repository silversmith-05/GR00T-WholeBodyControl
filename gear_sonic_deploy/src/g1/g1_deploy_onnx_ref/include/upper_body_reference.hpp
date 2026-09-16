#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <vector>

#include "policy_parameters.hpp"

// Keep the positions and their scope together in one thread-safe snapshot.
// The first three entries in upper-body order are waist yaw/roll/pitch.
struct UpperBodyJointPositions {
  std::array<double, 17> positions{};
  bool arms_only = false;

  void ApplyPositions(std::array<double, 29>& motion) const {
    for (std::size_t i = arms_only ? 3 : 0; i < positions.size(); ++i) {
      motion[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = positions[i];
    }
  }

  void ApplyVelocities(std::array<double, 29>& motion,
                       const std::array<double, 17>& velocities) const {
    for (std::size_t i = arms_only ? 3 : 0; i < positions.size(); ++i) {
      // arm_position is a static hold. Do not reuse old upper-body velocities
      // or overwrite the planner's waist velocities.
      motion[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] =
          arms_only ? 0.0 : velocities[i];
    }
  }
};

// Decode either the existing 17-DOF upper_body_position or 14-DOF arm_position.
// Kept independent of subscribers so wire decoding can be tested without I/O.
template <typename Scalar>
std::optional<UpperBodyJointPositions> DecodeUpperBodyJointPositions(
    const void* data, std::size_t size, bool needs_swap, bool arms_only) {
  const std::size_t count = arms_only ? 14 : 17;
  if (size != count * sizeof(Scalar)) return std::nullopt;
  UpperBodyJointPositions result;
  result.arms_only = arms_only;
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < count; ++i) {
    std::array<unsigned char, sizeof(Scalar)> scalar_bytes;
    std::memcpy(scalar_bytes.data(), bytes + i * sizeof(Scalar), sizeof(Scalar));
    if (needs_swap) std::reverse(scalar_bytes.begin(), scalar_bytes.end());
    Scalar value;
    std::memcpy(&value, scalar_bytes.data(), sizeof(Scalar));
    result.positions[i + (arms_only ? 3 : 0)] = static_cast<double>(value);
  }
  return result;
}
