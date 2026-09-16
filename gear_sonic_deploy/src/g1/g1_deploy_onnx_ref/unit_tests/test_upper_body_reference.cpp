#include <gtest/gtest.h>

#include "upper_body_reference.hpp"

namespace {
template <typename Scalar>
void CheckWireAndReference(bool arms_only, bool swap) {
  const int count = arms_only ? 14 : 17;
  std::vector<unsigned char> bytes(count * sizeof(Scalar));
  for (int i = 0; i < count; ++i) {
    const Scalar value = static_cast<Scalar>(0.125 * (i + 1));
    std::memcpy(bytes.data() + i * sizeof(Scalar), &value, sizeof(Scalar));
    if (swap) {
      std::reverse(bytes.begin() + i * sizeof(Scalar), bytes.begin() + (i + 1) * sizeof(Scalar));
    }
  }
  auto targets = DecodeUpperBodyJointPositions<Scalar>(bytes.data(), bytes.size(), swap, arms_only);
  ASSERT_TRUE(targets.has_value());
  EXPECT_EQ(targets->arms_only, arms_only);
  std::array<double, 29> positions, velocities;
  std::array<double, 17> old_velocities;
  for (int i = 0; i < 29; ++i) {
    positions[i] = 100 + i;
    velocities[i] = 200 + i;
  }
  old_velocities.fill(9.0);  // A previous full upper-body command must not leak.
  targets->ApplyPositions(positions);
  targets->ApplyVelocities(velocities, old_velocities);

  // Explicit independent joint list in policy order: waist then arms.
  const std::vector<int> frozen = arms_only
      ? std::vector<int>{11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
      : std::vector<int>{2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
  for (int joint = 0; joint < 29; ++joint) {
    const auto it = std::find(frozen.begin(), frozen.end(), joint);
    if (it == frozen.end()) {
      EXPECT_DOUBLE_EQ(positions[joint], 100 + joint);
      EXPECT_DOUBLE_EQ(velocities[joint], 200 + joint);
    } else {
      EXPECT_DOUBLE_EQ(positions[joint], 0.125 * (std::distance(frozen.begin(), it) + 1));
      EXPECT_DOUBLE_EQ(velocities[joint], arms_only ? 0.0 : 9.0);
    }
  }
  EXPECT_FALSE(DecodeUpperBodyJointPositions<Scalar>(bytes.data(), bytes.size() - 1, swap, arms_only));
  EXPECT_FALSE(DecodeUpperBodyJointPositions<Scalar>(bytes.data(), bytes.size(), swap, !arms_only));
}
}  // namespace

TEST(UpperBodyReference, ArmsLeaveEveryLegAndWaistPositionAndVelocityUnchanged) {
  for (bool swap : {false, true}) {
    CheckWireAndReference<float>(true, swap);
    CheckWireAndReference<double>(true, swap);
  }
}

TEST(UpperBodyReference, Existing17DofCommandsStillOverrideWaistAndArms) {
  for (bool swap : {false, true}) {
    CheckWireAndReference<float>(false, swap);
    CheckWireAndReference<double>(false, swap);
  }
}

TEST(UpperBodyReference, SwitchingBackToFullUpperBodyRestoresWaistOverride) {
  const std::array<float, 14> arms{};
  const std::array<float, 17> full{};
  auto targets = *DecodeUpperBodyJointPositions<float>(arms.data(), sizeof(arms), false, true);
  std::array<double, 29> motion;
  motion.fill(42.0);
  targets.ApplyPositions(motion);
  EXPECT_EQ(motion[2], 42.0);
  targets = *DecodeUpperBodyJointPositions<float>(full.data(), sizeof(full), false, false);
  targets.ApplyPositions(motion);
  EXPECT_EQ(motion[2], 0.0);
  EXPECT_EQ(motion[5], 0.0);
  EXPECT_EQ(motion[8], 0.0);
  EXPECT_EQ(motion[0], 42.0);
}
