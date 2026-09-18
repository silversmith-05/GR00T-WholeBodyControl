#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <unistd.h>
#include "arm_replay.hpp"
#include "arm_replay_config.hpp"
#include "arm_replay_tracking.hpp"

namespace {
class ArmReplayTest : public ::testing::Test {
 protected:
  std::string path;
  void SetUp() override {
    char pattern[] = "/tmp/sonic-arm-replay-test-XXXXXX";
    const int fd = mkstemp(pattern);
    ASSERT_GE(fd, 0);
    close(fd);
    path = pattern;
    write();
  }
  void TearDown() override { std::filesystem::remove(path); }
  void write(const std::string& header = arm_replay::csv_header(), double last_time = .04,
             const std::string& last_value = "0.6") {
    std::ofstream file(path);
    file << header << '\n';
    for (const auto& [time, value] : std::vector<std::pair<double, std::string>>{
             {0.0, "0.2"}, {.02, "0.4"}, {last_time, last_value}}) {
      file << time;
      for (int j = 0; j < 14; ++j) file << ',' << value;
      file << '\n';
    }
  }
  arm_replay::JointReference planner(double q = .9) {
    arm_replay::JointReference ref;
    for (int i = 0; i < 29; ++i) {
      ref.q[i] = q;
      ref.dq[i] = .01 * i;
    }
    return ref;
  }
};

TEST_F(ArmReplayTest, RejectsMalformedOrUnsafeInputBeforeAnyControl) {
  write("wrong header");
  EXPECT_THROW(arm_replay::Replay{path}, std::runtime_error);
  for (double time : {.01, .02, .5}) {
    write(arm_replay::csv_header(), time);
    EXPECT_THROW(arm_replay::Replay{path}, std::runtime_error);
  }
  for (const std::string value : {"nan", "inf", "5.0", "0.5junk", "0.5,"}) {
    write(arm_replay::csv_header(), .04, value);
    EXPECT_THROW(arm_replay::Replay{path}, std::exception);
  }
}

TEST_F(ArmReplayTest, RetainsPlannerReferenceBeforeReplayAndAfterCompletion) {
  arm_replay::Replay replay(path);
  const auto original = planner();
  for (double time : {0.0, 4.999, 11.05, 50.0}) {
    const auto ref = replay.reference_at(time, original);
    EXPECT_EQ(ref.q, original.q);
    EXPECT_EQ(ref.dq, original.dq);
  }
  EXPECT_EQ(replay.phase_at(0), arm_replay::Phase::Waiting);
  EXPECT_EQ(replay.phase_at(50), arm_replay::Phase::Done);  // no auto-repeat
  EXPECT_THROW(replay.phase_at(-1), std::runtime_error);
  EXPECT_THROW(replay.phase_at(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
}

TEST_F(ArmReplayTest, MapsArmsToPolicyOrderAndPreservesLegWaistReferences) {
  {
    std::ofstream file(path);
    file << arm_replay::csv_header() << '\n';
    for (double t : {0.0, .02, .04}) {
      file << t;
      for (int j = 0; j < 14; ++j) file << ',' << .2 + 10 * t + .001 * j;
      file << '\n';
    }
  }
  arm_replay::Replay replay(path);
  // Explicit expected policy indices in left-seven/right-seven hardware order.
  const std::array<int, 14> arm_indices{11,15,19,21,23,25,27,12,16,20,22,24,26,28};
  for (double time : {5.0, 6.5, 8.0, 8.021, 8.04, 9.54}) {
    const auto original = planner();
    const auto ref = replay.reference_at(time, original);
    for (int i = 0; i < 29; ++i) {
      if (std::find(arm_indices.begin(), arm_indices.end(), i) != arm_indices.end()) continue;
      EXPECT_DOUBLE_EQ(ref.q[i], original.q[i]);
      EXPECT_DOUBLE_EQ(ref.dq[i], original.dq[i]);
    }
  }
  const auto ref = replay.reference_at(8.02, planner());
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(ref.q[arm_indices[j]], .4 + .001*j, 1e-10);
    EXPECT_NEAR(ref.dq[arm_indices[j]], 10, 1e-10);
  }
}

TEST_F(ArmReplayTest, BlendsReferencesContinuouslyAndVelocitiesMatchTheirDerivative) {
  arm_replay::Replay replay(path);
  for (const auto& [time, expected] : std::vector<std::pair<double, double>>{
           {5, .9}, {6.5, .55}, {8, .2}, {8.02, .4}, {8.04, .6}, {9.54, .75}, {11.04, .9}}) {
    EXPECT_NEAR(replay.reference_at(time, planner()).q[11], expected, 1e-10);
  }
  const double eps = 1e-7;
  for (double t : {5.0, 6.5, 8.0, 8.005, 8.015, 8.02, 8.035, 8.04, 9.54, 11.04}) {
    auto before = planner(), after = planner();
    for (size_t j = 0; j < 29; ++j) {
      before.q[j] -= eps * before.dq[j];
      after.q[j] += eps * after.dq[j];
    }
    const auto a = replay.reference_at(t-eps, before);
    const auto b = replay.reference_at(t+eps, after);
    const auto ref = replay.reference_at(t, planner());
    EXPECT_NEAR(ref.dq[11], (b.q[11]-a.q[11])/(2*eps), 1e-4);
    EXPECT_NEAR(a.q[11], b.q[11], 1e-4);
    EXPECT_NEAR(a.dq[11], b.dq[11], 1e-3);
  }
}

TEST_F(ArmReplayTest, FutureWindowUsesEachFutureTimeAndDoesNotMutatePlanner) {
  {
    std::ofstream file(path);
    file << arm_replay::csv_header() << '\n';
    for (int k = 0; k <= 10; ++k) {
      file << k*.1;
      for (int j = 0; j < 14; ++j) file << ',' << .2 + .05*k;
      file << '\n';
    }
  }
  arm_replay::Replay replay(path);
  const auto base = planner();
  for (int k = 9; k >= 0; --k) {  // Lookups need not be monotonic.
    const auto ref = replay.reference_at(8.0 + .1*k, base);
    EXPECT_NEAR(ref.q[11], .2 + .05*k, 1e-10);
    if (k > 0) EXPECT_NEAR(ref.dq[11], .5, 1e-10);
  }
  EXPECT_EQ(base.q, planner().q);
  EXPECT_EQ(base.dq, planner().dq);
  for (double t = 8; t < 9; t += .001) {
    const auto ref = replay.reference_at(t, base);
    EXPECT_GE(ref.q[11], .2 - 1e-12);
    EXPECT_LE(ref.q[11], .7 + 1e-12);
    EXPECT_GE(ref.dq[11], -1e-12);
  }
}

TEST(ArmReplayConfig, RejectsInputsThatWouldIgnoreArmReference) {
  FullObservationConfig config;
  config.observations.emplace_back("token_state", true);
  config.encoder.dimension = 64;
  EncoderModeConfig mode("g1", 0);
  mode.required_observations = {"encoder_mode_4", "motion_joint_positions_10frame_step5",
      "motion_joint_velocities_10frame_step5", "motion_anchor_orientation_10frame_step5"};
  for (const auto& name : mode.required_observations) config.encoder.encoder_observations.emplace_back(name, true);
  config.encoder.encoder_modes.push_back(mode);
  EXPECT_NO_THROW(arm_replay::validate_encoder_config(config));
  auto broken = config;
  broken.encoder.encoder_modes[0].mode_id = 2;
  EXPECT_THROW(arm_replay::validate_encoder_config(broken), std::runtime_error);
  broken = config;
  broken.encoder.encoder_observations[2].enabled = false;
  EXPECT_THROW(arm_replay::validate_encoder_config(broken), std::runtime_error);
  broken = config;
  broken.encoder.encoder_modes[0].required_observations.pop_back();
  EXPECT_THROW(arm_replay::validate_encoder_config(broken), std::runtime_error);
  broken = config;
  broken.observations[0].enabled = false;
  EXPECT_THROW(arm_replay::validate_encoder_config(broken), std::runtime_error);
}

TEST(ArmReplayTracking, DistinguishesPerfectMotionTrackingFromNonzeroPdOffset) {
  arm_replay::TrackingLog tracking;
  arm_replay::TrackingLog pd("[ArmReplayPD]", "previous_control_target");
  std::ostringstream out;
  arm_replay::ArmAngles desired{}, actual{}, command{};
  desired.fill(.23); actual.fill(.23); command.fill(-.07);
  tracking.set_phase(arm_replay::Phase::Playing, 8, out);
  pd.set_phase(arm_replay::Phase::Playing, 8, out);
  tracking.observe(8.02, 8.019, 8.019, desired, actual, out);
  pd.observe(8.02, 8, 8.019, command, actual, out);
  tracking.finish("stopped", 9, out);
  pd.finish("stopped", 9, out);
  EXPECT_NE(out.str().find("[ArmReplayTracking] reference=motion_reference_at_feedback summary reason=stopped phase=playing frames=1 skipped=0 mae_rad=0.00000"), std::string::npos);
  EXPECT_NE(out.str().find("[ArmReplayPD] reference=previous_control_target summary reason=stopped phase=playing frames=1 skipped=0 mae_rad=0.30000"), std::string::npos);
}

TEST(ArmReplayTracking, ComputesUnsignedMetricsAcrossTimeAndArms) {
  arm_replay::TrackingStats stats;
  arm_replay::ArmAngles target{}, actual{};
  target[0] = 1.0;
  actual[13] = 2.0;
  stats.add(target, actual);
  target[0] = -1.0;
  actual[13] = -2.0;
  stats.add(target, actual);
  EXPECT_EQ(stats.frames, 2);
  EXPECT_NEAR(stats.mae(), 3.0 / 14.0, 1e-12);
  EXPECT_NEAR(stats.rmse(), std::sqrt(5.0 / 14.0), 1e-12);
  EXPECT_NEAR(stats.rmse(0, 7), std::sqrt(1.0 / 7.0), 1e-12);
  EXPECT_NEAR(stats.rmse(7, 14), std::sqrt(4.0 / 7.0), 1e-12);
  EXPECT_EQ(stats.worst_joint, 13);
  EXPECT_DOUBLE_EQ(stats.worst_error, -2.0);
  EXPECT_DOUBLE_EQ(stats.worst_target, 0.0);
  EXPECT_DOUBLE_EQ(stats.worst_actual, 2.0);
}

TEST(ArmReplayTracking, ThrottlesWindowsAndSummarizesPlayingOnlyOnce) {
  arm_replay::TrackingLog log;
  std::ostringstream out;
  arm_replay::ArmAngles target{}, actual{};
  actual.fill(1.0);
  log.observe(1, .98, .999, target, actual, out);  // Waiting is excluded.
  log.set_phase(arm_replay::Phase::BlendIn, 5, out);
  log.observe(5.02, 5, 5.019, target, actual, out);
  EXPECT_TRUE(out.str().empty());
  log.set_phase(arm_replay::Phase::Playing, 8, out);
  EXPECT_NE(out.str().find("phase=blend_in frames=1 skipped=0 mae_rad=1.00000"), std::string::npos);
  out.str("");
  actual.fill(.1);
  log.observe(8.02, 8, 8.019, target, actual, out);
  log.observe(8.98, 8.96, 8.979, target, actual, out);
  EXPECT_TRUE(out.str().empty());
  log.observe(9.02, 9, 9.019, target, actual, out);
  EXPECT_NE(out.str().find("phase=playing frames=3 skipped=0 mae_rad=0.10000 rmse_rad=0.10000"), std::string::npos);
  log.set_phase(arm_replay::Phase::BlendOut, 10, out);
  actual.fill(.5);
  log.observe(10.02, 10, 10.019, target, actual, out);
  log.set_phase(arm_replay::Phase::Done, 13, out);
  EXPECT_NE(out.str().find("summary reason=completed phase=playing frames=3 skipped=0 mae_rad=0.10000"), std::string::npos);
  EXPECT_NE(out.str().find("joint=right_wrist_yaw_joint sdk=28 mae_rad=0.10000"), std::string::npos);
  const auto finished = out.str();
  log.finish("stopped", 14, out);
  log.observe(14, 13.98, 13.999, target, actual, out);
  EXPECT_EQ(out.str(), finished);
}

TEST(ArmReplayTracking, SkipsOldDuplicateOrNonfiniteFeedbackWithoutPollutingMetrics) {
  arm_replay::TrackingLog log;
  std::ostringstream out;
  arm_replay::ArmAngles target{}, actual{};
  log.set_phase(arm_replay::Phase::Playing, 8, out);
  log.observe(8.02, 8, 8.019, target, actual, out);
  log.observe(8.02, 8, 8.019, target, actual, out);  // Duplicate snapshot.
  log.observe(8.04, 8.02, 8.019, target, actual, out);  // Predates target.
  log.observe(8.2, 8.04, 8.05, target, actual, out);  // More than 100 ms old.
  log.observe(8.2, 8.18, 8.3, target, actual, out);  // Future timestamp.
  actual[13] = std::numeric_limits<double>::quiet_NaN();
  log.observe(8.22, 8.2, 8.219, target, actual, out);
  actual[13] = 0.0;
  target[0] = std::numeric_limits<double>::infinity();
  log.observe(8.24, 8.22, 8.239, target, actual, out);
  log.finish("stopped", 8.25, out);
  EXPECT_NE(out.str().find("summary reason=stopped phase=playing frames=1 skipped=6 mae_rad=0.00000"), std::string::npos);
}

TEST(ArmReplayTracking, ReportsMissingSamplesRatherThanZeroError) {
  arm_replay::TrackingLog log;
  std::ostringstream out;
  arm_replay::ArmAngles target{}, actual{};
  log.set_phase(arm_replay::Phase::Playing, 8, out);
  log.observe(9, 8.98, 8.5, target, actual, out);
  EXPECT_NE(out.str().find("phase=playing frames=0 skipped=1 no_valid_samples"), std::string::npos);
  log.finish("stopped", 9, out);
  EXPECT_NE(out.str().find("summary reason=stopped phase=playing frames=0 skipped=1 no_valid_samples"), std::string::npos);
  EXPECT_EQ(out.str().find("mae_rad="), std::string::npos);
}
}  // namespace
