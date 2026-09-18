#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "policy_parameters.hpp"

// Hardware order, matching gear_sonic/data/assets/robot_description/urdf/g1/main.urdf.
namespace arm_replay {
inline constexpr const char* implementation = "ARM_REPLAY_G1_ENCODER_V1";
// The deployment build enables -ffast-math, under which std::isfinite may be
// optimized away. Check IEEE-754 exponent bits at this file/input boundary.
inline bool finite(double value) {
  return (std::bit_cast<uint64_t>(value) & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
inline constexpr std::array<const char*, 14> names = {
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};
inline constexpr std::array<double, 14> lower = {
    -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558,
    -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558};
inline constexpr std::array<double, 14> upper = {
    2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558,
    2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558};

struct Sample {
  double time;
  std::array<double, 14> q;
};

inline std::string csv_header() {
  std::string header = "time_s";
  for (const auto* name : names) header += "," + std::string(name) + "_q";
  return header;
}

// Prepared CSV only: raw lowstate dq/tau_est are intentionally never loaded.
inline std::vector<Sample> load_csv(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open arm replay: " + path);
  std::string line;
  auto read_line = [&]() {
    if (!std::getline(file, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
  };
  if (!read_line() || line != csv_header())
    throw std::runtime_error("Invalid arm replay header; use launch_arm_replay.py to prepare the recording");
  std::vector<Sample> samples;
  size_t line_number = 1;
  while (read_line()) {
    ++line_number;
    std::istringstream row(line);
    std::array<double, 15> values{};
    std::string cell;
    for (double& value : values) {
      if (!std::getline(row, cell, ',')) throw std::runtime_error("Short replay row " + std::to_string(line_number));
      size_t used = 0;
      value = std::stod(cell, &used);
      if (used != cell.size() || !finite(value))
        throw std::runtime_error("Invalid replay number at row " + std::to_string(line_number));
    }
    if (std::getline(row, cell, ',') || (!line.empty() && line.back() == ','))
      throw std::runtime_error("Extra replay column at row " + std::to_string(line_number));
    Sample sample{values[0], {}};
    if ((samples.empty() && sample.time != 0.0) ||
        (!samples.empty() && (sample.time <= samples.back().time || sample.time - samples.back().time > 0.100001)))
      throw std::runtime_error("Replay times must start at 0, increase, and have gaps <= 0.1 s");
    for (size_t j = 0; j < names.size(); ++j) {
      sample.q[j] = values[j + 1];
      if (sample.q[j] < lower[j] || sample.q[j] > upper[j])
        throw std::runtime_error("Replay angle outside URDF bounds: " + std::string(names[j]));
    }
    samples.push_back(sample);
  }
  if (samples.size() < 2) throw std::runtime_error("Arm replay needs at least two samples");
  return samples;
}

enum class Phase { Waiting, BlendIn, Playing, BlendOut, Done };
inline const char* phase_name(Phase phase) {
  switch (phase) {
    case Phase::Waiting: return "SONIC settling (5 s)";
    case Phase::BlendIn: return "blend arm reference to first recorded pose (3 s)";
    case Phase::Playing: return "G1 encoder tracking recorded arm reference";
    case Phase::BlendOut: return "blend arm reference back to IDLE planner (3 s)";
    case Phase::Done: return "complete; SONIC follows IDLE reference; press O to stop";
  }
  return "unknown";
}

// Full-body motion reference in IsaacLab order, never a low-level MotorCommand.
struct JointReference {
  std::array<double, 29> q{}, dq{};
};

class Replay {
 public:
  explicit Replay(const std::string& path) : samples_(load_csv(path)), slopes_(samples_.size()) {
    for (size_t k = 1; k < samples_.size(); ++k)
      for (size_t j = 0; j < names.size(); ++j)
        if (!finite((samples_[k].q[j] - samples_[k - 1].q[j]) /
                    (samples_[k].time - samples_[k - 1].time)))
          throw std::runtime_error("Invalid replay derivative");
    // Shape-preserving cubic Hermite interpolation. Interior derivatives use
    // the weighted harmonic mean; endpoint derivatives are zero to join holds.
    for (size_t k = 1; k + 1 < samples_.size(); ++k) {
      const double h0 = samples_[k].time - samples_[k - 1].time;
      const double h1 = samples_[k + 1].time - samples_[k].time;
      for (size_t j = 0; j < names.size(); ++j) {
        const double a = (samples_[k].q[j] - samples_[k - 1].q[j]) / h0;
        const double b = (samples_[k + 1].q[j] - samples_[k].q[j]) / h1;
        if (!finite(a) || !finite(b)) throw std::runtime_error("Invalid replay derivative");
        if (a * b > 0.0) {
          const double w0 = 2 * h1 + h0, w1 = h1 + 2 * h0;
          slopes_[k][j] = (w0 + w1) / (w0 / a + w1 / b);
        }
      }
    }
  }
  double duration() const { return samples_.back().time; }
  size_t size() const { return samples_.size(); }

  Phase phase_at(double elapsed) const {
    if (!finite(elapsed) || elapsed < 0.0) throw std::runtime_error("Invalid arm replay clock");
    if (elapsed < 5.0) return Phase::Waiting;
    if (elapsed < 8.0) return Phase::BlendIn;
    if (elapsed <= 8.0 + duration()) return Phase::Playing;
    if (elapsed < 11.0 + duration()) return Phase::BlendOut;
    return Phase::Done;
  }

  // Pure time lookup: callable in any order for current and future frames.
  // The encoder sees q and its analytic derivative; neither is sent directly
  // to motors. Legs/waist remain exactly the planner's reference values.
  JointReference reference_at(double elapsed, const JointReference& planner) const {
    const auto phase = phase_at(elapsed);
    auto result = planner;
    if (phase == Phase::Waiting || phase == Phase::Done) return result;
    auto pose = sample_at(std::clamp(elapsed - 8.0, 0.0, duration()));
    double weight = 1.0, weight_dot = 0.0;
    if (phase == Phase::BlendIn) {
      const double x = (elapsed - 5.0) / 3.0;
      weight = smooth(x);
      weight_dot = smooth_derivative(x) / 3.0;
    } else if (phase == Phase::BlendOut) {
      const double x = (elapsed - 8.0 - duration()) / 3.0;
      weight = 1.0 - smooth(x);
      weight_dot = -smooth_derivative(x) / 3.0;
    }
    for (size_t j = 0; j < names.size(); ++j) {
      const size_t index = isaaclab_to_mujoco[j + 15];  // hardware -> policy index
      result.q[index] = (1.0 - weight) * planner.q[index] + weight * pose.q[j];
      result.dq[index] = (1.0 - weight) * planner.dq[index] + weight * pose.dq[j] +
                         weight_dot * (pose.q[j] - planner.q[index]);
    }
    return result;
  }

 private:
  struct Pose { std::array<double, 14> q{}, dq{}; };
  Pose sample_at(double time) const {
    if (time <= 0.0) return {samples_.front().q, {}};
    if (time >= duration()) return {samples_.back().q, {}};
    auto upper = std::upper_bound(samples_.begin(), samples_.end(), time,
        [](double t, const Sample& sample) { return t < sample.time; });
    const size_t k = std::distance(samples_.begin(), upper) - 1;
    const double h = samples_[k + 1].time - samples_[k].time;
    const double x = (time - samples_[k].time) / h, x2 = x * x, x3 = x2 * x;
    Pose result;
    for (size_t j = 0; j < names.size(); ++j) {
      const double a = samples_[k].q[j], b = samples_[k + 1].q[j];
      const double da = slopes_[k][j], db = slopes_[k + 1][j];
      result.q[j] = (2*x3 - 3*x2 + 1)*a + (x3 - 2*x2 + x)*h*da +
                     (-2*x3 + 3*x2)*b + (x3 - x2)*h*db;
      result.dq[j] = ((6*x2 - 6*x)*a + (-6*x2 + 6*x)*b) / h +
                      (3*x2 - 4*x + 1)*da + (3*x2 - 2*x)*db;
    }
    return result;
  }
  static double smooth(double x) { return x * x * x * (10.0 + x * (-15.0 + 6.0 * x)); }
  static double smooth_derivative(double x) { return 30.0 * x * x * (1.0 - x) * (1.0 - x); }
  std::vector<Sample> samples_;
  std::vector<std::array<double, 14>> slopes_;
};
}  // namespace arm_replay
