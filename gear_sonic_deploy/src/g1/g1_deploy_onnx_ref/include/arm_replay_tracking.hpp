#pragma once

#include <iomanip>
#include <numeric>
#include <ostream>

#include "arm_replay.hpp"

namespace arm_replay {
using ArmAngles = std::array<double, 14>;

struct TrackingStats {
  size_t frames = 0;
  ArmAngles sum_abs{}, sum_squared{}, max_abs{};
  size_t worst_joint = 0;
  double worst_error = 0.0, worst_target = 0.0, worst_actual = 0.0;

  void add(const ArmAngles& target, const ArmAngles& actual) {
    for (size_t j = 0; j < names.size(); ++j) {
      const double error = target[j] - actual[j];
      sum_abs[j] += std::abs(error);
      sum_squared[j] += error * error;
      max_abs[j] = std::max(max_abs[j], std::abs(error));
      if ((frames == 0 && j == 0) || std::abs(error) > std::abs(worst_error)) {
        worst_joint = j;
        worst_error = error;
        worst_target = target[j];
        worst_actual = actual[j];
      }
    }
    ++frames;
  }

  double mae() const {
    return frames ? std::accumulate(sum_abs.begin(), sum_abs.end(), 0.0) / (14.0 * frames) : 0.0;
  }
  double rmse(size_t begin = 0, size_t end = 14) const {
    return frames ? std::sqrt(std::accumulate(sum_squared.begin() + begin, sum_squared.begin() + end, 0.0) /
                              ((end - begin) * static_cast<double>(frames))) : 0.0;
  }
};

// Owned by the control thread; finish() may also run after that thread is joined.
// This observes commands/feedback only and never changes a motor command.
class TrackingLog {
 public:
  explicit TrackingLog(const char* prefix = "[ArmReplayTracking]",
                       const char* reference = "motion_reference_at_feedback")
      : prefix_(prefix), reference_(reference) {}
  void set_phase(Phase phase, double elapsed, std::ostream& out) {
    if (finished_ || phase == phase_) return;
    flush(elapsed, out);
    phase_ = phase;
    window_start_ = elapsed;
    if (phase == Phase::Done) finish("completed", elapsed, out);
  }

  // All times use the same steady-clock origin. For trajectory error, evaluate
  // the reference at feedback_time and pass that as command_time too. For PD
  // error, use the preceding control command's generation time. Receive times
  // are not robot acquisition times, acknowledgements or latency compensation.
  void observe(double elapsed, double command_time, double feedback_time,
               const ArmAngles& target, const ArmAngles& actual, std::ostream& out) {
    if (finished_ || !active()) return;
    bool valid = finite(elapsed) && finite(command_time) && finite(feedback_time) &&
                 feedback_time >= command_time && feedback_time <= elapsed &&
                 elapsed - feedback_time <= 0.1 &&
                 (!last_feedback_time_ || feedback_time > *last_feedback_time_);
    for (size_t j = 0; j < names.size(); ++j) {
      valid = valid && finite(target[j]) && finite(actual[j]) &&
              finite(target[j] - actual[j]) && finite((target[j] - actual[j]) * (target[j] - actual[j]));
    }
    if (valid) {
      last_feedback_time_ = feedback_time;
      window_.add(target, actual);
      if (phase_ == Phase::Playing) playing_.add(target, actual);
    } else {
      ++window_skipped_;
      if (phase_ == Phase::Playing) ++playing_skipped_;
    }
    if (finite(elapsed) && elapsed - window_start_ >= 1.0) flush(elapsed, out);
  }

  void finish(const char* reason, double elapsed, std::ostream& out) {
    if (finished_) return;
    flush(elapsed, out);
    finished_ = true;
    std::ostringstream line;
    line << std::fixed << std::setprecision(5)
         << prefix_ << " reference=" << reference_ << " summary reason=" << reason << " phase=playing";
    append_stats(line, playing_, playing_skipped_);
    if (playing_.frames) {
      for (size_t j = 0; j < names.size(); ++j) {
        line << prefix_ << " joint=" << names[j] << " sdk=" << j + 15
             << " mae_rad=" << playing_.sum_abs[j] / playing_.frames
             << " rmse_rad=" << std::sqrt(playing_.sum_squared[j] / playing_.frames)
             << " max_abs_rad=" << playing_.max_abs[j] << '\n';
      }
    }
    out << line.str() << std::flush;
  }

 private:
  bool active() const {
    return phase_ == Phase::BlendIn || phase_ == Phase::Playing || phase_ == Phase::BlendOut;
  }
  const char* phase_label() const {
    switch (phase_) {
      case Phase::BlendIn: return "blend_in";
      case Phase::Playing: return "playing";
      case Phase::BlendOut: return "blend_out";
      default: return "inactive";
    }
  }
  static void append_stats(std::ostream& out, const TrackingStats& stats, size_t skipped) {
    out << " frames=" << stats.frames << " skipped=" << skipped;
    if (!stats.frames) {
      out << " no_valid_samples\n";
      return;
    }
    out << " mae_rad=" << stats.mae() << " rmse_rad=" << stats.rmse()
        << " left_rmse_rad=" << stats.rmse(0, 7) << " right_rmse_rad=" << stats.rmse(7, 14)
        << " max_abs_rad=" << std::abs(stats.worst_error)
        << " max_abs_deg=" << std::abs(stats.worst_error) * (180.0 / 3.14159265358979323846)
        << " worst=" << names[stats.worst_joint] << " sdk=" << stats.worst_joint + 15
        << " error_rad=" << stats.worst_error << " target_rad=" << stats.worst_target
        << " actual_rad=" << stats.worst_actual << '\n';
  }
  void flush(double elapsed, std::ostream& out) {
    if (window_.frames || window_skipped_) {
      std::ostringstream line;
      line << std::fixed << std::setprecision(5)
           << prefix_ << " reference=" << reference_ << " t=" << elapsed << "s phase=" << phase_label();
      append_stats(line, window_, window_skipped_);
      out << line.str() << std::flush;
    }
    window_ = {};
    window_skipped_ = 0;
    window_start_ = elapsed;
  }
  Phase phase_ = Phase::Waiting;
  TrackingStats window_, playing_;
  size_t window_skipped_ = 0, playing_skipped_ = 0;
  double window_start_ = 0.0;
  std::optional<double> last_feedback_time_;
  bool finished_ = false;
  const char* prefix_;
  const char* reference_;
};
}  // namespace arm_replay
