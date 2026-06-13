/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#include "smash/gpu_backend.h"

#include <cstdlib>
#include <iostream>
#include <mutex>

namespace smash {
namespace gpu {

// Implemented by exactly one of gpu_metal.mm / gpu_cuda.cu / gpu_none.cc,
// selected by CMake. Keeps the device code free of SMASH headers.
namespace detail {
bool backend_available();
const char *backend_name();
bool backend_gather(const GatherJob &job);
}  // namespace detail

namespace {
Mode config_mode = Mode::Auto;

/// Resolve the effective mode: SMASH_GPU env overrides the configured mode.
Mode effective_mode() {
  if (const char *e = std::getenv("SMASH_GPU")) {
    return mode_from_string(e);
  }
  return config_mode;
}
}  // namespace

Mode mode_from_string(const std::string &s) {
  std::string v;
  for (char c : s) {
    v += static_cast<char>(std::tolower(c));
  }
  if (v == "off" || v == "0" || v == "false" || v == "no") {
    return Mode::Off;
  }
  if (v == "on" || v == "1" || v == "true" || v == "yes" || v == "force") {
    return Mode::On;
  }
  return Mode::Auto;
}

void set_config_mode(Mode m) { config_mode = m; }

bool available() { return detail::backend_available(); }

const char *backend_name() { return detail::backend_name(); }

bool enabled() {
  // Decide once and report, so the choice is visible and not re-logged per step.
  static bool decided = false;
  static bool result = false;
  static std::once_flag once;
  std::call_once(once, [] {
    const Mode m = effective_mode();
    const bool have = detail::backend_available();
    if (m == Mode::Off) {
      result = false;
    } else if (m == Mode::On) {
      result = have;
      if (!have) {
        std::cerr << "[GPU] requested (Gpu: on / SMASH_GPU=on) but no usable "
                     "backend ("
                  << detail::backend_name() << "); falling back to CPU.\n";
      }
    } else {  // Auto
      result = have;
    }
    std::cout << "[GPU] mean-field path: "
              << (result ? "ENABLED" : "disabled") << " (backend "
              << detail::backend_name() << ", mode "
              << (m == Mode::Auto ? "auto" : m == Mode::On ? "on" : "off")
              << ")\n";
    decided = true;
  });
  (void)decided;
  return result;
}

bool run_gather(const GatherJob &job) { return detail::backend_gather(job); }

}  // namespace gpu
}  // namespace smash
