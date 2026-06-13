/*
 *
 *    Copyright (c) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

// Fallback GPU backend: compiled when neither Metal (Apple) nor CUDA is
// available. Reports no device so the dispatcher always uses the CPU path.

#include "smash/gpu_backend.h"

namespace smash {
namespace gpu {
namespace detail {

bool backend_available() { return false; }
const char *backend_name() { return "none"; }
bool backend_gather(const GatherJob &) { return false; }

}  // namespace detail
}  // namespace gpu
}  // namespace smash
