#pragma once

#include <string>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"

namespace refractir::reify {

  /// One corner path of a kernel's control flow. `path` is the block labels
  /// from `^entry` to the returning block; `skeleton` is the kernel's scalar
  /// skeleton as SIR text, its i32 parameters turned into ranged symbols and
  /// `require` lines pinning the corner, so `symirsolve --path` on it yields
  /// scalars that make the kernel take `path`.
  struct SteeringTarget {
    std::string name;
    std::vector<std::string> path;
    std::string skeleton;
  };

  /// The targets of one checked kernel, in a fixed order: `baseline` (every
  /// runtime loop at one trip, every branch on its first arm), then per loop
  /// with a parameter endpoint `loop<k>-empty-equal`, `loop<k>-empty-inverted`
  /// (an empty range whose start lies past its stop, the shape of upstream
  /// F2), `loop<k>-one` and `loop<k>-most`; per scalar branch `branch<k>-then`
  /// and `branch<k>-else`; per window indexed by such a loop `window<k>-below`
  /// and `window<k>-above` (its first tile straddling the start or the end of
  /// the active rows). Tensor statements are dropped from the skeleton: under
  /// the tensor profiles no tensor value reaches a branch condition or an
  /// integer local, so control flow is the skeleton's alone. Throws
  /// std::invalid_argument when the entry is missing or its control flow is
  /// not the profile's counted-loop form.
  std::vector<SteeringTarget>
  steeringTargets(const Program &program, const std::string &entry, const TensorProfile &profile);

} // namespace refractir::reify
