#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace refractir {

  /// Compile-time physical storage mapping for one logical tensor buffer.
  struct TritonBufferLayout {
    std::string name;
    std::vector<std::uint32_t> elementStrides;
    std::uint64_t elementOffset = 0;
    std::uint64_t storageSpan = 0;
    std::vector<std::uint32_t> axisOrder;
  };

} // namespace refractir
