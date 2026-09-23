#include "interp/tensor_io.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"
#include "interp/tensor.hpp"

namespace refractir {
  namespace {

    std::uint64_t byteLength(const TensorTypeInfo &type) {
      const std::uint64_t count = type.shape.empty() ? 1 : tensorElementCount(type.shape);
      if (count > std::numeric_limits<std::uint64_t>::max() / 4)
        throw std::runtime_error("tensor byte length overflows uint64");
      return count * 4;
    }

    std::uint32_t readWord(const std::vector<char> &bytes, std::size_t offset) {
      const auto b0 = static_cast<std::uint8_t>(bytes[offset]);
      const auto b1 = static_cast<std::uint8_t>(bytes[offset + 1]);
      const auto b2 = static_cast<std::uint8_t>(bytes[offset + 2]);
      const auto b3 = static_cast<std::uint8_t>(bytes[offset + 3]);
      return static_cast<std::uint32_t>(b0) | (static_cast<std::uint32_t>(b1) << 8) |
             (static_cast<std::uint32_t>(b2) << 16) | (static_cast<std::uint32_t>(b3) << 24);
    }

    void writeWord(std::ostream &output, std::uint32_t word) {
      const char bytes[4] = {
          static_cast<char>(word & 0xff), static_cast<char>((word >> 8) & 0xff),
          static_cast<char>((word >> 16) & 0xff), static_cast<char>((word >> 24) & 0xff)
      };
      output.write(bytes, sizeof(bytes));
      if (!output)
        throw std::runtime_error("failed to write tensor output bytes");
    }

    RuntimeValue decodeScalar(std::uint32_t word, TensorElement element) {
      RuntimeValue value;
      value.bits = 32;
      if (element == TensorElement::I32) {
        value.kind = RuntimeValue::Kind::Int;
        value.intVal = std::bit_cast<std::int32_t>(word);
      } else {
        const auto exponent = (word >> 23) & 0xffU;
        const auto fraction = word & 0x7fffffU;
        if (exponent == 0xffU)
          throw std::runtime_error("tensor input contains a non-finite f32 value");
        value.kind = RuntimeValue::Kind::Float;
        const std::uint64_t sign = static_cast<std::uint64_t>(word >> 31) << 63;
        std::uint64_t widened = sign;
        if (exponent != 0) {
          widened |= static_cast<std::uint64_t>(exponent + 896U) << 52;
          widened |= static_cast<std::uint64_t>(fraction) << 29;
        } else if (fraction != 0) {
          const auto highest = 31U - std::countl_zero(fraction);
          const auto normalizedFraction = fraction - (1U << highest);
          widened |= static_cast<std::uint64_t>(highest + 874U) << 52;
          widened |= static_cast<std::uint64_t>(normalizedFraction) << (52U - highest);
        }
        value.floatVal = std::bit_cast<double>(widened);
      }
      return value;
    }

    std::vector<RuntimeValue> flatten(const RuntimeValue &value, const TensorTypeInfo &type) {
      std::vector<RuntimeValue> values;
      const auto count = type.shape.empty() ? 1 : tensorElementCount(type.shape);
      if (type.shape.empty()) {
        if (value.kind == RuntimeValue::Kind::Tensor)
          throw std::runtime_error("tensor output has a tensor value where a scalar is required");
        values.push_back(value);
      } else {
        if (value.kind != RuntimeValue::Kind::Tensor || value.tensorShape != type.shape ||
            value.arrayVal.size() != count)
          throw std::runtime_error("tensor output shape does not match the checked result type");
        values = value.arrayVal;
      }
      for (auto &scalar: values) {
        if (scalar.kind == RuntimeValue::Kind::Undef)
          throw std::runtime_error("tensor output contains undef");
        if (type.element == TensorElement::I32) {
          if (scalar.kind != RuntimeValue::Kind::Int || scalar.intVal < INT32_MIN ||
              scalar.intVal > INT32_MAX)
            throw std::runtime_error("tensor output has a value outside the i32 type");
        } else if (scalar.kind != RuntimeValue::Kind::Float || scalar.bits != 32 ||
                   !std::isfinite(scalar.floatVal) ||
                   !std::isfinite(static_cast<float>(scalar.floatVal))) {
          throw std::runtime_error("tensor output has a non-finite or non-f32 value");
        }
      }
      return values;
    }

    std::string jsonEscape(std::string_view value) {
      std::string escaped;
      escaped.reserve(value.size());
      for (const unsigned char ch: value) {
        if (ch == '"' || ch == '\\') {
          escaped.push_back('\\');
          escaped.push_back(static_cast<char>(ch));
        } else if (ch < 0x20) {
          constexpr char digits[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(digits[ch >> 4]);
          escaped.push_back(digits[ch & 15]);
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
      }
      return escaped;
    }

    void writeShape(std::ostream &json, const Shape &shape) {
      json << '[';
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i)
          json << ',';
        json << shape[i];
      }
      json << ']';
    }

    const FunDecl &findEntry(const Program &program, std::string_view entryName) {
      if (!entryName.empty() && entryName.front() == '@')
        entryName.remove_prefix(1);
      for (const auto &function: program.funs)
        if (function.name.name.size() == entryName.size() + 1 &&
            function.name.name.front() == '@' &&
            function.name.name.compare(1, entryName.size(), entryName) == 0)
          return function;
      throw std::runtime_error("Entry function not found: " + std::string(entryName));
    }

  } // namespace

  std::vector<RuntimeValue>
  readTensorArguments(std::istream &input, const std::vector<TensorTypeInfo> &types) {
    std::uint64_t expected = 0;
    for (const auto &type: types) {
      const auto length = byteLength(type);
      if (length > std::numeric_limits<std::uint64_t>::max() - expected)
        throw std::runtime_error("tensor input byte length overflows uint64");
      expected += length;
    }
    if (expected > std::numeric_limits<std::size_t>::max() ||
        expected > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
      throw std::runtime_error("tensor input byte length exceeds the host stream limits");

    std::vector<char> bytes(static_cast<std::size_t>(expected));
    if (expected != 0) {
      input.read(bytes.data(), static_cast<std::streamsize>(expected));
      if (input.bad())
        throw std::runtime_error("failed to read tensor input bytes");
      const auto received = static_cast<std::uint64_t>(input.gcount());
      if (received != expected)
        throw std::runtime_error(
            "tensor input has " + std::to_string(received) + " bytes; expected exactly " +
            std::to_string(expected)
        );
    }
    if (input.peek() != std::char_traits<char>::eof())
      throw std::runtime_error(
          "tensor input has more than " + std::to_string(expected) + " bytes; expected exactly " +
          std::to_string(expected)
      );
    if (input.bad())
      throw std::runtime_error("failed to read tensor input bytes");

    std::vector<RuntimeValue> values;
    values.reserve(types.size());
    std::size_t offset = 0;
    for (const auto &type: types) {
      const auto count = type.shape.empty() ? 1 : tensorElementCount(type.shape);
      std::vector<RuntimeValue> leaves;
      leaves.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i, offset += 4)
        leaves.push_back(decodeScalar(readWord(bytes, offset), type.element));
      if (type.shape.empty()) {
        values.push_back(std::move(leaves.front()));
      } else {
        RuntimeValue tensor;
        tensor.kind = RuntimeValue::Kind::Tensor;
        tensor.tensorShape = type.shape;
        tensor.arrayVal = std::move(leaves);
        tensor.bits = 32;
        values.push_back(std::move(tensor));
      }
    }
    return values;
  }

  void
  writeTensorResult(std::ostream &output, const RuntimeValue &value, const TensorTypeInfo &type) {
    for (const auto &scalar: flatten(value, type)) {
      if (type.element == TensorElement::I32) {
        writeWord(output, std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(scalar.intVal)));
      } else {
        writeWord(output, std::bit_cast<std::uint32_t>(static_cast<float>(scalar.floatVal)));
      }
    }
  }

  std::string describeTensorEntry(const Program &program, const std::string &entryName) {
    const auto &entry = findEntry(program, entryName);
    std::ostringstream json;
    const auto unprefixedEntryName = entry.name.name.empty() || entry.name.name.front() != '@'
                                         ? entry.name.name
                                         : entry.name.name.substr(1);
    json << "{\"entry\":\"" << jsonEscape(unprefixedEntryName) << "\",\"inputs\":[";
    for (std::size_t i = 0; i < entry.params.size(); ++i) {
      if (i)
        json << ',';
      const auto info = tensorTypeInfo(entry.params[i].type);
      const auto paramName =
          entry.params[i].name.name.empty() || entry.params[i].name.name.front() != '%'
              ? entry.params[i].name.name
              : entry.params[i].name.name.substr(1);
      json << "{\"name\":\"" << jsonEscape(paramName) << "\",\"dtype\":\""
           << (info.element == TensorElement::I32 ? "i32" : "f32") << "\",\"shape\":";
      writeShape(json, info.shape);
      json << ",\"byte_length\":" << byteLength(info) << '}';
    }
    const auto output = tensorTypeInfo(entry.retType);
    json << "],\"output\":{\"name\":\"return\",\"dtype\":\""
         << (output.element == TensorElement::I32 ? "i32" : "f32") << "\",\"shape\":";
    writeShape(json, output.shape);
    json << ",\"byte_length\":" << byteLength(output) << "}}\n";
    return json.str();
  }

} // namespace refractir
