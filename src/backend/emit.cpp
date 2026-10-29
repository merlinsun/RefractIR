#include "backend/emit.hpp"

#include <chrono>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "analysis/reducibility.hpp"
#include "analysis/tensor.hpp"
#include "backend/c_backend.hpp"
#include "backend/c_vec_lowering.hpp"
#include "backend/py_backend.hpp"
#include "backend/py_vec_lowering.hpp"
#include "backend/triton_backend.hpp"
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "frontend/pipeline.hpp"

namespace fs = std::filesystem;

namespace refractir {

  namespace {

    class LayoutJsonReader {
    public:
      explicit LayoutJsonReader(std::string input) : input_(std::move(input)) {}

      std::vector<TritonBufferLayout> read() {
        expect('{');
        const auto key = readString();
        if (key != "buffers")
          fail("root object must contain 'buffers' first");
        expect(':');
        expect('[');
        std::vector<TritonBufferLayout> layouts;
        skipSpace();
        if (!consume(']')) {
          do {
            layouts.push_back(readLayout());
            skipSpace();
            if (consume(']'))
              break;
            expect(',');
          } while (true);
        }
        expect('}');
        skipSpace();
        if (position_ != input_.size())
          fail("trailing data after root object");
        return layouts;
      }

    private:
      std::string input_;
      std::size_t position_ = 0;

      [[noreturn]] void fail(const std::string &message) const {
        throw std::runtime_error(
            "invalid Triton layout manifest at byte " + std::to_string(position_) + ": " +
            message
        );
      }

      void skipSpace() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\t' ||
                input_[position_] == '\r' || input_[position_] == '\n'))
          ++position_;
      }

      bool consume(char expected) {
        skipSpace();
        if (position_ < input_.size() && input_[position_] == expected) {
          ++position_;
          return true;
        }
        return false;
      }

      void expect(char expected) {
        if (!consume(expected))
          fail(std::string("expected '") + expected + "'");
      }

      static void appendUtf8(std::string &out, std::uint32_t codepoint) {
        if (codepoint <= 0x7f) {
          out += static_cast<char>(codepoint);
        } else if (codepoint <= 0x7ff) {
          out += static_cast<char>(0xc0 | (codepoint >> 6));
          out += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else if (codepoint <= 0xffff) {
          out += static_cast<char>(0xe0 | (codepoint >> 12));
          out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
          out += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else {
          out += static_cast<char>(0xf0 | (codepoint >> 18));
          out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
          out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
          out += static_cast<char>(0x80 | (codepoint & 0x3f));
        }
      }

      std::uint32_t readHex4() {
        if (position_ + 4 > input_.size())
          fail("truncated unicode escape");
        std::uint32_t codepoint = 0;
        for (unsigned i = 0; i < 4; ++i) {
          const char ch = input_[position_++];
          codepoint <<= 4;
          if (ch >= '0' && ch <= '9')
            codepoint |= static_cast<std::uint32_t>(ch - '0');
          else if (ch >= 'a' && ch <= 'f')
            codepoint |= static_cast<std::uint32_t>(ch - 'a' + 10);
          else if (ch >= 'A' && ch <= 'F')
            codepoint |= static_cast<std::uint32_t>(ch - 'A' + 10);
          else
            fail("invalid unicode escape");
        }
        return codepoint;
      }

      std::string readString() {
        expect('"');
        std::string value;
        while (position_ < input_.size()) {
          const auto ch = static_cast<unsigned char>(input_[position_++]);
          if (ch == '"')
            return value;
          if (ch < 0x20)
            fail("control character in string");
          if (ch != '\\') {
            value += static_cast<char>(ch);
            continue;
          }
          if (position_ == input_.size())
            fail("truncated string escape");
          switch (input_[position_++]) {
          case '"': value += '"'; break;
          case '\\': value += '\\'; break;
          case '/': value += '/'; break;
          case 'b': value += '\b'; break;
          case 'f': value += '\f'; break;
          case 'n': value += '\n'; break;
          case 'r': value += '\r'; break;
          case 't': value += '\t'; break;
          case 'u': {
            auto codepoint = readHex4();
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
              if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                  input_[position_ + 1] != 'u')
                fail("high surrogate has no low surrogate");
              position_ += 2;
              const auto low = readHex4();
              if (low < 0xdc00 || low > 0xdfff)
                fail("invalid low surrogate");
              codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
              fail("unexpected low surrogate");
            }
            appendUtf8(value, codepoint);
            break;
          }
          default: fail("invalid string escape");
          }
        }
        fail("unterminated string");
      }

      std::uint64_t readUnsigned() {
        skipSpace();
        const auto begin = position_;
        while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
          ++position_;
        if (begin == position_)
          fail("expected a nonnegative integer");
        std::uint64_t value = 0;
        const auto result = std::from_chars(input_.data() + begin, input_.data() + position_, value);
        if (result.ec != std::errc{})
          fail("integer is out of range");
        return value;
      }

      std::vector<std::uint32_t> readArray() {
        expect('[');
        std::vector<std::uint32_t> values;
        skipSpace();
        if (consume(']'))
          return values;
        do {
          const auto value = readUnsigned();
          if (value > std::numeric_limits<std::uint32_t>::max())
            fail("axis value is out of range");
          values.push_back(static_cast<std::uint32_t>(value));
          skipSpace();
          if (consume(']'))
            return values;
          expect(',');
        } while (true);
      }

      TritonBufferLayout readLayout() {
        TritonBufferLayout layout;
        std::set<std::string> fields;
        expect('{');
        do {
          const auto key = readString();
          if (!fields.insert(key).second)
            fail("duplicate layout field '" + key + "'");
          expect(':');
          if (key == "name") {
            layout.name = readString();
          } else if (key == "role") {
            const auto role = readString();
            if (role == "input")
              layout.role = TritonBufferRole::Input;
            else if (role == "output")
              layout.role = TritonBufferRole::Output;
            else
              fail("layout role must be 'input' or 'output'");
          } else if (key == "element_strides") {
            layout.elementStrides = readArray();
          } else if (key == "element_offset") {
            layout.elementOffset = readUnsigned();
          } else if (key == "storage_span") {
            layout.storageSpan = readUnsigned();
          } else if (key == "axis_order") {
            layout.axisOrder = readArray();
          } else {
            fail("unknown layout field '" + key + "'");
          }
          skipSpace();
          if (consume('}'))
            break;
          expect(',');
        } while (true);
        for (const auto *required: {"name", "element_strides", "element_offset", "storage_span",
                                    "axis_order"})
          if (!fields.contains(required))
            fail(std::string("missing layout field '") + required + "'");
        if (layout.name.empty())
          fail("layout name must not be empty");
        return layout;
      }
    };

    void note(bool verbose, const std::string &msg) {
      if (verbose)
        std::cerr << "emit: " << msg << "\n";
    }

    // A program must be well-formed before a backend walks it: the backends
    // read the types the checker assigns and assume every branch resolves.
    bool checked(
        Program &prog, bool verbose, TensorConsumer consumer, TensorProfileId profileId
    ) {
      DiagBag diags;
      const auto profile = tensorProfile(profileId);
      CheckOptions checkOptions;
      checkOptions.tensorProfile = profileId;
      if (checkProgram(prog, diags, checkOptions) &&
          checkTensorCapabilities(prog, consumer, profile, diags))
        return true;
      if (verbose) {
        std::cerr << "emit: the program did not pass the checks:\n";
        for (const auto &d: diags.diags)
          if (d.level == DiagLevel::Error)
            std::cerr << "  error: " << d.message << "\n";
      }
      return false;
    }

    // Reconstructing control flow is total only on reducible CFGs. Callers
    // filter or repair upstream; verifying here turns a violation into a
    // clean failure instead of malformed output.
    bool everyFunReducible(const Program &prog, bool verbose) {
      for (const auto &f: prog.funs)
        if (!ReducibilityResult::isReducible(f)) {
          note(verbose, "structured lowering needs reducible control flow: " + f.name.name);
          return false;
        }
      return true;
    }

    bool openOut(std::ofstream &ofs, const fs::path &p, bool verbose) {
      ofs.open(p);
      if (!ofs) {
        note(verbose, "cannot open " + p.string());
        return false;
      }
      return true;
    }

  } // namespace

  std::vector<TritonBufferLayout> readTritonBufferLayouts(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open Triton layout manifest " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
      throw std::runtime_error("cannot read Triton layout manifest " + path.string());
    return LayoutJsonReader(contents.str()).read();
  }

  bool emitC(
      Program &prog, const fs::path &outDir, const std::string &primaryStem, const EmitOptions &opts
  ) {
    if (!checked(prog, opts.verbose, TensorConsumer::C, opts.tensorProfile))
      return false;
    if (opts.structuredLowering && !everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makeCVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the C target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    // The split and single-file forms differ only in where the output goes and
    // which emit call makes it; how the backend lowers is identical. Under
    // splitBySource the backend opens its own files, so `sink` stays closed.
    std::ofstream sink;
    if (!opts.splitBySource && !openOut(sink, outDir / (primaryStem + ".c"), opts.verbose))
      return false;

    CBackend cb(sink);
    cb.setNoRequire(!opts.keepRequire);
    cb.setNoUbGuards(opts.noUbGuards);
    cb.setNoMainMangle(opts.emitMain);
    cb.setStructuredLowering(opts.structuredLowering);
    cb.setVecLowering(std::move(vl));
    try {
      if (opts.splitBySource)
        cb.emitSplit(prog, outDir.string(), primaryStem);
      else
        cb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the C backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitWasm(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!checked(prog, opts.verbose, TensorConsumer::Wasm, opts.tensorProfile))
      return false;
    if (opts.structuredLowering && !everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makeWasmVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the WASM target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    std::ofstream ofs;
    if (!openOut(ofs, outFile, opts.verbose))
      return false;

    WasmBackend wb(ofs);
    wb.setNoRequire(!opts.keepRequire);
    wb.setNoUbGuards(opts.noUbGuards);
    wb.setNoMainMangle(opts.emitMain);
    wb.setStructuredLowering(opts.structuredLowering);
    wb.setVecLowering(std::move(vl));
    try {
      wb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the WASM backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitPy(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!checked(prog, opts.verbose, TensorConsumer::Python, opts.tensorProfile))
      return false;
    // Python has no goto, so the backend structures whatever it is given.
    if (!everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makePyVecLowering(opts.vecLowering.empty() ? "array" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the python target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    std::ofstream ofs;
    if (!openOut(ofs, outFile, opts.verbose))
      return false;

    PyBackend pb(ofs);
    pb.setNoRequire(!opts.keepRequire);
    pb.setNoUbGuards(opts.noUbGuards);
    pb.setNoMainMangle(opts.emitMain);
    pb.setVecLowering(std::move(vl));
    try {
      pb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the python backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitTriton(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (opts.keepRequire) {
      note(opts.verbose, "tensor-profile: --keep-require is not supported for Triton emission");
      return false;
    }
    if (opts.splitBySource || opts.structuredLowering || !opts.vecLowering.empty()) {
      note(
          opts.verbose,
          "tensor-profile: the selected code-generation option does not apply to Triton"
      );
      return false;
    }
    if (!checked(prog, opts.verbose, TensorConsumer::Triton, opts.tensorProfile))
      return false;

    std::ostringstream rendered;
    try {
      TritonBackend backend(
          rendered, opts.entry, opts.tensorSoftmaxLowering, tensorProfile(opts.tensorProfile),
          opts.tritonBufferLayouts, opts.tritonLoops
      );
      backend.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, e.what());
      return false;
    }

    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temporaryPath = outFile;
    temporaryPath += ".tmp." + std::to_string(ticks);
    std::ofstream temporary(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!temporary) {
      note(opts.verbose, "cannot open temporary output " + temporaryPath.string());
      return false;
    }
    temporary << rendered.str();
    temporary.close();
    if (!temporary) {
      std::error_code ignored;
      fs::remove(temporaryPath, ignored);
      note(opts.verbose, "cannot finish temporary output " + temporaryPath.string());
      return false;
    }
    std::error_code error;
    fs::rename(temporaryPath, outFile, error);
    if (error) {
      std::error_code ignored;
      fs::remove(temporaryPath, ignored);
      note(opts.verbose, "cannot replace " + outFile.string() + ": " + error.message());
      return false;
    }
    return true;
  }

  bool emitSirFile(
      const fs::path &sirPath, const std::string &target, const fs::path &outPath,
      const EmitOptions &opts
  ) {
    // Read separately so a missing file keeps its own message rather than
    // surfacing as a compilation failure.
    std::ifstream in(sirPath);
    if (!in) {
      note(opts.verbose, "cannot open " + sirPath.string());
      return false;
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string src = buf.str();

    try {
      Program prog = parseSource(src);
      if (target == "c") {
        // outPath names a file, so the split form has nowhere to go.
        EmitOptions single = opts;
        single.splitBySource = false;
        return emitC(prog, outPath.parent_path(), outPath.stem().string(), single);
      }
      if (target == "wasm")
        return emitWasm(prog, outPath, opts);
      if (target == "python")
        return emitPy(prog, outPath, opts);
      if (target == "triton")
        return emitTriton(prog, outPath, opts);
      note(opts.verbose, "unknown target " + target);
      return false;
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("compilation failed: ") + e.what());
      return false;
    }
  }

} // namespace refractir
