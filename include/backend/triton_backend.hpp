#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"
#include "backend/triton_layout.hpp"

namespace refractir {

  // How a counted loop's bound reaches the kernel. A literal bound is a constant
  // Triton folds; a runtime bound is a kernel parameter defaulting to the same
  // literal, so the compiler sees an unknown trip count while the launch and the
  // semantics stay those of the literal program.
  enum class TritonLoopBound { Literal, Runtime };

  // How every counted loop is emitted. `hint` is one `tl.range` keyword
  // argument, such as `num_stages=2`; it changes how Triton schedules the loop,
  // never what the loop computes.
  struct TritonLoopOptions {
    TritonLoopBound bound = TritonLoopBound::Literal;
    std::string hint;
  };

  // Grid tiling (tensor-v5): with `gridBlock` set, the kernel runs `cdiv(N, block)`
  // programs over the leading axis of extent N that the returned tensor has.
  // Each program takes rows `pid * BLOCK + arange(BLOCK)` of every tensor with
  // that leading extent, masked by `rows < N` with N a runtime argument, so a
  // non-divisible extent leaves a partial last tile. Zero emits one program.
  //
  // Stride arguments: with `strideArguments` set, every tensor buffer's strides
  // and element offset are kernel arguments (`<buffer>_stride<axis>`,
  // `<buffer>_offset`) defaulting to its layout, so a launch can lay a buffer
  // out differently without recompiling the kernel's source. Triton still
  // specializes them, so a misaligned offset takes its other code path.
  //
  // Boundary dtypes: each entry names an f32 tensor parameter (without `%`) or
  // `return` and the format its buffer holds, `float16` or `bfloat16`. An input
  // is loaded in that format and widened; the output must be the result of the
  // matching `narrow`, so storing it in that format loses nothing.
  struct TritonLaunchOptions {
    std::uint32_t gridBlock = 0;
    bool strideArguments = false;
    std::vector<std::pair<std::string, std::string>> boundaryDtypes;
  };

  // The `tl.range` keyword for `num_stages=N`, `loop_unroll_factor=N` (N in
  // [1, 8]), or `flatten`; nothing for any other text.
  std::optional<std::string> parseTritonLoopHint(const std::string &text);

  class TritonBackend {
  public:
    TritonBackend(
        std::ostream &out, std::string entry, std::string softmaxLowering,
        TensorProfile profile = {}, std::vector<TritonBufferLayout> layouts = {},
        TritonLoopOptions loops = {}, TritonLaunchOptions launch = {}
    );

    void emit(const Program &program);

    // After `emit`: the launch the kernel needs, as JSON: its grid, constexpr
    // and scalar arguments, and the relations sidecar's `launch.tiles` entry.
    [[nodiscard]] std::string launchRecord() const;

  private:
    struct Value {
      TypePtr type;
      std::string expression;
      std::string validity;
    };

    std::ostream &out_;
    std::string entry_;
    std::string softmaxLowering_;
    TensorProfile profile_;
    std::vector<TritonBufferLayout> layouts_;
    TritonLoopOptions loops_;
    TritonLaunchOptions launch_;
    // Stride and offset arguments with their layout defaults, in emission
    // order, and each buffer's argument names for the launch record.
    std::vector<std::pair<std::string, std::int64_t>> strideParameters_;
    std::vector<std::pair<std::string, std::vector<std::string>>> bufferStrides_;
    std::vector<std::pair<std::string, std::string>> bufferOffsets_;
    // The tiled leading extent N; zero without grid tiling.
    std::uint32_t gridExtent_ = 0;
    // Runtime loop-bound parameters and their literal defaults, in loop order.
    std::vector<std::pair<std::string, std::int32_t>> boundParameters_;
    // The lowest value each counted loop's induction variable takes in its body.
    std::unordered_map<std::string, std::int64_t> inductionMinimum_;
    const FunDecl *function_ = nullptr;
    std::unordered_map<std::string, TypePtr> types_;
    std::unordered_map<std::string, Value> values_;
    std::unordered_map<std::string, std::string> parameterPointers_;
    std::unordered_map<std::string, std::string> parameterOffsets_;
    std::unordered_set<std::string> usedPythonNames_;
    std::vector<std::string> body_;
    std::uint32_t temporaryId_ = 0;
    std::uint32_t indent_ = 0;

    void emitFunction(const FunDecl &function);
    void emitInput(const ParamDecl &param);
    // `buffer`, when stride arguments are on, names the kernel buffer whose
    // strides and offset become arguments (its pointer without `_ptr`).
    std::string emitOffsets(
        const std::string &name, const Shape &shape, const TritonBufferLayout *layout = nullptr,
        std::string *validity = nullptr, bool includeOffsets = true, const std::string &buffer = {}
    );
    std::string emitValidity(const std::string &name, const Shape &shape);
    void emitInstruction(const Instr &instruction);
    void emitInitializer(const LetDecl &let);
    std::string lowerExpr(const Expr &expr);
    std::string lowerAtom(const Atom &atom);
    std::string lowerTensorAtom(const TensorAtom &atom);
    std::string signedZeroExtremum(bool minimum, const std::string &left, const std::string &right);
    std::string signedZeroReduction(
        bool minimum, const std::string &lanes, std::uint32_t axis, const char *keep
    );
    std::string lowerCoef(const Coef &coef);
    std::string lowerLValue(const LValue &lvalue);
    /// The value a counted loop's induction variable holds after the loop.
    std::string loopExitValue(
        const TensorCountedLoop &loop, const std::string &start, const std::string &bound
    );
    /// Whether a window index may be negative, so its lanes need a lower mask.
    bool windowIndexMayBeNegative(const Coef &index) const;
    // The execution tile of a logical shape: the profile's padded tile, with
    // the tiled leading axis replaced by the block under grid tiling.
    Shape tileShapeOf(const Shape &shape) const;
    // A tile shape as Python text, `BLOCK` standing for a tiled leading axis.
    std::string tileText(const Shape &logical) const;
    // Refuse a function grid tiling cannot run program by program.
    void validateGridTiling(const FunDecl &function);
    // The format a buffer holds at the boundary, or empty for f32.
    std::string boundaryDtype(const std::string &buffer) const;
    // Refuse a half output that is not the result of the matching narrowing.
    void requireNarrowedReturn(
        const FunDecl &function, const Expr &returned, const std::string &dtype
    ) const;
    std::string nextTemporary(const std::string &stem);
    std::string freshPythonName(const std::string &base);
    std::string pythonName(const std::string &sirName) const;
    std::string dtypeName(const TypePtr &type) const;
    const TritonBufferLayout *findLayout(
        const std::string &sirName, TritonBufferRole role = TritonBufferRole::Unspecified
    ) const;
    Shape valueShape(const TypePtr &type) const;
    TypePtr valueType(const Coef &coef) const;
    std::string valueValidity(const Coef &coef) const;
    const FunDecl &findEntry(const Program &program) const;
    void validateEntry(const FunDecl &function) const;
    void addLine(std::string line);
  };

} // namespace refractir
