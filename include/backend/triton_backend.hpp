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

  // The `tl.range` keyword for `num_stages=N`, `loop_unroll_factor=N` (N in
  // [1, 8]), or `flatten`; nothing for any other text.
  std::optional<std::string> parseTritonLoopHint(const std::string &text);

  class TritonBackend {
  public:
    TritonBackend(
        std::ostream &out, std::string entry, std::string softmaxLowering,
        TensorProfile profile = {}, std::vector<TritonBufferLayout> layouts = {},
        TritonLoopOptions loops = {}
    );

    void emit(const Program &program);

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
    // Runtime loop-bound parameters and their literal defaults, in loop order.
    std::vector<std::pair<std::string, std::int32_t>> boundParameters_;
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
    std::string emitOffsets(
        const std::string &name, const Shape &shape, const TritonBufferLayout *layout = nullptr,
        std::string *validity = nullptr, bool includeOffsets = true
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
