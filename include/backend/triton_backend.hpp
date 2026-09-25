#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"
#include "backend/triton_layout.hpp"

namespace refractir {

  class TritonBackend {
  public:
    TritonBackend(
        std::ostream &out, std::string entry, std::string softmaxLowering,
        TensorProfile profile = {}, std::vector<TritonBufferLayout> layouts = {}
    );

    void emit(const Program &program);

  private:
    struct Value {
      TypePtr type;
      std::string expression;
    };

    std::ostream &out_;
    std::string entry_;
    std::string softmaxLowering_;
    TensorProfile profile_;
    std::vector<TritonBufferLayout> layouts_;
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
        const std::string &name, const Shape &shape, const TritonBufferLayout *layout = nullptr
    );
    void emitInstruction(const Instr &instruction);
    void emitInitializer(const LetDecl &let);
    std::string lowerExpr(const Expr &expr);
    std::string lowerAtom(const Atom &atom);
    std::string lowerTensorAtom(const TensorAtom &atom);
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
    const FunDecl &findEntry(const Program &program) const;
    void validateEntry(const FunDecl &function) const;
    void addLine(std::string line);
  };

} // namespace refractir
