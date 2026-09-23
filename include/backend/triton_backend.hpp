#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast/ast.hpp"

namespace refractir {

  class TritonBackend {
  public:
    TritonBackend(std::ostream &out, std::string entry, std::string softmaxLowering);

    void emit(const Program &program);

  private:
    struct Value {
      TypePtr type;
      std::string expression;
    };

    std::ostream &out_;
    std::string entry_;
    std::string softmaxLowering_;
    const FunDecl *function_ = nullptr;
    std::unordered_map<std::string, TypePtr> types_;
    std::unordered_map<std::string, Value> values_;
    std::vector<std::string> body_;
    std::uint32_t temporaryId_ = 0;

    void emitFunction(const FunDecl &function);
    void emitInput(const ParamDecl &param);
    void emitOffsets(const std::string &name, const Shape &shape);
    void emitInstruction(const Instr &instruction);
    void emitInitializer(const LetDecl &let);
    std::string lowerExpr(const Expr &expr);
    std::string lowerAtom(const Atom &atom);
    std::string lowerTensorAtom(const TensorAtom &atom);
    std::string lowerCoef(const Coef &coef);
    std::string lowerLValue(const LValue &lvalue);
    std::string nextTemporary(const std::string &stem);
    std::string pythonName(const std::string &sirName) const;
    std::string dtypeName(const TypePtr &type) const;
    Shape valueShape(const TypePtr &type) const;
    TypePtr valueType(const Coef &coef) const;
    const FunDecl &findEntry(const Program &program) const;
    void validateEntry(const FunDecl &function) const;
    void addLine(std::string line);
  };

} // namespace refractir
