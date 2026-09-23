#include "backend/triton_backend.hpp"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {
  namespace {

    std::string pythonShape(const Shape &shape) {
      std::string result = "(";
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i)
          result += ", ";
        result += std::to_string(shape[i]);
      }
      if (shape.size() == 1)
        result += ",";
      return result + ")";
    }

    std::string printFloat(double value) { return formatDouble(value); }

    std::string localId(const LocalOrSymId &id) {
      return std::visit([](const auto &name) { return name.name; }, id);
    }

    std::uint32_t productAfter(const Shape &shape, std::size_t axis) {
      return std::accumulate(shape.begin() + axis + 1, shape.end(), 1u, std::multiplies<>());
    }

    bool isTensorAtom(const Atom &atom) { return std::holds_alternative<TensorAtom>(atom.v); }

    bool expressionHasCall(const Expr &expr) {
      if (std::holds_alternative<CallAtom>(expr.first.v))
        return true;
      return std::any_of(expr.rest.begin(), expr.rest.end(), [](const Expr::Tail &tail) {
        return std::holds_alternative<CallAtom>(tail.atom.v);
      });
    }

    bool functionHasTensorAtom(const FunDecl &function) {
      for (const auto &block: function.blocks) {
        for (const auto &instruction: block.instrs) {
          const bool found = std::visit(
              [](const auto &ins) {
                using T = std::decay_t<decltype(ins)>;
                if constexpr (std::is_same_v<T, AssignInstr>) {
                  if (isTensorAtom(ins.rhs.first))
                    return true;
                  return std::any_of(
                      ins.rhs.rest.begin(), ins.rhs.rest.end(),
                      [](const auto &tail) { return isTensorAtom(tail.atom); }
                  );
                } else if constexpr (std::is_same_v<T, StoreInstr>) {
                  return expressionHasCall(ins.ptr) || expressionHasCall(ins.val);
                } else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                     std::is_same_v<T, RequireInstr>) {
                  return isTensorAtom(ins.cond.lhs.first) || isTensorAtom(ins.cond.rhs.first);
                }
                return false;
              },
              instruction
          );
          if (found)
            return true;
        }
        if (const auto ret = std::get_if<RetTerm>(&block.term); ret && ret->value) {
          if (isTensorAtom(ret->value->first))
            return true;
          if (std::any_of(ret->value->rest.begin(), ret->value->rest.end(), [](const auto &tail) {
                return isTensorAtom(tail.atom);
              }))
            return true;
        }
      }
      return false;
    }

    bool typeHasTensor(const TypePtr &type) { return TypeUtils::asTensor(type) != nullptr; }

  } // namespace

  TritonBackend::TritonBackend(std::ostream &out, std::string entry, std::string softmaxLowering) :
      out_(out), entry_(std::move(entry)), softmaxLowering_(std::move(softmaxLowering)) {}

  const FunDecl &TritonBackend::findEntry(const Program &program) const {
    auto it = std::find_if(program.funs.begin(), program.funs.end(), [&](const FunDecl &function) {
      return function.name.name == "@" + entry_ || function.name.name == entry_;
    });
    if (it == program.funs.end())
      throw std::runtime_error("tensor-profile: entry function was not found: " + entry_);
    return *it;
  }

  void TritonBackend::validateEntry(const FunDecl &function) const {
    if (function.blocks.size() != 1 ||
        !std::holds_alternative<RetTerm>(function.blocks.front().term))
      throw std::runtime_error("tensor-profile: Triton entry must have one block ending in ret");
    if (function.syms.size() != 0)
      throw std::runtime_error("tensor-profile: materialize all symbols before Triton emission");
    const bool tensorEntry =
        typeHasTensor(function.retType) || functionHasTensorAtom(function) ||
        std::any_of(function.params.begin(), function.params.end(), [](const ParamDecl &p) {
          return typeHasTensor(p.type);
        });
    if (!tensorEntry)
      throw std::runtime_error("tensor-profile: Triton entry must contain a tensor operation");
    auto validateType = [&](const TypePtr &type, const std::string &where) {
      if (!type)
        throw std::runtime_error("tensor-profile: missing type for " + where);
      if (const auto tensor = TypeUtils::asTensor(type)) {
        if (!tensor->elem || dtypeName(tensor->elem).empty())
          throw std::runtime_error("tensor-profile: unsupported tensor element type for " + where);
      } else if (dtypeName(type).empty()) {
        throw std::runtime_error("tensor-profile: unsupported scalar type for " + where);
      }
    };
    for (const auto &param: function.params)
      validateType(param.type, param.name.name);
    validateType(function.retType, "return value");
    for (const auto &let: function.lets)
      validateType(let.type, let.name.name);
    for (const auto &instruction: function.blocks.front().instrs) {
      if (std::holds_alternative<StoreInstr>(instruction))
        throw std::runtime_error("tensor-profile: Triton entry cannot use pointer stores");
      const bool hasCall = std::visit(
          [](const auto &ins) {
            using T = std::decay_t<decltype(ins)>;
            if constexpr (std::is_same_v<T, AssignInstr>)
              return expressionHasCall(ins.rhs);
            else if constexpr (std::is_same_v<T, StoreInstr>)
              return expressionHasCall(ins.ptr) || expressionHasCall(ins.val);
            else if constexpr (std::is_same_v<T, AssumeInstr> || std::is_same_v<T, RequireInstr>)
              return expressionHasCall(ins.cond.lhs) || expressionHasCall(ins.cond.rhs);
            return false;
          },
          instruction
      );
      if (hasCall)
        throw std::runtime_error("tensor-profile: Triton entry cannot contain calls");
    }
    const auto &ret = std::get<RetTerm>(function.blocks.front().term);
    if (!ret.value)
      throw std::runtime_error("tensor-profile: Triton entry must return a value");
    if (expressionHasCall(*ret.value))
      throw std::runtime_error("tensor-profile: Triton entry cannot return a call");
  }

  std::string TritonBackend::pythonName(const std::string &sirName) const {
    std::size_t begin = 0;
    std::string result;
    if (sirName.size() > 1 && sirName[0] == '%' && sirName[1] == '?') {
      result = "sym_";
      begin = 2;
    } else if (!sirName.empty() && (sirName[0] == '%' || sirName[0] == '@')) {
      begin = 1;
    }
    for (std::size_t i = begin; i < sirName.size(); ++i) {
      const unsigned char c = static_cast<unsigned char>(sirName[i]);
      result += std::isalnum(c) || c == '_' ? static_cast<char>(c) : '_';
    }
    return "_sir_" + result;
  }

  std::string TritonBackend::dtypeName(const TypePtr &type) const {
    if (const auto bits = TypeUtils::getIntBitWidth(type); bits && *bits == 32)
      return "tl.int32";
    if (const auto bits = TypeUtils::getFloatBitWidth(type); bits && *bits == 32)
      return "tl.float32";
    return {};
  }

  Shape TritonBackend::valueShape(const TypePtr &type) const {
    if (const auto tensor = TypeUtils::asTensor(type))
      return tensor->shape;
    return {};
  }

  TypePtr TritonBackend::valueType(const Coef &coef) const {
    if (const auto id = std::get_if<LocalOrSymId>(&coef)) {
      const auto it = types_.find(localId(*id));
      if (it == types_.end())
        throw std::runtime_error("tensor-profile: unresolved operand " + localId(*id));
      return it->second;
    }
    return nullptr;
  }

  void TritonBackend::addLine(std::string line) { body_.push_back(std::move(line)); }

  std::string TritonBackend::nextTemporary(const std::string &stem) {
    return "_sir_" + stem + "_tmp" + std::to_string(temporaryId_++);
  }

  void TritonBackend::emitOffsets(const std::string &name, const Shape &shape) {
    if (shape.empty())
      return;
    std::vector<std::string> terms;
    terms.reserve(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const auto axisName = name + "_d" + std::to_string(axis);
      std::string range = "tl.arange(0, " + std::to_string(shape[axis]) + ")";
      if (shape.size() > 1) {
        range += "[";
        for (std::size_t dim = 0; dim < shape.size(); ++dim) {
          if (dim)
            range += ", ";
          range += dim == axis ? ":" : "None";
        }
        range += "]";
      }
      addLine(axisName + " = " + range);
      const auto stride = productAfter(shape, axis);
      terms.push_back(axisName + (stride == 1 ? "" : " * " + std::to_string(stride)));
    }
    std::string offsets = name + "_offsets = ";
    for (std::size_t i = 0; i < terms.size(); ++i) {
      if (i)
        offsets += " + ";
      offsets += terms[i];
    }
    addLine(std::move(offsets));
  }

  void TritonBackend::emitInput(const ParamDecl &param) {
    const auto variable = pythonName(param.name.name);
    const auto pointer = variable.substr(5) + "_ptr";
    const auto shape = valueShape(param.type);
    types_[param.name.name] = param.type;
    emitOffsets(variable, shape);
    const auto address = shape.empty() ? pointer : pointer + " + " + variable + "_offsets";
    addLine(variable + " = tl.load(" + address + ")");
    values_[param.name.name] = Value{param.type, variable};
  }

  std::string TritonBackend::lowerCoef(const Coef &coef) {
    if (const auto integer = std::get_if<IntLit>(&coef))
      return std::to_string(integer->value);
    if (const auto floating = std::get_if<FloatLit>(&coef))
      return printFloat(floating->value);
    if (const auto id = std::get_if<LocalOrSymId>(&coef)) {
      const auto name = localId(*id);
      const auto it = values_.find(name);
      if (it == values_.end())
        throw std::runtime_error("tensor-profile: unresolved or uninitialized operand " + name);
      return it->second.expression;
    }
    throw std::runtime_error("tensor-profile: null is not a Triton scalar value");
  }

  std::string TritonBackend::lowerLValue(const LValue &lvalue) {
    if (!lvalue.accesses.empty())
      throw std::runtime_error("tensor-profile: indexed lvalues are not supported by Triton");
    const auto it = values_.find(lvalue.base.name);
    if (it == values_.end())
      throw std::runtime_error(
          "tensor-profile: unresolved or uninitialized local " + lvalue.base.name
      );
    return it->second.expression;
  }

  std::string TritonBackend::lowerExpr(const Expr &expr) {
    std::string result = lowerAtom(expr.first);
    for (const auto &tail: expr.rest)
      result = "(" + result + (tail.op == AddOp::Plus ? " + " : " - ") + lowerAtom(tail.atom) + ")";
    return result;
  }

  std::string TritonBackend::lowerAtom(const Atom &atom) {
    return std::visit(
        [&](const auto &arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, CoefAtom>) {
            return lowerCoef(arg.coef);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return lowerLValue(arg.rval);
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            return lowerTensorAtom(arg);
          } else if constexpr (std::is_same_v<T, OpAtom>) {
            if (arg.op == AtomOpKind::Mod)
              throw std::runtime_error(
                  "tensor-profile: scalar modulo is unsupported in Triton entry"
              );
            std::string op;
            switch (arg.op) {
              case AtomOpKind::Mul:
                op = "*";
                break;
              case AtomOpKind::Div:
                op = "/";
                break;
              case AtomOpKind::And:
                op = "&";
                break;
              case AtomOpKind::Or:
                op = "|";
                break;
              case AtomOpKind::Xor:
                op = "^";
                break;
              case AtomOpKind::Shl:
                op = "<<";
                break;
              case AtomOpKind::Shr:
              case AtomOpKind::LShr:
                op = ">>";
                break;
              case AtomOpKind::Mod:
                break;
            }
            return "(" + lowerCoef(arg.coef) + " " + op + " " + lowerLValue(arg.rval) + ")";
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return "(~" + lowerLValue(arg.rval) + ")";
          } else {
            throw std::runtime_error("tensor-profile: unsupported atom in Triton entry");
          }
        },
        atom.v
    );
  }

  std::string TritonBackend::lowerTensorAtom(const TensorAtom &atom) {
    if (!atom.resolvedType)
      throw std::runtime_error("tensor-profile: tensor atom is missing checked type metadata");
    std::vector<std::string> operands;
    operands.reserve(atom.operands.size());
    for (const auto &operand: atom.operands)
      operands.push_back(lowerCoef(operand));
    const auto elementType = TypeUtils::asTensor(atom.resolvedType)
                                 ? TypeUtils::asTensor(atom.resolvedType)->elem
                                 : atom.resolvedType;
    const bool isInteger = TypeUtils::getIntBitWidth(elementType).has_value();
    auto tupleText = [](const std::vector<std::uint32_t> &values) {
      Shape shape(values.begin(), values.end());
      return pythonShape(shape);
    };
    if (atom.op == TensorOp::Iota) {
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(atom.attribute.size()));
      const Shape shape(atom.attribute.begin(), atom.attribute.end());
      const auto range = nextTemporary("iota_axis");
      const auto reshaped = nextTemporary("iota_axis_shape");
      addLine(range + " = tl.arange(0, " + std::to_string(shape[axis]) + ")");
      Shape axisShape(shape.size(), 1);
      axisShape[axis] = shape[axis];
      addLine(reshaped + " = tl.reshape(" + range + ", " + pythonShape(axisShape) + ")");
      return "tl.broadcast_to(" + reshaped + ", " + pythonShape(shape) + ")";
    }
    if (atom.op == TensorOp::Extract) {
      const auto sourceType = valueType(atom.operands.front());
      const auto sourceShape = valueShape(sourceType);
      const auto offset = [&]() {
        if (sourceShape.size() != atom.attribute.size())
          throw std::runtime_error("tensor-profile: extraction rank differs from its source");
        std::uint32_t linear = 0;
        for (std::size_t i = 0; i < sourceShape.size(); ++i) {
          if (atom.attribute[i] >= sourceShape[i])
            throw std::runtime_error("tensor-profile: extraction coordinate is out of bounds");
          linear = linear * sourceShape[i] + atom.attribute[i];
        }
        return linear;
      }();
      const auto flat = nextTemporary("extract_flat");
      const auto index = nextTemporary("extract_index");
      const auto selected = nextTemporary("extract_selected");
      const auto scalar = nextTemporary("extract_scalar");
      addLine(
          flat + " = tl.reshape(" + operands.front() + ", (" +
          std::to_string(tensorElementCount(sourceShape)) + ",))"
      );
      addLine(index + " = tl.full((1,), " + std::to_string(offset) + ", tl.int32)");
      addLine(selected + " = tl.gather(" + flat + ", " + index + ", axis=0)");
      addLine(scalar + " = tl.reshape(" + selected + ", ())");
      return scalar;
    }
    if (atom.op == TensorOp::Expand) {
      const auto sourceType = valueType(atom.operands.front());
      const auto sourceRank = static_cast<std::uint32_t>(valueShape(sourceType).size());
      const auto axis = normalizeTensorAxis(atom.axis, sourceRank + 1);
      return "tl.expand_dims(" + operands.front() + ", axis=" + std::to_string(axis) + ")";
    }
    if (atom.op == TensorOp::Broadcast)
      return "tl.broadcast_to(" + operands.front() + ", " + tupleText(atom.attribute) + ")";
    if (atom.op == TensorOp::Transpose)
      return "tl.permute(" + operands.front() + ", " + tupleText(atom.attribute) + ")";
    if (atom.op == TensorOp::Add)
      return "(" + operands[0] + " + " + operands[1] + ")";
    if (atom.op == TensorOp::Sub)
      return "(" + operands[0] + " - " + operands[1] + ")";
    if (atom.op == TensorOp::Mul)
      return "(" + operands[0] + " * " + operands[1] + ")";
    if (atom.op == TensorOp::Div)
      return "(" + operands[0] + " / " + operands[1] + ")";
    if (atom.op == TensorOp::And)
      return "(" + operands[0] + " & " + operands[1] + ")";
    if (atom.op == TensorOp::Or)
      return "(" + operands[0] + " | " + operands[1] + ")";
    if (atom.op == TensorOp::Xor)
      return "(" + operands[0] + " ^ " + operands[1] + ")";
    if (atom.op == TensorOp::Min)
      return "tl.minimum(" + operands[0] + ", " + operands[1] + ")";
    if (atom.op == TensorOp::Max)
      return "tl.maximum(" + operands[0] + ", " + operands[1] + ")";
    if (atom.op == TensorOp::Exp)
      return "tl.exp(" + operands.front() + ")";
    if (atom.op == TensorOp::Sum || atom.op == TensorOp::ReduceMin ||
        atom.op == TensorOp::ReduceMax) {
      const auto sourceType = valueType(atom.operands.front());
      const auto sourceShape = valueShape(sourceType);
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(sourceShape.size()));
      const auto keep = atom.keepDims ? "True" : "False";
      if (atom.op == TensorOp::Sum) {
        std::string value = operands.front();
        if (isInteger)
          value = "(" + value + ").to(tl.int64)";
        auto sum =
            "tl.sum(" + value + ", axis=" + std::to_string(axis) + ", keep_dims=" + keep + ")";
        return isInteger ? "(" + sum + ").to(tl.int32)" : sum;
      }
      return std::string(atom.op == TensorOp::ReduceMin ? "tl.min(" : "tl.max(") +
             operands.front() + ", axis=" + std::to_string(axis) + ", keep_dims=" + keep + ")";
    }
    if (atom.op == TensorOp::Softmax) {
      const auto sourceType = valueType(atom.operands.front());
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(valueShape(sourceType).size()));
      if (softmaxLowering_ == "direct")
        return "tl.softmax(" + operands.front() + ", dim=" + std::to_string(axis) + ")";
      const auto maximum = nextTemporary("softmax_max");
      const auto shifted = nextTemporary("softmax_shifted");
      const auto exponentials = nextTemporary("softmax_exp");
      const auto denominator = nextTemporary("softmax_sum");
      addLine(
          maximum + " = tl.max(" + operands.front() + ", axis=" + std::to_string(axis) +
          ", keep_dims=True)"
      );
      addLine(shifted + " = " + operands.front() + " - " + maximum);
      addLine(exponentials + " = tl.exp(" + shifted + ")");
      addLine(
          denominator + " = tl.sum(" + exponentials + ", axis=" + std::to_string(axis) +
          ", keep_dims=True)"
      );
      return "(" + exponentials + " / " + denominator + ")";
    }
    (void) isInteger;
    throw std::runtime_error("tensor-profile: unsupported tensor operator for Triton");
  }

  void TritonBackend::emitInitializer(const LetDecl &let) {
    types_[let.name.name] = let.type;
    if (!let.init || let.init->kind == InitVal::Kind::Undef)
      return;
    const auto variable = pythonName(let.name.name);
    const auto shape = valueShape(let.type);
    const auto dtype =
        dtypeName(TypeUtils::asTensor(let.type) ? TypeUtils::asTensor(let.type)->elem : let.type);
    if (let.init->kind == InitVal::Kind::Aggregate) {
      if (!TypeUtils::asTensor(let.type))
        throw std::runtime_error("tensor-profile: only flat tensor initializers are supported");
      const auto &items = std::get<std::vector<InitValPtr>>(let.init->value);
      if (items.size() != tensorElementCount(shape))
        throw std::runtime_error("tensor-profile: tensor initializer size mismatch");
      const auto index = nextTemporary("init_index");
      const auto flat = nextTemporary("init_flat");
      addLine(index + " = tl.arange(0, " + std::to_string(items.size()) + ")");
      const auto literal = [&](const InitVal &item) {
        if (item.kind == InitVal::Kind::Int)
          return std::to_string(std::get<IntLit>(item.value).value);
        if (item.kind == InitVal::Kind::Float)
          return printFloat(std::get<FloatLit>(item.value).value);
        if (item.kind == InitVal::Kind::Local)
          return lowerLValue(LValue{std::get<LocalId>(item.value), {}, item.span});
        if (item.kind == InitVal::Kind::Sym)
          return lowerCoef(Coef{LocalOrSymId{std::get<SymId>(item.value)}});
        throw std::runtime_error("tensor-profile: unsupported tensor initializer leaf");
      };
      addLine(
          flat + " = tl.full((" + std::to_string(items.size()) + ",), " + literal(*items.front()) +
          ", " + dtype + ")"
      );
      auto current = flat;
      for (std::size_t i = 1; i < items.size(); ++i) {
        const auto next = nextTemporary("init_values");
        addLine(
            next + " = tl.where(" + index + " == " + std::to_string(i) + ", " + literal(*items[i]) +
            ", " + current + ")"
        );
        current = next;
      }
      addLine(variable + " = tl.reshape(" + current + ", " + pythonShape(shape) + ")");
    } else if (let.init->kind == InitVal::Kind::Int || let.init->kind == InitVal::Kind::Float) {
      std::string value = let.init->kind == InitVal::Kind::Int
                              ? std::to_string(std::get<IntLit>(let.init->value).value)
                              : printFloat(std::get<FloatLit>(let.init->value).value);
      if (TypeUtils::asTensor(let.type))
        addLine(variable + " = tl.full(" + pythonShape(shape) + ", " + value + ", " + dtype + ")");
      else
        addLine(variable + " = " + value);
    } else if (let.init->kind == InitVal::Kind::Local) {
      addLine(
          variable + " = " +
          lowerLValue(LValue{std::get<LocalId>(let.init->value), {}, let.init->span})
      );
    } else if (let.init->kind == InitVal::Kind::Sym) {
      addLine(variable + " = " + lowerCoef(Coef{LocalOrSymId{std::get<SymId>(let.init->value)}}));
    } else if (let.init->kind == InitVal::Kind::Atom) {
      addLine(variable + " = " + lowerAtom(*std::get<AtomPtr>(let.init->value)));
    } else {
      throw std::runtime_error("tensor-profile: unsupported local initializer");
    }
    values_[let.name.name] = Value{let.type, variable};
  }

  void TritonBackend::emitInstruction(const Instr &instruction) {
    std::visit(
        [&](const auto &ins) {
          using T = std::decay_t<decltype(ins)>;
          if constexpr (std::is_same_v<T, AssignInstr>) {
            if (!ins.lhs.accesses.empty())
              throw std::runtime_error(
                  "tensor-profile: Triton assignment must target a whole local"
              );
            const auto type = types_.find(ins.lhs.base.name);
            if (type == types_.end())
              throw std::runtime_error("tensor-profile: assignment target is not declared");
            const auto variable = pythonName(ins.lhs.base.name);
            addLine(variable + " = " + lowerExpr(ins.rhs));
            values_[ins.lhs.base.name] = Value{type->second, variable};
          } else if constexpr (std::is_same_v<T, AssumeInstr> || std::is_same_v<T, RequireInstr>) {
            // The reference validates constraints for concrete inputs.
          } else {
            throw std::runtime_error("tensor-profile: Triton entry cannot use pointer stores");
          }
        },
        instruction
    );
  }

  void TritonBackend::emitFunction(const FunDecl &function) {
    function_ = &function;
    values_.clear();
    types_.clear();
    body_.clear();
    temporaryId_ = 0;
    for (const auto &param: function.params)
      emitInput(param);
    for (const auto &let: function.lets)
      emitInitializer(let);
    for (const auto &instruction: function.blocks.front().instrs)
      emitInstruction(instruction);

    const auto &ret = std::get<RetTerm>(function.blocks.front().term);
    const auto result = lowerExpr(*ret.value);
    const auto returnShape = valueShape(function.retType);
    if (returnShape.empty()) {
      addLine("tl.store(out_ptr, " + result + ")");
    } else {
      emitOffsets("_sir_out", returnShape);
      addLine("tl.store(out_ptr + _sir_out_offsets, " + result + ")");
    }

    out_ << "import triton\n";
    out_ << "import triton.language as tl\n\n";
    out_ << "@triton.jit\n";
    out_ << "def kernel(";
    for (std::size_t i = 0; i < function.params.size(); ++i) {
      if (i)
        out_ << ", ";
      out_ << pythonName(function.params[i].name.name).substr(5) << "_ptr";
    }
    if (!function.params.empty())
      out_ << ", ";
    out_ << "out_ptr):\n";
    for (const auto &line: body_)
      out_ << "    " << line << "\n";
  }

  void TritonBackend::emit(const Program &program) {
    if (softmaxLowering_ != "direct" && softmaxLowering_ != "decomposed")
      throw std::runtime_error("tensor-profile: softmax lowering must be direct or decomposed");
    const auto &function = findEntry(program);
    validateEntry(function);
    emitFunction(function);
  }

} // namespace refractir
