#include "backend/triton_backend.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>

#include "analysis/cfg.hpp"
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

    std::string bufferKey(std::string name) {
      if (!name.empty() && (name.front() == '%' || name.front() == '@'))
        name.erase(name.begin());
      return name;
    }

    void validateBufferLayout(
        const TritonBufferLayout &layout, const Shape &shape, const std::string &where
    ) {
      if (shape.empty())
        throw std::runtime_error("tensor-profile: layout for " + where + " requires a tensor");
      if (layout.elementStrides.size() != shape.size())
        throw std::runtime_error("tensor-profile: layout rank does not match " + where);
      if (layout.axisOrder.size() != shape.size())
        throw std::runtime_error("tensor-profile: axis order rank does not match " + where);
      if (layout.storageSpan == 0)
        throw std::runtime_error("tensor-profile: storage span for " + where + " must be positive");
      if (std::any_of(layout.elementStrides.begin(), layout.elementStrides.end(),
                      [](std::uint32_t stride) { return stride == 0; }))
        throw std::runtime_error("tensor-profile: layout strides must be positive for " + where);

      std::vector<std::uint32_t> expectedOrder(shape.size());
      std::iota(expectedOrder.begin(), expectedOrder.end(), 0);
      std::stable_sort(expectedOrder.begin(), expectedOrder.end(), [&](std::uint32_t a,
                                                                       std::uint32_t b) {
        if (layout.elementStrides[a] != layout.elementStrides[b])
          return layout.elementStrides[a] < layout.elementStrides[b];
        return a < b;
      });
      auto suppliedOrder = layout.axisOrder;
      std::sort(suppliedOrder.begin(), suppliedOrder.end());
      std::vector<std::uint32_t> axes(shape.size());
      std::iota(axes.begin(), axes.end(), 0);
      if (suppliedOrder != axes || layout.axisOrder != expectedOrder)
        throw std::runtime_error(
            "tensor-profile: axis order must be a permutation ordered by increasing strides for " +
            where
        );

      std::uint64_t lastAddress = layout.elementOffset;
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto extent = static_cast<std::uint64_t>(shape[axis] - 1);
        const auto stride = static_cast<std::uint64_t>(layout.elementStrides[axis]);
        if (extent != 0 && stride > (std::numeric_limits<std::uint64_t>::max() - lastAddress) /
                                        extent)
          throw std::runtime_error("tensor-profile: storage span address overflows for " + where);
        lastAddress += extent * stride;
      }
      if (layout.elementOffset >= layout.storageSpan || lastAddress >= layout.storageSpan)
        throw std::runtime_error(
            "tensor-profile: storage span does not cover the last address for " + where
        );

      const auto elements = tensorElementCount(shape);
      std::set<std::uint64_t> addresses;
      for (std::uint32_t linear = 0; linear < elements; ++linear) {
        std::uint32_t remainder = linear;
        std::uint64_t address = layout.elementOffset;
        for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
          const auto axis = reverse - 1;
          const auto coordinate = remainder % shape[axis];
          remainder /= shape[axis];
          address += static_cast<std::uint64_t>(coordinate) * layout.elementStrides[axis];
        }
        if (!addresses.insert(address).second)
          throw std::runtime_error("tensor-profile: layout addresses overlap for " + where);
      }
    }

  } // namespace

  TritonBackend::TritonBackend(
      std::ostream &out, std::string entry, std::string softmaxLowering, TensorProfile profile,
      std::vector<TritonBufferLayout> layouts, TritonLoopOptions loops
  ) :
      out_(out), entry_(std::move(entry)), softmaxLowering_(std::move(softmaxLowering)),
      profile_(profile), layouts_(std::move(layouts)), loops_(std::move(loops)) {}

  std::optional<std::string> parseTritonLoopHint(const std::string &text) {
    if (text == "flatten")
      return "flatten=True";
    for (const std::string key: {"num_stages", "loop_unroll_factor"}) {
      const auto prefix = key + "=";
      if (text.size() == prefix.size() + 1 && text.starts_with(prefix) && text.back() >= '1' &&
          text.back() <= '8')
        return text;
    }
    return std::nullopt;
  }

  const FunDecl &TritonBackend::findEntry(const Program &program) const {
    auto it = std::find_if(program.funs.begin(), program.funs.end(), [&](const FunDecl &function) {
      return function.name.name == "@" + entry_ || function.name.name == entry_;
    });
    if (it == program.funs.end())
      throw std::runtime_error("tensor-profile: entry function was not found: " + entry_);
    return *it;
  }

  void TritonBackend::validateEntry(const FunDecl &function) const {
    if (profile_.id == TensorProfileId::V1) {
      if (function.blocks.size() != 1 ||
          !std::holds_alternative<RetTerm>(function.blocks.front().term))
        throw std::runtime_error("tensor-profile: Triton entry must have one block ending in ret");
    } else {
      DiagBag loopDiagnostics;
      if (!analyzeTensorLoops(function, profile_, loopDiagnostics))
        throw std::runtime_error(
            loopDiagnostics.diags.empty() ? "tensor-profile: invalid v2 control flow"
                                          : loopDiagnostics.diags.front().message
        );
    }
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
        try {
          (void) tensorExecutionTileShape(tensor->shape, profile_);
        } catch (const std::exception &error) {
          throw std::runtime_error(std::string(error.what()) + " for " + where);
        }
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
    bool hasReturn = false;
    for (const auto &block: function.blocks) {
      for (const auto &instruction: block.instrs) {
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
      if (const auto *branch = std::get_if<BrTerm>(&block.term); branch && branch->cond &&
          (expressionHasCall(branch->cond->lhs) || expressionHasCall(branch->cond->rhs)))
        throw std::runtime_error("tensor-profile: Triton entry cannot contain calls");
      if (const auto *ret = std::get_if<RetTerm>(&block.term)) {
        if (!ret->value)
          throw std::runtime_error("tensor-profile: Triton entry must return a value");
        if (expressionHasCall(*ret->value))
          throw std::runtime_error("tensor-profile: Triton entry cannot return a call");
        hasReturn = true;
      }
    }
    if (!hasReturn)
      throw std::runtime_error("tensor-profile: Triton entry must return a value");

    std::set<std::pair<TritonBufferRole, std::string>> seenLayouts;
    for (const auto &layout: layouts_) {
      const auto key = bufferKey(layout.name);
      const auto parameter = std::find_if(
          function.params.begin(), function.params.end(), [&](const ParamDecl &param) {
            return bufferKey(param.name.name) == key;
          }
      );
      const bool hasInput = parameter != function.params.end();
      const bool hasOutput = key == "return" && typeHasTensor(function.retType);
      auto role = layout.role;
      if (role == TritonBufferRole::Unspecified)
        role = hasInput ? TritonBufferRole::Input
                        : hasOutput ? TritonBufferRole::Output
                                    : TritonBufferRole::Unspecified;
      if (role == TritonBufferRole::Unspecified ||
          (role == TritonBufferRole::Input && !hasInput) ||
          (role == TritonBufferRole::Output && !hasOutput))
        throw std::runtime_error("tensor-profile: layout names unknown buffer " + layout.name);
      if (!seenLayouts.emplace(role, key).second)
        throw std::runtime_error("tensor-profile: duplicate layout for buffer " + layout.name);
      const Shape shape = role == TritonBufferRole::Output
                              ? valueShape(function.retType)
                              : valueShape(parameter->type);
      validateBufferLayout(layout, shape, layout.name);
    }
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

  std::string TritonBackend::valueValidity(const Coef &coef) const {
    if (const auto id = std::get_if<LocalOrSymId>(&coef)) {
      const auto it = values_.find(localId(*id));
      if (it != values_.end())
        return it->second.validity;
    }
    return {};
  }

  void TritonBackend::addLine(std::string line) {
    line.insert(0, indent_ * 4, ' ');
    body_.push_back(std::move(line));
  }

  std::string TritonBackend::freshPythonName(const std::string &base) {
    std::string candidate = base;
    for (std::uint32_t suffix = 1; usedPythonNames_.contains(candidate); ++suffix)
      candidate = base + "_" + std::to_string(suffix);
    usedPythonNames_.insert(candidate);
    return candidate;
  }

  // SIR resolves an equal-magnitude signed-zero tie in min and max with IEEE
  // minNum/maxNum: -0.0 forces the minimum and +0.0 the maximum. Triton leaves
  // that tie unspecified, so its interpreter and compiled paths may disagree.
  // Under round-to-nearest, +0.0 + -0.0 = +0.0 and -0.0 + -0.0 = -0.0, so the
  // sum of two zeros is the maximum's tie-break and the negated sum of their
  // negations is the minimum's; every other pair keeps the Triton call.
  std::string TritonBackend::signedZeroExtremum(
      bool minimum, const std::string &left, const std::string &right
  ) {
    const auto a = nextTemporary("extremum_left");
    const auto b = nextTemporary("extremum_right");
    addLine(a + " = " + left);
    addLine(b + " = " + right);
    const auto tie = minimum ? "-((-" + a + ") + (-" + b + "))" : "(" + a + " + " + b + ")";
    return "tl.where((" + a + " == 0.0) & (" + b + " == 0.0), " + tie + ", " +
           (minimum ? "tl.minimum(" : "tl.maximum(") + a + ", " + b + "))";
  }

  // The reduction form of the same tie-break. When the extreme value is zero,
  // its sign is decided by whether any zero lane has the winning sign: +0.0
  // for a maximum, -0.0 for a minimum. The sign is read from the lane's bits,
  // and an integer maximum over a 0/1 marker combines the lanes; summing the
  // zero lanes instead would depend on the identity a reduction starts from.
  std::string TritonBackend::signedZeroReduction(
      bool minimum, const std::string &lanes, std::uint32_t axis, const char *keep
  ) {
    const auto values = nextTemporary("extrema_lanes");
    const auto extreme = nextTemporary("extrema");
    const auto winning = nextTemporary("extrema_winning_zero");
    const auto arguments = ", axis=" + std::to_string(axis) + ", keep_dims=" + keep + ")";
    addLine(values + " = " + lanes);
    addLine(extreme + " = " + (minimum ? "tl.min(" : "tl.max(") + values + arguments);
    const auto sign = std::string(minimum ? " < 0" : " >= 0");
    addLine(
        winning + " = tl.max(tl.where((" + values + " == 0.0) & (" + values +
        ".to(tl.int32, bitcast=True)" + sign + "), 1, 0)" + arguments
    );
    const auto zero = minimum ? "tl.where(" + winning + " > 0, -0.0, 0.0)"
                              : "tl.where(" + winning + " > 0, 0.0, -0.0)";
    return "tl.where(" + extreme + " == 0.0, " + zero + ", " + extreme + ")";
  }

  std::string TritonBackend::nextTemporary(const std::string &stem) {
    return freshPythonName("_sir_" + stem + "_tmp" + std::to_string(temporaryId_++));
  }

  std::string TritonBackend::emitOffsets(
      const std::string &name, const Shape &shape, const TritonBufferLayout *layout,
      std::string *validity, bool includeOffsets
  ) {
    if (shape.empty())
      return {};
    const auto tileShape = tensorExecutionTileShape(shape, profile_);
    std::vector<std::string> terms;
    terms.reserve(shape.size());
    std::vector<std::string> validTerms;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const auto axisName = freshPythonName(name + "_d" + std::to_string(axis));
      std::string range = "tl.arange(0, " + std::to_string(tileShape[axis]) + ")";
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
      if (includeOffsets) {
        const auto stride = layout ? layout->elementStrides[axis] : productAfter(shape, axis);
        terms.push_back(axisName + (stride == 1 ? "" : " * " + std::to_string(stride)));
        if (layout && axis == 0 && layout->elementOffset != 0)
          terms.insert(terms.begin(), std::to_string(layout->elementOffset));
      }
      if (profile_.id == TensorProfileId::V3)
        validTerms.push_back(axisName + " < " + std::to_string(shape[axis]));
    }
    if (validity) {
      if (validTerms.empty()) {
        *validity = {};
      } else {
        *validity = freshPythonName(name + "_valid");
        std::string expression;
        for (std::size_t i = 0; i < validTerms.size(); ++i) {
          if (i)
            expression += " & ";
          expression += "(" + validTerms[i] + ")";
        }
        addLine(*validity + " = " + expression);
      }
    }
    if (!includeOffsets)
      return {};
    const auto offsetName = freshPythonName(name + "_offsets");
    std::string offsets = offsetName + " = ";
    for (std::size_t i = 0; i < terms.size(); ++i) {
      if (i)
        offsets += " + ";
      offsets += terms[i];
    }
    addLine(std::move(offsets));
    return offsetName;
  }

  std::string TritonBackend::emitValidity(const std::string &name, const Shape &shape) {
    if (profile_.id != TensorProfileId::V3 || shape.empty())
      return {};
    std::string validity;
    (void) emitOffsets(name, shape, nullptr, &validity, false);
    return validity;
  }

  void TritonBackend::emitInput(const ParamDecl &param) {
    const auto variable = pythonName(param.name.name);
    const auto pointer = freshPythonName(variable.substr(5) + "_ptr");
    parameterPointers_[param.name.name] = pointer;
    const auto shape = valueShape(param.type);
    types_[param.name.name] = param.type;
    std::string validity;
    parameterOffsets_[param.name.name] =
        emitOffsets(
            variable, shape, findLayout(param.name.name, TritonBufferRole::Input), &validity
        );
    values_[param.name.name] = Value{param.type, {}, validity};
  }

  const TritonBufferLayout *TritonBackend::findLayout(
      const std::string &sirName, TritonBufferRole role
  ) const {
    const auto key = bufferKey(sirName);
    const auto find = [&](TritonBufferRole candidateRole) {
      return std::find_if(layouts_.begin(), layouts_.end(), [&](const auto &layout) {
        return bufferKey(layout.name) == key && layout.role == candidateRole;
      });
    };
    auto it = find(role);
    if (it != layouts_.end())
      return &*it;
    if (role != TritonBufferRole::Input && role != TritonBufferRole::Output)
      return nullptr;

    it = find(TritonBufferRole::Unspecified);
    if (it == layouts_.end())
      return nullptr;
    if (role == TritonBufferRole::Output && function_ &&
        std::any_of(function_->params.begin(), function_->params.end(), [&](const auto &param) {
          return bufferKey(param.name.name) == key;
        }))
      return nullptr;
    return &*it;
  }

  std::string TritonBackend::lowerCoef(const Coef &coef) {
    if (const auto integer = std::get_if<IntLit>(&coef))
      return std::to_string(integer->value);
    if (const auto floating = std::get_if<FloatLit>(&coef))
      return printFloat(floating->value);
    if (const auto id = std::get_if<LocalOrSymId>(&coef)) {
      const auto name = localId(*id);
      if (const auto pointer = parameterPointers_.find(name); pointer != parameterPointers_.end()) {
        // Keep loads at their SIR use site so loads inside counted loops remain
        // inside the generated loop and exercise the same reduction structure.
        const auto shape = valueShape(types_.at(name));
        const auto offsets = parameterOffsets_.at(name);
        const auto address = shape.empty() ? pointer->second : pointer->second + " + " + offsets;
        const auto loaded = nextTemporary("input_load");
        if (profile_.id == TensorProfileId::V3 && !shape.empty()) {
          const auto validity = values_.at(name).validity;
          const auto elementType = TypeUtils::asTensor(types_.at(name))->elem;
          const auto fallback = TypeUtils::getFloatBitWidth(elementType) ? "0.0" : "0";
          addLine(
              loaded + " = tl.load(" + address + ", mask=" + validity + ", other=" + fallback + ")"
          );
        } else {
          addLine(loaded + " = tl.load(" + address + ")");
        }
        return loaded;
      }
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
    return lowerCoef(Coef{LocalOrSymId{lvalue.base}});
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
            const auto lhs = lowerCoef(arg.coef);
            const auto rhs = lowerLValue(arg.rval);
            if (arg.op == AtomOpKind::LShr)
              return "tl.cast((tl.cast(" + lhs + ", tl.uint32) >> " + rhs + "), tl.int32)";
            std::string op;
            switch (arg.op) {
              case AtomOpKind::Mul:
                op = "*";
                break;
              case AtomOpKind::Div:
                op = TypeUtils::getIntBitWidth(types_.at(arg.rval.base.name)).has_value() ? "//"
                                                                                          : "/";
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
                op = ">>";
                break;
              case AtomOpKind::LShr:
                break;
              case AtomOpKind::Mod:
                break;
            }
            return "(" + lhs + " " + op + " " + rhs + ")";
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
    if (atom.op == TensorOp::Window) {
      const auto *named = std::get_if<LocalOrSymId>(&atom.operands[0]);
      const auto *local = named ? std::get_if<LocalId>(named) : nullptr;
      if (local == nullptr)
        throw std::runtime_error("tensor-window: source is not a parameter");
      const auto sourceName = localId(*named);
      const auto pointer = parameterPointers_.find(sourceName);
      const auto sourceType = types_.find(sourceName);
      if (pointer == parameterPointers_.end() || sourceType == types_.end())
        throw std::runtime_error("tensor-window: source is not a tensor parameter");
      const auto sourceShape = valueShape(sourceType->second);
      if (sourceShape.empty())
        throw std::runtime_error("tensor-window: source has no leading extent");
      const auto *layout = findLayout(sourceName, TritonBufferRole::Input);
      if (atom.attribute.size() != 3)
        throw std::runtime_error("tensor-window: expected [rows, stride, active]");
      const auto rows = atom.attribute[0];
      const auto stride = atom.attribute[1];
      const auto active = atom.attribute[2];
      Shape resultShape = sourceShape;
      resultShape.front() = rows;
      const auto resultTile = tensorExecutionTileShape(resultShape, profile_);
      const auto rawIndex = nextTemporary("window_index");
      addLine(rawIndex + " = " + lowerCoef(atom.operands[1]) + " + tl.full((), 0, tl.int32)");
      const auto index64 = nextTemporary("window_index64");
      addLine(index64 + " = " + rawIndex + ".to(tl.int64)");

      std::string rowRange = "tl.arange(0, " + std::to_string(resultTile[0]) + ")";
      if (resultShape.size() > 1) {
        rowRange += "[";
        for (std::size_t dim = 0; dim < resultShape.size(); ++dim) {
          if (dim)
            rowRange += ", ";
          rowRange += dim == 0 ? ":" : "None";
        }
        rowRange += "]";
      }
      const auto rowName = nextTemporary("window_row");
      addLine(rowName + " = " + rowRange);
      const auto linear = nextTemporary("window_lin");
      addLine(
          linear + " = " + index64 + " * " + std::to_string(stride) + " + " + rowName +
          ".to(tl.int64)"
      );

      std::vector<std::string> maskTerms;
      if (profile_.id == TensorProfileId::V3)
        maskTerms.push_back("(" + rowName + " < " + std::to_string(rows) + ")");
      maskTerms.push_back("(" + linear + " < " + std::to_string(active) + ")");

      const auto stride0 = layout ? layout->elementStrides[0] : productAfter(sourceShape, 0);
      std::string address = linear + " * " + std::to_string(stride0);
      for (std::size_t axis = 1; axis < sourceShape.size(); ++axis) {
        std::string range = "tl.arange(0, " + std::to_string(resultTile[axis]) + ")";
        range += "[";
        for (std::size_t dim = 0; dim < sourceShape.size(); ++dim) {
          if (dim)
            range += ", ";
          range += dim == axis ? ":" : "None";
        }
        range += "]";
        const auto coordName = nextTemporary("window_d" + std::to_string(axis));
        addLine(coordName + " = " + range);
        if (profile_.id == TensorProfileId::V3)
          maskTerms.push_back(
              "(" + coordName + " < " + std::to_string(sourceShape[axis]) + ")"
          );
        const auto axisStride =
            layout ? layout->elementStrides[axis] : productAfter(sourceShape, axis);
        address += " + " + coordName + " * " + std::to_string(axisStride);
      }
      if (layout && layout->elementOffset != 0)
        address = std::to_string(layout->elementOffset) + " + " + address;
      std::string mask = maskTerms.front();
      for (std::size_t i = 1; i < maskTerms.size(); ++i)
        mask += " & " + maskTerms[i];
      const auto maskName = nextTemporary("window_mask");
      addLine(maskName + " = " + mask);
      std::string fill;
      if (atom.fill && std::holds_alternative<FloatLit>(*atom.fill))
        fill = printFloat(std::get<FloatLit>(*atom.fill).value);
      else if (atom.fill && std::holds_alternative<IntLit>(*atom.fill))
        fill = std::to_string(std::get<IntLit>(*atom.fill).value);
      else
        throw std::runtime_error("tensor-window: fill is not a literal");
      const auto value = nextTemporary("window");
      addLine(
          value + " = tl.load(" + pointer->second + " + " + address + ", mask=" + maskName +
          ", other=" + fill + ")"
      );
      return value;
    }

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
      const auto tileShape = tensorExecutionTileShape(shape, profile_);
      const auto range = nextTemporary("iota_axis");
      const auto reshaped = nextTemporary("iota_axis_shape");
      addLine(range + " = tl.arange(0, " + std::to_string(tileShape[axis]) + ")");
      Shape axisShape(shape.size(), 1);
      axisShape[axis] = tileShape[axis];
      addLine(reshaped + " = tl.reshape(" + range + ", " + pythonShape(axisShape) + ")");
      return "tl.broadcast_to(" + reshaped + ", " + pythonShape(tileShape) + ")";
    }
    if (atom.op == TensorOp::Extract) {
      const auto sourceType = valueType(atom.operands.front());
      const auto sourceShape = valueShape(sourceType);
      const auto sourceTileShape = tensorExecutionTileShape(sourceShape, profile_);
      const auto offset = [&]() {
        if (sourceShape.size() != atom.attribute.size())
          throw std::runtime_error("tensor-profile: extraction rank differs from its source");
        std::uint32_t linear = 0;
        for (std::size_t i = 0; i < sourceShape.size(); ++i) {
          if (atom.attribute[i] >= sourceShape[i])
            throw std::runtime_error("tensor-profile: extraction coordinate is out of bounds");
          linear = linear * sourceTileShape[i] + atom.attribute[i];
        }
        return linear;
      }();
      const auto flat = nextTemporary("extract_flat");
      const auto index = nextTemporary("extract_index");
      const auto selected = nextTemporary("extract_selected");
      const auto scalar = nextTemporary("extract_scalar");
      addLine(
          flat + " = tl.reshape(" + operands.front() + ", (" +
          std::to_string(tensorElementCount(sourceTileShape)) + ",))"
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
    if (atom.op == TensorOp::Broadcast) {
      const Shape shape(atom.attribute.begin(), atom.attribute.end());
      const auto targetTileShape = tensorExecutionTileShape(shape, profile_);
      const auto sourceType = valueType(atom.operands.front());
      const auto sourceShape = valueShape(sourceType);
      std::string operand = operands.front();
      if (sourceShape.size() < shape.size()) {
        // The Triton primitive takes one rank on both sides, so a source of
        // lower rank gains leading singleton axes before the call. The SIR rule
        // is the right-aligned broadcast these axes state. A rank-zero value is
        // a Triton scalar, which has no block rank to reshape, so a full block
        // of that value supplies the rank instead.
        Shape aligned(shape.size() - sourceShape.size(), 1);
        const auto reshaped = nextTemporary("broadcast_rank");
        if (sourceShape.empty()) {
          addLine(
              reshaped + " = tl.full(" + pythonShape(aligned) + ", " + operand + ", " +
              (isInteger ? "tl.int32" : "tl.float32") + ")"
          );
        } else {
          const auto sourceTileShape = tensorExecutionTileShape(sourceShape, profile_);
          aligned.insert(aligned.end(), sourceTileShape.begin(), sourceTileShape.end());
          addLine(reshaped + " = tl.reshape(" + operand + ", " + pythonShape(aligned) + ")");
        }
        operand = reshaped;
      }
      return "tl.broadcast_to(" + operand + ", " + pythonShape(targetTileShape) + ")";
    }
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
    if (atom.op == TensorOp::Rem)
      return "(" + operands[0] + " % " + operands[1] + ")";
    if (atom.op == TensorOp::And)
      return "(" + operands[0] + " & " + operands[1] + ")";
    if (atom.op == TensorOp::Or)
      return "(" + operands[0] + " | " + operands[1] + ")";
    if (atom.op == TensorOp::Xor)
      return "(" + operands[0] + " ^ " + operands[1] + ")";
    if (atom.op == TensorOp::Min || atom.op == TensorOp::Max) {
      const bool minimum = atom.op == TensorOp::Min;
      if (!isInteger)
        return signedZeroExtremum(minimum, operands[0], operands[1]);
      return std::string(minimum ? "tl.minimum(" : "tl.maximum(") + operands[0] + ", " +
             operands[1] + ")";
    }
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
        if (profile_.id == TensorProfileId::V3) {
          const auto validity = valueValidity(atom.operands.front());
          if (validity.empty())
            throw std::runtime_error("tensor-profile: v3 reduction operand lacks validity");
          const auto fallback = TypeUtils::getFloatBitWidth(elementType) ? "0.0" : "0";
          value = "tl.where(" + validity + ", " + value + ", " + fallback + ")";
        }
        if (isInteger)
          value = "(" + value + ").to(tl.int64)";
        auto sum =
            "tl.sum(" + value + ", axis=" + std::to_string(axis) + ", keep_dims=" + keep + ")";
        return isInteger ? "(" + sum + ").to(tl.int32)" : sum;
      }
      std::string value = operands.front();
      const bool minimum = atom.op == TensorOp::ReduceMin;
      if (profile_.id == TensorProfileId::V3) {
        // An inactive lane must not win an extrema reduction, so the mask
        // replaces it with the identity of the operation before the call.
        const auto validity = valueValidity(atom.operands.front());
        if (validity.empty())
          throw std::runtime_error("tensor-profile: v3 reduction operand lacks validity");
        std::string neutral;
        if (isInteger)
          neutral = minimum ? "2147483647" : "-2147483648";
        else
          neutral = minimum ? "float(\"inf\")" : "-float(\"inf\")";
        value = "tl.where(" + validity + ", " + value + ", " + neutral + ")";
      }
      if (!isInteger)
        return signedZeroReduction(minimum, value, axis, keep);
      return std::string(minimum ? "tl.min(" : "tl.max(") + value +
             ", axis=" + std::to_string(axis) + ", keep_dims=" + keep + ")";
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
    const auto tileShape = shape.empty() ? Shape{} : tensorExecutionTileShape(shape, profile_);
    const auto dtype =
        dtypeName(TypeUtils::asTensor(let.type) ? TypeUtils::asTensor(let.type)->elem : let.type);
    // A scalar initializer reaches the declared tensor shape before any
    // shape-sensitive operator; a tensor initializer is a copy. `tl.full`
    // takes a scalar fill value (Triton's `splat` asserts on a block tensor),
    // so a tensor source must never become the fill value.
    const auto bindMaterialized = [&](const std::string &value, bool sourceIsTensor) {
      if (TypeUtils::asTensor(let.type) && !sourceIsTensor)
        addLine(
            variable + " = tl.full(" + pythonShape(tileShape) + ", " + value + ", " + dtype + ")"
        );
      else
        addLine(variable + " = " + value);
    };
    const auto initializerSourceIsTensor = [&](const std::string &name) {
      if (const auto it = types_.find(name); it != types_.end())
        return TypeUtils::asTensor(it->second) != nullptr;
      if (function_)
        for (const auto &sym: function_->syms)
          if (sym.name.name == name)
            return TypeUtils::asTensor(sym.type) != nullptr;
      return false;
    };
    if (let.init->kind == InitVal::Kind::Aggregate) {
      if (!TypeUtils::asTensor(let.type))
        throw std::runtime_error("tensor-profile: only flat tensor initializers are supported");
      const auto &items = std::get<std::vector<InitValPtr>>(let.init->value);
      if (items.size() != tensorElementCount(shape))
        throw std::runtime_error("tensor-profile: tensor initializer size mismatch");
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
      if (profile_.id == TensorProfileId::V3) {
        const auto tileElements = tensorElementCount(tileShape);
        const auto index = nextTemporary("init_index");
        auto current = nextTemporary("init_flat");
        addLine(index + " = tl.arange(0, " + std::to_string(tileElements) + ")");
        addLine(
            current + " = tl.full((" + std::to_string(tileElements) + ",), 0, " + dtype + ")"
        );
        for (std::size_t i = 0; i < items.size(); ++i) {
          std::uint32_t remaining = static_cast<std::uint32_t>(i);
          std::uint32_t tileIndex = 0;
          for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            const auto coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            tileIndex += coordinate * productAfter(tileShape, axis);
          }
          const auto next = nextTemporary("init_values");
          addLine(
              next + " = tl.where(" + index + " == " + std::to_string(tileIndex) + ", " +
              literal(*items[i]) + ", " + current + ")"
          );
          current = next;
        }
        addLine(variable + " = tl.reshape(" + current + ", " + pythonShape(tileShape) + ")");
      } else {
        const auto index = nextTemporary("init_index");
        const auto flat = nextTemporary("init_flat");
        addLine(index + " = tl.arange(0, " + std::to_string(items.size()) + ")");
        addLine(
            flat + " = tl.full((" + std::to_string(items.size()) + ",), " + literal(*items.front()) +
            ", " + dtype + ")"
        );
        auto current = flat;
        for (std::size_t i = 1; i < items.size(); ++i) {
          const auto next = nextTemporary("init_values");
          addLine(
              next + " = tl.where(" + index + " == " + std::to_string(i) + ", " +
              literal(*items[i]) + ", " + current + ")"
          );
          current = next;
        }
        addLine(variable + " = tl.reshape(" + current + ", " + pythonShape(shape) + ")");
      }
    } else if (let.init->kind == InitVal::Kind::Int || let.init->kind == InitVal::Kind::Float) {
      std::string value = let.init->kind == InitVal::Kind::Int
                              ? std::to_string(std::get<IntLit>(let.init->value).value)
                              : printFloat(std::get<FloatLit>(let.init->value).value);
      bindMaterialized(value, false);
    } else if (let.init->kind == InitVal::Kind::Local) {
      const auto name = std::get<LocalId>(let.init->value).name;
      bindMaterialized(lowerLValue(LValue{std::get<LocalId>(let.init->value), {}, let.init->span}),
                       initializerSourceIsTensor(name));
    } else if (let.init->kind == InitVal::Kind::Sym) {
      const auto name = std::get<SymId>(let.init->value).name;
      bindMaterialized(lowerCoef(Coef{LocalOrSymId{std::get<SymId>(let.init->value)}}),
                       initializerSourceIsTensor(name));
    } else if (let.init->kind == InitVal::Kind::Atom) {
      addLine(variable + " = " + lowerAtom(*std::get<AtomPtr>(let.init->value)));
    } else {
      throw std::runtime_error("tensor-profile: unsupported local initializer");
    }
    const auto validity = emitValidity(variable + "_initializer", shape);
    if (!validity.empty()) {
      const auto element = TypeUtils::asTensor(let.type)->elem;
      const auto fallback = TypeUtils::getFloatBitWidth(element) ? "0.0" : "0";
      addLine(variable + " = tl.where(" + validity + ", " + variable + ", " + fallback + ")");
    }
    values_[let.name.name] = Value{let.type, variable, validity};
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
            auto expression = lowerExpr(ins.rhs);
            const auto shape = valueShape(type->second);
            const auto validity = emitValidity(variable + "_result", shape);
            if (!validity.empty()) {
              const auto element = TypeUtils::asTensor(type->second)->elem;
              const auto fallback = TypeUtils::getFloatBitWidth(element) ? "0.0" : "0";
              expression = "tl.where(" + validity + ", " + expression + ", " + fallback + ")";
            }
            addLine(variable + " = " + expression);
            values_[ins.lhs.base.name] = Value{type->second, variable, validity};
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
    parameterPointers_.clear();
    parameterOffsets_.clear();
    usedPythonNames_.clear();
    boundParameters_.clear();
    temporaryId_ = 0;
    indent_ = 0;
    for (const auto &param: function.params)
      usedPythonNames_.insert(pythonName(param.name.name));
    for (const auto &let: function.lets)
      usedPythonNames_.insert(pythonName(let.name.name));
    usedPythonNames_.insert("out_ptr");
    for (const auto &param: function.params)
      emitInput(param);
    for (const auto &let: function.lets)
      emitInitializer(let);

    DiagBag diagnostics;
    const CFG cfg = CFG::build(function, diagnostics);
    const auto analysis = analyzeTensorLoops(function, profile_, diagnostics);
    if (!analysis)
      throw std::runtime_error("tensor-profile: failed to analyze Triton entry control flow");
    std::unordered_map<std::size_t, std::size_t> loopAtPreheader;
    std::unordered_map<std::size_t, std::size_t> loopAtLatch;
    for (std::size_t i = 0; i < analysis->loops.size(); ++i) {
      loopAtPreheader.emplace(analysis->loops[i].preheader, i);
      loopAtLatch.emplace(analysis->loops[i].latch, i);
    }

    auto emitReturn = [&](const RetTerm &ret) {
      if (!ret.value)
        throw std::runtime_error("tensor-profile: Triton entry must return a value");
      const auto result = lowerExpr(*ret.value);
      const auto returnShape = valueShape(function.retType);
      if (returnShape.empty()) {
        addLine("tl.store(out_ptr, " + result + ")");
      } else {
        std::string validity;
        const auto offsets = emitOffsets(
            "_sir_out", returnShape, findLayout("return", TritonBufferRole::Output), &validity
        );
        std::string store = "tl.store(out_ptr + " + offsets + ", " + result;
        if (!validity.empty())
          store += ", mask=" + validity;
        addLine(store + ")");
      }
    };

    std::function<void(std::size_t, std::optional<std::size_t>, int)> emitFlow;
    std::function<void(std::size_t)> emitLoop;
    emitLoop = [&](std::size_t loopIndex) {
      const auto &loop = analysis->loops[loopIndex];
      const auto &header = function.blocks[loop.header];
      const auto &branch = std::get<BrTerm>(header.term);
      const auto body = cfg.indexOf.at(branch.thenLabel.name);
      const auto variable = pythonName(loop.inductionVariable);
      const auto type = types_.at(loop.inductionVariable);
      std::string bound = std::to_string(loop.bound);
      if (loops_.bound == TritonLoopBound::Runtime) {
        bound = freshPythonName(variable + "_bound");
        boundParameters_.emplace_back(bound, loop.bound);
      }
      addLine(variable + " = tl.full((), " + std::to_string(loop.initialValue) + ", " +
              dtypeName(type) + ")");
      const auto hint = loops_.hint.empty() ? "" : ", " + loops_.hint;
      addLine(
          "for " + variable + " in tl.range(" + std::to_string(loop.initialValue) + ", " + bound +
          hint + "):"
      );
      const auto bodyStartLine = body_.size();
      ++indent_;
      emitFlow(body, loop.exit, static_cast<int>(loopIndex));
      if (body_.size() == bodyStartLine)
        addLine("pass");
      --indent_;
      addLine(variable + " = tl.full((), " + bound + ", " + dtypeName(type) + ")");
      values_[loop.inductionVariable] = Value{type, variable, {}};
    };

    emitFlow = [&](
                  std::size_t start, std::optional<std::size_t> stop, int activeLoop
              ) {
      std::size_t blockIndex = start;
      std::size_t steps = 0;
      while (true) {
        if (stop && blockIndex == *stop)
          return;
        if (++steps > cfg.blocks.size() + analysis->loops.size() + 1)
          throw std::runtime_error("tensor-profile: unstructured cycle in Triton entry");

        const auto latch = loopAtLatch.find(blockIndex);
        if (activeLoop >= 0 && latch != loopAtLatch.end() &&
            latch->second == static_cast<std::size_t>(activeLoop)) {
          const auto &loop = analysis->loops[static_cast<std::size_t>(activeLoop)];
          for (const auto &instruction: function.blocks[blockIndex].instrs) {
            const auto *assignment = std::get_if<AssignInstr>(&instruction);
            if (assignment && assignment->lhs.accesses.empty() &&
                assignment->lhs.base.name == loop.inductionVariable)
              continue;
            emitInstruction(instruction);
          }
          return;
        }

        for (const auto &instruction: function.blocks[blockIndex].instrs)
          emitInstruction(instruction);

        if (const auto nested = loopAtPreheader.find(blockIndex);
            nested != loopAtPreheader.end()) {
          emitLoop(nested->second);
          blockIndex = analysis->loops[nested->second].exit;
          continue;
        }

        const auto &term = function.blocks[blockIndex].term;
        if (const auto *ret = std::get_if<RetTerm>(&term)) {
          emitReturn(*ret);
          return;
        }
        if (const auto *branch = std::get_if<BrTerm>(&term)) {
          if (branch->isConditional)
            throw std::runtime_error(
                "tensor-profile: only counted-loop headers may branch conditionally"
            );
          const auto destination = cfg.indexOf.at(branch->dest.name);
          if (stop && destination == *stop)
            return;
          blockIndex = destination;
          continue;
        }
        throw std::runtime_error("tensor-profile: unreachable terminator in Triton entry");
      }
    };
    emitFlow(cfg.entry, std::nullopt, -1);

    out_ << "import triton\n";
    out_ << "import triton.language as tl\n\n";
    // A specialized integer argument equal to one becomes a constant, so the
    // runtime bounds are excluded from specialization to keep one-trip loops.
    out_ << "@triton.jit";
    if (!boundParameters_.empty()) {
      out_ << "(do_not_specialize=[";
      for (std::size_t i = 0; i < boundParameters_.size(); ++i)
        out_ << (i ? ", " : "") << '"' << boundParameters_[i].first << '"';
      out_ << "])";
    }
    out_ << "\n";
    out_ << "def kernel(";
    for (const auto &param: function.params)
      out_ << parameterPointers_.at(param.name.name) << ", ";
    out_ << "out_ptr";
    for (const auto &[name, literal]: boundParameters_)
      out_ << ", " << name << "=" << literal;
    out_ << "):\n";
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
