#include "analysis/structurizer.hpp"
#include <algorithm>
#include <cassert>
#include <ostream>

namespace refractir {

  namespace {

    using Node = ControlTree::Node;
    using NodePtr = ControlTree::NodePtr;

    template<typename T>
    NodePtr make(T &&alt) {
      return std::make_unique<Node>(Node{std::forward<T>(alt)});
    }

    // One pending construct the emission point is nested in. back() of
    // the stack is the innermost. A Loop entry is the loop construct
    // itself; a Join entry is a dominator-tree follower whose subtree
    // is pending after the current one.
    struct Ctx {
      bool isLoop;
      std::size_t block; // loop header / join block
      int loopId;        // only for loops
    };

    class Builder {
    public:
      Builder(const FunDecl &f, const CFG &cfg, const DomTree &dt, const LoopInfo &li) :
          f_(f), cfg_(cfg), dt_(dt) {
        const std::size_t n = cfg.blocks.size();
        headerLoop_.assign(n, -1);
        loopMember_.assign(li.loops.size(), std::vector<char>(n, 0));
        for (std::size_t l = 0; l < li.loops.size(); ++l) {
          headerLoop_[li.loops[l].header] = static_cast<int>(l);
          for (std::size_t b: li.loops[l].blocks)
            loopMember_[l][b] = 1;
        }
        fwdPreds_.assign(n, 0);
        for (std::size_t b = 0; b < n; ++b) {
          if (dt.rpoNumber[b] == DomTree::kNone)
            continue;
          for (std::size_t p: cfg.pred[b]) {
            if (dt.rpoNumber[p] != DomTree::kNone && dt.rpoNumber[p] < dt.rpoNumber[b])
              ++fwdPreds_[b];
          }
        }
      }

      NodePtr run() {
        std::vector<Ctx> ctx;
        return doTree(cfg_.entry, ctx);
      }

    private:
      NodePtr doTree(std::size_t x, std::vector<Ctx> &ctx) {
        const int lid = headerLoop_[x];

        // Followers: dominator-tree children that cannot be inlined at
        // their branch site. For a loop header every dominated block
        // outside the loop trails the Loop node (reaching one is a
        // break); inside the loop — or anywhere for a non-header —
        // only merge nodes (>= 2 forward predecessors) become
        // followers. dt.children is RPO-ordered, which is exactly the
        // order followers must be emitted in.
        std::vector<std::size_t> inF, outF;
        for (std::size_t c: dt_.children[x]) {
          if (lid != -1 && !loopMember_[lid][c])
            outF.push_back(c);
          else if (fwdPreds_[c] >= 2)
            inF.push_back(c);
        }

        for (auto it = outF.rbegin(); it != outF.rend(); ++it)
          ctx.push_back({false, *it, -1});
        if (lid != -1)
          ctx.push_back({true, x, lid});
        for (auto it = inF.rbegin(); it != inF.rend(); ++it)
          ctx.push_back({false, *it, -1});

        ControlTree::Seq inner;
        inner.items.push_back(make(ControlTree::BlockStmts{x}));
        inner.items.push_back(lowerTerm(x, ctx));
        for (std::size_t c: inF) {
          ctx.pop_back(); // c's join is no longer pending inside its own subtree
          inner.items.push_back(doTree(c, ctx));
        }

        ControlTree::Seq outer;
        if (lid != -1) {
          ctx.pop_back(); // the Loop entry
          outer.items.push_back(make(ControlTree::Loop{lid, x, make(std::move(inner))}));
        } else {
          outer = std::move(inner);
        }
        for (std::size_t c: outF) {
          ctx.pop_back();
          outer.items.push_back(doTree(c, ctx));
        }
        return make(std::move(outer));
      }

      NodePtr lowerTerm(std::size_t x, std::vector<Ctx> &ctx) {
        return std::visit(
            [&](const auto &t) -> NodePtr {
              using T = std::decay_t<decltype(t)>;
              if constexpr (std::is_same_v<T, RetTerm>) {
                return make(ControlTree::Return{x});
              } else if constexpr (std::is_same_v<T, UnreachableTerm>) {
                return make(ControlTree::Trap{x});
              } else {
                if (!t.isConditional)
                  return doBranch(x, cfg_.indexOf.at(CFG::labelKey(t.dest)), ctx);
                return make(ControlTree::If{
                    x, doBranch(x, cfg_.indexOf.at(CFG::labelKey(t.thenLabel)), ctx),
                    doBranch(x, cfg_.indexOf.at(CFG::labelKey(t.elseLabel)), ctx)});
              }
            },
            f_.blocks[x].term
        );
      }

      NodePtr doBranch(std::size_t x, std::size_t y, std::vector<Ctx> &ctx) {
        if (dt_.rpoNumber[y] <= dt_.rpoNumber[x]) {
          // Retreating edge; reducibility makes it a back edge, so the
          // target is a header of some loop on the stack.
          int crossed = 0;
          for (auto it = ctx.rbegin(); it != ctx.rend(); ++it) {
            if (!it->isLoop)
              continue;
            if (it->block == y)
              return make(ControlTree::Continue{y, crossed});
            ++crossed;
          }
          assert(false && "back edge target not on the loop stack");
        }

        // Forward edge to a pending join?
        int crossed = 0, skipped = 0;
        for (auto it = ctx.rbegin(); it != ctx.rend(); ++it) {
          if (it->isLoop) {
            ++crossed;
            skipped = 0; // a break lands directly after the loop
          } else if (it->block == y) {
            if (skipped > 0)
              return make(ControlTree::JumpJoin{y, crossed});
            if (crossed > 0)
              return make(ControlTree::Break{y, crossed});
            return make(ControlTree::FallThrough{y});
          } else {
            ++skipped;
          }
        }

        // Not pending anywhere: y is reached by this edge alone, so its
        // whole subtree inlines right here.
        assert(fwdPreds_[y] == 1 && dt_.idom[y] == x && "unexpected non-inlinable branch");
        return doTree(y, ctx);
      }

      const FunDecl &f_;
      const CFG &cfg_;
      const DomTree &dt_;
      std::vector<int> headerLoop_;
      std::vector<std::vector<char>> loopMember_;
      std::vector<int> fwdPreds_;
    };

    void dumpNode(
        std::ostream &os, const Node &node, const ControlTree &tree, const CFG &cfg, int indent
    ) {
      const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
      std::visit(
          [&](const auto &n) {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, ControlTree::Seq>) {
              for (const auto &item: n.items)
                if (item)
                  dumpNode(os, *item, tree, cfg, indent);
            } else if constexpr (std::is_same_v<T, ControlTree::BlockStmts>) {
              os << pad << "block " << cfg.blocks[n.block] << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::If>) {
              os << pad << (n.negate ? "if-not " : "if ") << cfg.blocks[n.block] << "\n";
              os << pad << "  then:\n";
              if (n.thenBr)
                dumpNode(os, *n.thenBr, tree, cfg, indent + 2);
              if (n.elseBr) {
                os << pad << "  else:\n";
                dumpNode(os, *n.elseBr, tree, cfg, indent + 2);
              }
            } else if constexpr (std::is_same_v<T, ControlTree::Loop>) {
              os << pad << "loop " << n.loopId << " header=" << cfg.blocks[n.header] << "\n";
              dumpNode(os, *n.body, tree, cfg, indent + 1);
            } else if constexpr (std::is_same_v<T, ControlTree::CondLoop>) {
              os << pad << (n.negate ? "while-not " : "while ") << n.loopId
                 << " header=" << cfg.blocks[n.header] << "\n";
              dumpNode(os, *n.body, tree, cfg, indent + 1);
            } else if constexpr (std::is_same_v<T, ControlTree::DoWhile>) {
              os << pad << (n.negate ? "dowhile-not " : "dowhile ") << n.loopId
                 << " latch=" << cfg.blocks[n.latch] << "\n";
              if (n.body)
                dumpNode(os, *n.body, tree, cfg, indent + 1);
            } else if constexpr (std::is_same_v<T, ControlTree::Break>) {
              os << pad << "break " << cfg.blocks[n.target] << " levels=" << n.levels << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::Continue>) {
              os << pad << "continue " << cfg.blocks[n.header] << " levels=" << n.levels << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::FallThrough>) {
              os << pad << "fallthrough " << cfg.blocks[n.target] << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::JumpJoin>) {
              os << pad << "jumpjoin " << cfg.blocks[n.target] << " levels=" << n.levels << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::Return>) {
              os << pad << "return " << cfg.blocks[n.block] << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::SetFlag>) {
              os << pad << "setflag " << tree.flagNames[n.flag] << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::FlagBreak>) {
              os << pad << "flagbreak " << tree.flagNames[n.flag]
                 << " final=" << (n.isFinal ? 1 : 0) << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::FlagContinue>) {
              os << pad << "flagcontinue " << tree.flagNames[n.flag] << "\n";
            } else if constexpr (std::is_same_v<T, ControlTree::Guarded>) {
              os << pad << "guarded";
              for (int f: n.flags)
                os << " " << tree.flagNames[f];
              os << ":\n";
              dumpNode(os, *n.body, tree, cfg, indent + 1);
            } else if constexpr (std::is_same_v<T, ControlTree::ResetFlag>) {
              os << pad << "resetflag " << tree.flagNames[n.flag] << "\n";
            } else {
              static_assert(std::is_same_v<T, ControlTree::Trap>);
              os << pad << "trap " << cfg.blocks[n.block] << "\n";
            }
          },
          node.v
      );
    }

  } // namespace

  ControlTree
  Structurizer::build(const FunDecl &f, const CFG &cfg, const DomTree &dt, const LoopInfo &li) {
    ControlTree tree;
    tree.root = Builder(f, cfg, dt, li).run();
    return tree;
  }

  void ControlTree::dump(std::ostream &os, const CFG &cfg, const std::string &funName) const {
    os << (lowered ? "lowered-tree " : "control-tree ") << funName << ":\n";
    if (root)
      dumpNode(os, *root, *this, cfg, 1);
  }

} // namespace refractir
