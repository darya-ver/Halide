// #include "IRVisitor.h"
// #include <Halide.h>

// #include "ExternFuncArgument.h"
// #include "Function.h"

#include "../../src/IRVisitor.h"
#include <Halide.h>

#include "../../src/ExternFuncArgument.h"
#include "../../src/Function.h"

#include <stdexcept>
#include <unordered_map>

using namespace Halide;
using namespace Internal;

#define DEPTH_COST 3
struct StmtCost {
    int cost;   // per line
    int depth;  // per nested loop

    // add other costs later, like integer-ALU cost, float-ALU cost, memory cost, etc.
};

class FindStmtCost : public IRVisitor {

private:
    // create variable that will hold mapping of stmt to cost
    std::unordered_map<const IRNode *, StmtCost> stmt_cost;

    // gets cost from `stmt_cost` map
    int get_cost(const IRNode *node) const {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            assert(false);
            return 0;
        }
        return it->second.cost;
    }

    // sets cost in `stmt_cost` map
    void set_cost(const IRNode *node, int cost) {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            stmt_cost.emplace(node, StmtCost{cost, 0});
        } else {
            it->second.cost = cost;
        }
    }

    // gets depth from `stmt_cost` map
    int get_depth(const IRNode *node) const {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            assert(false);
            return 0;
        }
        return it->second.depth;
    }

    // sets depth in `stmt_cost` map
    void set_depth(const IRNode *node, int depth) {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            // should never be setting depth for a stmt that doesn't exist
            assert(false);
        } else {
            it->second.depth = depth;
        }
    }

    // increment depth in `stmt_cost` map by 1
    void increment_depth(const IRNode *node) {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            // should never be setting depth for a stmt that doesn't exist
            assert(false);
        } else {
            it->second.depth += 1;
        }
    }

public:
    // calculates the total cost of a stmt
    int calculate_total_cost(const IRNode *node) const {
        auto it = stmt_cost.find(node);
        if (it == stmt_cost.end()) {
            assert(false);
            return 0;
        }
        int cost = it->second.cost;
        int depth = it->second.depth;

        return cost + DEPTH_COST * depth;
    }

    // TODO: decide if count of 1 or 0
    void visit(const IntImm *op) override {
        set_cost(op, 1);
    }

    void visit(const UIntImm *op) override {
        set_cost(op, 1);
    }

    void visit(const FloatImm *op) override {
        set_cost(op, 1);
    }

    void visit(const StringImm *op) override {
        set_cost(op, 1);
    }

    void visit(const Cast *op) override {
        op->value.accept(this);
        int tempVal = get_cost(op->value.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Variable *op) override {
        set_cost(op, 1);
    }

    void visit(const Add *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Sub *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Mul *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Div *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Mod *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Min *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Max *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const EQ *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const NE *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const LT *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const LE *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const GT *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const GE *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const And *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Or *op) override {
        op->a.accept(this);
        op->b.accept(this);
        int tempVal = get_cost(op->a.get()) + get_cost(op->b.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Not *op) override {
        op->a.accept(this);
        int tempVal = get_cost(op->a.get());
        set_cost(op, 1 + tempVal);
    }

    // TODO: do we agree on my counts?
    void visit(const Select *op) override {
        op->condition.accept(this);
        op->true_value.accept(this);
        op->false_value.accept(this);

        int tempVal = get_cost(op->condition.get()) + get_cost(op->true_value.get()) + get_cost(op->false_value.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Load *op) override {
        throw std::runtime_error("`Load` not supported");
        // op->predicate.accept(this);
        // op->index.accept(this);
        // int tempVal = get_cost(op->predicate.get()) + get_cost(op->index.get());
        // set_cost(op, 1 + tempVal);
    }

    void visit(const Ramp *op) override {
        throw std::runtime_error("`Ramp` not supported");
        // op->base.accept(this);
        // op->stride.accept(this);
        // int tempVal = get_cost(op->base.get()) + get_cost(op->stride.get());
        // set_cost(op, 1 + tempVal);
    }

    void visit(const Broadcast *op) override {
        throw std::runtime_error("`Broadcast` not supported");
        // op->value.accept(this);
        // int tempVal = get_cost(op->value.get());
        // set_cost(op, 1 + tempVal);
    }

    void visit(const Call *op) override {
        int tempVal = 0;
        for (const auto &arg : op->args) {
            arg.accept(this);
            tempVal += get_cost(arg.get());
        }

        // Consider extern call args
        if (op->func.defined()) {
            Function f(op->func);
            if (op->call_type == Call::Halide && f.has_extern_definition()) {
                for (const auto &arg : f.extern_arguments()) {
                    if (arg.is_expr()) {
                        arg.expr.accept(this);
                        tempVal += get_cost(arg.expr.get());
                    }
                }
            }
        }
        set_cost(op, tempVal);
    }

    void visit(const Let *op) override {
        op->value.accept(this);
        op->body.accept(this);
        int tempVal = get_cost(op->value.get()) + get_cost(op->body.get());
        set_cost(op, tempVal);
    }

    void visit(const LetStmt *op) override {
        op->value.accept(this);
        op->body.accept(this);
        int tempVal = get_cost(op->value.get()) + get_cost(op->body.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const AssertStmt *op) override {
        op->condition.accept(this);
        op->message.accept(this);
        int tempVal = get_cost(op->condition.get()) + get_cost(op->message.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const ProducerConsumer *op) override {
        op->body.accept(this);
        int tempVal = get_cost(op->body.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const For *op) override {
        op->min.accept(this);
        op->extent.accept(this);
        op->body.accept(this);
        int bodyCost = get_cost(op->body.get());

        // FIXME: how to take into account the different types of for loops?
        if (op->for_type == ForType::Parallel) {
            throw std::runtime_error("`For - parallel` not supported");
        }
        // if (op->for_type == ForType::Serial) {

        // }
        if (op->for_type == ForType::Unrolled) {
            throw std::runtime_error("`For - unrolled` not supported");
        }
        if (op->for_type == ForType::Vectorized) {
            throw std::runtime_error("`For - vectorized` not supported");
        }

        // TODO: should we recurse into the body to set the depth?
        increment_depth(op->body.get());
    }

    void visit(const Acquire *op) override {
        op->semaphore.accept(this);
        op->count.accept(this);
        op->body.accept(this);
        int tempVal = get_cost(op->semaphore.get()) + get_cost(op->count.get()) + get_cost(op->body.get());
        set_cost(op, tempVal);
    }

    void visit(const Store *op) override {
        op->predicate.accept(this);
        op->value.accept(this);
        op->index.accept(this);
        int tempVal = get_cost(op->predicate.get()) + get_cost(op->value.get()) + get_cost(op->index.get());
        set_cost(op, 1 + tempVal);
    }

    void visit(const Provide *op) override {
        throw std::runtime_error("`Provide` not supported");
        // op->predicate.accept(this);
        // int tempVal = get_cost(op->predicate.get());
        // for (const auto &value : op->values) {
        //     value.accept(this);
        //     tempVal += get_cost(value.get());
        // }
        // for (const auto &arg : op->args) {
        //     arg.accept(this);
        //     tempVal += get_cost(arg.get());
        // }
        // set_cost(op, 1 + tempVal);
    }

    void visit(const Allocate *op) override {
        throw std::runtime_error("`Allocate` not supported");
        // int tempVal = 0;
        // for (const auto &extent : op->extents) {
        //     extent.accept(this);
        //     tempVal += get_cost(extent.get());
        // }
        // op->condition.accept(this);
        // tempVal += get_cost(op->condition.get());

        // if (op->new_expr.defined()) {
        //     op->new_expr.accept(this);
        //     tempVal += get_cost(op->new_expr.get());
        // }
        // op->body.accept(this);
        // tempVal += get_cost(op->body.get());
        // set_cost(op, tempVal);
    }

    void visit(const Free *op) override {
        // TODO: i feel like this should be more than cost 1, but the only
        //       vars it has is the name, which isn't helpful in determining
        //       the cost of the free
        set_cost(op, 1);
    }

    void visit(const Realize *op) override {
        throw std::runtime_error("`Realize` not supported");
        // TODO: is this the same logic as For, where I add the depth?
        // int tempVal = 0;
        // for (const auto &bound : op->bounds) {
        //     bound.min.accept(this);
        //     bound.extent.accept(this);
        //     tempVal += get_cost(bound.min.get()) + get_cost(bound.extent.get());
        // }
        // op->condition.accept(this);
        // op->body.accept(this);
        // tempVal += get_cost(op->condition.get()) + get_cost(op->body.get());
        // set_cost(op, tempVal);
    }

    void visit(const Prefetch *op) override {
        throw std::runtime_error("`Prefetch` not supported");
        // TODO: similar question as one above
        // int tempVal = 0;
        // for (const auto &bound : op->bounds) {
        //     bound.min.accept(this);
        //     bound.extent.accept(this);
        //     tempVal += get_cost(bound.min.get()) + get_cost(bound.extent.get());
        // }
        // op->condition.accept(this);
        // op->body.accept(this);
        // tempVal += get_cost(op->condition.get()) + get_cost(op->body.get());
        // set_cost(op, tempVal);
    }

    void visit(const Block *op) override {
        throw std::runtime_error("`Block` not supported");
        // int tempVal = 0;
        // op->first.accept(this);
        // tempVal += get_cost(op->first.get());
        // if (op->rest.defined()) {
        //     op->rest.accept(this);
        //     tempVal += get_cost(op->rest.get());
        // }
        // set_cost(op, tempVal);
    }

    void visit(const Fork *op) override {
        throw std::runtime_error("`Fork` not supported");
        // int tempVal = 0;
        // op->first.accept(this);
        // tempVal += get_cost(op->first.get());
        // if (op->rest.defined()) {
        //     op->rest.accept(this);
        //     tempVal += get_cost(op->rest.get());
        // }
        // set_cost(op, tempVal);
    }

    void visit(const IfThenElse *op) override {
        // TODO: is this correct, based on discussion about if-then-else, as
        //       compared to Select?
        op->condition.accept(this);
        op->then_case.accept(this);
        int tempVal = get_cost(op->condition.get()) + get_cost(op->then_case.get());
        if (op->else_case.defined()) {
            op->else_case.accept(this);
            tempVal += get_cost(op->else_case.get());
        }
        set_cost(op, tempVal);
    }

    void visit(const Evaluate *op) override {
        op->value.accept(this);
        int tempVal = get_cost(op->value.get());
        set_cost(op, tempVal);
    }

    void visit(const Shuffle *op) override {
        throw std::runtime_error("`Shuffle` not supported");
        // int tempVal = 0;
        // for (const Expr &i : op->vectors) {
        //     i.accept(this);
        //     tempVal += get_cost(i.get());
        // }
        // set_cost(op, tempVal);
    }

    void visit(const VectorReduce *op) override {
        throw std::runtime_error("`VectorReduce` not supported");
        // op->value.accept(this);
        // int tempVal = get_cost(op->value.get());
        // set_cost(op, tempVal);
    }

    void visit(const Atomic *op) override {
        throw std::runtime_error("`Atomic` not supported");
        // op->body.accept(this);
        // int tempVal = get_cost(op->body.get());
        // set_cost(op, tempVal);
    }
};
