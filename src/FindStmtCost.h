#ifndef HALIDE_FIND_STMT_COST_H
#define HALIDE_FIND_STMT_COST_H

#include "Error.h"
#include "ExternFuncArgument.h"
#include "Function.h"
#include "IRVisitor.h"
#include "Module.h"

#include <unordered_map>

namespace Halide {
namespace Internal {

// Different classes of costs
enum StmtCostModel { Compute,
                     DataMovement,
                     ComputeInclusive,
                     DataMovementInclusive };

/*
 * StmtCost struct
 */
struct StmtCost {
    // DMC == Data Movement Cost, CC == Compute Cost
    static constexpr int NormalNodeCC = 1;
    static constexpr int NormalNodeDMC = 0;
    static constexpr int LoadDMC = 2;
    static constexpr int StoreDMC = 3;

    int depth;                         // per nested loop
    int computation_cost_inclusive;    // per entire node (includes cost of body)
    int data_movement_cost_inclusive;  // per entire node (includes cost of body)
    int computation_cost_exclusive;    // per line
    int data_movement_cost_exclusive;  // per line
};

/*
 * FindStmtCost class
 */
class FindStmtCost : public IRVisitor {

public:
    FindStmtCost()
        : current_loop_depth(0), max_computation_cost_inclusive(0),
          max_data_movement_cost_inclusive(0), max_computation_cost_exclusive(0),
          max_data_movement_cost_exclusive(0) {
    }

    // starts the traversal of the given node
    void generate_costs(const Module &m);

    // checks if node is IfThenElse or something else - will call get_if_node_cost if it is,
    // get_computation_cost/get_data_movement_cost otherwise
    int get_cost(const IRNode *node, StmtCostModel cost_model) const;

    // gets the depth of the node
    int get_depth(const IRNode *node) const;

    // gets max costs
    int get_max_cost(StmtCostModel cost_model) const;

    // prints node type
    std::string print_node(const IRNode *node) const;

private:
    std::unordered_map<const IRNode *, StmtCost> stmt_cost;  // key: node, value: cost
    int current_loop_depth;                                  // stores current loop depth level

    // these are used for determining the range of the cost
    int max_computation_cost_inclusive;
    int max_data_movement_cost_inclusive;
    int max_computation_cost_exclusive;
    int max_data_movement_cost_exclusive;

    // starts the traversal based on Module
    void traverse(const Module &m);

    // gets the total costs of a node
    int get_computation_cost(const IRNode *node, bool inclusive) const;
    int get_data_movement_cost(const IRNode *node, bool inclusive) const;

    // treats if nodes differently when visualizing cost - will have cost be:
    //      cost of condition + cost of then_case (exclude else_case in cost)
    int get_if_node_cost(const IfThenElse *op, StmtCostModel cost_model) const;

    // gets costs from `stmt_cost` map of children nodes and sum them up accordingly
    std::vector<int> get_costs_children(const IRNode *parent, const std::vector<const IRNode *> &children,
                                        bool inclusive) const;

    // sets inclusive/exclusive costs
    void set_costs(bool inclusive, const IRNode *node, const std::vector<const IRNode *> &children,
                   const std::function<int(int)> &calculate_cc, const std::function<int(int)> &calculate_dmc);

    // sets max computation cost and max data movement cost (inclusive and exclusive)
    void set_max_costs(const Module &m);

    // gets scaling factor for Load/Store based on lanes and bits
    int get_scaling_factor(uint8_t bits, uint16_t lanes) const;

    void visit_binary_op(const IRNode *op, const Expr &a, const Expr &b);

    void visit(const IntImm *op) override;
    void visit(const UIntImm *op) override;
    void visit(const FloatImm *op) override;
    void visit(const StringImm *op) override;
    void visit(const Cast *op) override;
    void visit(const Reinterpret *op) override;
    void visit(const Variable *op) override;
    void visit(const Add *op) override;
    void visit(const Sub *op) override;
    void visit(const Mul *op) override;
    void visit(const Div *op) override;
    void visit(const Mod *op) override;
    void visit(const Min *op) override;
    void visit(const Max *op) override;
    void visit(const EQ *op) override;
    void visit(const NE *op) override;
    void visit(const LT *op) override;
    void visit(const LE *op) override;
    void visit(const GT *op) override;
    void visit(const GE *op) override;
    void visit(const And *op) override;
    void visit(const Or *op) override;
    void visit(const Not *op) override;
    void visit(const Select *op) override;
    void visit(const Load *op) override;
    void visit(const Ramp *op) override;
    void visit(const Broadcast *op) override;
    void visit(const Call *op) override;
    void visit(const Let *op) override;
    void visit(const Shuffle *op) override;
    void visit(const VectorReduce *op) override;
    void visit(const LetStmt *op) override;
    void visit(const AssertStmt *op) override;
    void visit(const ProducerConsumer *op) override;
    void visit(const For *op) override;
    void visit(const Acquire *op) override;
    void visit(const Store *op) override;
    void visit(const Provide *op) override;
    void visit(const Allocate *op) override;
    void visit(const Free *op) override;
    void visit(const Realize *op) override;
    void visit(const Prefetch *op) override;
    void visit(const Block *op) override;
    void visit(const Fork *op) override;
    void visit(const IfThenElse *op) override;
    void visit(const Evaluate *op) override;
    void visit(const Atomic *op) override;
};

}  // namespace Internal
}  // namespace Halide

#endif  // FINDSTMTCOST_H
