#include "FindStmtCost.h"

using namespace Halide;
using namespace Internal;

/*
 * CostPreProcessor class
 */
void CostPreProcessor::traverse(const Module &m) {
    // recursively traverse all submodules
    for (const auto &s : m.submodules()) {
        traverse(s);
    }

    // traverse all functions
    for (const auto &f : m.functions()) {
        mutate(f.body);
    }
}

int CostPreProcessor::get_lock_access_count(const string name) const {
    auto it = lock_access_counts.find(name);
    if (it == lock_access_counts.end()) {
        cout << "name: " << name << endl;
        m_assert(false, "name not found in `lock_access_counts`");
        return 0;
    }
    return it->second;
}

void CostPreProcessor::increase_count(const string name) {
    auto it = lock_access_counts.find(name);
    if (it == lock_access_counts.end()) {
        lock_access_counts.emplace(name, 1);
    } else {
        it->second += 1;
    }
}

Stmt CostPreProcessor::visit(const Acquire *op) {
    stringstream name;
    name << op->semaphore;
    increase_count(name.str());
    return op;
}

Stmt CostPreProcessor::visit(const Atomic *op) {
    stringstream name;
    name << op->producer_name;
    increase_count(name.str());
    return op;
}

/*
 * FindStmtCost class
 */
void FindStmtCost::generate_costs(const Module &m) {
    cost_preprocessor.traverse(m);
    traverse(m);
}
void FindStmtCost::generate_costs(const Stmt &stmt) {
    cost_preprocessor.mutate(stmt);
    mutate(stmt);
}

int FindStmtCost::get_computation_range(const IRNode *op) const {

    // get max value of cost in stmt_cost map
    int max_cost = 0;
    for (auto const &pair : stmt_cost) {
        if (calculate_cost(pair.second) > max_cost) {
            max_cost = calculate_cost(pair.second);
        }
    }

    // divide max cost by 8 and round up to get ranges
    int range_size = (max_cost / NUMBER_COST_COLORS) + 1;
    int cost = calculate_computation_cost(op);
    int range = cost / range_size;
    return range;
}
int FindStmtCost::get_data_movement_range(const IRNode *op) const {

    // get max value of cost in stmt_cost map
    int max_cost = 0;
    for (auto const &pair : stmt_cost) {
        if (pair.second.data_movement_cost > max_cost) {
            max_cost = pair.second.data_movement_cost;
        }
    }

    // divide max cost by 8 and round up to get ranges
    int range_size = (max_cost / NUMBER_COST_COLORS) + 1;
    int cost = get_data_movement_cost(op);
    int range = cost / range_size;
    return range;
}

int FindStmtCost::get_depth(const IRNode *node) const {
    auto it = stmt_cost.find(node);
    if (it == stmt_cost.end()) {

        // sometimes, these constant values are created on the whim in
        // StmtToViz.cpp - return 1 to avoid crashing
        // TODO: what should this depth be?
        IRNodeType type = node->node_type;
        if (type == IRNodeType::IntImm || type == IRNodeType::UIntImm ||
            type == IRNodeType::FloatImm || type == IRNodeType::StringImm) {
            return 1;
        } else {
            print_node(node);
            m_assert(false, "node not found in stmt_cost");
            return 0;
        }
    }

    StmtCost cost_node = it->second;

    return cost_node.depth;
}
int FindStmtCost::calculate_computation_cost(const IRNode *node) const {
    auto it = stmt_cost.find(node);
    StmtCost cost_node;

    if (it == stmt_cost.end()) {
        // sometimes, these constant values are created on the whim in
        // StmtToViz.cpp - set cost_node to be fresh StmtCost to avoid crashing
        // TODO: what should this depth be?
        IRNodeType type = node->node_type;
        if (type == IRNodeType::IntImm || type == IRNodeType::UIntImm ||
            type == IRNodeType::FloatImm || type == IRNodeType::StringImm) {
            cost_node = StmtCost{1, 1, 0};
        } else {
            print_node(node);
            m_assert(false, "node not found in stmt_cost");
            return 0;
        }
    } else {
        cost_node = it->second;
    }

    return calculate_cost(cost_node);
}
int FindStmtCost::get_data_movement_cost(const IRNode *node) const {
    auto it = stmt_cost.find(node);
    if (it == stmt_cost.end()) {

        // sometimes, these constant values are created on the whim in
        // StmtToViz.cpp - return 0 to avoid crashing
        IRNodeType type = node->node_type;
        if (type == IRNodeType::IntImm || type == IRNodeType::UIntImm ||
            type == IRNodeType::FloatImm || type == IRNodeType::StringImm) {
            return 0;
        } else {
            print_node(node);
            m_assert(false, "node not found in stmt_cost");
            return 0;
        }
    }

    return it->second.data_movement_cost;
}

void FindStmtCost::traverse(const Module &m) {
    // recursively traverse all submodules
    for (const auto &s : m.submodules()) {
        traverse(s);
    }

    // traverse all functions
    for (const auto &f : m.functions()) {
        mutate(f.body);
    }

    return;
}

int FindStmtCost::get_computation_cost(const IRNode *node) const {
    auto it = stmt_cost.find(node);
    if (it == stmt_cost.end()) {

        // sometimes, these constant values are created on the whim in
        // StmtToViz.cpp - return 1 to avoid crashing
        IRNodeType type = node->node_type;
        if (type == IRNodeType::IntImm || type == IRNodeType::UIntImm ||
            type == IRNodeType::FloatImm || type == IRNodeType::StringImm) {
            return 1;
        } else {
            print_node(node);
            m_assert(false, "node not found in stmt_cost");
            return 0;
        }
    }
    return it->second.computation_cost;
}
void FindStmtCost::set_costs(const IRNode *node, int computation_cost, int data_movement_cost) {
    auto it = stmt_cost.find(node);
    if (it == stmt_cost.end()) {
        stmt_cost.emplace(node, StmtCost{current_loop_depth, computation_cost, data_movement_cost});
    } else {
        it->second.depth = current_loop_depth;
        it->second.computation_cost = computation_cost;
        it->second.data_movement_cost = data_movement_cost;
    }
}

int FindStmtCost::calculate_cost(StmtCost cost_node) const {
    int cost = cost_node.computation_cost;
    int depth = cost_node.depth;

    return cost + DEPTH_COST * depth;
}

int FindStmtCost::get_scaling_factor(uint8_t bits, uint16_t lanes) const {
    int bitsFactor = bits / 8;
    int lanesFactor = lanes / 8;
    if (bitsFactor == 0) {
        bitsFactor = 1;
    }
    if (lanesFactor == 0) {
        lanesFactor = 1;
    }
    return bitsFactor * lanesFactor;
}

void FindStmtCost::print_map(unordered_map<const IRNode *, StmtCost> const &m) {
    for (auto const &pair : m) {
        cout << "{" << pair.first << ": " << pair.second.computation_cost << "}\n";
    }
}

void FindStmtCost::visit_binary_op(const IRNode *op, const Expr &a, const Expr &b) {
    mutate(a);
    mutate(b);

    int tempVal = get_computation_cost(a.get()) + get_computation_cost(b.get());
    int dataMovementCost = get_data_movement_cost(a.get()) + get_data_movement_cost(b.get());
    set_costs(op, 1 + tempVal, dataMovementCost);
}

Expr FindStmtCost::visit(const IntImm *op) {
    set_costs(op, 1, 0);
    return op;
}

Expr FindStmtCost::visit(const UIntImm *op) {
    set_costs(op, 1, 0);
    return op;
}

Expr FindStmtCost::visit(const FloatImm *op) {
    set_costs(op, 1, 0);
    return op;
}

Expr FindStmtCost::visit(const StringImm *op) {
    set_costs(op, 1, 0);
    return op;
}

Expr FindStmtCost::visit(const Cast *op) {
    mutate(op->value);

    int tempVal = get_computation_cost(op->value.get());
    int dataMovementCost = get_data_movement_cost(op->value.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Variable *op) {
    set_costs(op, 1, 0);
    return op;
}

Expr FindStmtCost::visit(const Add *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Sub *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Mul *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Div *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Mod *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Min *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Max *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const EQ *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const NE *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const LT *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const LE *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const GT *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const GE *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const And *op) {
    visit_binary_op(op, op->a, op->b);

    return op;
}

Expr FindStmtCost::visit(const Or *op) {
    visit_binary_op(op, op->a, op->b);
    return op;
}

Expr FindStmtCost::visit(const Not *op) {
    mutate(op->a);

    int tempVal = get_computation_cost(op->a.get());
    int dataMovementCost = get_data_movement_cost(op->a.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

// TODO: do we agree on my counts?
Expr FindStmtCost::visit(const Select *op) {
    mutate(op->condition);
    mutate(op->true_value);
    mutate(op->false_value);

    int tempVal = get_computation_cost(op->condition.get()) +
                  get_computation_cost(op->true_value.get()) +
                  get_computation_cost(op->false_value.get());
    int dataMovementCost = get_data_movement_cost(op->condition.get()) +
                           get_data_movement_cost(op->true_value.get()) +
                           get_data_movement_cost(op->false_value.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Load *op) {
    uint8_t bits = op->type.bits();
    uint16_t lanes = op->type.lanes();
    int scalingFactor = get_scaling_factor(bits, lanes);

    mutate(op->predicate);
    mutate(op->index);

    int tempVal = get_computation_cost(op->predicate.get()) + get_computation_cost(op->index.get());
    int dataMovementCost =
        get_data_movement_cost(op->predicate.get()) + get_data_movement_cost(op->index.get());
    dataMovementCost += LOAD_COST;
    dataMovementCost *= scalingFactor;
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Ramp *op) {
    mutate(op->base);
    mutate(op->stride);

    int tempVal = get_computation_cost(op->base.get()) + get_computation_cost(op->stride.get());
    int dataMovementCost =
        get_data_movement_cost(op->base.get()) + get_data_movement_cost(op->stride.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Broadcast *op) {
    mutate(op->value);

    int tempVal = get_computation_cost(op->value.get());
    int dataMovementCost = get_data_movement_cost(op->value.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Call *op) {
    /*
        TODO: take into account this message from Maaz:

                > there are instructions in Halide IR that are not compute instructions but
                  data-movement instructions. Such as Shuffle::interleave or
       Shuffle::concatenate. I would ignore this for now but you should know about them.

    */
    int tempVal = 0;
    int dataMovementCost = 0;

    for (const auto &arg : op->args) {
        mutate(arg);
        tempVal += get_computation_cost(arg.get());
        dataMovementCost += get_data_movement_cost(arg.get());
    }

    // Consider extern call args
    if (op->func.defined()) {
        Function f(op->func);
        if (op->call_type == Call::Halide && f.has_extern_definition()) {
            for (const auto &arg : f.extern_arguments()) {
                if (arg.is_expr()) {
                    mutate(arg.expr);
                    tempVal += get_computation_cost(arg.expr.get());
                    dataMovementCost += get_data_movement_cost(arg.expr.get());
                }
            }
        }
    }

    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const Let *op) {

    // TODO: problem is that the variable in the let is not being tied to
    //       the value of the let, so it's not getting its context correctly
    //       set.

    mutate(op->value);
    mutate(op->body);

    int tempVal = get_computation_cost(op->value.get()) + get_computation_cost(op->body.get());
    int dataMovementCost =
        get_data_movement_cost(op->value.get()) + get_data_movement_cost(op->body.get());
    set_costs(op, tempVal, dataMovementCost);

    // add_variable_map(op->name, get_context(op->value.get()));

    return op;
}

Expr FindStmtCost::visit(const Shuffle *op) {
    int tempVal = 0;
    int dataMovementCost = 0;

    for (const Expr &i : op->vectors) {
        mutate(i);
        tempVal += get_computation_cost(i.get());
        dataMovementCost += get_data_movement_cost(i.get());
    }

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Expr FindStmtCost::visit(const VectorReduce *op) {
    mutate(op->value);

    int tempVal = get_computation_cost(op->value.get());
    int countCost = op->value.type().lanes() - 1;
    int dataMovementCost = get_data_movement_cost(op->value.get());
    set_costs(op, tempVal + countCost, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const LetStmt *op) {
    mutate(op->value);
    mutate(op->body);

    int tempVal = get_computation_cost(op->value.get());
    int dataMovementCost = get_data_movement_cost(op->value.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const AssertStmt *op) {
    mutate(op->condition);
    mutate(op->message);

    int tempVal =
        get_computation_cost(op->condition.get()) + get_computation_cost(op->message.get());
    int dataMovementCost =
        get_data_movement_cost(op->condition.get()) + get_data_movement_cost(op->message.get());
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const ProducerConsumer *op) {
    mutate(op->body);

    int tempVal = get_computation_cost(op->body.get());
    int dataMovementCost = get_data_movement_cost(op->body.get());
    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const For *op) {

    current_loop_depth += 1;

    mutate(op->min);
    mutate(op->extent);

    mutate(op->body);

    current_loop_depth -= 1;

    int bodyCost = get_computation_cost(op->body.get());
    int dataMovementCost = get_data_movement_cost(op->body.get());

    // TODO: how to take into account the different types of for loops?
    if (op->for_type == ForType::Parallel) {
        m_assert(false, "Parallel for loops are not supported yet");
    }
    if (op->for_type == ForType::Unrolled) {
        m_assert(false, "Unrolled for loops are not supported yet");
    }
    if (op->for_type == ForType::Vectorized) {
        m_assert(false, "Vectorized for loops are not supported yet");
    }
    set_costs(op, 1 + bodyCost, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Acquire *op) {
    /*
        TODO: change this

                * depends on contention (how many other accesses are there to this
                  particular semaphore?)
                * need to have separate FindStmtCost::visitor that FindStmtCost::visits
       everything and es the number of times each lock is accessed, and also keep track of the
       depth of said lock (the deeper, the more times it will be accessed)

                * do we need to recurse on body???
    */
    m_assert(false,
             "reached Acquire! take a look at its use - visit(Acquire) is not fully implemented");

    stringstream name;
    name << op->semaphore;
    int lock_cost = cost_preprocessor.get_lock_access_count(name.str());
    set_costs(op, lock_cost, 0);  // this is to remove the error of unused variable
    // TODO: do something with lock cost

    mutate(op->semaphore);
    mutate(op->count);
    mutate(op->body);
    int tempVal = get_computation_cost(op->semaphore.get()) +
                  get_computation_cost(op->count.get()) + get_computation_cost(op->body.get());
    int dataMovementCost = get_data_movement_cost(op->semaphore.get()) +
                           get_data_movement_cost(op->count.get()) +
                           get_data_movement_cost(op->body.get());
    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Store *op) {
    mutate(op->predicate);
    mutate(op->value);
    mutate(op->index);

    uint8_t bits = op->value.type().bits();
    uint16_t lanes = op->value.type().lanes();
    int scalingFactor = get_scaling_factor(bits, lanes);

    int tempVal = get_computation_cost(op->predicate.get()) +
                  get_computation_cost(op->value.get()) + get_computation_cost(op->index.get());
    int dataMovementCost = get_data_movement_cost(op->predicate.get()) +
                           get_data_movement_cost(op->value.get()) +
                           get_data_movement_cost(op->index.get());
    dataMovementCost += STORE_COST;
    dataMovementCost *= scalingFactor;
    set_costs(op, 1 + tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Provide *op) {
    int tempVal = get_computation_cost(op->predicate.get());
    int dataMovementCost = get_data_movement_cost(op->predicate.get());

    for (const auto &value : op->values) {
        mutate(value);
        tempVal += get_computation_cost(value.get());
        dataMovementCost += get_data_movement_cost(value.get());
    }
    for (const auto &arg : op->args) {
        mutate(arg);
        tempVal += get_computation_cost(arg.get());
        dataMovementCost += get_data_movement_cost(arg.get());
    }

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Allocate *op) {
    /*
        TODO: treat this node differently

            * loop depth is important
            * type of allocation is especially important (heap vs stack)
                  this can be found MemoryType of the Allocate node (might need
                  some nesting to find the node with this type)
            * could FindStmtCost::visit `extents` for costs, and n`
            * (in case of GPUShared type) visualize size of allocation in case
              the size of shared memory and goes into main memory

            * do we need to recurse on body???
    */
    int tempVal = 0;
    int dataMovementCost = 0;

    for (const auto &extent : op->extents) {
        mutate(extent);
        tempVal += get_computation_cost(extent.get());
        dataMovementCost += get_data_movement_cost(extent.get());
    }

    mutate(op->condition);
    tempVal += get_computation_cost(op->condition.get());
    dataMovementCost += get_data_movement_cost(op->condition.get());

    if (op->new_expr.defined()) {
        mutate(op->new_expr);
        tempVal += get_computation_cost(op->new_expr.get());
        dataMovementCost += get_data_movement_cost(op->new_expr.get());
    }

    mutate(op->body);
    tempVal += get_computation_cost(op->body.get());
    dataMovementCost += get_data_movement_cost(op->body.get());

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Free *op) {
    // TODO: i feel like this should be more than cost 1, but the only
    //       vars it has is the name, which isn't helpful in determining
    //       the cost of the free
    set_costs(op, 1, 0);
    return op;
}

Stmt FindStmtCost::visit(const Realize *op) {
    // TODO: is this the same logic as For, where I add the depth?
    int tempVal = 0;
    int dataMovementCost = 0;

    for (const auto &bound : op->bounds) {
        mutate(bound.min);
        mutate(bound.extent);
        tempVal += get_computation_cost(bound.min.get()) + get_computation_cost(bound.extent.get());
        dataMovementCost +=
            get_data_movement_cost(bound.min.get()) + get_data_movement_cost(bound.extent.get());
    }

    mutate(op->condition);
    mutate(op->body);
    tempVal += get_computation_cost(op->condition.get()) + get_computation_cost(op->body.get());
    dataMovementCost +=
        get_data_movement_cost(op->condition.get()) + get_data_movement_cost(op->body.get());

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Prefetch *op) {
    /*
        TODO: like caching? # of memory stores
    */
    int tempVal = 0;
    int dataMovementCost = 0;

    for (const auto &bound : op->bounds) {
        mutate(bound.min);
        mutate(bound.extent);

        tempVal += get_computation_cost(bound.min.get()) + get_computation_cost(bound.extent.get());
        dataMovementCost +=
            get_data_movement_cost(bound.min.get()) + get_data_movement_cost(bound.extent.get());
    }

    mutate(op->condition);
    mutate(op->body);
    tempVal += get_computation_cost(op->condition.get()) + get_computation_cost(op->body.get());
    dataMovementCost +=
        get_data_movement_cost(op->condition.get()) + get_data_movement_cost(op->body.get());

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Block *op) {
    // TODO: making this cost 1 is wrong - need to change this
    int tempVal = 0;
    int dataMovementCost = 0;

    mutate(op->first);
    tempVal += get_computation_cost(op->first.get());
    dataMovementCost += get_data_movement_cost(op->first.get());

    if (op->rest.defined()) {
        mutate(op->rest);
        tempVal += get_computation_cost(op->rest.get());
        dataMovementCost += get_data_movement_cost(op->rest.get());
    }

    // set_costs(op, tempVal, dataMovementCost);
    set_costs(op, 1, 0);

    return op;
}

Stmt FindStmtCost::visit(const Fork *op) {
    int tempVal = 0;
    int dataMovementCost = 0;

    mutate(op->first);
    tempVal += get_computation_cost(op->first.get());
    dataMovementCost += get_data_movement_cost(op->first.get());

    if (op->rest.defined()) {
        mutate(op->rest);
        tempVal += get_computation_cost(op->rest.get());
        dataMovementCost += get_data_movement_cost(op->rest.get());
    }

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const IfThenElse *op) {
    mutate(op->condition);
    mutate(op->then_case);

    int tempVal =
        get_computation_cost(op->condition.get()) + get_computation_cost(op->then_case.get());
    int dataMovementCost =
        get_data_movement_cost(op->condition.get()) + get_data_movement_cost(op->then_case.get());

    if (op->else_case.defined()) {
        mutate(op->else_case);
        tempVal += get_computation_cost(op->else_case.get());
        dataMovementCost += get_data_movement_cost(op->else_case.get());
    }

    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Evaluate *op) {
    mutate(op->value);

    int tempVal = get_computation_cost(op->value.get());
    int dataMovementCost = get_data_movement_cost(op->value.get());
    set_costs(op, tempVal, dataMovementCost);

    return op;
}

Stmt FindStmtCost::visit(const Atomic *op) {
    /*
        TODO: change this

                * make it similar to acquire
                * parallel vs vector is important
    */
    m_assert(false,
             "reached Atomic! take a look at its use - visit(Atomic) is not fully implemented");
    mutate(op->body);

    stringstream name;
    name << op->producer_name;
    int lock_cost = cost_preprocessor.get_lock_access_count(name.str());
    set_costs(op, lock_cost, 0);  // this is to remove the error of unused variable
    // TODO: do something with lock cost

    int tempVal = get_computation_cost(op->body.get());
    int dataMovementCost = get_data_movement_cost(op->body.get());
    set_costs(op, tempVal, dataMovementCost);

    return op;
}

void FindStmtCost::print_node(const IRNode *node) const {
    cout << "Crashing node has type: ";
    IRNodeType type = node->node_type;
    if (type == IRNodeType::IntImm) {
        cout << "IntImm type" << endl;
        auto node1 = dynamic_cast<const IntImm *>(node);
        cout << "value: " << node1->value << endl;
    } else if (type == IRNodeType::UIntImm) {
        cout << "UIntImm type" << endl;
    } else if (type == IRNodeType::FloatImm) {
        cout << "FloatImm type" << endl;
    } else if (type == IRNodeType::StringImm) {
        cout << "StringImm type" << endl;
    } else if (type == IRNodeType::Broadcast) {
        cout << "Broadcast type" << endl;
    } else if (type == IRNodeType::Cast) {
        cout << "Cast type" << endl;
    } else if (type == IRNodeType::Variable) {
        cout << "Variable type" << endl;
    } else if (type == IRNodeType::Add) {
        cout << "Add type" << endl;
    } else if (type == IRNodeType::Sub) {
        cout << "Sub type" << endl;
    } else if (type == IRNodeType::Mod) {
        cout << "Mod type" << endl;
    } else if (type == IRNodeType::Mul) {
        cout << "Mul type" << endl;
    } else if (type == IRNodeType::Div) {
        cout << "Div type" << endl;
    } else if (type == IRNodeType::Min) {
        cout << "Min type" << endl;
    } else if (type == IRNodeType::Max) {
        cout << "Max type" << endl;
    } else if (type == IRNodeType::EQ) {
        cout << "EQ type" << endl;
    } else if (type == IRNodeType::NE) {
        cout << "NE type" << endl;
    } else if (type == IRNodeType::LT) {
        cout << "LT type" << endl;
    } else if (type == IRNodeType::LE) {
        cout << "LE type" << endl;
    } else if (type == IRNodeType::GT) {
        cout << "GT type" << endl;
    } else if (type == IRNodeType::GE) {
        cout << "GE type" << endl;
    } else if (type == IRNodeType::And) {
        cout << "And type" << endl;
    } else if (type == IRNodeType::Or) {
        cout << "Or type" << endl;
    } else if (type == IRNodeType::Not) {
        cout << "Not type" << endl;
    } else if (type == IRNodeType::Select) {
        cout << "Select type" << endl;
    } else if (type == IRNodeType::Load) {
        cout << "Load type" << endl;
    } else if (type == IRNodeType::Ramp) {
        cout << "Ramp type" << endl;
    } else if (type == IRNodeType::Call) {
        cout << "Call type" << endl;
    } else if (type == IRNodeType::Let) {
        cout << "Let type" << endl;
    } else if (type == IRNodeType::Shuffle) {
        cout << "Shuffle type" << endl;
    } else if (type == IRNodeType::VectorReduce) {
        cout << "VectorReduce type" << endl;
    } else if (type == IRNodeType::LetStmt) {
        cout << "LetStmt type" << endl;
    } else if (type == IRNodeType::AssertStmt) {
        cout << "AssertStmt type" << endl;
    } else if (type == IRNodeType::ProducerConsumer) {
        cout << "ProducerConsumer type" << endl;
    } else if (type == IRNodeType::For) {
        cout << "For type" << endl;
    } else if (type == IRNodeType::Acquire) {
        cout << "Acquire type" << endl;
    } else if (type == IRNodeType::Store) {
        cout << "Store type" << endl;
    } else if (type == IRNodeType::Provide) {
        cout << "Provide type" << endl;
    } else if (type == IRNodeType::Allocate) {
        cout << "Allocate type" << endl;
    } else if (type == IRNodeType::Free) {
        cout << "Free type" << endl;
    } else if (type == IRNodeType::Realize) {
        cout << "Realize type" << endl;
    } else if (type == IRNodeType::Block) {
        cout << "Block type" << endl;
    } else if (type == IRNodeType::Fork) {
        cout << "Fork type" << endl;
    } else if (type == IRNodeType::IfThenElse) {
        cout << "IfThenElse type" << endl;
    } else if (type == IRNodeType::Evaluate) {
        cout << "Evaluate type" << endl;
    } else if (type == IRNodeType::Prefetch) {
        cout << "Prefetch type" << endl;
    } else if (type == IRNodeType::Atomic) {
        cout << "Atomic type" << endl;
    } else {
        cout << "Unknown type" << endl;
    }
}
