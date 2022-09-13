#ifndef ProducerConsumerHierarchy_H
#define ProducerConsumerHierarchy_H

#include "FindStmtCost.h"
#include "IRVisitor.h"

#include <set>
#include <unordered_map>

using namespace std;
using namespace Halide;
using namespace Internal;

#define MAX_CONDITION_LENGTH 30

struct StmtSize {
    map<string, string> writes;
    map<string, string> reads;
    bool empty() const {
        return writes.size() == 0 && reads.size() == 0;
    }

    string to_string() {
        string result = "";
        if (!empty()) {
            result += "Writes: ";
            for (auto it = writes.begin(); it != writes.end(); ++it) {
                result += it->first + ": " + it->second + "\n";
            }
            result += "\n";
            result += "Reads: ";
            for (auto it = reads.begin(); it != reads.end(); ++it) {
                result += it->first + ": " + it->second + "\n";
            }
        }
        return result;
    }
};

/*
 * StmtSizes class
 */
class StmtSizes : public IRVisitor {
public:
    vector<string> function_names;  // used for figuring out whether variable is a function call

    void generate_sizes(const Module &m);
    void generate_sizes(const Stmt &stmt);

    StmtSize get_size(const IRNode *node) const;

    // for coloring
    string string_span(string varName) const;
    string int_span(int64_t intVal) const;

    string print_node(const IRNode *node) const;

private:
    using IRVisitor::visit;

    unordered_map<const IRNode *, StmtSize> stmt_sizes;
    map<string, int> curr_load_values;

    void traverse(const Module &m);

    string get_simplified_string(string a, string b, string op);

    void set_write_size(const IRNode *node, string write_var, string write_size);
    void set_read_size(const IRNode *node, string read_var, string read_size);

    void visit(const Store *op) override;
    void add_load_value(const string &name, const int lanes);
    void visit(const Load *op) override;
};

/*
 * ProducerConsumerHierarchy class
 */
class ProducerConsumerHierarchy : public IRVisitor {

public:
    static const string prodConsCSS, scrollToFunctionJSVizToCode;

    ProducerConsumerHierarchy(FindStmtCost findStmtCostPopulated)
        : findStmtCost(findStmtCostPopulated) {
    }

    // generates the html for the producer-consumer hierarchy
    string generate_producer_consumer_html(const Module &m);
    string generate_producer_consumer_html(const Stmt &stmt);

    string generate_prodCons_js();

private:
    using IRVisitor::visit;

    string html;                // main html string
    StmtSizes pre_processor;    // generates the sizes of the nodes
    FindStmtCost findStmtCost;  // used to determine the color of each statement

    // used for getting anchor names
    int ifCount = 0;
    int producerConsumerCount = 0;
    int forCount = 0;
    int storeCount = 0;
    int allocateCount = 0;
    int functionCount = 0;

    // tooltip count
    int prodConsTooltipCount = 0;

    // for traversal of a Module object
    void startModuleTraversal(const Module &m);

    // opens and closes divs
    void open_box_div(string className, const IRNode *op);
    void close_box_div();
    void open_function_box_div();
    void close_function_box_div();
    void open_header_div();
    void open_box_header_title_div();
    void open_box_header_table_div();
    void open_store_div();
    void close_div();

    // header functions
    void open_header(const string &header, string anchorName);
    void close_header(string anchorName);
    void div_header(const string &header, StmtSize *size, string anchorName);
    void function_div_header(const string &functionName, string anchorName);
    vector<string> get_allocation_sizes(const Allocate *op) const;
    void allocate_div_header(const Allocate *op, const string &header, string anchorName);
    void for_loop_div_header(const For *op, const string &header, string anchorName);

    // opens and closes an if-tree
    void if_tree(const IRNode *op, const string &header, string anchorName);
    void close_if_tree();

    // different cost tables
    void read_write_table(StmtSize &size);
    void allocate_table(vector<string> &allocationSizes);
    void for_loop_table(string loop_size);

    // opens relative code links
    void see_code_button_div(string anchorName, bool putDiv = true);

    // tooltip
    string info_tooltip(string toolTipText, string className);

    // for cost colors - side bars
    void generate_computation_cost_div(const IRNode *op);
    void generate_memory_cost_div(const IRNode *op);
    void open_content_div();

    // for cost colors - side boxes
    string color_button(int colorRange);
    string computation_div(const IRNode *op);
    string data_movement_div(const IRNode *op);
    string tooltip_table(vector<pair<string, string>> &table);
    void cost_colors(const IRNode *op);

    void visit_function(const LoweredFunc &func);
    void visit(const Variable *op) override;
    void visit(const ProducerConsumer *op) override;
    string get_loop_iterator(const For *op) const;
    void visit(const For *op) override;
    void visit(const IfThenElse *op) override;
    void visit(const Store *op) override;
    void visit(const Load *op) override;
    void visit(const Allocate *op) override;
};

#endif
