#ifndef IRVISUALIZATION_H
#define IRVISUALIZATION_H

#include "FindStmtCost.h"
#include "IRVisitor.h"

#include <set>
#include <unordered_map>

using namespace std;
using namespace Halide;
using namespace Internal;

#define MAX_CONDITION_LENGTH 35
#define MAX_NUMBER_OF_NODES 15

#define NUMBER_COST_COLORS 20

struct StmtSize {
    map<string, string> writes;
    map<string, string> reads;

    bool empty() const {
        return writes.size() == 0 && reads.size() == 0;
    }
};

/*
 * GetReadWrite class
 */
class GetReadWrite : public IRVisitor {
public:
    vector<string> function_names;  // used for figuring out whether variable is a function call

    // generates the reads/writes for the module
    void generate_sizes(const Module &m);

    // returns the reads/writes for the given node
    StmtSize get_size(const IRNode *node) const;

    // for coloring
    string string_span(string var_name) const;
    string int_span(int64_t int_val) const;

    // prints nodes in error messages
    string print_node(const IRNode *node) const;

private:
    using IRVisitor::visit;

    unordered_map<const IRNode *, StmtSize> stmt_sizes;  // stores the sizes
    map<string, int> curr_load_values;                   // used when calculating store reads

    // starts traversal of the module
    void traverse(const Module &m);

    // used to simplify expressions with + and *, to not have too many parentheses
    string get_simplified_string(string a, string b, string op);

    // sets reads/writes for the given node
    void set_write_size(const IRNode *node, string write_var, string write_size);
    void set_read_size(const IRNode *node, string read_var, string read_size);

    void visit(const Store *op) override;
    void add_load_value(const string &name, const int lanes);
    void visit(const Load *op) override;
};

/*
 * IRVisualization class
 */
class IRVisualization : public IRVisitor {

public:
    static const string ir_viz_CSS, scroll_to_function_JS_viz_to_code;

    IRVisualization(FindStmtCost find_stmt_cost_populated)
        : find_stmt_cost(find_stmt_cost_populated), ir_viz_tooltip_count(0), if_count(0),
          producer_consumer_count(0), for_count(0), store_count(0), allocate_count(0),
          function_count(0) {
    }

    // generates the html for the IR Visualization
    string generate_ir_visualization_html(const Module &m);

    // returns the JS for the IR Visualization
    string generate_ir_visualization_js();

    // generates tooltip tables based on given node
    string generate_computation_cost_tooltip(const IRNode *op, string extraNote);
    string generate_data_movement_cost_tooltip(const IRNode *op, string extraNote);

    // returns the range of the node's cost based on the other nodes' costs
    int get_color_range(const IRNode *op, bool inclusive, bool is_computation) const;

    // returns color range when blocks are collapsed in code viz
    int get_combined_color_range(const IRNode *op, bool is_computation) const;

private:
    using IRVisitor::visit;

    stringstream html;            // main html string
    GetReadWrite get_read_write;  // generates the read/write sizes
    FindStmtCost find_stmt_cost;  // used to determine the color of each statement
    int num_of_nodes;             // keeps track of the number of nodes in the visualization
    int ir_viz_tooltip_count;     // tooltip count

    // used for getting anchor names
    int if_count;
    int producer_consumer_count;
    int for_count;
    int store_count;
    int allocate_count;
    int function_count;

    // for traversal of a Module object
    void start_module_traversal(const Module &m);

    // opens and closes divs
    string open_box_div(string class_name, const IRNode *op);
    string close_box_div() const;
    string open_function_box_div() const;
    string close_function_box_div() const;
    string open_header_div() const;
    string open_box_header_title_div() const;
    string open_box_header_table_div() const;
    string open_store_div() const;
    string open_body_div() const;
    string close_div() const;

    // header functions
    string open_header(const string &header, string anchor_name,
                       vector<pair<string, string>> info_tooltip_table);
    string close_header() const;
    string div_header(const string &header, StmtSize *size, string anchor_name,
                      vector<pair<string, string>> info_tooltip_table);
    string function_div_header(const string &function_name, string anchor_name) const;
    vector<string> get_allocation_sizes(const Allocate *op) const;
    string allocate_div_header(const Allocate *op, const string &header, string anchor_name,
                               vector<pair<string, string>> &info_tooltip_table);
    string for_loop_div_header(const For *op, const string &header, string anchor_name);

    // opens and closes an if-tree
    string if_tree(const IRNode *op, const string &header, string anchor_name);
    string close_if_tree() const;

    // different cost tables
    string read_write_table(StmtSize &size) const;
    string allocate_table(vector<string> &allocation_sizes) const;
    string for_loop_table(string loop_size) const;

    // generates code for button that will scroll to associated IR code line
    string see_code_button_div(string anchor_name, bool put_div = true) const;

    // info button with tooltip
    string info_button_with_tooltip(string tooltip_text, string button_class_name,
                                    string tooltip_class_name = "");

    // for cost colors - side bars of boxes
    string generate_computation_cost_div(const IRNode *op);
    string generate_memory_cost_div(const IRNode *op);
    string open_content_div() const;

    // gets cost percentages of a given node
    int get_cost_percentage(const IRNode *node, bool inclusive, bool is_computation) const;

    // builds the tooltip cost table based on given input table
    string tooltip_table(vector<pair<string, string>> &table, string extra_note = "");

    // for cost colors - side boxes of Load nodes
    string color_button(int color_range);
    string computation_div(const IRNode *op);
    string data_movement_div(const IRNode *op);
    string cost_colors(const IRNode *op);

    void visit_function(const LoweredFunc &func);
    void visit(const Variable *op) override;
    void visit(const ProducerConsumer *op) override;
    string get_loop_iterator(const For *op) const;
    void visit(const For *op) override;
    void visit(const IfThenElse *op) override;
    void visit(const Store *op) override;
    void visit(const Load *op) override;
    string get_memory_type(MemoryType mem_type) const;
    void visit(const Allocate *op) override;
};

#endif
