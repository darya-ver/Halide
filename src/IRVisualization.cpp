#include "IRVisualization.h"
#include "IROperator.h"
#include "Module.h"

using namespace std;
using namespace Halide;
using namespace Internal;

/*
 * StmtSizes class
 */

void StmtSizes::generate_sizes(const Module &m) {
    traverse(m);
}

StmtSize StmtSizes::get_size(const IRNode *node) const {
    auto it = stmt_sizes.find(node);

    // errors if node is not found
    if (it == stmt_sizes.end()) {
        internal_error << "\n\nStmtSizes::get_size - Node not found in StmtSizes: "
                       << print_node(node) << "\n\n";
        return StmtSize();
    }

    return it->second;
}

string StmtSizes::string_span(string varName) const {
    return "<span class='stringType'>" + varName + "</span>";
}
string StmtSizes::int_span(int64_t intVal) const {
    return "<span class='intType'>" + std::to_string(intVal) + "</span>";
}

void StmtSizes::traverse(const Module &m) {

    // recursively traverse all submodules
    for (const auto &s : m.submodules()) {
        traverse(s);
    }

    // traverse all functions
    for (const auto &f : m.functions()) {
        function_names.push_back(f.name);
        f.body.accept(this);
    }
}

string StmtSizes::get_simplified_string(string a, string b, string op) {
    if (op == "+") {
        return a + " + " + b;
    }

    else if (op == "*") {
        // check if b contains "+"
        if (b.find("+") != string::npos) {
            return a + "*(" + b + ")";
        } else {
            return a + "*" + b;
        }
    }

    else {
        internal_error << "\n"
                       << "StmtSizes::get_simplified_string - Unsupported operator: " << op << "\n";
        return "";
    }
}

void StmtSizes::set_write_size(const IRNode *node, string write_var, string write_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].writes[write_var] = write_size;
}
void StmtSizes::set_read_size(const IRNode *node, string read_var, string read_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].reads[read_var] = read_size;
}

void StmtSizes::visit(const Store *op) {

    // TODO: is this correct?
    uint16_t lanes = op->index.type().lanes();

    set_write_size(op, op->name, int_span(lanes));

    // empty curr_load_values
    curr_load_values.clear();
    op->value.accept(this);

    // set consume (for now, read values)
    for (const auto &load_var : curr_load_values) {
        set_read_size(op, load_var.first, int_span(load_var.second));
    }
}
void StmtSizes::add_load_value(const string &name, const int lanes) {
    auto it = curr_load_values.find(name);
    if (it == curr_load_values.end()) {
        curr_load_values[name] = lanes;
    } else {
        curr_load_values[name] += lanes;
    }
}
void StmtSizes::visit(const Load *op) {

    // TODO: make sure this logic is right
    int lanes = int(op->type.lanes());

    add_load_value(op->name, lanes);
}

/*
 * IRVisualization class
 */
string IRVisualization::generate_ir_visualization_html(const Module &m) {
    pre_processor.generate_sizes(m);

    html.str("");
    numOfNodes = 0;
    startModuleTraversal(m);

    return html.str();
}

void IRVisualization::startModuleTraversal(const Module &m) {

    // print main function first
    for (const auto &f : m.functions()) {
        if (f.name == m.name()) {
            visit_function(f);
        }
    }

    // print the rest of the functions
    for (const auto &f : m.functions()) {
        if (f.name != m.name()) {
            visit_function(f);
        }
    }
}

string IRVisualization::open_box_div(string className, const IRNode *op) {
    stringstream ss;

    ss << "<div class='box center " << className << "'";
    ss << ">";

    if (op != nullptr) {
        ss << generate_computation_cost_div(op);
        ss << generate_memory_cost_div(op);
    }

    ss << open_content_div();
    return ss.str();
}
string IRVisualization::close_box_div() const {
    stringstream ss;
    ss << close_div();  // body div (opened at end of close_header())
    ss << close_div();  // content div
    ss << close_div();  // main box div
    return ss.str();
}
string IRVisualization::open_function_box_div() const {
    return "<div class='center FunctionBox'> <div class='functionContent'>";
}
string IRVisualization::close_function_box_div() const {
    stringstream ss;
    ss << close_div();  // content div
    ss << close_div();  // main box div
    return ss.str();
}
string IRVisualization::open_header_div() const {
    return "<div class='boxHeader'>";
}
string IRVisualization::open_box_header_title_div() const {
    return "<div class='boxHeaderTitle'>";
}
string IRVisualization::open_box_header_table_div() const {
    return "<div class='boxHeaderTable'>";
}
string IRVisualization::open_store_div() const {
    return "<div class='store'>";
}
string IRVisualization::close_div() const {
    return "</div>";
}

string IRVisualization::open_header(const string &header, string anchorName) {
    stringstream ss;
    ss << open_header_div();

    numOfNodes++;

    // buttons div
    ss << "<div class='collapseExpandButtons'>";

    // expand button - hidden to start
    ss << "<button id='irViz" << numOfNodes
       << "-show' class='iconButton irVizToggle' onclick='toggleCollapse(" << numOfNodes
       << ")' style='display: none;'><i class='bi "
       << "bi-chevron-bar-down'></i></button>";

    // collapse button
    ss << "<button id='irViz" << numOfNodes
       << "-hide' class='iconButton irVizToggle' onclick='toggleCollapse(" << numOfNodes
       << ")' ><i class='bi bi-chevron-bar-up'></i></button>"
       << "</div>";

    ss << open_box_header_title_div();

    ss << "<span id='" << anchorName << "_viz'>";
    ss << header;
    ss << "</span>";

    ss << close_div();

    // spacing purposes
    ss << "<div class='spacing'></div>";

    ss << open_box_header_table_div();

    return ss.str();
}
string IRVisualization::close_header(string anchorName) const {
    stringstream ss;

    ss << close_div();  // header table div
    ss << see_code_button_div(anchorName);
    ss << close_div();  // header div

    // open body div
    ss << "<div id='irViz" << numOfNodes << "' class='boxBody'>";

    return ss.str();
}
string IRVisualization::div_header(const string &header, StmtSize *size, string anchorName) {
    stringstream ss;

    ss << open_header(header, anchorName);

    // add producer consumer size if size is provided
    if (size != nullptr) {
        ss << read_write_table(*size);
    }

    ss << close_header(anchorName);

    return ss.str();
}
string IRVisualization::function_div_header(const string &functionName, string anchorName) const {
    stringstream ss;

    ss << "<div class='functionHeader'>";

    ss << "<span id='" << functionName << "'>";
    ss << "<span id='" << anchorName << "_viz' style='display: inline-block;'>";
    ss << "<h4 style='margin-bottom: 0px;'> Func: " << functionName << "</h4>";
    ss << "</span>";
    ss << "</span>";

    ss << see_code_button_div(anchorName, false);

    ss << "</div>";

    return ss.str();
}
vector<string> IRVisualization::get_allocation_sizes(const Allocate *op) const {
    vector<string> sizes;

    stringstream type;
    type << "<span class='stringType'>" << op->type << "</span>";
    sizes.push_back(type.str());

    for (const auto &extent : op->extents) {
        stringstream ss;
        if (extent.as<IntImm>()) {
            ss << "<span class='intType'>" << extent << "</span>";
        } else {
            ss << "<span class='stringType'>" << extent << "</span>";
        }

        sizes.push_back(ss.str());
    }

    internal_assert(sizes.size() == op->extents.size() + 1);

    return sizes;
}
string IRVisualization::allocate_div_header(const Allocate *op, const string &header,
                                            string anchorName) {
    stringstream ss;

    ss << open_header(header, anchorName);

    vector<string> allocationSizes = get_allocation_sizes(op);
    ss << allocate_table(allocationSizes);

    ss << close_header(anchorName);

    return ss.str();
}
string IRVisualization::for_loop_div_header(const For *op, const string &header,
                                            string anchorName) {
    stringstream ss;

    ss << open_header(header, anchorName);

    string loopSize = get_loop_iterator(op);
    ss << for_loop_table(loopSize);

    ss << close_header(anchorName);

    return ss.str();
}

string IRVisualization::if_tree(const IRNode *op, const string &header, string anchorName) {
    stringstream ss;

    ss << "<li>";
    ss << "<span class='tf-nc if-node'>";

    ss << open_box_div("IfBox", op);
    ss << div_header(header, nullptr, anchorName);

    return ss.str();
}
string IRVisualization::close_if_tree() const {
    stringstream ss;
    ss << close_box_div();
    ss << "</span>";
    ss << "</li>";
    return ss.str();
}

string IRVisualization::read_write_table(StmtSize &size) const {
    stringstream readWriteTable;

    // open table
    readWriteTable << "<table class='costTable'>";

    // Prod | Cons
    readWriteTable << "<tr>";

    readWriteTable << "<th colspan='2' class='costTableHeader middleCol'>";
    readWriteTable << "Written";
    readWriteTable << "</th>";

    readWriteTable << "<th colspan='2' class='costTableHeader'>";
    readWriteTable << "Read";
    readWriteTable << "</th>";

    readWriteTable << "</tr>";

    // produces and consumes are empty
    if (size.empty()) {
        internal_error << "\n\n"
                       << "IRVisualization::read_write_table - size is empty"
                       << "\n";
    }

    // produces and consumes aren't empty
    else {
        vector<string> rows;

        // fill in producer variables
        for (const auto &produce_var : size.writes) {
            string ss;
            ss += "<td class='costTableData'>";
            ss += produce_var.first + ": ";
            ss += "</td>";

            ss += "<td class='costTableData middleCol'>";
            ss += produce_var.second;
            ss += "</td>";

            rows.push_back(ss);
        }

        // fill in consumer variables
        unsigned long rowNum = 0;
        for (const auto &consume_var : size.reads) {
            string ss;
            ss += "<td class='costTableData'>";
            ss += consume_var.first + ": ";
            ss += "</td>";

            ss += "<td class='costTableData'>";
            ss += consume_var.second;
            ss += "</td>";

            if (rowNum < rows.size()) {
                rows[rowNum] += ss;
            } else {
                // pad row with empty cells for produce
                string sEmpty;
                sEmpty += "<td colspan='2' class='costTableData middleCol'>";
                sEmpty += "</td>";

                rows.push_back(sEmpty + ss);
            }
            rowNum++;
        }

        // pad row with empty calls for consume
        rowNum = size.reads.size();
        while (rowNum < size.writes.size()) {
            string sEmpty;
            sEmpty += "<td class='costTableData'>";
            sEmpty += "</td>";
            sEmpty += "<td class='costTableData'>";
            sEmpty += "</td>";

            rows[rowNum] += sEmpty;
            rowNum++;
        }

        // add rows to readWriteTable
        for (const auto &row : rows) {
            readWriteTable << "<tr>";
            readWriteTable << row;
            readWriteTable << "</tr>";
        }
    }

    // close table
    readWriteTable << "</table>";

    return readWriteTable.str();
}
string IRVisualization::allocate_table(vector<string> &allocationSizes) const {
    stringstream allocateTable;

    // open table
    allocateTable << "<table class='costTable'>";

    // open header and data rows
    stringstream header;
    stringstream data;

    header << "<tr>";
    data << "<tr>";

    // iterate through all allocation sizes and add them to the header and data rows
    for (unsigned long i = 0; i < allocationSizes.size(); i++) {
        if (i == 0) {
            header << "<th class='costTableHeader middleCol'>";
            header << "Type";
            header << "</th>";

            data << "<td class='costTableHeader middleCol'>";
            data << allocationSizes[0];
            data << "</td>";
        } else {
            if (i < allocationSizes.size() - 1) {
                header << "<th class='costTableHeader middleCol'>";
                data << "<td class='costTableHeader middleCol'>";
            } else {
                header << "<th class='costTableHeader'>";
                data << "<td class='costTableHeader'>";
            }
            header << "Dim-" + std::to_string(i);
            header << "</th>";

            data << allocationSizes[i];
            data << "</td>";
        }
    }

    // close header and data rows
    header << "</tr>";
    data << "</tr>";

    // add header and data rows to allocateTable
    allocateTable << header.str();
    allocateTable << data.str();

    // close table
    allocateTable << "</table>";

    return allocateTable.str();
}
string IRVisualization::for_loop_table(string loop_size) const {
    stringstream forLoopTable;

    // open table
    forLoopTable << "<table class='costTable'>";

    // Loop Size
    forLoopTable << "<tr>";

    forLoopTable << "<th class='costTableHeader'>";
    forLoopTable << "Loop Span";
    forLoopTable << "</th>";

    forLoopTable << "</tr>";

    forLoopTable << "<tr>";

    // loop size
    forLoopTable << "<td class='costTableData'>";
    forLoopTable << loop_size;
    forLoopTable << "</td>";

    forLoopTable << "</tr>";

    // close table
    forLoopTable << "</table>";

    return forLoopTable.str();
}

string IRVisualization::see_code_button_div(string anchorName, bool putDiv) const {
    stringstream ss;
    if (putDiv) ss << "<div>";
    ss << "<button class='iconButton'";
    ss << "onclick='scrollToFunctionVizToCode(\"" << anchorName << "\")'>";
    ss << "<i class='bi bi-code-square'></i>";
    ss << "</button>";
    if (putDiv) ss << "</div>";
    return ss.str();
}

string IRVisualization::info_tooltip(string toolTipText, string className = "") {
    stringstream ss;

    // info-button
    irVizTooltipCount++;
    ss << "<button id='irVizButton" << irVizTooltipCount << "' ";
    ss << "aria-describedby='irVizTooltip" << irVizTooltipCount << "' ";
    ss << "class='info-button' role='button' ";
    ss << ">";
    ss << "<i class='bi bi-info'></i>";
    ss << "</button>";

    // tooltip span
    ss << "<span id='irVizTooltip" << irVizTooltipCount << "' ";
    ss << "class='tooltip";
    if (className != "") {
        ss << " " + className;
    }
    ss << "'";
    ss << "role='irVizTooltip" << irVizTooltipCount << "'>";
    ss << toolTipText;
    ss << "</span>";

    return ss.str();
}

string IRVisualization::generate_computation_cost_div(const IRNode *op) {
    stringstream ss;

    // skip if it's a store
    if (op->node_type == IRNodeType::Store) return "";

    irVizTooltipCount++;

    string tooltipText = findStmtCost.generate_computation_cost_tooltip(op, true, "");

    // tooltip span
    ss << "<span id='irVizTooltip" << irVizTooltipCount << "' class='tooltip CostTooltip' ";
    ss << "role='irVizTooltip" << irVizTooltipCount << "'>";
    ss << tooltipText;
    ss << "</span>";

    int computation_range = findStmtCost.get_computation_color_range(op, true);
    string className = "computation-cost-div CostColor" + std::to_string(computation_range);
    ss << "<div id='irVizButton" << irVizTooltipCount << "' ";
    ss << "aria-describedby='irVizTooltip" << irVizTooltipCount << "' ";
    ss << "class='" << className << "'>";

    ss << close_div();

    return ss.str();
}
string IRVisualization::generate_memory_cost_div(const IRNode *op) {
    stringstream ss;

    // skip if it's a store
    if (op->node_type == IRNodeType::Store) return "";

    irVizTooltipCount++;

    string tooltipText = findStmtCost.generate_data_movement_cost_tooltip(op, true, "");

    // tooltip span
    ss << "<span id='irVizTooltip" << irVizTooltipCount << "' class='tooltip CostTooltip' ";
    ss << "role='irVizTooltip" << irVizTooltipCount << "'>";
    ss << tooltipText;
    ss << "</span>";

    int data_movement_range = findStmtCost.get_data_movement_color_range(op, true);
    string className = "memory-cost-div CostColor" + std::to_string(data_movement_range);
    ss << "<div id='irVizButton" << irVizTooltipCount << "' ";
    ss << "aria-describedby='irVizTooltip" << irVizTooltipCount << "' ";
    ss << "class='" << className << "'>";

    ss << close_div();

    return ss.str();
}
string IRVisualization::open_content_div() const {
    return "<div class='content'>";
}

string IRVisualization::color_button(int colorRange) {
    stringstream ss;

    irVizTooltipCount++;
    ss << "<button id='irVizButton" << irVizTooltipCount << "' ";
    ss << "aria-describedby='irVizTooltip" << irVizTooltipCount << "' ";
    ss << "class='irVizColorButton CostColor" << colorRange << "' role='button' ";
    ss << ">";
    ss << "</button>";

    return ss.str();
}

string IRVisualization::computation_div(const IRNode *op) {
    // want exclusive cost (so that the colors match up with exclusive costs)
    int computation_range = findStmtCost.get_computation_color_range(op, false);

    stringstream ss;
    ss << color_button(computation_range);

    string tooltipText = findStmtCost.generate_computation_cost_tooltip(op, false, "");

    // tooltip span
    ss << "<span id='irVizTooltip" << irVizTooltipCount << "' class='tooltip CostTooltip' ";
    ss << "role='irVizTooltip" << irVizTooltipCount << "'>";
    ss << tooltipText;
    ss << "</span>";

    return ss.str();
}
string IRVisualization::data_movement_div(const IRNode *op) {
    // want exclusive cost (so that the colors match up with exclusive costs)
    int data_movement_range = findStmtCost.get_data_movement_color_range(op, false);

    stringstream ss;
    ss << color_button(data_movement_range);

    string tooltipText = findStmtCost.generate_data_movement_cost_tooltip(op, false, "");

    // tooltip span
    ss << "<span id='irVizTooltip" << irVizTooltipCount << "' class='tooltip CostTooltip' ";
    ss << "role='irVizTooltip" << irVizTooltipCount << "'>";
    ss << tooltipText;
    ss << "</span>";

    return ss.str();
}
string IRVisualization::tooltip_table(vector<pair<string, string>> &table) const {
    stringstream ss;
    ss << "<table class='tooltipTable'>";
    for (auto &row : table) {
        ss << "<tr>";
        ss << "<td class = 'left-table'>" << row.first << "</td>";
        ss << "<td class = 'right-table'> " << row.second << "</td>";
        ss << "</tr>";
    }
    ss << "</table>";
    return ss.str();
}
string IRVisualization::cost_colors(const IRNode *op) {
    stringstream ss;
    ss << computation_div(op);
    ss << data_movement_div(op);
    return ss.str();
}

void IRVisualization::visit_function(const LoweredFunc &func) {
    html << open_function_box_div();

    functionCount++;
    string anchorName = "loweredFunc" + std::to_string(functionCount);

    html << function_div_header(func.name, anchorName);

    html << "<div class='functionViz'>";
    func.body.accept(this);
    html << "</div>";

    close_function_box_div();
}
void IRVisualization::visit(const Variable *op) {
    // if op->name starts with "::", remove "::"
    string varName = op->name;
    if (varName[0] == ':' && varName[1] == ':') {
        varName = varName.substr(2);
    }

    // see if varName is in pre_processor.function_names
    if (std::count(pre_processor.function_names.begin(), pre_processor.function_names.end(),
                   varName)) {

        html << "<div class='box center FunctionCallBox'>";

        html << "Function Call";
        html << "<button class='function-scroll-button' role='button' ";
        html << "onclick='scrollToFunctionCodeToViz(\"" << varName << "\")'>";

        html << varName;
        html << "</button>";

        html << "</div>";
    }
}
void IRVisualization::visit(const ProducerConsumer *op) {
    html << open_box_div("ProducerConsumerBox", op);

    producerConsumerCount++;
    string anchorName = "producerConsumer" + std::to_string(producerConsumerCount);

    string header = (op->is_producer ? "Produce" : "Consume");
    header += " " + op->name;

    html << div_header(header, nullptr, anchorName);

    op->body.accept(this);

    html << close_box_div();
}
string IRVisualization::get_loop_iterator(const For *op) const {
    Expr min = op->min;
    Expr extent = op->extent;

    string loopIterator;

    // check if min and extend are of type IntImm
    if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::IntImm) {
        int64_t minValue = min.as<IntImm>()->value;
        int64_t extentValue = extent.as<IntImm>()->value;
        uint16_t range = uint16_t(extentValue - minValue);
        loopIterator = pre_processor.int_span(range);
    }

    else if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::Variable) {
        int64_t minValue = min.as<IntImm>()->value;
        string minName = pre_processor.int_span(minValue);
        string extentName = pre_processor.string_span(extent.as<Variable>()->name);

        if (minValue == 0) {
            loopIterator = extentName;
        } else {
            loopIterator = "(" + extentName + " - " + minName + ")";
        }
    }

    else if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::Add) {
        int64_t minValue = min.as<IntImm>()->value;
        string minName = pre_processor.int_span(minValue);
        string extentName = "(";

        // deal with a
        if (extent.as<Add>()->a.node_type() == IRNodeType::IntImm) {
            int64_t extentValue = extent.as<Add>()->a.as<IntImm>()->value;
            extentName += pre_processor.int_span(extentValue);
        } else if (extent.as<Add>()->a.node_type() == IRNodeType::Variable) {
            extentName += pre_processor.string_span(extent.as<Add>()->a.as<Variable>()->name);
        } else {
            internal_error << "\n"
                           << "In for loop: " << op->name << "\n"
                           << pre_processor.print_node(extent.as<Add>()->a.get()) << "\n"
                           << "StmtSizes::visit(const For *op): add->a isn't IntImm or Variable - "
                              "can't generate irViz hierarchy yet. \n\n";
        }

        extentName += "+";

        // deal with b
        if (extent.as<Add>()->b.node_type() == IRNodeType::IntImm) {
            int64_t extentValue = extent.as<Add>()->b.as<IntImm>()->value;
            extentName += pre_processor.int_span(extentValue);
        } else if (extent.as<Add>()->b.node_type() == IRNodeType::Variable) {
            extentName += pre_processor.string_span(extent.as<Add>()->b.as<Variable>()->name);
        } else {
            internal_error << "\n"
                           << "In for loop: " << op->name << "\n"
                           << pre_processor.print_node(extent.as<Add>()->b.get()) << "\n"
                           << "StmtSizes::visit(const For *op): add->b isn't IntImm or Variable - "
                              "can't generate irViz hierarchy yet. \n\n";
        }
        extentName += ")";

        if (minValue == 0) {
            loopIterator = extentName;
        } else {
            loopIterator = "(" + extentName + " - " + minName + ")";
        }

    }

    else {
        internal_error
            << "\n"
            << "In for loop: " << op->name << "\n"
            << pre_processor.print_node(op->min.get()) << "\n"
            << pre_processor.print_node(op->extent.get()) << "\n"
            << "StmtSizes::visit(const For *op): min and extent are not of type (IntImm) "
               "or (IntImm & Variable) or (IntImm & Add) - "
               "can't generate irViz hierarchy yet. \n\n";
    }

    return loopIterator;
}
void IRVisualization::visit(const For *op) {
    html << open_box_div("ForBox", op);

    forCount++;
    string anchorName = "for" + std::to_string(forCount);

    string header = "For (" + op->name + ")";

    html << for_loop_div_header(op, header, anchorName);

    op->body.accept(this);

    html << close_box_div();
}
void IRVisualization::visit(const IfThenElse *op) {
    // open main if tree
    html << "<div class='tf-tree tf-gap-sm tf-custom-irViz'>";
    html << "<ul>";
    html << "<li><span class='tf-nc if-node'>";
    html << "If";
    html << "</span>";
    html << "<ul>";

    string ifHeader;
    ifHeader += "if ";

    // anchor name
    ifCount++;
    string anchorName = "if" + std::to_string(ifCount);

    while (true) {
        stringstream condition;
        condition << op->condition;

        string conditionString = condition.str();
        // make condition smaller if it's too big
        if (conditionString.size() > MAX_CONDITION_LENGTH) {
            condition.str("");
            condition << "...";
            condition << info_tooltip(conditionString, "conditionTooltip");
        }

        ifHeader += condition.str();

        html << if_tree(op, ifHeader, anchorName);

        // then body
        op->then_case.accept(this);

        html << close_if_tree();

        // if there is no else case, we are done
        if (!op->else_case.defined()) {
            break;
        }

        // if else case is another ifthenelse, recurse and reset op to else case
        if (const IfThenElse *nested_if = op->else_case.as<IfThenElse>()) {
            op = nested_if;
            ifHeader = "";
            ifHeader += "else if ";

            // anchor name
            ifCount++;
            anchorName = "if" + std::to_string(ifCount);

        }

        // if else case is not another ifthenelse
        else {

            string elseHeader;
            elseHeader += "else ";

            // anchor name
            ifCount++;
            anchorName = "if" + std::to_string(ifCount);

            html << if_tree(op->else_case.get(), elseHeader, anchorName);

            op->else_case.accept(this);

            html << close_if_tree();
            break;
        }
    }

    // close main if tree
    html << "</ul>";
    html << "</li>";
    html << "</ul>";
    html << "</div>";
}
void IRVisualization::visit(const Store *op) {
    StmtSize size = pre_processor.get_size(op);

    storeCount++;
    string anchorName = "store" + std::to_string(storeCount);

    string header = "Store " + op->name;

    vector<pair<string, string>> tableRows;
    tableRows.push_back({"Vector Size", std::to_string(op->index.type().lanes())});
    tableRows.push_back({"Bit Size", std::to_string(op->index.type().bits())});

    header += info_tooltip(tooltip_table(tableRows));

    html << open_box_div("StoreBox", op);

    html << div_header(header, &size, anchorName);

    op->value.accept(this);

    html << close_box_div();
}
void IRVisualization::visit(const Load *op) {
    string header;
    vector<pair<string, string>> tableRows;

    if (op->type.is_scalar()) {
        header = "Scalar ";
    }

    else if (op->type.is_vector()) {
        if (op->index.node_type() == IRNodeType::Ramp) {
            const Ramp *ramp = op->index.as<Ramp>();

            tableRows.push_back({"Ramp lanes", std::to_string(ramp->lanes)});
            stringstream rampStride;
            rampStride << ramp->stride;
            tableRows.push_back({"Ramp stride", rampStride.str()});

            if (ramp->stride.node_type() == IRNodeType::IntImm) {
                int64_t stride = ramp->stride.as<IntImm>()->value;
                if (stride == 1) {
                    header = "Dense vector ";
                } else {
                    header = "Strided vector ";
                }
            } else {
                header = "Dense vector ";
            }
        } else {
            header = "Dense vector ";
        }
    }

    else {
        internal_error << "\n\nUnsupported type for Load: " << op->type << "\n\n";
    }

    header += "load " + op->name;

    if (findStmtCost.is_local_variable(op->name)) {
        tableRows.push_back({"Variable Type", "local var"});
    } else {
        tableRows.push_back({"Variable Type", "global var"});
    }

    tableRows.push_back({"Bit Size", std::to_string(op->index.type().bits())});
    tableRows.push_back({"Vector Size", std::to_string(op->index.type().lanes())});

    if (op->param.defined()) {
        tableRows.push_back({"Parameter", op->param.name()});
    }

    header += info_tooltip(tooltip_table(tableRows));

    html << open_store_div();
    html << cost_colors(op);
    html << header;
    html << close_div();
}
string IRVisualization::get_memory_type(MemoryType memType) const {
    if (memType == MemoryType::Auto) {
        return "Auto";
    } else if (memType == MemoryType::Heap) {
        return "Heap";
    } else if (memType == MemoryType::Stack) {
        return "Stack";
    } else if (memType == MemoryType::Register) {
        return "Register";
    } else if (memType == MemoryType::GPUShared) {
        return "GPUShared";
    } else if (memType == MemoryType::GPUTexture) {
        return "GPUTexture";
    } else if (memType == MemoryType::LockedCache) {
        return "LockedCache";
    } else if (memType == MemoryType::VTCM) {
        return "VTCM";
    } else if (memType == MemoryType::AMXTile) {
        return "AMXTile";
    } else {
        internal_error << "\n\n"
                       << "Unknown memory type"
                       << "\n";
        return "Unknown Memory Type";
    }
}
void IRVisualization::visit(const Allocate *op) {
    html << open_box_div("AllocateBox", op);

    allocateCount++;
    string anchorName = "allocate" + std::to_string(allocateCount);

    string header = "Allocate " + op->name;

    vector<pair<string, string>> tableRows;
    tableRows.push_back({"Memory Type", get_memory_type(op->memory_type)});

    if (!is_const_one(op->condition)) {
        stringstream conditionString;
        conditionString << op->condition;
        tableRows.push_back({"Condition", conditionString.str()});
    }
    if (op->new_expr.defined()) {
        internal_error << "\n"
                       << "IRVisualization: Allocate " << op->name
                       << " `op->new_expr.defined()` is not supported.\n\n";

        stringstream newExprString;
        newExprString << op->new_expr;
        tableRows.push_back({"New Expr", newExprString.str()});
    }
    if (!op->free_function.empty()) {
        internal_error << "\n"
                       << "IRVisualization: Allocate " << op->name
                       << " `!op->free_function.empty()` is not supported.\n\n";

        stringstream freeFunctionString;
        freeFunctionString << op->free_function;
        tableRows.push_back({"Free Function", freeFunctionString.str()});
    }

    tableRows.push_back({"Bit Size", std::to_string(op->type.bits())});
    tableRows.push_back({"Vector Size", std::to_string(op->type.lanes())});

    header += info_tooltip(tooltip_table(tableRows));

    html << allocate_div_header(op, header, anchorName);

    op->body.accept(this);

    html << close_box_div();
}

string IRVisualization::generate_irViz_js() {
    stringstream irVizJS;

    irVizJS << "\n// irViz JS\n"
            << "for (let i = 1; i <= " << irVizTooltipCount << "; i++) { \n"
            << "    const button = document.getElementById('irVizButton' + i); \n"
            << "    const tooltip = document.getElementById('irVizTooltip' + i); \n"
            << "    button.addEventListener('mouseenter', () => { \n"
            << "        showTooltip(button, tooltip); \n"
            << "    }); \n"
            << "    button.addEventListener('mouseleave', () => { \n"
            << "        hideTooltip(tooltip); \n"
            << "    } \n"
            << "    ); \n"
            << "    tooltip.addEventListener('focus', () => { \n"
            << "        showTooltip(button, tooltip); \n"
            << "    } \n"
            << "    ); \n"
            << "    tooltip.addEventListener('blur', () => { \n"
            << "        hideTooltip(tooltip); \n"
            << "    } \n"
            << "    ); \n"
            << "} \n"
            << "function toggleCollapse(id) {\n "
            << "    var buttonShow = document.getElementById('irViz' + id + '-show');\n"
            << "    var buttonHide = document.getElementById('irViz' + id + '-hide');\n"
            << "    var body = document.getElementById('irViz' + id);\n"
            << "    if (body.style.visibility != 'hidden') {\n"
            << "        body.style.visibility = 'hidden';\n"
            << "        body.style.height = '0px';\n"
            << "        body.style.width = '0px';\n"
            << "        buttonShow.style.display = 'block';\n"
            << "        buttonHide.style.display = 'none';\n"
            << "    } else {\n"
            << "        body.style = '';\n"
            << "        buttonShow.style.display = 'none';\n"
            << "        buttonHide.style.display = 'block';\n"
            << "    }\n"
            << "}\n ";

    return irVizJS.str();
}

/*
 * PRINT NODE
 */
string StmtSizes::print_node(const IRNode *node) const {
    stringstream ss;
    ss << "Node in question has type: ";
    IRNodeType type = node->node_type;
    if (type == IRNodeType::IntImm) {
        ss << "IntImm type";
        auto node1 = dynamic_cast<const IntImm *>(node);
        ss << "value: " << node1->value;
    } else if (type == IRNodeType::UIntImm) {
        ss << "UIntImm type";
    } else if (type == IRNodeType::FloatImm) {
        ss << "FloatImm type";
    } else if (type == IRNodeType::StringImm) {
        ss << "StringImm type";
    } else if (type == IRNodeType::Broadcast) {
        ss << "Broadcast type";
    } else if (type == IRNodeType::Cast) {
        ss << "Cast type";
    } else if (type == IRNodeType::Variable) {
        ss << "Variable type";
    } else if (type == IRNodeType::Add) {
        ss << "Add type";
        auto node1 = dynamic_cast<const Add *>(node);
        ss << "a: " << print_node(node1->a.get()) << endl;
        ss << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Sub) {
        ss << "Sub type" << endl;
        auto node1 = dynamic_cast<const Sub *>(node);
        ss << "a: " << print_node(node1->a.get()) << endl;
        ss << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Mod) {
        ss << "Mod type" << endl;
        auto node1 = dynamic_cast<const Mod *>(node);
        ss << "a: " << print_node(node1->a.get()) << endl;
        ss << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Mul) {
        ss << "Mul type" << endl;
        auto node1 = dynamic_cast<const Mul *>(node);
        ss << "a: " << print_node(node1->a.get()) << endl;
        ss << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Div) {
        ss << "Div type" << endl;
        auto node1 = dynamic_cast<const Div *>(node);
        ss << "a: " << print_node(node1->a.get()) << endl;
        ss << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Min) {
        ss << "Min type";
    } else if (type == IRNodeType::Max) {
        ss << "Max type";
    } else if (type == IRNodeType::EQ) {
        ss << "EQ type";
    } else if (type == IRNodeType::NE) {
        ss << "NE type";
    } else if (type == IRNodeType::LT) {
        ss << "LT type";
    } else if (type == IRNodeType::LE) {
        ss << "LE type";
    } else if (type == IRNodeType::GT) {
        ss << "GT type";
    } else if (type == IRNodeType::GE) {
        ss << "GE type";
    } else if (type == IRNodeType::And) {
        ss << "And type";
    } else if (type == IRNodeType::Or) {
        ss << "Or type";
    } else if (type == IRNodeType::Not) {
        ss << "Not type";
    } else if (type == IRNodeType::Select) {
        ss << "Select type";
    } else if (type == IRNodeType::Load) {
        ss << "Load type";
    } else if (type == IRNodeType::Ramp) {
        ss << "Ramp type";
    } else if (type == IRNodeType::Call) {
        ss << "Call type";
    } else if (type == IRNodeType::Let) {
        ss << "Let type";
    } else if (type == IRNodeType::Shuffle) {
        ss << "Shuffle type";
    } else if (type == IRNodeType::VectorReduce) {
        ss << "VectorReduce type";
    } else if (type == IRNodeType::LetStmt) {
        ss << "LetStmt type";
    } else if (type == IRNodeType::AssertStmt) {
        ss << "AssertStmt type";
    } else if (type == IRNodeType::ProducerConsumer) {
        ss << "ProducerConsumer type";
    } else if (type == IRNodeType::For) {
        ss << "For type";
    } else if (type == IRNodeType::Acquire) {
        ss << "Acquire type";
    } else if (type == IRNodeType::Store) {
        ss << "Store type";
    } else if (type == IRNodeType::Provide) {
        ss << "Provide type";
    } else if (type == IRNodeType::Allocate) {
        ss << "Allocate type";
    } else if (type == IRNodeType::Free) {
        ss << "Free type";
    } else if (type == IRNodeType::Realize) {
        ss << "Realize type";
    } else if (type == IRNodeType::Block) {
        ss << "Block type";
    } else if (type == IRNodeType::Fork) {
        ss << "Fork type";
    } else if (type == IRNodeType::IfThenElse) {
        ss << "IfThenElse type";
    } else if (type == IRNodeType::Evaluate) {
        ss << "Evaluate type";
    } else if (type == IRNodeType::Prefetch) {
        ss << "Prefetch type";
    } else if (type == IRNodeType::Atomic) {
        ss << "Atomic type";
    } else {
        ss << "Unknown type";
    }

    return ss.str();
}

const string IRVisualization::scrollToFunctionJSVizToCode = "\n \
// scroll to function - viz to code\n \
function makeVisible(element) { \n \
    if (!element) return; \n \
    if (element.className == 'mainContent') return; \n \
    if (element.style.visibility == 'hidden') { \n \
        element.style = ''; \n \
        show = document.getElementById(element.id + '-show'); \n \
        hide = document.getElementById(element.id + '-hide'); \n \
        show.style.display = 'none'; \n \
        hide.style.display = 'block'; \n \
        return; \n \
    } \n \
    makeVisible(element.parentNode); \n \
} \n \
 \n \
function scrollToFunctionVizToCode(id) { \n \
    var container = document.getElementById('IRCode-code'); \n \
    var scrollToObject = document.getElementById(id); \n \
    makeVisible(scrollToObject); \n \
    container.scrollTo({ \n \
        top: scrollToObject.offsetTop - 10, \n \
        behavior: 'smooth' \n \
    }); \n \
    scrollToObject.style.backgroundColor = 'yellow'; \n \
    scrollToObject.style.fontSize = '20px'; \n \
 \n \
    // change content for 1 second   \n \
    setTimeout(function () { \n \
        scrollToObject.style.backgroundColor = 'transparent'; \n \
        scrollToObject.style.fontSize = '12px'; \n \
    }, 1000); \n \
} \n \
";

const string IRVisualization::irVizCSS = "\n \
/* irViz CSS */\n \
.tf-custom-irViz .tf-nc { border-radius: 5px; border: 1px solid; }\n \
.tf-custom-irViz .tf-nc:before, .tf-custom-irViz .tf-nc:after { border-left-width: 1px; }\n \
.tf-custom-irViz li li:before { border-top-width: 1px; }\n \
.tf-custom-irViz .end-node { border-style: dashed; }\n \
.tf-custom-irViz .tf-nc { background-color: #e6eeff; }\n \
.tf-custom-irViz { font-size: 12px; } \n \
div.box { \n \
    border: 1px dashed grey; \n \
    border-radius: 5px; \n \
    margin: 5px; \n \
    padding: 5px; \n \
    display: flex; \n \
    width: max-content; \n \
} \n \
div.boxHeader { \n \
    padding: 5px; \n \
    display: flex; \n \
} \n \
div.memory-cost-div, \n \
div.computation-cost-div { \n \
    border: 1px solid rgba(0, 0, 0, 0); \n \
     width: 7px; \n \
} \n \
div.FunctionCallBox { \n \
    background-color: #fabebe; \n \
} \n \
div.FunctionBox { \n \
    background-color: #f0f0f0; \n \
    border: 1px dashed grey; \n \
    border-radius: 5px; \n \
    margin-bottom: 15px; \n \
    padding: 5px; \n \
    width: max-content; \n \
} \n \
div.functionHeader { \n \
    display: flex; \n \
    margin-bottom: 10px; \n \
} \n \
div.ProducerConsumerBox { \n \
    background-color: #99bbff; \n \
} \n \
div.ForBox { \n \
    background-color: #b3ccff; \n \
} \n \
div.StoreBox { \n \
    background-color: #f4f8bf; \n \
} \n \
div.AllocateBox { \n \
    background-color: #f4f8bf; \n \
} \n \
div.IfBox { \n \
    background-color: #e6eeff; \n \
} \n \
div.memory-cost-div:hover, \n \
div.computation-cost-div:hover { \n \
    border: 1px solid grey; \n \
} \n \
div.spacing { \n \
    flex-grow: 1; \n \
} \n \
table { \n \
    border-radius: 5px; \n \
    font-size: 12px; \n \
    border: 1px dashed grey; \n \
    border-collapse: separate; \n \
    border-spacing: 0; \n \
} \n \
.ifElseTable { \n \
    border: 0px; \n \
}  \n \
.costTable { \n \
    float: right; \n \
    text-align: center; \n \
    border: 0px; \n \
    background-color: rgba(150, 150, 150, 0.5); \n \
} \n \
.costTable td { \n \
    border-top: 1px dashed grey; \n \
} \n \
.costTableHeader, \n \
.costTableData { \n \
    border-collapse: collapse; \n \
    padding-top: 1px; \n \
    padding-bottom: 1px; \n \
    padding-left: 5px; \n \
    padding-right: 5px; \n \
} \n \
span.intType { color: #099; } \n \
span.stringType { color: #990073; } \n \
.middleCol { \n \
    border-right: 1px dashed grey; \n \
} \n \
div.content { \n \
    flex-grow: 1; \n \
} \n \
.irVizColorButton { \n \
    height: 15px; \n \
    width: 10px; \n \
    margin-right: 2px; \n \
    border: 1px solid rgba(0, 0, 0, 0); \n \
    vertical-align: middle; \n \
    border-radius: 2px; \n \
} \n \
.irVizColorButton:hover { \n \
    border: 1px solid grey; \n \
} \n \
div.boxHeaderTitle { \n \
    font-weight: bold; \n \
} \n \
.irVizToggle { \n \
    margin-right: 5px; \n \
} \n \
";
