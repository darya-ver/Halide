#include "ProducerConsumerHierarchy.h"
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
void StmtSizes::generate_sizes(const Stmt &stmt) {
    mutate(stmt);
}

StmtSize StmtSizes::get_size(const IRNode *node) const {
    auto it = stmt_sizes.find(node);

    // return empty size if not found (this just means it wasn't set)
    if (it == stmt_sizes.end()) {
        return StmtSize();
    }

    return it->second;
}
string StmtSizes::get_allocation_size(const IRNode *node, const string &name) const {
    StmtSize size = get_size(node);

    auto it = size.allocates.find(name);
    if (it == size.allocates.end()) {
        internal_error << "\n"
                       << print_node(node) << "\n"
                       << "StmtSizes::get_allocation_size - " << name << " not found in allocates\n"
                       << "\n\n";
    }
    return it->second;
}

void StmtSizes::traverse(const Module &m) {
    module_name = m.name();

    // recursively traverse all submodules
    for (const auto &s : m.submodules()) {
        traverse(s);
    }

    // traverse all functions
    for (const auto &f : m.functions()) {
        get_function_arguments(f);

        main_function_bodies[f.name] = f.body;
    }

    // start traversing the main function
    auto it = main_function_bodies.find(module_name);
    if (it == main_function_bodies.end()) {
        internal_error << "\n\n\n"
                       << "StmtSizes::traverse - main function not found\n\n\n";
    }
    mutate(it->second);
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
                       << "Unsupported operator: " << op << "\n";
        return "";
    }
}

void StmtSizes::get_function_arguments(const LoweredFunc &op) {
    for (size_t i = 0; i < op.args.size(); i++) {
        arguments.push_back(op.args[i].name);
    }
}

string StmtSizes::print_sizes() const {
    stringstream ss;
    for (const auto &it : stmt_sizes) {
        ss << it.first << ":" << endl;
        for (const auto &it2 : it.second.produces) {
            ss << "    produces " << it2.first << ": " << it2.second << endl;
        }
        for (const auto &it2 : it.second.consumes) {
            ss << "    consumes " << it2.first << ": " << it2.second << endl;
        }
    }
    return ss.str();
}
string StmtSizes::print_produce_sizes(StmtSize &stmtSize) const {
    stringstream ss;
    for (const auto &it : stmtSize.produces) {
        ss << "produces:" << it.first << ": " << it.second << "<br>";
    }
    return ss.str();
}
string StmtSizes::print_consume_sizes(StmtSize &stmtSize) const {
    stringstream ss;
    for (const auto &it : stmtSize.consumes) {
        ss << "consumes:" << it.first << ": " << it.second << "<br>";
    }
    return ss.str();
}

void StmtSizes::set_produce_size(const IRNode *node, string produce_var, string produce_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].produces[produce_var] = produce_size;
}
void StmtSizes::set_consume_size(const IRNode *node, string consume_var, string consume_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].consumes[consume_var] = consume_size;
}
void StmtSizes::set_allocation_size_old(const IRNode *node, string allocate_var,
                                        string allocate_size) {
    auto it = stmt_sizes.find(node);

    // error if size found and allocate_var already exists
    if (it != stmt_sizes.end()) {
        auto it2 = it->second.allocates.find(allocate_var);
        if (it2 != it->second.allocates.end()) {
            internal_error << "\n"
                           << print_node(node) << "\n"
                           << "StmtSizes::set_allocation_size - " << allocate_var
                           << " already found in allocates\n"
                           << "\n\n";
        }
    }

    // if size not found, create new entry
    else {
        stmt_sizes[node] = StmtSize();
    }

    // set allocation size
    stmt_sizes[node].allocates[allocate_var] = allocate_size;
}
void StmtSizes::set_for_loop_size(const IRNode *node, string for_loop_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].forLoopSize = for_loop_size;
}
void StmtSizes::set_allocation_size(const IRNode *node, string allocate_size) {
    auto it = stmt_sizes.find(node);
    if (it == stmt_sizes.end()) {
        stmt_sizes[node] = StmtSize();
    }
    stmt_sizes[node].allocationSizes.push_back(allocate_size);
}

bool StmtSizes::in_producer(const string &name) const {
    // check if name is in curr_producer_names
    for (const auto &it : curr_producer_names) {
        if (it == name) {
            return true;
        }
    }
    return false;
}
bool StmtSizes::in_consumer(const string &name) const {
    // check if name is in curr_consumer_names
    for (const auto &it : curr_consumer_names) {
        if (it == name) {
            return true;
        }
    }
    return false;
}
void StmtSizes::remove_producer(const string &name) {
    // remove name from curr_producer_names
    for (size_t i = 0; i < curr_producer_names.size(); i++) {
        if (curr_producer_names[i] == name) {
            curr_producer_names.erase(curr_producer_names.begin() + i);
            return;
        }
    }
}
void StmtSizes::remove_consumer(const string &name) {
    // remove name from curr_consumer_names
    for (size_t i = 0; i < curr_consumer_names.size(); i++) {
        if (curr_consumer_names[i] == name) {
            curr_consumer_names.erase(curr_consumer_names.begin() + i);
            return;
        }
    }
}

string StmtSizes::string_span(string varName) const {
    return "<span class='stringType'>" + varName + "</span>";
}
string StmtSizes::int_span(int64_t intVal) const {
    return "<span class='intType'>" + to_string(intVal) + "</span>";
}

void StmtSizes::bubble_up(const IRNode *from, const IRNode *to, string loopIterator = "") {
    StmtSize fromSize = get_size(from);
    StmtSize toSize = get_size(to);

    // produce sizes
    for (const auto &produce_var : fromSize.produces) {
        string fromVar = produce_var.first;
        string fromProduceSize = produce_var.second;

        // if there's loop iterator, add it to the produce var name
        if (loopIterator != "") {
            string toProduceSize = get_simplified_string(loopIterator, fromProduceSize, "*");
            set_produce_size(to, fromVar, toProduceSize);
        } else {
            // check if fromVar is in toSize.consumes
            auto it = toSize.produces.find(fromVar);
            if (it != toSize.produces.end()) {
                string currProduceSize = it->second;
                string toProduceSize = get_simplified_string(currProduceSize, fromProduceSize, "+");

                set_produce_size(to, fromVar, toProduceSize);
            } else {
                set_produce_size(to, fromVar, fromProduceSize);
            }
        }
    }

    // consume sizes
    for (const auto &consume_var : fromSize.consumes) {
        string fromVar = consume_var.first;
        string fromConsumeSize = consume_var.second;

        // if there's loop iterator, add it to the consume var name
        if (loopIterator != "") {
            string toConsumeSize = get_simplified_string(loopIterator, fromConsumeSize, "*");
            set_consume_size(to, fromVar, toConsumeSize);
        } else {
            // check if fromVar is in toSize.consumes
            auto it = toSize.consumes.find(fromVar);
            if (it != toSize.consumes.end()) {
                string currConsumeSize = it->second;
                string toConsumeSize = get_simplified_string(currConsumeSize, fromConsumeSize, "+");

                set_consume_size(to, fromVar, toConsumeSize);
            } else {
                set_consume_size(to, fromVar, fromConsumeSize);
            }
        }
    }

    // mainFunctionCalls
    for (const auto &mainFunctionCall : fromSize.mainFunctionCalls) {
        // only add if not already in toSize.mainFunctionCalls
        if (!std::count(toSize.mainFunctionCalls.begin(), toSize.mainFunctionCalls.end(),
                        mainFunctionCall)) {
            toSize.mainFunctionCalls.push_back(mainFunctionCall);
        }
    }
}

Expr StmtSizes::visit(const Call *op) {
    for (size_t i = 0; i < op->args.size(); i++) {
        Expr arg = op->args[i];
        mutate(arg);
        bubble_up(arg.get(), op);
    }
    return op;
}
Expr StmtSizes::visit(const Variable *op) {
    // if op->name starts with "::", remove "::"
    string varName = op->name;
    if (varName[0] == ':' && varName[1] == ':') {
        varName = varName.substr(2);
    }

    auto it = main_function_bodies.find(varName);
    if (it != main_function_bodies.end()) {
        StmtSize size = StmtSize();
        size.mainFunctionCalls.push_back(op->name);
        stmt_sizes[op] = size;

        mutate(it->second);

        bubble_up(it->second.get(), op);
    }
    return op;
}
Stmt StmtSizes::visit(const LetStmt *op) {
    mutate(op->value);
    mutate(op->body);

    bubble_up(op->value.get(), op);
    bubble_up(op->body.get(), op);

    return op;
}
Stmt StmtSizes::visit(const ProducerConsumer *op) {
    if (op->is_producer) {
        curr_producer_names.push_back(op->name);
    } else {
        curr_consumer_names.push_back(op->name);
    }

    mutate(op->body);

    bubble_up(op->body.get(), op);

    // remove name from curr_producer_names or curr_consumer_names
    if (op->is_producer) {
        remove_producer(op->name);
    } else {
        remove_consumer(op->name);
    }

    return op;
}
string StmtSizes::get_loop_iterator(const For *op) const {
    Expr min = op->min;
    Expr extent = op->extent;

    string loopIterator;

    // check if min and extend are of type IntImm
    if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::IntImm) {
        int64_t minValue = min.as<IntImm>()->value;
        int64_t extentValue = extent.as<IntImm>()->value;
        uint16_t range = uint16_t(extentValue - minValue);
        loopIterator = int_span(range);
    }

    else if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::Variable) {
        int64_t minValue = min.as<IntImm>()->value;
        string minName = int_span(minValue);
        string extentName = string_span(extent.as<Variable>()->name);

        // TODO: inline variable for extentName

        if (minValue == 0) {
            loopIterator = extentName;
        } else {
            loopIterator = "(" + extentName + " - " + minName + ")";
        }
    }

    else if (min.node_type() == IRNodeType::IntImm && extent.node_type() == IRNodeType::Add) {
        int64_t minValue = min.as<IntImm>()->value;
        string minName = int_span(minValue);
        string extentName = "(";

        // deal with a
        if (extent.as<Add>()->a.node_type() == IRNodeType::IntImm) {
            int64_t extentValue = extent.as<Add>()->a.as<IntImm>()->value;
            extentName += int_span(extentValue);
        } else if (extent.as<Add>()->a.node_type() == IRNodeType::Variable) {
            extentName += string_span(extent.as<Add>()->a.as<Variable>()->name);
        } else {
            internal_error << "\n"
                           << "In for loop: " << op->name << "\n"
                           << print_node(extent.as<Add>()->a.get()) << "\n"
                           << "StmtSizes::visit(const For *op): add->a isn't IntImm or Variable - "
                              "can't generate ProdCons hierarchy yet. \n\n";
        }

        extentName += "+";

        // deal with b
        if (extent.as<Add>()->b.node_type() == IRNodeType::IntImm) {
            int64_t extentValue = extent.as<Add>()->b.as<IntImm>()->value;
            extentName += int_span(extentValue);
        } else if (extent.as<Add>()->b.node_type() == IRNodeType::Variable) {
            extentName += string_span(extent.as<Add>()->b.as<Variable>()->name);
        } else {
            internal_error << "\n"
                           << "In for loop: " << op->name << "\n"
                           << print_node(extent.as<Add>()->b.get()) << "\n"
                           << "StmtSizes::visit(const For *op): add->b isn't IntImm or Variable - "
                              "can't generate ProdCons hierarchy yet. \n\n";
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
            << print_node(op->min.get()) << "\n"
            << print_node(op->extent.get()) << "\n"
            << "StmtSizes::visit(const For *op): min and extent are not of type (IntImm) "
               "or (IntImm & Variable) or (IntImm & Add) - "
               "can't generate ProdCons hierarchy yet. \n\n";
    }

    return loopIterator;
}
Stmt StmtSizes::visit(const For *op) {
    mutate(op->body);
    StmtSize bodySize = get_size(op->body.get());

    // don't do anything if body is empty
    if (bodySize.empty()) {
        return op;
    }

    string loopIterator = get_loop_iterator(op);

    bubble_up(op->body.get(), op, loopIterator);

    set_for_loop_size(op, loopIterator);

    return op;
}
Stmt StmtSizes::visit(const Store *op) {

    // TODO: is this correct? should i be getting it from `index`?
    uint16_t lanes = op->index.type().lanes();

    if (in_producer(op->name)) {
        set_produce_size(op, op->name, int_span(lanes));
    }

    // empty curr_load_values
    curr_load_values.clear();
    curr_loads.clear();
    mutate(op->value);

    // set consume (for now, read values)
    for (const auto &load_var : curr_load_values) {
        set_consume_size(op, load_var.first, int_span(load_var.second));
    }

    // iterate through curr_loads to get unique loads
    for (const auto &load_var : curr_loads) {
        string vectorName = load_var.first;
        vector<set<int>> loadValues = load_var.second;
        set<int> finalLoadValuesUnique;

        for (const set<int> &loadValueSet : loadValues) {
            finalLoadValuesUnique.insert(loadValueSet.begin(), loadValueSet.end());
        }

        if (SHOW_UNIQUE_LOADS) {
            set_consume_size(op, vectorName + "_unique", int_span(finalLoadValuesUnique.size()));
        }
    }

    return op;
}
void StmtSizes::add_load_value(const string &name, const int lanes) {
    auto it = curr_load_values.find(name);
    if (it == curr_load_values.end()) {
        curr_load_values[name] = lanes;
    } else {
        curr_load_values[name] += lanes;
    }
}
void StmtSizes::add_load_value_unique_loads(const string &name, set<int> &load_values) {
    curr_loads[name].push_back(load_values);
}
Expr StmtSizes::visit(const Load *op) {

    // see if the variable is in the arguments variable
    if (std::count(arguments.begin(), arguments.end(), op->name)) {
        curr_consumer_names.push_back(op->name);
    }

    if (in_consumer(op->name)) {
        // TODO: make sure this logic is right
        int lanes;

        if (op->index.as<Ramp>()) {
            lanes = op->index.as<Ramp>()->lanes;

            // calculate unique loads if all args are concrete
            if (op->index.as<Ramp>()->base.as<IntImm>() &&
                op->index.as<Ramp>()->stride.as<IntImm>()) {
                int64_t baseValue = op->index.as<Ramp>()->base.as<IntImm>()->value;
                int64_t strideValue = op->index.as<Ramp>()->stride.as<IntImm>()->value;

                set<int> load_values;
                for (int i = baseValue; i < baseValue + (lanes * strideValue); i += strideValue) {
                    load_values.insert(i);
                }

                if (SHOW_UNIQUE_LOADS) {
                    add_load_value_unique_loads(op->name, load_values);
                }
            }
        } else {
            lanes = int(op->type.lanes());
        }

        add_load_value(op->name, lanes);
    }

    return op;
}
Stmt StmtSizes::visit(const Block *op) {
    mutate(op->first);
    mutate(op->rest);

    bubble_up(op->first.get(), op);
    bubble_up(op->rest.get(), op);

    return op;
}
Stmt StmtSizes::visit(const Allocate *op) {

    mutate(op->body);
    StmtSize bodySize = get_size(op->body.get());

    bubble_up(op->body.get(), op);

    // set allocate stuff
    stringstream type;
    type << "<span class='stringType'>" << op->type << "</span>";
    set_allocation_size(op, type.str());

    for (const auto &extent : op->extents) {
        // TODO: inline these as well if they are variables
        stringstream ss;
        if (extent.as<IntImm>()) {
            ss << "<span class='intType'>" << extent << "</span>";
        } else {
            ss << "<span class='stringType'>" << extent << "</span>";
        }

        set_allocation_size(op, ss.str());
    }

    return op;
}
Stmt StmtSizes::visit(const IfThenElse *op) {
    mutate(op->then_case);
    mutate(op->else_case);

    bubble_up(op->then_case.get(), op);
    if (op->else_case.defined()) {
        bubble_up(op->else_case.get(), op);
    }

    return op;
}

/*
 * ProducerConsumerHierarchy class
 */
string ProducerConsumerHierarchy::generate_producer_consumer_html(const Module &m) {
    pre_processor.generate_sizes(m);

    html.str(string());
    startModuleTraversal();

    string html_string = html.str();
    return html_string;
}
string ProducerConsumerHierarchy::generate_producer_consumer_html(const Stmt &stmt) {
    pre_processor.generate_sizes(stmt);

    html.str(string());
    mutate(stmt);

    return html.str();
}

void ProducerConsumerHierarchy::startModuleTraversal() {

    auto it = pre_processor.main_function_bodies.find(pre_processor.module_name);
    if (it == pre_processor.main_function_bodies.end()) {
        internal_error << "ProducerConsumerHierarchy::generate_producer_consumer_html - main "
                          "function not found\n";
    }

    // start with the main function
    html << "<a name=\"" << it->first << "\"></a>";
    html << "<h3>" << it->first << "</h3>";
    html << "<div class='functionViz'>";
    mutate(it->second);
    html << "</div>";

    // mutate all other functions
    for (const auto &func : pre_processor.main_function_bodies) {
        if (func.first != pre_processor.module_name) {
            html << "<a name=\"" << func.first << "\"></a>";
            html << "<h3>" << func.first << "</h3>";
            html << "<div class='functionViz'>";
            mutate(func.second);
            html << "</div>";
        }
    }
}

void ProducerConsumerHierarchy::open_box_div(string backgroundColor, string className,
                                             const IRNode *op) {
    html << "<div style='";
    html << "background-color: " << backgroundColor << "; ";
    html << "' ";
    html << "class='box center " << className << "'";
    html << ">";

    generate_computation_cost_div(op);
    generate_memory_cost_div(op);

    open_content_div();
}
void ProducerConsumerHierarchy::close_box_div() {
    close_div();  // content div
    close_div();  // main box div
}
void ProducerConsumerHierarchy::open_header_div() {
    html << "<div class='boxHeader'>";
}
void ProducerConsumerHierarchy::open_box_header_title_div() {
    html << "<div class='boxHeaderTitle'>";
}
void ProducerConsumerHierarchy::open_box_header_table_div() {
    html << "<div class='boxHeaderTable'>";
}
void ProducerConsumerHierarchy::open_store_div() {
    html << "<div class='store'>";
}
void ProducerConsumerHierarchy::close_div() {
    html << "</div>";
}

void ProducerConsumerHierarchy::open_header(const IRNode *op, const string &header,
                                            string anchorName) {
    open_header_div();

    open_box_header_title_div();

    html << header;
    if (anchorName != "") {
        see_code_button(anchorName);
    }
    close_div();

    // spacing purposes
    html << "<div class='spacing'></div>";

    open_box_header_table_div();
}
void ProducerConsumerHierarchy::close_header() {
    close_div();  // header table div
    close_div();  // header div
}
void ProducerConsumerHierarchy::div_header(const IRNode *op, const string &header, StmtSize &size,
                                           string anchorName) {

    open_header(op, header, anchorName);

    // add producer consumer size if size is provided
    if (!size.empty()) {
        prod_cons_table(size);
    }

    close_header();
}
void ProducerConsumerHierarchy::allocate_div_header(const Allocate *op, const string &header,
                                                    StmtSize &size, string anchorName) {
    open_header(op, header, anchorName);

    vector<string> &allocationSizes = size.allocationSizes;
    allocate_table(allocationSizes);

    close_header();
}
void ProducerConsumerHierarchy::for_loop_div_header(const For *op, const string &header,
                                                    StmtSize &size, string anchorName) {
    open_header(op, header, anchorName);

    string loopSize = pre_processor.get_size(op).forLoopSize;
    for_loop_table(loopSize);

    close_header();
}

void ProducerConsumerHierarchy::if_tree(const IRNode *op, const string &header, StmtSize &size,
                                        string anchorName = "") {
    html << "<li>";
    html << "<span class='tf-nc if-node'>";

    open_box_div(IF_COLOR, "IfBox", op);
    div_header(op, header, size, anchorName);
}
void ProducerConsumerHierarchy::close_if_tree() {
    close_box_div();
    html << "</span>";
    html << "</li>";
}

void ProducerConsumerHierarchy::prod_cons_table(StmtSize &size) {
    // open table
    html << "<table class='costTable' style='background-color: rgba(150, 150, 150, 0.5)'>";

    // Prod | Cons
    html << "<tr>";

    html << "<th colspan='2' class='costTableHeader middleCol'>";
    html << "Written";
    html << "</th>";

    html << "<th colspan='2' class='costTableHeader'>";
    html << "Read";
    html << "</th>";

    html << "</tr>";

    // produces and consumes are empty - add row with values 0
    if (size.empty()) {
        internal_error << "\n\n"
                       << "ProducerConsumerHierarchy::prod_cons_table - size is empty"
                       << "\n";
    }

    // produces and consumes aren't empty
    else {
        vector<string> rows;

        // fill in producer variables
        for (const auto &produce_var : size.produces) {
            stringstream ss;
            ss << "<td class='costTableData'>";
            ss << produce_var.first << ": ";
            ss << "</td>";

            ss << "<td class='costTableData middleCol'>";
            ss << produce_var.second;
            ss << "</td>";

            rows.push_back(ss.str());
        }

        // fill in consumer variables
        unsigned long rowNum = 0;
        for (const auto &consume_var : size.consumes) {
            stringstream ss;
            ss << "<td class='costTableData'>";
            ss << consume_var.first << ": ";
            ss << "</td>";

            ss << "<td class='costTableData'>";
            ss << consume_var.second;
            ss << "</td>";

            if (rowNum < rows.size()) {
                rows[rowNum] += ss.str();
            } else {
                // pad row with empty cells for produce
                stringstream sEmpty;
                sEmpty << "<td colspan='2' class='costTableData middleCol'>";
                sEmpty << "</td>";

                rows.push_back(sEmpty.str() + ss.str());
            }
            rowNum++;
        }

        // pad row with empty calls for consume
        rowNum = size.consumes.size();
        while (rowNum < size.produces.size()) {
            stringstream sEmpty;
            sEmpty << "<td class='costTableData'>";
            sEmpty << "</td>";
            sEmpty << "<td class='costTableData'>";
            sEmpty << "</td>";

            rows[rowNum] += sEmpty.str();
            rowNum++;
        }

        // add rows to html
        for (const auto &row : rows) {
            html << "<tr>";
            html << row;
            html << "</tr>";
        }
    }

    // close table
    html << "</table>";
}
void ProducerConsumerHierarchy::allocate_table(vector<string> &allocationSizes) {
    // open table
    html << "<table class='costTable' style='background-color: rgba(150, 150, 150, 0.5)'>";

    stringstream header;
    stringstream data;

    // open header and data rows
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
            header << "Dim-" << i;
            header << "</th>";

            data << allocationSizes[i];
            data << "</td>";
        }
    }

    // close header and data rows
    header << "</tr>";
    data << "</tr>";

    // add header and data rows to html
    html << header.str();
    html << data.str();

    // close table
    html << "</table>";
}
void ProducerConsumerHierarchy::for_loop_table(string loop_size) {
    // open table
    html << "<table class='costTable' style='background-color: rgba(150, 150, 150, 0.5)'>";

    // Loop Size
    html << "<tr>";

    html << "<th class='costTableHeader'>";
    html << "Loop Size";
    html << "</th>";

    html << "</tr>";

    html << "<tr>";

    // loop size
    html << "<td class='costTableData'>";
    html << loop_size;
    html << "</td>";

    html << "</tr>";

    // close table
    html << "</table>";
}

void ProducerConsumerHierarchy::see_code_button(string anchorName) {
    html << "<button class='see-code-button' onclick='";
    html << "window.open(\"" << output_file_name << "#" << anchorName << "\", \"_blank\")";
    html << "' style='margin-left: 5px'>";
    html << "<i class='bi bi-code-square'></i>";
    html << "</button>";
}

string ProducerConsumerHierarchy::info_tooltip(string toolTipText, string className = "") {
    prodConsTooltipCount++;

    stringstream ss;

    // info-button
    ss << "<button id='prodConsButton" << prodConsTooltipCount << "' ";
    ss << "aria-describedby='prodConsTooltip" << prodConsTooltipCount << "' ";
    ss << "class='info-button' role='button' ";
    ss << ">";
    ss << "<i class='bi bi-info'></i>";
    ss << "</button>";

    ss << "<span id='prodConsTooltip" << prodConsTooltipCount << "' class='tooltip prodConsTooltip";
    if (className != "") {
        ss << " " << className;
    }
    ss << "'";
    ss << "role='prodConsTooltip" << prodConsTooltipCount << "'>";
    ss << toolTipText;
    ss << "</span>";

    return ss.str();
}

void ProducerConsumerHierarchy::generate_computation_cost_div(const IRNode *op) {
    // skip if it's a store
    if (op->node_type == IRNodeType::Store) return;

    int computation_range = findStmtCost.get_computation_range(op);
    string className = "computation-cost-div CostColor" + to_string(computation_range);
    html << "<div class='" << className << "'";
    html << "style='width: 10px;'>";

    close_div();
}
void ProducerConsumerHierarchy::generate_memory_cost_div(const IRNode *op) {
    // skip if it's a store
    if (op->node_type == IRNodeType::Store) return;

    // html << "<div class='memory-cost-div'>";
    int memory_range = findStmtCost.get_data_movement_range(op);
    string className = "memory-cost-div CostColor" + to_string(memory_range);
    html << "<div class='" << className << "'";
    html << "style='width: 10px;'>";

    close_div();
}
void ProducerConsumerHierarchy::open_content_div() {
    html << "<div class='content'>";
}

void ProducerConsumerHierarchy::open_span(string className) {
    html << "<span class='" << className << "'>";
}
void ProducerConsumerHierarchy::close_span() {
    html << "</span>";
}
void ProducerConsumerHierarchy::cost_color_spacer() {
    open_span("CostColorSpacer");
    html << ".";
    close_span();
}
void ProducerConsumerHierarchy::cost_colors(const IRNode *op) {
    cost_color_spacer();

    int computation_range = findStmtCost.get_computation_range(op);
    open_span("CostColor" + to_string(computation_range) + " CostComputation");
    html << ".";
    close_span();

    cost_color_spacer();

    int data_movement_range = findStmtCost.get_data_movement_range(op);
    open_span("CostColor" + to_string(data_movement_range) + " CostMovement");
    html << ".";
    close_span();

    cost_color_spacer();
}

Expr ProducerConsumerHierarchy::visit(const Variable *op) {
    // if op->name starts with "::", remove "::"
    string varName = op->name;
    if (varName[0] == ':' && varName[1] == ':') {
        varName = varName.substr(2);
    }

    auto it = pre_processor.main_function_bodies.find(varName);
    if (it != pre_processor.main_function_bodies.end()) {

        StmtSize size = pre_processor.get_size(op);

        if (!SHOW_CUMULATIVE_COST) {
            size = StmtSize();
        }

        html << "<div class='box center FunctionCallBox' ";
        html << "style='background-color: " << FUNCTION_CALL_COLOR << ";";
        html << "padding: 5px;'>";

        html << "Function Call";
        html << "<button class='info-button' role='button' ";
        html << "onclick=\"location.href='./" << output_file_name << "#" << varName << "'\">";
        html << varName;
        html << "</button>";

        html << "</div>";

        return op;
    }
    return op;
}
Stmt ProducerConsumerHierarchy::visit(const ProducerConsumer *op) {
    open_box_div(op->is_producer ? PRODUCER_COLOR : CONSUMER_COLOR, "ProducerConsumerBox", op);

    producerConsumerCount++;
    string anchorName = "producerConsumer" + std::to_string(producerConsumerCount);

    stringstream header;
    header << (op->is_producer ? "Produce" : "Consume");
    header << " " << op->name;
    StmtSize size = pre_processor.get_size(op);

    if (!SHOW_CUMULATIVE_COST) {
        size = StmtSize();
    }

    div_header(op, header.str(), size, anchorName);

    mutate(op->body);

    close_box_div();

    return op;
}

Stmt ProducerConsumerHierarchy::visit(const For *op) {
    open_box_div(FOR_COLOR, "ForBox", op);

    forCount++;
    string anchorName = "for" + std::to_string(forCount);

    StmtSize size = pre_processor.get_size(op);

    stringstream header;
    header << "For (" << op->name << ")";

    if (SHOW_CUMULATIVE_COST) {
        div_header(op, header.str(), size, anchorName);
    } else {
        for_loop_div_header(op, header.str(), size, anchorName);
    }

    mutate(op->body);

    close_box_div();

    return op;
}

Stmt ProducerConsumerHierarchy::visit(const IfThenElse *op) {
    StmtSize thenSize = pre_processor.get_size(op->then_case.get());
    StmtSize elseSize;

    if (op->else_case.defined()) {
        elseSize = pre_processor.get_size(op->else_case.get());
    }

    // only start the if-tree if either case is not empty
    // (aka won't print if both cases are empty)
    // (we can't just exit early though because we have to go through all if-stmts to
    //  get accurate count for the anchor names)
    bool opened = false;
    if (!thenSize.empty() || !elseSize.empty()) {
        // open main if tree
        html << "<div class='tf-tree tf-gap-sm tf-custom-prodCons' style='font-size: 12px;'>";
        html << "<ul>";
        html << "<li><span class='tf-nc if-node'>";
        html << "If";
        html << "</span>";
        html << "<ul>";
        opened = true;
    }

    stringstream ifHeader;
    ifHeader << "if ";

    // anchor name
    ifCount++;
    string anchorName = "if" + std::to_string(ifCount);

    while (true) {

        thenSize = pre_processor.get_size(op->then_case.get());

        if (!thenSize.empty()) {
            // TODO: inline condition
            stringstream condition;
            condition << op->condition;

            // make condition smaller if it's too big
            if (condition.str().size() > MAX_CONDITION_LENGTH) {
                condition.str("");
                condition << "... ";

                stringstream cond;
                cond << op->condition;
                condition << info_tooltip(cond.str(), "conditionTooltip");
            }

            ifHeader << condition.str();

            if (!SHOW_CUMULATIVE_COST) {
                thenSize = StmtSize();
            }
            if_tree(op->then_case.get(), ifHeader.str(), thenSize, anchorName);

            // then body
            mutate(op->then_case);

            close_if_tree();
        }

        // if there is no else case, we are done
        if (!op->else_case.defined()) {
            break;
        }

        // if else case is another ifthenelse, recurse and reset op to else case
        if (const IfThenElse *nested_if = op->else_case.as<IfThenElse>()) {
            op = nested_if;
            ifHeader.str("");
            ifHeader << "else if ";

            // anchor name
            ifCount++;
            anchorName = "if" + std::to_string(ifCount);

        }

        // if else case is not another ifthenelse
        else {
            elseSize = pre_processor.get_size(op->else_case.get());

            if (!elseSize.empty()) {
                stringstream elseHeader;
                elseHeader << "else ";

                // anchor name
                ifCount++;
                anchorName = "if" + std::to_string(ifCount);

                if (!SHOW_CUMULATIVE_COST) {
                    elseSize = StmtSize();
                }
                if_tree(op->else_case.get(), elseHeader.str(), elseSize, anchorName);

                mutate(op->else_case);

                close_if_tree();
            }
            break;
        }
    }

    // close main if tree
    if (opened) {
        html << "</ul>";
        html << "</li>";
        html << "</ul>";
        html << "</div>";
    }
    return op;
}

Stmt ProducerConsumerHierarchy::visit(const Store *op) {
    StmtSize size = pre_processor.get_size(op);

    storeCount++;
    string anchorName = "store" + std::to_string(storeCount);

    stringstream header;
    header << "Store " << op->name;

    open_box_div(STORE_COLOR, "StoreBox", op);

    div_header(op, header.str(), size, anchorName);

    mutate(op->value);

    close_box_div();

    return op;
}
Expr ProducerConsumerHierarchy::visit(const Load *op) {
    int lanes;

    // TODO: make sure this is right
    if (op->index.as<Ramp>()) {
        lanes = op->index.as<Ramp>()->lanes;
    } else {
        lanes = int(op->type.lanes());
    }

    stringstream header;
    header << "Load " << op->name << " ";

    // tooltip table
    stringstream tooltipTable;
    tooltipTable << "<table class='tooltipTable'>";

    tooltipTable << "<tr>";
    tooltipTable << "<td class = 'left-table'> Variable Type</td>";
    tooltipTable << "<td class = 'right-table'> ";
    if (findStmtCost.is_local_variable(op->name)) {
        tooltipTable << "local var";
    } else {
        tooltipTable << "global var";
    }
    tooltipTable << "</td>";
    tooltipTable << "</tr>";

    tooltipTable << "<tr>";
    tooltipTable << "<td class = 'left-table'> Load Size</td>";
    tooltipTable << "<td class = 'right-table'> " << lanes << "</td>";
    tooltipTable << "</tr>";

    tooltipTable << "</table>";

    header << info_tooltip(tooltipTable.str());

    open_store_div();
    cost_colors(op);
    html << header.str();
    close_div();

    return op;
}

string get_memory_type(MemoryType memType) {
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
Stmt ProducerConsumerHierarchy::visit(const Allocate *op) {
    open_box_div(ALLOCATE_COLOR, "AllocateBox", op);

    allocateCount++;
    string anchorName = "allocate" + std::to_string(allocateCount);

    StmtSize size = pre_processor.get_size(op);

    stringstream header;
    header << "Allocate " << op->name << " ";

    // memory type tooltip table
    stringstream tooltipTable;
    tooltipTable << "<table class='tooltipTable'>";
    tooltipTable << "<tr>";
    tooltipTable << "<td class = 'left-table'> Memory Type</td>";
    tooltipTable << "<td class = 'right-table'> " << get_memory_type(op->memory_type) << "</td>";
    tooltipTable << "</tr>";

    if (!is_const_one(op->condition)) {
        tooltipTable << "<tr>";
        tooltipTable << "<td class = 'left-table'> Condition</td>";
        tooltipTable << "<td class = 'right-table'> " << op->condition << "</td>";
        tooltipTable << "</tr>";
    }
    if (op->new_expr.defined()) {
        internal_error << "\n"
                       << "ProducerConsumerHierarchy: Allocate " << op->name
                       << " `op->new_expr.defined()` is not supported.\n\n";

        tooltipTable << "<tr>";
        tooltipTable << "<td class = 'left-table'> New Expr</td>";
        tooltipTable << "<td class = 'right-table'> " << op->new_expr << "</td>";
        tooltipTable << "</tr>";
    }
    if (!op->free_function.empty()) {
        internal_error << "\n"
                       << "ProducerConsumerHierarchy: Allocate " << op->name
                       << " `!op->free_function.empty()` is not supported.\n\n";

        tooltipTable << "<tr>";
        tooltipTable << "<td class = 'left-table'> Free Function</td>";
        tooltipTable << "<td class = 'right-table'> " << op->free_function << "</td>";
        tooltipTable << "</tr>";
    }

    tooltipTable << "</table>";

    header << info_tooltip(tooltipTable.str());

    allocate_div_header(op, header.str(), size, anchorName);

    close_box_div();

    mutate(op->body);

    return op;
}

string ProducerConsumerHierarchy::generate_prodCons_js() {
    stringstream prodConsJS;

    prodConsJS << "// prodCons JS\n";
    prodConsJS << "for (let i = 1; i <= " << prodConsTooltipCount << "; i++) { \n";
    prodConsJS << "    const button = document.querySelector('#prodConsButton' + i); \n";
    prodConsJS << "    const tooltip = document.querySelector('#prodConsTooltip' + i); \n";
    prodConsJS << "    button.addEventListener('mouseenter', () => { \n";
    prodConsJS << "        showTooltip(button, tooltip); \n";
    prodConsJS << "    }); \n";
    prodConsJS << "    button.addEventListener('mouseleave', () => { \n";
    prodConsJS << "        hideTooltip(tooltip); \n";
    prodConsJS << "    } \n";
    prodConsJS << "    ); \n";
    prodConsJS << "    tooltip.addEventListener('focus', () => { \n";
    prodConsJS << "        showTooltip(button, tooltip); \n";
    prodConsJS << "    } \n";
    prodConsJS << "    ); \n";
    prodConsJS << "    tooltip.addEventListener('blur', () => { \n";
    prodConsJS << "        hideTooltip(tooltip); \n";
    prodConsJS << "    } \n";
    prodConsJS << "    ); \n";
    prodConsJS << "} \n";

    return prodConsJS.str();
}

/*
 * PRINT NODE
 */
string StmtSizes::print_node(const IRNode *node) const {
    stringstream s;
    s << "Crashing node has type: ";
    IRNodeType type = node->node_type;
    if (type == IRNodeType::IntImm) {
        s << "IntImm type";
        auto node1 = dynamic_cast<const IntImm *>(node);
        s << "value: " << node1->value;
    } else if (type == IRNodeType::UIntImm) {
        s << "UIntImm type";
    } else if (type == IRNodeType::FloatImm) {
        s << "FloatImm type";
    } else if (type == IRNodeType::StringImm) {
        s << "StringImm type";
    } else if (type == IRNodeType::Broadcast) {
        s << "Broadcast type";
    } else if (type == IRNodeType::Cast) {
        s << "Cast type";
    } else if (type == IRNodeType::Variable) {
        s << "Variable type";
    } else if (type == IRNodeType::Add) {
        s << "Add type";
        auto node1 = dynamic_cast<const Add *>(node);
        s << "a: " << print_node(node1->a.get()) << endl;
        s << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Sub) {
        s << "Sub type" << endl;
        auto node1 = dynamic_cast<const Sub *>(node);
        s << "a: " << print_node(node1->a.get()) << endl;
        s << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Mod) {
        s << "Mod type" << endl;
        auto node1 = dynamic_cast<const Mod *>(node);
        s << "a: " << print_node(node1->a.get()) << endl;
        s << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Mul) {
        s << "Mul type" << endl;
        auto node1 = dynamic_cast<const Mul *>(node);
        s << "a: " << print_node(node1->a.get()) << endl;
        s << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Div) {
        s << "Div type" << endl;
        auto node1 = dynamic_cast<const Div *>(node);
        s << "a: " << print_node(node1->a.get()) << endl;
        s << "b: " << print_node(node1->b.get()) << endl;
    } else if (type == IRNodeType::Min) {
        s << "Min type";
    } else if (type == IRNodeType::Max) {
        s << "Max type";
    } else if (type == IRNodeType::EQ) {
        s << "EQ type";
    } else if (type == IRNodeType::NE) {
        s << "NE type";
    } else if (type == IRNodeType::LT) {
        s << "LT type";
    } else if (type == IRNodeType::LE) {
        s << "LE type";
    } else if (type == IRNodeType::GT) {
        s << "GT type";
    } else if (type == IRNodeType::GE) {
        s << "GE type";
    } else if (type == IRNodeType::And) {
        s << "And type";
    } else if (type == IRNodeType::Or) {
        s << "Or type";
    } else if (type == IRNodeType::Not) {
        s << "Not type";
    } else if (type == IRNodeType::Select) {
        s << "Select type";
    } else if (type == IRNodeType::Load) {
        s << "Load type";
    } else if (type == IRNodeType::Ramp) {
        s << "Ramp type";
    } else if (type == IRNodeType::Call) {
        s << "Call type";
    } else if (type == IRNodeType::Let) {
        s << "Let type";
    } else if (type == IRNodeType::Shuffle) {
        s << "Shuffle type";
    } else if (type == IRNodeType::VectorReduce) {
        s << "VectorReduce type";
    } else if (type == IRNodeType::LetStmt) {
        s << "LetStmt type";
    } else if (type == IRNodeType::AssertStmt) {
        s << "AssertStmt type";
    } else if (type == IRNodeType::ProducerConsumer) {
        s << "ProducerConsumer type";
    } else if (type == IRNodeType::For) {
        s << "For type";
    } else if (type == IRNodeType::Acquire) {
        s << "Acquire type";
    } else if (type == IRNodeType::Store) {
        s << "Store type";
    } else if (type == IRNodeType::Provide) {
        s << "Provide type";
    } else if (type == IRNodeType::Allocate) {
        s << "Allocate type";
    } else if (type == IRNodeType::Free) {
        s << "Free type";
    } else if (type == IRNodeType::Realize) {
        s << "Realize type";
    } else if (type == IRNodeType::Block) {
        s << "Block type";
    } else if (type == IRNodeType::Fork) {
        s << "Fork type";
    } else if (type == IRNodeType::IfThenElse) {
        s << "IfThenElse type";
    } else if (type == IRNodeType::Evaluate) {
        s << "Evaluate type";
    } else if (type == IRNodeType::Prefetch) {
        s << "Prefetch type";
    } else if (type == IRNodeType::Atomic) {
        s << "Atomic type";
    } else {
        s << "Unknown type";
    }

    return s.str();
}

const string ProducerConsumerHierarchy::prodConsCSS = "\n \
/* ProdCons CSS */\n \
.tf-custom-prodCons .tf-nc { border-radius: 5px; border: 1px solid; }\n \
.tf-custom-prodCons .tf-nc:before, .tf-custom-prodCons .tf-nc:after { border-left-width: 1px; }\n \
.tf-custom-prodCons li li:before { border-top-width: 1px; }\n \
.tf-custom-prodCons .end-node { border-style: dashed; }\n \
.tf-custom-prodCons .tf-nc { background-color: #e6eeff; }\n \
div.box { \n \
    border: 1px dashed grey; \n \
    border-radius: 5px; \n \
    margin: 10px; \n \
    display: flex; \n \
} \n \
div.boxHeader { \n \
    padding: 5px; \n \
    display: flex; \n \
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
";
