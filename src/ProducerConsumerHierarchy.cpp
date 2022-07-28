#include "ProducerConsumerHierarchy.h"
#include "Module.h"

using namespace std;
using namespace Halide;
using namespace Internal;

void ProducerConsumerHierarchy::generate_producer_consumer_html(const Module &m) {
    start_html();
    traverse(m);
    end_html();
}
void ProducerConsumerHierarchy::generate_producer_consumer_html(const Stmt &stmt) {
    start_html();
    mutate(stmt);
    end_html();
}

void ProducerConsumerHierarchy::print_hierarchy() {
    cout << endl;
    cout << "Hierarchy HTML: ";
    cout << endl;
    cout << html.str();
    cout << endl;
}

string ProducerConsumerHierarchy::get_producer_consumer_html(const Expr &startNode) {
    start_html();
    mutate(startNode);
    end_html();

    return html.str();
}
string ProducerConsumerHierarchy::get_producer_consumer_html(const Stmt &startNode) {
    start_html();
    mutate(startNode);
    end_html();

    return html.str();
}

void ProducerConsumerHierarchy::traverse(const Module &m) {
    // recursively traverse all submodules
    for (const auto &s : m.submodules()) {
        traverse(s);
    }
    // traverse all functions
    for (const auto &f : m.functions()) {
        mutate(f.body);
    }
}

void ProducerConsumerHierarchy::start_html() {
    html.str(string());
    html << "<html>";
    html << "<head>";
    html << "<link rel=\\'stylesheet\\' ";
    html << "href=\\'https://unpkg.com/treeflex/dist/css/treeflex.css\\'>";
    html << "</head>";
    html << "<style>";
    html << "body { font-family: Consolas, \\'Liberation Mono\\', Menlo, Courier, monospace;}";
    html << "table, th, td { ";
    html << "border: 1px solid black;";
    html << "border-collapse: collapse;";
    html << "padding: 15px;";
    html << "border-spacing: 15px;";
    html << "background-color: rgba(150, 150, 150, 0.15);";
    html << "} ";
    html << "</style>";
    html << "<body>";
}
void ProducerConsumerHierarchy::end_html() {
    html << "</body></html>";
}

void ProducerConsumerHierarchy::open_table() {
    html << "<table>";
}
void ProducerConsumerHierarchy::close_table() {
    html << "</table>";
}

void ProducerConsumerHierarchy::table_header(const string &header) {
    html << "<tr><th>" << header << "</th></tr>";
}

void ProducerConsumerHierarchy::open_table_row() {
    html << "<tr>";
    html << "<td>";
}
void ProducerConsumerHierarchy::close_table_row() {
    html << "</td>";
    html << "</tr>";
}

Stmt ProducerConsumerHierarchy::visit(const ProducerConsumer *op) {
    open_table();
    stringstream header;
    header << (op->is_producer ? "Produce" : "Consumer");
    header << " " << op->name;
    table_header(header.str());
    open_table_row();
    mutate(op->body);
    close_table_row();
    close_table();

    return op;
}
