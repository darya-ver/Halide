#ifndef ProducerConsumerHierarchy_H
#define ProducerConsumerHierarchy_H

#include "IRMutator.h"

#include <unordered_map>

using namespace std;
using namespace Halide;
using namespace Internal;

struct StmtSize {
    map<string, string> produces;
    map<string, string> consumes;

    bool empty() const {
        return produces.size() == 0 && consumes.size() == 0;
    }
};

/*
 * StmtSizes class
 */
class StmtSizes : public IRMutator {
public:
    StmtSizes() = default;
    ~StmtSizes() = default;

    void generate_sizes(const Module &m);
    void generate_sizes(const Stmt &stmt);

    StmtSize get_size(const IRNode *node) const;
    bool are_bounds_set();

    string print_sizes() const;
    string print_produce_sizes(StmtSize &stmtSize) const;
    string print_consume_sizes(StmtSize &stmtSize) const;

private:
    using IRMutator::visit;

    unordered_map<const IRNode *, StmtSize> stmt_sizes;
    bool bounds_set = false;
    bool in_producer = false;
    bool in_consumer = false;
    string curr_consumer;

    void traverse(const Module &m);

    string get_simplified_string(string a, string b, string op);

    void set_produce_size(const IRNode *node, string produce_var, string produce_size);
    void set_consume_size(const IRNode *node, string consume_var, string consume_size);

    Stmt visit(const LetStmt *op) override;
    Stmt visit(const ProducerConsumer *op) override;
    Stmt visit(const For *op) override;
    Stmt visit(const Store *op) override;
    Expr visit(const Load *op) override;
    Stmt visit(const Allocate *op) override;
    Stmt visit(const Block *op) override;
    Stmt visit(const IfThenElse *op) override;
};

/*
 * ProducerConsumerHierarchy class
 */
class ProducerConsumerHierarchy : public IRMutator {

public:
    ProducerConsumerHierarchy() = default;
    ~ProducerConsumerHierarchy() = default;

    // generates the html for the producer-consumer hierarchy
    string generate_producer_consumer_html(const Module &m);
    string generate_producer_consumer_html(const Stmt &stmt);

private:
    using IRMutator::visit;

    std::stringstream html;   // main html string
    StmtSizes pre_processor;  // generates the sizes of the nodes

    // starts the traversal of the tree and returns the generated html
    string get_producer_consumer_html(const Expr &startNode);
    string get_producer_consumer_html(const Stmt &startNode);

    // for traversal of a Module object
    void traverse(const Module &m);

    // starts and ends the html file
    void start_html();
    void end_html();

    // opens and closes a table
    void open_table();
    void close_table();

    // creates a table header row with given header string
    void table_header(const string &header, StmtSize &size);
    void prod_cons_table(StmtSize &size);
    // void double_table_header(const string &header);

    // opens and closes a row
    void open_table_row();
    void close_table_row();

    // opens and closes a data cell
    void open_table_data();
    void close_table_data();

    Stmt visit(const ProducerConsumer *op) override;
    Stmt visit(const For *op) override;
    Stmt visit(const IfThenElse *op) override;
};

#endif
