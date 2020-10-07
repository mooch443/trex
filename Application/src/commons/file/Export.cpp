#include "Export.h"

namespace file {
Table::Table(const std::vector<std::string> &header)
    : _header(header)
{ }

void Table::add(const Row &row) {
    _rows.push_back(row);
}
}
