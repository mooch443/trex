#include "CSVExport.h"
#include <fstream>

namespace file {
bool CSVExport::save(const file::Path &filename) const {
    auto &t = table();
    auto &header = t.header();
    
    
    std::ofstream out(filename.add_extension("csv").str());
    if (!out.is_open())
        U_EXCEPTION("Cannot open file '%S'.", &filename.str());
    
    auto write_line = [&out](const Row& row) {
        for (size_t i=0; i<row.size(); i++) {
            auto &column = row[i];
            
            if (utils::contains(column, ",")) {
                out << "\"" << column << "\"";
            } else {
                out << column;
            }
            
            if (i<row.size()-1) {
                out << ",";
            }
        }
        out << "\n";
    };
    
    Row head_row;
    for (auto &h : header)
        head_row.add(h);

    write_line(head_row);
    for (auto &row : table().rows()) {
        write_line(row);
    }
    
    return true;
}
}
