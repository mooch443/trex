#include "Table.h"

namespace gui {
    Table::Table()
        : _default_font(0.75)
    {
        
    }
    
    void Table::update() {
        if(!content_changed())
            return;
        
        begin();
        
        using column_t = size_t;
        //using row_t = size_t;
        
        std::map<column_t, Bounds> title_bounds;
        for(const auto & [index, title] : _columns) {
            title_bounds[index] = advance(new Text(title, Vec2(), White, Font(_default_font.size * 1.25, Style::Bold)))->bounds();
        }
        
        end();
    }
    
    void Table::add_column(Column col) {
        
    }
    
    void Table::add_row(const Row& row) {
        
    }
}
