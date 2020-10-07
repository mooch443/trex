#pragma once

#include <gui/DrawStructure.h>
#include <gui/types/Entangled.h>

namespace gui {
    class Table : public Entangled {
    public:
        struct Column {
            size_t index;
            std::string title;
        };
        
        class Row {
        protected:
            GETTER_SETTER(size_t, index)
            
        private:
            std::map<size_t, std::string> items;
            std::map<size_t, bool> rich_text;
            
        public:
            Row(size_t index = 0) : _index(index) {}
            
            void set(size_t index, const std::string& value, bool rich_text = false) {
                items[index] = value;
                if(rich_text)
                    this->rich_text[index] = true;
            }
            
            std::string item(size_t i) const {
                auto it = items.find(i);
                if(it == items.end())
                    return std::string();
                return it->second;
            }
            
            bool is_rich_text(size_t i) const {
                auto it = rich_text.find(i);
                if(it == rich_text.end() || !it->second)
                    return false;
                return true;
            }
        };
        
    private:
        std::set<Column> _columns;
        std::set<Row> _rows;
        Font _default_font;
        
        std::map<size_t, std::shared_ptr<StaticText>> _rich_cache;
        
    public:
        Table();
        virtual ~Table() {}
        
        void add_column(Column);
        void add_row(const Row&);
        
        void update() override;
    };
};
