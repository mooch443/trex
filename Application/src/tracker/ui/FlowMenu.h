#pragma once

#include <gui/types/PieChart.h>

namespace cmn::gui {
    class FlowMenu : public Entangled {
    public:
        class Layer {
        private:
            friend class FlowMenu;
            
            size_t _index;
            std::map<std::string, size_t> _links;
            std::vector<std::string> _names;
            std::string _title;
            
        public:
            Layer(const std::string&, const std::vector<std::string>&);
        };
        
    protected:
        std::function<void(size_t, const std::string&)> _clicked_leaf;
        std::vector<Layer> _layers;
        
        PieChart _pie;
        long_t _current;
        
    public:
        FlowMenu(float radius = 10, const decltype(_clicked_leaf)& clicked_leaf = [](auto,auto&){});
        
        size_t add_layer(Layer&& layer);
        void link(size_t from_layer, const std::string& item, size_t to_layer);
        void unlink(size_t from_layer, const std::string& item);
        void display_layer(size_t);
        
    protected:
        void clicked(size_t);
        void check_layer_index(size_t) const;
        std::vector<PieChart::Slice> generate_layer(size_t);
        
        void update() override;
    };
}
