#pragma once
#include <commons.pc.h>
#include <gui/types/ScrollableList.h>
#include <misc/SpriteMap.h>
#include <gui/types/ListItemTypes.h>
#include <file/PathArray.h>

class RecentItems {
public:
    struct Item {
        std::string _name;
        cmn::sprite::Map _options;

        nlohmann::json to_object() const;
        std::string toStr() const;
        static std::string class_name() { return "RecentItems::Item"; }
        
        operator gui::DetailItem() const {
            gui::DetailItem item;
            item.set_name(std::string(file::Path(_name).filename()));
            item.set_detail(_name);
            return item;
        }
    };
    
protected:
    GETTER(std::vector<Item>, items)

    void add(std::string name, const cmn::sprite::Map& options);
    void write();

    std::string toStr() const;
    static std::string class_name() { return "RecentItems"; }

public:
    static void open(const file::PathArray&, const cmn::sprite::Map& settings);
    static void set_select_callback(std::function<void(Item)>);
    bool has(std::string) const;
    void show(gui::ScrollableList<gui::DetailItem>& list);
    static RecentItems read();
};
