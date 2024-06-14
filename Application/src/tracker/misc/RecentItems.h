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
        cmn::timestamp_t _created;
        cmn::sprite::Map _options;

        nlohmann::json to_json() const;
        std::string toStr() const;
        static std::string class_name() { return "RecentItems::Item"; }
        
        operator cmn::gui::DetailItem() const {
            cmn::gui::DetailItem item;
            item.set_name(std::string(cmn::file::Path(_name).filename()));
            item.set_detail(_name);
            return item;
        }
    };
    
protected:
    GETTER(std::vector<Item>, items);

    void add(std::string name, const cmn::sprite::Map& options);
    void write();

    std::string toStr() const;
    static std::string class_name() { return "RecentItems"; }

public:
    static void open(const cmn::file::PathArray&, const cmn::sprite::Map& settings);
    static void set_select_callback(std::function<void(Item)>);
    bool has(std::string) const;
    void show(cmn::gui::ScrollableList<cmn::gui::DetailItem>& list);
    static RecentItems read();
};
