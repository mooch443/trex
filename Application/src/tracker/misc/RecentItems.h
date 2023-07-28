#pragma once
#include <commons.pc.h>
#include <nlohmann/json.hpp>
#include <gui/types/ScrollableList.h>
#include <misc/SpriteMap.h>

class RecentItems {
    struct Item {
        std::string _name;
        cmn::sprite::Map _options;

        nlohmann::json to_object() const;
        std::string toStr() const;
        static std::string class_name() { return "RecentItems::Item"; }
    };
    GETTER(std::vector<Item>, items)

        void add(std::string name, const cmn::sprite::Map& options);
    void write();

    std::string toStr() const;
    static std::string class_name() { return "RecentItems"; }

public:
    static void open(std::string name, const cmn::sprite::Map& settings);
    bool has(std::string) const;
    void show(gui::ScrollableList<>& list);
    static RecentItems read();
};
