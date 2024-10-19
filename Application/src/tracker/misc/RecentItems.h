#pragma once
#include <commons.pc.h>
#include <gui/types/ScrollableList.h>
#include <misc/SpriteMap.h>
#include <gui/types/ListItemTypes.h>
#include <file/PathArray.h>
#include <file/Path.h>

/// JSON representations of the recent items data
struct RecentItemJSON {
    std::variant<uint64_t, std::string> created{cmn::timestamp_t::now().get()}, modified{0llu};
    std::string name, output_prefix;
    std::string output_dir, filename;
    std::map<std::string, glz::json_t> settings;

    cmn::sprite::Map _options;
    
    glz::json_t to_json() const;
    std::string toStr() const;
    static std::string class_name() { return "RecentItem"; }
    
    operator cmn::gui::DetailTooltipItem() const {
        cmn::gui::DetailTooltipItem item;
        item.set_name(std::string(cmn::file::Path(name).filename()));
        item.set_detail(name);
        return item;
    }
    
    cmn::timestamp_t t_modified() const;
    cmn::timestamp_t t_created() const;
};

template <>
struct glz::meta<RecentItemJSON> {
   using T = RecentItemJSON;
   static constexpr auto value = object(
            &T::created, 
            &T::modified, 
            &T::name, &T::output_prefix,
            &T::output_dir, &T::filename,
            &T::settings,
            "options", hide{&T::_options});
};

struct RecentItemFile {
    std::vector<RecentItemJSON> entries;
    std::variant<uint64_t, std::string> modified{0llu};
};

class RecentItems {
protected:
    GETTER(std::vector<RecentItemJSON>, items);

    void add(std::string name, const cmn::sprite::Map& options);
    void write();

    std::string toStr() const;
    static std::string class_name() { return "RecentItems"; }

public:
    static void open(const cmn::file::PathArray&, const cmn::sprite::Map& settings);
    static void set_select_callback(std::function<void(RecentItemJSON)>);
    bool has(std::string) const;
    void show(cmn::gui::ScrollableList<cmn::gui::DetailTooltipItem>& list);
    static RecentItems read();
};
