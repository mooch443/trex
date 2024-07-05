#include "DrawPreviewImage.h"
#include <gui/GUICache.h>
#include <tracking/PPFrame.h>
#include <gui/types/Textfield.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Checkbox.h>
#include <gui/types/SettingsTooltip.h>
#include <grabber/misc/default_config.h>
#include <tracking/LockGuard.h>
#include <tracking/Individual.h>

namespace cmn::gui {
namespace meta {

constexpr double default_element_width = 200;

struct LabeledField {
    gui::derived_ptr<gui::Text> _text;
    std::string _docs;
    //gui::derived_ptr<gui::HorizontalLayout> _joint;
    
    LabeledField(const std::string& name = "")
        : _text(std::make_shared<gui::Text>(Str(name)))
          //_joint(std::make_shared<gui::HorizontalLayout>(std::vector<Layout::Ptr>{_text, _text_field}))
    {
        _text->set_font(Font(0.6f, Style::Bold));
        _text->set_color(White);
    }
    
    virtual ~LabeledField() {}
    
    virtual void add_to(std::vector<Layout::Ptr>& v) {
        v.push_back(_text);
    }
    virtual void update() {}
    virtual Drawable* representative() { return _text.get(); }
};
struct LabeledTextField : public LabeledField {
    gui::derived_ptr<gui::Textfield> _text_field;
    sprite::Reference _ref;
    LabeledTextField(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_text_field);
    }
    void update() override;
    Drawable* representative() override { return _text_field.get(); }
};
struct LabeledDropDown : public LabeledField {
    gui::derived_ptr<gui::Dropdown> _dropdown;
    sprite::Reference _ref;
    LabeledDropDown(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_dropdown);
    }
    void update() override;
    Drawable* representative() override { return _dropdown.get(); }
};
struct LabeledCheckbox : public LabeledField {
    gui::derived_ptr<gui::Checkbox> _checkbox;
    sprite::Reference _ref;
    LabeledCheckbox(const std::string& name = "");
    void add_to(std::vector<Layout::Ptr>& v) override {
        LabeledField::add_to(v);
        v.push_back(_checkbox);
    }
    void update() override;
    Drawable* representative() override { return _checkbox.get(); }
};

LabeledCheckbox::LabeledCheckbox(const std::string& name)
    : LabeledField(name),
      _checkbox(std::make_shared<gui::Checkbox>(Str{name})),
      _ref(GlobalSettings::map()[name])
{
    _docs = GlobalSettings::docs()[name];
    
    _checkbox->set_checked(_ref.value<bool>());
    _checkbox->set_font(Font(0.7f));

    _checkbox->on_change([this](){
        try {
            _ref.get() = _checkbox->checked();

        } catch(...) {}
    });
}

void LabeledCheckbox::update() {
    _checkbox->set_checked(_ref.value<bool>());
}

LabeledTextField::LabeledTextField(const std::string& name)
    : LabeledField(name),
      _text_field(std::make_shared<gui::Textfield>(Box(0, 0, default_element_width, 28))),
      _ref(GlobalSettings::map()[name])
{
    _text_field->set_placeholder(name);
    _text_field->set_font(Font(0.7f));
    
    _docs = GlobalSettings::docs()[name];

    update();
    _text_field->on_text_changed([this](){
        try {
            _ref.get().set_value_from_string(_text_field->text());

        } catch(...) {}
    });
}

void LabeledTextField::update() {
    auto str = _ref.get().valueString();
    if(str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        str = str.substr(1,str.length()-2);
    }
    _text_field->set_text(str);
}

LabeledDropDown::LabeledDropDown(const std::string& name)
    : LabeledField(name),
      _dropdown(std::make_shared<gui::Dropdown>(Box(0, 0, default_element_width, 28))),
      _ref(GlobalSettings::map()[name])
{
    _docs = GlobalSettings::docs()[name];

    _dropdown->textfield()->set_font(Font(0.7f));
    assert(_ref.get().is_enum());
    std::vector<Dropdown::TextItem> items;
    int index = 0;
    for(auto &name : _ref.get().enum_values()()) {
        items.push_back(Dropdown::TextItem(name, index++));
    }
    _dropdown->set_items(items);
    _dropdown->select_item(Dropdown::RawIndex{narrow_cast<long>(_ref.get().enum_index()())});
    _dropdown->textfield()->set_text(_ref.get().valueString());
    
    _dropdown->on_select([this](auto index, auto) {
        if(not index.valid())
            return;
        
        try {
            _ref.get().set_value_from_string(_ref.get().enum_values()().at((size_t)index.value));
        } catch(...) {}
        
        _dropdown->set_opened(false);
    });
}

void LabeledDropDown::update() {
    _dropdown->select_item(Dropdown::RawIndex{narrow_cast<long>(_ref.get().enum_index()())});
}

}
}

namespace cmn::gui {
namespace DrawPreviewImage {

Entangled preview;
using namespace default_config;

std::map<std::string, std::unique_ptr<meta::LabeledField>> fields;
VerticalLayout layout;
SettingsTooltip tooltip;

Image::Ptr convert_image_to_rgba(Image::Ptr&& image,
                         meta_encoding_t::Class from, bool is_r3g3b2)
{
    if(image->dims == 4) {
        /// must already be in RGBA format!
        FormatWarning("Null operation since the image is already in 4-channel format!");
        return std::move(image);
    }
    
    auto ptr = Image::Make(image->rows, image->cols, 4);
    cv::Mat output = ptr->get();
    
    switch(from) {
        case meta_encoding_t::data::values::r3g3b2: {
            if(image->channels() == 1) [[likely]] {
                /// got a greyscale r3g3b2 image input -> just convert
                convert_from_r3g3b2<4, 1, true>(image->get(), output);
                return ptr;
                
            } else if(image->channels() == 3) {
                /// got a 3 channel format, so maybe we have a RGB image?
                //cv::cvtColor(image->get(), output, cv::COLOR_BGR2BGRA);
                //return ptr;
            }
            
            break;
        }
        case meta_encoding_t::data::values::rgb8: {
            if(image->channels() == 3) [[likely]] {
                /// this is the most likely option
                cv::cvtColor(image->get(), output, cv::COLOR_BGR2BGRA);
                return ptr;
                
            } else if(image->channels() == 1) {
                /// likely a grayscale image
                /// but could also be r3g3b2!
                if(is_r3g3b2) {
                    convert_from_r3g3b2<4, 1, true>(image->get(), output);
                } else {
                    cv::cvtColor(image->get(), output, cv::COLOR_GRAY2BGRA);
                }
                return ptr;
            }
            
            break;
        }
        case meta_encoding_t::data::values::gray: {
            if(image->channels() == 3) {
                /// this was likely from a rgb8 image
                cv::cvtColor(image->get(), output, cv::COLOR_BGR2BGRA);
                return ptr;
            } else if(image->channels() == 1) {
                if(is_r3g3b2) {
                    convert_from_r3g3b2<4, 1, true>(image->get(), output);
                } else {
                    cv::cvtColor(image->get(), output, cv::COLOR_GRAY2BGRA);
                }
                return ptr;
            }
            break;
        }
    }
    
    throw InvalidArgumentException("Illegal conversion from ", *image, " to RGBA from ", from);
}

std::tuple<Image::Ptr, Vec2> make_image(pv::BlobWeakPtr blob,
                                        const track::Midline* midline,
                                        const track::constraints::FilterCache* filters,
                                        const track::Background* background)
{
    const auto normalize = SETTING(individual_image_normalization).value<individual_image_normalization_t::Class>();
    auto output_shape = FAST_SETTING(individual_image_size);
    auto transform = midline ? midline->transform(normalize) : gui::Transform();
    
    auto &&[image, pos] = constraints::diff_image(normalize, blob, transform, filters ? filters->median_midline_length_px : 0, output_shape, background);
    
    if(not image)
        return {nullptr, Vec2{}};
    
    return {
        convert_image_to_rgba(std::move(image), Background::meta_encoding(), false),
        pos
    };
}

void draw(const Background* average, const PPFrame& pp,Frame_t frame, DrawStructure& graph) {
    if(not SETTING(gui_show_individual_preview)) {
        return; //! function is disabled
    }
    
    if(fields.empty()) {
#define ADD_FIELD(TYPE, NAME) fields[NAME] = std::make_unique<meta:: TYPE>(NAME)
        ADD_FIELD(LabeledTextField, "individual_image_size");
        ADD_FIELD(LabeledTextField, "individual_image_scale");
        ADD_FIELD(LabeledDropDown, "individual_image_normalization");
        ADD_FIELD(LabeledCheckbox, "track_background_subtraction");
        ADD_FIELD(LabeledDropDown, "meta_encoding");
        
        std::vector<Layout::Ptr> objects;
        for(auto &[key, obj] : fields)
            obj->add_to(objects);
        layout.set_children(objects);
    }
    
    
    static bool first = true;
    
    auto& cache = GUICache::instance();
    Loc offset(5);
    
    /*PPFrame pp;
    try {
        pv::Frame vframe;
        pp.set_index(frame);
        GUI::video_source()->read_frame(vframe, frame);
        Tracker::preprocess_frame(std::move(vframe), pp, nullptr, PPFrame::NeedGrid::NoNeed);
    } catch(const UtilsException& e) {
        UNUSED(e);
#ifndef NDEBUG
        FormatError("DrawPreviewImage failed for frame ", frame, ": ", e.what());
#endif
    }*/
    
    LockGuard guard(ro_t{}, "DrawPreviewImage", 100);
    if(!guard.locked() && !first) {
        graph.wrap_object(preview);
        return;
    }
    
    auto size = graph.dialog_window_size();
    
    static StaticText text(Str("Select individuals to preview their images using the settings shown below. Adjusting these settings here will affect <b>visual identification</b>, <b>categorization</b> and <b>tracklet images</b>.\n\nTry to keep images as small as possible, while still capturing all important details."), Loc(offset), SizeLimit(240, 0), Font(0.65));
    static Button button(Str("x"), Box(5, 5, 25, 25));
    button.set_scale(graph.scale().reciprocal());
    offset.x += button.local_bounds().width + 10;
    
    preview.update([&](Entangled& e) {
        ExternalImage *ptr{nullptr};
        auto bds = e.add<Text>(Str("Image settings"), offset, TextClr(White.alpha(200)), Font(0.75, Style::Bold), Scale(graph.scale().reciprocal()))->local_bounds();
        
        offset.y += bds.height + 10;
        offset.x = 5;
        
        if(cache.selected.empty()) {
            text.set_pos(offset);
            text.set_scale(graph.scale().reciprocal());
            e.advance_wrap(text);
            offset.y += text.local_bounds().height;
            offset.x = 5;
        }
        
        auto lock = cache.lock_individuals();
        for(auto idx : cache.selected) {
            // check whether this id has an image for the current frame
            if(not cache.active_ids.contains(idx))
                continue;
            
            auto it = lock.individuals.find(idx);
            if(it == lock.individuals.end())
                continue;
            
            auto fish = it->second;
            if(!fish->has(frame))
                continue;
            
            auto blob = fish->compressed_blob(frame);
            auto pixels = pp.bdx_to_ptr(blob->blob_id());
            if(!pixels) {
                FormatWarning("Cannot find ", blob->blob_id(), " in frame ", frame, ".");
                continue;
            }
            
            auto midline = fish->midline(frame);
            
            auto segment = fish->segment_for(frame);
            if(!segment)
                U_EXCEPTION("Cannot find segment for frame ", frame, " in fish ", idx, " despite finding a blob ", *blob);
            
            auto filters = constraints::local_midline_length(fish, segment->range);
            auto &&[image, pos] = make_image(pixels, midline.get(), filters.get(), average);
            
            if(!image || image->empty())
                continue;
            
            auto scale = graph.scale().reciprocal().mul(200.0 / image->cols, 200.0 / image->rows);
            ptr = e.add<ExternalImage>(std::move(image), offset, scale);
            
            e.add<Text>(Str(Identity::Temporary(idx).name()), Loc(offset + Vec2(5, 2)), TextClr(White.alpha(200)), Font(0.5), Scale(graph.scale().reciprocal()));
            
            offset.x += ptr->local_bounds().width + 5;
            if(offset.x >= size.width * 0.25) {
                offset.y += ptr->local_bounds().height + 5;
                offset.x = 5;
            }
        }
        
        if(offset.x > 5) {
            offset.x = 5;
            if(ptr) {
                offset.y += ptr->local_bounds().height + 5;
            }
        }
        
        e.advance_wrap(button);
        
        layout.set_scale(graph.scale().reciprocal());
        layout.set_pos(offset);
        e.advance_wrap(layout);
    });
    
    preview.auto_size({5.0,5.0});
    
    if(first) {
        preview.set_pos(graph.dialog_window_size().mul(0.5, 0.4));
        preview.set_origin(Vec2(0.5, 0));
        preview.set_clickable(true);
        preview.set_draggable();
        preview.set_background(DarkCyan.exposure(0.5).alpha(100), Red.alpha(50));
        
        text.set_clickable(false);
        text.set_background(Black.alpha(5), Transparent);
        
        button.on_click([](auto){
            SETTING(gui_show_individual_preview) = false;
        });
        button.set_font(Font(0.5, Style::Regular, Align::Center));
        first = false;
    }
    
    graph.wrap_object(preview);
    
    for(auto &[k, f] : fields) {
        if(f->representative()
           && f->representative()->hovered())
        {
            tooltip.set_other(f->representative());
            tooltip.set_parameter(k);
            graph.wrap_object(tooltip);
            break;
        }
    }
}

}
}
