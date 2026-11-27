#include <commons.pc.h>
#include <gui/types/StaticText.h>

// Baseline implementation compared with cmn::utils::find_splitting_point.
// Both versions use the linked commons library for text measurements.
namespace benchmark_local {
using namespace cmn;
using namespace cmn::gui;
using cmn::utils::calculate_bounds;

// Inline helpers that match the original set membership:
inline bool is_whitespace(char c) {
    // Characters allowed when close to the end of the segment.
    return c == ' ' || c == ':' || c == ',' || c == '/' || c == '\\';
}

inline bool is_extended_whitespace(char c) {
    // Characters allowed when further away from the end.
    return c == ' ' || c == '-' || c == ':' || c == ',' ||
           c == '/' || c == '\\' || c == '.' || c == '_';
}

size_t find_splitting_point(std::string_view str, const Float2_t w,
                              const Float2_t max_w, Drawable* reference,
                              const Font& font)
{
    Float2_t cw = w;
    size_t L = str.length();
    size_t idx = L;
    
    while (cw > max_w && idx > 1) {
        L = idx;
        do {
            --idx;
        } while ( idx &&
                  ((L - idx <= 10 && not is_whitespace(str[idx - 1])) ||
                   (L - idx > 10 && not is_extended_whitespace(str[idx - 1]))) );
        
        // If we couldn’t find a proper breaking point, exit the loop.
        if (idx == 0)
            break;
        
        // Test splitting at idx:
        Bounds bounds = calculate_bounds(
                                         std::string_view(str.data(), idx),
            //StaticText::RichString::parse(std::string_view(str.data(), idx)),
            reference, font
        );
        cw = utils::calculate_width(bounds);
    }
    
    if(not idx) {
        // If no acceptable break was found, try to subdivide the segment.
        cw = w;
        
        if(cw > max_w) {
            // we have to break it up.
            size_t len = str.length();
            size_t middle = len * 0.5;
            idx = middle;
            
            while (true) {
                if(len <= 1)
                    break;
                
                Bounds bounds = calculate_bounds(
                     std::string_view(str.data(), middle),
                    //StaticText::RichString::parse(std::string_view(str.data(), middle)),
                    reference, font
                );
                cw = utils::calculate_width(bounds);
                
                if(cw <= max_w) {
                    middle = middle + len * 0.25;
                    len = len * 0.5;
                } else { // cw > max_w
                    middle = middle - len * 0.25;
                    len = len * 0.5;
                }
            }
            
            idx = middle;
            if (!idx && not str.empty())
                idx = 1;
        }
        // Otherwise, the entire segment fits on a new line.
    }
    
    return idx;
}

using Settings = StaticText::Settings;
using RichString = StaticText::RichString;

void add_string(
        Drawable* reference,
        const Settings& _settings,
        RichString&& ptr,
        std::vector<RichString>& strings,
        Vec2& offset)
{
    if (_settings.max_size.x > 0 && not ptr.str.empty()) {
        Bounds bounds = utils::calculate_bounds(ptr.str, reference, ptr.font);
        const auto w = utils::calculate_width(bounds);
        const Float2_t max_w = _settings.max_size.x - _settings.margins.x - _settings.margins.x - offset.x;
        
        //Print("** ", utils::ShortenText(ptr->parsed, 15)," w=", w, " max=",max_w, " font=",ptr->font);

        if (w > max_w) {
            size_t idx = benchmark_local::find_splitting_point(ptr.str, w, max_w, reference, ptr.font);
            
            offset.y++;
            offset.x = 0;

            if (idx) {
                /// in case we need to add a next line
                std::optional<RichString> add_after;
                
                if(ptr.str.length() > idx) {
                    auto trimmed = utils::ltrim(ptr.str.substr(idx));
                    if(trimmed.length() > 0) {
                        // if there is some remaining non-whitespace
                        // string, add it recursively
                        RichString tmp;
                        tmp.str = trimmed;
                        tmp.font = ptr.font;
                        tmp.pos.x = ptr.pos.x;
                        tmp.pos.y = ptr.pos.y + 1;
                        tmp.clr = ptr.clr;
                        
                        add_after = std::move(tmp);
                    }
                }
                
                /// shorten the current line segment
                ptr.str = ptr.str.substr(0, idx);
                strings.push_back(std::move(ptr));
                
                /// ensure the order remains correct here, add
                /// the next line after the current line
                if(add_after) {
                    add_string(reference, _settings, std::move(add_after.value()), strings, offset);
                }
                
                return;

            } else {
                // put the whole text in the next line
                ptr.pos.y++;
            }
        }
        
        offset.x += w;
    }

    strings.push_back(std::move(ptr));
}

}
