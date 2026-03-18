#include "GraphElement.h"
#include <gui/DrawGraph.h>

namespace cmn::gui {

using namespace dyn;

struct GraphPoints {
    std::string name;
    std::variant<std::vector<Vec2>, std::string> pts;
    
    const std::vector<Vec2>& get_points() const {
        assert(not is_pattern());
        return std::get<std::vector<Vec2>>(pts);
    }
    const std::string& get_pattern() const {
        assert(is_pattern());
        return std::get<std::string>(pts);
    }
    
    bool is_pattern() const {
        return std::holds_alternative<std::string>(pts);
    }

    glz::json_t to_json() const {
        glz::json_t out;
        //out["name"] = glz::json_t(name);
        //out["pts"] = cvt2json(pts);
        
        // Serialize this object to a JSON string first (leveraging the glz::meta below)
        auto s = glz::write_json(*this);
        if (!s) {
            // Bubble up a helpful error if serialization fails
            throw RuntimeError(glz::format_error(s.error(), ""));
        }
        // Parse the JSON string into a dynamic glz::json_t
        auto ec = glz::read_json(out, *s);
        if (ec != glz::error_code::none) {
            throw RuntimeError(glz::format_error(ec, *s));
        }
        return out;
    }
    std::string toStr() const {
        return glz::write_json(*this).value_or("null");
    }
    
    static GraphPoints fromStr(cmn::StringLike auto&& str) {
        GraphPoints pts;
        auto error = glz::read_json(pts, str);
        if(error != glz::error_code::none)
            throw RuntimeError(glz::format_error(error, str));
        return pts;
    }
    
    static void realize(Graph& graph, std::vector<GraphPoints>&& points, PatternMapType& patterns, const Context& context, State& state) {
        size_t i = 0;
        for(auto &pts : points) {
            if(auto it = patterns.find("points_pts"+Meta::toStr(i));
               it != patterns.end())
            {
                assert(pts.is_pattern());
                pts.pts = Meta::fromStr<std::vector<Vec2>>(it->second.realize(context, state));
                
            } else {
                assert(not pts.is_pattern());
            }
            
            assert(not pts.is_pattern());
            graph.add_points(pts.name, pts.get_points());
            
            ++i;
        }
    }
};

struct GraphLine {
    std::string name;
    std::variant<std::vector<Vec2>, std::string> pts;
    std::optional<bool> extrapolate;
    std::optional<bool> display_points;
    
    bool is_pattern() const {
        return std::holds_alternative<std::string>(pts);
    }
    
    const std::vector<Vec2>& get_line() const {
        assert(not is_pattern());
        return std::get<std::vector<Vec2>>(pts);
    }
    const std::string& get_pattern() const {
        assert(is_pattern());
        return std::get<std::string>(pts);
    }

    glz::json_t to_json() const {
        glz::json_t out;
        //out["name"] = glz::json_t(name);
        //out["pts"] = cvt2json(pts);
        
        // Serialize this object to a JSON string first (leveraging the glz::meta below)
        auto s = glz::write_json(*this);
        if (!s) {
            // Bubble up a helpful error if serialization fails
            throw RuntimeError(glz::format_error(s.error(), ""));
        }
        // Parse the JSON string into a dynamic glz::json_t
        auto ec = glz::read_json(out, *s);
        if (ec != glz::error_code::none) {
            throw RuntimeError(glz::format_error(ec, *s));
        }
        return out;
    }
    std::string toStr() const {
        return glz::write_json(*this).value_or("null");
    }
    
    static GraphLine fromStr(cmn::StringLike auto&& str) {
        GraphLine pts;
        auto error = glz::read_json(pts, str);
        if(error != glz::error_code::none)
            throw RuntimeError(glz::format_error(error, str));
        return pts;
    }
    
    void moveTo(Graph& graph) && {
        assert(not is_pattern());
        auto fn = Graph::Function{
            name, Graph::Type::CONTINUOUS, [pts = get_line(), extrapolate = this->extrapolate](float x) -> float {
                std::optional<Vec2> prev, after;
                auto fy = GlobalSettings::invalid();
                
                for(auto &pt : pts) {
                    if(pt.x == x)
                        return pt.y;
                    
                    if((not prev || pt.x > prev->x) && pt.x < x)
                        prev = pt;
                    
                    if((not after || pt.x < after->x) && pt.x > x)
                        after = pt;
                }
                
                if(prev && after) {
                    auto m = (after->y - prev->y) / (after->x - prev->x);
                    fy = (x - prev->x) * m + prev->y;
                    
                } else if(not extrapolate.has_value() || not *extrapolate) {
                    /// pass if we do not want to extrapolate
                    /// return GlobalSettings::invalid();
                    
                } else if(prev) {
                    std::optional<Vec2> second_prev;
                    for(auto &pt : pts) {
                        if((not second_prev || pt.x > second_prev->x) && pt.x < prev->x)
                            second_prev = pt;
                    }
                    
                    if(second_prev) {
                        auto m = (prev->y - second_prev->y) / (prev->x - second_prev->x);
                        fy = (x - prev->x) * m + prev->y;
                    }
                } else if(after) {
                    std::optional<Vec2> second_after;
                    for(auto &pt : pts) {
                        if((not second_after || pt.x < second_after->x) && pt.x > after->x)
                            second_after = pt;
                    }
                    
                    if(second_after) {
                        auto m = (second_after->y - after->y) / (second_after->x - after->x);
                        fy = (x - after->x) * m + after->y;
                    }
                }
                
                return fy;
            }
        };
        
        //fn._points = std::make_shared<Graph::Points>(Graph::Points{fn, points});
        auto f = graph.add_function(fn);
        
        if(display_points.has_value() && *display_points) {
            graph.add_points("", get_line(), f._color);
        }
    }
    
    static void realize(Graph& graph, std::vector<GraphLine>&& lines, PatternMapType& patterns, const Context& context, State& state) {
        size_t i = 0;
        for(auto &line : lines) {
            if(auto it = patterns.find("lines_pts"+Meta::toStr(i));
               it != patterns.end())
            {
                assert(line.is_pattern());
                line.pts = Meta::fromStr<std::vector<Vec2>>(it->second.realize(context, state));
            } else {
                assert(not line.is_pattern());
            }
            
            std::move(line).moveTo(graph);
            
            ++i;
        }
    }
};

GraphElement::GraphElement()
{
    name = "graph";
    
    create = [this](LayoutContext& context){
        return _create(context);
    };
    update = [this](Layout::Ptr& o,
                    const Context& context,
                    State& state,
                    auto& patterns)
    {
        return _update(o, context, state, patterns);
    };
}

GraphElement::~GraphElement() {
}

Layout::Ptr GraphElement::_create(LayoutContext& context) {
    [[maybe_unused]] auto center = context.get(int64_t(), "center");
    [[maybe_unused]] auto xrange = context.get(Range<float>(), "xrange");
    [[maybe_unused]] auto yrange = context.get(Range<float>(), "yrange");
    [[maybe_unused]] auto title = context.get(std::string(), "title");
    auto draggable = context.get(false, "draggable");

    auto ptr = derived_ptr<Graph>(new Graph{Bounds(context.pos, context.size), title, xrange, yrange});
    ptr->set_zero(center);

    if(context.has("points")) {
        /// add a points based function
        auto points = context.get(std::vector<GraphPoints>{}, "points");
        ptr->add_custom_data("points", (void*)new std::vector<GraphPoints>(points), [](void* ptr) {
            delete (std::vector<GraphPoints>*)ptr;
        });
        
        size_t i = 0;
        for(auto &point : points) {
            if(point.is_pattern()) {
                auto pattern = context.state.pattern().set(context.hash, "points_pts"+Meta::toStr(i), Pattern::prepare(std::get<std::string>(point.pts)));
                point.pts = Meta::fromStr<std::vector<Vec2>>(pattern.realize(context.context, context.state));
            }
            
            ptr->add_points(point.name, std::move(point.get_points()));
            
            ++i;
        }
        
    } else {
        ptr->remove_custom_data("points");
    }
    
    if(context.has("lines")) {
        /// add a points based function
        auto lines = context.get(std::vector<GraphLine>{}, "lines");
        ptr->add_custom_data("lines", (void*)new std::vector<GraphLine>(lines), [](void* ptr) {
            delete (std::vector<GraphLine>*)ptr;
        });
        
        size_t i = 0;
        for(auto &line : lines) {
            if(line.is_pattern()) {
                auto pattern = context.state.pattern().set(context.hash, "lines_pts"+Meta::toStr(i), Pattern::prepare(line.get_pattern()));
                line.pts = Meta::fromStr<std::vector<Vec2>>(pattern.realize(context.context, context.state));
            }
            
            ptr->add_function(Graph::Function(line.name, Graph::Type::CONTINUOUS, [pts = line.get_line()](float x) {
                float mind = FLT_MAX;
                auto y = GlobalSettings::invalid();
                for(auto &pt : pts) {
                    auto d = abs(pt.x - x);
                    if(d < mind) {
                        mind = d;
                        y = pt.y;
                    }
                }
                return y;
            }));
            
            ++i;
        }
        
    } else {
        ptr->remove_custom_data("lines");
    }
    
    ptr->set_draggable(draggable);

    return ptr;
}

template<typename T>
void apply_pattern(auto& patterns, StringLike auto && name, auto&& fn, const Context& context, State& state) {
    if(auto it = patterns.find(name);
       it != patterns.end())
    {
        fn(name, Meta::fromStr<T>(it->second.realize(context, state)));
    }
}

bool GraphElement::_update(Layout::Ptr& o,
            const Context& context,
            State& state,
            PatternMapType& patterns)
{
    auto graph = o.to<Graph>();
    
    apply_pattern<int64_t>(patterns, "center", [&](auto, auto&& center){
        graph->set_zero(center);
    }, context, state);
    
    std::optional<Rangef> xrange, yrange;
    apply_pattern<Rangef>(patterns, "xrange", [&](auto, auto&& obj){
        xrange = std::move(obj);
    }, context, state);
    apply_pattern<Rangef>(patterns, "yrange", [&](auto, auto&& obj){
        yrange = std::move(obj);
    }, context, state);
    
    if(xrange || yrange) {
        graph->set_ranges(xrange ? *xrange : Rangef(-FLT_MAX, FLT_MAX),
                          yrange ? *yrange : Rangef(-FLT_MAX, FLT_MAX));
    }
    
    apply_pattern<std::string>(patterns, "title", [&](auto, auto&& obj){
        graph->set_title(std::move(obj));
    }, context, state);
    
    apply_pattern<bool>(patterns, "draggable", [&](auto, bool obj){
        graph->set_draggable(obj);
    }, context, state);
    
    std::optional<std::vector<GraphPoints>> points;
    std::optional<std::vector<GraphLine>> lines;
    
    auto _points = (std::vector<GraphPoints>*)graph->custom_data("points");
    if(_points)
        points = *_points;
    auto _lines = (std::vector<GraphLine>*)graph->custom_data("lines");
    if(_lines)
        lines = *_lines;
    
    if(points || lines) {
        graph->clear();
        
        if(points)
            GraphPoints::realize(*graph, std::move(*points), patterns, context, state);
        if(lines)
            GraphLine::realize(*graph, std::move(*lines), patterns, context, state);
    }
    
    return false;
}

}



template<>
struct glz::meta<cmn::gui::GraphPoints> {
  using T = cmn::gui::GraphPoints;
  static constexpr auto value = glz::object(
    "name", &T::name,
    "pts", &T::pts
  );
};

template<>
struct glz::meta<cmn::gui::GraphLine> {
  using T = cmn::gui::GraphLine;
  static constexpr auto value = glz::object(
    "name", &T::name,
    "pts", &T::pts
  );
};
