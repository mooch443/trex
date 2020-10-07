#include "DrawSFBase.h"
#include <misc/GlobalSettings.h>
#include <gui/types/Textfield.h>
#include <misc/Timer.h>

static std::recursive_mutex cache_lock;

namespace gui {
#if WITH_SFML
    const sf::Font& SFBase::font() {
        static bool _font_loaded = false;
        static sf::Font _font;
        if(!_font_loaded) {
			file::Path path("fonts/Avenir.ttc");
			if (!path.exists())
				U_EXCEPTION("Cannot find file '%S'", &path.str());
            if(!_font.loadFromFile(path.str())) {
                U_EXCEPTION("Cannot load '%S'", &path.str());
            }
            else
                _font_loaded = true;
        }
        return _font;
    }
    
    static bool setting_nowindow;
    static bool setting_nowindow_updated = false;
    
    SFBase::RenderState::RenderState(SFBase *base, DrawStructure &s)
        : _obj(NULL),
          _cache(NULL),
          _target(base->has_texture() ? (sf::RenderTarget*)&base->_texture : &base->window()),
          _base(base),
          _graph(s)
    {}
    
    void SFBase::RenderState::set_cache(sf::Drawable *d) {
        if(_cache)
            _cache = nullptr;
        
        if(d) {
            _cache = std::shared_ptr<CacheObject>(new Cache(d));
            _obj->insert_cache(_base, _cache);
            
        } else {
            _obj->remove_cache(_base);
        }
    }
    
    void SFBase::RenderState::set_object(gui::Drawable *o) {
        if(o->type() == Type::SINGLETON)
            o = static_cast<SingletonObject*>(o)->ptr();
        
        _obj = o;
        auto ptr = o->cached(_base);
        _cache = ptr;
        //_cache = static_cast<Cache*>(ptr.get());
        
        Transform bounds;
        if(_base->has_texture())
            bounds.scale(2, 2);
        bounds.translate(_base->offset(_graph));
        bounds.scale(_graph.scale());
        bounds.combine(o->global_transform());
        
        if(o->type() == Type::TEXT)
            bounds.scale(1/_graph.scale().x, 1/_graph.scale().y);
        
        _render.transform = bounds;
    }
    
    Vec2 SFBase::offset(const gui::DrawStructure &graph) const {
        Vec2 real_size = Vec2(_window.getView().getSize().x, _window.getView().getSize().y).div(graph.scale());
        Vec2 virtual_size = Vec2(graph.width(), graph.height());
        
        if(GlobalSettings::has("gui_no_offset"))
            return Vec2();
        
        return ((real_size - virtual_size) * 0.5).mul(graph.scale());
    }
    
    Event SFBase::toggle_fullscreen(DrawStructure& graph) {
        std::lock_guard<std::recursive_mutex> guard(graph.lock());
        for(auto o : graph.collect()) {
            o->clear_cache();
        }
        _is_fullscreen = !_is_fullscreen;
        
        if(_is_fullscreen) {
            _video_mode = sf::VideoMode(_window.getSize().x, _window.getSize().y);
            
            auto modes = sf::VideoMode::getFullscreenModes();
            Debug("Supported:");
            for(auto &mode: modes)
                Debug("%dx%d @ %d", mode.width, mode.height, mode.bitsPerPixel);
            
            auto mode = modes.front();
            Debug("using (first) mode: %dx%d @ %d", mode.width, mode.height, mode.bitsPerPixel);
            _window.create(mode, _title, sf::Style::Fullscreen);
            _window.setView(sf::View(sf::FloatRect(0, 0, mode.width, mode.height)));
            _window.setActive(false);
            
            Event event(WINDOW_RESIZED);
            event.size.width = mode.width;
            event.size.height = mode.height;
            
            graph.event(event);
            return event;
            
        } else {
            _window.create(_video_mode, _title);
            _window.setView(sf::View(sf::FloatRect(0, 0, _video_mode.width, _video_mode.height)));
            _window.setActive(false);
            
            Event event(WINDOW_RESIZED);
            event.size.width = _video_mode.width;
            event.size.height = _video_mode.height;
            
            graph.event(event);
            return event;
        }
    }
    
    void SFBase::RenderState::draw() {
        if(_cache) {
            auto rect = dynamic_cast<sf::RectangleShape*>(((Cache*)_cache.get())->tp());
            if(rect && rect->getOutlineThickness() > 0) {
                auto pt = _render.transform.transformRect(sf::FloatRect(0,0,1,1));
                Vec2 scale = Vec2(pt.width, pt.height).reciprocal();
                rect->setOutlineThickness(max(1, max(scale.y, scale.x)));
            }
            
            _target->draw(*((Cache*)_cache.get())->tp(), _render);
            _cache->set_changed(false);
        }
    }
    
    SFBase::Cache::Cache(sf::Drawable*d) : _tp(d) {
    }
    
    SFBase::Cache::~Cache() {
        delete _tp;
    }
    
    Path::Path()
    {}
    
    void Path::draw(const std::vector<Vertex>& vertices, float thickness, float scale, sf::VertexArray& array)
    {
        array.setPrimitiveType(sf::LineStrip);
        
        if(thickness > 1.0) {
            array = SFBase::GenerateTrianglesStrip(vertices, thickness, scale);
        }
        else {
            array.clear();
            for (auto &vt : vertices)
                array.append(sf::Vertex(vt.position() * scale, vt.color()));
        }
    }
    
    SFBase::SFBase(const std::string& title, const Size2& size, uint32_t style)
        : SFBase(title, Bounds(Vec2(-1, -1), size), style)
    {}
    
    SFBase::SFBase(const std::string& title, const Bounds& bounds, uint32_t style)
        : _window(sf::VideoMode(bounds.width, bounds.height), title, style),
          _is_fullscreen(false), _title(title),
    //texture_size(0, 0),
          _has_texture(false),
          _auto_display(true)
    {
        if(bounds.x >= 0 || bounds.y >= 0)
            _window.setPosition(sf::Vector2i(bounds.x, bounds.y));
        else
            _window.setPosition(sf::Vector2i(_window.getPosition().x, (sf::VideoMode::getDesktopMode().height - _window.getSize().y) * 0.5));
        _window.requestFocus();
        _window.setVerticalSyncEnabled(true);
        
        if(SETTING(gui_transparent_background))
            _has_texture = true;
        
        GlobalSettings::map().register_callback(this, [this](auto&, auto& key, auto& value){
            if(key == "gui_transparent_background") {
                _has_texture = value.template value<bool>();
            }
        });
        
        /*const std::string vertexShader = \
            "void main() \
            { \
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \
                gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;\
                gl_FrontColor = gl_Color; \
            }";
        
        /const std::string fragmentShader = "uniform sampler2D texture; uniform float amount; \
        void main() \
        { \
            vec2 uv = gl_TexCoord[0].st; \
            vec4 color = vec4(0.0, 0.0, 0.0, 1.0); \
            \
            int j = 0; \
        vec2 mv = vec2(0.0, 0.0); \
            float sum = 0.0; \
            int kernel_size = 1; \
            for(int i=-kernel_size; i<=kernel_size; i++) { \
                vec4 value = texture2D(texture, uv + mv); \
                float factor = float(kernel_size)+1.0 - abs(float(i)); \
                color += value * factor; \
                sum += factor; \
            } \
            color /= sum; \
            \
            gl_FragColor = color; \
            gl_FragColor.a = 0.25; \
        }";*/
        
        /*const std::string fragmentShader = \
        "uniform sampler2D texture; uniform float amount; \
            \
            void main() \
            { \
                vec4 pixel = texture2D(texture, gl_TexCoord[0].xy); \
                gl_FragColor = gl_Color * pixel; \
                gl_FragColor.a = amount; \
            }";
        
        // load only the vertex shader
        if (!shader.loadFromMemory(vertexShader, sf::Shader::Vertex))
        {
            // error...
            U_EXCEPTION("Cannot load vertex shader.");
        }
        
        // load only the fragment shader
        if (!shader.loadFromMemory(fragmentShader, sf::Shader::Fragment))
        {
            // error...
            U_EXCEPTION("Cannot load fragment shader.");
        }
        
        // load both shaders
        if (!shader.loadFromMemory(vertexShader, fragmentShader))
        {
            // error...
            U_EXCEPTION("Cannot load shaders.");
        }
        
        shader.setUniform("texture", sf::Shader::CurrentTexture);
        shader.setUniform("amount", float(0.5));*/
        
        _window.setActive(false);
    }
    
    SFBase::~SFBase() { }
    
    void SFBase::paint(gui::DrawStructure &s) {
        Timer timer;
        
        {
            std::lock_guard<std::recursive_mutex> guard(cache_lock);
            font(); // load font if not yet done
        }
        
        std::unique_lock<std::recursive_mutex> lock(s.lock());
        _window.clear(Black);
        if(_has_texture) {
            if(_texture.getSize().x != _window.getView().getSize().x * 2 || _texture.getSize().y != _window.getView().getSize().y * 2) {
                _texture.create(_window.getView().getSize().x * 2, _window.getView().getSize().y * 2, sf::ContextSettings(0, 0, 8));
                Debug("Updated texture %dx%d", _texture.getSize().x, _texture.getSize().y);
            }
            
            _texture.clear(Transparent);
        }
        
        s.before_paint(this);
        
        auto objects = s.collect();
        
        _last_draw_objects = 0;
        _last_draw_added = 0;
        _last_draw_repaint = 0;
        
        for (size_t i=0; i<objects.size(); i++) {
            auto o =  objects[i];
            
            RenderState state(this, s);
            state.set_object(o);
            
            if(state.cache() && !state.cache()->changed()) {
                state.draw();
                _last_draw_objects++;
                
            } else {
                repaint_object(state);
            }
            
            if(o->type() == Type::ENTANGLED && state.cache()) {
                auto entangled = static_cast<Entangled*>(o);
                
                if(entangled->background()
                   && entangled->background()->lineclr() != Transparent)
                {
                    sf::RectangleShape shape;
                    shape.setSize(entangled->size());
                    shape.setFillColor(Transparent);
                    shape.setOutlineColor(entangled->background()->lineclr());
                    shape.setOutlineThickness(1);
                    state.target()->draw(shape, state.render());
                }
            }
        }
        
        _last_draw_ms = timer.elapsed()*1000;
        
        if(_has_texture) {
            _texture.display();
            sf::Sprite sprite(_texture.getTexture());
            sprite.setScale(0.5, 0.5);
            _window.draw(sprite);
        }
        
        if(_auto_display)
            _window.display();
    }
    
    sf::VertexArray SFBase::GenerateTrianglesStrip(const std::vector<Vertex>& points, float thickness, float scale)
    {
        sf::VertexArray array(sf::PrimitiveType::TriangleStrip);
        
        for (size_t i=0; i<points.size(); i++) {
            auto ptr = &points.at(i);
            
            auto idx0 = i;
            auto idx1 = i+1;
            if(idx1 >= points.size()) {
                idx1 = idx0 ? (idx0 - 1) : 0;
            }
        
            /*auto norm = points.at(idx1).position() - points.at(idx0).position();
            norm /= length(norm);
            norm = Vec2(-norm.y, norm.x);
            
            gui::Color clr(ptr->color());
            gui::Color change(0, 0, 0, 0);
            float max_change = 0;
            uint8_t min_alpha = 255;
            change = clr;
            
            while (idx1 < points.size()-1) {
                auto line = points.at(idx1).position() - points.at(idx0).position();
                auto clr1 = points.at(idx1).color();
                
                min_alpha = min(clr1.a, min_alpha);
                
                float d = length(clr1 - clr);
                if(d >= max_change) {
                    change = clr1;
                    max_change = d;
                }
                
                auto len = length(line);
                line = line / len;
                
                if (len >= thickness * 2 || DEGREE(cmn::abs(dot(norm, line))) >= 0.15)
                    break;
                
                idx1++;
                i++;
            }*/
            
            Vec2 current = points.at(idx0).position();
            Vec2 next = points.at(idx1).position();
            
            auto direction = next-current;
            direction = direction / length(direction) * thickness / 2.f;
            
            Vec2 normal(-direction.y, direction.x);
            
            //change[3] = min_alpha;
            array.append(Vertex((- normal + Vec2(*ptr)) * scale, ptr->color()));
            array.append(Vertex((  normal + Vec2(*ptr)) * scale, ptr->color()));
        }
        
        return array;
    }

    uint32_t SFBase::line_spacing(const Font& font) {
        return SFBase::font().getLineSpacing(25 * font.size);
    }
    
    Bounds SFBase::text_bounds(const std::string& t, Drawable* obj, const Font& font) {
        if(!setting_nowindow_updated) {
            setting_nowindow_updated = true;
            setting_nowindow = GlobalSettings::map().has("nowindow") ? SETTING(nowindow).value<bool>() : false;
        }
        
        if(setting_nowindow)
            return Bounds(0, 0, t.length() * 11.3 * font.size, 26 * font.size);
        
        std::lock_guard<std::recursive_mutex> guard(cache_lock);
        static sf::Text text("", SFBase::font());
        
        if(obj) {
            try {
                auto gscale = obj->global_text_scale();
                text.setScale(gscale.reciprocal());
                text.setCharacterSize(font.size * gscale.x * 25);
            } catch(const UtilsException& ex) {
                Warning("Not initialising scale of (probably StaticText) fully because of a UtilsException.");
                text.setCharacterSize(font.size * 25);
                text.setScale(1, 1);
            }
            
        } else {
            text.setCharacterSize(font.size * 25);
            text.setScale(1, 1);
        }
        
        text.setString(t);
        text.setStyle(font.style);
        //text.setScale(real_scale);
        
        sf::FloatRect bounds;
        try {
            bounds = text.getGlobalBounds();// getLocalBounds();
        } catch(const std::exception& e) {
            Except("size:%f scale:0x%X '%s'", font.size, obj, e.what());
            return Bounds(0,0,1,1);
        }
        
        return Bounds(bounds.left, bounds.top, bounds.width, Base::default_line_spacing(font)); //
        //Bounds((bounds.left) / stage_scale.x / real_scale.x, (bounds.top) / stage_scale.y / real_scale.y, (bounds.width) / stage_scale.x / real_scale.x, Base::default_line_spacing(font));//(bounds.height) / stage_scale);
    }
    
    bool entangled_will_texture(Entangled* e) {
        assert(e);
        if(e->scroll_enabled() && e->size().max() > 0) {
            return true;
        }
        
        return false;
    }
    
    bool type_is_cachable(Drawable* obj) {
        assert(obj);
        if(obj->type() == Type::ENTANGLED && !entangled_will_texture(static_cast<Entangled*>(obj)))
            return false;
        return true;
    }
    
    void SFBase::repaint_object(RenderState state) {
        _last_draw_objects ++;
        _last_draw_repaint ++;
        
        switch (state.obj()->type()) {
            case Type::ENTANGLED: {
                SpriteWithRenderTexture *tex = NULL;
                bool tex_dirty = false;
                auto ptr = static_cast<Entangled*>(state.obj());
                Vec2 inside_scale(1);
                
                for(auto child : ptr->children()) {
                    auto c = child->cached(this);
                    if((!c && type_is_cachable(child)) || (c && c->changed())) {
                        tex_dirty = true;
                        break;
                    }
                }
                
                auto bg = ptr->background();
                if(bg && !tex_dirty) {
                    if(!bg->cached(this) || bg->cached(this)->changed())
                        tex_dirty = true;
                }
                
                if(entangled_will_texture(ptr)) {
                    const float interface_scale = gui::interface_scale();
                    
                    Drawable* current = ptr;
                    while (current) {
                        inside_scale = inside_scale.mul(current->scale());
                        current = current->parent();
                        if(!current) {
                            inside_scale = inside_scale.mul(state.graph().scale() / interface_scale);
                        }
                    }
                    
                    sf::Vector2u size(ptr->global_bounds().width * state.graph().scale().x / interface_scale,
                                      ptr->global_bounds().height * state.graph().scale().y / interface_scale);
                    
                    tex = state.get_cache<SpriteWithRenderTexture>();
                    if(!tex) {
                        tex = new SpriteWithRenderTexture(size.x, size.y);
                        state.set_cache(tex);
                        tex_dirty = true;
                    }
                    
                    if(tex->texture().getSize() != size) {
                        //Debug("Resizing texture %dx%d", size.x, size.y);
                        tex->texture().create(size.x, size.y);
                        tex->sprite().setTexture(tex->texture().getTexture(), true);
                        tex->texture().setSmooth(false);
                        tex_dirty = true;
                    }
                    
                    Color clear_color = Transparent;
                    if(ptr->background()) {
                        clear_color = ptr->background()->fillclr();
                        //if(!clear_color.r && !clear_color.g && !clear_color.b)
                        //    clear_color = clear_color.red(1).green(1).blue(1);
                    }
                    
                    if(!tex_dirty)
                        break;
                    tex->texture().clear(clear_color);
                    
                } else {
                    if(state.cache()) {
                        //Debug("Deleting old texture");
                        state.set_cache(NULL);
                    }
                    
                    if(ptr->background()) {
                        RenderState bg_state(this, state.graph());
                        bg_state.set_object(ptr->background());
                        
                        if(bg_state.cache() && !bg_state.cache()->changed())
                            bg_state.draw();
                        else
                            repaint_object(bg_state);
                    }
                }
                
                RenderState child_state(this, state.graph());
                if(tex)
                    child_state.set_target(&tex->texture());
                
                for(auto c: ptr->children()) {
                    if(ptr->scroll_enabled()) {
                        auto b = c->local_bounds();
                        
                        //! Skip drawables that are outside the view
                        //  TODO: What happens to Drawables that dont have width/height?
                        float x = b.x;
                        float y = b.y;
                        
                        if(y < -b.height || y > ptr->height()
                           || x < -b.width || x > ptr->width())
                        {
                            continue;
                        }
                    }
                    
                    child_state.set_object(c);
                    
                    if(tex) {
                        child_state.render().transform = Transform();
                        child_state.render().transform.scale(inside_scale);
                        child_state.render().transform.combine(c->local_transform());
                        if(c->type() == Type::TEXT)
                            child_state.render().transform.scale(1/state.graph().scale().x, 1/state.graph().scale().y);
                    }
                    
                    if(child_state.cache() && !child_state.cache()->changed()) {
                        child_state.draw();
                    } else {
                        repaint_object(child_state);
                    }
                    
                    if(c->type() == Type::ENTANGLED && child_state.cache()) {
                        auto entangled = static_cast<Entangled*>(c);
                        
                        if(entangled->background()
                           && entangled->background()->lineclr() != Transparent)
                        {
                            sf::RectangleShape shape;
                            shape.setSize(entangled->size());
                            shape.setFillColor(Transparent);
                            shape.setOutlineColor(entangled->background()->lineclr());
                            shape.setOutlineThickness(1);
                            child_state.target()->draw(shape, child_state.render());
                        }
                    }
                }
                
                if(tex) {
                    tex->texture().display();
                    tex->texture().setActive(false);
                    tex->sprite().setScale(inside_scale.reciprocal());
                }
                
                break;
            }
                
            case Type::TEXT: {
                auto ptr = static_cast<Text*>(state.obj());
                
                auto text = state.get_cache<sf::Text>();
                if(!text) {
                    text = new sf::Text(ptr->txt(), font());
                    state.set_cache(text);
                } else
                    text->setString(ptr->txt());
                
                auto scale = ptr->global_text_scale();
                text->setScale(scale.reciprocal().mul(state.graph().scale()));
                
                text->setCharacterSize(ptr->font().size * scale.x * 25);
                text->setStyle(ptr->font().style);
                text->setFillColor(ptr->color());
                
                break;
            }
                
            case Type::IMAGE: {
                auto ptr = static_cast<ExternalImage*>(state.obj());
                
                SpriteWithTexture *sprite = state.get_cache<SpriteWithTexture>();
                if(!sprite) {
                    sprite = new SpriteWithTexture;
                    state.set_cache(sprite);
                }
                
                if(sprite->texture().getSize().x != ptr->source()->cols || sprite->texture().getSize().y != ptr->source()->rows)
                {
                    sf::Image sfimg;
                    sfimg.create(ptr->source()->cols, ptr->source()->rows, ptr->source()->data());
                    
                    sprite->texture().loadFromImage(sfimg);
                    sprite->texture().setSmooth(true);
                    
                    sprite->sprite().setTexture(sprite->texture(), true);
                } else if(ptr->source()->timestamp() != sprite->timestamp()) {
                    sprite->texture().update(ptr->source()->data());
                    sprite->set_timestamp(ptr->source()->timestamp());
                }
                
                if (ptr->color().a > 0) {
                    sprite->sprite().setColor(ptr->color());
                } else if(sprite->sprite().getColor().a > 0)
                    sprite->sprite().setColor(White);
                
                break;
            }
                
            case Type::CIRCLE: {
                auto ptr = static_cast<Circle*>(state.obj());
                auto shape = state.get_cache<sf::CircleShape>();
                
                if(shape)
                    shape->setRadius(ptr->radius());
                else {
                    shape = new sf::CircleShape(ptr->radius());
                    state.set_cache(shape);
                }
                
                shape->setOutlineColor(ptr->color());
                shape->setOutlineThickness(1);
                shape->setFillColor(ptr->fillclr());
                shape->setOrigin(0, 0);
                
                break;
            }
                
            case Type::RECT: {
                auto ptr = static_cast<Rect*>(state.obj());
                
                auto shape = state.get_cache<sf::RectangleShape>();
                if(shape)
                    shape->setSize(ptr->size());
                else {
                    shape = new sf::RectangleShape(ptr->size());
                    state.set_cache(shape);
                }
                
                shape->setFillColor(ptr->fillclr());
                if(ptr->lineclr().a > 0) {
                    shape->setOutlineColor(ptr->lineclr());
                    shape->setOutlineThickness(1.5);
                } else
                    shape->setOutlineThickness(0);
                
                break;
            }
                
            case Type::VERTICES: {
                auto ptr = static_cast<Vertices*>(state.obj());
                
                auto shape = state.get_cache<sf::VertexArray>();
                if(!shape) {
                    shape = new sf::VertexArray;
                    state.set_cache(shape);
                }
                
                shape->setPrimitiveType((sf::PrimitiveType)ptr->primitive());
                if(dynamic_cast<Line*>(state.obj())) {
                    Path::draw(ptr->points(), static_cast<Line*>(state.obj())->thickness(), 1, *shape);
                    
                } else {
                    shape->clear();
                    for(auto &v : ptr->points()) {
                        shape->append(v);
                    }
                }
                
                break;
            }
                
            case Type::POLYGON: {
                auto ptr = static_cast<Polygon*>(state.obj());
                
                auto shape = state.get_cache<sf::ConvexShape>();
                if(!shape) {
                    shape = new sf::ConvexShape;
                    state.set_cache(shape);
                }
                
                if((!ptr->relative() && shape->getPointCount())
                   || shape->getPointCount() != ptr->relative()->size())
                {
                    if(ptr->relative()) {
                        shape->setPointCount(ptr->relative()->size());
                        for(size_t i=0; i<ptr->relative()->size(); ++i)
                            shape->setPoint(i, ptr->relative()->at(i));
                    } else
                        shape->setPointCount(0);
                } else if(shape->getPointCount() == ptr->relative()->size()) {
                    for(size_t i=0; i<ptr->relative()->size(); ++i)
                            shape->setPoint(i, ptr->relative()->at(i));
                }
                
                shape->setFillColor(ptr->fill_clr());
                shape->setOrigin(0, 0);
                
                break;
            }
                
            default: {
                auto type = state.obj()->type().name();
                U_EXCEPTION("Unknown type '%s' in SFBase.", type);
            }
        }
        
        state.draw();
    }
#endif
}

