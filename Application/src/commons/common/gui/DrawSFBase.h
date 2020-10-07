#ifndef _DRAW_SF_BASE_H
#define _DRAW_SF_BASE_H

#include "DrawBase.h"

#if WITH_SFML
#include <SFML/Graphics.hpp>

namespace gui {
    class Path {
    private:
        Path();
        
    public:
        static void draw(const std::vector<Vertex>& vertices, float thickness, float scale, sf::VertexArray& array);
    };
    
    class SpriteWithTexture : public sf::Drawable {
        GETTER_NCONST(sf::Sprite, sprite)
        GETTER_NCONST(sf::Texture, texture)
        GETTER_SETTER(size_t, timestamp)
        
    public:
        SpriteWithTexture() : _timestamp(0) {}
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const {
            target.draw(_sprite, states);
        }
    };
    
    class SpriteWithRenderTexture : public sf::Drawable {
        GETTER_NCONST(sf::Sprite, sprite)
        GETTER_NCONST(sf::RenderTexture, texture)
        
    public:
        SpriteWithRenderTexture(float width, float height) {
            _texture.create(width, height);
            _texture.setSmooth(true);
            _sprite.setTexture(_texture.getTexture(), true);
        }
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const {
            target.draw(_sprite, states);
        }
    };
    
    class SFBase : public Base {
    public:
        class Cache : public CacheObject {
        protected:
            GETTER_PTR(sf::Drawable*, tp)
            
        public:
            Cache(sf::Drawable *d);
            ~Cache();
        };
        
        //static std::set<Cache*> caches;
    private:
        class RenderState {
        protected:
            GETTER_PTR(Drawable*, obj)
            GETTER_PTR(SFBase::Cache::Ptr, cache)
            GETTER_SETTER_PTR(sf::RenderTarget*, target)
            GETTER_NCONST(sf::RenderStates, render)
            
        private:
            SFBase *_base;
            
        protected:
            GETTER_NREF(DrawStructure&, graph)
            
        public:
            RenderState(SFBase *base, DrawStructure &s);
            
            void set_object(Drawable* o);
            void set_cache(sf::Drawable *d);
            void draw();
            
            template<typename T>
            T* get_cache() {
                if(!_cache)
                    return NULL;
                
                return static_cast<T*>(((Cache*)_cache.get())->tp());
            }
        };
        
        GETTER_NCONST(sf::RenderWindow, window)
        
        GETTER(float, last_draw_ms)
        GETTER(size_t, last_draw_repaint)
        GETTER(size_t, last_draw_added)
        GETTER(size_t, last_draw_objects)
        GETTER(bool, is_fullscreen)
        GETTER(std::string, title)
        sf::VideoMode _video_mode;
        //sf::Shader shader;
        GETTER(sf::RenderTexture, texture)
        GETTER(bool, has_texture)
        //sf::Vector2f texture_size;
        
        GETTER_SETTER(bool, auto_display)

    public:
        static const sf::Font& font();
        
        SFBase(const std::string& title, const Bounds& bounds, uint32_t style = sf::Style::Default);
        SFBase(const std::string& title, const Size2& size, uint32_t style = sf::Style::Default);
        ~SFBase();

        virtual void paint(DrawStructure& s) override;
        void set_title(std::string str) override {
            _window.setTitle(str);
        }
        Size2 size() const {
            auto s = _window.getSize();
            return Size2(s.x, s.y);
        }
        
        Vec2 offset(const DrawStructure& graph) const;
        Event toggle_fullscreen(DrawStructure& graph);
        Size2 window_dimensions() override {
            auto s = _window.getSize();
            return Size2(s.x, s.y);
        }
        
        static sf::VertexArray GenerateTrianglesStrip(const std::vector<Vertex>& points, float thickness, float scale);
        
        Bounds text_bounds(const std::string& text, Drawable* obj, const Font& font) override;
        uint32_t line_spacing(const Font& font) override;
        
    protected:
        void repaint_object(RenderState state);
    };
}
#endif

#endif
