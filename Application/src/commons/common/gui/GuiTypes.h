#ifndef _GUI_TYPES_H
#define _GUI_TYPES_H

#include <misc/defines.h>

#include <gui/types/Drawable.h>
#include <misc/Image.h>
#include <gui/colors.h>

#define CHANGE_SETTER(NAME) virtual void set_ ## NAME (const decltype( _ ## NAME ) & NAME) { \
if ( _ ## NAME == NAME ) return; _ ## NAME = NAME; set_dirty(); }

namespace gui {
    enum PrimitiveType
    {
        Points,
        Lines,
        LineStrip,
        Triangles,
        TriangleStrip
    };

class VertexArray : public Drawable {
public:
    enum MEMORY {
        TRANSPORT,
        COPY
    };
    
protected:
    const std::vector<Vertex>* _transport;
    std::shared_ptr<std::vector<Vertex>> _points;
    std::vector<Vertex> _original_points;
    GETTER(PrimitiveType, primitive)
    GETTER(bool, size_calculated)
    GETTER(float, thickness)
    
public:
    VertexArray(const std::vector<Vertex>& p, PrimitiveType primitive, MEMORY memory);
    virtual ~VertexArray();
    
    virtual const std::vector<Vertex>& points() {
        if(_transport)
            U_EXCEPTION("Vertices must be prepare()d before first use.");
        assert(_points);
        
        return *_points;
    }
    
    void set_thickness(float t);
    bool operator!=(const VertexArray& other) const;
    std::ostream &operator <<(std::ostream &os) override;
    void set_pos(const Vec2&) override;
    void set_size(const Size2&) override;
    void set_bounds(const Bounds&) override;
    //virtual bool is_same_type(Drawable* d) const { return dynamic_cast<Vertices*>(d) != nullptr; }
    
    bool swap_with(Drawable* d) override;
    
    friend class Section;
    friend class Entangled;
    
    //! Moves points from the _transport array to a copied _points array.
    virtual void prepare();
    
    void set_parent(SectionInterface* p) final override {
        if(p != parent()) {
            Drawable::set_parent(p);
            
            if(p && _transport)
                prepare();
        }
    }
    
protected:
    void update_size();
    virtual std::string vertex_sub_type() const = 0;
};
    
    class Vertices final : public VertexArray {
    public:
        Vertices(const std::vector<Vertex>& p, PrimitiveType primitive, MEMORY memory);
        Vertices(const Vec2& p0, const Vec2& p1, const Color& clr);
        
        ~Vertices() {}
        
    protected:
        std::string vertex_sub_type() const override { return "Vertices"; }
    };
    
    class Line final : public VertexArray {
    private:
        std::shared_ptr<std::vector<Vertex>> _processed_points;
        float _process_scale;
        GETTER(float, max_scale)
        
    public:
        Line(const Vec2& pos0, const Vec2& pos1, const Color& color, float t = 1, MEMORY memory = COPY)
            : Line({Vertex(pos0, color), Vertex(pos1, color)}, t, memory)
        {}
        
        Line(const std::vector<Vertex>& p, float t, MEMORY memory = TRANSPORT)
            : VertexArray(p, PrimitiveType::LineStrip, memory),
              _processed_points(NULL), _process_scale(-1),
              _max_scale(1)
        { if(_thickness != 1) set_thickness(t); }
        
        ~Line() {}
        
        const std::vector<Vertex>& points() override final;
        std::ostream &operator <<(std::ostream &os) override;
        decltype(_processed_points) processed_points() const { return _processed_points; }
        const decltype(_points)& raw_points() const { return _points; }
        
    protected:
        bool swap_with(Drawable* d) override;
        void update_bounds() override;
        void prepare() override;
        
        //virtual bool is_same_type(Drawable* d) const override { return dynamic_cast<Line*>(d) != nullptr; }
        
        //! Moves points from the _transport array to a copied _raw_points array.
        //void prepare() override;
        std::string vertex_sub_type() const override { return "Line"; }
    };
    
    class Polygon : public Drawable {
        GETTER_PTR(std::shared_ptr<std::vector<Vec2>>, vertices)
        GETTER_PTR(std::shared_ptr<std::vector<Vec2>>, relative)
        GETTER(Color, fill_clr)
        GETTER(Color, border_clr)
        bool _size_calculated;
        
    public:
        Polygon(std::shared_ptr<std::vector<Vec2>> vertices);
        Polygon(const std::vector<Vertex>& vertices);
        void set_fill_clr(const Color& clr);
        void set_border_clr(const Color& clr);
        void set_vertices(const std::vector<Vec2>& vertices);
    protected:
        bool swap_with(Drawable* d) override;
        void update_size();
        bool in_bounds(float x, float y) override;
    };
    
    class Rect final : public Drawable {
    private:
        GETTER(Color, lineclr)
        GETTER(Color, fillclr)
        
    public:
        Rect(const Bounds& size = Bounds(), const Color &fill = Black, const Color &line = Transparent)
            : gui::Drawable(Type::RECT, size), _lineclr(line), _fillclr(fill)
        {}
        virtual ~Rect() {}
        
        CHANGE_SETTER(lineclr)
        CHANGE_SETTER(fillclr)
        
        std::ostream &operator <<(std::ostream &os) override;
        
    protected:
        bool swap_with(Drawable* d) override {
            if(!Drawable::swap_with(d))
                return false;
            
            auto ptr = static_cast<const Rect*>(d);
            set_fillclr(ptr->_fillclr);
            set_lineclr(ptr->_lineclr);
            
            return true;
        }
    };
    
    class Circle final : public Drawable {
    private:
        GETTER(float, radius)
        GETTER(Color, line_clr)
        GETTER(Color, fill_clr)
        
    public:
        Circle(const Vec2& pos = Vec2(),
               float radius = 1,
               const Color& line_color = White,
               const Color& fill_color = Transparent)
            : gui::Drawable(Type::CIRCLE, Bounds(pos, Size2(radius*2)), Vec2(0.5)),
                _radius(radius),
                _line_clr(line_color),
                _fill_clr(fill_color)
        { }
        virtual ~Circle() {}
        
        void set_radius(float radius) {
            if(_radius != radius) {
                _radius = radius;
                set_size(Size2(_radius * 2));
            }
        }
        
        CHANGE_SETTER(line_clr)
        CHANGE_SETTER(fill_clr)
        
        bool in_bounds(float x, float y) override {
            auto size = global_bounds();
            return euclidean_distance(Vec2(x, y), size.pos() + size.size().mul(Vec2(1) - origin())) <= size.width * 0.5f;
        }
        
        std::ostream &operator <<(std::ostream &os) override;
        
    private:
        bool swap_with(Drawable* d) override {
            if(!Drawable::swap_with(d))
                return false;
            
            auto ptr = static_cast<const Circle*>(d);
            if(int(ptr->_radius) != int(_radius)) {
                _radius = ptr->_radius;
                set_size(Size2(_radius * 2));
            }
            set_line_clr(ptr->_line_clr);
            set_fill_clr(ptr->_fill_clr);
            
            return true;
        }
    };
    
    class Text final : public Drawable {
    private:
        GETTER(std::string, txt)
        GETTER(Color, color)
        GETTER(Font, font)
        Bounds _text_bounds;
        bool _bounds_calculated;
        
    public:
        Text(const std::string& txt = "", const Vec2& pos = Vec2(), const Color& color = Black, const Font& font = Font(), const Vec2& scale = Vec2(1), const Vec2& origin = Vec2(FLT_MAX));
        virtual ~Text() {}
        
        void set_txt(const std::string& txt) {
            if(txt == _txt)
                return;
            
            _txt = txt;
            _bounds_calculated = false;
			set_bounds_changed();
        }
        
        CHANGE_SETTER(color)
        
        void set_font(const Font& font) {
            if(font == _font)
                return;
            
            if(font.align != _font.align) {
                if(font.align == Align::Center)
                    set_origin(Vec2(0.5, 0.5));
                else if(font.align == Align::Left)
                    set_origin(Vec2());
                else if(font.align == Align::Right)
                    set_origin(Vec2(1, 0));
            }
            
            _font = font;
            _bounds_calculated = false;
            set_bounds_changed();
        }
        
        std::ostream &operator <<(std::ostream &os) override;
        
        const Bounds& text_bounds() {
            refresh_dims();
            return _text_bounds;
        }
        
        Size2 size() override {
            //update_bounds();
            refresh_dims();
            return Drawable::size();
        }
        
        Drawable& operator=(const Drawable& other) override {
            Drawable::operator=(other);
            
            auto ptr = static_cast<const Text*>(&other);
            set_txt(ptr->_txt);
            set_color(ptr->_color);
            set_font(ptr->_font);
            
            return *this;
        }
        
    private:
        //! Hide set_size method, which doesn't apply to Texts.
        //  Their size is determined by the text itself and automatically updated.
        void set_size(const Size2& size) override {
            Drawable::set_size(size);
        }
        
        void update_bounds() override {
            refresh_dims();
            Drawable::update_bounds();
        }
        
        
        bool swap_with(Drawable* d) override {
            if(d->type() != Type::TEXT)
                return false;
            
            auto ptr = static_cast<const Text*>(d);
            
            set_pos(d->pos());
            set_origin(d->origin());
            set_scale(d->scale());
            
            set_txt(ptr->_txt);
            set_color(ptr->_color);
            set_font(ptr->_font);
            
            if(ptr->_bounds_calculated) {
                _bounds_calculated = true;
                _text_bounds = ptr->_text_bounds;
                set_size(Size2(_text_bounds.pos() + _text_bounds.size()));
            }
            
            return true;
        }
        
        void refresh_dims();
    };
    
    class ExternalImage : public Drawable {
    public:
        using Ptr = Image::UPtr;
    private:
        GETTER(std::string, url)
        Ptr _source;
        GETTER(Color, color)
        
    public:
        ExternalImage() : ExternalImage(Image::Make(), Vec2()) {}
        ExternalImage(Ptr&& source, const Vec2& pos = Vec2(), const Vec2& scale = Vec2(1,1), const Color& color = Transparent);
        ~ExternalImage() { }
        ExternalImage(const ExternalImage& e) = delete;
        
        CHANGE_SETTER(url)
        CHANGE_SETTER(color)
        
        virtual const Image* source() const { return _source.get(); }
        //virtual void set_source(const Image& source);
        virtual void set_source(Ptr&& source);
        //void set_source(const cv::Mat& source);
        void set_bounds_changed() override;
        std::ostream &operator <<(std::ostream &os) override;
        void update_with(const gpuMat&);
        void update_with(const Image&);
        
    private:
        bool swap_with(Drawable* d) override {
            if(!Drawable::swap_with(d))
                return false;
            
            auto ptr = static_cast<ExternalImage*>(d);
            
            // only swap images with the same dimensions
            if(source() && ptr->source() && ptr->source()->dims != source()->dims)
                return false;
            
            set_url(ptr->_url);
            set_color(ptr->_color);
            
            if(!(*ptr->_source == *_source)) {
                std::swap(ptr->_source, _source);
                clear_cache();set_dirty();
            }
            
            set_scale(ptr->_scale);
            
            return true;
        }
    };
}

#endif

