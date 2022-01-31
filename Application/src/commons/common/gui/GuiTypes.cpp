#include "GuiTypes.h"
#include <misc/Base64.h>
#include "DrawBase.h"
#include <gui/DrawSFBase.h>
#include <JSON.h>
#include <misc/CircularGraph.h>

namespace gui {
VertexArray::VertexArray(const std::vector<Vertex>& p, PrimitiveType primitive, MEMORY memory)
    : Drawable(Type::VERTICES), _transport(&p), _points(NULL), _primitive(primitive),_size_calculated(false), _thickness(1)
{
    if(memory == COPY)
        prepare();
}

VertexArray::~VertexArray() {
    
}

void VertexArray::prepare() {
    if(!_transport)
        return;
    
    _original_points.resize(0);
    _original_points.insert(_original_points.end(), _transport->begin(), _transport->end());
    
    _points = nullptr;
    _transport = NULL;
    
    update_size();
}

void VertexArray::update_size() {
    if(_points && !_points->empty())
        return;
    
    if(_points) {
        _points->resize(0);
        _points->insert(_points->end(), _original_points.begin(), _original_points.end());
    } else
        _points = std::make_shared<std::vector<Vertex>>(_original_points);
    
    Vec2 maximum(-std::numeric_limits<Float2_t>::max()), minimum(std::numeric_limits<Float2_t>::max());
    float x, y;
    
    for(auto &p : *_points) {
        x = p.position().x;
        y = p.position().y;
            
        if(x < minimum.x) minimum.x = x;
        if(y < minimum.y) minimum.y = y;
        if(maximum.x < x) maximum.x = x;
        if(maximum.y < y) maximum.y = y;
    }
    
    maximum.x -= minimum.x;
    maximum.y -= minimum.y;
            
    for(auto &p : *_points)
        p.position() -= minimum;
    
    set_bounds(Bounds(minimum, maximum));
    _size_calculated = true;
}
            
void VertexArray::set_pos(const Vec2 &pos) {
    Drawable::set_pos(pos);
}
void VertexArray::set_size(const Size2 &size) {
    Drawable::set_size(size);
}
void VertexArray::set_bounds(const Bounds &bounds) {
    Drawable::set_bounds(bounds);
}

bool VertexArray::operator!=(const VertexArray& other) const {
    const auto rp0 = _transport ? _transport : &_original_points;
    const auto rp1 = other._transport ? other._transport : &other._original_points;
    
    const auto &lhs = *rp0;
    const auto &rhs = *rp1;
    
    return _thickness != other._thickness || other.primitive() != primitive() || (!rp0 && rp1) || (rp0 && !rp1) || (rp0 && lhs != rhs);
}

bool VertexArray::swap_with(gui::Drawable *d) {
    if(d->type() != type())
        return false;
    
    if(d == this)
        U_EXCEPTION("Cannot swap object with itself.");
    
    auto ptr = dynamic_cast<VertexArray*>(d);
    if(!ptr)
        return false;
    
    if(ptr->vertex_sub_type() != vertex_sub_type())
        return false;
        
    if(_transport)
        U_EXCEPTION("Cannot use Line in this way. Transport is supposed to be temporary.");
    
    /*set_pos(d->pos());*/
    set_origin(d->origin());
    set_rotation(d->rotation());
    set_clickable(d->clickable());
    
    // this would reset the bounds, which is not wanted behavior
    //if(!Drawable::swap_with(d))
    //    return false;
    
    
    
    if(*this != *ptr) {
        set_thickness(ptr->thickness());
        _primitive = ptr->_primitive;
        
        if(ptr->_transport) {
            if(_original_points != *ptr->_transport) {
                _original_points.resize(0);
                _original_points.insert(_original_points.end(), ptr->_transport->begin(), ptr->_transport->end());
                if(_points)
                    _points->resize(0);
            }
            
        } else {
            if(_original_points != ptr->_original_points)
            {
                std::swap(ptr->_original_points, _original_points);
                if(_points)
                    _points->resize(0);
            }
        }
        //if(_points)
        //_points->resize(0);
        //_size_calculated = false;
        update_size();
        set_dirty();
    } //else
       // Debug("Swap with itself?");
    
    return true;
}

void VertexArray::set_thickness(float t) {
    if(t != _thickness) {
        _thickness = t;
        set_dirty();
    }
}

Vertices::Vertices(const std::vector<Vertex>& p, PrimitiveType primitive, MEMORY memory)
    : VertexArray(p, primitive, memory)
{ }


Vertices::Vertices(const Vec2& p0, const Vec2& p1, const Color& clr)
    : Vertices({
        Vertex(p0, clr),
        Vertex(p1, clr)
    }, LineStrip, MEMORY::COPY)
{ }

Polygon::Polygon(std::shared_ptr<std::vector<Vec2>> vertices)
    : Drawable(Type::POLYGON), _vertices(vertices), _fill_clr(Transparent), _border_clr(Transparent), _size_calculated(false)
{
    update_size();
}

Polygon::Polygon(const std::vector<Vertex>& vertices)
    : Drawable(Type::POLYGON), _fill_clr(Transparent), _border_clr(Transparent), _size_calculated(false)
{
    _vertices = std::make_shared<std::vector<Vec2>>(vertices.begin(), vertices.end());
    if(!vertices.empty()) {
        set_border_clr(vertices.front().color());
        set_fill_clr(vertices.front().color().alpha(100));
    }
    update_size();
}

bool Polygon::in_bounds(float x, float y) {
    if(Drawable::in_bounds(x, y)) {
        std::vector<Vec2> points;
        points.reserve(_relative->size());
        auto transform = global_transform();
        for(auto &pt : *_relative) {
            points.push_back(transform.transformPoint(pt));
        }
        return pnpoly(points, Vec2(x, y));
    }
    return false;
}

//void Polygon::update_size() {
//    Drawable::update_bounds();
//}

void Polygon::set_fill_clr(const Color& clr) {
    if(clr != _fill_clr) {
        _fill_clr = clr;
        set_dirty();
    }
}

void Polygon::set_border_clr(const Color& clr) {
    if(clr != _border_clr) {
        _border_clr = clr;
        set_dirty();
    }
}

void Polygon::set_vertices(const std::vector<Vec2> &vertices) {
    if(!_vertices)
        _vertices = std::make_shared<std::vector<Vec2>>();
    if(vertices != *_vertices) {
        *_vertices = vertices;
        _size_calculated = false;
        update_size();
        set_dirty();
    }
}

void Polygon::update_size() {
    if(_size_calculated)
        return;
    
    if(!_vertices || _vertices->empty()) {
        set_bounds(Bounds());
        _relative = nullptr;
        return;
    }
    
    //auto ptr = poly_convex_hull(_vertices.get());
    //if(ptr)
    //    _vertices = ptr;
    //else
    //    _vertices->clear();
    
    Bounds b(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX);
    if(!_relative) _relative = std::make_shared<std::vector<Vec2>>();
    _relative->resize(_vertices->size());
    
    for(auto &p : *_vertices) {
        if(b.x > p.x) b.x = p.x;
        if(b.y > p.y) b.y = p.y;
        
        if(b.width < p.x) b.width = p.x;
        if(b.height < p.y) b.height = p.y;
    }
    
    b << Size2(b.size() - b.pos());
    
    for(size_t i=0; i<_vertices->size(); ++i)
        (*_relative)[i] = (*_vertices)[i] - b.pos();
    
    set_bounds(b);
    _size_calculated = true;
}

bool Polygon::swap_with(Drawable* d) {
    if(d->type() != Type::POLYGON)
        return false;
    
    Polygon* p = static_cast<Polygon*>(d);
    
    set_origin(d->origin());
    set_rotation(d->rotation());
    set_clickable(d->clickable());
    set_border_clr(p->border_clr());
    set_fill_clr(p->fill_clr());
    
    if((p->_vertices && !_vertices)
       || (!p->_vertices && _vertices)
       || *p->_vertices != *_vertices)
    {
        set_dirty();
        
        std::swap(p->_vertices, _vertices);
        _size_calculated = false;
        update_size();
    }
    
    return true;
}

std::ostream & Line::operator <<(std::ostream &os) {
    if(!_points || _points->empty()) {
        os << "\"S\",[]," << Color() << "," << thickness()*0.75;
        return os;
    }
    
    os << "\"S\",";
    os << "[";
    for (size_t i=0; i<_points->size(); i++) {
        auto &p = (*_points)[i];
        if(i)
            os << ",";
        os << p.position();
        //os << int(p.position().x) << "," << int(p.position().y);
    }
    
    auto &p = _points->front();
    os << "], " << p.color() << "," << thickness()*0.75;
    return os;
}

bool Line::swap_with(gui::Drawable *d) {
    //return false;
    
    auto ptr = dynamic_cast<Line*>(d);
    if(!ptr)
        return false; // wrong object type
    
    //_max_scale = ptr->_max_scale;
    bool diff = *ptr != *this;
    //if(diff)
    //    return false;
    
    if(!VertexArray::swap_with(d))
        return false;
    
    //if(*ptr != *this) {
        if(diff) {
            std::swap(_processed_points, ptr->_processed_points);
            std::swap(_process_scale, ptr->_process_scale);
            std::swap(_max_scale, ptr->_max_scale);
            
            points();
        }
        
        /*if(ptr->_processed_points) {
            std::swap(_processed_points, ptr->_processed_points);
            
        } else if(!_points->empty()) {
            if(_processed_points)
                _processed_points = nullptr;
            points();
        }*/
    //}
    
    
    
    
    return true;
}

void Line::update_bounds() {
    if(!bounds_changed())
        return;
    
    VertexArray::update_bounds();
    _max_scale = global_text_scale().max();
}

void reduce_vertex_line(const std::vector<Vertex>& points, std::vector<Vertex>& array, float threshold)
{
    array.clear();
    
    if(points.empty())
        return;
    
    Vertex last_added_point(points.front());
    array.push_back(last_added_point);
    
    if(points.size() == 1)
        return;
    
    Vec2 previous_vec((last_added_point.position() - points.back().position()).normalize());
    
    Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
    for(auto &v : points) {
        auto &pt = v.position();
        if(pt.x < bounds.x) bounds.x = pt.x;
        if(pt.y < bounds.y) bounds.y = pt.y;
        if(pt.x > bounds.width) bounds.width = pt.x;
        if(pt.y > bounds.height) bounds.height = pt.y;
    }
    
    bounds << Size2(bounds.size() - bounds.pos());
    float dim = bounds.size().length();
    float cumlen = 0;
    Vec2 prev_vertex = last_added_point;
    
    for (size_t i=1; i<points.size()-1; i++) {
        Vertex p1(points.at(i));
        
        auto line = p1.position() - last_added_point.position();
        auto color_diff = p1.color().diff(last_added_point.color()) / 4.f / 255.f;
        
        auto len = line.length();
        line = line / len;
        
        cumlen += (p1.position() - prev_vertex).length();
        prev_vertex = p1.position();
        
        // returns [-1,1] values from 180°-0° (upper half of the unit circle)
        // 90°-180° is < 0, while 0°-90° is >= 0
        // move it to [-2,0], take absolute and multiply by 0.5, so we get 0-180° -> [0,1]
        float a = cmn::abs(previous_vec.dot(line) - 1) * 0.5f;
        //Debug("%f (%f,%f) -> (%f,%f) : %f | %f, len:%f", previous_vec.dot(line), previous_vec.x, previous_vec.y, line.x, line.y, a, threshold, len);
        if (cumlen < dim * threshold * 0.005 && color_diff < threshold * 0.01 && a < threshold * 0.1)
            continue;
        
        cumlen = 0;
        
        //change[3] = min_alpha;
        last_added_point = p1;
        previous_vec = line;
        array.push_back(last_added_point);
    }
    
    array.push_back(points.back());
    
    //Debug("Reduced from %d -> %d (%f, %f) -> %f real", points.size(), array.size(), threshold, dim, dim * threshold);
}

void Line::prepare() {
    if(!_transport)
        return;
    
    if(_processed_points)
        _processed_points = nullptr;
    
    VertexArray::prepare();
}

const std::vector<Vertex>& Line::points() {
    if(_transport)
        U_EXCEPTION("Line must be prepare()d before first use.");
    assert(_points);
    
    auto _s = 1 / _max_scale;
    if(!_processed_points || _process_scale != _s) {
        if(!_processed_points)
            _processed_points = std::make_shared<std::vector<Vertex>>();
        _process_scale = _s;
        
        reduce_vertex_line(*_points, *_processed_points, _s);
    }
    
    return *_processed_points;
}

std::ostream & Rect::operator <<(std::ostream &os) {
    Drawable::operator<<(os);
    
    os <<
    "" << fillclr() << ","
    "" << lineclr() << "";
    return os;
}

std::ostream & Circle::operator <<(std::ostream &os) {
    Drawable::operator<<(os);
    
    os << line_clr();
    if(fill_clr().a > 0)
        os << "," << fill_clr();
    return os;
}

std::ostream & Text::operator <<(std::ostream &os) {
    //Drawable::operator<<(os);
    
    using namespace nlohmann;
    auto t = txt();
    //t = utils::find_replace(t, "/", "");
    
    json j = t;
    
    os <<
    j.dump() << ","
    ""<<font().size<<","
    "" << color() << "";
    if(/*font().align != Align::Left || */_font.style) {
        os << ",\"";
        
        //if(font().align == Align::Center)
        //    os << "c";
        //else if(font().align == Align::Right)
        //    os << "r";
        if(_font.style)
            os << ((_font.style & Style::Bold) ? "b" : ((_font.style & Style::Italic) ? "i" : ""));
        os << "\"";
    }
    return os;
}

Text::Text(const std::string& txt, const Vec2& pos, const Color& color, const Font& font, const Vec2& scale, const Vec2& origin)
: gui::Drawable(Type::TEXT, Bounds(pos),
                font.align == Align::Center
                    ? Vec2(0.5, 0.5)
                    : (font.align == Align::Right
                       ? Vec2(1.0, 0.0)
                       : (font.align == Align::VerticalCenter ? Vec2(0, 0.5) : Vec2()))
                ),
      _txt(txt), _color(color), _font(font), _bounds_calculated(false)
{
    set_scale(scale);
    if(origin.x != FLT_MAX)
        set_origin(origin);
    //refresh_dims(); //! will be done when parent is set
}

void Text::refresh_dims() {
    if(_bounds_calculated)
        return;
    
    if(!parent() || !parent()->stage()) {
        set_bounds_changed();
    } else {
        _bounds_calculated = true;
    }
    
    if(_txt.empty()) {
        _text_bounds = Bounds(Vec2(), Size2(0, Base::default_line_spacing(font())));
    }
    else {
        _text_bounds = Base::default_text_bounds(txt(), this, font());
        //set_size(_text_bounds.pos() /** 1.25f*/ + _text_bounds.size());
    }
    set_size(_text_bounds.size());
}

ExternalImage::ExternalImage(Ptr&& source, const Vec2& pos, const Vec2& scale, const Color& color)
    : gui::Drawable(Type::IMAGE, Bounds(pos, Vec2(source ? source->cols : 0, source ? source->rows : 0))), _url(""), _source(nullptr), _color(color)
{
    if(scale.x != 1 || scale.y != 1)
        set_scale(scale);
    set_source(std::move(source));
}

void ExternalImage::update_with(const gpuMat& mat) {
    if(mat.channels() > 0 && mat.channels() != 4 && mat.channels() != 1 && mat.channels() != 2)
        U_EXCEPTION("Only support greyscale, RG, or RGBA images.");
    
    if(!_source) {
        _source = Image::Make(mat.rows, mat.cols, mat.dims);
        set_size(Vec2(_source->cols, _source->rows));
    } else if(   (int)_source->cols != mat.cols
              || (int)_source->rows != mat.rows
              || (int)_source->dims != mat.channels())
    {
        _source->create((uint)mat.rows, (uint)mat.cols, (uint)mat.dims);
        set_size(Vec2(_source->cols, _source->rows));
    }
    
    cv::Mat local;
    mat.copyTo(local);
    _source->set(_source->index(), local);
    
    for(auto& c : _cache) {
        c.second->set_changed(true);
    }
    
    set_dirty();
}

void ExternalImage::update_with(const Image& mat) {
    if(mat.dims > 0 && mat.dims != 4 && mat.dims != 1 && mat.dims != 2)
        U_EXCEPTION("Only support greyscale, RG, or RGBA images.");
    
    if(!_source) {
        _source = Image::Make(mat.rows, mat.cols, mat.dims);
        set_size(Vec2(_source->cols, _source->rows));
        
    } else if(   _source->cols != mat.cols
              || _source->rows != mat.rows
              || _source->dims != mat.dims)
    {
        _source->create(mat.rows, mat.cols, mat.dims);
        set_size(Vec2(_source->cols, _source->rows));
    }
    
    _source->set(mat.index(), mat.data());
    
    for(auto& c : _cache) {
        c.second->set_changed(true);
    }
    
    set_dirty();
}

void ExternalImage::set_source(Ptr&& source) {
    if(_source == source)
        return;
    
    if(source && source->dims > 0 && source->dims != 4 && source->dims != 1 && source->dims != 2)
        U_EXCEPTION("Only support greyscale, RG, or RGBA images.");
    
    //if(_source && source && _source->array_size() >= source->array_size()) {
    if(_source && source)
        _source->set(std::move(*source));
    else
        _source = std::move(source);
    //} else
    //    _source = std::move(source);
    
    if(_source)
        set_size(_source->bounds().size());
    else
        set_size(Size2());
    
    for(auto& c : _cache) {
        c.second->set_changed(true);
    }
    
    set_dirty();
}

/*void ExternalImage::set_source(const Image& source) {
    if(_source == &source)
        return;
    
    if(source.dims > 0 && source.dims != 4 && source.dims != 1 && source.dims != 2)
        U_EXCEPTION("Only support greyscale, RG, or RGBA images.");
            
    if(!_source) {
        _source = new Image(source.rows, source.cols, source.dims);
        set_size(Vec2(_source->cols, _source->rows));
        
    } else if(   _source->cols != source.cols
              || _source->rows != source.rows
              || _source->dims != source.dims)
    {
        _source->create(source.rows, source.cols, source.dims);
        set_size(Vec2(_source->cols, _source->rows));
    }
    
    if(source.dims == 1 || source.dims == 2) {
        //_source->set_channel(0, source.data());
        //_source->set_channel(1, source.data());
        //_source->set_channel(2, source.data());
        //_source->set_channel(3, [&source](size_t i){ return source.data()[i] ? 255 : 0; });
        _source->set(source.index(), source.data());
        
    } else if(source.dims == 4 || source.dims == 0) {
        if(source.dims)
            _source->set(0, source.data());
    } else
        U_EXCEPTION("Only support greyscale, RG, or RGBA images.");
    
    for(auto& c : _cache) {
        c.second->set_changed(true);
    }
    
    set_dirty();
}*/

void ExternalImage::set_bounds_changed() {
    if(_bounds_changed)
        return;
    
    if(_parent)
        _parent->set_dirty();
    _bounds_changed = true;
    
    if(parent() && parent()->type() == Type::ENTANGLED) {
        static_cast<Entangled*>(parent())->set_content_changed(true);
    }
}

std::ostream & ExternalImage::operator <<(std::ostream &os) {
    //Drawable::operator<<(os);
    
    if(_source->empty()) {
        os << "\"\"";
        return os;
    }
    
    std::vector<uchar> conv;
    to_png(*_source, conv);
    
    auto str = base64_encode(conv.data(), conv.size());
    os << "\"" << str << "\"";
    if(color().a > 0)
        os << "," << color();
    
    return os;
}
std::ostream & VertexArray::operator <<(std::ostream &os) {
    char primitive_type = 'S';
    switch (primitive()) {
        case LineStrip: break;
        case Triangles: primitive_type = 'T'; break;
        case Lines:     primitive_type = 'L'; break;
            
        default:
            Warning("Serialization: Unknown primitive type.");
            os << "0," << "\"L\"," << "[]," << Color();
            return os;
    }
    
    auto &points = this->points();
    
    if(points.empty()) {
        os << "\"L\",[]," << Color();
        return os;
    }
    
    os << "\"" << primitive_type << "\",";
    os << "[";
    for (size_t i=0; i<_points->size(); i++) {
        auto &p = (*_points)[i];
        if(i)
            os << ",";
        os << p.position();
        //os << int(p.position().x) << "," << int(p.position().y);
    }
    
    auto &p = _points->front();
    os << "], " << p.color();
    return os;
}
}
