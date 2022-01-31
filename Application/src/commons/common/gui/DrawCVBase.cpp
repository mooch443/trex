#include "DrawCVBase.h"
#include <misc/checked_casts.h>

namespace gui {
    IMPLEMENT(CVBase::_static_pixels);
    
    CVBase::CVBase(cv::Mat& window, const cv::Mat& background)
        : _window(window)
    {
        if (background.empty()) {
            _overlay = cv::Mat::zeros(window.rows, window.cols, CV_8UC4);
        } else {
            assert(background.type() == CV_8UC4);
            background.copyTo(_overlay);
        }
        
        assert(_window.type() == CV_8UC4);
    }
    
    void CVBase::draw_image(gui::ExternalImage* ptr) {
        auto mat = ptr->source()->get();
        assert(mat.type() == CV_8UC4);
        
        if (ptr->color().a > 0) {
            auto &color = ptr->color();
            
            for (int i = 0; i < mat.rows; ++i)
            {
                uchar* pixel = mat.ptr<uchar>(i);
                for (int j = 0; j < mat.cols; ++j)
                {
                    uchar &b = *pixel++;
                    uchar &g = *pixel++;
                    uchar &r = *pixel++;
                    uchar &a = *pixel++;
                    
                    b = (uchar)saturate(float(b) * (color.b / 255.f));
                    g = (uchar)saturate(float(g) * (color.g / 255.f));
                    r = (uchar)saturate(float(r) * (color.r / 255.f));
                    a = (uchar)saturate(float(a) * (color.a / 255.f));
                }
            }
        }
        
        if (ptr->scale().x != 1.0f && ptr->scale().x > 0)
            resize_image(mat, ptr->scale().x);
        
        std::vector<cv::Mat> split;
        cv::split(mat, split);
        
        auto pos = ptr->pos();
        if(pos.x + mat.cols <= _window.cols && pos.y + mat.rows <= _window.rows)
            mat.copyTo(_window(Bounds(pos.x, pos.y, mat.cols, mat.rows)), split[3]);
        
        else if(   pos.x < _window.cols
                && pos.x + _window.cols >= 0
                && pos.y < _window.rows
                && pos.y + _window.rows >= 0)
        {
            
            float w = floor(min(mat.cols + pos.x, min(_window.cols - pos.x, (float)mat.cols)));
            float h = floor(min(mat.rows + pos.y, min(_window.rows - pos.y, (float)mat.rows)));
            
            float mx = pos.x < 0 ? -pos.x : 0;
            float my = pos.y < 0 ? -pos.x : 0;
            
            cv::Mat small = mat(Bounds(mx, my, w, h));
            cv::Mat big = _window(Bounds(pos.x >= 0 ? pos.x : 0, pos.y >= 0 ? pos.y : 0, w, h));
            
            small.copyTo(big, split[3](Bounds(mx, my, w, h)));
            
        } else {
            Debug("Didnt draw %f,%f %dx%d. (%dx%d) in window %dx%d", pos.x, pos.y, mat.cols, mat.rows, mat.cols, mat.rows, _window.cols, _window.rows);
        }
    }
    
    void CVBase::display() {
        tf::imshow("cvbase", _window);
    }
    
    void CVBase::paint(gui::DrawStructure &s) {
        std::unique_lock<std::recursive_mutex> lock(s.lock());
        s.before_paint(this);
        
        auto objects = s.collect();
        
        for (auto o : objects) {
            draw(s, o);
        }
    }
    
    void CVBase::draw(DrawStructure& s, Drawable*o) {
        switch (o->type()) {
            case Type::IMAGE:
                draw_image(static_cast<ExternalImage*>(o));
                break;
                
            case Type::RECT: {
                auto ptr = static_cast<Rect*>(o);
                auto &rect = ptr->bounds();
                
                if (ptr->fillclr().a > 0) {
                    cv::rectangle(_window, (cv::Rect2f)rect, cv::Scalar(ptr->fillclr().r, ptr->fillclr().g, ptr->fillclr().b, ptr->fillclr().a), cv::FILLED);
                }
                
                if(ptr->lineclr().a > 0) {
                    cv::rectangle(_window, (cv::Rect2f)rect, cv::Scalar(ptr->lineclr().r, ptr->lineclr().g, ptr->lineclr().b, ptr->lineclr().a), 1);
                }
                
                break;
            }
                
            case Type::VERTICES: {
                auto ptr = static_cast<Vertices*>(o);
                int t = 1;
                if(dynamic_cast<Line*>(o))
                    t = max(1, min(static_cast<Line*>(o)->thickness(), CV_MAX_THICKNESS));
                
                if(ptr->primitive() != LineStrip && ptr->primitive() != Lines)
                    U_EXCEPTION("Does not support other primitive types yet.");
                
                if(ptr->primitive() == LineStrip) {
                    Vec2 prev;
                    for(size_t i=0; i<ptr->points().size(); i++) {
                        auto &p = ptr->points().at(i);
                        auto &c = p.color();
                        
                        if(i)
#if CV_MAJOR_VERSION >= 3
                            DEBUG_CV(cv::line(_window, prev, (cv::Point2f)p.position(), cv::Scalar(c.b, c.g, c.r, c.a), t, cv::LINE_AA));
#else
                        DEBUG_CV(cv::line(_window, prev, (cv::Point2f)p.position(), cv::Scalar(c.b, c.g, c.r, c.a), t));
#endif
                        prev = p.position();
                    }
                    
                } else {
                    for(size_t i=0; i<ptr->points().size(); i++) {
                        auto &p = ptr->points().at(i);
                        auto &c = p.color();
                        
#if CV_MAJOR_VERSION >= 3
                        DEBUG_CV(cv::line(_window, (cv::Point2f)ptr->points().at(i < ptr->points().size()-1 ? i+1 : 0), (cv::Point2f)p.position(), cv::Scalar(c.b, c.g, c.r, c.a), t, cv::LINE_AA));
#else
                        DEBUG_CV(cv::line(_window, (cv::Point2f)ptr->points().at(i < ptr->points().size()-1 ? i+1 : 0), (cv::Point2f)p.position(), cv::Scalar(c.b, c.g, c.r, c.a), t));
#endif
                    }
                }
                
                break;
            }
                
            case Type::TEXT: {
                auto ptr = static_cast<Text*>(o);
                auto &color = ptr->color();
                
                cv::putText(_window, ptr->txt(), (cv::Point2f)Vec2(ptr->pos()), cv::FONT_HERSHEY_PLAIN, ptr->font().size, cv::Scalar(color.b, color.g, color.r, color.a), 1
#if CV_MAJOR_VERSION >= 3
                            , cv::LINE_AA
#endif
                            );
                break;
            }
                
            case Type::CIRCLE: {
                auto ptr = static_cast<Circle*>(o);
                auto &color = ptr->line_clr();
                cv::circle(_window, (cv::Point2f)Vec2(ptr->pos()), narrow_cast<int>(ptr->radius()), cv::Scalar(color.b, color.g, color.r, color.a), 1
#if CV_MAJOR_VERSION >= 3
                           , cv::LINE_AA
#endif
                           );
                
                
                break;
            }
                
            case Type::ENTANGLED: {
                auto ptr = static_cast<Entangled*>(o);
                for(auto c : ptr->children())
                    draw(s, c);
                
                break;
            }
                
            default: {
                auto type = o->type().name();
                U_EXCEPTION("Unknown type '%s' in CVBase.", type)
            }
        }
    }
}
