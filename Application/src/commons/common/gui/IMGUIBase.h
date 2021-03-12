#pragma once

#include "CrossPlatform.h"
#include <gui/DrawBase.h>
#include <misc/Timer.h>
#include <thread>

#include "MetalImpl.h"
#if TREX_METAL_AVAILABLE
using default_impl_t = gui::MetalImpl;
#else
#include "GLImpl.h"
using default_impl_t = gui::GLImpl;
#endif

namespace gui {
    class IMGUIBase : public Base {
    protected:
        struct baseFunctor {
            virtual void operator()()=0;
            virtual ~baseFunctor() {}
        };
        
        template<typename T>
        class functor : public baseFunctor {
            T f;
        public:
            template<typename U>
            functor(U&& f)
                :    f(std::forward<U>(f))
            {}
            void operator()() override {
                f();
            }
        };
        
        GETTER_NCONST(std::shared_ptr<CrossPlatform>, platform)
        DrawStructure * _graph;
        CrossPlatform::custom_function_t _custom_loop;
        GETTER(Bounds, work_area)
        std::function<void(const gui::Event&)> _event_fn;
        size_t _objects_drawn, _skipped;
        std::unordered_map<Type::Class, size_t> _type_counts;
        Timer _last_debug_print;
        Size2 _last_framebuffer_size;
        float _dpi_scale;
        std::function<bool(const std::vector<file::Path>&)> _open_files_fn;
        
        struct DrawOrder {
            bool is_pop;
            size_t index;
            Drawable* ptr;
            gui::Transform transform;
            Bounds bounds;
            
            DrawOrder() {}
            DrawOrder(bool is_pop, size_t index, Drawable*ptr, const gui::Transform& transform, const Bounds& bounds)
            : is_pop(is_pop), index(index), ptr(ptr), transform(transform), bounds(bounds)
            {}
        };
        
        std::vector<DrawOrder> _draw_order;
        
        std::mutex _mutex;
        std::queue<std::unique_ptr<baseFunctor>> _exec_main_queue;
        
    public:
        template<typename impl_t = default_impl_t>
        IMGUIBase(std::string title, DrawStructure& base, CrossPlatform::custom_function_t custom_loop, std::function<void(const gui::Event&)> event_fn) : _custom_loop(custom_loop), _event_fn(event_fn)
        {
            set_graph(base);
            
            auto ptr = new impl_t([this](){
                if(_graph == NULL)
                    return;
                
                std::lock_guard<std::recursive_mutex> lock(_graph->lock());
                this->paint(*_graph);
                
                auto cache = _graph->root().cached(this);
                if(cache)
                    cache->set_changed(false);
                
                //_after_display();
            }, [this]() -> bool {
                std::lock_guard<std::recursive_mutex> lock(_graph->lock());
                _graph->before_paint(this);
                
                auto cache = _graph->root().cached(this);
                if(!cache) {
                    cache = std::make_shared<CacheObject>();
                    _graph->root().insert_cache(this, cache);
                }
                
                return cache->changed();
            });
            
            _platform = std::shared_ptr<impl_t>(ptr);
            init(title);
        }
        
        void set_graph(DrawStructure& base) {
            _graph = &base;
            
        }
        void init(const std::string& title);
        ~IMGUIBase();
        
        void set_open_files_fn(std::function<bool(const std::vector<file::Path>&)> fn) {
            _open_files_fn = fn;
        }
        
        void set_background_color(const Color&) override;
        void set_frame_recording(bool v) override;
        Image::Ptr current_frame_buffer() override;
        void loop();
        LoopStatus update_loop() override;
        virtual void paint(DrawStructure& s) override;
        void set_title(std::string) override;
        Bounds text_bounds(const std::string& text, Drawable*, const Font& font) override;
        uint32_t line_spacing(const Font& font) override;
        Size2 window_dimensions() override;
        float dpi_scale() override;
        template<typename F>
        void exec_main_queue(F&& fn) {
            std::lock_guard<std::mutex> guard(_mutex);
            _exec_main_queue.push(std::unique_ptr<baseFunctor>(new functor<F>(std::move(fn))));
            //_exec_main_queue.push(std::bind([](F& fn){ fn(); }, std::move(fn)));
        }
        Event toggle_fullscreen(DrawStructure& g) override;
        
    private:
        void redraw(Drawable* o, std::vector<DrawOrder>& draw_order, bool is_background = false);
        void draw_element(const DrawOrder& order);
        void event(const gui::Event& e);
    };
}
