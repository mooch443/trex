#ifndef _DRAW_STRUCTURE_H
#define _DRAW_STRUCTURE_H

#include <gui/GuiTypes.h>
#include <gui/colors.h>
#include <gui/Section.h>
#include <gui/DrawableCollection.h>
#include <gui/types/MiscShapes.h>
#include <gui/types/Layout.h>

namespace gui {
    //! Initializes the GUI error log
    void init_errorlog();
    void deinit_errorlog();
    
    class StaticText;
    class Button;
    class Dialog : public DrawableCollection {
    public:
        enum Result {
            OKAY,
            ABORT,
            SECOND,
            THIRD,
            FOURTH
        };
        
    protected:
        std::atomic_bool _closed;
        GETTER(Result, result)
        
        Rect _title_bg;
        derived_ptr<StaticText> _text;
        Text _title;
        derived_ptr<Button> _okay;
        derived_ptr<Button> _abort, _second, _third, _fourth;
        derived_ptr<HorizontalLayout> _buttons;
        GETTER(derived_ptr<Entangled>, custom)
        GETTER_NCONST(VerticalLayout, layout)
        std::function<bool(Result)> _callback;
        
    public:
        ~Dialog();
        
        std::future<void> wait() const {
            auto task = std::async(std::launch::async, [this](){
                cmn::set_thread_name("Dialog::wait()");
                while(!is_closed())
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
            });
            
            return task;
        }
        
        bool is_closed() const {
            return _closed.load();
        }
        
        void set_parent(SectionInterface* parent) override;
        void set_closed();
        
    protected:
        friend class DrawStructure;
        Dialog(DrawStructure& d, const std::function<bool(Result)>& callback, const std::string &text, const std::string& title, const std::string& okay, const std::string& abort, const std::string& second, const std::string& third, const std::string& fourth);
        void update(DrawStructure& d) override;
        
    public:
        void set_custom_element(derived_ptr<Entangled>&& e);
    };
    
    class DrawStructure {
    public:
        struct SectionGuard {
            DrawStructure &_s;
            Section* _section;
            
            SectionGuard(DrawStructure& s, const std::string& name) : _s(s)
            {
                _section = s.begin_section(name);
            }
            
            SectionGuard(DrawStructure& s, const std::string& name, bool reuse) : _s(s)
            {
                _section = s.begin_section(name, reuse);
            }
            
            ~SectionGuard() {
                _s.finalize_section(_section);
            }
        };
        
    protected:
        GETTER_NCONST(Section, root)
        GETTER_PTR(Section*, active_section)
        GETTER_PTR(Drawable*, hovered_object)
        GETTER_PTR(Drawable*, selected_object)

        GETTER(uint16_t, width)
        GETTER(uint16_t, height)
        GETTER_I(Vec2, scale, {0})
        GETTER(std::atomic_bool, changed)
        GETTER_SETTER_I(Size2, dialog_window_size, {0})
        GETTER_I(Vec2, mouse_position, {0})

        std::set<Drawable*> _end_objects;
        std::mutex _end_object_mutex;
        
        std::deque<Section*> _sections;
        std::deque<Dialog*> _dialogs;
        
        GETTER_NCONST(std::recursive_mutex, lock)
        std::set<Codes> pressed_keys;
        
    public:
        DrawStructure(uint16_t width = 0, uint16_t height = 0, Vec2 scale = Vec2(1, 1))
            : _root(this, NULL, "root"),
              _active_section(NULL),
              _hovered_object(NULL),
              _selected_object(NULL),
              _width(width),
              _height(height),
              _scale(scale),
              _changed(true)
        {}
        ~DrawStructure();
        
    private:
        Dialog* _dialog(const std::function<bool(Dialog::Result)>& callback, const std::string& text, const std::string& title = "Dialog", const std::string& okay = "Okay", const std::string& abort = "", const std::string& second = "", const std::string& third = "", const std::string& fourth = "");
        
    public:
        template<typename F, typename R = typename std::invoke_result_t<F, Dialog::Result>>
        Dialog* dialog(F&& callback, const std::string &text, const std::string& title = "Dialog", const std::string& okay = "Okay", const std::string& abort = "", const std::string& second = "", const std::string& third = "", const std::string& fourth = "", std::enable_if_t<std::is_same<R, void>::value, void> *  = nullptr)
        {
            std::function<bool(Dialog::Result)> fn = [callback](Dialog::Result r) {
                callback(r);
                return true;
            };
            return _dialog(fn, text, title, okay, abort, second, third, fourth);
        }
        
        template<typename F, typename R = typename std::invoke_result_t<F, Dialog::Result>>
        Dialog* dialog(F&& callback, const std::string &text, const std::string& title = "Dialog", const std::string& okay = "Okay", const std::string& abort = "", const std::string& second = "", const std::string& third = "", const std::string& fourth = "", std::enable_if_t<std::is_same<R, bool>::value, void> *  = nullptr)
        {
            return _dialog(callback, text, title, okay, abort, second, third, fourth);
        }
        
        Dialog* dialog(const std::string& text, const std::string& title = "Dialog", const std::string& okay = "Okay", const std::string& abort = "", const std::string& second = "", const std::string& third = "", const std::string& fourth = "");
        
        void close_dialogs();
        
        inline void section(const std::string& name, const std::function<void(DrawStructure&, Section*)>& fn) {
            fn(*this, begin_section(name));
            pop_section();
        }
        
        void draw_log_messages(const Bounds& screen);
        void set_size(const Size2& size);
        
        std::vector<Drawable*> collect();
        bool is_key_pressed(Codes code) const;
        
        void print(const Base*);
        
        void wrap_object(Drawable& d);
        void register_end_object(Drawable& d);
        void unregister_end_object(Drawable& d);

        template<typename T>
        T* add_object(T *ptr, typename std::enable_if<std::is_base_of<Drawable, T>::value, bool>::type = false) {
            if(ptr->type() == Type::SECTION)
                U_EXCEPTION("Cannot add Section using add_object. Use wrap_object instead.");
            
            if(!_active_section)
                begin_section("root");
            
            return _active_section->add(ptr);
        }
        //Drawable* _add_object(gui::Drawable *ptr);
        
        ExternalImage* image(const Vec2& pos, ExternalImage::Ptr&& image, const Vec2& scale = Vec2(1.0f, 1.0f), const Color& color = Transparent);
        
        Text* text(const std::string& txt, const Vec2& pos, const Color& color = White);
        Text* text(const std::string& txt, const Vec2& pos, const Color& color, const Font& font, const Vec2& scale = Vec2(1));
        
        Circle* circle(const Vec2& pos, float radius, const Color& color = White, const Color& fill_color = Transparent);
        
        Line* line(const Vec2& pos0, const Vec2& pos1, float thickness, const Color& color = White);
        Line* line(const std::vector<Vec2>& points, float thickness, const Color& color = White);
        Line* line(const std::vector<Vertex>& points, float thickness);
        
        Rect* rect(const Vec2& pos, const Vec2& size, const Color& inside, const Color& outside = Transparent);
        Rect* rect(const Bounds& rect, const Color& inside = White, const Color& outside = Transparent, const Vec2& scale = Vec2(1.f, 1.f));
        
        Vertices* vertices(const std::vector<Vec2>& points, const Color& color, PrimitiveType type);
        Vertices* vertices(const std::vector<Vertex>& points, PrimitiveType type = LineStrip);
        Vertices* line(const Vec2& pos0, const Vec2& pos1, const Color& color = White);
        
        void set_scale(float s) { if(_scale == Vec2(s,s)) return; _scale = Vec2(s, s); all_changed(); }
        void set_scale(const Vec2& s) { if(_scale == s) return; _scale = s; all_changed(); }
        
        Drawable* find(const std::string& name);
        Drawable* find(float x, float y);
        
        Drawable* mouse_move(float x, float y);
        Drawable* mouse_down(bool left_button);
        Drawable* mouse_up(bool left_button);
        
        //! Handles events like mouse move/up/down and returns true if
        //  the given event was handled.
        bool event(Event e);
        
        void select(Drawable*);
        
        bool key_down(Codes code, bool shift);
        bool key_up(Codes code, bool shift);
        bool text_entered(char c);
        void scroll(const Vec2& delta);
        
        void update_dialogs();
        
        // may only be called from gui::Base derived classes
        template<typename T>
        void before_paint(T*, typename std::enable_if<std::is_base_of<Base, T>::value, T>::type* = NULL) {
            if(_active_section)
                finalize_section(_active_section);
        }
        
        void do_hover(Drawable *d) {
            Event e(HOVER);
            
            if(d) {
                auto &gb = d->global_bounds();
                _mouse_position = gb.pos();
                
                e.hover.x = gb.x;
                e.hover.y = gb.y;
                e.hover.hovered = true;
                
            } else {
                e.hover.x = e.hover.y = 0;
                e.hover.hovered = false;
            }
            
            do_hover(d, e);
        }
        
    protected:
        friend class Drawable;
        friend class Section;
        friend class Base;
        friend class Entangled;
        friend class SectionInterface;
        
        void update_hover();
        void all_changed();
        void clear_hover();
        
        bool remove_wrapped(gui::Drawable *d);
        void clear() { _root.clear(); }
        
        Section* begin_section(const std::string& name, bool reuse = false);
        void finalize_section(const std::string& name);
        void finalize_section(const Section* section);
        
        //! Called automatically if the given object is removed from
        //  the draw hierarchy and might still be selected/hovered/etc.
        void erase(Drawable *d);
        
        void set_active_section(Section* section) {
            /*if(!section)
                Debug("active_section = empty");
            else
                Debug("Setting active_section = '%S'", &section->HasName::name());*/
            _active_section = section;
        }
        
        void push_section(Section* section, bool reuse = false) {
            //if(section && section != &_root && section != _sections.front())
            _sections.push_front(section);
            set_active_section(section);
            section->begin(reuse);
        }
        
        void pop_section() {
            if(!_sections.empty()) {
                assert(_sections.front() == _active_section);

                if (_active_section == &_root) {
                    std::unique_lock guard(_end_object_mutex);
                    for (auto& o : _end_objects) {
                        wrap_object(*o);
                    }
                }

                _active_section->end();

                _sections.pop_front();
                
                if (!_sections.empty()) {
                    set_active_section(_sections.front());
                }
                else {
                    

                    set_active_section(NULL);
                }
            } else
                U_EXCEPTION("Popping empty stack.");
        }
        
        void do_hover(Drawable *d, Event e) {
            if(d != _hovered_object) {
                if(_hovered_object) {
                    e.hover.hovered = false;
                    _hovered_object->hover(e);
                }
                
                _hovered_object = d;
            }
            
            if(_hovered_object) {
                e.hover.hovered = true;
                _hovered_object->hover(e);
            }
        }
        
    public:
        void set_dirty(const Base* base);
    };
}

#endif
