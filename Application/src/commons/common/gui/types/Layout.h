#pragma once

#include <gui/types/Entangled.h>

namespace gui {
    template<class Base>
    class derived_ptr {
    public:
        std::shared_ptr<Base> ptr;
        Base* raw_ptr;

        derived_ptr(std::shared_ptr<Base> share = nullptr) : ptr(share), raw_ptr(nullptr) {}
        derived_ptr(Base* raw) : ptr(nullptr), raw_ptr(raw) {}
        
        Base& operator*() const { return ptr ? *ptr : *raw_ptr; }
        Base* get() const { return ptr ? ptr.get() : raw_ptr; }
        template<typename T> T* to() const { auto ptr = dynamic_cast<T*>(get()); if(!ptr) U_EXCEPTION("Cannot cast object to specified type."); return ptr; }
        
        bool operator==(Base* raw) const { return get() == raw; }
        //bool operator==(decltype(ptr) other) const { return ptr == other; }
        bool operator==(derived_ptr<Base> other) const { return get() == other.get(); }
        bool operator<(Base* raw) const { return get() < raw; }
        bool operator<(derived_ptr<Base> other) const { return get() < other.get(); }
        
        operator bool() const { return get() != nullptr; }
        Base* operator ->() const { return get(); }
        
        template<typename T>
        operator derived_ptr<T> () {
            if(ptr)
                return derived_ptr<T>(std::static_pointer_cast<T>(ptr));
            return derived_ptr<T>(static_cast<T*>(raw_ptr));
        }
    };
    
    class Layout : public Entangled {
    public:
        typedef derived_ptr<Drawable> Ptr;
        
    private:
        std::vector<Ptr> _objects;
        
    public:
        template<typename T, typename... Args>
        static Layout::Ptr Make(Args&&... args) {
            return Layout::Ptr(std::make_shared<T>(std::forward<Args>(args)...));
        }
        
    public:
        Layout(const std::vector<Layout::Ptr>&);
        virtual ~Layout() { clear_children(); }
        
        void update() override;
        void add_child(size_t pos, Layout::Ptr ptr);
        void add_child(Layout::Ptr ptr);
        
        void remove_child(Layout::Ptr ptr);
        void remove_child(Drawable* ptr) override;
        void set_children(const std::vector<Layout::Ptr>&);
        void clear_children() override;
        
        
        virtual void update_layout() {}
        virtual void auto_size(Margin margins) override;
    };
    
    class HorizontalLayout : public Layout {
    public:
        enum Policy {
            CENTER, TOP, BOTTOM
        };
        
    protected:
        GETTER(Bounds, margins)
        GETTER(Policy, policy)
        
    public:
        HorizontalLayout(const std::vector<Layout::Ptr>& objects = {},
                         const Vec2& position = Vec2(),
                         const Bounds& margins = {5, 5, 5, 5});
        
        void set_policy(Policy);
        void set_margins(const Bounds&);
        virtual std::string name() const override { return "HorizontalLayout"; }
        
        void update_layout() override;
    };
    
    class VerticalLayout : public Layout {
    public:
        enum Policy {
            CENTER, LEFT, RIGHT
        };
        
    protected:
        GETTER(Bounds, margins)
        GETTER(Policy, policy)
        
    public:
        VerticalLayout(const std::vector<Layout::Ptr>& objects = {},
                         const Vec2& position = Vec2(),
                         const Bounds& margins = {5, 5, 5, 5});
        
        void set_policy(Policy);
        void set_margins(const Bounds&);
        virtual std::string name() const override { return "VerticalLayout"; }
        
        void update_layout() override;
    };
}
