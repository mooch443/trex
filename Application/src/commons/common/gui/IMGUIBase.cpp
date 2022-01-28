#include "IMGUIBase.h"
#include <gui/DrawStructure.h>

#include <algorithm>
#include <limits>

#include "MetalImpl.h"
#include "GLImpl.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#define IM_NORMALIZE2F_OVER_ZERO(VX,VY)     { float d2 = VX*VX + VY*VY; if (d2 > 0.0f) { float inv_len = 1.0f / ImSqrt(d2); VX *= inv_len; VY *= inv_len; } }
#define IM_FIXNORMAL2F(VX,VY)               { float d2 = VX*VX + VY*VY; if (d2 < 0.5f) d2 = 0.5f; float inv_lensq = 1.0f / d2; VX *= inv_lensq; VY *= inv_lensq; }

#define GLFW_INCLUDE_GL3  /* don't drag in legacy GL headers. */
#define GLFW_NO_GLU       /* don't drag in the old GLU lib - unless you must. */

#include <GLFW/glfw3.h>

#include <misc/metastring.h>
#include <misc/GlobalSettings.h>

#if GLFW_VERSION_MAJOR <= 3 && GLFW_VERSION_MINOR <= 2
#define GLFW_HAVE_MONITOR_SCALE false
#include <GLFW/glfw3native.h>
#else
#define GLFW_HAVE_MONITOR_SCALE true
#endif

#include <misc/checked_casts.h>
#include <gui/colors.h>

namespace gui {


size_t cache_misses = 0, cache_finds = 0;

void cache_miss() {
    ++cache_misses;
}

void cache_find() {
    ++cache_finds;
}

void clear_cache() {
    //Debug("Misses: %lu vs. finds: %lu", cache_misses, cache_finds);
    cache_finds = cache_misses = 0;
}

class PolyCache : public CacheObject {
    GETTER_NCONST(std::vector<ImVec2>, points)
};

#ifndef NDEBUG
    class TextureCache;
    std::set<ImTextureID> all_gpu_texture;
#endif

    class TextureCache : public CacheObject {
        GETTER(TexturePtr, texture)
        GETTER_NCONST(Size2, size)
        GETTER_NCONST(Size2, gpu_size)
        GETTER_PTR(IMGUIBase*, base)
        GETTER_PTR(CrossPlatform*, platform)
        GETTER_PTR(Drawable*, object)
        GETTER_SETTER(uint32_t, channels)
        
        static std::unordered_map<const IMGUIBase*, std::set<std::shared_ptr<TextureCache>>> _all_textures;
        static std::mutex _texture_mutex;
        
    public:
        static std::shared_ptr<TextureCache> get(IMGUIBase* base) {
            auto ptr = std::make_shared<TextureCache>();
            ptr->set_base(base);
            ptr->platform() = base->platform().get();
            
            std::lock_guard<std::mutex> guard(_texture_mutex);
            _all_textures[base].insert(ptr);
            
            return ptr;
        }
        
        static void remove_base(IMGUIBase* base) {
            std::unique_lock<std::mutex> guard(_texture_mutex);
            for(auto & tex : _all_textures[base]) {
                tex->set_ptr(nullptr);
                tex->set_base(nullptr);
                tex->platform() = nullptr;
            }
            _all_textures.erase(base);
        }
        
    public:
        TextureCache()
            : _texture(nullptr), _base(nullptr), _platform(nullptr), _object(nullptr), _channels(0)
        {
            
        }
        
        TextureCache(TexturePtr&& ptr)
            : _texture(std::move(ptr)), _base(nullptr), _platform(nullptr), _object(nullptr), _channels(0)
        {
            
        }
        
    public:
        void set_base(IMGUIBase* base) {
            _base = base;
        }
        
        void set_ptr(TexturePtr&& ptr) {
            /*if(_texture) {
                if(_platform) {
                    assert(_base);
                    _base->exec_main_queue([ptr = std::move(_texture), cache = this, base = _platform]() mutable
                    {
                        if(ptr) {
#ifndef NDEBUG
                            std::lock_guard<std::mutex> guard(_texture_mutex);
                            auto it = all_gpu_texture.find(ptr->ptr);
                            if(it != all_gpu_texture.end()) {
                                all_gpu_texture.erase(it);
                            } else
                                Warning("Cannot find GPU texture of %X", cache);
#endif
                            ptr = nullptr;
                            //base->clear_texture(std::move(ptr));
                        }
                    });
                } else
                    Except("Cannot clear texture because platform is gone.");
            }*/
            
#ifndef NDEBUG
            if(_texture) {
                auto it = all_gpu_texture.find(_texture->ptr);
                if(it != all_gpu_texture.end()) {
                    all_gpu_texture.erase(it);
                } else
                    Warning("Cannot find GPU texture of %X", this);
            }
#endif
            
            _texture = std::move(ptr);
            
#ifndef NDEBUG
            if(_texture) {
                all_gpu_texture.insert(_texture->ptr);
            }
#endif
        }
        
        static Size2 gpu_size_of(const ExternalImage* image) {
            if(!image || !image->source())
                return Size2();
            return Size2(next_pow2(sign_cast<uint64_t>(image->source()->bounds().width)),
                         next_pow2(sign_cast<uint64_t>(image->source()->bounds().height)));
        }
        
        void update_with(const ExternalImage* image) {
            auto image_size = image->source()->bounds().size();
            auto image_channels = image->source()->dims;
            auto gpusize = gpu_size_of(image);
            //static size_t replacements = 0, adds = 0, created = 0;
            if(!_texture) {
                auto id = _platform->texture(image->source());
                set_ptr(std::move(id));
                _size = image_size;
                _channels = image_channels;
                //++created;
                
            } else if(gpusize.width > _texture->width
                      || gpusize.height > _texture->height
                      || gpusize.width < _texture->width/4
                      || gpusize.height < _texture->height/4
                      || channels() != image_channels)
            {
                //Debug("Texture of size %dx%d does not fit %.0fx%.0f", _texture->width, _texture->height, gpusize.width, gpusize.height);
                auto id = _platform->texture(image->source());
                set_ptr(std::move(id));
                _size = image_size;
                _channels = image_channels;
                //++adds;
                
            } else if(changed()) {
                _platform->update_texture(*_texture, image->source());
                set_changed(false);
                //++replacements;
            }
            
            //if(replacements%100 == 0)
            //    Debug("Replace: %lu, Add: %lu, Created: %lu", replacements, adds, created);
        }
        
        void set_object(Drawable* o) {
            auto pobj = _object;
            if(pobj && pobj != o) {
                _object = o;
                pobj->remove_cache(_base);
                if(o)
                    Warning("Exchanging cache for object");
            }
        }
        
        ~TextureCache() {
            set_ptr(nullptr);
            
            if(_base ) {
                std::lock_guard<std::mutex> guard(_texture_mutex);
                auto &tex = _all_textures[_base];
                for(auto it=tex.begin(); it != tex.end(); ++it) {
                    if(it->get() == this) {
                        tex.erase(it);
                        break;
                    }
                }
            }
        }
    };

    std::unordered_map<const IMGUIBase*, std::set<std::shared_ptr<TextureCache>>> TextureCache::_all_textures;
    std::mutex TextureCache::_texture_mutex;

    constexpr Codes glfw_key_map[GLFW_KEY_LAST + 1] {
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Space,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, // apostroph (39)
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Comma, // 44
        Codes::Subtract,
        Codes::Period,
        Codes::Slash,
        Codes::Num0, Codes::Num1, Codes::Num2, Codes::Num3, Codes::Num4, Codes::Num5, Codes::Num6, Codes::Num7, Codes::Num8, Codes::Num9,
        Codes::Unknown,
        Codes::SemiColon, //(69),
        Codes::Unknown,
        Codes::Equal, // (61)
        Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::A, Codes::B, Codes::C, Codes::D, Codes::E, Codes::F, Codes::G, Codes::H, Codes::I, Codes::J, Codes::K, Codes::L, Codes::M, Codes::N, Codes::O, Codes::P, Codes::Q, Codes::R, Codes::S, Codes::T, Codes::U, Codes::V, Codes::W, Codes::X, Codes::Z, Codes::Y,
        Codes::LBracket,
        Codes::BackSlash,
        Codes::RBracket, // (93)
        Codes::Unknown, Codes::Unknown,
        Codes::Unknown, // (grave accent, 96)
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, // world 1 (161)
        Codes::Unknown, // world 2 (162)
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, // 255
        
        Codes::Escape,
        Codes::Return,
        Codes::Tab,
        Codes::BackSpace,
        Codes::Insert,
        Codes::Delete,
        Codes::Right,
        Codes::Left,
        Codes::Down,
        Codes::Up,
        Codes::PageUp,
        Codes::PageDown,
        Codes::Home,
        Codes::End, // 269
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        Codes::Unknown, // capslock (280)
        Codes::Unknown, // scroll lock (281)
        Codes::Unknown, // num_lock (282)
        Codes::Unknown, // print screen (283)
        Codes::Pause, // 284
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, 
        // 290
        Codes::F1, Codes::F2, Codes::F3, Codes::F4, Codes::F5, Codes::F6, Codes::F7, Codes::F8, Codes::F9, Codes::F10, Codes::F11, Codes::F12, Codes::F13, Codes::F14, Codes::F15, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, // 314
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::Numpad0, Codes::Numpad1, Codes::Numpad2, Codes::Numpad3, Codes::Numpad4, Codes::Numpad5, Codes::Numpad6, Codes::Numpad7, Codes::Numpad8, Codes::Numpad9, // 329
        
        // numpad
        Codes::Comma,
        Codes::Divide,
        Codes::Multiply,
        Codes::Subtract,
        Codes::Add,
        Codes::Return,
        Codes::Equal, // 336
        
        Codes::Unknown, Codes::Unknown, Codes::Unknown,
        
        Codes::LShift, // 340
        Codes::LControl,
        Codes::LAlt,
        Codes::LSystem,
        Codes::RShift,
        Codes::RControl,
        Codes::RAlt,
        Codes::RSystem,
        Codes::Menu // 348
    };

    /*struct CachedFont {
        ImFont *ptr;
        float base_size;
        float line_spacing;
    };*/

    std::unordered_map<uint32_t, ImFont*> _fonts;
    float im_font_scale = 2;

    std::unordered_map<GLFWwindow*, IMGUIBase*> base_pointers;

void IMGUIBase::update_size_scale(GLFWwindow* window) {
    auto base = base_pointers.at(window);
    std::lock_guard lock_guard(base->_graph->lock());
    
    int x, y;
    glfwGetWindowPos(window, &x, &y);
    
    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    
    GLFWmonitor* monitor = glfwGetWindowMonitor(window);
    if(!monitor) {
        int count;
        auto monitors = glfwGetMonitors(&count);
        int mx, my, mw, mh;
#ifndef NDEBUG
        Debug("Window is at %d, %d (%dx%d).", x, y, fw, fh);
#endif
        
        for (int i=0; i<count; ++i) {
            auto name = glfwGetMonitorName(monitors[i]);
            glfwGetMonitorWorkarea(monitors[i], &mx, &my, &mw, &mh);
#ifndef NDEBUG
            Debug("Monitor '%s': %d,%d %dx%d", name, mx, my, mw, mh);
#endif
            if(Bounds(mx+5, my+5, mw-10, mh-10).overlaps(Bounds(x+5, y+5, fw-10, fh-10))) {
                monitor = monitors[i];
                break;
            }
        }
        
        if(!monitor) {
            // assume fullscreen?
            Debug("No monitor found.");
            return;
        }
        
    } else {
        int mx, my, mw, mh;
        glfwGetMonitorWorkarea(monitor, &mx, &my, &mw, &mh);
#ifndef NDEBUG
        Debug("FS Monitor: %d,%d %dx%d", mx, my, mw, mh);
#endif
    }
    
    float xscale, yscale;
#if GLFW_HAVE_MONITOR_SCALE
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
#else
    xscale = yscale = 1;
#endif
    
#ifndef NDEBUG
    Debug("Content scale: %f, %f", xscale, yscale);
#endif
    
    int width = base->_graph->width(), height = base->_graph->height();
    
    const float base_scale = 32;
    float dpi_scale = 1 / max(xscale, yscale);//max(float(fw) / float(width), float(fh) / float(height));
    im_font_scale = max(1, dpi_scale) * 0.75f;
    base->_dpi_scale = dpi_scale;
    base->_graph->set_scale(1 / dpi_scale);
    
    base->_last_framebuffer_size = Size2(fw, fh).mul(base->_dpi_scale);
    
    //Debug("dpi_scale:%f gui::interface_scale:%f xscale:%f yscale:%f", dpi_scale, gui::interface_scale(), xscale, yscale);
    
    {
        Event e(EventType::WINDOW_RESIZED);
        e.size.width = fw * dpi_scale;
        e.size.height = fh * dpi_scale;

        base->event(e);
    }
    
    base->_graph->set_dirty(NULL);
}

    ImU32 cvtClr(const gui::Color& clr) {
        return ImColor(clr.r, clr.g, clr.b, clr.a);
    }

    void IMGUIBase::init(const std::string& title, bool soft) {
        _platform->init();
        
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int count;
        auto monitors = glfwGetMonitors(&count);
        int maximum = 0;
        GLFWmonitor* choice = glfwGetPrimaryMonitor();
        for (int i = 0; i < count; ++i) {
            int width, height;
            glfwGetMonitorPhysicalSize(monitors[i], &width, &height);
            auto name = glfwGetMonitorName(monitors[i]);
            if (width * height > maximum) {
                choice = monitors[i];
                maximum = width * height;
            }
        }
        
        if (choice)
            monitor = choice;

        float xscale, yscale;
#if GLFW_HAVE_MONITOR_SCALE
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
#else
        xscale = yscale = 1;
#endif
        
        int width = _graph->width(), height = _graph->height();
        int mx, my, mw, mh;
#if GLFW_HAVE_MONITOR_SCALE
        glfwGetMonitorWorkarea(monitor, &mx, &my, &mw, &mh);
#else
        mx = my = 0;
        glfwGetMonitorPhysicalSize(monitor, &mw, &mh);
#endif
        //mw -= mx;
        //mh -= my;
        
#if WIN32
        my += mh * 0.04;
        mh *= 0.95; //! task bar
#endif
        
#ifdef WIN32
        width *= xscale;
        height *= yscale;
#endif
        _work_area = Bounds(mx, my, mw, mh);
        
        if(width / float(mw) >= height / float(mh)) {
            if(width > mw) {
                float ratio = float(height) / float(width);
                width = mw;
                height = int(width * ratio);
            }
        } else {
            if(height > mh) {
                float ratio = float(width) / float(height);
                height = mh;
                width = int(ratio * height);
            }
        }
        
        if(!_platform->window_handle())
            _platform->create_window(title.c_str(), width, height);
        else {
            glfwSetWindowSize(_platform->window_handle(), width, height);
            set_title(title);
        }
        
        glfwSetWindowPos(_platform->window_handle(), mx + (mw - width) / 2, my + (mh - height) / 2);
        
        glfwSetDropCallback(_platform->window_handle(), [](GLFWwindow* window, int N, const char** texts){
            std::vector<file::Path> _paths;
            for(int i=0; i<N; ++i)
                _paths.push_back(texts[i]);
            if(base_pointers.count(window)) {
                if(base_pointers[window]->_open_files_fn) {
                    base_pointers[window]->_open_files_fn(_paths);
                }
            }
        });
        _platform->set_open_files_fn([this](auto& files) -> bool {
            if(_open_files_fn)
                return _open_files_fn(files);
            return false;
        });
        
        file::Path path("fonts/Quicksand-");
        if (!path.add_extension("ttf").exists())
            Except("Cannot find file '%S'", &path.str());
        
        auto& io = ImGui::GetIO();
        //io.FontAllowUserScaling = true;
        //io.WantCaptureMouse = false;
        //io.WantCaptureKeyboard = false;
        
        const float base_scale = 32;
        //float dpi_scale = max(float(fw) / float(width), float(fh) / float(height));
        float dpi_scale = 1 / max(xscale, yscale);
        im_font_scale = max(1, dpi_scale) * 0.75f;
        _dpi_scale = dpi_scale;
        
        int fw, fh;
        glfwGetFramebufferSize(_platform->window_handle(), &fw, &fh);
        _last_framebuffer_size = Size2(fw, fh).mul(_dpi_scale);
        
        if (!soft) {
            io.Fonts->Clear();
            _fonts.clear();
        }

        if(_fonts.empty()) {
            ImFontConfig config;
            config.OversampleH = 3;
            config.OversampleV = 1;

            auto load_font = [&](int no, std::string suffix) {
                config.FontNo = no;
                if (no > 0)
                    config.MergeMode = false;

                auto full = path.str() + suffix + ".ttf";
                auto ptr = io.Fonts->AddFontFromFileTTF(full.c_str(), base_scale * im_font_scale, &config);
                if (!ptr) {
                    Warning("Cannot load font '%S' with index %d.", &path.str(), config.FontNo);
                    ptr = io.Fonts->AddFontDefault();
                    im_font_scale = max(1, dpi_scale) * 0.5f;
                }
                ptr->FontSize = base_scale * im_font_scale;

                return ptr;
            };

            _fonts[Style::Regular] = load_font(0, "");
            _fonts[Style::Italic] = load_font(0, "i");
            _fonts[Style::Bold] = load_font(0, "b");
            _fonts[Style::Bold | Style::Italic] = load_font(0, "bi");
        }

        _platform->post_init();
        _platform->set_title(title);
        
        base_pointers[_platform->window_handle()] = this;
        
        glfwSetKeyCallback(_platform->window_handle(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
            auto base = base_pointers.at(window);
            
            Event e(EventType::KEY);
            e.key.pressed = action == GLFW_PRESS || action == GLFW_REPEAT;
            assert(key <= GLFW_KEY_LAST);
            e.key.code = glfw_key_map[key];
            e.key.shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
            
            base->event(e);
            base->_graph->set_dirty(NULL);
        });
        glfwSetCursorPosCallback(_platform->window_handle(), [](GLFWwindow* window, double xpos, double ypos) {
            auto base = base_pointers.at(window);
            
            Event e(EventType::MMOVE);
            auto &io = ImGui::GetIO();
            e.move.x = float(xpos * io.DisplayFramebufferScale.x) * base->_dpi_scale;
            e.move.y = float(ypos * io.DisplayFramebufferScale.y) * base->_dpi_scale;
            
            base->event(e);
            
            base->_graph->set_dirty(NULL);
            /**/
        });
        glfwSetMouseButtonCallback(_platform->window_handle(), [](GLFWwindow* window, int button, int action, int mods) {
            if(button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_RIGHT)
                return;
            
            Event e(EventType::MBUTTON);
            e.mbutton.pressed = action == GLFW_PRESS;
            
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            
            auto base = base_pointers.at(window);
            auto &io = ImGui::GetIO();
            e.mbutton.x = float(xpos * io.DisplayFramebufferScale.x) * base->_dpi_scale;
            e.mbutton.y = float(ypos * io.DisplayFramebufferScale.y) * base->_dpi_scale;
            e.mbutton.button = GLFW_MOUSE_BUTTON_RIGHT == button ? 1 : 0;
            
            base->event(e);
            base->_graph->set_dirty(NULL);
        });
        glfwSetScrollCallback(_platform->window_handle(), [](GLFWwindow* window, double xoff, double yoff) {
            Event e(EventType::SCROLL);
            e.scroll.dy = float(yoff);
            e.scroll.dx = float(xoff);
            
            auto base = base_pointers.at(window);
            base->event(e);
            base->_graph->set_dirty(NULL);
        });
        glfwSetCharCallback(_platform->window_handle(), [](GLFWwindow* window, unsigned int c) {
            Event e(EventType::TEXT_ENTERED);
            e.text.c = char(c);
            
            auto base = base_pointers.at(window);
            base->event(e);
            base->_graph->set_dirty(NULL);
        });
        glfwSetWindowSizeCallback(_platform->window_handle(), [](GLFWwindow* window, int, int)
        {
            IMGUIBase::update_size_scale(window);
        });
        
        glfwSetWindowPosCallback(_platform->window_handle(), [](GLFWwindow* window, int, int)
        {
            IMGUIBase::update_size_scale(window);
        });
        
        exec_main_queue([window = _platform->window_handle()](){
            IMGUIBase::update_size_scale(window);
        });

        Debug("IMGUIBase::init complete");
    }

    IMGUIBase::~IMGUIBase() {
        while(!_exec_main_queue.empty()) {
            (_exec_main_queue.front())();
            _exec_main_queue.pop();
        }
        
        TextureCache::remove_base(this);
        
        while(!_exec_main_queue.empty()) {
            (_exec_main_queue.front())();
            _exec_main_queue.pop();
        }
        
        base_pointers.erase(_platform->window_handle());
    }

    void IMGUIBase::set_background_color(const Color& color) {
        if(_platform)
            _platform->set_clear_color(color);
    }

    void IMGUIBase::event(const gui::Event &e) {
        if(_graph->event(e) && e.type != EventType::MMOVE)
            return;
        
        try {
            _event_fn(e);
        } catch(const std::invalid_argument& e) { }
    }

    void IMGUIBase::loop() {
        LoopStatus status = LoopStatus::IDLE;
        
        // Main loop
        while (status != LoopStatus::END)
        {
            status = update_loop();
            if(status != gui::LoopStatus::UPDATED)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    LoopStatus IMGUIBase::update_loop() {
        {
            std::unique_lock<std::mutex> guard(_mutex);
            while (!_exec_main_queue.empty()) {
                auto fn = std::move(_exec_main_queue.front());
                
                guard.unlock();
                (fn)();
                guard.lock();
                _exec_main_queue.pop();
            }
        }
        
        return _platform->update_loop(_custom_loop);
    }

    void IMGUIBase::set_frame_recording(bool v) {
        _frame_recording = v;
        if(v) {
            _platform->set_frame_capture_enabled(true);
        } else if(_platform->frame_capture_enabled())
            _platform->set_frame_capture_enabled(false);
    }

    const Image::UPtr& IMGUIBase::current_frame_buffer() {
        return _platform->current_frame_buffer();
    }

    void IMGUIBase::paint(DrawStructure& s) {
        int fw, fh;
        auto window = _platform->window_handle();
        glfwGetFramebufferSize(window, &fw, &fh);
        fw *= _dpi_scale;
        fh *= _dpi_scale;
        
        if(fw > 0 && fh > 0 && (fw != _last_framebuffer_size.width || fh != _last_framebuffer_size.height))
        {
#ifndef NDEBUG
            Debug("Changed framebuffer size to %dx%d", fw, fh);
#endif
            _last_framebuffer_size = Size2(fw, fh);
        }
        
        std::unique_lock<std::recursive_mutex> lock(s.lock());
        auto objects = s.collect();
        _objects_drawn = 0;
        _skipped = 0;
        _type_counts.clear();
        _draw_order.clear();
        _rotation_starts.clear();
        
        for (size_t i=0; i<objects.size(); i++) {
            auto o =  objects[i];
            redraw(o, _draw_order);
        }
        
        std::sort(_draw_order.begin(), _draw_order.end(), [](const auto& A, const auto& B) {
            return A.ptr->z_index() < B.ptr->z_index() || (A.ptr->z_index() == B.ptr->z_index() && A.index < B.index);
        });
        
        //Debug("----");
        for(auto & order : _draw_order) {
            //Debug("Drawing %lu (%d) (%s)", order.index, order.ptr->z_index(), order.ptr->type().name());
            draw_element(order);
        }
        
#ifndef NDEBUG
        if(_last_debug_print.elapsed() > 60) {
            auto str = Meta::toStr(_type_counts);
            Debug("%d drawn, %d skipped, types: %S", _objects_drawn, _skipped, &str);
            _last_debug_print.reset();
        }
#endif
    }

bool LineSegementsIntersect(const Vec2& p, const Vec2& p2, const Vec2& q, const Vec2& q2, Vec2& intersection)
{
    constexpr bool considerCollinearOverlapAsIntersect = false;
    
    auto r = p2 - p;
    auto s = q2 - q;
    auto rxs = cross(r, s);
    auto qpxr = cross(q - p, r);
    
    // If r x s = 0 and (q - p) x r = 0, then the two lines are collinear.
    if (rxs == 0 && qpxr == 0)
    {
        // 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
        // then the two lines are overlapping,
        if constexpr (considerCollinearOverlapAsIntersect)
            if ((0 <= (q - p).dot(r) && (q - p).dot(r) <= r.dot(r)) || (0 <= (p - q).dot(s) && (p - q).dot(s) <= s.dot(s)))
                return true;
        
        // 2. If neither 0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
        // then the two lines are collinear but disjoint.
        // No need to implement this expression, as it follows from the expression above.
        return false;
    }
    
    // 3. If r x s = 0 and (q - p) x r != 0, then the two lines are parallel and non-intersecting.
    if (rxs == 0 && qpxr != 0)
        return false;
    
    // t = (q - p) x s / (r x s)
    auto t = cross(q - p,s)/rxs;
    
    // u = (q - p) x r / (r x s)
    
    auto u = cross(q - p, r)/rxs;
    
    // 4. If r x s != 0 and 0 <= t <= 1 and 0 <= u <= 1
    // the two line segments meet at the point p + t r = q + u s.
    if (rxs != 0 && (0 <= t && t < 1) && (0 <= u && u < 1))
    {
        // We can calculate the intersection point using either t or u.
        intersection = p + t*r;
        
        // An intersection was found.
        return true;
    }
    
    // 5. Otherwise, the two line segments are not parallel but do not intersect.
    return false;
}

/**
 Source: https://github.com/ocornut/imgui/issues/760
 */
void PolyFillScanFlood(ImDrawList *draw, std::vector<ImVec2> *poly, std::vector<ImVec2>& output, ImColor color, const int gap = 1, const int strokeWidth = 1) {
    using namespace std;

    vector<ImVec2> scanHits;
    static ImVec2 min, max; // polygon min/max points
    auto &io = ImGui::GetIO();
    bool isMinMaxDone = false;
    const auto polysize = poly->size();
    if(polysize < 3)
        return; // smaller shapes (lines and points) cannot be filled

    // find the orthagonal bounding box
    // probably can put this as a predefined
    if (!isMinMaxDone) {
        min.x = min.y = FLT_MAX;
        max.x = max.y = -FLT_MAX;
        for (auto p : *poly) {
            if (p.x < min.x) min.x = p.x;
            if (p.y < min.y) min.y = p.y;
            if (p.x > max.x) max.x = p.x;
            if (p.y > max.y) max.y = p.y;
        }
        isMinMaxDone = true;
    }
    
    /*struct Edge {
        int yMin;
        int yMax;
        float xHit;
        float mInv;
        
        Edge() {}
        
        Edge(int yMin, int yMax, float xHit, float mInv)
            : yMin(yMin), yMax(yMax), xHit(xHit), mInv(mInv)
        {}
        
        bool operator<(const Edge& other) const {
            return yMin < other.yMin || (yMin == other.yMin && xHit < other.xHit);
        }
    };
    std::vector<Edge> segments;
    segments.reserve(poly->size());
    
    for (size_t i=0; i<poly->size(); ++i) {
        auto prev = i ? (i - 1) : (poly->size()-1);
        
        auto &A = (*poly)[prev];
        auto &B = (*poly)[i];
        
        if(Vec2(A) == Vec2(B))
            continue;
        
        if(cmn::min(A.x, B.x) > io.DisplaySize.x)
            continue;
        
        if(cmn::max(A.x, B.x) < 0)
            continue;
        
        if(A.y <= B.y) {
            if(B.y < 0 || A.y > io.DisplaySize.y)
                continue;
            else
                segments.emplace_back(A.y, B.y, A.x, (B.x - A.x) / (B.y - A.y));
        } else {
            if(A.y < 0 || B.y > io.DisplaySize.y)
                continue;
            else
                segments.emplace_back(B.y, A.y, B.x, (A.x - B.x) / (A.y - B.y));
        }
    }
    
    std::sort(segments.begin(), segments.end());*/
    
    // Vertically clip
    if (min.y < 0) min.y                = 0;
    if (max.y > io.DisplaySize.y) max.y = io.DisplaySize.y;
    
    // traverse all y-coordinates
    /*std::vector<Vec2> intersections;
    int y = min.y;
    std::vector<Edge> AET;
    
    while (!segments.empty()) {
        AET.clear();
        
        for(auto it = segments.begin(); it != segments.end(); ) {
            if(it->yMin == y) {
                AET.push_back(*it);
            }
            
            if(it->yMax == y)
                it = segments.erase(it);
            else ++it;
        }
        
        std::sort(AET.begin(), AET.end(), [](const Edge&A, const Edge&B){
            return A.xHit < B.xHit;
        });
        
        bool parity = false;
        float x;
        for(auto &edge : AET) {
            if(parity) {
                //Debug("Line %f,%d - %f,%d", x,y, edge.xHit, y);
                draw->AddLine(Vec2(x,y), Vec2(edge.xHit, y), color, strokeWidth);
            } else {
                x = edge.xHit;
            }
            parity = !parity;
        }
        
        for(auto &edge : segments) {
            if(!std::isinf(edge.mInv))
                edge.xHit = edge.xHit + edge.mInv;
        }
        
        ++y;
    }
    
    return;*/

    // Bounds check
    if ((max.x < 0) || (min.x > io.DisplaySize.x) || (max.y < 0) || (min.y > io.DisplaySize.y)) return;


    // so we know we start on the outside of the object we step out by 1.
    min.x -= 1;
    max.x += 1;

    // Initialise our starting conditions
    int y = int(min.y);

    // Go through each scan line iteratively, jumping by 'gap' pixels each time
    while (y < max.y) {

        scanHits.resize(0);

        {
            int jump = 1;
            ImVec2 fp = poly->at(0);

            for (size_t i = 0; i < polysize - 1; i++) {
                ImVec2 pa = (*poly)[i];
                ImVec2 pb = (*poly)[i+1];

                // jump double/dud points
                if (pa.x == pb.x && pa.y == pb.y) continue;

                // if we encounter our hull/poly start point, then we've now created the
                // closed
                // hull, jump the next segment and reset the first-point
                if ((!jump) && (fp.x == pb.x) && (fp.y == pb.y)) {
                    if (i < polysize - 2) {
                        fp   = (*poly)[i + 2];
                        jump = 1;
                        i++;
                    }
                } else {
                    jump = 0;
                }

                // test to see if this segment makes the scan-cut.
                if ((pa.y > pb.y && y < pa.y && y > pb.y) || (pa.y < pb.y && y > pa.y && y < pb.y)) {
                    ImVec2 intersect;

                    intersect.y = y;
                    if (pa.x == pb.x) {
                        intersect.x = pa.x;
                    } else {
                        intersect.x = (pb.x - pa.x) / (pb.y - pa.y) * (y - pa.y) + pa.x;
                    }
                    scanHits.push_back(intersect);
                }
            }

            // Sort the scan hits by X, so we have a proper left->right ordering
            sort(scanHits.begin(), scanHits.end(), [](ImVec2 const &a, ImVec2 const &b) { return a.x < b.x; });

            // generate the line segments.
            {
                auto l = scanHits.size(); // we need pairs of points, this prevents segfault.
                for (size_t i = 0; i+1 < l; i += 2) {
                    output.push_back(scanHits[i]);
                    output.push_back(scanHits[i+1]);
                    draw->AddLine(scanHits[i], scanHits[i + 1], color, strokeWidth);
                }
            }
        }
        y += gap;
    } // for each scan line
    scanHits.clear();
}

void ImRotateStart(int& rotation_start_index, ImDrawList* list)
{
    rotation_start_index = list->VtxBuffer.Size;
}

ImVec2 ImRotationCenter(int rotation_start_index, ImDrawList* list)
{
    ImVec2 l(FLT_MAX, FLT_MAX), u(-FLT_MAX, -FLT_MAX); // bounds

    const auto& buf = list->VtxBuffer;
    for (int i = rotation_start_index; i < buf.Size; i++)
        l = ImMin(l, buf[i].pos), u = ImMax(u, buf[i].pos);

    return ImVec2((l.x+u.x)/2, (l.y+u.y)/2); // or use _ClipRectStack?
}

ImVec2 operator-(const ImVec2& l, const ImVec2& r) { return{ l.x - r.x, l.y - r.y }; }

void ImRotateEnd(int rotation_start_index, ImDrawList* list, float rad, ImVec2 center)
{
    //auto center = ImRotationCenter(list);
    float s=cos(rad), c=sin(rad);
    center = ImRotate(center, s, c) - center;

    auto& buf = list->VtxBuffer;
    for (int i = rotation_start_index; i < buf.Size; i++)
        buf[i].pos = ImRotate(buf[i].pos, s, c) - center;
}

bool operator!=(const ImVec4& A, const ImVec4& B) {
    return A.w != B.w || A.x != B.x || A.y != B.y || A.z != B.z;
}

void IMGUIBase::draw_element(const DrawOrder& order) {
    auto list = ImGui::GetForegroundDrawList();
    /*if(order.type == DrawOrder::POP) {
        if(list->_ClipRectStack.size() > 1) {
            //Debug("Popped cliprect %.0f,%.0f", list->_ClipRectStack.back().x, list->_ClipRectStack.back().y);
            //list->PopClipRect();
        } else
            Warning("Cannot pop too many rects.");
        return;
    }*/
    
    if(order.type == DrawOrder::END_ROTATION) {
        auto o = order.ptr;
        assert(o->has_global_rotation());
        if(!_rotation_starts.count(o)) {
            Warning("Cannot find object.");
            return;
        }
        auto && [rotation_start, center] = _rotation_starts.at(o);
        ImRotateEnd(rotation_start, list, o->rotation(), center);
        return;
    }
    
    auto o = order.ptr;
    Vec2 center;
    auto bds = order.bounds;
    auto &io = ImGui::GetIO();
    Transform transform(order.transform);
    
    int rotation_start;
    
    if(order.type == DrawOrder::START_ROTATION) {
        ImRotateStart(rotation_start, list);
        
        // generate position without rotation
        Vec2 scale = (_graph->scale() / gui::interface_scale() / _dpi_scale) .div(Vec2(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y));
        
        transform = Transform();
        transform.scale(scale);
        transform.combine(o->parent()->global_transform());
        transform.translate(o->pos());
        transform.scale(o->scale());
        transform.translate(-o->size().mul(o->origin()));
        
        bds = transform.transformRect(Bounds(Vec2(), o->size()));
        center = bds.pos() + bds.size().mul(o->origin());
        
        if(o->type() == Type::ENTANGLED) {
            _rotation_starts[o] = { rotation_start, center };
        }
        
        return;
    }
    
    if(!o->visible())
        return;
    
    ++_objects_drawn;
    auto cache = o->cached(this);
    
    if(o->rotation() && o->type() != Type::ENTANGLED) {
        ImRotateStart(rotation_start, list);
        
        // generate position without rotation
        Vec2 scale = (_graph->scale() / gui::interface_scale() / _dpi_scale) .div(Vec2(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y));
        
        transform = Transform();
        transform.scale(scale);
        transform.combine(o->parent()->global_transform());
        transform.translate(o->pos());
        transform.scale(o->scale());
        transform.translate(-o->size().mul(o->origin()));
        
        bds = transform.transformRect(Bounds(Vec2(), o->size()));
        center = bds.pos() + bds.size().mul(o->origin());
    }
    
    bool pushed_rect = false;
    //if(order._clip_rect.w > 0 && order._clip_rect.z > 0 && (list->_ClipRectStack.empty() || list->_ClipRectStack.back() != order._clip_rect))
    if(order._clip_rect.w > 0 && order._clip_rect.z > 0) {
        //list->AddRect(ImVec2(order._clip_rect.x, order._clip_rect.y),
        //              ImVec2(order._clip_rect.w, order._clip_rect.z), cvtClr(Red));
        list->PushClipRect(ImVec2(order._clip_rect.x, order._clip_rect.y),
                           ImVec2(order._clip_rect.w, order._clip_rect.z), false);
        pushed_rect = true;
    }

    auto i_ = list->VtxBuffer.Size;
    
    switch (o->type()) {
        case Type::CIRCLE: {
            auto ptr = static_cast<Circle*>(o);
            
            // calculate a reasonable number of segments based on global bounds
            auto e = 0.25f;
            auto r = max(1, order.bounds.width * 0.5f);
            auto th = acos(2 * SQR(1 - e / r) - 1);
            int64_t num_segments = (int64_t)ceil(2*M_PI/th);
            
            if(num_segments <= 1)
                break;
            
            // generate circle path
            auto centre = ImVec2(ptr->radius(), ptr->radius());
            const float a_max = (float)M_PI*2.0f * ((float)num_segments - 1.0f) / (float)num_segments;
            
            list->PathArcTo(centre, ptr->radius(), 0.0f, a_max, (int)num_segments - 1);
            
            // transform according to local transform etc.
            for (auto i=0; i<list->_Path.Size; ++i) {
                list->_Path.Data[i] = order.transform.transformPoint(list->_Path.Data[i]);
            }
            
            // generate vertices (1. filling + 2. outline)
            if(ptr->fill_clr() != Transparent)
                list->AddConvexPolyFilled(list->_Path.Data, list->_Path.Size, (ImColor)ptr->fill_clr());
            if(ptr->line_clr() != Transparent)
                list->AddPolyline(list->_Path.Data, list->_Path.Size, (ImColor)ptr->line_clr(), true, 1);
            
            // reset path
            list->_Path.Size = 0;
            
            break;
        }
            
        case Type::POLYGON: {
            auto ptr = static_cast<Polygon*>(o);
            static std::vector<ImVec2> points;
            if(ptr->relative()) {
                points.clear();
                for(auto &pt : *ptr->relative()) {
                    points.push_back(order.transform.transformPoint(pt));
                }
                
                if(points.size() >= 3) {
                    points.push_back(points.front());
                    static std::vector<ImVec2> output;
                    output.clear();
                    
                    if(!cache) {
                        cache = std::make_shared<PolyCache>();
                        o->insert_cache(this, cache);
                    }
                    
                    if(cache->changed()) {
                        PolyFillScanFlood(list, &points, output, ptr->fill_clr());
                        ((PolyCache*)cache.get())->points() = output;
                        cache->set_changed(false);
                    } else {
                        auto& output = ((PolyCache*)cache.get())->points();
                        for(size_t i=0; i<output.size(); i+=2) {
                            list->AddLine(output[i], output[i+1], (ImColor)ptr->fill_clr(), 1);
                        }
                    }
                    
                } else if(cache) {
                    o->remove_cache(this);
                }
                
                //list->AddConvexPolyFilled(points.data(), points.size(), (ImColor)ptr->fill_clr());
                if(ptr->border_clr() != Transparent)
                    list->AddPolyline(points.data(), (int)points.size(), (ImColor)ptr->border_clr(), true, 1);
            }
            
            break;
        }
            
        case Type::TEXT: {
            auto ptr = static_cast<Text*>(o);
            
            if(ptr->txt().empty())
                break;
            
            auto font = _fonts.at(ptr->font().style);
            
            list->AddText(font, ptr->global_text_scale().x * font->FontSize * (ptr->font().size / im_font_scale / _dpi_scale / io.DisplayFramebufferScale.x), bds.pos(), (ImColor)ptr->color(), ptr->txt().c_str());
            
            break;
        }
            
        case Type::ENTANGLED: {
            //list->AddRect(ImVec2(bds.x, bds.y), ImVec2(bds.x + bds.width, bds.y + bds.height), cvtClr(Red));
            //list->PushClipRect(ImVec2(bds.x, bds.y), ImVec2(bds.x + bds.width, bds.y + bds.height), false);
            
            //Debug("Pushing cliprect of %.0f,%.0f", list->_ClipRectStack.back().x, list->_ClipRectStack.back().y);
            break;
        }
            
        case Type::IMAGE: {
            if(!cache) {
                auto tex_cache = TextureCache::get(this);
                cache = tex_cache;
                o->insert_cache(this, cache);
                
                //tex_cache->platform() = _platform.get();
                //tex_cache->set_base(this);
                tex_cache->set_object(o);
                
                //_all_textures[this].insert(tex_cache);
            }
            
            if(!static_cast<ExternalImage*>(o)->source())
                break;
            
            auto image_size = static_cast<ExternalImage*>(o)->source()->bounds().size();
            if(image_size.empty())
                break;
            
            auto tex_cache = (TextureCache*)cache.get();
            tex_cache->update_with(static_cast<ExternalImage*>(o));
            tex_cache->set_changed(false);
            
            ImU32 col = IM_COL32_WHITE;
            uchar a = static_cast<ExternalImage*>(o)->color().a;
            if(a > 0 && static_cast<ExternalImage*>(o)->color() != White)
                col = (ImColor)static_cast<ExternalImage*>(o)->color();
            
            auto I = list->VtxBuffer.size();
            list->AddImage(tex_cache->texture()->ptr,
                           ImVec2(0, 0),
                           ImVec2(o->width(), o->height()),
                           ImVec2(0, 0),
                           ImVec2(tex_cache->texture()->image_width / float(tex_cache->texture()->width),
                                  tex_cache->texture()->image_height / float(tex_cache->texture()->height)),
                                  col);
            for (auto i = I; i<list->VtxBuffer.size(); ++i) {
                list->VtxBuffer[i].pos = transform.transformPoint(list->VtxBuffer[i].pos);
            }
            break;
        }
            
        case Type::RECT: {
            auto ptr = static_cast<Rect*>(o);
            auto &rect = order.bounds;
            
            if(rect.size().empty())
                break;
            
            if(ptr->fillclr().a > 0)
                list->AddRectFilled((ImVec2)transform.transformPoint(Vec2()),
                                    (ImVec2)transform.transformPoint(o->size()),
                                    cvtClr(ptr->fillclr()));
            
            if(ptr->lineclr().a > 0)
                list->AddRect((ImVec2)transform.transformPoint(Vec2()),
                              (ImVec2)transform.transformPoint(o->size()),
                              cvtClr(ptr->lineclr()));
            
            break;
        }
            
        case Type::VERTICES: {
            auto ptr = static_cast<Vertices*>(o);
            
            // Non Anti-aliased Stroke
            auto &points = ptr->points();
            auto points_count = points.size();
            
            if(points_count <= 1)
                break;
            
            auto count = points_count - 1;
            float thickness = ptr->thickness();
            const ImVec2 uv = list->_Data->TexUvWhitePixel;
            
            const auto idx_count = count*6;
            const auto vtx_count = count*4;      // FIXME-OPT: Not sharing edges
            list->PrimReserve(idx_count, vtx_count);
            assert(idx_count > 0 && vtx_count > 0);
            
            //Transform transform;
            //transform.scale(_graph->scale());
            //transform.combine(o->global_transform());
            //auto transform = o->global_transform();

            for (size_t i1 = 0; i1 < count; i1++)
            {
                const size_t i2 = (i1+1) == points_count ? 0 : i1+1;
                auto p1 = order.transform.transformPoint(points[i1].position());
                auto p2 = order.transform.transformPoint(points[i2].position());
                
                //const auto& p1 = points[i1].position();
                //const auto& p2 = points[i2].position();
                auto col = cvtClr(points[i1].color());

                float dx = p2.x - p1.x;
                float dy = p2.y - p1.y;
                
                
                IM_NORMALIZE2F_OVER_ZERO(dx, dy);
                dx *= (thickness * 0.5f);
                dy *= (thickness * 0.5f);

                list->_VtxWritePtr[0].pos.x = p1.x + dy; list->_VtxWritePtr[0].pos.y = p1.y - dx; list->_VtxWritePtr[0].uv = uv; list->_VtxWritePtr[0].col = col;
                list->_VtxWritePtr[1].pos.x = p2.x + dy; list->_VtxWritePtr[1].pos.y = p2.y - dx; list->_VtxWritePtr[1].uv = uv; list->_VtxWritePtr[1].col = col;
                list->_VtxWritePtr[2].pos.x = p2.x - dy; list->_VtxWritePtr[2].pos.y = p2.y + dx; list->_VtxWritePtr[2].uv = uv; list->_VtxWritePtr[2].col = col;
                list->_VtxWritePtr[3].pos.x = p1.x - dy; list->_VtxWritePtr[3].pos.y = p1.y + dx; list->_VtxWritePtr[3].uv = uv; list->_VtxWritePtr[3].col = col;
                list->_VtxWritePtr += 4;

                list->_IdxWritePtr[0] = (ImDrawIdx)(list->_VtxCurrentIdx); list->_IdxWritePtr[1] = (ImDrawIdx)(list->_VtxCurrentIdx+1); list->_IdxWritePtr[2] = (ImDrawIdx)(list->_VtxCurrentIdx+2);
                list->_IdxWritePtr[3] = (ImDrawIdx)(list->_VtxCurrentIdx); list->_IdxWritePtr[4] = (ImDrawIdx)(list->_VtxCurrentIdx+2); list->_IdxWritePtr[5] = (ImDrawIdx)(list->_VtxCurrentIdx+3);
                list->_IdxWritePtr += 6;
                list->_VtxCurrentIdx += 4;
            }
            
            list->_Path.Size = 0;
            
            break;
        }
            
        default:
            break;
    }
    
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
    if(SETTING(gui_blur_enabled)) {
        bool blur = false;
        auto p = o;
        while(p) {
            if(p->tagged(Effects::blur)) {
                blur = true;
                break;
            }
            
            p = p->parent();
        }
        
        auto e = list->VtxBuffer.Size;
        for(auto i=i_; i<e; ++i) {
            list->VtxBuffer[i].mask = blur;
        }
    }
#endif
    
    if(!list->CmdBuffer.empty()) {
        if(list->CmdBuffer.back().ElemCount == 0) {
            (void)list->CmdBuffer;
            ///Debug("Empty cmd buffer.");
        }
    }
    
#ifdef DEBUG_BOUNDARY
    list->AddRect(bds.pos(), bds.pos() + bds.size(), (ImColor)Red.alpha(125));
    std::string text;
    if(o->parent() && o->parent()->background() == o) {
        if(dynamic_cast<Entangled*>(o->parent()))
            text = dynamic_cast<Entangled*>(o->parent())->name() + " " + Meta::toStr(o->parent()->bounds());
        else
            text = Meta::toStr(*o->parent());
    } else
        text = Meta::toStr(*o);
    auto font = _fonts.at(Style::Regular);
    auto _font = Font(0.3, Style::Regular);
    
    list->AddText(font, font->FontSize * (_font.size / im_font_scale / _dpi_scale / io.DisplayFramebufferScale.x), bds.pos(), (ImColor)White.alpha(125), text.c_str());
#endif
    
    if(o->type() != Type::ENTANGLED && o->has_global_rotation()) {
        ImRotateEnd(rotation_start, list, o->rotation(), center);
    }
    
    if(pushed_rect) {
        assert(!list->_ClipRectStack.empty());
        list->PopClipRect();
    }
}

    void IMGUIBase::redraw(Drawable *o, std::vector<DrawOrder>& draw_order, bool is_background, ImVec4 clip_rect) {
        static auto entangled_will_texture = [](Entangled* e) {
            assert(e);
            if(e->scroll_enabled() && e->size().max() > 0) {
                return true;
            }
            
            return false;
        };
        
        if(o->type() == Type::SINGLETON)
            o = static_cast<SingletonObject*>(o)->ptr();
        o->set_visible(false);
        
        auto &io = ImGui::GetIO();
        Vec2 scale = (_graph->scale() / gui::interface_scale() / _dpi_scale) .div(Vec2(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y));
        Transform transform;
        transform.scale(scale);
        
        if(is_background && o->parent() && o->parent()->type() == Type::ENTANGLED)
        {
            auto p = o->parent();
            if(p)
                transform.combine(p->global_transform());
            
        } else {
            transform.combine(o->global_transform());
        }
        
        auto bounds = transform.transformRect(Bounds(0, 0, o->width(), o->height()));
        
        //bool skip = false;
        auto cache = o->cached(this);
        auto dim = window_dimensions() / _dpi_scale;
        
        if(!Bounds(0, 0, dim.width-0, dim.height-0).overlaps(bounds)) {
            ++_skipped;
            return;
        }
        
        o->set_visible(true);
        ++_type_counts[o->type()];
        
        switch (o->type()) {
            case Type::ENTANGLED: {
                auto ptr = static_cast<Entangled*>(o);
                if(ptr->rotation() != 0)
                    draw_order.emplace_back(DrawOrder::START_ROTATION, draw_order.size(), o, transform, bounds, clip_rect);
                
                auto bg = static_cast<Entangled*>(o)->background();
                if(bg) {
                    redraw(bg, draw_order, true, clip_rect);
                }
                
                if(entangled_will_texture(ptr)) {
                    clip_rect = bounds;
                    
                    //draw_order.emplace_back(DrawOrder::DEFAULT, draw_order.size(), ptr, transform, bounds, clip_rect);
                    
                    for(auto c : ptr->children()) {
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
                        
                        redraw(c, draw_order, false, clip_rect);
                    }
                    
                    //draw_order.emplace_back(DrawOrder::POP, draw_order.size(), ptr, transform, bounds, clip_rect);
                    
                } else {
                    for(auto c : ptr->children())
                        redraw(c, draw_order, false, clip_rect);
                }
                
                if(ptr->rotation() != 0) {
                    draw_order.emplace_back(DrawOrder::END_ROTATION, draw_order.size(), ptr, transform, bounds, clip_rect);
                }
                
                break;
            }
                
            default:
                draw_order.emplace_back(DrawOrder::DEFAULT, draw_order.size(), o, transform, bounds, clip_rect);
                break;
        }
    }

    Bounds IMGUIBase::text_bounds(const std::string& text, Drawable* obj, const Font& font) {
        /*Vec2 scale(1, 1);
        
        if(obj) {
            try {
                //scale = obj->global_text_scale();
                //text.setScale(gscale.reciprocal());
                //text.setCharacterSize(font.size * gscale.x * 25);
                //font_size = font.size * gscale.x * font_size;
                
            } catch(const UtilsException& ex) {
                Warning("Not initialising scale of (probably StaticText) fully because of a UtilsException.");
                //text.setCharacterSize(font.size * 25);
                //text.setScale(1, 1);
            }
            
        } else {
            //text.setCharacterSize(font.size * 25);
            //text.setScale(1, 1);
        }*/
        
        if(_fonts.empty()) {
            Warning("Trying to retrieve text_bounds before fonts are available.");
            return gui::Base::text_bounds(text, obj, font);
        }
        
        auto imfont = _fonts.at(font.style);
        ImVec2 size = imfont->CalcTextSizeA(imfont->FontSize * font.size / im_font_scale, FLT_MAX, -1.0f, text.c_str(), text.c_str() + text.length(), NULL);
        // Round
        //size.x = max(0, (float)(int)(size.x - 0.95f));
        size.y = line_spacing(font);
        //Debug("font.size = %f, FontSize = %f, im_font_scale = %f, size = (%f, %f) '%S'", font.size, im_font->FontSize, im_font_scale, size.x, size.y, &text);
        //return text_size;
        //auto size = ImGui::CalcTextSize(text.c_str());
        return Bounds(Vec2(), size);
    }

    uint32_t IMGUIBase::line_spacing(const Font& font) {
        //Debug("font.size = %f, FontSize = %f, im_font_scale = %f", font.size, im_font->FontSize, im_font_scale);
        if(_fonts.empty()) {
            Warning("Trying to get line_spacing without a font loaded.");
            return Base::line_spacing(font);
        }
        return sign_cast<uint32_t>(font.size * _fonts.at(font.style)->FontSize / im_font_scale);
    }

    void IMGUIBase::set_title(std::string title) {
        _title = title;
        
        exec_main_queue([this, title](){
            if(_platform)
                _platform->set_title(title);
        });
    }

    Size2 IMGUIBase::window_dimensions() {
        //auto window = _platform->window_handle();
        //int width, height;
        //glfwGetWindowSize(window, &width, &height);
        
        
        //glfwGetFramebufferSize(window, &width, &height);
        //Size2 size(width, height);
        
        return _last_framebuffer_size;
    }

Size2 IMGUIBase::real_dimensions() {
    if(_dpi_scale > 0)
        return _last_framebuffer_size.div(_dpi_scale);
    return _last_framebuffer_size;
}

    Event IMGUIBase::toggle_fullscreen(DrawStructure &graph) {
        static int _wndSize[2];
        static int _wndPos[2];
        
        Debug("Enabling full-screen.");
        _platform->toggle_full_screen();
        
        Event event(WINDOW_RESIZED);
        
        // backup window position and window size
        glfwGetWindowPos( _platform->window_handle(), &_wndPos[0], &_wndPos[1] );
        glfwGetFramebufferSize(_platform->window_handle(), &_wndSize[0], &_wndSize[1]);
        //glfwGetWindowSize( _platform->window_handle(), &_wndSize[0], &_wndSize[1] );
        
        event.size.width = _wndSize[0];
        event.size.height = _wndSize[1];
        graph.event(event);
        
        return event;
    }
}
